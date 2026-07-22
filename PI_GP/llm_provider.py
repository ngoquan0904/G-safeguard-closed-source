"""
Tầng gọi LLM dùng chung cho TA_GP / MA_GP / PI_GP.

Mục đích:
  1. Route 1 `model_type` tới đúng API shape + region (3 shape khác nhau).
  2. Retry có backoff khi 429/5xx  -> không sập khi chạm rate limit.
  3. Giới hạn concurrency          -> giảm tần suất bị 429 ngay từ đầu.

Chữ ký `llm_invoke(prompt, model_type)` / `allm_invoke(prompt, model_type)` giữ
NGUYÊN như bản cũ trong agents.py, nên mọi call site không phải sửa.

Routing matrix (đã verify bằng request thật, 2026-07-21):

    model_type prefix     route                                        region
    ------------------    -----------------------------------------    ---------
    anthropic.*           mantle  /anthropic/v1/messages               us-east-1
    deepseek.*            runtime /model/{id}/converse                 us-west-2
    openai.*              runtime /model/{id}/converse                 us-east-1
    google.*              mantle  /openai/v1/chat/completions          us-east-1
    (còn lại)             BASE_URL tự host (vLLM Llama)                 -

Bằng chứng phủ định (vì sao KHÔNG thể chỉ đổi BASE_URL):
    chat/completions + anthropic.claude-haiku-4-5 -> 400 "does not support the '/v1/chat/completions' API"
    chat/completions + deepseek.v3-v1:0           -> 400 "isn't supported on this route"
    converse(us-east-1) + deepseek.v3-v1:0        -> 400 "The provided model identifier is invalid."

Auth: dùng bearer token (AWS_BEARER_TOKEN_BEDROCK), KHÔNG phải SigV4. Vì vậy ở
đây gọi HTTP trực tiếp thay vì dùng AnthropicBedrockMantle/boto3 — các client đó
ký SigV4 từ AWS credentials, không nhận bearer token.
"""

import asyncio
import os
import random
import time
import urllib.parse

import httpx

# ──────────────────────────────────────────────────────────────────────────
# Cấu hình
# ──────────────────────────────────────────────────────────────────────────

MAX_CONCURRENCY = int(os.getenv("LLM_MAX_CONCURRENCY", "4"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "6"))
BASE_BACKOFF = float(os.getenv("LLM_BASE_BACKOFF", "2.0"))
MAX_BACKOFF = float(os.getenv("LLM_MAX_BACKOFF", "60.0"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "180"))
DEFAULT_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}

_MANTLE = "https://bedrock-mantle.{region}.api.aws"
_RUNTIME = "https://bedrock-runtime.{region}.amazonaws.com"

# (prefix, kind, region). Khớp theo thứ tự -> prefix cụ thể đặt trước.
_ROUTES = [
    ("anthropic.", "anthropic_messages", "us-east-1"),
    ("deepseek.", "bedrock_converse", "us-west-2"),
    ("openai.", "bedrock_converse", "us-east-1"),
    ("google.", "openai_chat_mantle", "us-east-1"),
]


class LLMError(RuntimeError):
    """Lỗi gọi LLM đã hết số lần retry, hoặc lỗi không thể retry."""

    def __init__(self, message, status=None, body=None):
        super().__init__(message)
        self.status = status
        self.body = body


def resolve_route(model_type: str):
    """model_type -> (kind, region). Trả ('openai_chat_selfhosted', None) nếu không khớp."""
    for prefix, kind, region in _ROUTES:
        if model_type.startswith(prefix):
            return kind, region
    return "openai_chat_selfhosted", None


def _bedrock_token() -> str:
    tok = os.getenv("AWS_BEARER_TOKEN_BEDROCK")
    if not tok:
        raise LLMError(
            "Thiếu AWS_BEARER_TOKEN_BEDROCK — bắt buộc cho model Bedrock "
            "(anthropic.* / deepseek.* / openai.* / google.*)."
        )
    return tok


def _need_text(v):
    if not v:
        raise LLMError("Response rỗng (không có text).", status=200)
    return v


def _extract_openai_message(d):
    """Lấy text từ response OpenAI-compat.

    Reasoning model (gpt-oss, deepseek-r1, ...) trả `content = null` và đưa nội
    dung vào `reasoning_content`. Không xử lý thì mọi agent response thành None
    và parser vỡ ở tận cuối pipeline — rất khó truy.
    """
    msg = d["choices"][0]["message"]
    for key in ("content", "reasoning_content"):
        v = msg.get(key)
        if v:
            return v
    raise LLMError(
        "Response không có text: cả 'content' lẫn 'reasoning_content' đều rỗng. "
        f"finish_reason={d['choices'][0].get('finish_reason')!r}. "
        "Nếu là reasoning model, thử tăng max_tokens hoặc bật --reasoning-parser trên vLLM.",
        status=200,
    )


def _split_system(messages):
    """OpenAI messages -> (system_text, messages_không_system).

    Anthropic Messages API và Bedrock Converse đều tách `system` ra field riêng,
    KHÔNG nhận role=system trong mảng messages.
    """
    system_parts, rest = [], []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            rest.append(m)
    return "\n\n".join(p for p in system_parts if p), rest


# ──────────────────────────────────────────────────────────────────────────
# Dựng request theo từng shape
# ──────────────────────────────────────────────────────────────────────────

def _build(model_type, messages, temperature, max_tokens, extra=None):
    """-> (url, headers, payload, extractor)

    `extra` = tham số sampling chỉ vLLM hiểu (top_p / presence_penalty / top_k...).
    CHỈ áp dụng cho route self-hosted; Bedrock sẽ 400 nếu nhận các field này,
    nên chúng bị bỏ qua ở mọi route khác — có chủ đích, không phải quên.
    """
    kind, region = resolve_route(model_type)
    # Bedrock bắt buộc có max_tokens -> áp default. Self-hosted thì bỏ hẳn field
    # nếu caller không set, để giữ nguyên hành vi vLLM trước đây.
    bedrock_max = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

    if kind == "anthropic_messages":
        system, rest = _split_system(messages)
        if not rest:
            raise LLMError("Anthropic route cần ít nhất 1 message không phải system.")
        payload = {
            "model": model_type,
            "max_tokens": bedrock_max,
            "temperature": temperature,
            "messages": [
                {"role": m["role"], "content": m.get("content", "")} for m in rest
            ],
        }
        if system:
            payload["system"] = system
        return (
            _MANTLE.format(region=region) + "/anthropic/v1/messages",
            {
                "Authorization": f"Bearer {_bedrock_token()}",
                "content-type": "application/json",
                "anthropic-version": "2023-06-01",
            },
            payload,
            lambda d: _need_text(d["content"][0].get("text")),
        )

    if kind == "bedrock_converse":
        system, rest = _split_system(messages)
        if not rest:
            raise LLMError("Converse route cần ít nhất 1 message không phải system.")
        payload = {
            "messages": [
                {"role": m["role"], "content": [{"text": m.get("content", "")}]}
                for m in rest
            ],
            "inferenceConfig": {"maxTokens": bedrock_max, "temperature": temperature},
        }
        if system:
            payload["system"] = [{"text": system}]
        # model id chứa ':' -> phải URL-encode trong path
        mid = urllib.parse.quote(model_type, safe="")
        return (
            _RUNTIME.format(region=region) + f"/model/{mid}/converse",
            {
                "Authorization": f"Bearer {_bedrock_token()}",
                "Content-Type": "application/json",
            },
            payload,
            lambda d: _need_text(d["output"]["message"]["content"][0].get("text")),
        )

    if kind == "openai_chat_mantle":
        return (
            _MANTLE.format(region=region) + "/openai/v1/chat/completions",
            {
                "Authorization": f"Bearer {_bedrock_token()}",
                "content-type": "application/json",
            },
            {
                "model": model_type,
                "max_tokens": bedrock_max,
                "temperature": temperature,
                "messages": messages,
            },
            _extract_openai_message,
        )

    # self-hosted vLLM (Llama) — giữ nguyên hành vi cũ, kể cả extra_body
    base = os.getenv("BASE_URL")
    if not base:
        raise LLMError(f"Thiếu BASE_URL cho model self-hosted '{model_type}'.")
    payload = {
        "model": model_type,
        "temperature": temperature,
        "messages": messages,
        # vLLM-only; Bedrock sẽ reject nên chỉ gắn ở route này
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if extra:
        payload.update(extra)
    return (
        base.rstrip("/") + "/chat/completions",
        {
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'EMPTY')}",
            "content-type": "application/json",
        },
        payload,
        _extract_openai_message,
    )


def _retry_after(resp, attempt):
    """Giây cần chờ: ưu tiên header Retry-After, không có thì exponential + jitter."""
    if resp is not None:
        ra = resp.headers.get("retry-after")
        if ra:
            try:
                return min(float(ra), MAX_BACKOFF)
            except ValueError:
                pass
    return min(BASE_BACKOFF * (2 ** attempt), MAX_BACKOFF) * (0.5 + random.random())


def _should_retry(status):
    return status in RETRY_STATUS


# ──────────────────────────────────────────────────────────────────────────
# Semaphore theo từng event loop (tránh bind nhầm loop)
# ──────────────────────────────────────────────────────────────────────────

_sems: dict = {}


def _sem():
    loop = asyncio.get_running_loop()
    s = _sems.get(loop)
    if s is None:
        s = asyncio.Semaphore(MAX_CONCURRENCY)
        _sems[loop] = s
    return s


# ──────────────────────────────────────────────────────────────────────────
# Public API — giữ nguyên chữ ký cũ
# ──────────────────────────────────────────────────────────────────────────

async def allm_invoke(prompt, model_type: str, temperature: float = 0.0,
                      max_tokens: int | None = None, extra: dict | None = None) -> str:
    url, headers, payload, extract = _build(model_type, prompt, temperature, max_tokens, extra)
    last = None
    async with _sem():
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for attempt in range(MAX_ATTEMPTS):
                try:
                    r = await client.post(url, headers=headers, json=payload)
                except (httpx.TimeoutException, httpx.TransportError) as e:
                    last = LLMError(f"network: {e!r}")
                    if attempt == MAX_ATTEMPTS - 1:
                        break
                    await asyncio.sleep(_retry_after(None, attempt))
                    continue

                if r.status_code == 200:
                    try:
                        out = extract(r.json())
                        if not out:
                            raise LLMError(f"'{model_type}' trả về text rỗng.", status=200,
                                           body=r.text[:500])
                        return out
                    except (KeyError, IndexError, ValueError) as e:
                        raise LLMError(
                            f"Không parse được response của '{model_type}': {e!r}",
                            status=200, body=r.text[:500],
                        ) from e

                last = LLMError(
                    f"HTTP {r.status_code} từ '{model_type}': {r.text[:300]}",
                    status=r.status_code, body=r.text[:500],
                )
                if not _should_retry(r.status_code) or attempt == MAX_ATTEMPTS - 1:
                    break
                await asyncio.sleep(_retry_after(r, attempt))
    raise last if last else LLMError(f"gọi '{model_type}' thất bại không rõ lý do")


def llm_invoke(prompt, model_type: str, temperature: float = 0.0,
               max_tokens: int | None = None, extra: dict | None = None) -> str:
    """Bản đồng bộ. Cùng logic retry, không có semaphore (đã tuần tự sẵn)."""
    url, headers, payload, extract = _build(model_type, prompt, temperature, max_tokens, extra)
    last = None
    with httpx.Client(timeout=TIMEOUT) as client:
        for attempt in range(MAX_ATTEMPTS):
            try:
                r = client.post(url, headers=headers, json=payload)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = LLMError(f"network: {e!r}")
                if attempt == MAX_ATTEMPTS - 1:
                    break
                time.sleep(_retry_after(None, attempt))
                continue

            if r.status_code == 200:
                try:
                    out = extract(r.json())
                    if not out:
                        raise LLMError(f"'{model_type}' trả về text rỗng.", status=200,
                                       body=r.text[:500])
                    return out
                except (KeyError, IndexError, ValueError) as e:
                    raise LLMError(
                        f"Không parse được response của '{model_type}': {e!r}",
                        status=200, body=r.text[:500],
                    ) from e

            last = LLMError(
                f"HTTP {r.status_code} từ '{model_type}': {r.text[:300]}",
                status=r.status_code, body=r.text[:500],
            )
            if not _should_retry(r.status_code) or attempt == MAX_ATTEMPTS - 1:
                break
            time.sleep(_retry_after(r, attempt))
    raise last if last else LLMError(f"gọi '{model_type}' thất bại không rõ lý do")


# ──────────────────────────────────────────────────────────────────────────
# Preflight — dùng trong smoke test để phân biệt sai region / sai key / hết quota
# ──────────────────────────────────────────────────────────────────────────

def preflight(model_type: str) -> tuple[bool, str]:
    kind, region = resolve_route(model_type)
    where = f"route={kind} region={region or 'BASE_URL'}"
    try:
        out = llm_invoke(
            [{"role": "system", "content": "Be terse."},
             {"role": "user", "content": "Reply with exactly: OK"}],
            model_type, max_tokens=16,
        )
        return True, f"✅ {model_type}  {where}  -> {out.strip()[:40]!r}"
    except LLMError as e:
        s = e.status
        if s in (401, 403):
            hint = "SAI/HẾT HẠN CREDENTIAL (AWS_BEARER_TOKEN_BEDROCK)"
        elif s == 404:
            hint = "TÊN MODEL SAI (model không tồn tại)"
        elif s == 400:
            hint = "SAI ROUTE HOẶC SAI REGION (model có tồn tại nhưng không phục vụ ở đây)"
        elif s == 429:
            hint = "HẾT QUOTA / RATE LIMIT"
        elif s and s >= 500:
            hint = "LỖI SERVER — hoặc REGION SAI (Bedrock hay trả 5xx khi model không có ở region này)"
        else:
            hint = "LỖI MẠNG hoặc không xác định"
        return False, f"❌ {model_type}  {where}\n     HTTP {s}: {hint}\n     {e}"


if __name__ == "__main__":
    import sys
    models = sys.argv[1:] or [
        "anthropic.claude-haiku-4-5",
        "deepseek.v3-v1:0",
    ]
    ok = True
    for m in models:
        good, msg = preflight(m)
        print(msg)
        ok &= good
    sys.exit(0 if ok else 1)
