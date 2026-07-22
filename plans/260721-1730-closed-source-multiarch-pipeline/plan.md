---
title: Closed-source backbones (deepseek.v3, haiku-4.5) + pipeline parity 3 module + mini-test + rate-limit resilience
status: in-progress
created: 2026-07-21
implemented: 2026-07-22
owner: ngoquan0904
blockedBy: []
blocks: []
tags: [bedrock, closed-source, pipeline-parity, rate-limit, resume, mini-test]
---

# Closed-source backbones + pipeline parity + resilience

## Mục tiêu
1. Chạy thêm 2 backbone closed-source: `deepseek.v3-v1:0` và `anthropic.claude-haiku-4-5`.
2. `run_all.sh` của **MA_GP** và **PI_GP** đạt parity với **TA_GP**: gendata 1 lần → defense trên nhiều kiến trúc.
3. Mini-test full luồng trước khi chạy full.
4. Chạm rate limit không sập hệ thống.

> Kế thừa [[260626-1500-ma-gp-escalation]] và [[260627-1000-pi-gp-escalation]] (đều `completed`). Không sửa logic escalation/GNN — plan này chỉ động vào **tầng gọi LLM** và **tầng orchestration**.

## Quyết định đã chốt với user
| Câu hỏi | Chốt | Hệ quả |
|---|---|---|
| GNN retrain theo backbone? | **Không** — tái dùng checkpoint train trên data Llama, chỉ đổi backbone lúc defense | Gendata + train **giữ nguyên Llama, chạy 1 lần**. Closed-source **chỉ xuất hiện ở phase defense**. Đây là thí nghiệm **cross-model generalization** của detector |
| Phạm vi topology | **ĐÚNG 1 topology = `star`** (chốt lại, xem Phụ lục A/B) | **10** defense run/backbone thay vì 40 |
| Chống sập | **Retry backoff + checkpoint resume** | Cần cả 2, không chỉ retry |

### Ma trận cuối cùng
| Module | defense run/backbone |
|---|---|
| TA_GP | 1 topo × 2 GNN = 2 |
| MA_GP | 1 topo × 2 GNN = 2 |
| PI_GP | 3 dataset × 1 topo × 2 GNN = 6 |
| **Tổng** | **10 run = 38.400 call/backbone** |

**Chi phí: deepseek ~$53 + haiku ~$115 = ~$168** (giảm $504 so với phương án 4-topology).

### Thiết kế thí nghiệm sau khi cắt (quan trọng cho paper)
Không mất tính tổng quát vì hai chiều được chứng minh **tách biệt**, không cần cross product:
- **Tổng quát hoá theo topology** → đã chứng minh đầy đủ trên backbone Llama (4 topology, kết quả đã có).
- **Tổng quát hoá theo backbone** → chứng minh trên `star` (3 backbone).

Viết rõ như vậy trong paper thì thiết kế này hợp lệ. Điều **không** được tuyên bố: "phương pháp tổng quát trên mọi topology × mọi backbone" — chưa đo cross product đó.

---

## Phát hiện đã verify bằng thực nghiệm (không phải giả định)

### F1 — Ba API shape khác nhau, region khác nhau
Đã probe trực tiếp bằng key Bedrock:

| Model | Route | Region | Shape | Trạng thái |
|---|---|---|---|---|
| `Meta-Llama-3.1-70B-...` (hiện tại) | self-hosted `BASE_URL` | — | OpenAI chat | ✅ đang chạy |
| `google.gemma-4-31b` | mantle `/openai/v1/chat/completions` | us-east-1 | OpenAI chat | ✅ 200, **drop-in** |
| `deepseek.v3-v1:0` | runtime `/model/{id}/converse` | **us-west-2** | Bedrock Converse | ✅ 200 |
| `anthropic.claude-haiku-4-5` | mantle `/anthropic/v1/messages` | **us-east-1** | Anthropic Messages | ✅ 200 |

Bằng chứng phủ định quan trọng — **không thể chỉ đổi `BASE_URL`**:
```
chat/completions + anthropic.claude-haiku-4-5 -> 400 "does not support the '/v1/chat/completions' API"
chat/completions + deepseek.v3-v1:0           -> 400 "isn't supported on this route"
converse(us-east-1) + deepseek.v3-v1:0        -> 400 "The provided model identifier is invalid."  (region!)
```
Phân biệt 400 vs 404 cho biết model **tồn tại nhưng sai route**, không phải sai tên. Region sai trả **500/400 gây hiểu nhầm là lỗi credential** — đã mất thời gian vì lỗi này, phải chống bằng preflight (Phase 2).

### F2 — ⚠️ Rate limit KHÔNG làm sập, nó làm **mất data âm thầm**
`main_defense_for_different_topology.py:141-147` (và bản v2, cả 3 module):
```python
try:
    ...
    communication_data_no_defense = await no_defense_communication(...)
    communication_data_defense, identified_attackers = await defense_communication(...)
except Exception as e:
    print(e)
    continue          # <-- nuốt 429 và BỎ NGUYÊN sample
```
Một lỗi 429 ở bất kỳ agent nào → **drop toàn bộ sample**, vòng lặp vẫn chạy tiếp, file kết quả vẫn ghi, exit code vẫn 0. Hậu quả: `evaluation_results.csv` có số liệu tính trên **n < 60 mà không ai biết**, và n khác nhau giữa các topology/backbone → **so sánh sai lệch, không phát hiện được**.

Đây mới là rủi ro thật của requirement 4 — nguy hiểm hơn crash, vì crash thì biết mà sửa. Fix bắt buộc: đếm và **fail loud** khi tỉ lệ drop vượt ngưỡng.

### F3 — Không có giới hạn concurrency, không có retry
`asyncio.gather(*tasks)` (12 call site trên 4 file agents) bắn đồng thời 1 request/agent = 8 concurrent, **không semaphore, không retry**. Với Bedrock TPM quota chung, đây là nguồn 429 chính.

### F4 — Gap parity so với TA_GP
| | Gendata | Loop | Isolation lỗi |
|---|---|---|---|
| **TA_GP** (chuẩn) | 1 lần | 4 topology | subshell + `FAILED[]`, skip không kéo sập |
| **MA_GP** | ❌ **comment hết**, checkpoint hardcode | ❌ chỉ `tree` | ❌ `set -e` top-level → 1 lỗi chết cả run |
| **PI_GP** | ✅ per-dataset | ⚠️ 3 dataset nhưng **chỉ `random`** | ✅ đã có subshell |

→ MA_GP thiếu cả 3; PI_GP chỉ thiếu vòng topology (cần lồng dataset × topology).

### F5 — 🔴 Secret hardcode trong repo
`TA_GP/evaluate_output.py:12-13` chứa **JWT thật** của `assistant-stream.vnpt.vn` commit thẳng vào git. Phải rút ra env var + xoay key. (Ngoài scope kỹ thuật nhưng chặn việc public repo cho paper.)

### F6 — Quy mô & chi phí
`samples=60`, 8 agent, ~4 turn, ×2 (no_defense + defense) = **3.840 call/defense-run**.

| Module | defense run/backbone |
|---|---|
| TA_GP | 4 topo × 2 GNN = 8 |
| MA_GP | 4 topo × 2 GNN = 8 |
| PI_GP | 3 ds × 4 topo × 2 GNN = 24 |
| **Tổng** | **40 run = 153.600 call/backbone** |

Ước tính (~1500 in / 300 out per call):
- haiku-4.5: **~$461**
- deepseek.v3: **~$211**
- **Tổng 2 backbone mới ≈ $672** (chưa tính judge LLM ở `evaluate_output.py`)

Con số này là lý do Phase 2 (mini-test) và Phase 5 (chạy theo lô) là bắt buộc, không phải nice-to-have.

---

## Kiến trúc: tầng provider dùng chung

Điểm chèn tối ưu là `llm_invoke` / `allm_invoke` — **giống hệt nhau ở cả 3 module**, mọi agent call đều đi qua đây. Viết **1 file `llm_provider.py`, copy sang 3 module** (giữ convention "mỗi module self-contained" của repo; không tạo shared package để tránh phá cấu trúc import hiện tại).

```
allm_invoke(prompt, model_type)
        │
        ├── route theo prefix của model_type
        │     "anthropic.*"  -> AnthropicMantleAdapter (us-east-1)
        │     "deepseek.*"   -> BedrockConverseAdapter (us-west-2)
        │     "openai.*"     -> BedrockConverseAdapter (us-east-1)
        │     "google.*"     -> OpenAIAdapter (mantle us-east-1)
        │     còn lại        -> OpenAIAdapter (BASE_URL self-hosted)  # Llama, không đổi
        │
        ├── Semaphore(MAX_CONCURRENCY)     # F3
        └── retry: 429/5xx -> exponential backoff + jitter, tôn trọng retry-after
```
Chữ ký `allm_invoke(prompt, model_type)` **giữ nguyên** → 12 call site trong `agents.py` không phải sửa.

---

## Phase 1 — Tầng provider: adapter + retry + concurrency

**File mới:** `llm_provider.py` (copy vào `TA_GP/`, `MA_GP/`, `PI_GP/`)

1. `MODEL_ROUTES: dict[prefix -> (adapter, region, endpoint)]` theo bảng F1.
2. Ba adapter, cùng interface `async def invoke(messages, model, **kw) -> str`:
   - `OpenAIAdapter` — giữ nguyên hành vi hiện tại (`AsyncOpenAI`).
   - `AnthropicMantleAdapter` — tách `messages[0]` role=system ra field `system` (Anthropic API tách riêng, **không nhận system trong messages**); `max_tokens` bắt buộc.
   - `BedrockConverseAdapter` — map sang `{"messages":[{"role","content":[{"text":...}]}], "system":[{"text":...}], "inferenceConfig":{"maxTokens","temperature"}}`; đọc `output.message.content[0].text`. **URL-encode `:` trong model id** (`deepseek.v3-v1%3A0`).
3. Retry: `tenacity` — `retry_if_exception(status in {429,500,502,503,529})`, `wait_exponential_jitter(initial=2, max=60)`, `stop_after_attempt(6)`. Ưu tiên header `retry-after` nếu có.
4. `_SEM = asyncio.Semaphore(int(os.getenv("LLM_MAX_CONCURRENCY", "4")))` bọc mọi call.
5. Đổi `agents.py` (3 module, 4 file): `from llm_provider import allm_invoke, llm_invoke` thay cho định nghĩa local. **Không đổi chữ ký.**

**Done when:** `python -c "import asyncio,llm_provider; print(asyncio.run(llm_provider.allm_invoke([{'role':'system','content':'terse'},{'role':'user','content':'say OK'}], 'anthropic.claude-haiku-4-5')))"` in ra `OK`, lặp lại với `deepseek.v3-v1:0` và model Llama hiện tại.

⚠️ Rủi ro: temperature/top_p hiện khác nhau giữa 3 module (TA dùng `temperature=0.7, presence_penalty=1.5, extra_body.top_k`; MA/PI dùng `temperature=0`). `extra_body` là param riêng của vLLM — **Bedrock sẽ reject**. Adapter phải **lọc bỏ** param không hỗ trợ theo từng route, không truyền mù.

## Phase 2 — Mini-test full luồng (chạy TRƯỚC mọi full run)

**File mới:** `smoke_test.sh` + `preflight.py` (mỗi module)

1. `preflight.py --model_type X` — gọi **1 request thật** tới đúng route/region của model, in rõ `✅ route=... region=...` hoặc fail với thông báo phân biệt được **sai region vs sai credential vs hết quota** (chính là lỗi đã vấp: region sai trả 500 trông như lỗi server).
2. `smoke_test.sh` chạy **full luồng với tham số tí hon**:
   ```bash
   SAMPLES=2 NUM_GRAPHS=1 EPOCHS=1 TOPOLOGIES=(random chain) ./smoke_test.sh
   ```
   → gendata (1 graph) → merge → build dataset → train 1 epoch → defense 2 sample × 2 topology × 2 GNN → evaluate → ghi CSV vào **`../evaluation_results_SMOKE.csv`** (tách khỏi CSV thật).
3. Thêm `--samples` vào MA_GP/PI_GP `main_defense*` nếu chưa có (TA_GP đã có, default 60).
4. Smoke phải verify **cả 3 backbone** trong cùng 1 lệnh: `./smoke_test.sh --all-backbones`.

**Done when:** smoke chạy < 10 phút, chi phí < $1, và **fail đúng chỗ** khi cố tình đặt sai region (test tiêu cực bắt buộc).

## Phase 3 — Chống mất data âm thầm + resume

**Sửa:** `main_defense_for_different_topology.py` + `_v2.py` (cả 3 module = 6 file)

1. **Fail loud** — thay `except: continue`:
   ```python
   except Exception as e:
       skipped.append({"idx": i, "err": repr(e)})
       print(f"⚠️  SKIP sample {i}: {e!r}")
       continue
   # sau vòng lặp:
   n_ok, n_skip = len(final_dataset_nd), len(skipped)
   print(f"SAMPLES_OK: {n_ok}   SAMPLES_SKIPPED: {n_skip}")
   if n_skip / (n_ok + n_skip) > float(os.getenv("MAX_SKIP_RATIO", "0.1")):
       print(f"❌ drop ratio {n_skip}/{n_ok+n_skip} vượt ngưỡng — kết quả KHÔNG dùng được")
       sys.exit(2)
   ```
2. **Ghi cột `n_samples` vào CSV** trong `evaluate_output.py` — để mọi dòng kết quả tự mang theo cỡ mẫu, không bao giờ so sánh nhầm giữa các n khác nhau.
3. **Checkpoint resume** — ghi từng sample vào JSONL ngay khi xong thay vì `json.dump` một lần cuối:
   - `{save_path}.partial.jsonl`, append sau mỗi sample.
   - Khởi động: nếu file tồn tại → load, `done_idx = set(...)`, skip các sample đã có.
   - Kết thúc: gộp JSONL → JSON cuối, xoá `.partial`.
   - → mất điện/OOM/429-kéo-dài giữa chừng, chạy lại **không mất phần đã xong**.

**Done when:** giết process giữa chừng (`kill -9`), chạy lại, log hiện `RESUMED: skip N samples` và kết quả cuối đủ 60 sample.

## Phase 4 — Parity `run_all.sh`

### 4a. MA_GP (thiếu nhiều nhất)
Viết lại theo đúng khung TA_GP:
1. **Bỏ comment** toàn bộ Phase 1 (gendata train/test → merge → build → train v1/v2), **xoá 2 checkpoint hardcode**.
2. **Bỏ `set -e` top-level**; Phase 1 giữ fail-fast, Phase 2 dùng subshell + `FAILED[]`.
3. Thêm vòng `TOPOLOGIES=(random chain tree star)` (hiện chỉ `tree`).
4. Thêm vòng backbone (xem 4c).

### 4b. PI_GP (chỉ thiếu vòng topology)
Lồng thêm 1 cấp: `for DS in datasets { gendata+train 1 lần; for TOPO in 4 topology { defense v1+v2; evaluate } }`.
→ Đúng yêu cầu "**gendata 1 lần, defense trên nhiều kiến trúc**": gendata/train nằm **ngoài** vòng topology.

### 4c. Vòng backbone (cả 3 module)
```bash
GENDATA_MODEL="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"   # phase 1: cố định
BACKBONES=("$GENDATA_MODEL" "deepseek.v3-v1:0" "anthropic.claude-haiku-4-5")

# PHASE 1 — gendata + train, CHỈ với GENDATA_MODEL, chạy 1 lần
# PHASE 2 — for BK in BACKBONES: for TOPO in TOPOLOGIES: defense v1+v2 + evaluate
```
`evaluate_output.py` đã tách `model_type` từ tên file (`main_defense` có `replace('/','_')`) → CSV tự phân biệt backbone.

✅ **Đã verify: `:` trong `deepseek.v3-v1:0` KHÔNG cần xử lý gì thêm.** Đã test cả 3 điểm nghi ngờ:
- regex `model_type_(.+?)\.json$` → parse ra đúng `deepseek.v3-v1:0`
- `cut -d':' -f2-` trong run_all.sh → `-f2-` giữ mọi field sau dấu `:` đầu, path còn nguyên
- ext4 chấp nhận `:` trong tên file

→ **Không đổi `safe_model_type`.** (Chỉ cần xử lý nếu sau này chạy trên Windows/SMB share.)

**Done when:** `bash -n run_all.sh` pass cả 3; smoke chạy end-to-end ra CSV có đủ cột `graph_type` × `model_type` × `defense_model`.

## Phase 5 — Chạy full theo lô, kiểm soát chi phí

1. `run_matrix.sh` ở repo root: chạy tuần tự 3 module × 3 backbone, **checkpoint theo từng (module, backbone, topology)**, resume được ở cấp lô.
2. Trước mỗi lô in **ước tính chi phí + số call**, yêu cầu `--yes` để chạy không hỏi.
3. Xin **quota increase** Bedrock trước (miễn phí, xem `service-quotas`) thay vì hạ concurrency — đó là cách đúng để tăng throughput.
4. Thứ tự chạy đề xuất: **deepseek trước** (rẻ hơn ~2.2×) để phát hiện lỗi orchestration bằng tiền rẻ, rồi mới tới haiku.

**Done when:** CSV cuối có đủ 120 tổ hợp với cột `n_samples` đồng nhất = 60.

## Phase 6 — Dọn secret (chặn public repo)
1. Rút JWT khỏi `TA_GP/evaluate_output.py` (và kiểm tra 2 file evaluate còn lại) → `os.getenv("JUDGE_API_KEY")`.
2. **Xoay key** đã lộ — nó nằm trong git history, không chỉ working tree.
3. Thêm `.env.example`; đảm bảo `.env` trong `.gitignore`.

---

## Thứ tự thực thi
```
Phase 1 (provider) ──> Phase 2 (mini-test) ──> Phase 3 (resilience)
                                                     │
                          Phase 4 (parity run_all) ◄─┘
                                     │
                          Phase 5 (full matrix)
Phase 6 (secret) — độc lập, làm bất cứ lúc nào, BẮT BUỘC trước khi public
```
Phase 2 phải xong **trước** Phase 5 — đó là toàn bộ lý do nó tồn tại.

## Rủi ro chính
| Rủi ro | Mức | Giảm thiểu |
|---|---|---|
| ~~`:` trong `deepseek.v3-v1:0` phá filename/regex~~ | ~~Cao~~ → **Loại bỏ** | Đã verify an toàn qua regex + `cut` + ext4. Không cần làm gì |
| Prompt tuned cho Llama cho kết quả khác trên haiku/deepseek | **Cao** | Đây là bản chất thí nghiệm cross-model, không phải bug — nhưng phải ghi rõ trong paper là detector train trên data Llama |
| Anthropic API tách `system` khỏi `messages` — adapter map sai sẽ mất system prompt **âm thầm** (không lỗi, chỉ tệ kết quả) | **Cao** | Assert trong adapter: nếu `messages[0].role=="system"` mà field `system` rỗng → raise. Test trong smoke |
| `extra_body`/`presence_penalty` bị Bedrock reject | Trung bình | Adapter lọc param theo route (Phase 1.5) |
| Chi phí vượt $672 do retry lặp | Trung bình | Circuit breaker + `stop_after_attempt(6)` + in cost mỗi lô |
| Sai region trả 500 gây debug nhầm hướng | Trung bình | `preflight.py` phân loại lỗi rõ ràng (Phase 2.1) |
| Drop ratio khác nhau giữa backbone → so sánh lệch | **Cao** | Cột `n_samples` + fail khi vượt `MAX_SKIP_RATIO` (Phase 3) |

---

## Phụ lục C — Kết quả implement thực tế (2026-07-22)

### C1. Những gì mini test bắt được (đều KHÔNG đoán trước được)
| # | Vấn đề | Cách xử lý |
|---|---|---|
| 1 | `torch_scatter` không build được với torch 2.13 (extension phải compile, chưa có wheel) | `scatter_compat.py` dùng `torch_geometric.utils.scatter`; đã verify khớp số học kể cả index rỗng → 0 |
| 2 | MA_GP + PI_GP hardcode `map_location=torch.device('cuda')` → crash trên máy CPU-only | `_DEVICE = cuda if available else cpu`, sửa 4 file |
| 3 | `tee /dev/tty` fail khi chạy headless (không có TTY) | đổi sang `/dev/stderr` |
| 4 | **`get_sentence_embedding()` khởi tạo SentenceTransformer BÊN TRONG hàm** — gọi ~1.900 lần/defense run, mỗi lần nạp lại model 90MB | lazy singleton: **5.54s → 28ms/lần**, tiết kiệm ~175 phút/run |
| 5 | JWT thật hardcode trong 3 `evaluate_output.py` | rút ra `.env` (chmod 600) + `.gitignore` |

Mục 4 là thứ đáng giá nhất và hoàn toàn không nhìn ra được nếu chỉ đọc code — chỉ lộ khi chạy thật trên CPU.

### C2. Tốc độ thực đo (2 sample, topology star, MA_GP)
| Backbone | giây/sample |
|---|---|
| deepseek.v3-v1:0 | **~49s** |
| anthropic.claude-haiku-4-5 | **~107s** |

deepseek nhanh hơn 2.2× **và** rẻ hơn 2.2× → củng cố thứ tự "deepseek trước" ở Phase 5.4.

Ước tính full run (`--samples 60`, star, v1+v2):

| Module | deepseek | haiku |
|---|---|---|
| TA_GP | ~1.6h | ~3.6h |
| MA_GP | ~1.6h | ~3.5h |
| PI_GP (×3 dataset) | ~4.9h | ~10.7h |
| **Tổng** | **~8h** | **~18h** |

≈ **26h tuần tự**, hoặc ~15h nếu chạy 3 module song song ở 3 terminal.

⚠️ **Nút thắt là vòng lặp sample chạy tuần tự** — chỉ 8 agent *trong* 1 sample là song song (semaphore=4). Muốn nhanh hơn nữa phải song song hoá ở tầng sample; nằm ngoài scope hiện tại.

### C3. Đã verify hoạt động
- Routing 3 shape + preflight phân loại lỗi đúng (sai region / sai key / sai model / hết quota).
- **System prompt tới nơi trên cả 2 route** — test bằng codeword bí mật trong system prompt; đây là rủi ro "Cao" đã nêu ở bảng Rủi ro, nay đã đóng.
- Llama bị skip sạch khi không có vLLM server, không kéo sập run — đúng thiết kế preflight.
- `SAMPLES_OK / SAMPLES_SKIPPED` in ra đúng; partial JSONL ghi từng sample → resume chạy được.
- MA_GP end-to-end: deepseek + haiku × v1 + v2 → CSV 16 dòng (2 backbone × 2 GNN × 4 turn).

### C4. Bài học vận hành (lỗi do người chạy, không phải code)
- **Không được khởi động 2 `run_all.sh` cùng module song song** — ghi đè lẫn nhau, gây `❌ v1 không ra file` gây hiểu nhầm là bug.
- **`pkill -f "run_all.sh"` tự sát**: `-f` khớp toàn bộ command line, nên nếu lệnh dọn dẹp nằm cùng một shell có chuỗi `run_all.sh` thì nó giết chính mình (exit 144). Dùng pattern hẹp hơn (`main_defense_for_diff`).
- `nohup ... &` bên trong một tool call vẫn bị kill khi call kết thúc — phải dùng cơ chế background của harness.

---

## Phụ lục A — Phân tích tương quan topology (đo trên `evaluation_results (2).csv`, n=104 dòng, backbone Llama)

### A1. Topology chỉ giải thích ~10% variance
| Metric | Variance giữa condition (attack/turn/GNN) | Variance giữa 4 topology |
|---|---|---|
| `det_f1` | **89.7%** | **10.3%** |
| `safe_rate_defense` | **87.4%** | **12.6%** |

→ Topology là yếu tố **thứ yếu**. Gần 90% biến thiên đến từ attack type, turn, và defense model.

### A2. Tương quan cặp rất cao
`det_f1`: r = **+0.905 … +0.961**, mean|Δ| = 2.1–4.8%
`safe_rate_defense`: r = **+0.748 … +0.899**, mean|Δ| = 2.2–2.9%

### A3. Kết luận KHÔNG đảo chiều theo topology
Tỉ lệ `TemporalGAT_v2` thắng `MyGAT_v1`:

| Topology | det_f1 | safe_rate_defense |
|---|---|---|
| random | 8/9 (89%) | 8/12 (67%) |
| chain | 8/12 (67%) | 10/16 (62%) |
| tree | **9/9 (100%)** | **11/12 (92%)** |
| star | **9/9 (100%)** | 9/12 (75%) |

→ TemporalGAT thắng trên **cả 4** topology. Chạy 1 topology hay 4 topology đều ra **cùng một kết luận**.

### A4. ⚠️ Tự sửa: `random + star` là cặp TỆ NHẤT
RMSE khi dùng tập con để ước lượng trung bình 4-topology (`det_f1`):

| Tập con | RMSE |
|---|---|
| **chain + star** | **0.0108** ✅ |
| **random + tree** | **0.0108** ✅ |
| random + chain | 0.0129 |
| tree + star | 0.0129 |
| ~~random + star~~ | **0.0208** ❌ |
| ~~chain + tree~~ | 0.0208 ❌ |
| 1 topology (tốt nhất: random) | 0.0227 |

**Đề xuất `random + star` ở bản plan trước là SAI.** Lý do: `random` và `star` có tương quan cao nhất (r=+0.961) → **dư thừa nhau**, gộp lại không thêm thông tin. Cặp tốt là cặp *bù trừ* nhau, không phải cặp giống nhau. `chain+star` giảm sai số **gần 2×** so với `random+star` với cùng chi phí.

### A5. Khuyến nghị (bối cảnh: nếu chạy 2 topology)
**Chạy 2 topology `chain + star` cho 2 backbone mới**, không phải 4:
- Sai số ước lượng 0.0108 — nhỏ hơn mean|Δ| giữa các topology, tức **nằm trong nhiễu sẵn có**.
- `star` là discriminator mạnh nhất (100%/75% win) → thể hiện rõ ưu thế phương pháp.
- `chain` là case khó nhất (67%/62%) → **không cherry-pick**, tăng độ tin cậy khi review.
- **Bao trọn dải điểm**: `chain` = det_f1 thấp nhất (0.8572), `star` = cao nhất (0.8803), và đây là **cặp duy nhất khác nhau có ý nghĩa thống kê** (paired t=-2.53, df=17, p<0.05). Các cặp còn lại không significant → báo cáo chain+star là báo cáo cả biên dưới lẫn biên trên của khoảng thực sự đo được.
- Tiết kiệm **~$336** và một nửa thời gian.

**Giả định cần kiểm chứng:** phân tích trên đo bằng backbone Llama. Chưa có bằng chứng topology hành xử y hệt trên deepseek/haiku. Cách khử rủi ro rẻ nhất:

> Chạy **deepseek đủ 4 topology** (~$211, backbone rẻ hơn 2.2×) → tính lại A1–A4 trên chính data đó. Nếu variance topology vẫn ~10% và không đảo chiều → chạy **haiku chỉ `chain+star`** (~$230). **Tổng ~$441**, vẫn tiết kiệm $231 so với full, mà *chứng minh được* việc cắt topology là hợp lệ thay vì giả định.

---

## Phụ lục B — Chọn 1 topology duy nhất: vì sao là `star` (CHỐT)

Tiêu chí: topology thể hiện rõ nhất **cả hiệu quả tấn công lẫn hiệu quả phòng thủ**.

### B1. `star` thắng toàn bộ 4 tiêu chí (n=24 condition paired)
| topo | ASR no_def ↑ | ASR sau defense | ASR giảm ↑ | giảm % ↑ | det_f1 ↑ |
|---|---|---|---|---|---|
| **star** | **0.1867** | 0.1403 | **0.0463** | **24.8%** | **0.8803** |
| random | 0.1851 | 0.1405 | 0.0446 | 24.1% | 0.8636 |
| chain | 0.1737 | 0.1430 | 0.0307 | 17.7% | 0.8572 |
| tree | 0.1681 | 0.1382 | 0.0298 | 17.7% | 0.8627 |

### B2. Lý do quyết định: `star` là topology DUY NHẤT cứu được PI
Defense gain theo attack type:

| Attack | random | chain | tree | **star** |
|---|---|---|---|---|
| `memory_attack` | +0.013 | −0.012 | +0.022 | **+0.030** |
| `pi_mmlu` | −0.002 | −0.006 | −0.010 | **+0.021** |
| `tool_attack` | **+0.123** | +0.110 | +0.077 | +0.089 |

→ Với `pi_mmlu`, **mọi topology trừ star đều cho gain ÂM** (defense làm tệ hơn). Nếu PI_GP chạy `random`, kết luận sẽ là "defense không có tác dụng". Đây là lý do quyết định chọn `star` khi chỉ được chạy 1.

### B3. Đánh đổi đã chấp nhận
`tool_attack` (TA_GP) mạnh nhất trên `random` (+0.123 so với +0.089 của star). Chọn `star` cho đồng bộ toàn bộ → TA_GP mất ~28% effect size. Chấp nhận được vì TA_GP vẫn dương mạnh trên star, và đồng bộ 1 topology giúp bảng kết quả so sánh được giữa 3 module.

### B4. Cảnh báo: turn 0 luôn cho gain ÂM
| turn | random | chain | tree | star |
|---|---|---|---|---|
| 0 | −0.008 | −0.005 | −0.009 | −0.005 |
| 1 | +0.032 | +0.023 | +0.050 | **+0.068** |
| 2 | +0.084 | +0.049 | +0.027 | +0.057 |
| 3 | +0.070 | +0.056 | +0.050 | +0.065 |

Turn 0 chưa có tín hiệu tấn công → GNN chỉ tạo false positive. **Phải nói rõ trong paper** thay vì để reviewer tự phát hiện; nếu không sẽ bị hỏi "sao defense làm giảm safe rate".

### B5. ⚠️ Giới hạn bằng chứng — phải kiểm chứng khi chạy
Khuyến nghị `star` cho PI dựa **chỉ trên `mmlu`**. Coverage thực tế trong CSV:

| | topology đã đo |
|---|---|
| `pi_mmlu` | random, chain, tree, star ✅ |
| `pi_csqa` | **chỉ chain** |
| `pi_gsm8k` | **không có dòng nào** |

→ Chưa có bằng chứng `star` là lựa chọn đúng cho `csqa` và `gsm8k`. **Hành động bắt buộc:** ở lần chạy Llama đầu tiên trên `star`, kiểm tra defense gain của csqa/gsm8k. Nếu âm → phải chạy thêm topology khác cho 2 dataset đó trước khi kết luận.

## Ghi chú phản biện
Cắt từ 4 xuống 1 topology tiết kiệm $504 và **không làm yếu kết luận**, vì topology chỉ chiếm ~10% variance (A1) và không đảo chiều so sánh GNN (A3). Rủi ro còn lại **không nằm ở chi phí mà ở B5**: chọn `star` dựa trên bằng chứng từ 1/3 dataset của PI. Đây là điểm cần verify sớm, không phải giả định để yên.
