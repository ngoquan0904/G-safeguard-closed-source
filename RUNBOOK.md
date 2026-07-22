# RUNBOOK — chạy 3 kịch bản tấn công

## Chuẩn bị (1 lần)

### 1. Thay key Bedrock
```bash
nano .env      # sửa AWS_BEARER_TOKEN_BEDROCK="ABSK..."
```
⚠️ Key đang có trong `.env` **đã bị lộ** (dán vào chat) — phải revoke trong IAM và tạo key mới.
`.env` đã `chmod 600` và nằm trong `.gitignore`.

### 2. (Tuỳ chọn) Bật vLLM cho baseline Llama
Nếu `BASE_URL` không có server, preflight sẽ **tự skip Llama** và chỉ chạy deepseek + haiku —
run vẫn chạy bình thường, chỉ là không có cột so sánh Llama.

### 3. Môi trường
Đã có sẵn `.venv` (torch CPU + torch_geometric + sentence-transformers). Không cần cài gì thêm.
`run_all.sh` tự dùng `../.venv/bin/python`.

---

## Chạy

```bash
cd TA_GP  && ./run_all.sh      # tool attack
cd ../MA_GP && ./run_all.sh    # memory attack
cd ../PI_GP && ./run_all.sh    # prompt injection (3 dataset)
```

Mỗi script tự: nạp `.env` → preflight 3 backbone → tìm checkpoint mới nhất →
defense star (v1 + v2) → evaluate → append `../evaluation_results.csv`.

**Luôn smoke trước khi chạy full lần đầu:**
```bash
./run_all.sh --smoke      # 2 sample, ~4 phút, <$0.10, ghi vào evaluation_results_SMOKE.csv
```

### Tuỳ chọn
| Cờ | Ý nghĩa |
|---|---|
| `--smoke` | 2 sample, CSV riêng |
| `--samples N` | đổi cỡ mẫu (mặc định 60) |
| `--topo X` | đổi topology (mặc định `star`) |
| `--backbones "a b"` | chọn backbone thủ công |

---

## Thời gian & chi phí

| Module | deepseek | haiku |
|---|---|---|
| TA_GP | ~1.6h | ~3.6h |
| MA_GP | ~1.6h | ~3.5h |
| PI_GP | ~4.9h | ~10.7h |

**~26h tuần tự**, ~15h nếu chạy 3 module song song ở 3 terminal (khác thư mục, an toàn).
Chi phí ~$168 cho cả 2 backbone mới.

Muốn nhanh hơn: tăng `LLM_MAX_CONCURRENCY` trong `.env` (mặc định 4) nếu quota Bedrock cho phép.

---

## Khi có sự cố

**Ngắt giữa chừng** → chạy lại đúng lệnh đó. Log sẽ in `RESUMED: đã có N sample`,
phần đã xong được bỏ qua (lưu trong `*.partial.jsonl`).

**Rate limit** → tự retry 6 lần (backoff + jitter, tôn trọng `Retry-After`).
Nếu vẫn hỏng >10% sample, run đó **dừng với exit 2** và in
`❌ KẾT QUẢ KHÔNG DÙNG ĐƯỢC` thay vì âm thầm ghi CSV thiếu mẫu.
Chỉnh ngưỡng: `MAX_SKIP_RATIO` trong `.env`.

**Một backbone/dataset hỏng** → chỉ cái đó bị skip, phần còn lại chạy tiếp;
cuối log in danh sách bị skip.

---

## ⚠️ Hai lỗi vận hành dễ mắc

1. **Không chạy 2 `run_all.sh` cùng một module song song** — chúng ghi đè nhau và
   gây lỗi `❌ v1 không ra file` trông y như bug. Song song **khác module** thì an toàn.

2. **Đừng dùng `pkill -f "run_all.sh"`** — `-f` khớp cả command line của chính shell
   đang gõ lệnh, nên nó tự giết mình (exit 144). Dùng `pkill -f main_defense_for_diff`.

---

## Kiểm tra kết quả

```bash
# mỗi dòng phải có num_samples đồng nhất giữa các backbone
awk -F, 'NR>1{print $2, $3, $4, "n="$6}' evaluation_results.csv | sort | uniq -c
```
Nếu `num_samples` khác nhau giữa các backbone → có sample bị drop, **không so sánh trực tiếp được**.
