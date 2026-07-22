"""
Tải sẵn data cho csqa + gsm8k về đúng layout mà PI_GP mong đợi.
Idempotent: đã có thì bỏ qua.

Yêu cầu: pip install pyarrow datasets huggingface_hub
Chạy:    python download_datasets.py
"""
import os
import sys

# ── csqa: gen_csqa.py đọc parquet ở ./datasets/commonsense_qa/data/*.parquet ──
CSQA_DIR = "./datasets/commonsense_qa/data"
CSQA_FILES = {
    "train": os.path.join(CSQA_DIR, "train-00000-of-00001.parquet"),
    "validation": os.path.join(CSQA_DIR, "validation-00000-of-00001.parquet"),
    "test": os.path.join(CSQA_DIR, "test-00000-of-00001.parquet"),
}

# ── gsm8k: gen_gsm8k.py gọi load_dataset("./datasets/gsm8k", "main") ──
GSM8K_DIR = "./datasets/gsm8k"


def download_csqa():
    if all(os.path.exists(p) for p in CSQA_FILES.values()):
        print("[csqa] đã có đủ parquet, bỏ qua.")
        return
    from datasets import load_dataset
    os.makedirs(CSQA_DIR, exist_ok=True)
    print("[csqa] tải tau/commonsense_qa ...")
    ds = load_dataset("tau/commonsense_qa")
    for split, path in CSQA_FILES.items():
        ds[split].to_parquet(path)
        print(f"[csqa]   -> {path} ({ds[split].num_rows} rows)")
    print("[csqa] xong.")


def download_gsm8k():
    main_dir = os.path.join(GSM8K_DIR, "main")
    if os.path.isdir(main_dir) and any(f.endswith(".parquet") for f in os.listdir(main_dir)):
        print("[gsm8k] đã có parquet ở main/, bỏ qua.")
        return
    from huggingface_hub import snapshot_download
    os.makedirs(GSM8K_DIR, exist_ok=True)
    print("[gsm8k] tải openai/gsm8k ...")
    snapshot_download(
        repo_id="openai/gsm8k",
        repo_type="dataset",
        local_dir=GSM8K_DIR,
        allow_patterns=["main/*", "README.md", ".gitattributes"],
    )
    print("[gsm8k] xong.")


if __name__ == "__main__":
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        print("❌ Thiếu 'pyarrow'. Chạy: pip install pyarrow datasets huggingface_hub")
        sys.exit(1)

    ok = True
    for name, fn in [("csqa", download_csqa), ("gsm8k", download_gsm8k)]:
        try:
            fn()
        except Exception as e:
            ok = False
            print(f"⚠️  [{name}] tải thất bại: {e}")
    print("✅ download_datasets done." if ok else "⚠️  download_datasets có lỗi (xem trên).")
