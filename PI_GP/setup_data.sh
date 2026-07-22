#!/bin/bash
# Cài thư viện cần thiết + tải data csqa/gsm8k. Chạy 1 lần trước run_all.sh.
set -e
pip install -q pyarrow datasets huggingface_hub
python download_datasets.py
echo "✅ setup_data done — giờ chạy: bash ./run_all.sh"
