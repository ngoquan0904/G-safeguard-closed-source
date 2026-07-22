#!/bin/bash
# PI_GP — defense trên 1 topology (star) × nhiều backbone × 3 dataset.
# KHÔNG gen data, KHÔNG train: dùng lại test dataset + checkpoint đã có.
# Chạy:  ./run_all.sh        (thêm --smoke để test nhanh 2 sample)
set -uo pipefail
cd "$(dirname "$0")"
MODULE="PI_GP"
ATTACK="pi_escalation"          # bị ghi đè thành pi_{dataset}_escalation
CKPT_DIR="checkpoint"           # thực tế dùng checkpoint/{dataset}
DATASETS=("mmlu" "csqa" "gsm8k")
source ../scripts_common.sh
