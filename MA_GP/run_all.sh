#!/bin/bash
# MA_GP — defense trên 1 topology (star) × nhiều backbone.
# KHÔNG gen data, KHÔNG train: dùng lại test dataset + checkpoint đã có.
# Chạy:  ./run_all.sh        (thêm --smoke để test nhanh 2 sample)
set -uo pipefail
cd "$(dirname "$0")"
MODULE="MA_GP"
ATTACK="memory_attack_escalation"
CKPT_DIR="checkpoint/memory_attack_v2"
DATASETS=("")
source ../scripts_common.sh
