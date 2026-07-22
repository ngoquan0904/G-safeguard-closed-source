#!/bin/bash
# TA_GP — defense trên 1 topology (star) × nhiều backbone.
#
# KHÔNG gen data, KHÔNG train: dùng lại test dataset + checkpoint đã có
# (đúng thiết kế "tái dùng checkpoint Llama, chỉ đổi backbone lúc defense").
#
# Chỉ cần chạy:  ./run_all.sh
#   ./run_all.sh --smoke              chạy nhanh 2 sample để kiểm luồng
#   ./run_all.sh --backbones "a b"    chọn backbone thủ công

set -uo pipefail
cd "$(dirname "$0")"

MODULE="TA_GP"
ATTACK="tool_attack_escalation"
CKPT_DIR="checkpoint/tool_attack"
DATASETS=("")                     # TA_GP chỉ có 1 dataset -> chuỗi rỗng
source ../scripts_common.sh
