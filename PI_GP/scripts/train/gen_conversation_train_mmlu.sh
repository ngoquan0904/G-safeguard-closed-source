#!/bin/bash
# TRAIN data (mmlu) — chạy song song mọi cấu hình (attackers 1..4 x sparsity).
MODEL_TYPE="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"
for attackers in 1 2 3 4; do
  for sparsity in 0.2 0.4 0.6 0.8 1.0; do
    python gen_graph.py --num_nodes 8 --sparsity $sparsity --num_graphs 20 --num_attackers $attackers --samples 40 --model_type "$MODEL_TYPE" --phase train --dataset mmlu &
  done
done
wait
echo "=== TRAIN mmlu done ==="
