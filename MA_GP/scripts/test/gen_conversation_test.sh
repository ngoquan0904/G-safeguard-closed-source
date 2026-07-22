#!/bin/bash
# Sinh dữ liệu hội thoại cho phase TEST — chạy song song mọi cấu hình.
# attackers 3 x sparsity {0.2,0.4,0.6,0.8,1.0} = 5 process chạy đồng thời.

MODEL_TYPE="hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4"

for attackers in 3; do
  for sparsity in 0.2 0.4 0.6 0.8 1.0; do
    python gen_graph.py \
      --num_nodes 8 \
      --sparsity $sparsity \
      --num_graphs 20 \
      --num_attackers $attackers \
      --samples 12 \
      --model_type "$MODEL_TYPE" \
      --phase test &
  done
done

wait
echo "=== TEST generation done ==="
