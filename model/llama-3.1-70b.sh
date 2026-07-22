#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m vllm.entrypoints.openai.api_server \
    --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 \
    --tensor-parallel-size 4 \
    --gpu-memory-utilization 0.75 \
    --swap-space 16 \
    --max-model-len 16384
