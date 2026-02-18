#!/bin/bash
CUDA_VISIBLE_DEVICES=2 \
vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
    --enable-log-requests \
    --max-model-len 32768 \
    --gpu-memory-utilization .95
