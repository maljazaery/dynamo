#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -e
trap 'echo Cleaning up...; kill 0' EXIT

# Default values
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
ENABLE_PREFIX_CACHE=true
# Multimodal embedding cache: not set = disabled (0). Set via --multimodal-embedding-cache-capacity-gb to enable.
PD_EMBEDDING_CACHE_ARGS=""
SINGLE_GPU=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_NAME=$2
            shift 2
            ;;
        --single-gpu)
            SINGLE_GPU=true
            shift
            ;;
        --no-prefix-cache)
            ENABLE_PREFIX_CACHE=false
            shift
            ;;
        --multimodal-embedding-cache-capacity-gb)
            if [[ -n "${2:-}" && "$2" =~ ^[0-9]+\.?[0-9]*$ ]]; then
                PD_EMBEDDING_CACHE_ARGS="--multimodal-embedding-cache-capacity-gb $2"
                shift 2
            else
                echo "ERROR: --multimodal-embedding-cache-capacity-gb requires a positive number (GB)." >&2
                exit 1
            fi
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Disaggregated multimodal serving with separate Encode and aggregated PD worker"
            echo ""
            echo "Options:"
            echo "  --model <model_name>          Specify the VLM model to use (default: $MODEL_NAME)"
            echo "                                LLaVA 1.5 7B, Qwen2.5-VL, and Phi3V models have predefined templates"
            echo "  --single-gpu                  Run encode and PD workers on the same GPU (for small models, e.g. 2B)"
            echo "  --no-prefix-cache             Disable prefix caching on the PD worker (default: enabled)"
            echo "  --multimodal-embedding-cache-capacity-gb <gb>  Enable embedding cache on PD worker, capacity in GB (default: disabled)"
            echo "  -h, --help                    Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --model llava-hf/llava-1.5-7b-hf"
            echo "  $0 --model microsoft/Phi-3.5-vision-instruct"
            echo "  $0 --model Qwen/Qwen2.5-VL-7B-Instruct"
            echo "  $0 --model $MODEL_NAME --no-prefix-cache  # no prefix cache, no embedding cache"
            echo "  $0 --model $MODEL_NAME --multimodal-embedding-cache-capacity-gb 2"
            echo "  $0 --model Qwen/Qwen2-VL-2B-Instruct --single-gpu  # single GPU for encode + PD"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# PD worker: pass --no-enable-prefix-caching only when prefix cache is disabled
if [[ "$ENABLE_PREFIX_CACHE" == "true" ]]; then
    PD_PREFIX_CACHE_ARGS=""
else
    PD_PREFIX_CACHE_ARGS="--no-enable-prefix-caching"
fi

# PD worker max sequence length (encoder cache budget follows this). Override via PD_MAX_MODEL_LEN.
PD_MAX_MODEL_LEN=${PD_MAX_MODEL_LEN:-16384}


echo "=================================================="
echo "Disaggregated Multimodal Serving (E + PD)"
echo "=================================================="
echo "Model: $MODEL_NAME"
echo "=================================================="


# Start frontend (no router mode)
echo "Starting frontend..."
python -m dynamo.frontend &

EXTRA_ARGS=""

# Embedding transfer: 1 = local file (safetensors), 0 = NIXL RDMA
export TRANSFER_LOCAL=${TRANSFER_LOCAL:-1}

# GPU assignments (override via environment variables)
if [[ "$SINGLE_GPU" == "true" ]]; then
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-0}
    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-0}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.4}
    DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.4}
    EXTRA_ARGS="--enforce-eager"
else
    DYN_ENCODE_WORKER_GPU=${DYN_ENCODE_WORKER_GPU:-1}
    DYN_PD_WORKER_GPU=${DYN_PD_WORKER_GPU:-2}
    DYN_ENCODE_GPU_MEM=${DYN_ENCODE_GPU_MEM:-0.9}
    DYN_PD_GPU_MEM=${DYN_PD_GPU_MEM:-0.9}
fi

# Start encode worker
echo "Starting encode worker on GPU $DYN_ENCODE_WORKER_GPU (GPU mem: $DYN_ENCODE_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20097 \
CUDA_VISIBLE_DEVICES=$DYN_ENCODE_WORKER_GPU \
python -m dynamo.vllm \
  --multimodal-encode-worker \
  --enable-multimodal \
  --model "$MODEL_NAME" \
  --gpu-memory-utilization "$DYN_ENCODE_GPU_MEM" \
  $EXTRA_ARGS \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20080"}' &

# Start PD worker (aggregated prefill+decode, routes to encoder for embeddings)
echo "Starting PD worker on GPU $DYN_PD_WORKER_GPU (GPU mem: $DYN_PD_GPU_MEM)..."
VLLM_NIXL_SIDE_CHANNEL_PORT=20098 \
CUDA_VISIBLE_DEVICES=$DYN_PD_WORKER_GPU \
python -m dynamo.vllm \
  --route-to-encoder \
  --multimodal-worker \
  --enable-multimodal \
  --enable-mm-embeds \
  --model "$MODEL_NAME" \
  --max-model-len "$PD_MAX_MODEL_LEN" \
  --gpu-memory-utilization "$DYN_PD_GPU_MEM" \
  $PD_PREFIX_CACHE_ARGS \
  $PD_EMBEDDING_CACHE_ARGS \
  $EXTRA_ARGS \
  --kv-events-config '{"publisher":"zmq","topic":"kv-events","endpoint":"tcp://*:20081"}' &

echo "=================================================="
echo "All components started. Waiting for initialization..."
echo "=================================================="

# Wait for all background processes to complete
wait
