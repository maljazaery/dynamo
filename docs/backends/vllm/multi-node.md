<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Multi-node Examples

This guide covers deploying vLLM across multiple nodes using Dynamo's distributed capabilities.

## Prerequisites

Multi-node deployments require:
- Multiple nodes with GPU resources
- Network connectivity between nodes (faster the better)
- Firewall rules allowing NATS/ETCD communication

## Infrastructure Setup

### Step 1: Start NATS/ETCD on Head Node

Start the required services on your head node. These endpoints must be accessible from all worker nodes:

```bash
# On head node (node-1)
docker compose -f deploy/docker-compose.yml up -d
```

Default ports:
- NATS: 4222
- ETCD: 2379

### Step 2: Configure Environment Variables

Set the head node IP address and service endpoints. **Set this on all nodes** for easy copy-paste:

```bash
# Set this on ALL nodes - replace with your actual head node IP
export HEAD_NODE_IP="<your-head-node-ip>"

# Service endpoints (set on all nodes)
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"
```

## Deployment Patterns

### Multi-node Aggregated Serving

Deploy vLLM workers across multiple nodes for horizontal scaling:

**Node 1 (Head Node)**: Run ingress and first worker
```bash
# Start ingress
python -m dynamo.frontend --router-mode kv

# Start vLLM worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager
```

**Node 2**: Run additional worker
```bash
# Start vLLM worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager
```

### Multi-node Disaggregated Serving

Deploy prefill and decode workers on separate nodes for optimized resource utilization:

**Node 1**: Run ingress and decode worker
```bash
# Start ingress
python -m dynamo.frontend --router-mode kv &

# Start prefill worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --is-decode-worker
```

**Node 2**: Run prefill worker
```bash
# Start decode worker
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --enforce-eager \
  --is-prefill-worker
```

### Multi-node Tensor/Pipeline Parallelism

When the total world size (`tensor_parallel_size × pipeline_parallel_size`) exceeds the number of GPUs on a single node, you need to span vLLM across multiple machines. This uses vLLM's native multiprocessing (`mp`) backend with the `--headless` flag for worker nodes.

This applies whenever `TP × PP > GPUs on a single node` — for example, TP=16 across two 8-GPU nodes, or TP=8 with PP=2 across two 8-GPU nodes.

**How it works:**
- The **head node** (node-rank 0) runs the full `dynamo.vllm` process with the Dynamo frontend, engine core, and scheduler
- **Worker nodes** run `dynamo.vllm --headless`, which launches vLLM workers only — no Dynamo endpoints, no NATS/etcd connections
- All nodes coordinate via `torch.distributed` using the `mp` backend

**Infrastructure requirements:**
- The head node still needs NATS and etcd for the Dynamo frontend
- Worker nodes only need network connectivity to the head node for `torch.distributed`
- The model must be downloaded on all nodes (each node loads weights locally)

For more details on vLLM's multi-node multiprocessing, see the
[vLLM parallelism docs](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/?h=#running-vllm-with-multiprocessing).

#### Example: TP=16 across 2× 8-GPU nodes

Pure tensor parallelism spanning two nodes:

**Node 1 (Head Node)**: Run ingress and vLLM engine
```bash
# Start ingress
python -m dynamo.frontend --router-mode kv &

# Start vLLM head (node-rank 0)
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 16 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr ${HEAD_NODE_IP} \
  --master-port 29500 \
  --enforce-eager
```

**Node 2 (Worker)**: Run headless vLLM worker
```bash
python -m dynamo.vllm --headless \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 16 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr ${HEAD_NODE_IP} \
  --master-port 29500 \
  --enforce-eager
```

#### Example: TP=8, PP=2 across 2× 8-GPU nodes

Pipeline parallelism across nodes with tensor parallelism within each node:

**Node 1 (Head Node)**: Run ingress and vLLM engine
```bash
# Start ingress
python -m dynamo.frontend --router-mode kv &

# Start vLLM head (node-rank 0)
python -m dynamo.vllm \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 0 \
  --master-addr ${HEAD_NODE_IP} \
  --master-port 29500 \
  --enforce-eager
```

**Node 2 (Worker)**: Run headless vLLM worker
```bash
python -m dynamo.vllm --headless \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --tensor-parallel-size 8 \
  --pipeline-parallel-size 2 \
  --distributed-executor-backend mp \
  --nnodes 2 \
  --node-rank 1 \
  --master-addr ${HEAD_NODE_IP} \
  --master-port 29500 \
  --enforce-eager
```
