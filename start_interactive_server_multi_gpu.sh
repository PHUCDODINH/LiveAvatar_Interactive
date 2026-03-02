#!/bin/bash
# Interactive LiveAvatar Server - Multi-GPU Launch Script (TPP mode)
# This script runs the interactive server using 5 GPUs (1 VAE + 4 DiT).

export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4}
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=OFF

echo "=========================================="
echo "Starting Interactive Avatar Server in Multi-GPU mode"
echo "=========================================="

export ENABLE_COMPILE=true
export ENABLE_FP8=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the interactive server with 5 processes
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun \
    --nproc_per_node=5 \
    --master_port=29505 \
    interactive_avatar_server.py
