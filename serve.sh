#!/usr/bin/env bash
# Qwen3.8-Flash-Next on 3x unlocked CMP 170HX: pipeline parallel 3, MTP-3
# spec decode, PLE n-gram table offloaded to pinned host RAM.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2

# Rank 0 must cover every PLE layer and nothing more. The n-gram table lives
# in host RAM, so the other ranks carry the layers.
export VLLM_PP_LAYER_PARTITION=16,17,15
# No P2P between these cards. Every hop goes over the host bridge anyway.
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# Big transient shapes (248k-vocab logits) hit allocator-OOM retries on
# near-full cards, and expandable segments serve them from fresh segments.
# vLLM forbids this only with KV connectors, and we run none.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve "${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    --served-model-name qwen3.8-flash-next-fp8 \
    --port 8000 \
    --pipeline-parallel-size 3 \
    --engram-config '{"cpu_offload": true}' \
    --moe-backend humming \
    --enable-prefix-caching \
    --gpu-memory-utilization 0.94 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    "$@"
