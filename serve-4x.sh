#!/usr/bin/env bash
# Qwen3.8-Flash-Next on 4x unlocked CMP 170HX: pipeline parallel 4, MTP-3
# spec decode, PLE n-gram table offloaded to pinned host RAM.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md. All three 4-card lanes are booted
# and benched (results.md, 4-card set).
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# The KV pool caps at min_r (rank KV bytes / rank layer count), and a
# rank's KV bytes shrink as its layer count grows. The model has one PLE
# layer, at decoder index 1, so rank 0 needs at least 2 layers. Measured
# at 1M (boot-4x-1m.log): 12,12,12,12 pools 2,541,795 tokens / 2.54x;
# 13,12,12,11 pools 2,595,975 / 2.60x (+2.1%). The extra layer costs
# rank 0 2.53 GiB of weights while rank 3, which also carries the MTP
# drafter, gives the same back and gains the room. Rank 0 is now the
# binding stage, so a second shifted layer has nowhere useful to go.
export VLLM_PP_LAYER_PARTITION=13,12,12,11
# No P2P between these cards. Every hop goes over the host bridge anyway.
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
# Big transient shapes (248k-vocab logits) hit allocator-OOM retries on
# near-full cards, and expandable segments serve them from fresh segments.
# vLLM forbids this only with KV connectors, and we run none.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

exec vllm serve "${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    --served-model-name qwen3.8-flash-next-fp8 \
    --port 8000 \
    --pipeline-parallel-size 4 \
    --engram-config '{"cpu_offload": true}' \
    --moe-backend humming \
    --enable-prefix-caching \
    --prefix-cache-retention-interval 16000 \
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
