#!/usr/bin/env bash
# Qwen3.8-Flash-Next on 4x unlocked CMP 170HX: pipeline parallel 4, MTP-3
# spec decode, PLE n-gram table offloaded to pinned host RAM.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md. The 1M and 512k lanes are booted
# and benched (results.md, 4-card set); this 262k lane awaits a boot.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# The model has a single PLE layer, at decoder index 1, so any split with
# at least 2 layers on rank 0 satisfies the PLE-on-rank-0 rule. Balance
# the layer counts instead: the KV pool caps at min_r (rank KV bytes /
# rank layer count), and a rank's KV bytes shrink as its layer count
# grows. Measured with 16,12,11,9 (boot-4x-1m.log): rank 0 stranded at
# 12.76 GiB of KV for 16 layers while rank 3 sat on 28.13 GiB for 9,
# capping the pool at 1.568M tokens. The equal 12s measure 2,541,795
# tokens / 2.54x (+62%). Rank 3 is now the tightest (12 layers + the MTP
# drafter's 3.74 GiB); shifting a layer there costs the donor rank more
# than rank 3 gains, so 12,12,12,12 is the practical optimum.
export VLLM_PP_LAYER_PARTITION=12,12,12,12
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
