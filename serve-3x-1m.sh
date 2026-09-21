#!/usr/bin/env bash
# 1M context variant of serve-3x.sh. YaRN factor 4.0, the vendor's own recipe.
# The override rules and the patch 9 startup check are in serve-3x-512k.sh.
# Expect "Using max model len 1000000" twice at startup.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md.
#
# This is the box ceiling. Measured concurrency is 1.21x: one 1M request at a
# time. A second long request preempts the first, and Mamba align mode makes
# the preempted request replay its whole prompt, about 2 min. Keep other
# traffic short.
#
# YaRN costs VRAM before the pool is sized. The cos/sin cache is 1 GiB per
# rank at factor 4.0, and it comes out of the KV pool. Two failure modes
# point opposite ways:
#   "To serve at least one request with the model's max seq len (1000000)"
#     -> the pool is too small. Raise --gpu-memory-utilization, or use 768k
#        (about 1.6x) instead.
#   OOM retries repeating during inference
#     -> transient peaks have nowhere to go. Lower
#        --gpu-memory-utilization, or drop --speculative-config.
# 0.94 sits between the two. expandable_segments is the first defense against
# the second mode.
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
    --max-model-len 1000000 \
    --hf-overrides '{"text_config":{"max_position_embeddings":1000000,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
    --max-num-seqs 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    "$@"
