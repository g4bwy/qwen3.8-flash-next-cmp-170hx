#!/usr/bin/env bash
# 512k context variant of serve-4x.sh. The checkpoint is native to 262144
# positions, so this adds static YaRN factor 2.0.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md.
#
# Three rules for the --hf-overrides below (why: README Gotchas):
# 1. rope_parameters must nest under text_config. The flat form from the
#    model card is a silent no-op on this checkpoint.
# 2. max_position_embeddings rises to 524288 in the same override. Since
#    vLLM #56446 the yarn limit is max_position_embeddings itself, and
#    original_max_position_embeddings stays mandatory.
# 3. Patch 7 forwards the override to the MTP drafter. The startup log must
#    print "Using max model len 524288" for the drafter too. A stray 262144
#    there means the drafter runs unscaled RoPE.
#
# The KV pool tracks the SMALLEST per-rank budget divided by that rank's
# share of the layer KV. PP=4 gives the pool a fourth card's worth of HBM:
# expect roughly 20-25% more tokens than the PP=3 pool of 1.20M, so about
# 2.8x worst-case concurrency instead of the measured 2.30x. Read the
# "GPU KV cache size" line at startup before trusting that estimate.
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# The KV pool caps at min_r (rank KV bytes / rank layer count), and a
# rank's KV bytes shrink as its layer count grows. PLE no longer
# constrains the split: patch 2 (upstream PR #56444) is pipeline-rank-free. Measured
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
# The 173 GiB checkpoint is read once into VRAM; drop its page cache as each
# shard is consumed so the load does not evict cache worth keeping. Requires
# patches/0009-safetensors-drop-pagecache.patch.
export VLLM_SAFETENSORS_DROP_CACHE=1

# --prefix-match-unit 16 drops the prefix hit floor from 1,600 tokens to 32.
# Measured rationale: serve-4x-1m.sh header and README, Prefix caching.
exec vllm serve "${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    --served-model-name qwen3.8-flash-next-fp8 \
    --port 8000 \
    --pipeline-parallel-size 4 \
    --engram-config '{"cpu_offload": true}' \
    --moe-backend humming \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --prefix-cache-retention-interval 16000 \
    --prefix-match-unit 16 \
    --gpu-memory-utilization 0.94 \
    --max-model-len 524288 \
    --hf-overrides '{"text_config":{"max_position_embeddings":524288,"rope_parameters":{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":262144}}}' \
    --max-num-seqs 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
    -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    "$@"
