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
# 3. Patch 9 forwards the override to the MTP drafter. The startup log must
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
