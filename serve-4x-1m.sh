#!/usr/bin/env bash
# 1M context variant of serve-4x.sh. YaRN factor 4.0, the vendor's own
# recipe. Expect "Using max model len 1000000" for the drafter too.
# The override rules and the patch 9 startup check are in serve-4x-512k.sh.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md.
#
# PP=3 held 1.21x here: one 1M request at a time. PP=4 adds a fourth
# card's KV, roughly 1.4-1.5x by the pool math in serve-4x-512k.sh. A
# second full-length request then fits, but admission gating still runs it
# at half speed, and a Mamba align-mode preemption replays its whole
# prompt, about 2 min. Keep other traffic short until measured.
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
    --prefix-cache-retention-interval 16000 \
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
