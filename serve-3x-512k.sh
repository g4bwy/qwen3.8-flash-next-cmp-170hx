#!/usr/bin/env bash
# 512k context variant of serve-3x.sh. The checkpoint is native to 262144
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
#    print "Using max model len 524288" twice, target then draft. A second
#    262144 there means the drafter runs unscaled RoPE.
#
# The KV pool does not grow with the context. Worst-case concurrency is 2.30x
# on this box. More than about two long requests in flight preempt and
# re-prefill.
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
