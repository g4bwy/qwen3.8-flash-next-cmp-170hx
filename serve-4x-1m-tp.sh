#!/usr/bin/env bash
# TP=4 + EP variant of serve-4x-1m.sh. Same 1M window, same YaRN factor 4.0,
# same MTP-3 drafter and prefix-caching flags. The parallel layout changes:
# every card holds a quarter of every layer instead of a quarter of the layers.
#
# Why try this on a box with no NVLink: PP=4 moves one hidden-state tensor per
# stage boundary per step, so it barely exercises inter-card communication, and
# P2P measured flat to slightly negative (2026-10-09, bench-4x-1m-p2p-n1.json:
# prefill -2 to -3%, decode unchanged). TP=4 puts about two collectives per
# layer in the step path, roughly 100 collectives per decode step, so it is the
# layout where a peer-memory all-reduce could finally pay off.
#
# First boot 2026-10-09 14:24 (boot-4x-1m-tp4.log): clean, weights 33.59 GiB
# per rank on all four, KV pool 1,741,935 tokens / 1.74x against PP=4 2.60x,
# attention block size 800 against 1600. The pool is the cost of TP here: the
# QSA indexer cache is replicated on every rank. Still above the 1M window, so
# one full-length request fits; two no longer do.
#
# What has to go well, and what to read in the log:
#   1. NCCL must be pushed onto peer DMA. `nvidia-smi topo -m` reports every
#      GPU pair here as NODE: PCIe plus the interconnect between PCIe host
#      bridges inside one NUMA node. NCCL grants peer DMA for 2-rank comms at
#      its default level, which is why PP=4 showed `P2P/CUMEM` on rank 0's hop,
#      but it refuses it for the collective rings a TP group uses: at TP=4 all
#      eight channels came up `isAllDirectP2p 0` / `via SHM/direct/direct`.
#      `NCCL_P2P_LEVEL=SYS` below is the setting a 2-card CMP 170HX box at the
#      same NODE distance needed; it reported 1.8 to 1.9x faster TP prefill
#      with P2P than without. Proof to read: `isAllDirectP2p 1` and
#      `via P2P/` on the 4-rank channels.
#      vLLM's own custom all-reduce is not an option at TP=4 regardless of the
#      driver: custom_all_reduce.py:236 turns it off for
#      `world_size > 2 and not fully_connected`, and fully_connected is an
#      NVLink mesh test. It is also the backend that breaks across root
#      complexes once peer access exists (see the --disable-custom-all-reduce
#      note in the args), so do not chase it.
#   2. EP is live: 512 routed experts split 128 per rank, and humming moved from
#      "indexed" to the "grouped_contiguous" GEMM path (confirmed by
#      `Using grouped_contiguous gemm for humming moe`, fused_humming_moe.py:111).
#      Grouped was never cleanly compared with indexed at decode M on these
#      cards, so a decode change here is two variables at once: collectives and
#      GEMM type.
#   3. KV pool cost of TP. QSA shards KV only when num_key_value_heads divides
#      TP; below that it replicates (qsa.py:213-219). The QSA indexer cache is
#      ReplicatedLinear, so it is always full size on every rank. The log cannot
#      say which of the two bit; either way the pool is 1.74x instead of 2.60x.
#   4. The PLE n-gram table shards over the ETP group and adds one all-reduce
#      per step for its rows (ngram_embedding.py:481). One layer, small
#      payload, but it is a new sync point that PP did not have.
#   5. BLOCKER for TP > 1 as the series stands: patch 8 builds its draft head
#      as a sharded ParallelLMHead, so each rank returns 39,532/4 columns, and
#      expand_draft_logits() then scatters those into a full-width row. The
#      shapes disagree and the first draft step dies after the load. Either
#      hide the artifacts for this lane,
#        mv mtp_draft_vocab_ids.pt mtp_draft_vocab_ids.pt.off
#      which gives up the +8.7% single-stream the patch is worth and makes the
#      decode comparison against the PP table unfair, or give the head
#      disable_tp=True in patches/0008 (a no-op at TP=1, so PP lanes are
#      unaffected) and re-apply the series.
#      The 14:24 boot printed no `MTP drafter scores a ... draft vocabulary`
#      line, so that lane is running the full head: its decode sits about 8.7%
#      below the PP table before TP even enters the picture.
#
# Result 2026-10-09: not worth it on this box, kept for reference. With P2P on,
# prefill 1,434 / 1,456 tok/s against PP=4's 9,012 / 9,750 at the same depths,
# and single-stream decode 111.7 against 185.1 tok/s. P2P did its part: +20%
# prefill, -17% TTFT, +8% decode versus the same lane over host SHM. It stops
# there because GPU0/2/3 negotiate Gen2 x4, so a 4-rank ring moves about 25 MB
# per collective at roughly 2 GB/s and the lane sits at its wire limit. The same
# cards at x16 gave a reference box 1.8 to 1.9x faster TP prefill with P2P.
# Full tables: benchmark/results.md, "NCCL P2P, and TP=4 + EP". Boot this lane
# again only if the links become x16, or after making patch 8 TP-safe
# (disable_tp=True), which is the other thing holding these numbers down.
#
# Stop the PP server first, or override the port for a side by side boot:
#   ./serve-4x-1m-tp.sh --port 8001
set -euo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3

# PP-only knob, deliberately absent: VLLM_PP_LAYER_PARTITION.
# P2P stays enabled, that is the point of this lane. NCCL picks P2P/CUMEM for
# the collectives and falls back to host SHM if peer access is refused; check
# with NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TRANSPORT on the first boot.
export NCCL_IB_DISABLE=1
# Cross-root P2P for the collective rings. Without this NCCL stages every
# 4-rank message through host shared memory, and at TP=4 that is roughly 100
# collectives per decode step from four processes all memcpy-ing on cores 0-31.
export NCCL_P2P_LEVEL=SYS
# Required once peer access exists. vLLM's custom all-reduce allocates its own
# CUDA-IPC peer buffers; on a NODE-distance box that fails with "invalid
# argument", OOMs VRAM and crash-loops the workers, while NCCL keeps its P2P
# path, so nothing is lost. At TP=4 this flag only silences a warning, since
# custom all-reduce is already refused by the world-size gate.
export VLLM_ALLREDUCE_USE_FLASHINFER_PCIE_IPC=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_SAFETENSORS_DROP_CACHE=1

exec vllm serve "${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    --served-model-name qwen3.8-flash-next-fp8 \
    --port 8000 \
    --tensor-parallel-size 4 \
    --enable-expert-parallel \
    --disable-custom-all-reduce \
    --engram-config '{"cpu_offload": true}' \
    --moe-backend humming \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --prefix-cache-retention-interval 16000 \
    --prefix-match-unit 16 \
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
