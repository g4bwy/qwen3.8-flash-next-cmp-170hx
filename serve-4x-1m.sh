#!/usr/bin/env bash
# 1M context variant of the (now archived) 4-card 262k script. YaRN factor
# 4.0, the vendor's own recipe. Expect "Using max model len 1000000" for the
# drafter too. The override rules and the patch 4 startup check are in
# archive/serve-4x-512k.sh.
# Apply the patch series first: git am --keep-non-patch patches/00*.patch
# Details and measured numbers: README.md.
#
# PP=4 with fp8 KV: the QSA layers store e4m3 bytes in uint8 and decode them
# in-kernel on sm_80 (upstream #60602 + #60992), so the pool roughly doubles,
# 2.55M tokens -> 4.90M at matched settings, four 1M streams resident where a
# third used to evict the first. There is no prefill price for it. The
# 76.4 -> 85.5 s "10% tax" measured 2026-10-25 was a chunking artifact: a
# one-byte cache halves bytes per token, the hybrid page planner doubles the
# attention group block (1616 -> 3232 tokens, see `kv cache group sizes` in
# the boot log), and at the default 2048-token chunk budget the chunk ends
# stop landing on that block. Pass --max-num-batched-tokens 9728 (appended
# args win over the defaults here) and every dtype chunks at 9696 aligned
# rows: fp8 lands at 62.7 s against bf16's 64.0, the fastest prefill measured
# on this box (results.md, 2026-10-26). The scripts keep the shipped default
# until the cert battery clears: concurrency cells and preemption replay at
# fatter chunks are unmeasured. Unrelated to all of this, patch 8's int8
# activation hop moves about 12 s at either KV dtype, the two levers are
# orthogonal, the hop's gain is already banked here.
# A Mamba align-mode preemption still replays its whole prompt (~2 min), watch the
# preempt column and not only capacity.
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
# 0.94 sat between the two; since base 183625d4f9 the memory profiler
# budgets CUDA graph memory before sizing the pool, so 0.9453 restores the
# same effective pool the old 0.94 had (boot log prints the per-rank
# equivalence). expandable_segments is the first defense against
# the second mode.
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
# P2P works on these cards (CUMEM, driver-granted) and wins 2-8% per hop
# between two idle GPUs - but on the real PP=4 boot it measured SLOWER:
# cold 1M TTFT 94.7/94.9 s vs 91.2-92.1 s with staging (2026-10-16).
# Three full-duplex pipeline communicators plus the PLE's pinned-table
# DMA make the root complex a worse place for peer windows than for CPU
# staging copies. Keep the default disabled; the ${VAR:-1} form means an
# outer export overrides without editing the script, so future A/Bs are
# free. NCCL_IB_DISABLE keeps comm off-box regardless.
export NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1} NCCL_IB_DISABLE=1
# Big transient shapes (248k-vocab logits) hit allocator-OOM retries on
# near-full cards, and expandable segments serve them from fresh segments.
# vLLM forbids this only with KV connectors, and we run none.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The 173 GiB checkpoint is read once into VRAM; drop its page cache as each
# shard is consumed so the load does not evict cache worth keeping. Requires
# patches/0006-Model-Loader-Drop-consumed-shard-pages-from-the-page.patch.
export VLLM_SAFETENSORS_DROP_CACHE=1

# Ship the pipeline handoff as per-token per-stream int8 instead of the bf16
# 4-stream hyper-connection state: halves hop bytes on a link that is the
# prefill wall (cold 1M TTFT 91.8 -> 79.9 s). Series patch 8 reads this env;
# one export here covers all four stages because this script execs a single
# process tree. Without the env the engine is byte-identical to pre-patch.
export VLLM_QWEN4EXP_PP_INT8_HOP=${VLLM_QWEN4EXP_PP_INT8_HOP:-1}

# Series patches 11-13: int8 for the draft head, the dense linears, and patch 13's
# floor that lets the dense arm serve the captured decode buckets. +3-5% single-
# stream decode and -5% cold TTFT, quality at the within-boot floor (results.md,
# 2026-10-24). Rollback: set these three to 0/512 or drop the exports.
export VLLM_QWEN4EXP_DRAFT_W8A8=${VLLM_QWEN4EXP_DRAFT_W8A8:-1}
export VLLM_QWEN4EXP_DENSE_W8A8=${VLLM_QWEN4EXP_DENSE_W8A8:-1}
export VLLM_QWEN4EXP_DENSE_W8A8_MIN_M=${VLLM_QWEN4EXP_DENSE_W8A8_MIN_M:-1}

# RUST=1: experimental Rust frontend (vllm-rs) in place of the Python API
# server. Rollback: boot without it. Parity audit: results.md, 2026-10-20.
if [ "${RUST:-0}" = "1" ]; then
    export VLLM_USE_RUST_FRONTEND=1
fi

# PROF=1 arms the torch profiler (auto-stops after max_iterations engine
# steps, needs no code patch): boot with PROF=1, then POST /start_profile
# mid-decode. Traces land in logs/prof NEXT TO THIS SCRIPT'S OWN COPY
# (dirname of $0). Note: a copy under /ssd is invisible to the container;
# move the trace files into the repo afterwards, or set PROF_DIR to a
# path both sides see. Do not widen the window beyond ~150 steps -
# results.md, unbounded runs swap the machine.
PROF_ARGS=()
if [ "${PROF:-0}" = "1" ]; then
    export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1  # gpu_model_runner phase spans
    PROF_DIR="${PROF_DIR:-$(cd "$(dirname "$0")" && pwd)/logs/prof}"
    mkdir -p "$PROF_DIR"
    PROF_ARGS=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"$PROF_DIR\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true,\"max_iterations\":150}")
fi

# Experiment, not a series patch: cap one request's prefill rows per step so an
# arriving long prefill shares the pipeline with decode. Inert when only one request
# is eligible (scheduler.py:733-739), so single-stream TTFT is untouched by design.
# Unset or 0 keeps the engine default. Rollback: drop LPT from the command line.
LPT_ARGS=()
if [ -n "${LPT:-}" ] && [ "${LPT:-0}" != "0" ]; then
    case "$LPT" in
        *[!0-9]*) echo "serve-4x-1m.sh: LPT must be a non-negative integer, got '$LPT'" >&2
                  exit 1 ;;
    esac
    LPT_ARGS=(--long-prefill-token-threshold "$LPT")
fi

# Flag-only A/B of the QSA index cache. The selection scan streams the compressed
# key cache, and the lane default is "auto", which the QSA indexer resolves to bf16
# (nvidia/indexer_qsa.py:152-163 accepts only "bf16" or "fp8"). IKD=fp8 stores those
# rows as e4m3, halving the bytes the scan reads at a cost of selection accuracy, so
# it must be certified with the fixed-salt accept window, not on speed alone. The
# index cache is not the KV pool, so the pool line should not move. Unset keeps the
# shipped behaviour. Rollback: drop IKD.
IKD_ARGS=()
if [ -n "${IKD:-}" ]; then
    case "$IKD" in
        bf16|fp8) ;;
        *) echo "serve-4x-1m.sh: IKD must be bf16 or fp8, got '$IKD'" >&2; exit 1 ;;
    esac
    IKD_ARGS=(--attention-config "{\"indexer_kv_dtype\":\"$IKD\"}")
fi

# The default prefix match granularity is the GCD of the cacheable KV group
# block sizes, 1600 here, so a hit could not land finer than 1,600 tokens no
# matter how often a prompt repeats. --prefix-match-unit 16 drops the floor
# to 32 tokens with the MTP block drop: repeats and appends hit 99.8%+, the
# one-shot shallow anomaly is gone, pool and decode unchanged. Retention
# controls how often states are stored; this controls where hits land.
exec vllm serve "${MODEL:-Qwen/Qwen3.8-Flash-Next-FP8}" \
    ${PROF_ARGS[@]+"${PROF_ARGS[@]}"} \
    ${LPT_ARGS[@]+"${LPT_ARGS[@]}"} \
    ${IKD_ARGS[@]+"${IKD_ARGS[@]}"} \
    --served-model-name qwen3.8-flash-next-fp8 \
    --port 8000 \
    --pipeline-parallel-size 4 \
    --engram-config '{"cpu_offload": true, "host_file_gather": true, "host_file_gather_mmap": true}' \
    --moe-backend humming \
    --enable-prefix-caching \
    --mamba-cache-mode align \
    --prefix-cache-retention-interval 16160 \
    --block-size 1616 \
    --prefix-match-unit 16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.9453 \
    --max-model-len 1000000 \
    --hf-overrides '{"text_config":{"max_position_embeddings":1000000,"rope_parameters":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}}}' \
    --max-num-seqs 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
    -cc.cudagraph_mode=FULL_AND_PIECEWISE \
    --no-enable-flashinfer-autotune \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3 \
    "$@"
