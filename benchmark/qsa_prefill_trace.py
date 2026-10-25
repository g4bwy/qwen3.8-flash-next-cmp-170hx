#!/usr/bin/env python
"""Whole-step kernel attribution for a real Qwen4Exp prefill, offline, on host.

The lane runs int8 prefill at ~7.3k tok/s against bf16's 12.1k, a flat
per-token penalty, while every standalone int8 kernel measures within 6% of
bf16. This finds the missing 59 s: it boots the engine in-process (no HTTP,
nothing to crash), runs one real long prefill with vLLM's BUILT-IN torch
profiler armed, and aggregates the CUDA kernel times from the per-rank
chrome traces that the workers emit.

Why not wrap torch.profiler around generate(): in vLLM V1 the model runs in
EngineCore/worker subprocesses, so a frontend-process profiler records only
the frontend (283 us of socket waiting, zero kernels). The engine's own
profiler_config arms the workers.

Run ON THE HOST with the python that serves the lane (dev407 already contains
the int8 port; do NOT set PYTHONPATH to the source tree, it has no compiled
flash-attn). Stop the serving lane first; this wants all four cards.

  python benchmark/qsa_prefill_trace.py \
      --kv-cache-dtype int8_per_token_head --out /tmp/tr-int8
  python benchmark/qsa_prefill_trace.py \
      --kv-cache-dtype auto --out /tmp/tr-bf16

Each run prints the top CUDA kernels summed over ranks and writes
<out>.txt (table) plus the raw per-rank traces under <out>.prof/. Diff the
two tables: the kernel family carrying the int8 penalty is the answer.
"""
import argparse
import collections
import glob
import gzip
import json
import os
import shutil

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
os.environ.setdefault("VLLM_PP_LAYER_PARTITION", "13,12,12,11")
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("VLLM_SAFETENSORS_DROP_CACHE", "1")
os.environ.setdefault("VLLM_QWEN4EXP_PP_INT8_HOP", "1")
os.environ.setdefault("VLLM_QWEN4EXP_DRAFT_W8A8", "1")
os.environ.setdefault("VLLM_QWEN4EXP_DENSE_W8A8", "1")
os.environ.setdefault("VLLM_QWEN4EXP_DENSE_W8A8_MIN_M", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("MODEL", "/ssd/ai/models/Qwen/Qwen3.8-Flash-Next-FP8")


def aggregate(trace_dir: str) -> list[tuple[float, int, str]]:
    """(total_us, calls, name) per CUDA kernel across every rank's trace."""
    agg: dict[str, list] = collections.defaultdict(lambda: [0.0, 0])
    files = glob.glob(os.path.join(trace_dir, "*.json.gz")) + glob.glob(
        os.path.join(trace_dir, "*.json")
    )
    if not files:
        raise SystemExit(f"no trace files under {trace_dir}")
    for path in files:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            data = json.load(f)
        for e in data.get("traceEvents", []):
            cat = e.get("cat", "")
            if cat not in ("kernel", "gpu_memcpy", "gpu_memset"):
                continue
            row = agg[e.get("name", "?")[:110]]
            row[0] += e.get("dur", 0.0)
            row[1] += 1
    return sorted((v[0], v[1], k) for k, v in agg.items())[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", default="int8_per_token_head")
    ap.add_argument("--prompt-tokens", type=int, default=131_072,
                    help="prefill is flat per token, 128k is 64 chunks and "
                         "enough to see every per-step kernel's share")
    ap.add_argument("--out", default="/tmp/qsa-trace")
    args = ap.parse_args()

    prof_dir = f"{args.out}.prof"
    shutil.rmtree(prof_dir, ignore_errors=True)
    os.makedirs(prof_dir, exist_ok=True)

    llm = LLM(
        model=MODEL,
        max_model_len=1_000_000,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        block_size=1616,
        kv_cache_dtype=args.kv_cache_dtype,
        gpu_memory_utilization=0.9453,
        max_num_seqs=8,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        engram_config={"cpu_offload": True, "host_file_gather": True,
                       "host_file_gather_mmap": True},
        prefix_cache_retention_interval=16160,
        prefix_match_unit=16,
        speculative_config={"method": "mtp", "num_speculative_tokens": 4},
        hf_overrides={"text_config": {
            "max_position_embeddings": 1_000_000,
            "rope_parameters": {"rope_type": "yarn", "factor": 4.0,
                                "original_max_position_embeddings": 262_144}}},
        enable_flashinfer_autotune=False,
        compilation_config={"cudagraph_mode": "FULL_AND_PIECEWISE"},
        profiler_config={"profiler": "torch",
                         "torch_profiler_dir": prof_dir,
                         "torch_profiler_with_stack": False,
                         "ignore_frontend": True,
                         "max_iterations": 200},
        dtype="bfloat16",
    )

    ids = (torch.arange(args.prompt_tokens, dtype=torch.long)
           .remainder(150_000) + 1000).tolist()
    sp = SamplingParams(max_tokens=1, temperature=0.0)

    # Warm the kernels first so JIT never lands in the window (a short prompt
    # compiles the same constexpr shapes; the padded-cell int8 specializations
    # too).
    llm.generate([{"prompt_token_ids": ids[:4096]}], sp)
    torch.cuda.synchronize()

    llm.start_profile()
    llm.generate([{"prompt_token_ids": ids}], sp)
    llm.stop_profile()
    # Give the workers a beat to flush their traces.
    import time
    for _ in range(30):
        time.sleep(2)
        if glob.glob(os.path.join(prof_dir, "*.json.gz")):
            break
    rows = aggregate(prof_dir)
    grand = sum(r[0] for r in rows)

    lines = [
        f"=== CUDA kernel totals, dtype={args.kv_cache_dtype}, "
        f"prefill={args.prompt_tokens:,} tokens ===",
        f"{'us':>12}  {'share':>6}  {'calls':>8}  name",
    ]
    for us, calls, name in rows[:30]:
        lines.append(f"{us:12.1f}  {100 * us / grand:5.1f}%  {calls:8d}  {name}")
    lines.append(f"grand total {grand / 1e6:.3f} s summed over all ranks")
    text = "\n".join(lines)
    print("\n" + text)
    with open(f"{args.out}.txt", "w") as f:
        f.write(text + "\n")
    print(f"\nraw traces: {prof_dir}/   table: {args.out}.txt")


if __name__ == "__main__":
    main()
