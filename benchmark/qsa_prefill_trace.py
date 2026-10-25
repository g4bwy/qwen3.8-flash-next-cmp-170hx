#!/usr/bin/env python
"""Whole-step kernel attribution for a real Qwen4Exp prefill, offline.

Standalone microbenchmarks say the int8 port's kernels (fused store 1.06x,
unfused store 0.88x, sparse read 1.07x at the real TOPK=2048) are all at
parity with bf16. That contradicts the lane's 76 s -> 135 s prefill and means
the extra ~59 s is NOT in the QSA kernels. This script finds where it is:
it boots the engine in-process (no HTTP lane, nothing to crash) and profiles
ONE real prefill forward under torch.profiler, dumping the CUDA kernel table
sorted by device time for a chosen --kv-cache-dtype.

Run it once per dtype on the diff:

  CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=/home/gab/test/wt-int8kv \
    /home/gab/test/vllm/.venv/bin/python \
    benchmark/qsa_prefill_trace.py --kv-cache-dtype int8_per_token_head --out /tmp/tr-int8
  # ... --kv-cache-dtype auto --out /tmp/tr-bf16

Compare the top kernels between the two traces. Whatever kernel grew is the
59 s; if nothing in QSA grew, the gap is not the port.

The env block mirrors serve-4x-1m.sh so the shape matches the lane. Weight
load is ~173 GiB via engram cpu_offload; expect several minutes before the
first token. It does NOT run the serving API, so it cannot swap the box the
way a live-lane profiler did.
"""
import argparse
import os

# Match serve-4x-1m.sh env; set before importing vllm/torch cuda.
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
from torch.profiler import ProfilerActivity, profile  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.sampling_params import RequestOutputKind  # noqa: E402

MODEL = os.environ.get("MODEL", "/ssd/ai/models/Qwen/Qwen3.8-Flash-Next-FP8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-cache-dtype", default="int8_per_token_head")
    ap.add_argument("--prompt-tokens", type=int, default=250_000)
    ap.add_argument("--out", default="/tmp/qsa-trace")
    ap.add_argument("--steps", type=int, default=6,
                    help="engine steps in the profile window; prefill of a "
                    "long prompt is many 2048-chunks, a few is enough to price "
                    "the per-chunk kernels")
    a = ap.parse_args()

    llm = LLM(
        model=MODEL,
        max_model_len=1_000_000,
        tensor_parallel_size=1,
        pipeline_parallel_size=4,
        block_size=1616,
        kv_cache_dtype=a.kv_cache_dtype,
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
        dtype="bfloat16",
    )

    # Realistic random token block, deterministic, no tokenizer dependency.
    ids = (torch.arange(a.prompt_tokens, dtype=torch.long)
           .remainder(150_000) + 1000).tolist()
    sp = SamplingParams(max_tokens=1, temperature=0.0,
                        output_kind=RequestOutputKind.FINAL_ONLY)

    # Compile the exact kernels once (constexpr shapes do not depend on context
    # length), then profile the real long prefill.
    llm.generate(prompt_token_ids=[ids[:4096]], sampling_params=sp)
    torch.cuda.synchronize()

    os.makedirs(a.out, exist_ok=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        llm.generate(prompt_token_ids=[ids[:a.prompt_tokens]], sampling_params=sp)
        torch.cuda.synchronize()

    table = prof.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=30)
    print(f"\n=== self CUDA time, top 30 kernels, dtype={a.kv_cache_dtype} "
          f"prompt={a.prompt_tokens:,} ===")
    print(table)
    with open(f"{a.out}.txt", "w") as f:
        f.write(table)
    prof.export_chrome_trace(f"{a.out}.json")
    print(f"\nwrote {a.out}.txt and {a.out}.json")


if __name__ == "__main__":
    main()
