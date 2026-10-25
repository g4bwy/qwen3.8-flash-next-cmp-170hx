#!/usr/bin/env python
"""Is the per-token-head INT8 cache store slower than a plain cast?

The int8 lane is 2x on prefill and unchanged on decode. Decode writes ~1-5
tokens/step so a slow per-token store is invisible there; prefill writes the
whole ~1M prompt, so a slow store shows up as exactly this symptom. This times
the shipped writers on their own grid, no engine:

  bf16 : triton_reshape_and_cache_flash          (a cast + paged copy)
  int8 : triton_reshape_and_cache_flash_per_token_head_quant  (absmax + quant + scale)

CUDA-event timed over a graphed loop. Same token/head/dim shapes.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/gab/test/wt-int8kv \
        /home/gab/test/vllm/.venv/bin/python \
        benchmark/qsa_store_cost.py --tokens 2048
"""
import argparse
import torch

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
    triton_reshape_and_cache_flash_per_token_head_quant,
)
from vllm.v1.kv_cache_interface import KVQuantMode

HEAD_DIM = 256
HKV = 2
BLOCK = 1616


def make(tokens, dtype):
    nb = (tokens + BLOCK - 1) // BLOCK + 1
    g = torch.Generator(device="cuda").manual_seed(0)
    key = torch.randn(tokens, HKV, HEAD_DIM, generator=g, device="cuda",
                      dtype=torch.bfloat16)
    val = torch.randn(tokens, HKV, HEAD_DIM, generator=g, device="cuda",
                      dtype=torch.bfloat16)
    kc = torch.zeros(nb, BLOCK, HKV, HEAD_DIM, device="cuda", dtype=dtype)
    vc = torch.zeros(nb, BLOCK, HKV, HEAD_DIM, device="cuda", dtype=dtype)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    return key, val, kc, vc, slots


def timeit(fn, label):
    fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(6):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(50):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 50)
    print(f"  {label:42s} {best*1000:9.1f} us  ({best*1000/60:.2f} us/layer@12L)")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048)
    a = ap.parse_args()
    scale = torch.ones((), device="cuda")
    print(f"device {torch.cuda.get_device_name(0)}  store {a.tokens} tokens x "
          f"{HKV} kv-heads x {HEAD_DIM} dim")

    k, v, kc, vc, slots = make(a.tokens, torch.bfloat16)
    t_bf = timeit(
        lambda: triton_reshape_and_cache_flash(
            k, v, kc, vc, slots, "auto", scale, scale),
        "bf16 cast+copy")

    k2, v2, kc8, vc8, slots2 = make(a.tokens, torch.int8)
    ksc = torch.zeros(kc8.shape[:3], device="cuda", dtype=torch.float32)
    vsc = torch.zeros(vc8.shape[:3], device="cuda", dtype=torch.float32)
    t_i8 = timeit(
        lambda: triton_reshape_and_cache_flash_per_token_head_quant(
            k2, v2, kc8, vc8, ksc, vsc, slots2,
            KVQuantMode.INT8_PER_TOKEN_HEAD),
        "int8 per-token-head absmax+quant")
    print(f"\n  int8 / bf16 store ratio: {t_i8 / t_bf:.2f}x")


if __name__ == "__main__":
    main()
