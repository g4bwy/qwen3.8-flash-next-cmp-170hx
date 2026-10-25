#!/usr/bin/env python
"""Price the ONE remaining int8 prefill suspect: the fused int8 store.

Read measures 1.07x and the standalone per-token-head writer 0.88x, so the
whole prefill gap (int8 135 s vs bf16 76 s) can only live in the fused
qsa_prepare launch's IS_INT8_PTH branch (num_warps=1, absmax+quantize per
token,head). This times that exact launch, bf16 vs int8, same indexer work
(identical for both) and same main geometry, so the delta is purely the int8
store. Inputs are built with the parity script's own builders, rescaled to a
prefill-sized chunk.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/gab/test/wt-int8kv \
      /home/gab/test/vllm/.venv/bin/python \
      /home/gab/test/vllm/patchset-qwen38-pp/benchmark/qsa_fused_store_cost.py --tokens 2048
"""
import argparse
import importlib.util
import statistics
import sys

import torch

sys.path.insert(0, "/tmp")
_spec = importlib.util.spec_from_file_location("p", "/tmp/int8_fused_parity.py")
p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(p)  # builds INDEXER/MAIN at tiny sizes; we rebuild below

qsa_prepare = p.qsa_prepare


def build(num_tokens, main_hq):
    p.MAIN_HQ, p.MAIN_HK, p.MAIN_D = main_hq, 2, 256
    p.MAIN_PAGE = 16
    p.NUM_TOKENS = num_tokens
    p.NUM_BLOCKS = num_tokens // p.MAIN_PAGE + 2
    p.SLOTS = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    idx = p.make_indexer_args()
    main = p.make_main_args()
    return idx, main


def run(idx, main, cell, scales):
    args = dict(idx, q_out=torch.empty_like(idx["q_out"]))
    m = dict(main, main_kv_cache=cell.transpose(1, 2), **(scales or {}))
    return lambda: qsa_prepare(**args, **m)


def timeit(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(6):
        s = torch.cuda.Event(True); e = torch.cuda.Event(True)
        s.record()
        for _ in range(30):
            fn()
        e.record(); torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 30)
    return best * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--hq", type=int, default=24, help="main q heads (24 real)")
    a = ap.parse_args()
    idx, mrow = build(a.tokens, a.hq)
    cap = torch.cuda.get_device_capability(0)
    print(f"device sm_{cap[0]}{cap[1]}  tokens={a.tokens}  main Q/KV heads "
          f"{a.hq}/2  head_dim 256  fused launch num_warps=1")
    print(f"main_qkv {mrow['main_qkv'].shape}, "
          f"indexer work shared by both lanes")

    bf_cell = torch.zeros(p.NUM_BLOCKS, 2, p.MAIN_PAGE, 2 * 256,
                          dtype=torch.bfloat16, device="cuda")
    t_bf = timeit(run(idx, mrow, bf_cell, None))

    i8_cell = torch.zeros(p.NUM_BLOCKS, 2, p.MAIN_PAGE, 2 * 256 + 8,
                          dtype=torch.int8, device="cuda")
    ks, vs = p.load_scale_views(i8_cell)
    t_i8 = timeit(run(idx, mrow, i8_cell,
                      dict(main_k_scale_cache=ks, main_v_scale_cache=vs)))

    print(f"\n  fused prepare bf16 store: {t_bf:9.1f} us")
    print(f"  fused prepare int8 store: {t_i8:9.1f} us")
    print(f"  int8 / bf16 fused-store ratio: {t_i8 / t_bf:.2f}x")
    extra_s = (t_i8 - t_bf) * 1e-6 * 12 * (1_000_000 / a.tokens)
    print(f"  if this holds, extra over a 1M prefill (x12 layers): {extra_s:.1f} s")


if __name__ == "__main__":
    main()
