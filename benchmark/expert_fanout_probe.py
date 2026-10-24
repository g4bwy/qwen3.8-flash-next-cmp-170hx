#!/usr/bin/env python3
"""How many distinct experts does one MTP-4 decode step actually load?

The decode step anatomy attributes 7.4 ms of a 26.9 ms step to the grouped expert
kernels, and results.md bounds that at "~20 unique experts per layer-step" as an
inference from the time budget, never measured. The bound is what decides whether
the MoE byte axis has a lever in it: 20 unique experts means about 660 GB/s against
the 895 GB/s the dense path sustains in the same step, while 26 or more means the
kernels are already at the ceiling and there is nothing to win.

Server needs `--enable-return-routed-experts` (engine flag, no request opt-in
besides an optional skip). The response carries base64 `.npy` of shape
(tokens-1, layers, topk). This script reads it and reports the per-layer union over
sliding windows of `--rows` consecutive forwarded tokens, which is the set of experts
one step of that many rows has to read.

  python expert_fanout_probe.py --base-url http://fender.lan:8000 --depth 900000
  python expert_fanout_probe.py --depth 8000 --label shallow   # routing vs depth
"""

import argparse
import base64
import io
import json
import sys
import urllib.request

import numpy as np

sys.path.insert(0, ".")
from mtp_ab_probe import build_prompt, calibrate, tail_for  # noqa: E402

EXPERT_MB = 4.92  # fp8 block-128: gate_up 1280x2560 (3.28 MB) + down 2560x640 (1.64 MB)

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://127.0.0.1:8000")
ap.add_argument("--model", default=None)
ap.add_argument("--depth", type=int, default=900000, help="prompt tokens")
ap.add_argument("--gen", type=int, default=64, help="tokens to generate")
ap.add_argument(
    "--rows",
    type=int,
    default=5,
    help="rows per step: 1 + num_speculative_tokens for MTP-4",
)
ap.add_argument("--salt", type=int, default=110000)
ap.add_argument("--timeout", type=float, default=900.0)
ap.add_argument("--label", default="")
a = ap.parse_args()

base = a.base_url.rstrip("/")
model = (
    a.model
    or json.loads(urllib.request.urlopen(base + "/v1/models", timeout=15).read())[
        "data"
    ][0]["id"]
)
cpt = calibrate(base, model, 1, a.timeout)
prompt = build_prompt(a.depth, cpt, a.salt, tail_for("repeat", a.salt), task="repeat")
body = json.dumps(
    {
        "model": model,
        "prompt": prompt,
        "max_tokens": a.gen,
        "temperature": 0,
        "ignore_eos": True,
    }
).encode()
req = urllib.request.Request(
    base + "/v1/completions", data=body, headers={"Content-Type": "application/json"}
)
with urllib.request.urlopen(req, timeout=a.timeout) as resp:
    payload = json.loads(resp.read())

raw = (payload["choices"][0] or {}).get("routed_experts")
if not raw:
    print(
        "No routed_experts in the response. The server needs "
        "--enable-return-routed-experts. If it already has it, the Rust frontend "
        "may be dropping the field: retry against the Python frontend."
    )
    raise SystemExit(2)

arr = np.load(io.BytesIO(base64.b64decode(raw)))
n_tok, n_layers, topk = arr.shape
print(
    f"label={a.label or '-'} routed_experts shape {arr.shape} dtype={arr.dtype} "
    f"({n_tok} forwarded tokens x {n_layers} layers x topk {topk})"
)
if n_tok < a.rows + 2:
    print(f"need more generated tokens for a {a.rows}-row window; raise --gen")
    raise SystemExit(1)

# Per-layer union size over every sliding window of `rows` consecutive tokens.
per_layer = []
for layer in range(n_layers):
    rows = arr[:, layer, :]
    unions = np.empty(rows.shape[0] - a.rows + 1, dtype=np.int16)
    for i in range(unions.shape[0]):
        unions[i] = np.unique(rows[i : i + a.rows]).size
    per_layer.append((layer, int(unions.mean()), int(unions.min()), int(unions.max())))

means = np.array([p[1] for p in per_layer])
print(f"\nunique experts per layer over windows of {a.rows} consecutive tokens:")
print(
    f"  across {n_layers} layers: mean {means.mean():.1f}, "
    f"min-layer avg {means.min()}, max-layer avg {means.max()}"
)
print(
    f"  single token union is {topk} by construction, so a {a.rows}-row step "
    f"reuses {a.rows * topk} - {means.mean():.0f} = "
    f"{a.rows * topk - means.mean():.0f} slots through overlap"
)

bytes_step = means.mean() * EXPERT_MB
# 95 us gate_up + 53 us down per layer-step, rank-0 column of the 2026-10-13 trace.
for tag, us, weight_mb in (("gate_up", 95.0, 3.28), ("down", 53.0, 1.64)):
    gb = (bytes_step * (weight_mb / EXPERT_MB)) / (us * 1e-6) / 1e3
    print(
        f"  {tag}: {bytes_step * weight_mb / EXPERT_MB:5.1f} MB in {us:4.0f} us "
        f"= {gb:5.0f} GB/s"
    )
total = (bytes_step / ((95 + 53) * 1e-6)) / 1e3
print(f"  combined: {bytes_step:5.1f} MB in 148 us = {total:5.0f} GB/s")
print("  reference: dense bf16 in-step 895 GB/s, isolated GEMV 984-1070, bus 1593")
print(
    "  if the combined figure is at or above ~900 the grouped kernels are not "
    "leaving bandwidth on the table"
)
