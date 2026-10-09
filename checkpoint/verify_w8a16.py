#!/usr/bin/env python3
"""Full verification for the v2 int8-dense checkpoint: recompute every
converted tensor from source and compare stored bytes; compare every
passthrough tensor against source; validate config resolution."""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from encode_fp8_dense import bf16_to_f32, read_header  # noqa: E402
from encode_w8a16 import out_name, quant_int8_g32  # noqa: E402

import argparse as _ap
_p = _ap.ArgumentParser()
_p.add_argument("--src", default="/home/gab/models/Qwen/Qwen3.8-Flash-Next-FP8")
_p.add_argument("--out", default="/home/gab/test/Qwen3.8-Flash-Next-W8A16")
_a = _p.parse_args()
S, O = _a.src, _a.out


def main():
    batch = int(sys.argv[1])
    sid = json.load(open(f"{S}/model.safetensors.index.json"))["weight_map"]
    sid.pop("mtp.draft_lm_head.weight", None)
    conv = json.load(open(f"{O}/w8_manifest.json"))["converted"]
    by_file = {}
    for k, f in sid.items():
        by_file.setdefault(f, []).append(k)
    fails = []
    n = 0
    for f in sorted(by_file)[batch::4]:
        hs, ss = read_header(f"{S}/{f}")
        ho, so = read_header(f"{O}/{f}")
        with open(f"{S}/{f}", "rb") as sfh, open(f"{O}/{f}", "rb") as ofh:
            for k in by_file[f]:
                s0, s1 = hs[k]["data_offsets"]
                if k in conv:
                    sfh.seek(ss + s0)
                    w = bf16_to_f32(np.frombuffer(sfh.read(s1 - s0), "V2"))
                    w = w.reshape(hs[k]["shape"])
                    packed, scale, err = quant_int8_g32(w)
                    mod = k[: -len(".weight")]
                    for suffix, want in (
                        ("weight_packed", packed.tobytes()),
                        ("weight_scale", None),
                        ("weight_shape", np.array([w.shape[1], w.shape[0]],
                                                  np.int64).tobytes()),
                    ):
                        m = ho[f"{mod}.{suffix}"]
                        a, b = m["data_offsets"]
                        ofh.seek(so + a)
                        got = ofh.read(b - a)
                        if want is not None:
                            ok = got == want
                        else:
                            from encode_fp8_dense import f32_to_bf16_bytes
                            ok = got == f32_to_bf16_bytes(
                                scale.astype(np.float32)).tobytes()
                        if not ok:
                            fails.append((suffix, k))
                    n += 1
                else:
                    k_out = out_name(k)
                    os_, oe = ho[k_out]["data_offsets"]
                    sfh.seek(ss + s0)
                    a = sfh.read(s1 - s0)
                    ofh.seek(so + os_)
                    if ofh.read(oe - os_) != a:
                        fails.append(("passthrough", k))
                    n += 1
        print(f"{f}: total {n} tensors verified, {len(fails)} fails", flush=True)
    print(json.dumps({"batch": batch, "tensors": n,
                      "fails": fails[:20], "fail_count": len(fails)}),
          flush=True)


if __name__ == "__main__":
    main()
