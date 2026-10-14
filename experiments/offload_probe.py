#!/usr/bin/env python3
"""offload_probe: does the CPU KV tier actually serve a resumed session?

Sequence, all through /v1/completions, greedy, streaming (true TTFT):
  1. session prompt A at ~--length tokens: cold prefill, TTFT_cold,
     completion C1, host stores counted around the request.
  2. --settle seconds idle: async stores must drain before the GPU flush,
     otherwise late eviction cuts a session's store short (seen 2026-10-13:
     resume recovered only 71% of a session).
  3. fillers sized to exceed --pool-tokens GPU churn, each with
     kv_transfer_params.max_offload_tokens=0 (proven to hold stores at 0),
     so the GPU prefix cache forgets A without flooding the host region.
  4. resend A: expect TTFT_resume << TTFT_cold with a nonzero CPU->GPU
     counter delta, and C2 == C1 byte-for-byte.

Reads vllm:kv_offload_total_bytes_total{transfer_type=...} around every
phase, so a null result separates itself: stores without loads = lookup
miss; loads with no speedup = timing artifact; nothing at all = stores
never committed.

Requires a server booted with OFFLOAD=1 (pinned) or OFFLOAD=lazy.

usage: offload_probe.py [--length 150000] [--settle 25] [--pool-tokens
                         2689440] [--json out.json]
"""
import argparse
import json
import re
import sys
import time
import urllib.request

from mtp_ab_probe import build_prompt, calibrate, tail_for

BASE = "http://fender.lan:8000"


def counters(base):
    t = urllib.request.urlopen(base + "/metrics", timeout=10).read().decode()
    out = {}
    for m in re.finditer(
        r'vllm:kv_offload_total_bytes_total\{[^}]*transfer_type="([\w+]+)"\} ([\d.e+]+)',
        t,
    ):
        out[m.group(1)] = float(m.group(2))
    return out


def comp_stream(base, model, prompt, maxtok, timeout, extra=None):
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": maxtok,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if extra:
        body.update(extra)
    req = urllib.request.Request(
        base + "/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    ttfb = None
    text = []
    ptok = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            pay = line[5:].strip()
            if pay == "[DONE]":
                break
            d = json.loads(pay)
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens")
            for ch in d.get("choices", []):
                tok = ch.get("text", "")
                if tok:
                    if ttfb is None:
                        ttfb = time.time() - t0
                    text.append(tok)
    return (ttfb if ttfb else time.time() - t0), "".join(text), ptok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--model", default=None)
    ap.add_argument("--length", type=int, default=150000)
    ap.add_argument("--settle", type=float, default=25.0)
    ap.add_argument("--pool-tokens", type=int, default=2689440)
    ap.add_argument("--fill-margin", type=float, default=1.15)
    ap.add_argument("--decode", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    listed = json.loads(
        urllib.request.urlopen(a.base_url + "/v1/models", timeout=30).read()
    )
    model = a.model or listed["data"][0]["id"]
    cpt = calibrate(a.base_url, model, 1, a.timeout)
    salt = int(time.time()) // 10 * 10

    prompt_a = build_prompt(
        a.length, cpt, salt, tail_for("repeat", salt), task="repeat"
    )
    c0 = counters(a.base_url)
    t_cold, c1, n1 = comp_stream(
        a.base_url, model, prompt_a, a.decode, a.timeout
    )
    c1m = counters(a.base_url)
    stored = (c1m["GPU_to_CPU"] - c0["GPU_to_CPU"]) / 1e9
    print(f"[probe] A: {n1} tok | cold TTFT {t_cold:.1f} s | stored {stored:.2f} GB")

    print(f"[probe] settling async stores {a.settle:.0f} s")
    time.sleep(a.settle)

    need = int(a.pool_tokens * a.fill_margin)
    fill_len = 780000
    n_fill = max(1, -(-need // fill_len))
    for i in range(n_fill):
        p = build_prompt(
            fill_len,
            cpt,
            salt + 7919 * (i + 1),
            tail_for("repeat", salt + 7919 * (i + 1)),
            task="repeat",
        )
        ca = counters(a.base_url)
        t_f, _, n_f = comp_stream(
            a.base_url,
            model,
            p,
            1,
            a.timeout,
            extra={"kv_transfer_params": {"max_offload_tokens": 0}},
        )
        cb = counters(a.base_url)
        f_store = (cb["GPU_to_CPU"] - ca["GPU_to_CPU"]) / 1e9
        print(
            f"[probe] flusher {i + 1}/{n_fill}: {n_f} tok, {t_f:.1f} s, "
            f"stored-during {f_store:.2f} GB (want 0)"
        )

    c2 = counters(a.base_url)
    t_res, c2txt, _ = comp_stream(
        a.base_url, model, prompt_a, a.decode, a.timeout
    )
    c3m = counters(a.base_url)
    loaded = (c3m["CPU_to_GPU"] - c2["CPU_to_GPU"]) / 1e9
    equal = c1 == c2txt
    print(
        f"[probe] resume: TTFT {t_res:.1f} s | host->GPU {loaded:.2f} GB | "
        f"greedy {'IDENTICAL' if equal else 'DIVERGED'}"
    )
    if not equal:
        d = next(
            (i for i, (x, y) in enumerate(zip(c1, c2txt)) if x != y),
            min(len(c1), len(c2txt)),
        )
        print(
            f"[probe] first diff at char {d}: "
            f"cold={c1[max(0, d - 15):d + 30]!r} resume={c2txt[max(0, d - 15):d + 30]!r} "
            f"| lens {len(c1)}/{len(c2txt)}"
        )

    verdict = (
        "HOST-TIER WORKING"
        if loaded > 0 and t_res < t_cold / 2 and equal
        else "NO HOST LOAD (lookup miss or stores lost)"
        if loaded == 0
        else "LOADED BUT SLOW (check streaming timing vs settle)"
        if loaded > 0
        else "WRONG OUTPUT (state integrity failure - stop and investigate)"
    )
    print(f"[probe] verdict: {verdict}")
    res = {
        "salt": salt,
        "session_tokens": n1,
        "ttft_cold_s": round(t_cold, 2),
        "ttft_resume_s": round(t_res, 2),
        "stored_gb": round(stored, 2),
        "loaded_gb": round(loaded, 2),
        "greedy_equal": equal,
        "verdict": verdict,
        "pool_tokens": a.pool_tokens,
        "settle_s": a.settle,
    }
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"[probe] wrote {a.json}")
    return 0 if equal else 1


if __name__ == "__main__":
    sys.exit(main())
