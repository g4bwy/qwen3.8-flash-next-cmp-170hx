#!/usr/bin/env python3
"""Acceptance and decode-rate battery on real maki-corpus text.

Prompts are concatenations of corpus records (the model's own past
outputs), so draft acceptance is measured on the distribution the draft
vocabulary was built from, not synthetic repetition. Streams launch
staggered to create genuine joining windows: the phase where patch 8's
small head was measured to lose acceptance.

Each cell runs one sampling mode. "greedy" pins temperature 0. "sampled"
applies this deployment's client defaults (temperature 1.0, top_k 20,
top_p 0.95). Both modes report RATES (aggregate output tok/s, TTFT,
per-stream decode tok/s) next to acceptance, so a build that moves
throughput without moving acceptance still shows up in the cell.

Acceptance is reported twice:
  - acc win  : counters snapshotted after every stream has emitted its
    first token, then again when the first stream ends. Since base
    8407bc0bb6-era (and #62129) the spec-decode counters also include
    drafts emitted during prefill-phase steps, so only this inner bracket
    reads steady decode. It is the number to hold builds against.
  - acc full : the legacy whole-run diff, diluted toward the prefill
    ratio. Kept so older records stay comparable.

Prompt seeds are time-derived by default, so a re-run never hits the
prefix cache, and each mode shifts the seed by one, so modes never share
prefixes. Pin --seed to compare two builds on identical prompts.

usage: corpus_accept_probe.py --depth-tokens 45000 --window 4096
       --conc 4 --stagger 1.5 [--modes greedy,sampled] [--out file.json]
"""
import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.request

BASE = "http://127.0.0.1:8000"
MODEL = "qwen3.8-flash-next-fp8"

# Sampling knobs per mode, as the real clients use them.
MODE_SAMPLING = {
    "greedy": {"temperature": 0.0},
    "sampled": {"temperature": 1.0, "top_k": 20, "top_p": 0.95},
}

# Below this the inner bracket holds too few steps to mean anything.
MIN_BRACKET_S = 5.0


def metrics():
    txt = urllib.request.urlopen(BASE.rstrip("/") + "/metrics", timeout=30).read().decode()
    d = {}
    for line in txt.splitlines():
        m = re.match(r"vllm:spec_decode_num_drafts_total\S* (\S+)", line)
        if m:
            d["drafts"] = float(m.group(1))
        m = re.match(r"vllm:spec_decode_num_accepted_tokens_total\S* (\S+)", line)
        if m:
            d["accepted"] = float(m.group(1))
        m = re.match(
            r'vllm:spec_decode_num_accepted_tokens_per_pos_total{'
            r'.*?position="(\d+)"} (\S+)', line)
        if m:
            d[f"pos{m.group(1)}"] = float(m.group(2))
    return d


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_prompts(prompts):
    h = hashlib.sha256()
    for p in prompts:
        h.update(p.encode("utf-8", "replace"))
        h.update(b"\n\0\n")
    return h.hexdigest()


def read_corpus(path):
    """Longest-first pool of past model outputs; short records carry no signal."""
    pool = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = rec.get("output") or ""
            if len(t) > 500:
                pool.append(t)
    return pool


def build_prompts(pool, conc, chars, seed):
    rng = random.Random(seed)
    prompts = []
    for _ in range(conc):
        parts, total = [], 0
        while total < chars:
            t = rng.choice(pool)
            parts.append(t)
            total += len(t)
        prompts.append("\n\n".join(parts)[:chars])
    return prompts


def stream(prompt, window, stop_ev, model, sampling=None, firsts=None, idx=None):
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": window,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    payload.update(sampling or {})
    req = urllib.request.Request(
        BASE.rstrip("/") + "/v1/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    first = None
    ntok = 0
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            if stop_ev.is_set():
                break
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                obj = json.loads(body)
            except json.JSONDecodeError:
                continue
            ch = (obj.get("choices") or [{}])[0]
            if first is None and ch.get("text"):
                now = time.time()
                first = now - t0
                # Wall clock of the first token, so the main thread can
                # bracket the decode phase while the streams are running.
                if firsts is not None and idx is not None:
                    firsts[idx] = now
            usage = obj.get("usage")
            if usage:
                ntok = usage.get("completion_tokens", ntok)
    return {"start": t0, "ttft": first, "end": time.time(), "toks": ntok}


def accept(before, after):
    d = after.get("drafts", 0.0) - before.get("drafts", 0.0)
    a = after.get("accepted", 0.0) - before.get("accepted", 0.0)
    return (1 + a / d) if d > 0 else None, d, a


def median(xs):
    s = sorted(xs)
    if not s:
        return None
    m = len(s) // 2
    return s[m] if len(s) % 2 else (s[m - 1] + s[m]) / 2


def fmt(v, nd):
    return "n/a" if v is None else f"{v:.{nd}f}"


def per_stream(r):
    decode_s = r["end"] - (r["start"] + (r["ttft"] or 0))
    row = {"i": r["i"], "ttft": round(r["ttft"], 1) if r["ttft"] else None,
           "toks": r["toks"],
           "rate": round(r["toks"] / decode_s, 1) if r["toks"] and decode_s > 0 else None}
    if r.get("error"):
        row["error"] = r["error"]
    return row


def run_cell(a, pool, mode, sampling, seed):
    """One battery cell: `conc` staggered streams under one sampling mode."""
    chars = a.depth_tokens * 4
    prompts = build_prompts(pool, a.conc, chars, seed)
    print(f"[{mode}] seed {seed} prompts {sha256_prompts(prompts)[:12]} "
          f"({chars:,} chars x {a.conc}), launching at {time.strftime('%H:%M:%S')}",
          flush=True)

    before = metrics()
    stop = threading.Event()
    slots = [None] * a.conc
    firsts = {}
    threads = []

    def run(i):
        try:
            r = stream(prompts[i], a.window, stop, a.model, sampling, firsts, i)
        except Exception as exc:  # a dead stream must not hang the bracket watch
            r = {"start": time.time(), "ttft": None, "end": time.time(),
                 "toks": 0, "error": f"{type(exc).__name__}: {exc}"}
        r["i"] = i
        slots[i] = r

    t_launch = time.time()
    for i in range(a.conc):
        th = threading.Thread(target=run, args=(i,), daemon=True)
        th.start()
        threads.append(th)
        time.sleep(a.stagger)

    # Decode-window bracket: open once every stream is past its prefill,
    # close when the first stream is done. Full-window counters keep their
    # own snapshot below, so both figures come out of a single run.
    while len(firsts) < a.conc and any(th.is_alive() for th in threads):
        time.sleep(0.05)
    opened = len(firsts) >= a.conc
    bracket = {"opened": opened, "first_tokens": {}}
    win_open = win_close = None
    if opened:
        bracket["open_t"] = time.time()
        win_open = metrics()
    while any(th.is_alive() for th in threads) and all(r is None for r in slots):
        time.sleep(0.05)
    if opened:
        bracket["close_t"] = time.time()
        win_close = metrics()
    for th in threads:
        th.join()
    time.sleep(2)
    after = metrics()

    runs = [r for r in slots if r]
    acc_full, d_full, a_full = accept(before, after)
    acc_win = None
    if opened:
        window_s = bracket["close_t"] - bracket["open_t"]
        bracket["window_s"] = round(window_s, 1)
        bracket["open_rel_s"] = round(bracket["open_t"] - t_launch, 1)
        bracket["close_rel_s"] = round(bracket["close_t"] - t_launch, 1)
        acc_win, d_win, a_win = accept(win_open, win_close)
        bracket["drafts"], bracket["accepted"] = round(d_win), round(a_win)
        why = None
        if window_s < MIN_BRACKET_S:
            why = f"window {window_s:.1f}s < {MIN_BRACKET_S:g}s"
        elif not d_win:
            why = "no drafts inside the bracket"
        if why:
            # A collapsed bracket reads as a diluted or empty ratio, so the
            # figure is dropped and the reason recorded instead.
            bracket["collapsed"] = why
            acc_win = None
    bracket["first_tokens"] = {
        str(i): round(t - t_launch, 1) for i, t in sorted(firsts.items())}
    bracket.setdefault("window_s", None)

    start = min(r["start"] for r in runs) if runs else t_launch
    span = (max(r["end"] for r in runs) - start) if runs else 0.0
    out_tokens = sum(r["toks"] for r in runs)
    ttfts = [r["ttft"] for r in runs if r["ttft"]]
    streams = [per_stream(r) for r in sorted(runs, key=lambda x: x["i"])]
    rates = [s["rate"] for s in streams if s["rate"]]
    positions = sorted({k for k in before if k.startswith("pos")}
                       | {k for k in after if k.startswith("pos")})
    pos = {k: after.get(k, 0.0) - before.get(k, 0.0) for k in positions}
    cell = {
        "mode": mode,
        "seed": seed,
        "sampling": sampling,
        "prompts_sha256": sha256_prompts(prompts),
        "depth_tokens_est": a.depth_tokens,
        "window": a.window,
        "conc": a.conc,
        "stagger": a.stagger,
        "span_s": round(span, 1),
        "out_tokens": out_tokens,
        "agg_out_tps": round(out_tokens / span, 2) if span > 0 else None,
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 2) if ttfts else None,
        "rate_min": round(min(rates), 2) if rates else None,
        "rate_median": round(median(rates), 2) if rates else None,
        "rate_max": round(max(rates), 2) if rates else None,
        "acc_window": round(acc_win, 3) if acc_win is not None else None,
        "acc_full": round(acc_full, 3) if acc_full is not None else None,
        "mean_acceptance_length": round(acc_full, 3) if acc_full is not None else None,
        "drafts": round(d_full), "accepted": round(a_full),
        "bracket": bracket,
        "pos_shares": {k[3:]: round(v / a_full, 3) for k, v in pos.items()} if a_full else None,
        "per_stream": streams,
    }
    return cell


def print_table(cells):
    """Markdown rate summary; printed after every cell so a long run reports."""
    print("\n| mode | seed | span s | agg out tok/s | mean TTFT s "
          "| per-stream tok/s min/med/max | acc win | acc full |")
    print("|---|---:|---:|---:|---:|---|---:|---:|")
    for c in cells:
        spread = "/".join(fmt(c[k], 1) for k in ("rate_min", "rate_median", "rate_max"))
        print(f"| {c['mode']} | {c['seed']} | {fmt(c['span_s'], 1)} "
              f"| {fmt(c['agg_out_tps'], 1)} | {fmt(c['mean_ttft_s'], 2)} "
              f"| {spread} | {fmt(c['acc_window'], 3)} | {fmt(c['acc_full'], 3)} |",
              flush=True)


def print_bracket(c):
    b = c["bracket"]
    note = f" ({b['collapsed']})" if b.get("collapsed") else ""
    print(f"[{c['mode']}] decode-window bracket {fmt(b.get('window_s'), 1)} s "
          f"drafts={b.get('drafts')} accepted={b.get('accepted')} "
          f"-> acc win {fmt(c['acc_window'], 3)}{note}; "
          f"full window drafts={c['drafts']} accepted={c['accepted']} "
          f"-> acc full {fmt(c['acc_full'], 3)}", flush=True)


def main():
    global BASE
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE)
    ap.add_argument("--corpus", default="draft_corpus.jsonl")
    ap.add_argument("--model", default=MODEL, help="served model id")
    ap.add_argument("--depth-tokens", type=int, default=45000)
    ap.add_argument("--window", type=int, default=4096)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--stagger", type=float, default=1.5)
    ap.add_argument("--modes", default="greedy,sampled",
                    help="comma list of greedy,sampled; one cell per mode. "
                    "--temperature overrides this sweep")
    ap.add_argument("--temperature", type=float, default=None,
                    help="run one custom mode at this temperature instead of "
                    "the --modes sweep (back-compat with the single-mode cell)")
    ap.add_argument("--seed", type=int, default=None,
                    help="pin the prompt RNG for A/B across builds; default is "
                    "time-derived so re-runs miss the prefix cache")
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    BASE = a.base_url

    if a.temperature is not None:
        cells_spec = [("custom", {"temperature": a.temperature})]
    else:
        cells_spec = []
        for name in [m.strip() for m in a.modes.split(",") if m.strip()]:
            if name not in MODE_SAMPLING:
                raise SystemExit(f"unknown mode {name!r}; choose from "
                                 f"{', '.join(MODE_SAMPLING)}")
            cells_spec.append((name, dict(MODE_SAMPLING[name])))
    if not cells_spec:
        raise SystemExit("no modes to run")

    corpus = os.path.abspath(a.corpus)
    try:
        digest = sha256_file(corpus)
    except OSError as exc:
        raise SystemExit(f"cannot read corpus {corpus}: {exc}")
    pool = read_corpus(corpus)
    if not pool:
        raise SystemExit(f"no records with output > 500 chars in {corpus}")

    seed_base = a.seed if a.seed is not None else int(time.time()) // 10 * 10
    print(f"corpus {corpus} sha256 {digest[:12]} pool {len(pool)} records; "
          f"model {a.model} at {BASE}; base seed {seed_base} "
          f"({'pinned' if a.seed is not None else 'time-derived'})")

    results = []
    for idx, (mode, sampling) in enumerate(cells_spec):
        results.append(run_cell(a, pool, mode, sampling, seed_base + idx))
        print_bracket(results[-1])
        print_table(results)

    out = {
        "label": a.label, "model": a.model, "base_url": BASE,
        "corpus_file": corpus, "corpus_sha256": digest,
        "seed_base": seed_base, "seed_pinned": a.seed is not None,
        "depth_tokens_est": a.depth_tokens, "window": a.window,
        "conc": a.conc, "stagger": a.stagger,
        "modes": [c["mode"] for c in results],
        "results": results,
    }
    if len(results) == 1:
        # Keep the flat single-mode shape older records use.
        out.update({k: v for k, v in results[0].items() if k not in out})
        out["temperature"] = results[0]["sampling"].get("temperature")
    print(json.dumps(out, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
