#!/usr/bin/env python3
"""Check that a prefix-cache hit resumes the same state a cold prefill computes.

Reuse numbers say how much the cache saves. They cannot say whether the resumed
state is right, because a wrong state still answers fast. This tool answers that
question by asking the model to repeat needles it cannot guess.

How it works. Each prompt carries two random hex codes, NEEDLE-A in the first
block and NEEDLE-B at about half depth, inside thousands of distinct filler
lines. Greedy decoding, so the answer is fixed for a given state. A cold run, then
the same prompt warm from the cache, then the prompt grown by one word, which is
what a session turn does. All three must return both codes and the same payload.

Two phases:

  serial      cold baseline, then per prompt cold, warm and grown. The baseline
              matters: this checkpoint sometimes drops a hyphen, so scoring looks
              for the hex payload and needs a cold-path accuracy number before any
              warm miss can mean anything.
  concurrent  N sessions prefilled together, then repeated together. This is the
              case that matters under pipeline parallelism, where a deferred state
              copy can resolve to another request's slot and leak another
              session's needles.

Usage:
  ./prefix_state_check.py                                  # both phases
  ./prefix_state_check.py --depth 60000 --trials 1
  ./prefix_state_check.py --sessions 6 --skip-serial
"""

import argparse
import json
import random
import re
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

BASE_DEFAULT = "http://fender.lan:8000"
MODEL_DEFAULT = "qwen3.8-flash-next-fp8"
SUBJECTS = ["archivist", "surveyor", "lighthouse-keeper", "cartographer",
            "auditor", "bellringer", "glassblower", "seed-merchant"]
VERBS = "catalogued measured logged plotted audited rang fired weighed sorted counted repaired".split()
OBJECTS = ("amber-resin granite-blocks trawl-catches coast-charts wool-bales "
           "bell-ropes furnace-panes seed-crates tide-tables").split()
INSTR = "\nReply with exactly one line: NEEDLE-A <code> NEEDLE-B <code>\n"
HEX = re.compile(r"[0-9a-f]{7,8}")
COUNTERS = ("vllm:prompt_tokens_total", "vllm:prompt_tokens_cached_total",
            "vllm:generation_tokens_total")


def counters(base):
    text = urllib.request.urlopen(base + "/metrics", timeout=30).read().decode()
    out = {}
    for name in COUNTERS:
        m = re.search(re.escape(name) + r"\{[^}]*\} ([0-9.e+-]+)", text)
        out[name] = float(m.group(1)) if m else 0.0
    return out


def post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    return json.loads(urllib.request.urlopen(req, timeout=1800).read())


def ask(base, model, prompt, max_tokens=64):
    before = counters(base)
    reply = post(base + "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    after = counters(base)
    msg = reply["choices"][0]["message"]
    full = ((msg.get("reasoning") or "") + " " + (msg.get("content") or "")).strip()
    return {
        "full": full,
        "found": set(HEX.findall(full)),
        "prompt": reply["usage"]["prompt_tokens"],
        "cached": after["vllm:prompt_tokens_cached_total"] - before["vllm:prompt_tokens_cached_total"],
        "recomputed": (
            after["vllm:prompt_tokens_total"] - before["vllm:prompt_tokens_total"]
        ) - (
            after["vllm:prompt_tokens_cached_total"] - before["vllm:prompt_tokens_cached_total"]
        ),
    }


def build_prompt(lines):
    a, b = uuid.uuid4().hex[:8], uuid.uuid4().hex[:8]
    rng = random.Random(int(a[:4], 16))
    body = "".join(
        f"{i}: the {rng.choice(SUBJECTS)} {rng.choice(VERBS)} "
        f"{rng.choice(OBJECTS)} number {rng.randint(1000, 99999)}.\n"
        for i in range(lines)
    )
    cut = body.find("\n", len(body) // 2) + 1
    prompt = f"NEEDLE-A {a}\n{body[:cut]}NEEDLE-B {b}\n{body[cut:]}{INSTR}"
    return prompt, {a, b}


def row(label, r, want):
    ok = want <= r["found"]
    print(
        f"    {label:6} prompt={r['prompt']:6} cached={r['cached']:6.0f} "
        f"recomp={r['recomputed']:6.0f} needles_ok={ok!s:5} {r['full'][:44]!r}"
    )
    return ok


def serial_phase(base, model, depth, trials):
    lines = depth // 20
    print(f"\n== serial, depth ~{depth} tokens")
    hits = 0
    for _ in range(trials):
        prompt, want = build_prompt(lines)
        hits += row("cold", ask(base, model, prompt), want)
    print(f"  cold baseline: {hits}/{trials} recovered both needles")
    worst = 0
    for _ in range(2):
        prompt, want = build_prompt(lines)
        cold = row("cold", ask(base, model, prompt), want)
        warm = row("warm", ask(base, model, prompt), want)
        grown = row("grown", ask(base, model, prompt + "now\n"), want)
        if not (cold and warm and grown):
            worst += 1
    return worst


def concurrent_phase(base, model, depth, sessions):
    lines = depth // 20
    print(f"\n== concurrent, {sessions} sessions at ~{depth} tokens, fired together")
    built = [build_prompt(lines) for _ in range(sessions)]
    prompts = [p for p, _ in built]
    wants = [w for _, w in built]

    def fire():
        with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
            return list(pool.map(lambda p: ask(base, model, p), prompts))

    before = counters(base)
    cold = fire()
    middle = counters(base)
    warm = fire()
    after = counters(base)
    print(
        f"  cold round recomputed "
        f"{(middle['vllm:prompt_tokens_total'] - before['vllm:prompt_tokens_total']) - (middle['vllm:prompt_tokens_cached_total'] - before['vllm:prompt_tokens_cached_total']):.0f}"
        f", warm round recomputed "
        f"{(after['vllm:prompt_tokens_total'] - middle['vllm:prompt_tokens_total']) - (after['vllm:prompt_tokens_cached_total'] - middle['vllm:prompt_tokens_cached_total']):.0f}"
    )
    bad = 0
    for i, (c, w, want) in enumerate(zip(cold, warm, wants)):
        own_c, own_w = want <= c["found"], want <= w["found"]
        foreign = set()
        for j, other in enumerate(wants):
            if j != i:
                foreign |= other & w["found"]
        same = c["found"] == w["found"]
        ok = own_c and own_w and same and not foreign
        bad += not ok
        print(
            f"    session{i} cold_ok={own_c!s:5} warm_ok={own_w!s:5} same={same!s:5} "
            f"foreign={sorted(foreign) or '-'} {w['full'][:38]!r}"
        )
    print(
        "  note: per-row cached counts share one global counter across threads, "
        "so read the round totals above"
    )
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--depth", type=int, default=20000)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--sessions", type=int, default=4)
    ap.add_argument("--skip-serial", action="store_true")
    ap.add_argument("--skip-concurrent", action="store_true")
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    failures = 0
    if not args.skip_serial:
        failures += serial_phase(base, args.model, args.depth, args.trials)
    if not args.skip_concurrent:
        failures += concurrent_phase(base, args.model, args.depth, args.sessions)
    print(
        "\nVERDICT: "
        + ("cache hits and grown turns resume the same state as cold runs"
           if not failures else f"{failures} check(s) failed, investigate")
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
