#!/usr/bin/env python3
# Measure MTP draft acceptance at a chosen context depth, so a drafter that
# disagrees with the target about RoPE scaling can be told apart from one that
# does not. The prompt is rebuilt from a fixed corpus with a seeded RNG, so two
# runs on two builds see the same tokens and only the build differs.
#
#   ./benchmark/mtp_ab_probe.py --depth 200000 --out with9.json
#   # revert patch 9 (RoPE forwarding), restart, wait for idle, then:
#   ./benchmark/mtp_ab_probe.py --depth 200000 --out no9.json
#   ./benchmark/mtp_ab_probe.py --compare with9.json no9.json
#
# Run it against an idle server: the counters it reads are server-wide, so any
# other traffic in the window lands in the result.
import argparse
import json
import random
import re
import time
import urllib.request

CORPUS = [
    "The harbour lights came on one by one as the tide turned.",
    "A long corridor of locked doors opened onto a narrow stair.",
    "She kept the ledger in an oilskin sack beneath the seat.",
    "Nobody had told the night porter that the lift was out of service.",
    "Rain moved across the field in sheets, then stopped as suddenly.",
    "The boy counted the gulls and lost count twice before the third.",
    "Every third shelf in that room held nothing but empty jars.",
    "He wrote the number down and immediately forgot where he put it.",
    "The train was late, and the platform clock disagreed with it.",
    "Somewhere below the floor water was running and nobody could find the tap.",
    "They agreed to meet at the same table in the same month next year.",
    "The map showed a road that the satellite picture did not.",
    "Her handwriting grew smaller towards the bottom of every page.",
    "Two chairs, a lamp and a box of unmatched screws made the office.",
    "The bell rang twice and then gave up entirely.",
    "He had learned the names of the stars from a book with wrong pictures.",
    "Fog erased the far bank and left the near one looking borrowed.",
    "The recipe asked for an hour and the oven asked for none.",
    "A note on the door said back Thursday and was dated last March.",
    "The river had moved its bed a few feet since the summer.",
    "She read the same paragraph four times before it admitted anything.",
    "Boxes marked kitchen held only more boxes.",
    "The bridge was closed to traffic but not to the wind.",
    "He wound the clock and the house became louder for knowing the time.",
    "Something in the wall ticked on its own schedule.",
    "The last bus left ten minutes early, as if it had somewhere to be.",
    "They painted the fence twice and argued about the shade afterwards.",
    "Snow arrived overnight and made the yard unfamiliar at noon.",
    "The index of the book referenced pages that had never been printed.",
    "She kept a pencil behind her ear until it warped.",
    "The sign promised a view and delivered a car park.",
    "Down by the water the path dissolved into reeds and bottle glass.",
    "His whistle carried further than any note he could actually play.",
    "The kettle boiled, the phone rang, and both waited too long.",
    "Every drawer in the desk stuck except the empty one.",
    "The field guide called this bird common in this county; nobody had mentioned it.",
]

FILLER = "The tenant returned the key on a Tuesday and left no forwarding address. "
POSITION = re.compile(r'position="(\d+)"')
COUNTERS = {
    "vllm:spec_decode_num_drafts": "drafts",
    "vllm:spec_decode_num_draft_tokens": "draft_tokens",
    "vllm:spec_decode_num_accepted_tokens": "accepted",
    "vllm:num_preemptions": "preemptions",
    "vllm:generation_tokens": "gen_tokens",
}


def post(url, payload, timeout):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_text(url, timeout):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode()


def parse_metrics(text):
    """Sum the spec-decode counters over every label set."""
    out = {
        "drafts": 0.0,
        "draft_tokens": 0.0,
        "accepted": 0.0,
        "preemptions": 0.0,
        "gen_tokens": 0.0,
        "per_pos": {},
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.rsplit(" ", 1)
        if len(fields) != 2:
            continue
        name, raw = fields
        key = name.split("{", 1)[0]
        key = key[:-6] if key.endswith("_total") else key
        try:
            value = float(raw)
        except ValueError:
            continue
        if key in COUNTERS:
            out[COUNTERS[key]] += value
        elif key == "vllm:spec_decode_num_accepted_tokens_per_pos":
            pos = POSITION.search(name)
            if pos:
                index = int(pos.group(1))
                out["per_pos"][index] = out["per_pos"].get(index, 0.0) + value
    return out


def metrics(base, timeout):
    return parse_metrics(get_text(base.rstrip("/") + "/metrics", timeout))


def build_prompt(depth, chars_per_token, seed, tail, task="copy", marker_at=0.5):
    """Filler up to roughly depth tokens, ending on the tail instruction.

    With task="copy" a marker is planted part way through, so the continuation
    has to be retrieved from an absolute depth instead of copied from the window
    immediately behind it. That keeps acceptance off the ceiling, which is where
    a rope mismatch between target and drafter can show.
    """
    rng = random.Random(seed)
    budget = max(0, int(depth * chars_per_token) - len(tail) - 80)
    marker = f"[MARKER-{seed}]"
    parts = [f"Session {seed}. {FILLER}"]
    used = len(parts[0])
    placed = task != "copy"
    stop_at = budget * marker_at
    while used < budget:
        line = rng.choice(CORPUS)
        if not placed and used >= stop_at:
            line = f"{marker} {line}"
            placed = True
        parts.append(line)
        used += len(line) + 1
    return " ".join(parts) + " " + tail


def tail_for(task, seed):
    if task == "copy":
        return (
            f"Find the sentence that begins with [MARKER-{seed}] and reproduce it "
            "and every sentence after it, verbatim, in order, one per line, for as "
            "many as you can. Add nothing else."
        )
    return (
        "Repeat the following sentence verbatim, one per line, until the end of "
        "your response, and add nothing else: The lighthouse keeper logged the "
        "weather."
    )


def calibrate(base, model, seed, timeout):
    sample = build_prompt(512, 4.0, seed, "", task="repeat")
    body = post(
        f"{base}/v1/completions",
        {"model": model, "prompt": sample, "max_tokens": 1, "temperature": 0},
        timeout,
    )
    return len(sample) / body["usage"]["prompt_tokens"]


def accepted_at(pos, after, before):
    return after["per_pos"].get(pos, 0.0) - before["per_pos"].get(pos, 0.0)


def report(res):
    print(f"prompt tokens : {res['prompt_tokens']:,} (asked for {res['depth']:,})")
    print(f"completion    : {res['completion_tokens']} tokens in {res['seconds']:.1f}s")
    print(
        f"drafts        : {res['drafts']:.0f}, accepted {res['accepted']:.0f}"
        f" of {res['draft_tokens']:.0f} drafted"
    )
    print(f"mean acceptance length: {res['mean_acceptance_length']:.3f}")
    for pos, rate in sorted(res["per_position"].items()):
        print(f"  position {pos}: {rate:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default=None, help="default: first id from /v1/models")
    ap.add_argument("--depth", type=int, default=200000, help="target prompt tokens")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--task", choices=("copy", "repeat"), default="copy")
    ap.add_argument(
        "--marker-at",
        type=float,
        default=0.5,
        help="copy task: depth fraction the marker is planted at",
    )
    ap.add_argument("--seed", type=int, default=20260910)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument(
        "--settle",
        type=float,
        default=3.0,
        help="seconds to wait before"
        " reading the counters, so the last steps are flushed",
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        with open(args.compare[0]) as fa, open(args.compare[1]) as fb:
            a, b = json.load(fa), json.load(fb)
        if a.get("task") != b.get("task") or a.get("depth") != b.get("depth"):
            raise SystemExit(
                f"runs are not comparable: {a.get('task')}/{a.get('depth')} vs "
                f"{b.get('task')}/{b.get('depth')}"
            )
        print(f"{'metric':<26}{a['label']:>14}{b['label']:>14}{'delta':>10}")
        rows = [("prompt tokens", a["prompt_tokens"], b["prompt_tokens"])]
        rows.append(
            (
                "mean acceptance length",
                a["mean_acceptance_length"],
                b["mean_acceptance_length"],
            )
        )
        for pos in sorted(a["per_position"]):
            rows.append(
                (f"position {pos}", a["per_position"][pos], b["per_position"][pos])
            )
        for name, x, y in rows:
            print(f"{name:<26}{x:>14,.3f}{y:>14,.3f}{x - y:>+10.3f}")
        return

    base = args.base_url
    model = args.model
    if model is None:
        listed = json.loads(get_text(base + "/v1/models", args.timeout))
        model = listed["data"][0]["id"]

    tail = tail_for(args.task, args.seed)
    cpt = calibrate(base, model, args.seed, args.timeout)
    prompt = build_prompt(args.depth, cpt, args.seed, tail, args.task, args.marker_at)

    before = metrics(base, args.timeout)
    start = time.time()
    body = post(
        f"{base}/v1/completions",
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": args.max_tokens,
            "temperature": 0,
            "ignore_eos": True,
        },
        args.timeout,
    )
    elapsed = time.time() - start
    time.sleep(args.settle)
    after = metrics(base, args.timeout)

    drafts = after["drafts"] - before["drafts"]
    accepted = after["accepted"] - before["accepted"]
    drafted = after["draft_tokens"] - before["draft_tokens"]
    if drafts <= 0:
        raise SystemExit("no drafts counted; is speculative decoding enabled?")

    res = {
        "label": args.label or model,
        "base_url": base,
        "depth": args.depth,
        "seed": args.seed,
        "task": args.task,
        "marker_at": args.marker_at,
        "prompt_tokens": body["usage"]["prompt_tokens"],
        "completion_tokens": body["usage"]["completion_tokens"],
        "seconds": elapsed,
        "drafts": drafts,
        "draft_tokens": drafted,
        "accepted": accepted,
        "mean_acceptance_length": 1.0 + accepted / drafts,
        "per_position": {
            pos: accepted_at(pos, after, before) / drafts
            for pos in sorted(set(after["per_pos"]) | set(before["per_pos"]))
        },
    }
    report(res)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(res, fh, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
