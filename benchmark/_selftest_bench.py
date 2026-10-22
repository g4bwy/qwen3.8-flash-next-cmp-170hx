#!/usr/bin/env python3
"""Offline self-test for the bench.py protocol fixes.

No server, no GPU, no network. Checks that the repeat tail still builds the
exact prompt bytes older runs used, that the content canary scores synthetic
completions, that decode-window acceptance picks the right counter samples,
and that stream_run still hands back the keys its importers rely on.
"""
import hashlib
import io
import os
import re
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import bench  # noqa: E402
from mtp_ab_probe import REPEAT_SENTENCE, build_prompt, tail_for  # noqa: E402

FAILS = []


def check(name, cond):
    print(("PASS" if cond else "FAIL") + "  " + name)
    if not cond:
        FAILS.append(name)


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


# --- (a) prompts must not change --------------------------------------------
# Expected strings and sha256 sums are hardcoded from mtp_ab_probe.py before
# REPEAT_SENTENCE existed, so a tail edit that moves one byte trips here:
# old salts must keep reproducing the prompts earlier runs measured.
REPEAT_TAIL = (
    "Repeat the following sentence verbatim, one per line, until the end of "
    "your response, and add nothing else: The lighthouse keeper logged the "
    "weather."
)
check("repeat tail is byte-identical", tail_for("repeat", 777) == REPEAT_TAIL)
check(
    "repeat tail ignores the seed",
    tail_for("repeat", 1) == tail_for("repeat", 20260910) == REPEAT_TAIL,
)
check(
    "repeat tail ends on the constant",
    tail_for("repeat", 7).endswith(REPEAT_SENTENCE),
)
check(
    "repeat tail sha",
    sha(tail_for("repeat", 777))
    == "860c9b46748aa4271cc8924d4468e0982d39463c057dd8bb9970cb38a842cf79",
)
check(
    "copy tail untouched",
    sha(tail_for("copy", 777))
    == "ac641fd87677f4647b36c355b19c09ff396cc72e90316492d07a5d5816132318",
)
PROMPTS = [
    (
        (2048, 4.0, 777, "repeat"),
        "fbdc4e78e5287da2b386de10d8ab7b78d348d5a4606c6a3037b7f0da1edf4ef4",
    ),
    (
        (65536, 4.1, 9090, "repeat"),
        "18eecdd61623670483cd2cd9894705a86bbeee29f55677e758ff641b699e6044",
    ),
    (
        (4096, 4.0, 55, "copy"),
        "39b6d48f156f0d20233d96e053603c02e3e3da6c76b59af50e3b19dc8ae2a3ca",
    ),
]
for (depth, cpt, seed, task), want in PROMPTS:
    prompt = build_prompt(depth, cpt, seed, tail_for(task, seed), task=task)
    check(f"prompt sha {task} {depth}/{seed}", sha(prompt) == want)

# --- (b) content canary over synthetic completions --------------------------
# Semantics (live shape, 2026-10-22): judge the last `window` lines of each
# stream that shows the loop (>= min_hits exact lines); the reasoner preamble
# and the max_tokens-cut tail line never count; thinking-only streams read
# None, not a verdict. Tests pass small window/min_hits explicitly.
GOOD = REPEAT_SENTENCE
BAD = "The lighthouse keeper logged the sky."
check(
    "canary clean loop scores 1",
    bench.tail_match(["\n".join([GOOD] * 5)], window=3, min_hits=2) == 1.0,
)
check(
    "canary bad line at tail trips",
    bench.tail_match(["\n".join([GOOD] * 5 + [BAD, GOOD[:20]])], window=3, min_hits=2)
    < 0.98,
)
check(
    "canary old glitch outside window is history",
    bench.tail_match(["\n".join([BAD] * 5 + [GOOD] * 8)], window=4, min_hits=2) == 1.0,
)
check(
    "canary thinking-only stream is inconclusive",
    bench.tail_match(
        ["I will now output:\n" + GOOD + "\nStill planning more text here."],
        window=40,
        min_hits=5,
    )
    is None,
)
check(
    "canary mixed cell judged on the looping stream only",
    bench.tail_match(["thinking\n" + GOOD, "\n".join([GOOD] * 6)], window=5, min_hits=2)
    == 1.0,
)
check(
    "canary garbled tail after long clean loop",
    bench.tail_match(["\n".join([GOOD] * 10 + [BAD] * 3)], window=40, min_hits=5) < 0.98,
)
check("canary no text is None", bench.tail_match(["", ""]) is None)
check("canary blank text is None", bench.tail_match(["\n   \n\n"]) is None)
check("canary None text is None", bench.tail_match([None]) is None)
check("canary all wrong is None", bench.tail_match(["lorem ipsum\ndolor\n"]) is None)
check(
    "canary short loop under hits is inconclusive",
    bench.tail_match(["\n".join([GOOD] * 4)], window=3, min_hits=5) is None,
)
check("hash joins streams with NUL", bench.completion_hash(["a", "b"]) == sha("a\x00b"))
check(
    "hash tells joined streams apart",
    bench.completion_hash(["ab"]) != bench.completion_hash(["a", "b"]),
)

# --- (c) decode-window acceptance -------------------------------------------
T = 1000.0
# one sample per 0.5 s, 30 drafts and 90 accepted tokens per tick -> 1+90/30
counter_samples = [(T + i * 0.5, 30.0 * i, 90.0 * i) for i in range(80)]
check(
    "window acc hand value",
    abs(bench.window_acc(counter_samples, T + 5.0, T + 15.0) - 4.0) < 1e-9,
)
got = bench.window_acc(counter_samples, T + 5.0, T + 15.0)
check("window acc returns a float", isinstance(got, float))
check(
    "window acc drops samples outside the bounds",
    abs(bench.window_acc(counter_samples, T + 2.0, T + 12.0) - 4.0) < 1e-9,
)
check("window acc no pair in a thin window",
      bench.window_acc(counter_samples, T + 5.0, T + 5.2) is None)
check("window acc no samples at all", bench.window_acc([], T, T + 60) is None)
check("window acc span under 10 s", bench.window_acc(counter_samples, T, T + 9.9)
      is None)
check("window acc span of exactly 10 s",
      bench.window_acc(counter_samples, T, T + 10.0) is not None)
still = [(T + i * 0.5, 0.0, 0.0) for i in range(80)]
check("window acc zero drafts", bench.window_acc(still, T, T + 20) is None)
frozen = [(T + i * 0.5, 10.0, 10.0) for i in range(80)]
check("window acc drafts do not move", bench.window_acc(frozen, T, T + 20) is None)
check("window acc min_span argument",
      bench.window_acc(counter_samples, T, T + 30, min_span=40.0) is None)
skew = [(T + i * 1.0, 5.0 * i, 10.0 * i) for i in range(80)]
check("window acc 1 + accepted/drafts",
      abs(bench.window_acc(skew, T, T + 12) - 3.0) < 1e-9)
# run_cell glue: the all-decoding window is max(first) .. min(end)
runs = [{"first": T + 10.0, "end": T + 40.0}, {"first": T + 20.0, "end": T + 35.0}]
lo, hi = max(r["first"] for r in runs), min(r["end"] for r in runs)
check("window bounds are the stream overlap", (lo, hi) == (T + 20.0, T + 35.0))
check("overlap window yields a number",
      abs(bench.window_acc(counter_samples, lo, hi) - 4.0) < 1e-9)

# --- stream_run text contract (faked urlopen) -------------------------------
CHUNKS = [
    b'data: {"choices": [{"text": "The "}]}',
    b'data: {"choices": [{"text": "lighthouse keeper logged the "}]}',
    b"data: {\"choices\": []}",
    b'data: {"choices": [{"text": "weather.\\n"}]}',
    b'data: {"usage": {"prompt_tokens": 7, "completion_tokens": 4}}',
    b"data: [DONE]",
]


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def __iter__(self):
        return iter([c + b"\n" for c in CHUNKS])


_real_urlopen = urllib.request.urlopen
urllib.request.urlopen = lambda req, timeout=None: FakeResp()
try:
    r = bench.stream_run("http://x", "m", "p", 4, 1)
finally:
    urllib.request.urlopen = _real_urlopen
check(
    "stream_run concatenates the deltas",
    r.get("text") == "The lighthouse keeper logged the weather.\n",
)
check(
    "stream_run keeps the keys sustained_decode reads",
    {"start", "first", "end", "usage"} <= set(r),
)
check("stream_run usage survives", r["usage"]["completion_tokens"] == 4)
check("stream_run times stay ordered", r["start"] <= r["first"] <= r["end"])
check(
    "stream_run text feeds the canary",
    bench.tail_match([r["text"]], window=3, min_hits=1) == 1.0,
)

# --- table columns ----------------------------------------------------------
with open(os.path.join(HERE, "bench.py")) as fh:
    SRC = fh.read()

PIECE = re.compile(r'[fb]{0,2}"((?:[^"\\\n]|\\.)*)"')


def literal_block(src, needle):
    """Text of the run of string-literal lines that starts at the needle line."""
    lines = src.splitlines()
    start = next(i for i, ln in enumerate(lines) if needle in ln)
    out = ""
    for ln in lines[start:]:
        pieces = PIECE.findall(ln)
        if not pieces and out:
            break
        out += "".join(pieces)
    return out


def cells(text):
    return [c.strip() for c in text.split("|") if c.strip()]


head = literal_block(SRC, "| fill % |")
row = literal_block(SRC, "| {pct:g} |")
sep = literal_block(SRC, '"|---:')
check("separator has 15 columns", sep.count("---:") == 15)
check("header has 15 columns", len(cells(head)) == 15)
check("header ends with the new columns",
      cells(head)[-3:] == ["match", "acc win", "bg"])
check("row prints as many cells as the header", len(cells(row)) == len(cells(head)))
check(
    "row formats the new cells",
    "fmt_pct(row['tail_match_frac'])" in row
    and "fmt2(row['acc_window_len'])" in row
    and "row['bg_drafts_pre'] + row['bg_drafts_post']" in row,
)
check("fmt_pct percent no decimals", bench.fmt_pct(1.0) == "100")
check("fmt_pct rounds to nearest", bench.fmt_pct(0.9849) == "98")
check("fmt_pct missing", bench.fmt_pct(None) == "-")
check("fmt2 two decimals", bench.fmt2(3.14159) == "3.14")
check("fmt2 missing", bench.fmt2(None) == "-")

print()
print("SELFTEST " + ("FAILED: " + ", ".join(FAILS) if FAILS else "PASSED"))
sys.exit(1 if FAILS else 0)
