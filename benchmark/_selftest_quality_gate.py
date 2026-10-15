#!/usr/bin/env python3
"""Offline self-test for logprob_capture.py + quality_diff.py v2.

Fakes urllib at the module level (no server, no GPU, no network), runs two
capture passes (clean and perturbed), then diffs: identical vs identical must
be exact (determinism mode), identical vs perturbed must trip thresholds.
"""
import io
import json
import math
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import logprob_capture as lc  # noqa: E402
import quality_diff as qd  # noqa: E402

FAILS = []


def check(name, cond):
    print(("ok  " if cond else "FAIL") + " " + name)
    if not cond:
        FAILS.append(name)


# --- unit: union_kl against a hand-computed value --------------------------
ea = [["a", math.log(0.5), 1], ["b", math.log(0.5), 2]]
eb = [["a", math.log(0.7), 1], ["b", math.log(0.3), 2]]
got = qd.union_kl(ea, eb)
want = 0.5 * math.log(0.5 / 0.7) + 0.5 * math.log(0.5 / 0.3)
check("union_kl hand value", abs(got - want) < 1e-9)
check("rank1 picks min rank", qd.rank1([["x", -3.0, 2], ["y", -0.1, 1]]) ==
      ("y", -0.1))
# missing token gets worst-minus-1 proxy and both sides renormalize
ec = [["a", math.log(0.5), 1], ["b", math.log(0.5), 2], ["c", -9.0, 3]]
check("union grows kl when one side hides mass",
      qd.union_kl(ea, ec) is not None and qd.union_kl(ea, ec) >= 0)

# --- fake HTTP layer --------------------------------------------------------
NTOK = 90  # prompt tokens the fake server "sees"


def entry(i, perturb):
    # deterministic 3-token top set per position
    base = [[f"t{i}_1", -0.20 - (i % 7) * 0.01, 1],
            [f"t{i}_2", -1.10, 2],
            [f"t{i}_3", -2.40, 3]]
    if perturb and i % 25 == 0 and i:  # flip the rank-1 token every 25 pos
        base = [[f"t{i}_1", -1.30, 2], [f"tX{i}", -0.20, 1],
                [f"t{i}_3", -2.40, 3]]
    return {t: {"logprob": lp, "rank": rk, "decoded_token": t}
            for t, lp, rk in base}


class FakeResp(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


STATE = {"perturb": False, "greedy_flip": False}


def fake_urlopen(req, timeout=None):
    url = req.full_url if hasattr(req, "full_url") else req
    if url.endswith("/v1/models"):
        body = {"data": [{"id": "fake/model"}]}
        return FakeResp(json.dumps(body).encode())
    payload = json.loads(req.data.decode())
    prompt = payload["prompt"]
    ntok = max(10, len(prompt) // 4)
    if payload.get("prompt_logprobs"):
        pl = [None] + [entry(i, STATE["perturb"]) for i in range(1, ntok)]
        body = {"choices": [{"index": 0, "text": "", "logprobs": None,
                             "prompt_logprobs": pl}],
                "usage": {"prompt_tokens": ntok, "total_tokens": ntok}}
    else:
        txt = "The lighthouse keeper logged the weather."
        if STATE["greedy_flip"] and len(prompt) % 3 == 0:
            txt = "The lighthouse keeper LOGGED the weather."
        body = {"choices": [{"index": 0, "text": txt}],
                "usage": {"prompt_tokens": ntok, "total_tokens": ntok}}
    return FakeResp(json.dumps(body).encode())


urllib.request.urlopen = fake_urlopen
lc.urllib.request.urlopen = fake_urlopen
import mtp_ab_probe  # noqa: E402,I001
mtp_ab_probe.urllib.request.urlopen = fake_urlopen

# --- build a tiny corpus ----------------------------------------------------
CORPUS = os.path.join(HERE, "_selftest_corpus.jsonl")
with open(CORPUS, "w") as fh:
    for w in range(40):
        rec = {"output": f"Some corpus sentence number {w}. " * 60}
        fh.write(json.dumps(rec) + "\n")


def run_capture(out):
    sys.argv = ["x", "--base-url", "http://fake:8000", "--corpus", CORPUS,
                "--out", out, "--k", "8", "--windows", "4",
                "--tail-prompts", "2", "--tail-tokens", "8",
                "--depths", "300,600", "--depth-tokens", "8",
                "--label", "selftest"]
    lc.main()


A = os.path.join(HERE, "_selftest_a.jsonl")
B = os.path.join(HERE, "_selftest_b.jsonl")
C = os.path.join(HERE, "_selftest_c.jsonl")
STATE["perturb"] = False
run_capture(A)
run_capture(B)  # identical
STATE["perturb"] = True
STATE["greedy_flip"] = True
run_capture(C)  # perturbed


def run_diff(x, y, extra=()):
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "quality_diff.py"), x, y,
         "--label-a", "X", "--label-b", "Y", *extra],
        capture_output=True, text=True)
    if r.returncode not in (0, 1) or r.stderr:
        print("DIFF STDERR:\n" + r.stderr)
    return r


# --- determinism mode: A vs B must be exact ---------------------------------
r = run_diff(A, B, ["--require-greedy-equal", "--max-mean-kl", "1e-12",
                    "--min-top1-pct", "100"])
check("identical diff exits 0", r.returncode == 0)
check("identical diff: zero flips", "top-1 flips: 0 (" in r.stdout)
check("identical diff: KL mean 0", "mean 0.00000" in r.stdout)
check("identical diff: PASS verdict", "VERDICT: PASS" in r.stdout)
check("identical diff: per-depth rows", "300 tok:" in r.stdout
      and "600 tok:" in r.stdout)
print(r.stdout)

# --- perturbation must be caught -------------------------------------------
r = run_diff(A, C, ["--max-mean-kl", "1e-9", "--min-top1-pct", "99.5"])
check("perturbed diff exits 1", r.returncode == 1)
check("perturbed diff reports flips", "top-1 flips: 0," not in r.stdout)
check("perturbed diff reports mean KL", "mean 0.00000" not in r.stdout)
check("perturbed diff FAIL verdict", "VERDICT: FAIL" in r.stdout)
check("perturbed diff greedy divergence", "differ:" in r.stdout
      or "diverged: 0" not in r.stdout)
print(r.stdout)

# --- legacy k=1 file (no meta) still diffs ---------------------------------
LEG = os.path.join(HERE, "_selftest_legacy.jsonl")
row1 = {"kind": "logprobs", "win": 0, "n": 5,
        "top1": [None, ["a", -0.2], ["b", -1.0], ["c", -0.5], None]}
row2 = {"kind": "greedy", "prompt_i": 0, "text": "hello"}
with open(LEG, "w") as fh:
    fh.write(json.dumps(row1) + "\n")
    fh.write(json.dumps(row2) + "\n")
LEG2 = LEG + ".b"
with open(LEG2, "w") as fh:
    fh.write(json.dumps(row1) + "\n")
    fh.write(json.dumps(row2) + "\n")
r = run_diff(LEG, LEG2)
check("legacy top1 files diff", r.returncode == 0
      and "no KL" in r.stdout)

# gzip path: real saved reference loads without error
GZ = os.path.join(HERE, "quality", "capture_official_boot3.jsonl.gz")
if os.path.exists(GZ):
    meta, lp, gr = qd.load(GZ)
    check("gz reference loads", len(lp) > 100)

if not FAILS:
    for f in (A, B, C, LEG, LEG2, CORPUS):
        os.remove(f)
print()
print("SELFTEST", "FAILED: " + ", ".join(FAILS) if FAILS else "PASSED")
sys.exit(1 if FAILS else 0)
