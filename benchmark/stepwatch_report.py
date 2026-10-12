#!/usr/bin/env python3
"""Summarize a stepwatch JSONL: separate the three ways a cell goes slow.

step_ms drift = machine (clocks, power, thermal)
acc drift      = draft/target agreement (content, gates, staleness)
run/wait mix   = scheduling (queueing, admission)

usage: stepwatch_report.py cell.jsonl
"""
import json
import statistics as st
import sys

rows = [json.loads(l) for l in open(sys.argv[1])]
busy = [r for r in rows if r.get("step_ms")]
if not busy:
    print("no decode samples")
    sys.exit(0)


def series(key, src=busy):
    return [r[key] for r in src if r.get(key) is not None]


def line(name, vals, unit=""):
    if not vals:
        return
    med = st.median(vals)
    spread = (max(vals) - min(vals)) / med * 100 if med else 0
    print(f"{name:10s} median {med:8.2f}{unit}  min {min(vals):8.2f}  "
          f"max {max(vals):8.2f}  spread {spread:5.1f}%")


line("step_ms", series("step_ms"))
line("it_s", series("it_s"))
line("acc", series("acc"))
line("gen_tps", series("gen_tps"))
if series("sm_clk"):
    line("sm_clk", series("sm_clk"), " MHz")
    line("watts", series("watts"), " W")

pf = [r for r in rows if (r.get("prompt_tps") or 0) >= 10]
if pf:
    pfm = [r["step_ms"] for r in pf if r.get("step_ms")]
    print(f"prefill/mixed samples: {len(pf)} "
          f"(avg tok/iter {st.mean(series('tok_it', pf)):.0f}, "
          f"step_ms median {st.median(pfm) if pfm else 0:.1f})")
dec = [r for r in busy if (r.get("prompt_tps") or 0) < 10 and (r.get("gen_tps") or 0) > 20]
if dec:
    print(f"pure-decode samples: {len(dec)}, "
          f"step_ms median {st.median(series('step_ms', dec)):.1f}, "
          f"acc median {st.median(series('acc', dec)):.3f}")

# bimodality flag: coefficient of variation of decode step_ms
if len(dec) > 8:
    v = st.pvariance(series("step_ms", dec)) ** 0.5 / st.mean(series("step_ms", dec))
    print(f"decode step_ms CV: {v * 100:.1f}% " + ("(wide; cross-check acc column - content, not clocks)" if v > 0.06 else "(stable)"))
