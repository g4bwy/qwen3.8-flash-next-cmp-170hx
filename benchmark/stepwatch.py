#!/usr/bin/env python3
"""stepwatch: per-step engine telemetry without CUPTI.

The vLLM engine increments vllm:iteration_tokens_total once per scheduler
step. Sampling its cumulative counters fast gives step rate, mean step
duration, step token mix, acceptance, and queue state without touching
kernels or the engine. Run beside the server (localhost allows high rate;
remote over LAN is fine at default 20 Hz). Optionally records GPU clocks
and power via pynvml when run on the serving box.

usage: stepwatch.py --out cell.jsonl [--base-url URL] [--hz 20]
                    [--seconds N] [--nvml]
Send SIGINT (or Ctrl-C) to stop when running without --seconds.
"""
import argparse
import json
import re
import signal
import time
import urllib.request

WANT = {
    "it_count": r"^vllm:iteration_tokens_total_count\S* (\S+)",
    "it_sum": r"^vllm:iteration_tokens_total_sum\S* (\S+)",
    "it_b1": r"^vllm:iteration_tokens_total_bucket\{[^}]*le=\"1\.0\"[^}]*\} (\S+)",
    "it_b16": r"^vllm:iteration_tokens_total_bucket\{[^}]*le=\"16\.0\"[^}]*\} (\S+)",
    "drafts": r"^vllm:spec_decode_num_drafts_total\S* (\S+)",
    "accepted": r"^vllm:spec_decode_num_accepted_tokens_total\S* (\S+)",
    "gen": r"^vllm:generation_tokens_total\S* (\S+)",
    "prompt": r"^vllm:prompt_tokens_total\S* (\S+)",
    "running": r"^vllm:num_requests_running\S* (\S+)",
    "waiting": r"^vllm:num_requests_waiting\S* (\S+)",
    "preempt": r"^vllm:num_preemptions_total\S* (\S+)",
}


def sample(url, nvml_handles):
    txt = urllib.request.urlopen(url, timeout=5).read().decode(errors="replace")
    out = {}
    for key, pat in WANT.items():
        m = re.search(pat, txt, re.M)
        if m:
            out[key] = float(m.group(1))
    if nvml_handles:
        clocks, power = [], []
        for h in nvml_handles:
            try:
                clocks.append(h.clock())
                power.append(h.power())
            except Exception:  # noqa: BLE001
                pass
        if clocks:
            out["sm_clk"] = min(clocks)
            out["watts"] = sum(power)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hz", type=float, default=20.0)
    ap.add_argument("--seconds", type=float, default=0.0)
    ap.add_argument("--nvml", action="store_true")
    a = ap.parse_args()

    handles = []
    if a.nvml:
        try:
            import pynvml
            pynvml.nvmlInit()
            for i in range(pynvml.nvmlDeviceGetCount()):
                d = pynvml.nvmlDeviceGetHandleByIndex(i)
                handles.append(type("H", (), {
                    "clock": lambda d=d: pynvml.nvmlDeviceGetClockInfo(
                        d, pynvml.NVML_CLOCK_SM),
                    "power": lambda d=d: pynvml.nvmlDeviceGetPowerUsage(d) / 1000.0,
                })())
        except Exception as exc:  # noqa: BLE001
            print(f"nvml unavailable: {exc}")

    url = a.base_url.rstrip("/") + "/metrics"
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    prev_t = None
    prev = None
    t_end = time.time() + a.seconds if a.seconds else None
    with open(a.out, "w") as out:
        while not stop["now"]:
            try:
                cur = sample(url, handles)
            except Exception:  # noqa: BLE001
                time.sleep(0.2)
                continue
            t = time.time()
            if prev is not None:
                dt = t - prev_t
                if dt > 0:
                    row = {"t": round(t, 3)}
                    d_it = cur.get("it_count", 0) - prev.get("it_count", 0)
                    d_sum = cur.get("it_sum", 0) - prev.get("it_sum", 0)
                    d_b1 = cur.get("it_b1", 0) - prev.get("it_b1", 0)
                    d_b16 = cur.get("it_b16", 0) - prev.get("it_b16", 0)
                    d_dr = cur.get("drafts", 0) - prev.get("drafts", 0)
                    d_ac = cur.get("accepted", 0) - prev.get("accepted", 0)
                    d_g = cur.get("gen", 0) - prev.get("gen", 0)
                    d_p = cur.get("prompt", 0) - prev.get("prompt", 0)
                    row.update(
                        it_s=round(d_it / dt, 2),
                        step_ms=round(1000.0 * dt / d_it, 2) if d_it else None,
                        tok_it=round(d_sum / d_it, 1) if d_it else None,
                        it_small=round(d_b1 / d_it, 2) if d_it else None,
                        it_decode=round((d_b16 - d_b1) / d_it, 2) if d_it else None,
                        gen_tps=round(d_g / dt, 1),
                        prompt_tps=round(d_p / dt, 1),
                        acc=round(1 + d_ac / d_dr, 3) if d_dr else None,
                        run=cur.get("running"),
                        wait=cur.get("waiting"),
                    )
                    for k in ("sm_clk", "watts"):
                        if k in cur:
                            row[k] = cur[k]
                    if d_it or d_g or d_p:
                        out.write(json.dumps(row) + "\n")
                        out.flush()
            prev, prev_t = cur, t
            if t_end and time.time() > t_end:
                break
            time.sleep(max(0.0, 1.0 / a.hz - (time.time() - t)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
