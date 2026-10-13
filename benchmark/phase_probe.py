#!/usr/bin/env python3
"""phase_probe: run one fixed-salt decode cell and profile its steady state.

Wraps bench.py (single stream, 100% fill), watches /metrics, and POSTs
/start_profile once prefill is done and decode is flowing. The engine's
profiler config must carry max_iterations (see serve-4x-1m.sh PROF=1) so
the window self-closes and the box never writes an unbounded trace.

Requires: server booted with PROF=1 serve-4x-1m.sh.
Reads:    /tmp/prof on the server after the run (parse with phase_report.py).

usage: phase_probe.py [--base-url http://fender.lan:8000]
                      [--salt 7770] [--decode 4096] [--settle 3.0]
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request

HERE = __import__("pathlib").Path(__file__).parent


def metrics(url):
    txt = urllib.request.urlopen(url, timeout=5).read().decode()
    out = {}
    for line in txt.splitlines():
        for key in (
            "vllm:prompt_tokens_total",
            "vllm:generation_tokens_total",
            "vllm:num_requests_running",
        ):
            if line.startswith(key + " ") or line.startswith(key + "{"):
                out[key] = float(line.rsplit(" ", 1)[1])
    return out


def post(url):
    req = urllib.request.Request(url, data=b"", method="POST")
    return urllib.request.urlopen(req, timeout=30).read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://fender.lan:8000")
    ap.add_argument("--ctx", type=int, default=1000000)
    ap.add_argument("--salt", type=int, default=7770)
    ap.add_argument("--decode", type=int, default=4096)
    ap.add_argument("--settle", type=float, default=3.0,
                    help="seconds of steady decode before the trigger")
    ap.add_argument("--json", default=None, help="bench dump path for the cell")
    a = ap.parse_args()

    bench_cmd = [
        sys.executable, str(HERE / "bench.py"),
        "--base-url", a.base_url, "--ctx", str(a.ctx),
        "--fills", "100", "--concurrency", "1",
        "--salt", str(a.salt), "--decode", str(a.decode),
        "--label", f"phase-probe salt {a.salt}",
    ]
    if a.json:
        bench_cmd += ["--json", a.json]

    # NOTE: the engine must be booted with PROF=1 (max_iterations set), or
    # /start_profile opens an unbounded window - see the results.md warning.
    metrics(a.base_url + "/metrics")
    print("server reachable")

    proc = subprocess.Popen(bench_cmd)
    meturl = a.base_url + "/metrics"
    prev = None
    stable = 0.0
    triggered = False
    t0 = time.time()
    try:
        while proc.poll() is None and time.time() - t0 < 3600:
            time.sleep(1.0)
            try:
                cur = metrics(meturl)
            except Exception:  # noqa: BLE001
                continue
            if prev:
                d_prompt = cur["vllm:prompt_tokens_total"] - prev["vllm:prompt_tokens_total"]
                d_gen = cur["vllm:generation_tokens_total"] - prev["vllm:generation_tokens_total"]
                decode_only = d_prompt == 0 and d_gen > 0
                stable = stable + 1.0 if decode_only else 0.0
                if stable >= a.settle and not triggered:
                    triggered = True
                    print(f"[probe] steady decode after {time.time()-t0:.0f}s, "
                          f"POST /start_profile (auto-stop at max_iterations)")
                    try:
                        post(meturl.replace("/metrics", "/start_profile"))
                        print("[probe] profiler armed; window self-closes")
                    except Exception as exc:  # noqa: BLE001
                        print(f"[probe] start_profile FAILED: {exc}. Is the "
                              f"server booted with PROF=1?")
                        proc.terminate()
                        return 2
            prev = cur
    finally:
        rc = proc.wait()
    print(f"[probe] cell exited rc={rc}; traces should be in /tmp/prof "
          f"on the server (grep 'Stopping profiler' in the boot log to confirm)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
