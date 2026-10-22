# benchmark/

Measurement tools and the record behind the numbers in the top-level README.
Everything here is stdlib Python talking HTTP to a running server, except the
kernel-level benches noted below. Reproduce commands are at the end of
[results.md](results.md).

## The record

- `results.md` - the full chronicle: grids, methodology, verdicts, salt law,
  step anatomy, reproduce commands. README.md summarizes. This file proves.
- `bench-*.json`, `mtp-4x1m-*.json`, `sustained-decode.json` - raw per-cell
  dumps (bench.py `--json`) behind those tables. Names carry the config they
  measured. Stale-looking names are history, kept for provenance.
- `quality/*.jsonl.gz` - raw logprob captures behind the W8A16 quality gate.

## Load cells

- `bench.py` - fill sweep x concurrency -> the table with TTFT, prefill,
  wall, out tok/s, sustained, acceptance, preempts. Salt is time-derived by
  default. Pin `--salt` for comparisons and rotate it for headline ranges (salt
  law). Dumps per-cell JSON and a live progress line. `--warmup` runs one
  throwaway short cell before the grid, which kills the first-MTP-step JIT
  stall, and `--cooldown SECS` sleeps between cells. A background-traffic gate
  probes the spec-decode counters before and after each cell and prints a
  WARNING line on nonzero background drafts. Appended columns: `match` (share
  of sentence lines in the last 40 lines of each stream that reached the
  repeat loop; reasoner preamble and the max_tokens-cut tail line excluded,
  thinking-only cells read `-` as inconclusive), `acc win` (mean
  acceptance length inside the all-decoding window) and `bg` (background
  drafts seen pre+post). The legacy `acc` column is full-window and counts
  prefill-phase drafts on bases at or after `8407bc0bb6`, so it dilutes toward
  the prefill ratio on deep-fill cells. Use `acc win`. The matching per-cell
  JSON keys are `tail_match_frac`, `completion_sha256`, `acc_window_len`,
  `bg_drafts_pre` and `bg_drafts_post`.
- `sustained_decode.py` - batched-decode capacity plateaus at chosen depth
  and stream counts.

## Telemetry and step anatomy

- `stepwatch.py` (+ `stepwatch_report.py`) - per-step rate, mean step
  duration, token mix, acceptance from `/metrics` deltas, no CUPTI. Run it
  ON the serving box: a 20 Hz remote poll aliases a 21 ms step into a 32 ms
  ghost. `--nvml` adds clocks and power.
- `phase_probe.py` - runs the fixed-salt 100%-fill cell and fires
  `/start_profile` in steady decode; needs a `PROF=1` boot (capped window,
  self-closing).
- `phase_report.py` - parses the four per-rank chrome traces from that
  window into the decode step budget (cadence, spin vs compute, kernel
  categories, duration histogram). Run where the traces land (logs/prof).
- `humming_decode_bench.py` - kernel-level: dense/humming GEMM shapes at
  decode m, CUDA-event timed, min-composed scoring. Needs CUDA; not part of
  the HTTP tools.

## Acceptance and quality

- `mtp_ab_probe.py` - acceptance A/B at depth; doubles as the shared client
  lib for the other probes.
- `corpus_accept_probe.py` - acceptance over the real-text corpus
  (`../draft_corpus.jsonl`), greedy or sampled. `--modes greedy,sampled` runs
  both in one invocation and prints a per-mode table with rates as well as
  acceptance, so it is the real-content comparator for the battery. The seed
  is time-derived by default, so runs do not reuse the prefix cache.
- `logprob_capture.py` + `quality_diff.py` - the quality gate: teacher-forced
  fingerprints per boot (top-1 by default, `--k 8` adds top-k sets so the diff
  also reports union KL, and `--depths 8200,16400 --lp-stride 4` adds
  long-context rows where a KV format change bites). Diff a boot against its own repeat
  run first for the noise floor, then against the stored reference; threshold
  flags make the diff exit non-zero. `--require-greedy-equal` is the
  determinism mode (issue #54521 stress: same boot twice, all metrics must be
  zero). `_selftest_quality_gate.py` exercises both files with a fake server,
  no GPU needed.

## Prefix caching

- `prefix_cache_probe.py` - cached vs recomputed tokens per conversation
  turn.
- `prefix_state_check.py` - cache-hit state correctness; exits non-zero.

## Conventions

- All HTTP tools default to `http://127.0.0.1:8000`. Always pass
  `--base-url http://fender.lan:8000` when the server is the remote box.
- bench.py probes the spec-decode counters before and after every cell and
  flags contamination, but the box must still be idle. A probe catches traffic
  during the probe window, not during the cell, and PP decode is CPU-sensitive
  (ZMQ hop RPCs between stages), so external load silently inflates step time.
- The owner still says when the machine is quiet.
- Kernel-level changes read fixed-salt acceptance FIRST, timing second:
  timing can look great while values rot (L2 eviction incident, results.md).
- `sampler.py`, `picked.json`, `decode_probe.py` are untracked local scratch.
