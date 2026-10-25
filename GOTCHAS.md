# Gotchas

Everything that bit this lane and would bite the next boot blind. Moved out of
README.md on 2026-10-26 so it can grow without inflating the main doc.
Measurements behind the claims: [benchmark/results.md](benchmark/results.md).

- The PLE table defaults to pinned host RAM as fp8 (`float8_e4m3fn`,
  `pinned=True`), several GB, measured before sizing other host workloads. The
  serve scripts run the file-gather path (patches 9-10) instead, so the table
  is page cache the kernel can reclaim, not pinned RSS.
- The `--hf-overrides` lines are base-sensitive. Since #56446 the YaRN limit is
  `max_position_embeddings` itself. On an earlier base drop that key, because
  the old rule computed `original x factor`.
- RoPE overrides must nest under `text_config`. The flat form on the model card
  writes an attribute nothing reads on this checkpoint, and the server then runs
  unscaled RoPE at length. Nesting merges per key, so `mrope_section` survives
  and vLLM builds `MRotaryEmbedding` with yarn scaling.
- On 1M, transformers warns that factor 4.0 does not match the implicit ratio
  3.81. Nothing is wrong: 1,000,000 is not 4 x 262,144, and the explicit
  factor wins.
- The first MTP decode step compiles `_fill_num_accepted_kernel` and stalls;
  warm up before the first measured cell. `bench.py --warmup` now runs that
  throwaway cell for you. Scatter after that is content, not the machine:
  pinned-salt repeats land within ~2%, rotating salts draw the prompt lottery
  at depth 4+ (see the salt law in results.md).
- The 1M pool holds 4.96 full requests on fp8 KV, 2.55 on bf16. Anything past
  residency queues, and a request preempted mid-decode replays its whole prompt
  under Mamba align mode, roughly 2 min per 1M at the default chunk budget.
- A long prefill that lands during decode starves every stream to under
  20 tok/s until it finishes. Schedule long prefills early, not between decodes.
- A bench fill of 100% means (ctx - decode) x 0.99. Anything larger is a plain
  HTTP 400 from the server.
- `RUST=1 ./serve-4x-1m.sh` boots the experimental Rust frontend (vllm-rs):
  warm 1M TTFT 4.7 -> 1.4 s, no /v1/messages route; audit in results.md.
- On bases between #59260 (2026-10-14) and #62129 (2026-10-18), engine init
  hangs with PP>1 plus spec decode: the last rank stops after its
  kernel-warmup lines, py-spy shows `warmup_kernels -> execute_model ->
  recv`, while the other ranks log "Graph capturing finished". The upstream
  fix gates the greedy warmup round on `speculative_config` so every stage
  mirrors it; backport that one line or boot with `--kernel-config
  '{"enable_jit_warmup": false}'` if you must stay on a windowed base. The
  current series base is past the fix.
- A KV cache whose cell size is not a multiple of 16 bytes pays a silent
  read-kernel penalty on Triton: head strides like 520 fail the
  divisible-by-16 specialization, vectorized gather loads drop out, and the
  QSA sparse read ran 4.9x slower on the int8 per-token-head lane until a
  whole-step trace (`benchmark/qsa_prefill_trace.py`, arms the worker-side
  profiler because a frontend torch.profiler sees zero V1 kernels) caught it.
  Scale slots in packed cells get 16 bytes even when the payload is 4. The
  general rule: time a strided-view kernel with the production strides, never
  with a freshly allocated contiguous stand-in.
