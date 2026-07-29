# R6.5 attempt_gpu2 — INFRA_INCOMPLETE (foreign OOM at model load)

**Executed**: 2026-07-29T12:06 UTC on GPU 2.
**Machine verdict**: `AMBIGUOUS` (preserved verbatim under
[`verdict.md`](verdict.md) and [`verdict.json`](verdict.json)).

## What happened

R6.5 pre-launch check found GPU 2 idle (0 MiB, 0 util, no compute
PIDs). Runner launched the first server (ratio_0p2 / stock_default,
PID 49213). Between the pre-launch check and the sglang scheduler's
memory-pool init step, a foreign tenant landed on GPU 2 and grew to
~24 GB. Sglang then failed inside
`_resolve_memory_pool_config` with:

```
ValueError: Loaded weights leave no GPU memory for the KV cache
under --mem-fraction-static=0.8387624218749999. Raise
--mem-fraction-static above 0.862 (minimum viable = 1 -
available/pre = 0.8614). If using speculative decoding, draft
weights are now counted.
```

The Qwen3-VL-8B weights alone occupy ~93 GB in bf16; adding the
foreign ~24 GB leaves too little space for the KV cache under any
reasonable `mem-fraction-static`. The scheduler quit via
`Received sigquit from a child process` and `kill_process_tree`
tore down our own process cleanly.

Because of the `set -uo pipefail || return 1` chain in the runner,
subsequent ratio launches (0.5 and 0.8) also refused (`launch
failed for stock_default`), the whole R6.5 loop exited early, and
the verdict step reported `AMBIGUOUS` (< 3 ratios recorded, missing
server logs for the failed cells).

## Foreign-PID discipline

Never signalled. The tenant PIDs (2480524, 2491290, later a growing
set as the run progressed) live outside this container's PID
namespace (visible via `nvidia-smi`, absent from `/proc`). The
runner's TRACKED_PGIDS set only ever included our own PGIDs and
the `cleanup` EXIT trap signalled only those.

## Files preserved

- `verdict.md`, `verdict.json` — machine verdict AMBIGUOUS from
  `R6_5_verdict.py`.
- `raw/preflight.log`, `raw/launch_context.json` — provenance
  captured at start of run.
- `raw/ratio_0p2/stock_default/server.log` — the OOM crash trace.
- `raw/ratio_0p5/stock_default/{server.log, client.log}` —
  subsequent refused-launch records.

## Why this is INFRA and not FIX

The failure occurred **before any inference** — sglang could not
allocate its KV cache due to shared-GPU memory contention. It is
not a bug in either stock or fork. Neither the correctness fix nor
the PCG semantics were exercised. R6.1–R6.4 already established
the fix's safety and text/image performance story.

Per the user directive
("infrastructure failure — stop and report; wait for user"), R6.5
is halted until either GPU 2 becomes idle again or a different GPU
is authorized. R6.5 attempt should be retried under a fully-free
GPU with the same predeclared ratios (0.2, 0.5, 0.8) and identical
seed (42) so the analytical R6.4 prediction (fork wins across the
mix range) can be validated.
