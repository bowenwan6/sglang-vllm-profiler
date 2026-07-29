# R6.3 attempt_gpu6 — INFRA_INCOMPLETE (superseded)

**Executed**: 2026-07-29 UTC on GPU 6 (empty at pre-launch, foreign
tenant arrived mid-run).
**Runner SHA**: pre-fix `run_R6_3_image_and_sweep.sh` (before commit
`dd93c43`).
**Verdict as generated**: `FAIL` (machine, preserved verbatim under
[`verdict.md`](verdict.md) and [`verdict.json`](verdict.json)).

**Do not** reuse the numbers in this directory as the R6.3 headline.
attempt_gpu6 is preserved as evidence of the two runner bugs discovered
here and fixed in `dd93c43`; the authoritative R6.3 run is
`attempt_gpu2/` on GPU 2.

## What actually happened

### R6.3a — completed cleanly on both variants
| variant | reps | mean_ttft_ms | median | CV% | safety |
|---|---|---|---|---|---|
| stock_default | 3/3 | 98.381 | 92.321 | 13.24 | 0/0/0 |
| fork_pcg | 3/3 | 93.230 | 84.287 | 22.44 | 0/0/0 |

Fork/stock delta = **-5.15 ms** (ratio **0.9476**, fork 5.2 % *faster*
in mean TTFT at 720p 1-image 128→128 c=1). Both CVs are large (13 %
and 22 %) — GPU 6's neighbour landscape shifted during the ~30 min
that R6.3a occupied. These numbers are indicative only; the
authoritative R6.3a is on GPU 2 with the fixed runner.

### R6.3b — half of the sweep cells were skipped by CLI validation

The runner passed `--image-resolution 224p` for six of the twelve
cells. `sglang.benchmark.serving` validates against `{4k, 1080p, 720p,
360p}` (see `image.py:90` `parse_image_resolution`) and raised
`ValueError: Unsupported image resolution: 224p` before serving any
request. Those six cells (all `_r224p_*` variants) have empty
`bench.jsonl` and no numbers.

The six `_r720p_*` cells that did complete:

| cell | stock_default (ms) | fork_pcg (ms) | ratio | fork ≤ stock? |
|---|---|---|---|---|
| `cell_t128_r720p_c1` | 107.082 | 109.925 | 1.0265 | ❌ |
| `cell_t128_r720p_c4` | 183.497 | 182.591 | **0.9951** | ✅ (tie) |
| `cell_t512_r720p_c1` | 86.339 | 89.561 | 1.0373 | ❌ |
| `cell_t512_r720p_c4` | 181.545 | 179.402 | **0.9882** | ✅ (fork -1.2 %) |
| `cell_t2048_r720p_c1` | 143.162 | 147.960 | 1.0335 | ❌ |
| `cell_t2048_r720p_c4` | 379.357 | 402.617 | 1.0613 | ❌ |

Winning-looking cells (need confirmation reps under the revised
framework): `cell_t128_r720p_c4` and `cell_t512_r720p_c4` — both
concurrency-4 cells. Discovery-strength only. The authoritative sweep
lands in `attempt_gpu2/`.

### R6.3c — mixed-safety client ran against a nonexistent server

At the R6.3c launch step the runner's pre-launch check reported
`GPU 6 pre-launch: mem=709MiB util=8% pids=[433771]`. PID 433771
belongs to a foreign tenant (visible via `nvidia-smi` but absent from
our container's `/proc`; nvidia-smi later showed PID 433771 sustaining
90 GiB and 100 % util on GPU 6, i.e. an actively-scheduled foreign
process). `pre_launch_idle` correctly refused → `launch_server`
returned 1.

The bug: the R6.3c launch site did not check that return code. The
mixed-safety client then made 100 requests to `http://127.0.0.1:30003`
where nothing was listening; every request failed with
`http_status: None`. `client_summary.json` records
`request_failures: 100` and `completed: 0`. The
`c_mixed_safety/server.log` file does not exist (because no server was
ever launched).

R6.3c cannot be evaluated from this attempt. The runner fix in
`dd93c43` adds a 20 × 15 s launch-retry loop and refuses to invoke
the client if the server does not come up.

## Foreign-PID discipline

PID 433771 was never signalled. All process management stayed within
`TRACKED_PGIDS` for the runner's own descendants (sweep servers
232537 and 227777 were TERM/KILL-ed by our own teardown before
exiting). The `cleanup` EXIT trap fired at shell exit and touched
only our own PGIDs.

## Files preserved

- `verdict.md`, `verdict.json` — machine verdict from the pre-fix
  runner; `FAIL` (mixed-safety), 720p sweep numbers populated, 224p
  sweep numbers empty.
- `raw/preflight.log`, `raw/launch_context.json` — provenance
  captured at start of run.
- `raw/a_rebaseline/{stock_default,fork_pcg}/` — three reps each,
  bench.jsonl present, server.log per variant, indicative only.
- `raw/b_sweep/_server_{stock_default,fork_pcg}/server.log` — sweep
  server logs (one per variant, shared across cells).
- `raw/b_sweep/cell_t*_r720p_c*/` — bench.jsonl + bench.log per cell
  per variant (six cells × two variants).
- `raw/b_sweep/cell_t*_r224p_c*/{bench.log}` — bench.log documents
  the CLI ValueError; bench.jsonl absent.
- `raw/c_mixed_safety/{client.log,client_summary.json,
  fork_pcg_interleaved.jsonl}` — all 100 requests recorded as
  failures; server.log absent.

None of this data is deleted. attempt_gpu2 supersedes for the
headline finding.
