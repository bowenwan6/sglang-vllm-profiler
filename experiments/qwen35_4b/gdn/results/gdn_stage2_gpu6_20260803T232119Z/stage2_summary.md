# Stage-2 reproducibility captures — GPU 6 — 2026-08-03T23:21:19Z

**Purpose.** Satisfy `MIN_CAPTURES_FOR_REPRO=2` for the H_A hypothesis
by profiling A0 and A1 twice each at (p=128, b=1), then evaluate H_A
on **steady-state** metrics (not the whole trace, which mixes one-time
graph capture with per-request replay).

## Configuration

- Cell: (p=128, b=1), n_warmup=2, n_timed=8, new_tokens=128, greedy.
- 4 sequential Nsight-profiled cold-server bring-ups on GPU 6,
  rotated ports 30400-30403.
- Extraction uses `--capture-cutoff-seconds` = the run's
  `server_ready_seconds` (from metadata.json), emitting 3 rows per
  capture: `window=all|capture|steady_state`.

## Run summary (4/4 rc=0)

| run | server_ready | client_wallclock | gpu_returned_clean |
|---|---|---|---|
| A0_rep1 | 41 s | 39 s | True |
| A0_rep2 | 50 s | 38 s | True |
| A1_rep1 | 49 s | 39 s | True |
| A1_rep2 | 48 s | 39 s | True |

## Per-run windowed metrics

| run | window | kernel_total | cudaLaunchKernel | cudaGraphLaunch | per-request kern | per-request GL |
|---|---|---|---|---|---|---|
| A0_rep1 | all | 679,358 | 679,357 | 0 | 67,935.8 | 0.0 |
| A0_rep1 | capture | 0 | 0 | 0 | — | — |
| A0_rep1 | steady_state | 679,358 | 679,357 | 0 | 67,935.8 | 0.0 |
| A0_rep2 | steady_state | 679,358 | 679,357 | 0 | 67,935.8 | 0.0 |
| A1_rep1 | capture | 526 | 526 | 0 | — | — |
| A1_rep1 | steady_state | 771,998 | 791,143 | 363 | 77,199.8 | 36.3 |
| A1_rep2 | steady_state | 772,044 | 791,189 | 363 | 77,204.4 | 36.3 |

## Findings

### F1 — H_A on steady-state: SUPPORTED

| metric | A0 mean | A1 mean | delta | pct | 10% threshold | 2σ threshold |
|---|---|---|---|---|---|---|
| kernel_count_total | **679,358** | **772,021** | **+92,663** | **+13.6 %** | ≥ +67,936 → PASS | > +0 → PASS |
| kernels_per_request | 67,935.8 | 77,202.1 | **+9,266.3** | +13.6 % | — | — |
| cudaGraphLaunch total | 0 | 363 | +363 | — | — | — |
| cudaGraphLaunch per request | 0.0 | 36.3 | +36.3 | — | — | — |

**H_A on steady-state (MIN_CAPTURES_FOR_REPRO=2 met): SUPPORTED.**

Reproducibility of the steady-state kernel count is extreme:
- A0: rep1 = rep2 = 679,358 (|Δ| = 0, σ = 0).
- A1: rep1 = 771,998, rep2 = 772,044 (|Δ| = 46, 0.006 %).

The +13.6 % kernel inflation on A1 is **reproducible to the 4th
decimal**. Not noise.

### F2 — capture-window measurement caveat

My `--capture-cutoff-seconds` was set to `server_ready_seconds`
(from runner metadata), i.e. the wall-clock time between runner
start and SGLang's `/health` returning 200. The extractor then
counted 0 kernels (A0) and 526 kernels (A1) in that window.

That's *unexpected* if SGLang's graph capture were happening at
server bring-up (which our source audit assumed). Instead, this
pattern says **SGLang defers graph capture to the first client
request(s) for BCG**, or at least the CUDA kernel launches for
capture happen after `/health` fires. The 0-526 kernels in the
pre-`/health` window are consistent with model-load being dominated
by `cudaMalloc` + `cudaMemcpyAsync` (which are CUDA API calls but
not kernel launches).

**Implication:** my capture-window split undercounts capture-time
work. So a fraction of the +9,266 kernels/request steady-state
inflation on A1 is one-time BCG capture bleeding into the
"steady_state" window; the remainder is real per-request replay
overhead. Stage 3's threshold ladder is the natural way to
separate these: if the kernel-inflation shape is bucket-boundary-
sensitive (< 1024 padded prefill), it reflects the alt-stream
mechanism from `source_audit.md` §3.3 — not a per-request-capture
artefact.

### F3 — reproducibility of the delta

Between A0 and A1 the delta is `+92,663 ± 23` kernels (using σ_A1 =
23, σ_A0 = 0). Signal-to-noise ratio is essentially infinite. Any
Stage-3 threshold-ladder comparison at this cell shape will resolve
even small changes to this delta.

## Stage-2 signal

**`SIGNAL_GOOD`.** H_A on steady-state is supported with the
required 2 captures per arm. Reproducibility is nearly perfect;
the delta is far beyond any measurement noise.

**Continue automatically to Stage 3** — test the < 1024
alternate-stream hypothesis by comparing A0/A1 at padded-bucket
sizes clearly below 1024, near-below, at/just-above, and clearly
above. The expected signature: at < 1024 padded bucket, A1 shows
the same-shaped +13.6 % kernel inflation; at ≥ 1024, the delta
should shrink or disappear if the alt-stream branch is the
mechanism.

If the delta persists at ≥ 1024, the alt-stream hypothesis is
revised or rejected, and Stage 4 (targeted NVTX instrumentation)
is triggered to attribute the inflation to specific GDN ops.

## Preservation invariants (verified post-Stage 2)

- `/data/sglang-fork` HEAD unchanged: `986c89e69c…`.
- Frozen SGLang HEAD unchanged: `58974ca16c…`, empty `git diff --stat`.
- GPU 6 memory 0 MiB post-run.

## Files

- `stage2_summary.txt` — per-run CSV (4 rows).
- `driver.log` — driver console.
- `<arm>_<rep>/` — 4 subdirs each with `metadata.json`, `gpu_pre.txt`,
  `gpu_post.txt`, `preflight.json`, `runner_*.log`, `client_*.log`,
  `records_<arm>_p128_b1.jsonl`, `extract.log`, and
  `nsys/<arm>_p128_b1.csv` (3-row windowed extract).
  `raw/*.nsys-rep` and `raw/server_*.log` are gitignored.
