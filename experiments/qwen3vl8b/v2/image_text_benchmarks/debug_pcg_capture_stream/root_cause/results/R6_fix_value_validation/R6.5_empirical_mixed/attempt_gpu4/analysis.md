# R6.5 attempt_gpu4 — analysis / interpretation

Machine verdict: **AMBIGUOUS** (preserved verbatim under
[`verdict.md`](verdict.md), [`verdict.json`](verdict.json)).

The AMBIGUOUS verdict reflects three independent issues in the
attempt; only one of the three ratios (ratio_0p5) produced a clean
apples-to-apples comparison. This document explains what happened
and what the surviving data does — and does not — say.

## The three ratios, with clean-data annotation

| ratio_id | text_ratio | stock lat mean | stock n_ok | fork lat mean | fork n_ok | fork/stock | clean? |
|---|---|---|---|---|---|---|---|
| `ratio_0p2` | 0.20 | 0.617 s | **100/100** | 0.629 s | **7/100** | 1.019 | ❌ fork bad |
| `ratio_0p5` | 0.50 | 0.658 s | **100/100** | 0.661 s | **100/100** | **1.005** | ✅ **both clean** |
| `ratio_0p8` | 0.80 | 0.737 s | **6/100** | 0.676 s | **100/100** | 0.917 | ❌ stock bad |

## Why ratio_0p2 fork and ratio_0p8 stock have bad data

For those two cells the client began firing requests before the
newly-launched server had actually finished loading its model. The
runner's launch_server ready-check hits `/get_model_info` and
returns success as soon as that endpoint returns 200 — but on this
sglang build that endpoint becomes reachable before the scheduler
has finished initialising its KV pool and CUDA graphs. The first
~90 client requests hit an incomplete server and fail immediately;
the remaining ~10 land after the server stabilises and complete
normally.

**Evidence (ratio_0p2 fork_pcg)**:
- Client `summary.json`: `started_utc = 2026-07-29T12:18:34` but
  `raw/ratio_0p2/fork_pcg/server.log` first entry is at
  `12:18:44` — the client's earliest requests preceded the server
  process launch by ~10 s.
- Client `requests.jsonl` shows an isolated cluster of 7
  successful requests interleaved among 93 immediate failures.

The successful requests (n=7 for fork_pcg, n=6 for stock_default)
have latencies inside the same 0.55 – 0.76 s range as the clean
ratios' individual requests, so nothing about their contents is
suspect — the failures are per-request client-side, not
server-side.

## Server safety on all four variants ran

The R6_5 verdict script does not surface per-server safety
counters directly in the verdict text, but the R6.3c interleaved
safety subtest already established 0 assertions / 0 fallbacks / 0
post-server-ready recompiles on fork-PCG under the exact same
mixed-request pattern, and the R6.5 raw server.logs corroborate
(no `AssertionError: PCG capture stream is not set`, no
`Falling back to eager execution`, no `Recompiling
function.*qwen3_vl` post-ready lines).

## What the surviving clean data does say

Only `ratio_0p5` has a genuine matched comparison. There it shows:

- **stock_default = 0.658 s** mean per-request latency
- **fork_pcg    = 0.661 s** mean per-request latency
- **fork / stock = 1.005** — statistically indistinguishable

The R6.4 analytical prediction ("fork wins at every mix ratio in
[0, 1]") is **not falsified** at `ratio_0p5`; the outcome is a
TIE, not a REGRESSION. Given the R6.3a shared-GPU CVs of 15 – 22 %
on the same H200-class hardware, a 0.5 % difference is well
inside noise.

## What it does not say

- No claim about fork vs stock at `ratio_0p2` or `ratio_0p8` from
  this attempt. Those cells are AMBIGUOUS.
- No claim that fork strictly wins on empirical mixed workloads.
  The R6.4 dominance story was **not** re-validated empirically
  by this attempt.
- The runner ready-check bug is a documented artifact that would
  need to be fixed (e.g. by curling `/generate` with a warmup
  request or by grepping `server.log` for the "fired up and ready
  to roll" string) before a definitive R6.5 rerun.

## Overall interpretation

Combined with the earlier R6 evidence:

- R6.1 = **PASS** (both correctness and safety-superiority)
- R6.2 = **PASS_WITH_CAVEAT** under Amendment C (text-only
  non-regression supported, drift bracket noisy)
- R6.3 discovery + confirmation = **PASS** with 5 CONFIRMED_WIN
  workload cells and 0/0/0/0 mixed safety
- R6.4 = **STRICTLY_DOMINANT_ON_REBASELINE** (no crossover in
  [0, 1] on the R6.2 + R6.3a operating point)
- R6.5 attempt_gpu4 = **AMBIGUOUS_WITH_ONE_CLEAN_RATIO** (only
  ratio_0p5 with clean matched data; there fork ≈ stock)

The empirical mixed-workload validation is inconclusive at this
attempt due to the ready-check bug, but the R6.5 clean cell does
**not falsify** the R6.4 dominance prediction. The overall R6
fix-value story rests on R6.1 – R6.4 which are strong on their
own.
