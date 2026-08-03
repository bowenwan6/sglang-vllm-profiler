# Phase-6 smallest-cell A1/A2/A3 comparison — GPU 6 — 2026-08-03T17:14:58Z

**Purpose.** Execution_plan.md §3 — correctness (Gate 1) + coarse Nsight
metrics for A1/A2/A3 at the smallest cell (p=128, b=1), plus R3
Nsight-overhead disclosure via A0_nsys. Uses Phase-5 baseline
`gdn_a0_baseline_gpu6_20260803T162338Z/p128_b1_rep1` as the A0 reference.

## Configuration

- Cell: prompt_target=128 tokens (actual ~72–98 tokens on the Qwen
  tokenizer per the char-heuristic underscoring finding F3 in Phase 5),
  batch=1, n_warmup=2, n_timed=8, new_tokens=128, greedy.
- Gate-1 tolerance: **0.05** (Phase-5 noise floor = 0.0 at b=1, so
  `max(0.05, 3·0) = 0.05`).
- Per-arm arm-flags:
  - **A0** eager_eager: `--cuda-graph-backend-prefill=disabled --cuda-graph-backend-decode=disabled`
  - **A1** bcg_eager:   `--cuda-graph-backend-prefill=breakable --cuda-graph-backend-decode=disabled`
  - **A2** eager_dcg:   `--cuda-graph-backend-prefill=disabled --cuda-graph-backend-decode=full`
  - **A3** bcg_dcg:     `--cuda-graph-backend-prefill=breakable --cuda-graph-backend-decode=full`
- Frozen SGLang `58974ca16c…` (empty diff pre and post). Model `Qwen/Qwen3.5-4B @ 851bf6e8…`.

## Run summary (7 runs, all rc=0)

| run | arm | mode | server_ready | client_wallclock | gpu_returned_clean |
|---|---|---|---|---|---|
| A1_unprof | A1 | unprof | 49 s | 32 s | True |
| A1_nsys | A1 | nsys | 50 s | 39 s | True |
| A2_unprof | A2 | unprof | 37 s | 5 s | True |
| A2_nsys | A2 | nsys | 44 s | 5 s | True |
| A3_unprof | A3 | unprof | 47 s | 5 s | True |
| A3_nsys | A3 | nsys | 55 s | 6 s | True |
| A0_nsys | A0 | nsys | 37 s | 38 s | True |

(A2/A3 short wallclock is expected — full decode CG is ~6× faster than
eager decode.)

## R3 — Nsight-overhead disclosure

**Result: PASS.** Nsight profiling itself does not perturb tokens.

Comparison Phase-5 A0 rep1 (unprofiled) vs Phase-6 A0_nsys (profiled),
using the T4-hardened `gdn_correctness.gate_pairwise`:

| prompt id | verdict | tokens equal | max abs logprob diff |
|---|---|---|---|
| g1_short_qa_c256      | PASS | True | **0.0** |
| g1_short_qa_c4096     | PASS | True | **0.0** |
| g2_short_code_c256    | PASS | True | **0.0** |
| g2_short_code_c4096   | PASS | True | **0.0** |
| g3_short_multiturn_c256 | PASS | True | **0.0** |
| g3_short_multiturn_c4096 | PASS | True | **0.0** |
| g4_long_prose_c256    | PASS | True | **0.0** |
| g4_long_prose_c4096   | PASS | True | **0.0** |

`nsys profile --trace=cuda,nvtx,osrt,cublas,cudnn --sample=none` on a
Qwen3.5-4B eager server produces bit-identical tokens and per-token
logprobs to the same server without profiling. **All downstream
arm-vs-A0 comparisons can use A0 baseline records as reference; nsys
overhead does not confound.**

## Gate 1 — A0 (Phase-5 baseline) vs A1 / A2 / A3

| arm | passed / total | max abs logprob diff (across all timed pairs) | token mismatches |
|---|---|---|---|
| **A1** bcg_eager    | **5 / 8** | 1.33 | 2 prompts |
| **A2** eager_dcg    | **2 / 8** | 1.50 | 2 prompts |
| **A3** bcg_dcg      | **1 / 8** | 1.50 | 3 prompts |

**Gate 1 verdict: FAIL for every non-A0 arm.** Per plan §3 early-stop
rule, this yields a **provisional verdict of `FAIL_BCG_GDN_CORRECTNESS`**.

Per-prompt-tier breakdown (short = ~72 actual tokens, long = ~95):

|  | A0 vs A1 | A0 vs A2 | A0 vs A3 |
|---|---|---|---|
| SHORT (c256) — pass/fail | 2/2 | 1/3 | 0/4 |
| SHORT — max lp diff range | [0.00, 0.77] | [0.02, 0.91] | [0.08, 0.77] |
| LONG (c4096) — pass/fail  | 3/1 | 1/3 | 1/3 |
| LONG — max lp diff range  | [0.00, 1.33] | [0.03, 1.50] | [0.04, 1.50] |

Both prompt lengths show divergence; the effect is not length-specific.

## Coarse Nsight metrics (per-arm, one capture)

| arm | kernel_count_total | cudaLaunchKernel-family | cudaGraphLaunch | p50 launch gap | p95 launch gap |
|---|---|---|---|---|---|
| **A0** eager    | 679,282 | 679,281 |   **0** | 51.1 µs | 115.6 µs |
| **A1** BCG      | 772,406 | 791,552 | **363** | 53.7 µs | 118.9 µs |
| **A2** decode CG|  62,348 |  74,308 | **714** | 48.6 µs | 156.1 µs |
| **A3** both     | 173,057 | 204,163 |**1,246**| 58.0 µs | 160.6 µs |

**Findings:**

- **BCG really engaged (A1, A3).** `cudaGraphLaunch=363` on A1 = ~36
  graph launches per prefill × 10 requests. Confirms the runner's
  `--cuda-graph-backend-prefill=breakable` took effect.
- **Full decode CG really engaged (A2, A3).** `cudaGraphLaunch=714`
  on A2 = ~70–90 decode steps captured as a graph × 8 timed requests.
- **A1 launches 16.5 % more kernels than A0** (791,552 vs 679,281).
  Directional support for hypothesis `H_A` — kernel-count inflation
  under BCG prefill. **However, `MIN_CAPTURES_FOR_REPRO=2` is not met
  (only 1 capture per arm)** so `gdn_verdict.py` would not flag `H_A`
  on this data.
- **A2 launches 89 % fewer kernels than A0** (74,308 vs 679,281) —
  decode CG bundles ~128 decode kernels per step into one replay.
  Not a defect, just how full decode CG works.
- **p95 launch gap: A1 ≈ A0 (118 vs 116 µs).** No `H_C` support at
  this cell. Decode-CG arms (A2, A3) have p95 gap ~155–160 µs, which
  reflects the CPU wait between discrete graph completions — expected.

## Interpretation and provisional verdict

**Per plan §3 early-stop rule**: any Gate-1 FAIL → provisional
`FAIL_BCG_GDN_CORRECTNESS`. That rule fires here.

**But the mechanism is not BCG-only**. **A2 (eager prefill + decode
CG) also fails Gate 1**. A2 uses no BCG at all — its divergence must
come from decode CG. So the pattern is **"graph-replay disturbs
greedy-boundary token picks"**, not "BCG mis-handles GDN". Two
supporting facts:

- **A0 vs A0-nsys is bit-identical** (Nsight overhead ruled out).
- **A0 vs A0 self-repeat at Phase 5 (both unprofiled) is bit-identical**
  (baseline noise floor = 0.0).

So the drift is real cross-arm, not measurement noise. But it is
consistent with **numerical non-associativity in graph-replayed
reductions** flipping greedy top-1 picks when the top-1/top-2 logprob
gap is very small (< 0.05).

- **This is a real correctness effect at temperature=0 greedy**.
- **It is not unique to BCG** — full decode CG shows it too.
- **It may be "expected" numerical behavior** of any CUDA-graph-based
  inference, not a Qwen3.5/GDN-specific bug.

**Kernel-inflation direction for H_A** is preserved but unconfirmed
(1 capture only). Would need `MIN_CAPTURES_FOR_REPRO=2` runs of A0 and
A1 to firm up.

## Preservation invariants (verified post-Phase-6)

- `/data/sglang-fork` HEAD unchanged: `986c89e69c…`.
- Frozen SGLang HEAD unchanged: `58974ca16c…`, empty `git diff --stat`.
- GPU 6 memory 0 MiB post-run. GPU allowlist unchanged.

## Recommended next step (deferred to user direction per SIGNAL_AMBIGUOUS)

Two candidate paths, both non-destructive and within scope:

1. **Phase-7 threshold-ladder (evidence-triggered per plan §4)** —
   verify H_A support with `MIN_CAPTURES_FOR_REPRO=2` by running A0
   and A1 twice each at (p=128, b=1). If H_A firms up, move to
   p ~ 900 and p ~ 1200 to test the 1024-token alt-stream branch
   hypothesis directly.
2. **Correctness-first diagnosis** — reduce the failing Gate-1 prompt
   to a minimal reproducer, then check whether the same prompt at
   temperature > 0 (top-1 sampled) still diverges. If YES, real
   correctness bug. If NO, it's greedy-boundary numerical noise
   (expected under graph replay).

**Provisional verdict recorded: `FAIL_BCG_GDN_CORRECTNESS`** (per
strict plan §3), with the qualifier that A2 evidence suggests the
mechanism is shared with decode CG rather than BCG-specific. Full
verdict deferred to Phase 7 / final report.

## Files

- `driver.log`, `smallcell_summary.txt` — per-run driver artefacts.
- `<arm>_<unprof|nsys>/` — 7 subdirs.
  - `metadata.json`, `gpu_pre.txt`, `gpu_post.txt`, `preflight.json`,
    `runner_*.log`, `client_*.log`, `records_*.jsonl` — small
    evidence at cell root.
  - `raw/server_*.log`, `raw/*.nsys-rep` — heavyweight artefacts
    (gitignored).
  - `nsys/*.csv` (nsys mode only) — extracted per-cell metrics.
