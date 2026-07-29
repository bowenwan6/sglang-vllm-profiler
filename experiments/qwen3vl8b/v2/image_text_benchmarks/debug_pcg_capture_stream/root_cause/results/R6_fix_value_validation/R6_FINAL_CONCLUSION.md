# R6 Fix-value Validation — Final Conclusion (corrected after provenance audit)

**Audited verdict**:

- ✅ `CORRECTNESS_AND_SAFETY_FIX_PASS`
- ⚠️ `PERFORMANCE_VALUE_PROMISING_BUT_NOT_FINAL`
- ❌ Overall mixed-workload **performance dominance is NOT established**.

An earlier `R6_FINAL_CONCLUSION.md` on this branch reported an
unqualified `R6 = PASS` and claimed empirical mixed-workload
dominance. A subsequent audit (recorded here) found provenance and
statistical-strength defects in the R6.5 and R6.4 conclusions that
that document treated as established. This document supersedes it.
Historical machine-generated verdict `.json` / `.md` files
(preserved verbatim on this branch) still contain the original
wording; the summary below is the authoritative current statement.

## What the fix actually accomplishes (established)

Fork branch `fix/pcg-vlm-deepstack-warmup` HEAD
`986c89e69c25882ab6f3d396f8eb306f38f2c8d2` against stock
`da802ddcafe55e25b3e1db86b1e0444afc3e05bc`:

1. **Stock `da802ddca` reproduces the repeated-shape multimodal PCG
   capture-stream assertion.** On the historical R1/E2a workload
   (720p × 32 requests, `--random-input-len 128
   --random-range-ratio 1.0 --num-prompts 32 --warmup-requests 30`,
   `--enforce-piecewise-cuda-graph`) stock crashes with the exact
   `AssertionError: PCG capture stream is not set` at
   `cuda_piecewise_backend.py:172` on the second-same-shape prefill
   call after the post-server-ready multimodal recompile
   (`num_finished_warmup 1→2`, `runtime_shape=1024`,
   server_log:44322 for the recorded attempt).
2. **Fork `986c89e69` completes the identical bench cleanly.**
   Under the exact same recipe: 30 warmup + 32 measured requests,
   0 assertions, 0 fallbacks, 0 post-server-ready inflight
   recompiles, 32/32 bench-completed.
3. **Cache-matched cold-cache correctness comparisons pass.**
   Fork-PCG vs fork-default bit-identical on the image path; all
   cross-config comparisons fit inside the matched-repeat
   determinism envelope; the interleaved mixed-modality leg on one
   fork-PCG server shows 0 assertions / 0 fallbacks / 0 request
   failures / 0 post-server-ready recompiles.

⇒ **`R6.1 = PASS`** — safety-superiority + correctness combined.

## Text-only non-regression (`R6.2 = PASS_WITH_CAVEAT`)

Case A text-only benchmark on the Qwen3-VL server (`caseA_short.jsonl`,
128→128 tokens, c=1, n=400, 5 reps, 4 variants).

**Original machine verdict was FAIL** on a single axis: the
`stock_default → stock_default_repeat` drift bracket returned
**3.050 %**, over the pre-declared 3.0 % threshold by 0.05 pp
(preserved verbatim in
[`R6.2_text_only_caseA/attempt_gpu0/verdict.md`](R6.2_text_only_caseA/attempt_gpu0/verdict.md)).
`R6.2 Protocol Amendment C`
([`R6.2_text_only_caseA/protocol_amendment_C_shared_gpu_drift_gate.md`](R6.2_text_only_caseA/protocol_amendment_C_shared_gpu_drift_gate.md))
was drafted after the run and reclassifies the drift bracket as a
shared-GPU nuisance-control (buckets: `≤3 %` clean; `3–5 %`
`PASS_WITH_CAVEAT`; `>5 %` rerun / AMBIGUOUS). Under Amendment C,
R6.2 is classified `PASS_WITH_CAVEAT — TEXT_NON_REGRESSION_SUPPORTED`.
Amendment C is a post-hoc reclassification and is transparently
recorded as such; the original machine FAIL is preserved.

Substantive numbers (unchanged):

| variant | mean TTFT (ms) | CV % |
|---|---|---|
| `stock_default` | 26.86 | 5.91 |
| `stock_pcg` | 18.35 | 2.29 |
| `fork_pcg` | 17.65 | 2.02 |
| `stock_default_repeat` | 27.68 | 2.51 |

**Fork-PCG / stock-PCG mean-TTFT ratio = 0.9617.** This supports
text-path non-regression relative to stock-PCG. The 3.8 %
mean-of-means difference is **within the shared-GPU noise band**
that Amendment C was drafted to accommodate; treat it as
"non-regression, likely equivalent," **not** as a statistically
proven 3.8 % speedup.

## Image-path cost + workload sweep + mixed safety (`R6.3`)

Executed on GPU 2 (attempt_gpu2) 2026-07-29T09:41–10:57 UTC, then
3-rep confirmation on the discovery-winning cells
(attempt_gpu2_confirm).

**R6.3c mixed-modality safety = PASS.** Interleaved 50 text + 50
image requests on one fork-PCG server: 0 request failures, 0
assertions, 0 fallbacks, 0 post-server-ready recompiles. This is
the strongest R6.3 claim.

**R6.3a rebaseline (720p 1 image, 128→128 c=1 n=400 × 3):**

| variant | mean TTFT (ms) | CV % |
|---|---|---|
| `stock_default` | 94.35 | 15.51 |
| `fork_pcg` | 87.01 | 21.85 |

The mean-of-means ratio is 0.9222 (fork appears 7.8 % faster), but
per-variant CVs of 15 – 22 % on 3 reps make the point estimate
noisy. **Do not cite the 7.8 % figure as an established
performance headline.**

**R6.3b sweep (12 cells, n=100 per cell, single rep for
discovery + 3 rep confirmation):**

Machine numbers are recorded per cell in
[`R6.3_image_and_sweep/attempt_gpu2/verdict.md`](R6.3_image_and_sweep/attempt_gpu2/verdict.md)
and
[`R6.3_image_and_sweep/attempt_gpu2_confirm/verdict.md`](R6.3_image_and_sweep/attempt_gpu2_confirm/verdict.md).

Cleanest promising cell after confirmation:

- **`cell_t512_r360p_c1`**: stock 65.14 ms (CV **4.01 %**) vs fork
  59.52 ms (CV **4.64 %**) — fork **~8.6 %** mean improvement with
  low CV on both sides. This is the strongest per-cell performance
  signal in R6.3 that survives noise.

The headline-candidate `cell_t512_r360p_c4` (ratio 0.7806, ~22 %
margin) has per-variant CVs of **27.07 % / 32.93 %** after
confirmation; its point estimate is **exploratory only**. The
prior conclusion document elevated it to a "headline" — that
framing is retracted. `cell_t128_r360p_c4` (ratio 0.8176, ~18 %
margin) has per-variant CVs of **15.00 % / 10.37 %** — better
than the 22 %-margin cell but still noisier than a defensible
headline.

Loss regime: every `t2048_*` cell (long text) shows fork ≥ stock.
This is reported transparently.

Overall R6.3 reading:

- Mixed-modality safety = **PASS**;
- Performance results = **promising but exploratory**; the strongest
  per-cell datapoint is `cell_t512_r360p_c1` ~8.6 %; wider claims
  (aggregate 34 % TTFT reduction, 22 % headline, etc.) are not
  supported by the CV profile of a shared-GPU run.

## Analytical crossover (`R6.4 = AMBIGUOUS`)

`R6_4_crossover.py` machine verdict: **AMBIGUOUS**. Preserved
verbatim in [`R6.4_analytical_crossover/crossover.md`](R6.4_analytical_crossover/crossover.md).

Rep-level arithmetic means from R6.2 + R6.3a:

- G (retained text gain) = mean(stock text) − mean(fork text) = +9.21 ms
- C (image path cost)    = mean(fork image) − mean(stock image) = −7.34 ms
- p* (point estimate)    = C / (G + C) = **−3.91** (outside `[0, 1]`)
- Bootstrap 95 % CI on p*: **[−12.39, +15.44]** — **statistically
  unidentifiable**.

The prior conclusion described R6.4 as
`STRICTLY_DOMINANT_ON_R6.2/R6.3a_OPERATING_POINT`. **That
`STRICTLY_DOMINANT` framing is retracted here.** The −3.91 estimate
is a point number from noisy rebaseline inputs; the bootstrap
interval spans 27+ units of `p` and includes 0 as well as values
outside `[0, 1]` in every direction, so the analytical p*
framework does not identify a crossover — neither in either
direction, nor an inability to exist. `R6.4` stands as **AMBIGUOUS**.

## Empirical mixed-workload validation (`R6.5 = INVALID_MIXED_PROVENANCE / AMBIGUOUS`)

The R6.5 empirical mix-ratio sweep was attempted several times on
2026-07-29 UTC. All attempts are preserved.

### Provenance defect in the auto-generated "PASS" verdict

An earlier version of this document reported R6.5 attempt_gpu2 as
`PASS` on the basis of the auto-generated
[`R6.5_empirical_mixed/attempt_gpu2/verdict.json`](R6.5_empirical_mixed/attempt_gpu2/verdict.json).
Post-hoc audit of the raw provenance rejects that conclusion:

| item | `prelaunch_utc` / `started_utc` |
|---|---|
| attempt_gpu2 `raw/launch_context.json` `prelaunch_utc` | `2026-07-29T12:27:18Z` |
| `ratio_0p2/stock_default/summary.json started_utc` | `12:27:53Z` (after launch — OK) |
| `ratio_0p2/fork_pcg/summary.json started_utc` | `12:30:20Z` (after launch — OK) |
| `ratio_0p5/stock_default/summary.json started_utc` | **`12:17:32Z`** (10 min **before** launch) |
| `ratio_0p5/fork_pcg/summary.json started_utc` | **`12:19:27Z`** (before launch) |
| `ratio_0p8/stock_default/summary.json started_utc` | **`12:21:09Z`** (before launch) |
| `ratio_0p8/fork_pcg/summary.json started_utc` | **`12:23:54Z`** (before launch) |

Only `ratio_0p2` is a true attempt_gpu2 measurement. `ratio_0p5`
and `ratio_0p8` are stale artifacts from an earlier attempt (the
attempt_gpu4 run that took place ~12:17–12:25) whose per-ratio
`summary.json` files persisted into the attempt_gpu2 output
directory and were then read by `R6_5_verdict.py` alongside the
new attempt's data. **The verdict script did not enforce run IDs
or timestamp checks against the launch context.** The 3-of-3
"agreement" that produced the machine PASS therefore combined
different runs.

Even the two stale ratios are also of dubious provenance because
the corresponding attempt_gpu4 run was concurrently affected by
the `launch_server` `/get_model_info` ready-race that produced 93 %
and 94 % request-failure rates in other ratios of the same run
(see
[`R6.5_empirical_mixed/attempt_gpu4/analysis.md`](R6.5_empirical_mixed/attempt_gpu4/analysis.md)).

### R6.5 GPU 4 status

The GPU 4 attempt (attempt_gpu4) machine verdict was
**AMBIGUOUS**. Only one ratio (`ratio_0p5`) had both variants
complete cleanly:

- stock 0.6583 s vs fork 0.6613 s — ratio 1.005 (statistically
  tied, does **not** falsify R6.4's prediction, and does **not**
  confirm mixed-workload dominance either).

`ratio_0p2/fork_pcg` (7/100 completed) and
`ratio_0p8/stock_default` (6/100 completed) failed with
client-side ready-race errors — not server bugs; also not
sufficient to compute matched comparisons.

### R6.5 audited overall verdict

- `INVALID_MIXED_PROVENANCE` — the attempt_gpu2 machine PASS
  combined stale ratios from a prior run;
- `AMBIGUOUS` — only one clean matched ratio survives across
  every 2026-07-29 R6.5 attempt, and that ratio is statistically
  tied;
- **R6.5 does NOT validate mixed-workload dominance.**

A clean isolated rerun (per-ratio pre-launch idle check +
launch-context enforcement in the verdict script + a
model-serving-ready readiness check rather than `/get_model_info`)
would be required only if we want to retain a
mixed-workload-dominance claim. This is deferred; it is not
performed in this merge-preparation step.

## Overall corrected R6 verdict

- **Correctness + safety** ✅ Fix eliminates the historical
  capture-stream assertion crash on the workload that reliably
  reproduces it on stock; correctness envelope preserved. Mixed
  safety subtest 0/0/0/0.
- **Text non-regression** ✅ Under Amendment C, fork does not
  regress text-only PCG. Do not overclaim a proven speedup.
- **Image-path cost** — Not proven to exist at the R6.3a
  operating point (mean-of-means ratio 0.92 but CVs 15–22 %).
  Long-text cells (`t2048_*`) show fork ≥ stock.
- **Confirmed workload win with acceptable noise** — Only
  `cell_t512_r360p_c1` (~8.6 %, CV ~4 %) meets a reasonable
  statistical bar. Bigger-margin cells (~22 %) had CVs ~27–33 %
  and are exploratory.
- **Analytical crossover** — AMBIGUOUS; p* unidentifiable at 95 %
  bootstrap.
- **Empirical mixed-workload result** — INVALID_MIXED_PROVENANCE /
  AMBIGUOUS; no clean 3-ratio matched comparison exists.

**Final overall reading**: `CORRECTNESS_AND_SAFETY_FIX_PASS` +
`PERFORMANCE_VALUE_PROMISING_BUT_NOT_FINAL`. Overall performance
dominance is **not established**.

## Runtime provenance correction

Earlier statements listed:

- `flashinfer 0.6.8.post1` — **incorrect**. Actual installed
  version at the time of the R6 runs is **`flashinfer 0.6.12`**.
- `sgl_kernel` was not called out — actual installed version is
  **`sgl_kernel 0.4.4`**.

Everything else (torch 2.11.0+cu130, LD_PRELOAD-pinned host
libcuda 595.71.05, Qwen3-VL-8B snapshot 0c351dd, stock / fork
SHAs) stands.

## Current upstream status (2026-07-19)

SGLang upstream PR **#30868** was merged on 2026-07-19 and
addresses the same root cause with:

- multimodal deepstack warmup,
- dynamic-shape stabilization,
- a missing-capture-stream eager fallback.

Our fork's 7-commit `fix/pcg-vlm-deepstack-warmup` stack is
**likely superseded** for current upstream. We should NOT upstream
the old fork branch unchanged. If we want to file anything against
current SGLang, we should:

1. Re-verify the exact repeated-shape crash on current upstream
   (post-#30868);
2. If the assertion no longer reproduces, close this investigation
   as "already fixed upstream" and archive R6 as historical
   provenance;
3. If some residual case still reproduces, file a much smaller PR
   scoped to that residual case, not the whole branch.

Verification against current upstream is deferred to a separate
task; it is explicitly **not** performed as part of this
documentation correction.

## Preserved failure records (informative, unchanged)

- `R6.3_image_and_sweep/attempt_gpu6/` — INFRA_INCOMPLETE (runner
  bugs since fixed in `dd93c43`)
- `R6.3_image_and_sweep/attempt_gpu2_partial_orphaned_20260729T094128Z/`
  — session-teardown killed the runner mid-rep-2; **not tracked**
  (local scratch only, preserved untouched in the working tree)
- `R6.5_empirical_mixed/attempt_gpu2_partial_oom_20260729T120743Z/`,
  `_partial_foreign_20260729T120917Z/`,
  `_partial_tangled_20260729T121208Z/` — three consecutive
  foreign-tenant OOM / contention failures on GPU 2
- `R6.5_empirical_mixed/attempt_gpu4/` — client-side ready-race
  (`/get_model_info` returns 200 before scheduler is ready), only
  ratio_0p5 clean

All retained on-branch as informative failure records. Historical
machine-generated `.json` / `.md` verdicts are preserved verbatim;
this document supersedes their high-level framing.

## Change log for this correction

- Removed unqualified `R6 = PASS` framing.
- Removed / retracted `R6.4 STRICTLY_DOMINANT_ON_...` language.
- Removed `R6.5 = PASS` framing; replaced with
  `INVALID_MIXED_PROVENANCE / AMBIGUOUS` with the launch-context
  timestamp evidence spelled out.
- Deranked the R6.3 22 %-margin cell (`cell_t512_r360p_c4`) from
  headline to exploratory; promoted `cell_t512_r360p_c1` (~8.6 %,
  low CV) as the cleanest promising cell.
- Corrected runtime versions (flashinfer `0.6.12`, sgl_kernel
  `0.4.4`).
- Added SGLang PR #30868 upstream-status paragraph.
- Preserved every original machine verdict `.json` / `.md` file
  and every commit on this branch. Nothing under
  `results/R6_fix_value_validation/` was deleted.
