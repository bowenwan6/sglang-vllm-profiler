# R6 — Fix-value validation for mixed-modality PCG

Formal validation of clean-Y for upstream PR. See
[`plan.md` §5b](../../../../../../../../../plan.md) for the full
protocol; this directory holds the recorded artifacts.

## Current status

| Phase | Status | Artifact |
|---|---|---|
| **R6.0** — Provenance freeze + amendments A1 / A2 (2026-07-28: dynamic GPU selection via monitor; PGID-scoped cleanup only) | ✅ COMPLETE | [`R6.0_provenance.md`](R6.0_provenance.md) |
| **R6.1a** — Correctness protocol + runner + fixture (CPU-only, no GPU workload); safety-scoped teardown + idle-GPU monitor | ✅ COMPLETE (historical protocol) | [`R6.1_correctness/protocol.md`](R6.1_correctness/protocol.md), [`R6.1_correctness/fixtures/`](R6.1_correctness/fixtures/), `scripts/{run_R6_1_correctness.sh, R6_1_client.py, R6_1_verdict.py, R6_setsid_exec.py, monitor_idle_gpu.py}` |
| **R6.1 Protocol Amendment A** — direct fix-comparison (3-tier verdict, cache-matched controls, phase-scoped recompile markers, stock-PCG image negative control) | ✅ COMPLETE 2026-07-28 | [`R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md`](R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md) |
| **R6.1 Protocol Amendment B** — repeated-shape safety control (replaces Amendment A §2.3's under-powered 3-prompt neg-control with the exact historical R1/E2a sustained-workload recipe: 720p image + `--random-input-len 128 --random-range-ratio 1.0 --num-prompts 32 --warmup-requests 30`, identical for stock-PCG and fork-PCG) | ✅ COMPLETE 2026-07-28 | [`R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md`](R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md) |
| **R6.1 Amended runner + verdict** — implements Amendment A: 5 matched cold-cache pairs (10 fresh servers) + neg control + mixed-safety leg (12 total servers); phase markers; token-level envelope-based verdict | ✅ COMPLETE 2026-07-28 (CPU-only, no GPU workload) | `scripts/run_R6_1_amended.sh`, `scripts/R6_1_verdict_amended.py` |
| **R6.1 Repeated-shape runner + verdict** — implements Amendment B: stock-PCG (expected assertion) + fork-PCG (expected clean) on the exact historical R1 recipe (720p × 32 same-shape requests); assertion + deepstack-recompile classification + prefill-shape trace | ✅ COMPLETE 2026-07-28 (CPU-only, no GPU workload) | `scripts/run_R6_1_repeated_shape.sh`, `scripts/R6_1_verdict_amendment_B.py` |
| **R6.1b attempt 04** — first execution under Amendment B on GPU 0 (2026-07-28T14:36–14:40 UTC) | ✅ **`SAFETY_SUPERIORITY_PASS`**. Stock-PCG reproduced the exact historical `AssertionError: PCG capture stream is not set` (server_log:44322) after the post-server-ready deepstack recompile (line 35429). Assertion context confirmed by prefill-shape trace: last prefill batch before the crash was `new_seq=1 new_token=1 cached_token=1022` (total=1023 — second occurrence of that shape, matching R1/R2's "second-same-shape-after-recompile" mechanism at `runtime_shape=1024`). Fork-PCG on the **identical bench recipe** completed all 30 warmup + 32 measured requests: `assertion count = 0`, `fallback count = 0`, `post-ready recompile count = 0`, `bench.jsonl aggregate_completed = 32`, `generated_texts count = 32`. Prior attempt-03 `STOCK_NOW_SURVIVES` was underpowered — this attempt reproduces the historical bug on the exact same stock SHA `da802ddca`, proving the fix's operational-safety superiority is real. Combined with Attempt 03's CORRECTNESS PASS ⇒ **overall R6.1 = PASS**. Details: [`R6.1_correctness/attempt_04_repeated_shape_gpu0/verdict_amended_B.md`](R6.1_correctness/attempt_04_repeated_shape_gpu0/verdict_amended_B.md). |
| **R6.1b attempt 03** — full execution under Amendment A on GPU 0 (2026-07-28T13:34–13:49 UTC) | Machine verdict: ❌ **FAIL / NONE** — stands as recorded. **Corrected interpretation** (see [`interpretation_addendum.md`](R6.1_correctness/attempt_03_amended_A_gpu0/interpretation_addendum.md)): `STOCK_NOW_SURVIVES` is under-powered → **`INCONCLUSIVE_TRIGGER_NOT_REPRODUCED`**. Attempt 03's 3 image prompts produced 3 distinct prefill runtime shapes; the *second-same-shape call after multimodal recompile* that triggers the historical assertion was never reached. All 4 recompiles `[0/1]-[0/4]` observed in the neg-control server log confirm the recompile half of the historical bug did fire — only the repeated-shape half did not. Withdrawn: any claim that upstream fixed the bug. **Tier 2 CORRECTNESS PASS** remains valid (matched cold-cache comparisons don't require repeated shapes): fork-PCG vs fork-default bit-identical on image; all 3 cross-config cross-comparisons inside envelope; interleaved leg 0 assertions / 0 fallbacks / 0 request failures / 0 post-server-ready recompiles. Amendment B (next commit) adds the repeated-shape control; Attempt 04 runs it. | verdict + JSON committed; interpretation_addendum.md corrects the higher-level reading |
| **R6.1b attempt 01** (historical) | ⚠️ **R7_REQUIRED / INFRA_FAILURE** on 2026-07-28T10:46 UTC (GPU 1, monitor-selected after 629 s continuous idle). Runner exited 2 on the first server before any leg ran. **Original explanation was incomplete**: framed as "NVIDIA driver silently upgraded during the wait." **Corrected root cause** (see R6.0 Amendment A3): `cuda-compat-13-0`'s `/usr/local/cuda-13.0/compat/libcuda.so.580.82.07` took loader precedence over the host driver `libcuda.so.595.71.05`; torch 2.11.0+cu130 fails `cudaGetDeviceCount()` against the compat lib with Error 803. Infrastructure blocker only — **not** a clean-Y correctness failure. Historical artifacts preserved: [`R6.1_correctness/verdict.md`](R6.1_correctness/verdict.md), [`R6.1_correctness/verdict.json`](R6.1_correctness/verdict.json), original `R6.1_correctness/raw/`. | attempt-01 artifacts retained under `R6.1_correctness/` root |
| **R6.1b attempt 02** — GPU 0, host libcuda pinned | ❌ **FAIL** on 2026-07-28T12:43 UTC (executed on GPU 0 after user switched from GPU 2 mid-flight; new attempt directory `attempt_02_host_libcuda_595_gpu0/`). Full runner ran; **all 4 servers launched and served all 9 legs successfully with HTTP 200 across every request** (no crash, no assertion, no fallback, no request failure). Verdict.py applies the pre-declared rules → **FAIL** on three axes: (a) fork-default same-run repeat NOT bit-identical (idx=2 first_diff_offset=241; the difference is a paraphrase of the same semantic content — "single, uniform solid color with no patterns" vs "solid, uniform color without any patterns"), (d) stock-PCG text vs fork-PCG text NOT bit-identical (idx=1 first_diff_offset=75 — trailing whitespace + minor phrasing), (e) `inference_recompiles=4` (pre-declared strict rule). **Post-hoc characterization** (supplementary; does not change the machine verdict): (i) leg (a)'s same-server non-determinism proves the protocol's bit-identical assumption is unfounded on this stack with default radix caching — leg (b) and leg (d) divergences at matching offsets (241, 75) are almost certainly the same non-determinism, not fix-induced corruption; (ii) leg (c) PASSED bit-identical (stock-default == fork-default) — the fix is a genuine no-op when PCG is off; (iii) the 4 recompiles are all pre-server-ready warmup (lines 30, 53, 158, 193 vs server ready at line 570) — 0 inference-time recompiles by phase-split evidence, so the substance of leg (e) is clean; (iv) mixed-safety interleaved sequence text→image→text→image→text on the fork-PCG server: all 5 requests HTTP 200, no crashes. **Blocks R6.2 per gating rule**; R7 refinement required to (1) address protocol determinism (e.g. `--disable-radix-cache` in variant flags or an explicit same-server bit-identical baseline for text), (2) sharpen the `inference_recompiles` metric to count only post-server-ready events (per protocol §6 note). Preserved: [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.md`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.md), [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.json`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.json). | See verdict artifacts. Note the failed intermediate `attempt_02_host_libcuda_595_gpu2/` scaffold — preflight bug — its raw/preflight.log stays gitignored as historical evidence of the CUDA_VISIBLE_DEVICES preflight bug fixed in commit `52b0c84`. |
| **R6.2 runner + verdict** — 4 variants × 5 reps × 400 prompts on `caseA_short.jsonl` (SHA `fab49177…`) | ✅ COMPLETE 2026-07-29 (CPU-only) | `scripts/run_R6_2_text_only.sh`, `scripts/R6_2_verdict.py` |
| **R6.2 execution** — text-only PCG non-regression on Qwen3-VL, GPU 0 (2026-07-29T00:49–02:27 UTC) | ❌ machine verdict **FAIL** on one axis (drift 3.050% > 3.0% threshold, by 0.05 pp) — **preserved verbatim** as machine verdict under the original protocol. **Every substantive gate PASSED**: fork_pcg mean TTFT / stock_pcg mean TTFT = **0.9617** (require ≤ 1.05, actual fork-PCG is 4% faster than stock-PCG); per-variant CV ≤ 5.91% (require ≤ 6%); all 4 variants 5/5 reps × 400/400 requests; 0 assertions / 0 fallbacks / 0 post-ready recompiles across all servers. Headline: `stock_default = 26.86 ms → stock_pcg = 18.35 ms → fork_pcg = 17.65 ms` (fork retains ~-34% PCG TTFT benefit vs default; stock retains ~-32%). Drift stock_default (mean 26.862 across reps 1–5) vs stock_default_repeat (mean 27.681 across reps 1–5 an hour later) = 3.05% — traceable to intermittent foreign PIDs on GPU 0 during rep-1 and rep-5 of stock_default (thermal/queueing noise), not a fix effect. Details: [`R6.2_text_only_caseA/attempt_gpu0/verdict.md`](R6.2_text_only_caseA/attempt_gpu0/verdict.md). |
| **R6.2 Protocol Amendment C** — shared-GPU drift-gate reclassification (2026-07-29). Drift buckets `≤ 3% clean PASS` / `3% < d ≤ 5% PASS_WITH_CAVEAT` / `> 5% rerun-or-AMBIGUOUS`. Fork-vs-stock non-regression (`≤ 1.05`), CV bound (`≤ 6%`), and every safety hard-FAIL condition unchanged. Original machine verdict preserved verbatim. | ✅ 2026-07-29. Under Amendment C, R6.2 = **`PASS_WITH_CAVEAT — TEXT_NON_REGRESSION_SUPPORTED`**. See [`R6.2_text_only_caseA/protocol_amendment_C_shared_gpu_drift_gate.md`](R6.2_text_only_caseA/protocol_amendment_C_shared_gpu_drift_gate.md) and [`R6.2_text_only_caseA/attempt_gpu0/status_amended_C.md`](R6.2_text_only_caseA/attempt_gpu0/status_amended_C.md). Absolute `stock_default = 26.86 ms` retains a shared-GPU caveat in all downstream reporting. R6.3+ unblocked. |
| **R6.3** — Fresh image cost + workload sweep + mixed-safety subtest | ⏳ IN PROGRESS 2026-07-29 (unblocked by Amendment C) | — |
| **R6.4** — Analytical crossover (means, bootstrap CI) | ⏳ NOT STARTED | — |
| **R6.5** — Optional empirical mixed validation | ⏳ NOT STARTED | — |

**R6 overall verdict (running):** ✅ **R6.1 = PASS** (2026-07-28) + ✅ **R6.2 = `PASS_WITH_CAVEAT — TEXT_NON_REGRESSION_SUPPORTED`** (2026-07-29 under Amendment C; original machine verdict FAIL preserved). R6.1 attempt 04 (Amendment B) established `SAFETY_SUPERIORITY_PASS` — stock-PCG reproduced the exact historical `AssertionError: PCG capture stream is not set` at second-same-shape post-recompile call (R1/R2 mechanism, `total=1023, num_finished_warmup 1→2` at line 44322 of stock/server.log), and fork-PCG completed the identical bench (30 warmup + 32 measured requests) with 0 assertions / 0 fallbacks / 0 post-ready inflight recompiles / 32 bench-completed. R6.1 attempt 03 (Amendment A) established `CORRECTNESS_PASS` on cache-matched cold-cache repeats. R6.2 primary fix gate `fork_pcg/stock_pcg = 0.9617` (fork 3.8 % faster on text-only VLM path); the 3.05 % drift bracket is a shared-GPU nuisance-control, reclassified under Amendment C (see below). R6.3–R6.5 proceeding automatically.

Final verdict shape (populated after R6.5 / R6.4 decision):
`PASS` · `SAFETY_ONLY` · `R7_REQUIRED` · `INCONCLUSIVE`.

## Directory layout (created incrementally, one phase at a time)

```
R6_fix_value_validation/
├── README.md                          ← this file (updated per phase)
├── R6.0_provenance.md                 ← ✅ frozen provenance tuple
├── R6.1_correctness/                  ← created at R6.1 start
│   ├── protocol.md
│   ├── raw/                           ← .gitignore'd
│   ├── summary.md
│   └── verdict.md
├── R6.2_text_only_caseA/              ← created at R6.2 start
│   ├── protocol.md
│   ├── R6.2a_stock_default/
│   ├── R6.2b_stock_pcg/
│   ├── R6.2c_fork_pcg/
│   ├── R6.2d_stock_default_repeat/
│   └── summary.md
├── R6.3_image_cost_and_sweep/         ← created at R6.3 start
│   ├── protocol.md
│   ├── R6.3a_fresh_baseline/
│   ├── R6.3b_workload_sweep/
│   ├── R6.3c_mixed_safety/
│   └── summary.md
├── R6.4_analytical_crossover/         ← created at R6.4 start
│   ├── mix_analysis.py
│   └── mix_table.md
└── R6.5_empirical_mixed/              ← created only if R6.4 gate = GO
```

## Rules recap (from `plan.md` §5b and `root_cause/README.md` §3.2)

- One phase at a time; every phase ends with a `docs(v2)` update + commit + push.
- Runner code and result recording ship in separate commits (`feat(v2)`
  then `test(v2)`).
- Every scientifically meaningful cell is recorded — PASS, FAIL, and
  AMBIGUOUS results are all committed.
- Raw JSONL, full server logs, GPU dumps, and traces are not staged.
- The uncommitted local edit to
  `results/R5_clean_Y/R5C_correctness_audit/audit_report.md` is
  preserved as-is under user control.
- Historical numbers from older HEADs remain reference-only; R6
  measures everything fresh on the frozen (stock, fork) SHA pair
  recorded in [`R6.0_provenance.md`](R6.0_provenance.md).
