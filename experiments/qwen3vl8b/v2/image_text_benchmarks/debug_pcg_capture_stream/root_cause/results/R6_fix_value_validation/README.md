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
| **R6.1 Amended runner + verdict** — implements Amendment A: 5 matched cold-cache pairs (10 fresh servers) + neg control + mixed-safety leg (12 total servers); phase markers; token-level envelope-based verdict | ✅ COMPLETE 2026-07-28 (CPU-only, no GPU workload) | `scripts/run_R6_1_amended.sh`, `scripts/R6_1_verdict_amended.py` |
| **R6.1b attempt 03** — full execution under Amendment A on GPU 0 (2026-07-28T13:34–13:49 UTC) | ❌ overall **FAIL / NONE** — but the failure category has flipped from a fix defect to an **upstream reframing**. **Tier 1 SAFETY_SUPERIORITY = FAIL** because negative control classified as **`STOCK_NOW_SURVIVES`**: stock-PCG served all 3 image requests HTTP 200 on the current frozen `da802ddca`, so the historical first-image failure does not currently reproduce (server log shows the same `input_deepstack_embeds is None` recompile at [0/1]–[0/4] as before, but no capture-stream assertion). Upstream must have addressed the underlying missing-capture-stream issue between the historical failing HEAD `62c505a196` and current `da802ddca`. **Tier 2 CORRECTNESS = PASS**: fork-PCG interleaved leg 0 assertions / 0 fallbacks / 0 request failures / 0 post-server-ready recompiles; all 3 cross-config cross-comparisons (`stock_default_vs_fork_default__image_cold`, `stock_pcg_vs_fork_pcg__text_cold`, `fork_default_vs_fork_pcg__image_cold`) inside the union envelope; fork_pcg vs fork_default on image is **bit-identical on every prompt**. Equivalence-class observation on image prompt 2: `{stock_default_A, fork_default_A, fork_pcg_A}` are all one class; **stock-PCG neg-control (`neg_stock_pcg`) is a separate class** — the fork's proactive-warmup approach produces outputs bit-identical to eager, while stock-PCG's post-hoc-recompile approach diverges slightly on that prompt. Same-config non-determinism envelope exposed one hot cell: `fork_pcg_text` prompt 1 tok_lev=42 (trailing whitespace + minor phrasing between A and B). R6.2 – R6.5 remain blocked per Amendment A §2.4 (overall PASS required). See [`R6.1_correctness/attempt_03_amended_A_gpu0/verdict_amended.md`](R6.1_correctness/attempt_03_amended_A_gpu0/verdict_amended.md). | verdict + JSON committed |
| **R6.1b attempt 01** (historical) | ⚠️ **R7_REQUIRED / INFRA_FAILURE** on 2026-07-28T10:46 UTC (GPU 1, monitor-selected after 629 s continuous idle). Runner exited 2 on the first server before any leg ran. **Original explanation was incomplete**: framed as "NVIDIA driver silently upgraded during the wait." **Corrected root cause** (see R6.0 Amendment A3): `cuda-compat-13-0`'s `/usr/local/cuda-13.0/compat/libcuda.so.580.82.07` took loader precedence over the host driver `libcuda.so.595.71.05`; torch 2.11.0+cu130 fails `cudaGetDeviceCount()` against the compat lib with Error 803. Infrastructure blocker only — **not** a clean-Y correctness failure. Historical artifacts preserved: [`R6.1_correctness/verdict.md`](R6.1_correctness/verdict.md), [`R6.1_correctness/verdict.json`](R6.1_correctness/verdict.json), original `R6.1_correctness/raw/`. | attempt-01 artifacts retained under `R6.1_correctness/` root |
| **R6.1b attempt 02** — GPU 0, host libcuda pinned | ❌ **FAIL** on 2026-07-28T12:43 UTC (executed on GPU 0 after user switched from GPU 2 mid-flight; new attempt directory `attempt_02_host_libcuda_595_gpu0/`). Full runner ran; **all 4 servers launched and served all 9 legs successfully with HTTP 200 across every request** (no crash, no assertion, no fallback, no request failure). Verdict.py applies the pre-declared rules → **FAIL** on three axes: (a) fork-default same-run repeat NOT bit-identical (idx=2 first_diff_offset=241; the difference is a paraphrase of the same semantic content — "single, uniform solid color with no patterns" vs "solid, uniform color without any patterns"), (d) stock-PCG text vs fork-PCG text NOT bit-identical (idx=1 first_diff_offset=75 — trailing whitespace + minor phrasing), (e) `inference_recompiles=4` (pre-declared strict rule). **Post-hoc characterization** (supplementary; does not change the machine verdict): (i) leg (a)'s same-server non-determinism proves the protocol's bit-identical assumption is unfounded on this stack with default radix caching — leg (b) and leg (d) divergences at matching offsets (241, 75) are almost certainly the same non-determinism, not fix-induced corruption; (ii) leg (c) PASSED bit-identical (stock-default == fork-default) — the fix is a genuine no-op when PCG is off; (iii) the 4 recompiles are all pre-server-ready warmup (lines 30, 53, 158, 193 vs server ready at line 570) — 0 inference-time recompiles by phase-split evidence, so the substance of leg (e) is clean; (iv) mixed-safety interleaved sequence text→image→text→image→text on the fork-PCG server: all 5 requests HTTP 200, no crashes. **Blocks R6.2 per gating rule**; R7 refinement required to (1) address protocol determinism (e.g. `--disable-radix-cache` in variant flags or an explicit same-server bit-identical baseline for text), (2) sharpen the `inference_recompiles` metric to count only post-server-ready events (per protocol §6 note). Preserved: [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.md`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.md), [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.json`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.json). | See verdict artifacts. Note the failed intermediate `attempt_02_host_libcuda_595_gpu2/` scaffold — preflight bug — its raw/preflight.log stays gitignored as historical evidence of the CUDA_VISIBLE_DEVICES preflight bug fixed in commit `52b0c84`. |
| **R6.2** — Text-only Case A matched control (4 variants × 5 reps) | ⏳ NOT STARTED | — |
| **R6.3** — Fresh image cost + workload sweep + mixed-safety subtest | ⏳ NOT STARTED | — |
| **R6.4** — Analytical crossover (means, bootstrap CI) | ⏳ NOT STARTED | — |
| **R6.5** — Optional empirical mixed validation | ⏳ NOT STARTED | — |

**R6 overall verdict:** ❌ **FAIL at R6.1 (attempt 03) — but the failure category has flipped**. Correctness tier PASSED; safety-superiority tier FAILED because the **historical stock-PCG first-image capture-stream assertion does not reproduce on the current stock HEAD `da802ddca`** (`STOCK_NOW_SURVIVES` classification). Fork's clean-Y appears **functionally redundant** on the current upstream state — a fix commit landed upstream between the historical failing HEAD `62c505a196` and current `da802ddca` that addresses the underlying issue by a different mechanism. Fork's approach still has a correctness edge on image prompt 2 (fork-PCG output == eager output bit-identical; stock-PCG output diverges slightly). Value-claim reframe required before proceeding — R6.2 – R6.5 remain blocked per Amendment A §2.4 (overall PASS required for performance claims). Recommended R7 investigation: bisect upstream between `62c505a196` and `da802ddca` to locate the commit that changed stock behaviour; decide whether fork PR is redundant, complementary (defense-in-depth), or subsumed. See [`R6.1_correctness/attempt_03_amended_A_gpu0/verdict_amended.md`](R6.1_correctness/attempt_03_amended_A_gpu0/verdict_amended.md).

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
