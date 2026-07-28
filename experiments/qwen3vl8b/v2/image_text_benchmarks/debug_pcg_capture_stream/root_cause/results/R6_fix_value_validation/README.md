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
| **R6.1b attempt 01** (historical) | ⚠️ **R7_REQUIRED / INFRA_FAILURE** on 2026-07-28T10:46 UTC (GPU 1, monitor-selected after 629 s continuous idle). Runner exited 2 on the first server before any leg ran. **Original explanation was incomplete**: framed as "NVIDIA driver silently upgraded during the wait." **Corrected root cause** (see R6.0 Amendment A3): `cuda-compat-13-0`'s `/usr/local/cuda-13.0/compat/libcuda.so.580.82.07` took loader precedence over the host driver `libcuda.so.595.71.05`; torch 2.11.0+cu130 fails `cudaGetDeviceCount()` against the compat lib with Error 803. Infrastructure blocker only — **not** a clean-Y correctness failure. Historical artifacts preserved: [`R6.1_correctness/verdict.md`](R6.1_correctness/verdict.md), [`R6.1_correctness/verdict.json`](R6.1_correctness/verdict.json), original `R6.1_correctness/raw/`. | attempt-01 artifacts retained under `R6.1_correctness/` root |
| **R6.1b attempt 02** — GPU 0, host libcuda pinned | ❌ **FAIL** on 2026-07-28T12:43 UTC (executed on GPU 0 after user switched from GPU 2 mid-flight; new attempt directory `attempt_02_host_libcuda_595_gpu0/`). Full runner ran; **all 4 servers launched and served all 9 legs successfully with HTTP 200 across every request** (no crash, no assertion, no fallback, no request failure). Verdict.py applies the pre-declared rules → **FAIL** on three axes: (a) fork-default same-run repeat NOT bit-identical (idx=2 first_diff_offset=241; the difference is a paraphrase of the same semantic content — "single, uniform solid color with no patterns" vs "solid, uniform color without any patterns"), (d) stock-PCG text vs fork-PCG text NOT bit-identical (idx=1 first_diff_offset=75 — trailing whitespace + minor phrasing), (e) `inference_recompiles=4` (pre-declared strict rule). **Post-hoc characterization** (supplementary; does not change the machine verdict): (i) leg (a)'s same-server non-determinism proves the protocol's bit-identical assumption is unfounded on this stack with default radix caching — leg (b) and leg (d) divergences at matching offsets (241, 75) are almost certainly the same non-determinism, not fix-induced corruption; (ii) leg (c) PASSED bit-identical (stock-default == fork-default) — the fix is a genuine no-op when PCG is off; (iii) the 4 recompiles are all pre-server-ready warmup (lines 30, 53, 158, 193 vs server ready at line 570) — 0 inference-time recompiles by phase-split evidence, so the substance of leg (e) is clean; (iv) mixed-safety interleaved sequence text→image→text→image→text on the fork-PCG server: all 5 requests HTTP 200, no crashes. **Blocks R6.2 per gating rule**; R7 refinement required to (1) address protocol determinism (e.g. `--disable-radix-cache` in variant flags or an explicit same-server bit-identical baseline for text), (2) sharpen the `inference_recompiles` metric to count only post-server-ready events (per protocol §6 note). Preserved: [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.md`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.md), [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.json`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/verdict.json). | See verdict artifacts. Note the failed intermediate `attempt_02_host_libcuda_595_gpu2/` scaffold — preflight bug — its raw/preflight.log stays gitignored as historical evidence of the CUDA_VISIBLE_DEVICES preflight bug fixed in commit `52b0c84`. |
| **R6.2** — Text-only Case A matched control (4 variants × 5 reps) | ⏳ NOT STARTED | — |
| **R6.3** — Fresh image cost + workload sweep + mixed-safety subtest | ⏳ NOT STARTED | — |
| **R6.4** — Analytical crossover (means, bootstrap CI) | ⏳ NOT STARTED | — |
| **R6.5** — Optional empirical mixed validation | ⏳ NOT STARTED | — |

**R6 overall verdict:** ❌ **FAIL at R6.1 (attempt 02)** by pre-declared verdict rules; refined via forensic analysis (see [`R6.1_correctness/attempt_02_host_libcuda_595_gpu0/analysis.md`](R6.1_correctness/attempt_02_host_libcuda_595_gpu0/analysis.md)). Substantive safety of the fix is demonstrably clean (0 crashes, 0 assertions, 0 fallbacks, 0 request failures, 0 post-server-ready recompiles across all 4 servers × 9 legs). The FAIL was on the bit-identical axes, but the forensic analysis proves the divergences are cache-state artefacts, not fix-induced corruption: (i) `a1_vs_c` (fork-default cold vs stock-default cold) is bit-identical on every image prompt → fix is a genuine no-op when PCG is off; (ii) same-config warm-vs-cold within a single fork-default server (a1_vs_a2, tok_lev=4 on prompt 2) is *larger* variance than cross-config cold-vs-cold (a1_vs_b, tok_lev=2) — so the cross-config differences fit inside the same cache-state envelope, not a PCG effect. Attempt 02 also **never issued an image request to stock-PCG** so the historical first-image capture-stream failure is neither reproduced nor ruled out on the current frozen stock SHA. **R6 continues under a refined R6.1 protocol** (formalised in the next commit) that adds: cache-matched repeats, phase-scoped recompile markers, a direct stock-PCG image negative control, and three-tier evidence claims (SAFETY_SUPERIORITY / TEXT_NON_REGRESSION / WORKLOAD_PERFORMANCE_WIN).

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
