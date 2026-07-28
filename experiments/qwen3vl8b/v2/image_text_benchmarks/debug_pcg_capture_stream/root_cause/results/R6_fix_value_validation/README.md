# R6 — Fix-value validation for mixed-modality PCG

Formal validation of clean-Y for upstream PR. See
[`plan.md` §5b](../../../../../../../../../plan.md) for the full
protocol; this directory holds the recorded artifacts.

## Current status

| Phase | Status | Artifact |
|---|---|---|
| **R6.0** — Provenance freeze + amendments A1 / A2 (2026-07-28: dynamic GPU selection via monitor; PGID-scoped cleanup only) | ✅ COMPLETE | [`R6.0_provenance.md`](R6.0_provenance.md) |
| **R6.1a** — Correctness protocol + runner + fixture (CPU-only, no GPU workload); safety-scoped teardown + idle-GPU monitor | ✅ COMPLETE | [`R6.1_correctness/protocol.md`](R6.1_correctness/protocol.md), [`R6.1_correctness/fixtures/`](R6.1_correctness/fixtures/), `scripts/{run_R6_1_correctness.sh, R6_1_client.py, R6_1_verdict.py, R6_setsid_exec.py, monitor_idle_gpu.py}` |
| **R6.1b** — Correctness gate execution + verdict (GPU is auto-selected by `monitor_idle_gpu.py` after 600 s continuous idle) | ⚠️ **R7_REQUIRED / INFRA_FAILURE** on 2026-07-28T10:46 UTC — NVIDIA driver silently upgraded during the 3.4 h monitor wait (R6.0: `570.172.08` → runner-time: `595.71.05`); torch 2.11.0+cu130 fails `cudaGetDeviceCount()` with Error 803; stock-default server exited before HTTP readiness; no leg executed. Runner exited 2. No process outside our launch was signalled. See [`R6.1_correctness/verdict.md`](R6.1_correctness/verdict.md). Retry requires (a) resolving the driver/torch ABI mismatch and (b) a `docs(v2): update R6 provenance` commit — not attempted here. | [`R6.1_correctness/verdict.md`](R6.1_correctness/verdict.md), [`R6.1_correctness/verdict.json`](R6.1_correctness/verdict.json) |
| **R6.2** — Text-only Case A matched control (4 variants × 5 reps) | ⏳ NOT STARTED | — |
| **R6.3** — Fresh image cost + workload sweep + mixed-safety subtest | ⏳ NOT STARTED | — |
| **R6.4** — Analytical crossover (means, bootstrap CI) | ⏳ NOT STARTED | — |
| **R6.5** — Optional empirical mixed validation | ⏳ NOT STARTED | — |

**R6 overall verdict:** ⚠️ **R7_REQUIRED (INFRA_FAILURE at R6.1b, driver drift)** — see R6.1b row and [`R6.1_correctness/verdict.md`](R6.1_correctness/verdict.md). Downstream phases R6.2 – R6.5 remain blocked until R6.1b is retried under an amended R6.0 provenance tuple.

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
