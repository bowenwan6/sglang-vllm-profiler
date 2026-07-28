# R6 — Fix-value validation for mixed-modality PCG

Formal validation of clean-Y for upstream PR. See
[`plan.md` §5b](../../../../../../../../../plan.md) for the full
protocol; this directory holds the recorded artifacts.

## Current status

| Phase | Status | Artifact |
|---|---|---|
| **R6.0** — Provenance freeze | ✅ COMPLETE | [`R6.0_provenance.md`](R6.0_provenance.md) |
| **R6.1a** — Correctness protocol + runner + fixture (CPU-only, no GPU workload) | ✅ COMPLETE | [`R6.1_correctness/protocol.md`](R6.1_correctness/protocol.md), [`R6.1_correctness/fixtures/`](R6.1_correctness/fixtures/), `scripts/{run_R6_1_correctness.sh, R6_1_client.py, R6_1_verdict.py}` |
| **R6.1b** — Correctness gate execution + verdict | 🔒 BLOCKED on explicit user approval of a GPU ID | (executes runner; produces `R6.1_correctness/{verdict.md, verdict.json}`) |
| **R6.2** — Text-only Case A matched control (4 variants × 5 reps) | ⏳ NOT STARTED | — |
| **R6.3** — Fresh image cost + workload sweep + mixed-safety subtest | ⏳ NOT STARTED | — |
| **R6.4** — Analytical crossover (means, bootstrap CI) | ⏳ NOT STARTED | — |
| **R6.5** — Optional empirical mixed validation | ⏳ NOT STARTED | — |

**R6 overall verdict:** ⏳ pending — awaiting R6.1 through R6.5.

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
