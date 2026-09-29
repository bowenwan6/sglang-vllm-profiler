# Methodology Correction — KAPI Instrumentation Confound (2026-05-26)

This document corrects the **strength of conclusions** in earlier summaries. It does **not** delete or
alter any raw evidence: all `raw/*.json`, trace files, and logs are retained as provenance. Only the
*interpretation* of certain measurements is downgraded.

## What was wrong

Early SGLang benchmarks were collected with **`SGLANG_KERNEL_API_LOGLEVEL=1` (KAPI logging) enabled on
the SGLang side only**, while vLLM had **no equivalent instrumentation**. Phase 5 Case A later showed
that KAPI logging materially inflates SGLang TTFT (it logs every kernel-API call, penalizing the
eager/direct-launch path). Therefore any SGLang-vs-vLLM latency comparison drawn from KAPI-instrumented
SGLang runs is **instrumentation-confounded** and cannot stand as a clean cross-framework conclusion.

Affected (instrumented) runs:
- **Phase 1 baseline** — `run_phase1.py` set `SGLANG_KERNEL_API_LOGLEVEL=1`.
- **Phase 2 Case C W500** — `run_phase2_caseC_w500.py` set `SGLANG_KERNEL_API_LOGLEVEL=1`.

(The clean Phase-5 confirmation/rerun runners explicitly remove these env vars.)

## Correction table

| Topic | Correction |
|---|---|
| Affected measurements | Phase 1 SGLang baseline; Phase 2 Case C W500 (SGLang side) |
| Confound | SGLang-only KAPI logging; vLLM had no corresponding instrumentation |
| Superseded claim | Case C "stable **1.32×** SGLang-slower TTFT gap" (249.1 ms vs 189.0 ms) |
| Clean replacement | Case C clean interleaved rerun (no KAPI, no profiler): pooled S0 **~192.2 ms**, pooled S2 **~193.6 ms**, vLLM **~189.8 ms** → **no material median TTFT gap and no Case-A-like S2 benefit** observed; S0/vLLM show ~17% session variance, so small effects are unresolved (not strict parity, not a proven gap) |
| Surviving validated finding | **Case A clean S2 intervention** (`--enforce-piecewise-cuda-graph`) materially reduces TTFT (~19.2 → ~11.7 ms) with **TPOT unchanged**, 0 failures; reaches the vLLM TTFT range (stable superiority NOT claimed, S2 CV ~10–12%); `--enforce-piecewise-cuda-graph` is a **testing lever, not a production fix** |
| Phase 4 traces | Still valid for **structure**: both frameworks' GPU time is dominated by the same FP8 GEMM family (shared absolute cost, not a proven gap source); dispatch/graph/compile mechanisms differ structurally. Case C traces must **not** be cited as explaining the (now-superseded) 1.32× gap |
| Remaining validation need | Clean (no-KAPI, no-profiler) cross-framework baseline for **Case B and Case D** before any four-workload TTFT-ratio headline; the early 4.89× / 3.20× / 1.32× / 1.33× ratios are **exploratory, instrumentation-confounded**, not clean final results |

## Status of headline claims after correction

- ❌ "SGLang TTFT is 4.89× / 3.20× / 1.32× / 1.33× slower across four workloads" → **exploratory /
  instrumentation-confounded discovery signal**, not a clean conclusion.
- ❌ "Case C has a stable 1.32× batched gap" → **superseded**; clean rerun shows no material median gap.
- ✅ "Forcing prefill piecewise CUDA-graph coverage materially reduces Case A (c=1) TTFT, TPOT
  unchanged" → **clean validated finding** (testing-lever; production scope open).
- ⚠️ "H1 generalizes / does-not-generalize to Case C" → soften to **"no Case-A-like TTFT benefit
  observed for Case C under current clean interleaved test; smaller effects unresolved under observed
  ~17% session variance."**
- ⏳ Case B / Case D: **no clean cross-framework baseline yet** → excluded from any formal headline.

## What is preserved

Raw benchmark JSON, trace files, kernel-API logs (where retained), and all per-phase scripts are
unchanged — they document exactly what was run, including the instrumented conditions. This correction
changes only the **evidence level** attached to the conclusions, not the historical record.
