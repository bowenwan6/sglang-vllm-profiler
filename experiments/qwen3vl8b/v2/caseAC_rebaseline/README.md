# v2 / Round 2 — Case A/C rebaseline (issue #2)

Production-default **overlap-ON** rebaseline for Qwen3-VL. Demotes v1's `--disable-overlap-schedule`
headline to an optional ablation and re-tests whether the PCG lever still helps against the production
baseline.

- **Issue:** #2 (parent #1) on `bowenwan6/sglang-vllm-profiler`.
- **Status:** **COMPLETE — Case A + C run on GPU 1, 0 failures, acceptance PASS.**
- **Results:** [`results/summary.md`](results/summary.md) (combined), [`results/caseA_summary.md`](results/caseA_summary.md),
  [`results/caseC_summary.md`](results/caseC_summary.md). Aggregates in `results/case{A,C}_results.json`.
- **Protocol:** [`protocol.md`](protocol.md) — decision-complete 12-section plan (goal, questions,
  variants, run design, metrics, acceptance, stop conditions, command templates).
- **Model:** `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` (same as v1; verify in env snapshot before runs).

**Headline:** PCG still helps on the production-default overlap-ON baseline — Case A TTFT 21.9 → 14.0 ms
(into vLLM's 13.1 ms band, TPOT flat). Case C (c=16 batched) shows **no Case-A-like benefit** (boundary
confirmed). The fix is workload-shape-dependent → selective enablement (Issue #5).

Key design points (see protocol for detail):

- Headline baseline = SGLang **default (overlap-ON)**; `--disable-overlap-schedule` is ablation-only.
- Case A: simple S0→S2→S0→vLLM bracket, reps 5 (tightens the v1 S2 CV weak spot).
- Case C: **interleaved** S0/S2 design (v1 saw ~17% batched session variance), reps 3 per block.
- Clean only — no KAPI, no profiler. Results go under `results/`; server logs under
  `logs/qwen3vl8b/v2/caseAC_rebaseline/`. v1 Phase 5 artifacts are never overwritten.

Raw per-rep dumps live in `results/raw/` (~123 MB, `--output-details` arrays for 400/2000 prompts) and
server logs in `logs/qwen3vl8b/v2/caseAC_rebaseline/`; these are local provenance and not committed by
default (the committed deliverables are the summaries + aggregate `case{A,C}_results.json`).
