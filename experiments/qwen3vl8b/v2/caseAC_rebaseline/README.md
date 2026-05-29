# v2 / Round 2 — Case A/C rebaseline (issue #2)

Production-default **overlap-ON** rebaseline for Qwen3-VL. Demotes v1's `--disable-overlap-schedule`
headline to an optional ablation and re-tests whether the PCG lever still helps against the production
baseline.

- **Issue:** #2 (parent #1) on `bowenwan6/sglang-vllm-profiler`.
- **Status:** **protocol drafted / pending approval / no runs executed.**
- **Protocol:** [`protocol.md`](protocol.md) — decision-complete 12-section plan (goal, questions,
  variants, run design, metrics, acceptance, stop conditions, command templates).
- **Model:** `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` (same as v1; verify in env snapshot before runs).

Key design points (see protocol for detail):

- Headline baseline = SGLang **default (overlap-ON)**; `--disable-overlap-schedule` is ablation-only.
- Case A: simple S0→S2→S0→vLLM bracket, reps 5 (tightens the v1 S2 CV weak spot).
- Case C: **interleaved** S0/S2 design (v1 saw ~17% batched session variance), reps 3 per block.
- Clean only — no KAPI, no profiler. Results go under `results/`; server logs under
  `logs/qwen3vl8b/v2/caseAC_rebaseline/`. v1 Phase 5 artifacts are never overwritten.

No results directory exists yet; it is created when the approved run starts.
