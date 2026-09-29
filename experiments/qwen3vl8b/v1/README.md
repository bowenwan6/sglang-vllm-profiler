# qwen3vl8b — Profiling Experiment

> ⚠️ **Methodology correction (2026-05-26):** SGLang TTFT figures from Phase 1 / Phase 2 Case C were collected with **SGLang-only KAPI logging** and are **instrumentation-confounded (provenance only)**. The Case C **“1.32× SGLang-slower” gap is SUPERSEDED** by the clean rerun (no material median gap; SGLang ≈ vLLM ≈ 190 ms). Data retained unchanged. See `experiments/qwen3vl8b/v1/methodology_correction.md`.


Round 1 (v1) of the Qwen3-VL-8B study: an SGLang-vs-vLLM latency profiling round on
`Qwen/Qwen3-VL-8B-Instruct` (text-only serving). Methodology is in `../../../plan.md`. An earlier
exploratory round was removed during the repo restructure (see the Historical note in `../../../plan.md`
§0); its numbers were measured under a different stack and are not comparable. Later rounds live in
`../v2/` (#2 rebaseline) and `../v3_issue4/` (#4 image+text).

## Environment (summary)
- Model: `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (sha256-verified)
- GPU: single H200, serialized. Phase 0/1 index **0**; Phase 2 index **7**; Case C W500 probe + Phase 3 index **1**.
- SGLang `0.0.0.dev1+g0c8049d9b` (system python3) · vLLM `0.21.0` (conda env `/opt/miniconda3/envs/profiling`)
- torch `2.11.0+cu130`, CUDA `13.0` (aligned across both frameworks)
- Full detail: `env_snapshot.md`.

## Phase status
| Phase | Status | Artifacts |
|---|---|---|
| 0 — Equivalence | ✅ **PASS** | `phase0/` (equivalence.md + Tier-A/B outputs + scripts) |
| 1 — Baseline | ✅ **complete** (24 runs, 0 failures) | `phase1/summary.md`, `phase1/raw/`, `phase1/scripts/` |
| 2 — Shaping / Variance gate | ✅ **complete** (incl. Case C W500 probe) | `phase2/summary.md`, `phase2/selected_cases.md`, `phase2/raw/` |
| 3 — Profiling / Trace collection | ✅ **complete** (SGLang DECODE + EXTEND, vLLM prefill/decode; Case B SGLang EXTEND unavailable — caveat) | `phase3/summary.md`, `phase3/extend_supplement_summary.md`, `phase3/caseB_trace_issue.md`, `phase3/metadata/`, `traces/` |
| 4 — Triage | ✅ **complete** (all 4 cases; hypotheses + ranked recommendations) | `analysis/`, `reports/03_profiling_analysis.md` |
| 5 — Validation | ✅ **complete** for the scoped clean Case A/C validation | `phase5/*/summary.md`, `analysis/phase5/h1_launch_gap/` |

## Artifact index
- `env_snapshot.md` — environment record (versions, backends, memory)
- `phase0/` — equivalence.md (Tier A/B/C matrix + verdict), model_files_sha256.txt, tier_a_results.txt,
  sglang_outputs.json, vllm_outputs.json, scripts/
- `phase1/`, `phase2/`, `phase3/`, `phase4/` — per-phase summaries, raw/, metadata/, scripts/
- `phase3/caseB_trace_issue.md` — provenance of the unavailable Case B SGLang EXTEND trace
- `analysis/` (Phase 4 triage, Phase 5 launch-gap analysis), `traces/` (Phase 3 torch-profiler traces,
  Git LFS), `logs/` (server and orchestrator logs, Git LFS), `reports/` (v1 narrative reports).
  Workloads: `../../../datasets/qwen3vl8b/`. Raw per-request bench JSON was removed from the tree on
  2026-09-29 and is in tag `archive/pre-cleanup-2026-09-29`.

## Outcome
Phase 5 validated H1 on clean runs: forcing SGLang's prefill piecewise CUDA graph brought Case A TTFT
into the vLLM range with TPOT unchanged, while Case C (c=16) showed no Case-A-like gain
(`phase5/*/summary.md`). Round 2 re-ran Case A/C on the production-default overlap schedule (#2,
`../v2/caseAC_rebaseline/`); the current headline numbers are in the root `README.md`.
