# qwen3vl8b — Profiling Experiment

The single retained experiment in this repo: an SGLang-vs-vLLM latency profiling round on
`Qwen/Qwen3-VL-8B-Instruct` (text-only serving). Methodology is in `../../plan.md`. An earlier
exploratory round was removed during the repo restructure (see the Historical note in `../../plan.md`
§0); its numbers were measured under a different stack and are not comparable.

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
| 3 — Profiling / Trace collection | ✅ **complete** (SGLang DECODE + EXTEND, vLLM prefill/decode; Case B SGLang EXTEND unavailable — caveat) | `phase3/summary.md`, `phase3/extend_supplement_summary.md`, `phase3/caseB_trace_issue.md`, `phase3/metadata/`, `../../traces/qwen3vl8b/` |
| 4 — Triage | ✅ **complete** (all 4 cases; hypotheses + ranked recommendations) | `../../analysis/qwen3vl8b/`, `../../reports/qwen3vl8b/03_profiling_analysis.md` |
| 5 — Validation | ⬜ not started (next) | `phase5/` |

## Artifact index
- `env_snapshot.md` — environment record (versions, backends, memory)
- `phase0/` — equivalence.md (Tier A/B/C matrix + verdict), model_files_sha256.txt, tier_a_results.txt,
  sglang_outputs.json, vllm_outputs.json, scripts/
- `phase1/`, `phase2/`, `phase3/`, `phase4/` — per-phase summaries, raw/, metadata/, scripts/
- `phase3/caseB_trace_issue.md` — provenance of the unavailable Case B SGLang EXTEND trace
- Sibling trees: `../../datasets/qwen3vl8b/`, `../../logs/qwen3vl8b/`, `../../traces/qwen3vl8b/`,
  `../../analysis/qwen3vl8b/`, `../../reports/qwen3vl8b/`, `../../configs/qwen3vl8b/`

## Next step
**Phase 5 validation** — validate the top hypothesis (H1: SGLang eager `aten::mm` dispatch vs vLLM
torch.compile/CUDA-graph) by measuring CPU launch-gap and testing graph/compile coverage on Cases A
and C. See `../../analysis/qwen3vl8b/ranked_recommendations.md`. Carry Case B caveats (SGLang EXTEND
unavailable; confidence ceiling M) and the attention-backend ceiling M.
