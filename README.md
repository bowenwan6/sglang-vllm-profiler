# profiling_lab

SGLang vs vLLM profiling on `Qwen/Qwen3-VL-8B-Instruct`. See `plan.md` for the execution plan.

## Layout

| Path | Role | Layer |
|---|---|---|
| `plan.md` | Execution plan — source of truth | Deliverable |
| `configs/` | YAML configs for `sglang-auto-benchmark` sweeps | Hand-edited |
| `datasets/` | Canonical autobench JSONL (shared across frameworks) | Raw |
| `logs/` | Kernel-API boundary logs, server stderr | Raw |
| `experiments/` | Metric artifacts: env snapshot, Phase 0/1/2/5 outputs | Mixed |
| `traces/` | Torch profiler traces (SGLang mapping+formal, vLLM prefill_like+decode_like) | Raw |
| `analysis/` | Processed interpretation: triage tables, breakdowns, hypotheses | Processed |
| `reports/` | Final human-facing deliverables | Deliverable |

## Reviewer reading order

1. `reports/05_recommendations.md`
2. `reports/02_benchmark_table.md`
3. `analysis/ranked_recommendations.md`
4. `analysis/{case}/decode_triage.md`, `extend_triage.md`
5. `analysis/{case}/vllm_crosscheck.md`
6. `traces/{case}/…` — only when challenging a specific triage row

## Artifact layers

- **Raw** (never hand-edited): `datasets/`, `logs/`, `traces/`, `experiments/*/raw/`, `experiments/phase2_shaping/live_results.jsonl`.
- **Processed** (regenerated from raw): `experiments/*/summary.md`, `analysis/**`.
- **Deliverable** (hand-edited): `reports/**`, `plan.md`, `experiments/env_snapshot.md`, `experiments/phase0_equivalence.md`.
