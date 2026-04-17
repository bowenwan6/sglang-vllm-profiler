# configs/

YAML configs driving `sglang-auto-benchmark run`.

- `phase2_shaping/` — tier-1 sweeps (≤4 candidates, 1 case × ≤2 axes) used to rule out configurational explanations for a Phase-1 gap.
- `phase5_validation/` — tier-2 sweeps (~10 candidates, resumable) scoped to a specific Phase-4 hypothesis. File naming: `{hypothesis_slug}.yaml`.

Every config pins `benchmark.dataset_path` to an existing `datasets/*.jsonl` so results remain directly comparable to the Phase-1 baseline.
