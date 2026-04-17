# analysis/

Interpretation layer. Regenerated from `traces/` + `experiments/`; treat as reproducible output.

## Global files

- `category_regex.md` — regex rules mapping kernel names to categories (attention / gemm / communication / norm / quantization / memory / scheduler). Authored once, applied **symmetrically** to SGLang and vLLM traces. Uncategorized bucket must stay < 2 % of GPU time.
- `vllm_source_map.md` — curated kernel-name → `vllm/…` module path. Populated incrementally: every vLLM kernel that crosses 1 % share in any triage gets an entry with a manually verified path. Trades completeness for correctness.
- `hypotheses.md` — structured hypotheses across all cases, de-duplicated. Schema in `plan.md` §Phase 4, Step 4.
- `ranked_recommendations.md` — top 5–10 hypotheses sorted by `confidence × impact × feasibility`. Sole input to `reports/05_recommendations.md`.

## Per-case files

```
analysis/{case}/
├── extend_triage.md      two-trace triage, EXTEND stage (kernel + overlap + fuse tables)
├── decode_triage.md      two-trace triage, DECODE stage
├── breakdown.md          category breakdown applied to SGLang + vLLM
└── vllm_crosscheck.md    per-hypothesis falsification / corroboration record
```

## Hypothesis admissibility

A hypothesis without all of {kernel name, Python source pointer, vLLM evidence, catalog classification, fairness-dependence tier} is **inadmissible** and does not enter `ranked_recommendations.md`.
