# experiments/

Metric artifacts.

| Path | Layer | Contents |
|---|---|---|
| `env_snapshot.md` | Deliverable | Versions, GPU, attention backends, chunked-prefill defaults, idle memory |
| `phase0_equivalence.md` | Deliverable | Tier A/B/C equivalence matrix results |
| `phase1/raw/` | Raw | One `bench_serving` JSON + `meta.json` per (case × framework × rep) |
| `phase1/summary.md` | Processed | 4×2 baseline table + fairness notes |
| `phase2_shaping/` | Raw + Processed | `auto_benchmark run` output dir (`live_results.jsonl`, `results.{jsonl,csv}`, `summary.md`) |
| `phase2/selected_cases.md` | Processed | Phase-3 entry gate: ≤2-case shortlist with the phenomenon to explain |
| `phase5/{hypothesis}/` | Raw + Processed | Per-hypothesis validation sweep |
