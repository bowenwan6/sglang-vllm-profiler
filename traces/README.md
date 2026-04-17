# traces/

Raw torch profiler artifacts. One subdirectory per selected case. Case subdirectories are created at Phase 3, *after* `experiments/phase2/selected_cases.md` has shortlisted them.

## Per-case layout

```
traces/{case}/
├── sglang_mapping/     graph-off, --profile-by-stage (EXTEND/ DECODE)
├── sglang_formal/      graph-on,  --profile-by-stage (EXTEND/ DECODE)
├── vllm/
│   ├── prefill_like/   window opened around N=8 concurrency-1 long-prompt requests
│   └── decode_like/    window opened inside steady-state decoding at target concurrency
└── collection_notes.md warmup protocol, iteration counts, anomalies
```

## Rules

- Trace files must be 20 MB – 500 MB. If > 1 GB, re-collect with fewer steps.
- Prefer rank-local TP-0 traces over merged traces (required by the triage skill).
- Preserve raw files forever; all interpretation lives in `analysis/`, not here.
