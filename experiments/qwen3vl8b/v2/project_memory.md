# v2 / Round 2 — Project Memory

Short running state for the v2 round. Source of truth for *direction* is `plan.md`; this file is the
quick "where are we" pointer. Last updated: 2026-05-29.

## Active state

- **Round:** v2 (Round 2), experiment `qwen3vl8b`, model `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd`.
- **v1 (Phase 0–5):** complete. v1 artifacts are frozen and never overwritten.
- **Issue #2 (default-overlap rebaseline): ✅ COMPLETE / PASS.** First v2 experiment done.
- GitHub issues #1–#5 on `bowenwan6/sglang-vllm-profiler` (@JustinTong0323). Dependency order:
  #2 → {#4, #3} → #5.

## Issue #2 result (clean, GPU 1, 0 failures)

Production-default **overlap-ON** is now the headline baseline; `--disable-overlap-schedule` is ablation
only.

| | SGLang default | SGLang + PCG | vLLM | no-overlap ablation |
|---|---:|---:|---:|---:|
| Case A (c=1) TTFT p50 | 21.94 ms | 14.04 ms | 13.12 ms | 19.07 ms |
| Case C (c=16) TTFT p50 pooled | 204.8 ms | 230.6 ms | 215.7 ms | — |

- **Case A:** PCG still helps on the production default — TTFT −36% (21.94 → 14.04 ms), TPOT flat,
  into vLLM band. v1's no-overlap baseline understated the production gap (default→vLLM gap is *larger*
  than v1's overlap-OFF gap). Confirms the v1 PCG finding is not an overlap-OFF artifact.
- **Case C:** no material gap / no Case-A-like PCG benefit at c=16 (batched CV ~14–15%). Wording stays at
  "no improvement," never "PCG hurts."
- `--enforce-piecewise-cuda-graph` is a **testing lever, not a production fix.**
- Details: `experiments/qwen3vl8b/v2/caseAC_rebaseline/results/{summary,caseA_summary,caseC_summary}.md`.

## Next recommended step

1. **#4 — Qwen3-VL image+text + `SGLANG_USE_CUDA_IPC_TRANSPORT=1` (priority):** realistic VLM production
   path; draft a protocol mirroring #2's clean methodology before any runs.
2. **#3 — Qwen3.5 transfer check** (parallel/after): does the PCG finding transfer to Qwen3.5?
3. **#5 — selective/default-on PCG PR** (after #4).

## Artifact note

- Raw per-rep dumps (`results/raw/`, ~123 MB) and server logs (`logs/qwen3vl8b/v2/`) are **generated but
  not committed** — pending an explicit decision (commit raw like v1, or `.gitignore`/clean up).
- Committed deliverables = summaries + aggregate `case{A,C}_results.json`.
