# Phase 4 — Ranked Recommendations (hypothesis-driven, for Phase 5)

Run: `qwen3vl8b`. Ranked by (impact × confidence) toward **closing the SGLang→vLLM TTFT gap**.
**These are ranked *hypotheses* and proposed *validations*, not validated optimizations.** Phase 5 tests
them. vLLM is the reference baseline — no vLLM changes are recommended. See `hypotheses.md` for evidence.

| Rank | Hypothesis | Closes the gap? | Impact | Confidence | Fairness dep. | Phase 5 action |
|---|---|---|---|---|---|---|
| 1 | **H1** — SGLang eager dispatch vs vLLM compile/graph | **Yes (primary)** | H | M | no | Measure CPU launch gaps on SGLang prefill (A, C); test CUDA-graph / piecewise-graph / torch.compile coverage |
| 2 | **H2** — nvjet FP8 GEMM → PR #22392 CUTLASS-FP8 | No (absolute only) | M (abs) / L (gap) | H (attrib) / L (gap) | no | A/B PR #22392 for absolute prefill/decode speedup; track separately from the gap |
| 3 | **H4** — Case B gap = bimodality + c=1 overhead | n/a (deprioritize) | L | M | no | Resolve Case B bimodality before any kernel claim; Case B EXTEND trace unavailable |
| 4 | **H3** — attention backend (FlashInfer vs FA3) | No | L | M (capped) | **yes** | None for gap-closing; documented ceiling M |

## Recommended Phase 5 sequence

1. **Validate H1 first (highest leverage).** On Cases **A** and **C**, measure SGLang inter-kernel CPU
   launch gaps in the prefill window (the `scheduler / CPU gap` that the GPU-time kernel table cannot
   show). Then test whether enabling/extending SGLang CUDA-graph / piecewise-graph (and/or torch.compile)
   coverage on the prefill+dispatch path narrows the measured TTFT gap. This directly targets the
   strongest, fairness-independent hypothesis behind the 1.56× (A) and 1.32× (C) residuals.
2. **Run H2 as a parallel absolute-speed track.** If PR #22392 (CUTLASS FP8 replacing nvjet) is
   mergeable, A/B it for absolute latency. Keep it **separate** from the gap question — vLLM pays the
   same nvjet cost, so this is not expected to close the gap.
3. **Do not invest kernel-level effort in Case B (H4) or attention (H3).** Case B is bimodal with no
   usable SGLang EXTEND trace; attention is a fairness-ceilinged (M) backend difference. Both are
   documented ceilings, not actionable gap-closers.

## Confidence ceilings (carried)

- **All attention-level claims: M** (FlashInfer 0.6.11 vs FlashAttention v3).
- **All Case B cross-framework claims: M** (bimodal both frameworks; SGLang EXTEND trace unavailable).
- **H1 confidence is M, not H**, until a direct CPU-launch-gap measurement confirms the dispatch-overhead
  attribution (kernel-share tables are necessary but not sufficient evidence).

## Status

Phase 4 triage complete for all 4 cases (A/C/B/D). No final optimization is asserted here — Phase 5
validation is required before any recommendation graduates from hypothesis to conclusion.
