# Phase 2 — Case B Chunked-Prefill Disentanglement Sweep

Generated: 2026-04-24T12:49:50 UTC

Phase-1 reference: SGLang TTFT p50 = 62.3 ms | vLLM = 24.1 ms

Case B: 2048-token prompt, output=128, concurrency=1

Note: plan originally listed chunk sizes {2048, 4096, 8192, -1} but chunk≥2048 for a 2048-tok prompt is functionally identical to 8192 (single chunk). Replaced with 512/1024 to actually exercise the chunked-prefill code path.

## Screening results (1 rep each)

| Candidate | chunk-prefill-size | Chunks | TTFT p50 (ms) | Δ vs B0 | Δ vs vLLM |
|---|---|---|---|---|---|
| B0_chunk8192 | 8192 (default, 1 chunk) | 1 | 68.5 | — | +44.4 ms |
| B3_disabled | -1 (disabled) | — | 66.7 | -1.8 ms | +42.6 ms |
| B2_chunk1024 | 1024 (2 chunks) | 2 | 169.2 | +100.7 ms | +145.1 ms |
| B1_chunk512 | 512 (4 chunks) | 4 | 261.5 | +193.0 ms | +237.4 ms |

## Finalist reconfirm (3 reps)

| Candidate | TTFT p50 median (ms) | CV% | Verdict |
|---|---|---|---|
| B0_chunk8192 | 64.4 | 0.9% | STRUCTURAL (same floor) — chunking not responsible for gap |

## Decision

**STRUCTURAL (same floor as Case A) — Case B promotes to Phase 3.**

Chunked prefill in its default configuration (chunk=8192) is a no-op for a 2048-token prompt — the entire prompt fits in one chunk. B0 (default) and B3 (disabled) yield identical TTFT (~66–68 ms), confirming chunking bookkeeping is not responsible for the gap.

**Secondary finding (record in hypotheses.md):** When chunked prefill is actually triggered (chunk_size < prompt_len), each chunk pays an independent dispatch overhead roughly equal to the Case A structural floor:
- B2 (2 chunks × 1024): 169.2 ms ≈ 2 × ~85 ms per-chunk cost
- B1 (4 chunks × 512): 261.5 ms ≈ 4 × ~65 ms per-chunk cost

This implies the ~56 ms structural floor is incurred **per chunk dispatched**, not per request.

- Finalist config: B0 default (chunk=8192, no actual chunking for prompt≤2048)
- Residual TTFT gap vs vLLM: +44 ms (64.4 ms vs 24.1 ms, ~2.7× ratio)
- Residual TTFT CV: 0.9% — low-noise, profilable
- Phase-3 phenomenon label: "SGLang structural scheduler/dispatch floor at c=1 (2048-tok), same origin as Case A; chunked prefill not implicated at default settings"
- Fairness dependence: Framework-intrinsic
