# Phase 2 — Selected Cases for Phase 3

Generated: 2026-04-24 UTC

## Phase-3 case shortlist

| Case | Phenomenon | Shaping applied | Residual TTFT (SGLang / vLLM) | Residual CV | Fairness dep. | Phase-3 decision |
|---|---|---|---|---|---|---|
| A — 128→128, c=1 | SGLang ~56 ms structural scheduler/dispatch floor at c=1; unresponsive to overlap/policy/stream flags | None (default) | 56.0 ms / 14.1 ms (4.0×) | 0.1% | Framework-intrinsic | **PROMOTE** (primary) |
| B — 2048→128, c=1 | Same structural floor as Case A; chunked prefill not implicated at default chunk=8192 | None (default) | 64.4 ms / 24.1 ms (2.7×) | 0.9% | Framework-intrinsic | **PROMOTE** (primary) |
| C — 512→128, c=16 | 1.49× TTFT gap at c=16; variance stabilized with warmup=100 | warmup 30→100 | 241.3 ms / 164.1 ms (1.47×) | 4.2% | Framework-intrinsic | **PROMOTE** (secondary) |
| D — 512→512, c=16 | Bimodal TTFT distribution under sustained c=16 + 512-tok decode | — | — | V2 CV=14.8% | Framework-intrinsic | **DROP** (bimodal) |

## Case D note

Bimodal pattern: periodic drop to ~160 ms vs steady ~243 ms across 5-rep V2. Consistent with a periodic server-side event under sustained decode load. Record in `analysis/hypotheses.md` as low-confidence Phase-4 finding candidate.

## Case B secondary finding

When chunked prefill is triggered (chunk_size < prompt_len), TTFT scales with chunk count — each chunk pays ~56–85 ms dispatch overhead. Implies structural floor is per-chunk, not per-request.

## Phase-3 protocol

| Case | Server flags | warmup | bench_n | Concurrency |
|---|---|---|---|---|
| A | default (flashinfer, chunk=8192) | 30 | 400 | 1 |
| B | default (flashinfer, chunk=8192) | 30 | 200 | 1 |
| C | default (flashinfer, chunk=8192) | **100** | 2000 | 16 |
