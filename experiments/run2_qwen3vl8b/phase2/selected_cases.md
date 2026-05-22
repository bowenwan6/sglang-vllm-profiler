# run2 Phase 3 Protocol — Locked from Phase 2

Generated: 2026-05-22 03:58 UTC

> This file is the authoritative Phase 3 input. Do not modify without re-running Phase 2.

## Case A — Short latency (128→128, c=1)

- **Decision**: PROMOTE to Phase 3
- **SGLang server flags**: `--disable-overlap-schedule`
- **warmup**: 30 · **reps**: 3 · **bench_n**: 400 · **concurrency**: 1
- **SGLang TTFT p50 (Phase 2)**: 19.6 (cv=3.2%)
- **SGLang TTFT p95 (Phase 2)**: 24.9
- **SGLang TTFT p99 (Phase 2)**: 25.6
- **vLLM TTFT p50 (Phase 1 reference)**: 12.6 ms
- **Residual gap**: 1.56×
- **vLLM ceiling**: attention backend differs (FlashInfer vs FA3) — ceiling M on kernel-level findings
- **Phase 3 rationale**: Largest gap (4.89× Phase 1); clean variance; primary profiling target

## Case B — Long prefill (2048→128, c=1)

- **Decision**: PROMOTE to Phase 3
- **SGLang server flags**: `(none)`
- **warmup**: 300 · **reps**: 5 · **bench_n**: 200 · **concurrency**: 1
- **SGLang TTFT p50 (Phase 2)**: 30.3 (cv=68.4% ⚠)
- **SGLang TTFT p95 (Phase 2)**: 35.1
- **SGLang TTFT p99 (Phase 2)**: 36.2
- **vLLM TTFT p50 (Phase 2 recheck, w=300)**: 21.5
- **Residual gap**: 1.41×
- **vLLM ceiling**: ceiling M — bimodality NOT fully resolved at w=300; compare ratio directionally only
- **Attention ceiling**: FlashInfer vs FA3 — ceiling M on kernel-level findings
- **Phase 3 rationale**: Second-largest gap; chunked-prefill path tests overhead at long input

## Case C — Batched (512→128, c=16)

- **Decision**: PROMOTE to Phase 3
- **SGLang server flags**: `(none — default config)`
- **warmup**: 300 · **reps**: 5 · **bench_n**: 2000 · **concurrency**: 16
- **SGLang TTFT p50 (Phase 2, w=300)**: 149.5 (cv=14.9% ⚠)
- **vLLM TTFT p50 (Phase 2 recheck, w=300)**: 189.0
- **Residual gap**: 0.79×
- **Gate**: MARGINAL ⚠ — verify before Phase 3
- **vLLM ceiling**: attention backend differs — ceiling M on kernel-level findings
- **Phase 3 rationale**: Batched decode tests scheduler throughput path; 1.32× gap warrants profiling

## Case D — Decode-heavy (512→512, c=16)

- **Decision**: PROMOTE to Phase 3
- **SGLang server flags**: `(none — default config)`
- **warmup**: 30 · **reps**: 3 · **bench_n**: 1000 · **concurrency**: 16
- **SGLang TTFT p50 (Phase 2, w=30)**: 206.2 (cv=3.3%)
- **vLLM TTFT p50 (Phase 1 reference)**: 189.7 ms
- **Residual gap**: 1.09×
- **Gate**: PASS ✅
- **vLLM ceiling**: attention backend differs — ceiling M on kernel-level findings
- **Phase 3 rationale**: Decode-heavy exposes continuous-batching overhead; TTFT gap 1.33× + p99 cv=47.1% in Phase 1

## Global Confidence Ceilings (inherited from Phase 1, confirmed in Phase 2)

| Variable | SGLang | vLLM | Ceiling |
|----------|--------|------|---------|
| Attention backend | FlashInfer 0.6.11.post1 | FlashAttention v3 | M: kernel-level attribution uncertain |
| FlashInfer version | 0.6.11.post1 | 0.6.8.post1 (sampling) | M: sampling kernel timing not directly comparable |
| vLLM prefix caching | ON (default, V1) | ON | Framework-intrinsic, not controlled |
| vLLM Case B bimodality | — | cv=85.9% at w=300, NOT resolved | M: compare directionally only |
