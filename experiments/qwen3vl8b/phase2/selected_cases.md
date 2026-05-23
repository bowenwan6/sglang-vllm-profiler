# Phase 3 Protocol — Locked from Phase 2

Generated: 2026-05-22 14:53 UTC

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
- **warmup**: 500 · **reps**: 5 · **bench_n**: 2000 · **concurrency**: 16
- **SGLang TTFT p50 (Phase 2, w=500)**: 249.1 (cv=2.9%)
- **vLLM TTFT p50 (Phase 2 recheck, w=300)**: 189.0
- **Residual gap**: 1.32× (SGLang slower)
- **Gate**: PASS ✅ (W500 probe, CV<5%)
- **W500 probe**: 5 reps, CV 2.9% — **cleanly profilable.** The W100/W300 "SGLang faster (0.79×)" reading was an under-warmup artifact; the stable W500 median (249.1 ms) restores the ~1.32× SGLang-slower gap seen in Phase-1 W30.
- **Warmup-mismatch caveat**: SGLang stabilized at W500 vs vLLM W300. vLLM is warmup-insensitive here (187.9→189.0 ms across W30→W300), so the ~1.32× comparison is sound as a stable reference. A strict same-warmup vLLM W500 is **only needed if a 'SGLang faster' claim is made** (it is not).
- **vLLM ceiling**: attention backend differs — ceiling M on kernel-level findings
- **Phase 3 rationale**: Batched decode tests scheduler throughput path; ~1.32× gap warrants profiling

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
