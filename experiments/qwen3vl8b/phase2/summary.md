# Phase 2 — Shaping / Variance Gate Summary

Generated: 2026-05-22 14:53 UTC

Active run `qwen3vl8b` · GPU 7 · model Qwen3-VL-8B-Instruct @ 0c351dd · TP=1 · bf16.

## Case A — Short latency (128→128, c=1) — Shaping

### Screen results (1 rep each)

| Variant | Flags | TTFT p50 (ms) |
|---------|-------|---------------|
| default | `(none)` | 22.2 |
| no_overlap | `--disable-overlap-schedule` | 19.5 |
| stream8 | `--stream-interval 8` | 22.2 |
| chunk_off | `--chunked-prefill-size -1` | 22.5 |
| chunk_64 | `--chunked-prefill-size 64` | 53.9 |

### Finalist results (3 reps each)

| Variant | TTFT p50 median (ms) | CV | TTFT p95 | TTFT p99 |
|---------|---------------------|-----|----------|----------|
| no_overlap | 19.6 | 3.2% | 24.9 | 25.6 |
| default | 21.8 | 1.7% | 26.2 | 26.3 |

**Winner**: `no_overlap` (flags: `--disable-overlap-schedule`)
**Phase 1 vLLM reference**: 12.6 ms
**Phase 2 residual gap**: 1.56× ↑

---

## Case B — Long prefill (2048→128, c=1) — Shaping

### Screen results (1 rep each)

| Variant | Flags | TTFT p50 (ms) |
|---------|-------|---------------|
| default | `(none)` | 64.9 |
| chunk_off | `--chunked-prefill-size -1` | 62.8 |
| chunk_1024 | `--chunked-prefill-size 1024` | 80.6 |
| chunk_512 | `--chunked-prefill-size 512` | 91.6 |

### Finalist results (3 reps each)

| Variant | TTFT p50 median (ms) | CV | TTFT p95 | TTFT p99 |
|---------|---------------------|-----|----------|----------|
| default | 30.3 | 68.4% | 35.1 | 36.2 |

**Winner**: `default` (flags: `none`)
**Phase 1 SGLang reference**: 66.7 ms (p50) · Phase 1 vLLM: 20.8 ms (cv=114.6% ⚠)
**Phase 2 residual gap (vs vLLM recheck)**: 1.41× ↑

---

## Case C — Batched (512→128, c=16) — Variance Gate

| Warmup | Reps | TTFT p50 median (ms) | CV | Gate |
|--------|------|---------------------|----|------|
| 30 | 3 | 166.1 | 12.5% | FAIL ⚠ |
| 100 | 3 | 140.0 | 15.2% | FAIL ⚠ |
| 300 | 3 | 149.5 | 14.9% | FAIL ⚠ |
| 500 | 5 | 249.1 | 2.9% | PASS ✅ |

**Recommended warmup for Phase 3**: 500 · reps: 5
**Gate passed**: YES ✅

> **W500 follow-up probe (SGLang only, GPU 1, 5 reps):** TTFT p50 CV dropped to 2.9% at median 249.1 ms — **clean, profilable.** Note the stable median (~249 ms) is *higher* than the noisy W100/W300 medians: the earlier "SGLang faster / 0.79× reversal" was an under-warmup artifact. At stable warmup SGLang is ~1.32× vs vLLM W300 (189.0 ms), matching the Phase-1 W30 ratio. vLLM not re-run at W500 (W300 recheck is warmup-insensitive: 187.9→189.0 ms across W30→W300), so this is a stable reference, not a strict same-warmup comparison.

---

## Case D — Decode-heavy (512→512, c=16) — Variance Gate

| Warmup | Reps | TTFT p50 median (ms) | CV | Gate |
|--------|------|---------------------|----|------|
| 30 | 3 | 206.2 | 3.3% | PASS ✅ |
| 100 | 3 | N/A | N/A | N/A |
| 300 | 3 | N/A | N/A | N/A |

**Recommended warmup for Phase 3**: 30 · reps: 3
**Gate passed**: YES ✅

---

## vLLM Recheck (Cases B and C, warmup=300, 5 reps)

| Case | vLLM p50 (ms) | CV | Bimodal resolved | Ceiling M |
|------|---------------|----|-----------------|-----------|
| caseB | 21.5 | 85.9% | NO | YES ⚠ |
| caseC | 189.0 | 1.9% | YES | no |

Phase 1 vLLM Case B reference: 20.8 ms (cv=114.6% ⚠)
