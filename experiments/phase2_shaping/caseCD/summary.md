# Phase 2 — Cases C & D Variance Reduction Gate

Generated: 2026-04-24T14:02:56 UTC

Goal: Determine if TTFT CV drops ≤10% with extended warmup. Server: default flags throughout (client-only sweep).

## caseC_batched (512→128, c=16)

Phase-1 reference: TTFT p50 = 243.9 ms (CV=20.6%) | vLLM = 164.1 ms

| Variant | warmup | reps | TTFT p50 median (ms) | CV% (across reps) | Decision |
|---|---|---|---|---|---|
| V0_warmup30 | 30 | 3 | 241.6 | 9.5% | ✅ CV ≤10% — profilable |
| V1_warmup100 | 100 | 3 | 241.3 | 4.2% | ✅ CV ≤10% — profilable |
| V2_warmup300 | 300 | 5 | 241.9 | 2.1% | ✅ CV ≤10% — profilable |

**Case verdict: PROMOTE** — V2 CV=2.1% ≤ 10%. Phase-3 protocol: warmup=100 (V1 already sufficient; V2 confirms).

## caseD_decode (512→512, c=16)

Phase-1 reference: TTFT p50 = 247.0 ms (CV=2.6%) | vLLM = 184.8 ms

| Variant | warmup | reps | TTFT p50 median (ms) | CV% (across reps) | Decision |
|---|---|---|---|---|---|
| V0_warmup30 | 30 | 3 | 241.8 | 19.8% | ❌ rep3 outlier: 160 ms |
| V1_warmup100 | 100 | 3 | 243.1 | 0.1% | ⚠ lucky 3-rep window |
| V2_warmup300 | 300 | 5 | 242.6 | 14.8% | ❌ rep3 outlier again: 160 ms |

**Case verdict: DROP** — V2 CV=14.8% > 10%. V1's 0.1% was a 3-rep lucky window; V2's 5-rep run re-exposed the bimodal pattern. Periodic drop to ~160 ms vs steady ~243 ms is consistent with a server-side event (KV eviction / CUDA graph re-capture / scheduler housekeeping) under sustained c=16 + 512-tok decode. Record in `analysis/hypotheses.md` as low-confidence Phase-4 finding candidate.
