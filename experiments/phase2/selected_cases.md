# Phase 2 — Selected Cases for Phase 3

Generated: 2026-04-24 UTC (updated after Step 2.4)

## Phase-3 case shortlist

| Case | Phenomenon | Shaping applied | SGLang TTFT | vLLM TTFT (recheck) | Ratio | SGLang CV | vLLM stability | Phase-3 decision |
|---|---|---|---|---|---|---|---|---|
| A — 128→128, c=1 | SGLang ~56 ms structural scheduler/dispatch floor at c=1 | None (default) | 56.0 ms | 14.1 ms (Phase-1, cv=3.3% ✅) | 4.0× | 0.1% | ✅ Clean (no recheck needed) | **PROMOTE** primary |
| B — 2048→128, c=1 | Same structural floor as Case A; per-chunk dispatch overhead secondary finding | None (default) | 64.4 ms | 24.3 ms (median of stable reps, bimodal ⚠) | ~2.7× | 0.9% | ⚠ CEILING M (across-rep CV=76%, rep1 outlier at 65.4 ms) | **PROMOTE** primary |
| C — 512→128, c=16 | 1.33× TTFT gap at c=16; variance stabilized | warmup 30→100 | 241.3 ms | 180.9 ms (recheck, cv=5.5% ✅) | 1.33× | 4.2% | ✅ Clean | **PROMOTE** secondary |
| D — 512→512, c=16 | Bimodal TTFT distribution under sustained decode load | — | — | — | — | V2 CV=14.8% | — | **DROP** |

## Key revision from Step 2.4

**Case C vLLM baseline revised:** Phase-1 measured 164.1 ms (warmup=30, insufficient for c=16).
Recheck with warmup=300 gives **180.9 ms** as the true stable baseline.
Ratio corrected from 1.49× → **1.33×**. Gap is real and stable (CV=5.5%), Phase-3 proceeds.

**Case B vLLM baseline bimodal:** 5 reps = [65.4, 24.2, 24.3, 23.9, 24.3] ms.
Rep1 is a periodic outlier (~65 ms); steady-state is ~24 ms but unpredictable.
All Phase-4 vLLM cross-checks for Case B carry **confidence ceiling M**.

## Case D note

Bimodal TTFT pattern (periodic ~160 ms outliers vs ~243 ms steady) under c=16 + 512-tok decode.
Record in `analysis/hypotheses.md` as low-confidence Phase-4 finding candidate.

## Phase-3 locked protocol

| Case | Server flags | warmup | bench_n | Concurrency | vLLM confidence ceiling |
|---|---|---|---|---|---|
| A | default (flashinfer, chunk=8192) | 30 | 400 | 1 | None |
| B | default (flashinfer, chunk=8192) | 30 | 200 | 1 | **M** (bimodal vLLM baseline) |
| C | default (flashinfer, chunk=8192) | **100** | 2000 | 16 | None (use 180.9 ms as vLLM baseline) |

## Phase 2 exit criteria — all verified ✅

1. ✅ Case A/B gap status defined — both STRUCTURAL, confirmed by shaping sweeps
2. ✅ Cases C/D variance status defined — C promoted (CV 4.2%), D dropped (bimodal CV 14.8%)
3. ✅ vLLM baselines documented for all promoted cases (A: Phase-1 stable; B: ceiling M; C: clean at 180.9 ms)
4. ✅ Selected cases list: 3 cases promoted (A primary, B primary, C secondary); 1 dropped (D)
5. ✅ experiments/phase2_shaping/** raw layer complete and append-only
