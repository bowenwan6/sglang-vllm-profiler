# Case A — Phase 5.2 H1 Clean (Uninstrumented) Confirmation — Results

Main experiment `qwen3vl8b` · Case A (128→128, c=1) · **GPU 6** · 2026-05-25 · **no KAPI, no profiler**.
All variants: fresh server, greedy correctness smoke passed, **0 failed requests**, GPU 6 freed
(<2000 MiB) after each. Protocol: dataset `caseA_short.jsonl` (sha `fab4917772e08744`),
`--num-prompts 400 --max-concurrency 1 --warmup-requests 30`, reps 3, greedy.

## Results

| Variant | TTFT p50 reps (ms) | TTFT p50 median | CV | TTFT p95 med | TTFT p99 med | TPOT p50 med | failures |
|---|---|---:|---:|---:|---:|---:|---:|
| **S0_before** (`--disable-overlap-schedule`) | 19.743 / 19.169 / 19.075 | **19.169** | 1.5% | — | — | ~5.5 | 0 |
| **S2_enforce_piecewise** (`+ --enforce-piecewise-cuda-graph`) | 14.167 / 11.354 / 11.682 | **11.682** | 10.1% | — | — | ~5.5 | 0 |
| **S0_after** (`--disable-overlap-schedule`) | 20.665 / 19.234 / 18.972 | **19.234** | 3.8% | — | — | ~5.5 | 0 |
| **V0_clean_anchor** (vLLM default) | 13.810 / 13.107 / 12.787 | **13.107** | 3.2% | — | — | ~5.3 | 0 |

(Full per-rep p95/p99/TPOT/throughput in `results.json` / `raw/`.)

## Decision-rule evaluation (protocol §Decision rules)

1. **Baseline reproducibility — STABLE & reproduces Phase 2.**
   - S0_before 19.169 ms vs S0_after 19.234 ms → drift **0.34%** (≤ 5%), CV low → bracket stable.
   - Both ≈ **Phase 2's 19.6 ms** → the uninstrumented baseline **reproduces** Phase 2. This confirms
     the earlier instrumented S0 (53.28 ms) was a **KAPI-instrumentation artifact**, not a real baseline.

2. **S2 efficacy — improves vs both bracket baselines (>5%), 0 errors, smoke OK.**
   - S2 11.682 ms vs S0_before 19.169 (**−39.0%**) and vs S0_after 19.234 (**−39.3%**).
   - → **H1 strengthened by the clean Case A intervention.** (The clean effect is **~39%**, smaller than
     the instrumented screen's ~63% — confirming instrumentation amplified the earlier number — but a
     substantial, real reduction remains.)
   - Caveat: S2 CV is 10.1% (rep1 14.17 ms is a high outlier; rep2/3 ~11.5 ms). Median is robust; variance
     is slightly elevated and worth a confirming rep if precision matters.

3. **Relative to vLLM — S2 reaches Case-A TTFT parity (slightly below).**
   - S0 baseline was 19.17 ms = **1.46×** the V0 anchor (13.11 ms).
   - S2 11.68 ms = **0.89×** V0 → **S2 closes the Case-A TTFT gap to vLLM and lands at/just below parity.**
   - Stated conservatively: forcing prefill piecewise CUDA-graph coverage **eliminates the Case-A TTFT
     gap to vLLM** in this config (c=1, greedy). This is one case; not generalized to batched/other shapes.

## H1 verdict: **strengthened** (clean Case A intervention)

Under an uninstrumented benchmark with a stable S0→S2→S0 bracket that reproduces the Phase-2 baseline,
forcing SGLang prefill **piecewise CUDA-graph coverage** (`--enforce-piecewise-cuda-graph`) causally
reduces Case A TTFT by ~39% (19.2 → 11.7 ms) and removes the gap to vLLM (to ~0.89×), 0 errors. This is
the first clean causal support for H1 — the prefill graph-coverage gap (auto-disabled for VLM models) is
a real, actionable contributor to SGLang's Case-A TTFT.

## Caveats (keep rigorous)

- `--enforce-piecewise-cuda-graph` is a **testing lever** that bypasses the VLM auto-disable (which
  exists for a reason). This validates the **root-cause locus and direction**; it is **not** yet a
  production-ready fix, and correctness was only smoke-checked (8 greedy tokens), not broadly validated.
- **Single case (A), c=1, greedy.** Generality to batched (Case C) is unconfirmed — that is the proposed
  next step.
- S2 CV 10.1% (one warm-up-like outlier rep); median-based conclusion holds but precision is moderate.
- vLLM remains the reference; "parity" is a Case-A TTFT statement, not a global ranking.

## Recommended next step (for approval — NOT executed)

Run the **same clean (uninstrumented) S0→S2→S0 + vLLM** bracket on **Case C** (512→128, c=16) to test
whether the piecewise-graph TTFT win generalizes to the batched path. Do not start Case C without
explicit approval.
