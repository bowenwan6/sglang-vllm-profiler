# Case A — S2 Stability Supplement — Results

Main experiment `qwen3vl8b` · Case A (128→128, c=1) · **GPU 6** · **reps=5** · no KAPI, no profiler ·
0 failed requests · GPU 6 freed after each. Protocol identical to clean confirmation.

## Results

| Variant | TTFT p50 reps (ms) | median | CV | TPOT p50 med | failures |
|---|---|---:|---:|---:|---:|
| **S2_stability** (`--disable-overlap-schedule --enforce-piecewise-cuda-graph`) | 13.70 / 16.91 / 13.36 / 13.22 / 11.77 | **13.358** | **12.3%** | ~5.5 | 0 |
| **V0_stability_anchor** (vLLM default) | 14.49 / 13.21 / 14.78 / 13.90 / 15.42 | **14.489** | 5.3% | ~5.3 | 0 |

Reference baselines (clean confirmation, reps=3): S0 ≈ **19.17 / 19.23 ms**.

## Decision-rule evaluation

- **Parity claim — NOT made.** Rule 1 requires S2 CV ≤ 5% AND S2 median ≤ vLLM × 1.05. Here S2 CV =
  **12.3% (> 5%)**. Although the S2 median (13.36 ms) is numerically just below the vLLM median
  (14.49 ms), the distributions **overlap heavily** (S2 range 11.8–16.9; vLLM range 13.2–15.4). So we do
  **not** claim stable parity or superiority over vLLM.
- **S2 substantially reduces the Case-A TTFT gap — confirmed (rule 2).** S2 (13.36 ms) is clearly below
  the S0 baseline (~19.2 ms) — a ~30% reduction here, consistent with the confirmation's ~39%. The
  S2-vs-S0 improvement is robust across runs; only the S2-vs-vLLM *relative position* is unresolved.
- **Stability:** clean (0 errors, no crash), but S2 run-to-run variance is **elevated (CV 12.3%)** — one
  high rep (16.9 ms). This is the reason a parity claim is withheld; it is not a failure.

## Conclusion

- **H1 (forcing prefill piecewise CUDA-graph coverage substantially lowers Case-A TTFT): supported and
  stable in direction/magnitude** (S0 ~19 ms → S2 ~13 ms across two independent clean runs).
- **"S2 stably beats vLLM": NOT established** — S2 reaches the vLLM TTFT range, but with CV 12.3% the
  relative ordering is within noise. State only: *S2 reaches the vLLM TTFT range in Case A.*
- No errors/crash → Case C clean confirmation may proceed (test generalization to c=16 batched).

## Caveats

`--enforce-piecewise-cuda-graph` is a testing lever (not a production fix); single case (A, c=1, greedy);
S2 variance is elevated (CV 12.3%) — a possible warm-up/scheduling sensitivity worth noting for any
future production-path design.
