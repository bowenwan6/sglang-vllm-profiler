# Case C — Clean Rerun (interleaved bracket) — Results

Main experiment `qwen3vl8b` · Case C (512→128, **c=16**) · **GPU 0** (GPU 6 was externally occupied) ·
w500 · no KAPI, no profiler · **0 failed requests** · GPU 0 freed after each. Interleaved
`S0_a → S2_a → S0_b → S2_b → S0_c` (reps 3 each) + clean vLLM anchor (reps 5). Dataset sha `265bde3e48793077`.

## Results

| Block | TTFT p50 reps (ms) | median | CV |
|---|---|---:|---:|
| C_S0_a (default) | 170.0 / 207.2 / 197.5 | 197.52 | 8.2% |
| C_S2_a (`--enforce-piecewise-cuda-graph`) | 203.3 / 184.2 / 195.3 | 195.28 | 4.0% |
| C_S0_b (default) | 171.4 / 208.1 / 210.8 | 208.11 | 9.1% |
| C_S2_b (`--enforce-piecewise-cuda-graph`) | 191.9 / 190.0 / 202.9 | 191.91 | 2.9% |
| C_S0_c (default) | 177.5 / 192.2 / 169.5 | 177.48 | 5.2% |
| C_V0_clean_anchor (vLLM, w500) | 141.8 / 189.8 / 195.4 / 174.5 / 190.4 | 189.79 | 11.0% |

Pooled: **S0 median ≈ 192.2 ms** (9 reps), **S2 median ≈ 193.6 ms** (6 reps).

## Decision-rule evaluation

1. **Drift gate — still FAILS, but now characterized.** The three S0 block medians (197.5 / 208.1 /
   177.5) span **17.3%** (> 5%). So Case-C TTFT at c=16 has **intrinsic run-to-run / session drift of
   ~17% even at warmup=500** — this is a property of the batched workload + fresh-server sessions, not a
   fixable artifact of this protocol. Baseline drift is therefore **unresolved**.
2. **S2 efficacy — no improvement (now well-sampled).** Despite the unresolved absolute drift, the
   interleaved design samples S2 twice, bracketed by three S0 blocks, so the *comparison* is robust:
   - Pooled S2 (193.6 ms) ≈ pooled S0 (192.2 ms) → **+0.7% (no benefit).**
   - Each S2 block sits squarely inside the S0 noise band (S0 reps span 169.5–210.8 ms): S2_a 195.3 is
     between S0_a/S0_b; S2_b 191.9 is between S0_b/S0_c.
   - → **S2 does NOT improve TTFT at c=16.** This confirms (with better sampling) the first run's
     directional read.
3. **Relative to vLLM:** vLLM median 189.8 ms (CV 11%). Pooled S0 (192) and S2 (194) are both ≈ vLLM —
   at c=16 SGLang is already in the vLLM TTFT band; there is no gap for S2 to close.

## Secondary observation (not a TTFT win)

S2 blocks are **more stable** than S0 (S2 CV 4.0% / 2.9% vs S0 CV 8.2% / 9.1% / 5.2%). Forcing piecewise
graph appears to **reduce batched run-to-run variance** without changing the median TTFT. Defensible as a
stability note, **not** a latency improvement.

## Conclusion

- **H1 does NOT generalize to batched Case C (c=16).** Forcing prefill piecewise CUDA-graph coverage
  gives **no median TTFT benefit** at c=16 (pooled S2 ≈ pooled S0 ≈ vLLM ≈ 190 ms). The validated H1
  effect remains **limited to Case A short-latency (c=1)**.
- The baseline-drift problem from the first run is now **characterized** (intrinsic ~17% session
  variance at c=16), and even so the interleaved comparison is unambiguous: no S2 gain at c=16.
- Mechanistically consistent with Phase 4: at c=16 each prefill kernel is large and the path is
  compute/GEMM-bound, so per-launch dispatch overhead (what graph coverage removes) is a small fraction
  of batched TTFT.

## Caveats

- Absolute Case-C TTFT at c=16 is intrinsically noisy across sessions (~17%); a within-process
  interleave (no server restart) would be needed to chase sub-5% effects, but the **no-benefit** verdict
  is robust to that noise.
- `--enforce-piecewise-cuda-graph` is a testing lever; vLLM anchor CV high (11%); single config
  (Qwen3-VL text-only, c=16, greedy). Prior `caseC_h1_confirmation/` left intact (not overwritten).
