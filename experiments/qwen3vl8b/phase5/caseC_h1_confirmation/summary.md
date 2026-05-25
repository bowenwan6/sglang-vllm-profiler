# Case C — Phase 5.2 H1 Clean Confirmation — Results (generalization test, c=16)

Main experiment `qwen3vl8b` · Case C (512→128, **c=16**) · **GPU 6** · w500 · reps=5 · no KAPI, no
profiler · **0 failed requests** · GPU 6 freed after each. Protocol: dataset `caseC_batched.jsonl`
(sha `265bde3e48793077`), `--num-prompts 2000 --max-concurrency 16 --warmup-requests 500`.

## Results

| Variant | TTFT p50 reps (ms) | median | CV | failures |
|---|---|---:|---:|---:|
| **C_S0_before** (default) | 209.3 / 211.8 / 212.4 / 211.5 / 167.9 | **211.47** | 8.6% | 0 |
| **C_S2_enforce_piecewise** (`--enforce-piecewise-cuda-graph`) | 204.0 / 205.4 / 205.9 / 205.1 / 206.0 | **205.40** | 0.3% | 0 |
| **C_S0_after** (default) | 174.1 / 168.3 / 174.0 / 203.6 / 183.9 | **174.11** | 6.9% | 0 |
| **C_V0_clean_anchor** (vLLM, w500) | 178.3 / 141.7 / 147.5 / 196.1 / 178.4 | **178.28** | 12.2% | 0 |

## Decision-rule evaluation

1. **Baseline reproducibility — FAILED (bracket unstable).** C_S0_before 211.47 ms vs C_S0_after
   174.11 ms → **17.7% drift** (≫ 5%). The default baseline moved substantially across the ~45-min run
   (warm-up / caching / environmental drift), so per protocol the S2 interpretation is **not licensed**
   → **Case C result is inconclusive**.
2. **S2 efficacy — no improvement.** Even setting the drift aside, C_S2 (205.40 ms) is only −2.9% vs
   C_S0_before and **+18% vs C_S0_after** (i.e. *worse* than the after-baseline). It does **not** improve
   >5% vs **both** brackets — it sits within/above the baseline band. So **H1 does NOT generalize to
   batched c=16** in this run. (Note S2's own CV is very low, 0.3% — S2 is stable at ~205 ms; it's the
   default S0 that drifted.)
3. **Relative to vLLM:** C_V0 median 178.3 ms (CV 12.2%). At c=16 the SGLang default baseline is already
   near vLLM (S0_after 174 ≈ vLLM 178; S0_before 211 is ~1.19×). S2 (205) is ~1.15× vLLM — no closing of
   any gap.

## Conclusion

- **H1 does NOT generalize to batched Case C.** Forcing prefill piecewise CUDA-graph coverage gives **no
  TTFT benefit at c=16** (and the baseline bracket drifted 17.7%, so the run is formally inconclusive,
  but there is clearly no S2 improvement). The validated H1 effect is therefore **limited to Case A
  short-latency (c=1)** behavior.
- This is consistent with the Phase-4 picture: at c=16 each prefill kernel is large and the path is more
  compute/GEMM-bound, so per-launch dispatch overhead (which graph coverage removes) is a much smaller
  fraction of TTFT than in the c=1 dispatch-bound case.
- Per the decision rule (bracket drift > 5% → inconclusive; no improvement → do not generalize), **the
  testing scope is not widened further** and no batched-path claim is made.

## Caveats

- Baseline drift (S0 211 → 174 ms across the run) indicates Case C TTFT at c=16 is sensitive to
  warm-up/run-order/environment even at w500; a tighter bracket (interleaved or more reps) would be
  needed to resolve small effects — but the *direction* (no S2 benefit) is clear.
- `--enforce-piecewise-cuda-graph` remains a testing lever. vLLM anchor CV is high (12.2%).
- Conclusion limited to this config (Qwen3-VL text-only, c=16, greedy, this GPU/session).
