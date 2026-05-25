# Case A — Phase 5.2 H1 Clean (Uninstrumented) Confirmation Protocol

Main experiment `qwen3vl8b` · Workload Case A (128→128, c=1) · GPU **3**.

## Why this run

The initial Phase 5.2 screen was **instrumentation-confounded**: KAPI logging
(`SGLANG_KERNEL_API_LOGLEVEL=1`) was enabled for SGLang variants only, plausibly inflating the S0→S2
improvement, and the S0 baseline (53.28 ms) did not reproduce Phase 2's 19.6 ms
(`caseA_h1_intervention/baseline_anomaly_audit.md`). This run removes **all** instrumentation and uses
an `S0 → S2 → S0` bracket to (a) check baseline stability/drift and (b) isolate the true S2 effect.

## Strict prohibitions

- **No** `SGLANG_KERNEL_API_LOGLEVEL`, **no** `SGLANG_KERNEL_API_LOGDEST`.
- **No** profiler trace capture / `sglang.profiler`.
- **No** KAPI logging of any kind.
- **No** SGLang source changes.

## Fixed protocol (identical across variants)

- `CUDA_VISIBLE_DEVICES=3`; model `0c351dd…`; dtype bf16; TP=1; SGLang attention backend flashinfer.
- Dataset `datasets/qwen3vl8b/caseA_short.jsonl`; `--num-prompts 400 --max-concurrency 1
  --warmup-requests 30`; reps 3; greedy `{"temperature":0,"top_p":1}`; `--output-details`.
- Fresh server per variant; confirm GPU 3 < 2000 MiB after each shutdown; one server at a time.
- S2 runs a minimal greedy correctness smoke check before benchmarking.

## Execution order (bracket)

1. **S0_before** — `--disable-overlap-schedule`
2. **S2_enforce_piecewise** — `--disable-overlap-schedule --enforce-piecewise-cuda-graph`
3. **S0_after** — `--disable-overlap-schedule`
4. **V0_clean_anchor** — vLLM default (after the bracket, so it doesn't interrupt the S0→S2→S0 drift check)

Not re-run this round: **S1 graph-off** (expensive, not the prefill-TTFT lever) and **S3 torch.compile**
(decide only if S2 confirms).

## Decision rules

1. **Baseline reproducibility:** compare `S0_before` vs `S0_after`. If |Δ| ≤ 5% and CV acceptable →
   bracket stable. Also compare to Phase 2's 19.6 ms; if far off, record **unresolved runtime drift**
   but still use the within-bracket contrast cautiously.
2. **S2 efficacy:** if S2 improves >5% vs **both** S0_before and S0_after, 0 failures, smoke passes,
   CV acceptable → **H1 strengthened by clean Case A intervention**. If S0 returns to ~19.6 ms and S2 no
   longer improves → the earlier 63% was mostly instrumentation artifact → **H1 not causally supported**.
   If S0_before vs S0_after drift >5% → **inconclusive; do NOT enter Case C**.
3. **Relative to vLLM:** if S2 helps but is still slower than `V0_clean_anchor` → report as **partial
   mitigation / an important gap component explained**, NOT parity; residual gap needs further work.

## Stop conditions

GPU 3 unavailable / not freed < 2000 MiB · server crash / OOM / CUDA error · smoke fail · failed
requests > 0 · flag unsupported · need for source change · conflicting uncommitted change.

## Outputs

`experiments/qwen3vl8b/phase5/caseA_h1_confirmation/{raw/,results.json,summary.md}`;
`logs/qwen3vl8b/phase5/caseA_h1_confirmation/` (ordinary **server** logs only — no KAPI).
