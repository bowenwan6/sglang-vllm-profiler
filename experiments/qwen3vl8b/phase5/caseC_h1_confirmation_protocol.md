# Case C — Phase 5.2 H1 Clean Confirmation Protocol (generalization to batched c=16)

Main experiment `qwen3vl8b` · Case C (512→128, **c=16**) · **GPU 6** · no KAPI, no profiler.

## Purpose

Test whether the Case-A H1 result (forcing prefill piecewise CUDA-graph coverage lowers TTFT)
**generalizes to batched serving (c=16)**. Same clean methodology as Case A: `S0 → S2 → S0` bracket +
clean vLLM anchor, all on GPU 6, no instrumentation.

## Locked protocol (Phase-2 Case C values; identical across variants)

- `CUDA_VISIBLE_DEVICES=6`; model `0c351dd…`; bf16; TP=1; SGLang flashinfer.
- Dataset `datasets/qwen3vl8b/caseC_batched.jsonl` (sha `265bde3e48793077`).
- `--max-concurrency 16 --num-prompts 2000 --warmup-requests 500`; **reps 5**; greedy
  `{"temperature":0,"top_p":1}`; `--output-details`.
- Fresh server per variant; GPU 6 < 2000 MiB after each shutdown; one server at a time.
- **No** `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST`; **no** profiler; no source changes.

## Variants (in order)

1. **C_S0_before** — SGLang **default** flags (Case C baseline; decode graph on, VLM piecewise prefill off)
2. **C_S2_enforce_piecewise** — `--enforce-piecewise-cuda-graph`
3. **C_S0_after** — SGLang default flags (drift check)
4. **C_V0_clean_anchor** — vLLM default, **same Case-C protocol (warmup 500, reps 5)** — do NOT reuse the
   old vLLM W300 reference for the final comparison.

## Decision rules

1. **Baseline reproducibility:** `C_S0_before` vs `C_S0_after` median drift ≤ 5% and CV acceptable →
   allowed to interpret S2. Otherwise inconclusive.
2. **S2 efficacy:** if `C_S2` improves >5% vs **both** S0 brackets, 0 failures, smoke passes, CV
   acceptable → **H1 generalized to batched Case C**. If no improvement → conclusion limited to Case A
   short-latency (do not generalize). If unstable → inconclusive; do not widen testing.
3. **Relative to vLLM:** report S2 vs `C_V0_clean_anchor` as partial mitigation / gap-component; not
   parity unless CV-stable and within 1.05×.

## Outputs

`experiments/qwen3vl8b/phase5/caseC_h1_confirmation/{raw/,results.json,summary.md}`;
`logs/qwen3vl8b/phase5/caseC_h1_confirmation/` (ordinary server logs only).

## Stop conditions

GPU 6 not idle / not freed < 2000 MiB · crash / OOM / CUDA error · smoke fail · failed requests > 0 ·
flag unsupported · need for source change.
