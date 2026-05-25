# Case C — Clean Rerun Protocol (resolve baseline drift)

Main experiment `qwen3vl8b` · Case C (512→128, **c=16**) · **GPU 6** · no KAPI, no profiler.

## Why this rerun

The first clean Case C run (`caseC_h1_confirmation/`) was **inconclusive**: the S0 baseline bracket
drifted **17.7%** (S0_before 211.47 ms → S0_after 174.11 ms > 5% gate), so the S2 effect could not be
judged. This rerun is designed to **detect and average out that drift** so S2 can be evaluated against a
time-local baseline.

## Design change: interleaved A-B-A-B-A bracket

Instead of one `S0 → S2 → S0`, run **five fresh-server blocks**:

`C_S0_a → C_S2_a → C_S0_b → C_S2_b → C_S0_c`

- Three S0 samples spread across the run let us **measure drift** (max pairwise diff of the three S0
  medians) and form a robust pooled baseline.
- Two S2 samples test **repeatability** of any S2 effect.
- A clean vLLM anchor runs last.
- Each block = fresh server, reps 3 (keeps total runtime reasonable while sampling the time axis).

## Locked protocol (Phase-2 Case C values; identical across SGLang blocks)

- `CUDA_VISIBLE_DEVICES=6`; model `0c351dd…`; bf16; TP=1; SGLang flashinfer.
- Dataset `datasets/qwen3vl8b/caseC_batched.jsonl` (sha `265bde3e48793077`).
- `--max-concurrency 16 --num-prompts 2000 --warmup-requests 500`; **reps 3 per block**; greedy
  `{"temperature":0,"top_p":1}`; `--output-details`.
- C_S0_* = SGLang **default** flags; C_S2_* = `--enforce-piecewise-cuda-graph`.
- vLLM anchor: default, same Case-C protocol, reps 5.
- Fresh server per block; GPU 6 < 2000 MiB after each shutdown; one server at a time.
- **No** `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST`; **no** profiler; no source changes.

## Decision rules

1. **Drift gate:** compute the three S0 medians (a/b/c). If **max pairwise diff ≤ 5%** → bracket stable,
   pooled S0 baseline = median of all S0 reps. If > 5% → still report, but mark **drift unresolved**;
   judge S2 only against the time-adjacent S0 blocks and state reduced confidence.
2. **S2 efficacy:** pooled S2 median (a+b) vs pooled/adjacent S0. If S2 improves **>5%** consistently in
   **both** S2 blocks, 0 failures, smoke OK, CV acceptable → **H1 generalizes to batched Case C**. If S2
   ≈ S0 (within noise) → **no batched benefit; H1 limited to Case A**. If S2 blocks disagree → inconclusive.
3. **Relative to vLLM:** report S2 vs the clean vLLM anchor; partial mitigation only, not parity unless
   CV-stable and within 1.05×.

## Outputs (do NOT overwrite the prior `caseC_h1_confirmation/`)

`experiments/qwen3vl8b/phase5/caseC_clean_rerun/{raw/,results.json,summary.md}`;
`logs/qwen3vl8b/phase5/caseC_clean_rerun/` (ordinary server logs only).

## Stop conditions

GPU 6 not idle / not freed < 2000 MiB · crash / OOM / CUDA error · smoke fail · failed requests > 0 ·
flag unsupported · need for source change · conflicting uncommitted change.
