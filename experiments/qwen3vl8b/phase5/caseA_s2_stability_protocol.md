# Case A — S2 Stability Supplement Protocol

Main experiment `qwen3vl8b` · Case A (128→128, c=1) · **GPU 6** · no KAPI · no profiler.

## Purpose

The clean Case A confirmation already showed **S2 (`--enforce-piecewise-cuda-graph`) is substantially
better than S0** (~39% lower TTFT). This supplement is **not** re-litigating that; it only adds reps to
judge whether **S2's position relative to vLLM is stable**, because the confirmation's S2 CV was 10.1%
(one warm-up-like outlier rep). reps=5 per variant.

## Protocol (identical to clean confirmation, reps=5)

- `CUDA_VISIBLE_DEVICES=6`; model `0c351dd…`; bf16; TP=1; SGLang flashinfer.
- Dataset `datasets/qwen3vl8b/caseA_short.jsonl`; `--num-prompts 400 --max-concurrency 1
  --warmup-requests 30`; **reps 5**; greedy `{"temperature":0,"top_p":1}`; `--output-details`.
- Fresh server per variant; GPU 6 < 2000 MiB after each shutdown; one server at a time.
- **No** `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST`; **no** profiler; no source changes.

## Variants

1. **S2_stability** — `--disable-overlap-schedule --enforce-piecewise-cuda-graph` (reps 5)
2. **V0_stability_anchor** — vLLM default (reps 5), same GPU 6 / protocol

## Decision rules

- If **S2 CV ≤ 5%** AND **S2 median ≤ vLLM median × 1.05** → may state
  *"Case A reaches TTFT parity with vLLM under enforced piecewise graph coverage."*
- If S2 still clearly beats the original S0 (~19 ms) but CV > 5% → keep
  *"S2 substantially reduces the Case-A TTFT gap"*; do **not** claim stable parity/superiority.
- If S2 shows errors / crash / clear instability → **stop, do not enter Case C**, report stability risk.

## Outputs

`experiments/qwen3vl8b/phase5/caseA_s2_stability/{raw/,results.json,summary.md}`;
`logs/qwen3vl8b/phase5/caseA_s2_stability/` (ordinary server logs only).

## Stop conditions

GPU 6 not idle / not freed < 2000 MiB · crash / OOM / CUDA error · smoke fail · failed requests > 0 ·
flag unsupported · need for source change.
