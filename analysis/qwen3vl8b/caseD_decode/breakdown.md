# Case D — Category Breakdown

Run: `qwen3vl8b` · Case D `caseD_decode` (512→512, c=16). Buckets per `analysis/category_regex.md`.
Lowest-priority decode-heavy sanity check (gap 1.09×).

## SGLang GPU-time share by canonical category

| Category | EXTEND (prefill) | DECODE | Notes |
|---|---:|---:|---|
| **gemm** | **~72%** | **~85%** | nvjet FP8 family; `unquant.py:138`. Largest, both stages. |
| attention | ~9.7% | 4.2% | FlashInfer BatchPrefill. Ceiling M. |
| norm | 3.3% | 3.7% | fused_add_rmsnorm (+ qknorm). |
| memory | <2% | 1.8% | flashinfer copies. |
| quantization / communication / sampling | ~0 | ~0 | bf16, TP=1, greedy. |
| scheduler / CPU gap | n/a | n/a | not in GPU-time table. |
| uncategorized | ~1% (rope) | 2.8% (act) | act_and_mul + mrope. |

**Largest cost category: GEMM** (~72% / ~85%) — identical profile to Case C.

## vLLM reference

prefill: nvjet in inductor AOT regions + FA3; decode: nvjet via cudaGraphLaunch (~60%+) + FA3 13.1%.

## Interpretation

1. **GEMM dominates both frameworks** — same as all cases; not the differentiator.
2. **Case D adds no new gap signal.** Structurally identical to Case C; the only difference (512 vs 128
   output) makes the run more decode-dominated, which **shrinks** the relative gap to 1.09% because the
   fixed first-token overhead is amortized over more decode steps. This corroborates the
   TTFT-fixed-overhead thesis rather than revealing a new bottleneck.
3. Dispatch/compile difference (eager `aten::mm` vs inductor/graph) holds, as in Case C.
