# Case C — Category Breakdown

Run: `run2_qwen3vl8b` · Case C `caseC_batched` (512→128, c=16). Buckets per `analysis/category_regex.md`.

## SGLang GPU-time share by canonical category

| Category | EXTEND (prefill) | DECODE | Notes |
|---|---:|---:|---|
| **gemm** | **~73%** | **~85%** | nvjet FP8 family + splitKreduce; `unquant.py:138`. **Largest, both stages; decode even more so.** |
| attention | ~11.6% | ~4.3% | FlashInfer BatchPrefill (extend+decode masks) + MergeStates. Ceiling M. |
| norm | 3.2% | 3.7% | fused_add_rmsnorm (+ qknorm). |
| memory | ~4.3% | 1.8% | **radix-cache / allocator copies** (extend, batched-specific) + flashinfer copies. Mostly hidden/overlapped. |
| quantization | ~0 | ~0 | bf16. |
| communication | 0 | 0 | TP=1. |
| sampling | <1% | <1% | greedy. |
| scheduler / CPU gap | n/a | n/a | not in GPU-time table; see interpretation. |
| uncategorized | ~1.3% (rope) | ~2.8% (activation) + rope | act_and_mul + mrope. |

**Largest cost category: GEMM** — ~73% (EXTEND) / ~85% (DECODE), even more GEMM-dominated than Case A.

## vLLM reference (single-trace)

| Category | prefill_like | decode_like | Notes |
|---|---:|---:|---|
| gemm | ~80% (nvjet, inductor AOT regions) | ~57%+ nvjet + 10% FA3 | compiled/graphed dispatch |
| attention | ~5% (FA3) | ~10% (FA3) | ceiling M |
| norm | ~3.3% | ~2.7% | inductor fused |

## Interpretation

1. **GEMM dominates both frameworks** (even more than Case A) → not the cross-framework differentiator;
   PR #22392 is an absolute SGLang lead only.
2. **The Case C differentiator is dispatch/compilation, not category mix.** vLLM runs the same GEMM
   category under **torch.compile inductor AOT regions** (prefill) and **CUDA graph** (decode); SGLang
   runs it **eagerly** (`aten::mm`). The category breakdown is nearly identical between frameworks —
   the gap lives in *how* the GEMM category is launched/fused, not in *which* category dominates.
3. **Batched memory-management category is SGLang-specific** (radix-cache/allocator, ~4.3% of EXTEND)
   but **86–96% overlapped** → not a gap source.
4. Attention/norm/sampling at parity or below noise.
