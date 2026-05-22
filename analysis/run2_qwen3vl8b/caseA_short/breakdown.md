# Case A — Category Breakdown

Run: `run2_qwen3vl8b` · Case A `caseA_short`. Buckets per `analysis/category_regex.md`, applied to the
SGLang kernel tables (EXTEND + DECODE, ≥1% rows). vLLM shown for reference (single-trace).

## SGLang GPU-time share by canonical category

| Category | EXTEND (prefill) | DECODE | Notes |
|---|---:|---:|---|
| **gemm** | **83.5%** | **74.6%** | nvjet FP8 family + splitKreduce; all at `unquant.py:138 apply` / lm_head. **Largest category, both stages.** |
| attention | 7.3% | 12.8% | FlashInfer BatchPrefill + MergeStates. Ceiling M (FlashInfer vs FA3). |
| norm | 3.1% | 5.4% | fused_add_rmsnorm (+ qknorm in decode). |
| memory | <1% | 1.7% | elementwise copy (decode). |
| quantization | ~0 | ~0 | bf16 weights; no fp8-scale kernels above cutoff. |
| communication | 0 | 0 | TP=1, no NCCL (as expected). |
| sampling | <1% | <1% | greedy; negligible. |
| scheduler / CPU gap | n/a (not in GPU-time table) | n/a | **See below — likely where the residual gap lives.** |
| uncategorized | ~1.5% (mrope) | ~3.0% (mrope + activation) | RoPE + act_and_mul; small residual. |

**Largest cost category: GEMM** — ~84% (EXTEND) / ~75% (DECODE) of GPU kernel time, dominated by the
`nvjet_sm90_*` FP8 matmul family at `quantization/unquant.py:138 apply`.

## vLLM reference (single-trace, for comparison)

| Category | prefill_like | decode_like | Notes |
|---|---:|---:|---|
| gemm | ~60%+ | ~73% | same nvjet family, via `cudaGraphLaunch` |
| attention | ~11% (FA3 fwd+combine) | ~6.5% | FA3 cutlass — ceiling M |
| norm | ~3% | ~3% | triton fused_add_rms_norm |
| quantization | ~1.6% | ~1.6% | reshape_and_cache_flash |

## Interpretation

1. **GEMM is the dominant GPU cost in both frameworks** — so it is **not** the cross-framework
   differentiator. The PR #22392 CUTLASS-FP8 path (catalog hit, 72–81% of SGLang GEMM time) is a real
   *absolute* speedup lead for SGLang, but vLLM pays the same nvjet cost, so it does not explain the
   1.56× TTFT gap on its own.
2. The **`scheduler / CPU gap`** category is **invisible in the GPU-time kernel table** but is the most
   likely home of Case A's residual TTFT gap: SGLang dispatches GEMMs eagerly (`aten::mm` /
   `cudaLaunchKernelExC`) while vLLM runs them under `cudaGraphLaunch` (captured CUDA graph). For a
   128-token prefill where each kernel is sub-millisecond, per-launch CPU overhead is the plausible
   driver. Confirming this needs **inter-kernel CPU-gap / launch-overhead analysis** (a Phase 5
   probe), not a kernel-share table.
3. All other categories (attention, norm, memory, sampling) are at parity or below the noise floor.
