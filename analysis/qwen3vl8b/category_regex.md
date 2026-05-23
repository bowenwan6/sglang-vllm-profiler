# Shared Kernel Category Regex — Phase 4

Shared category definitions for bucketing torch-profiler kernel-table rows into comparable cost
classes across cases and frameworks. Authored against the **Case A** kernel names (pilot); reused
verbatim for Case C/B/D so categories stay comparable.

The triage script (`analyze_llm_torch_profile.py`) already emits a per-row `Category` column, but its
labels are finer-grained and framework-specific (e.g. `rope`, `activation`, `quantize`, `other`).
This file maps every observed kernel into the **9 canonical categories** required by the Phase 4 plan.

| Canonical category | Regex (case-insensitive, matched against kernel name) | Notes |
|---|---|---|
| attention | `flashinfer.*(prefill\|decode\|attention)`, `BatchPrefillWithPagedKVCache`, `MergeStates`, `FlashAttnFwd`, `FlashAttnFwdCombine`, `merge_attn_states`, `paged.*attn`, `\bmha\b` | Includes FlashInfer (SGLang) + FA3 cutlass (vLLM) attention and their state-merge epilogues. **Ceiling M** on cross-framework attention claims (FlashInfer vs FA3). |
| gemm | `nvjet`, `cutlass.*(gemm\|mm)`, `cublas`, `splitKreduce`, `\bmm\b`, `matmul`, `default_unquantized_gemm` | Dense linear-layer matmuls (qkv/o/gate/up/down proj + lm_head). Dominant category both stages. |
| memory | `direct_copy`, `elementwise.*copy`, `aten::copy_`, `reshape_and_cache` (when not fp8), `index_select`, `\bcat\b`, `memset`, `memcpy` | Copies, gathers, kv-cache writes, reshapes. |
| scheduler / CPU gap | (not a kernel name) — derived from inter-kernel CPU launch gaps / `cudaLaunchKernel` idle | **Not visible in the GPU-time kernel table.** Captured only via gap/overlap analysis; tracked separately, not as a kernel-table row. |
| norm | `rmsnorm`, `layernorm`, `\bnorm\b`, `fused_add_rms_norm`, `fused_qknorm`, `qk_norm` | RMSNorm / LayerNorm / QK-norm (incl. fused residual-add variants). |
| quantization | `quant`, `dequant`, `fp8`, `int8`, `scaled_mm.*scale`, `Fp8KVCache` | fp8/int8 scale & cache-quant kernels. Mostly empty at bf16; `reshape_and_cache_flash<...Fp8KVCacheDataType...>` counts here. |
| communication | `nccl`, `allreduce`, `all_reduce`, `allgather`, `all_gather`, `reduce_scatter`, `sendrecv` | NCCL collectives. Expected empty at TP=1. |
| sampling | `sample`, `argmax`, `topk`, `top_k`, `top_p`, `softmax.*sampl`, `sampler` | Sampling-side kernels. Negligible at greedy. |
| uncategorized | everything else (e.g. `mrope`/`rope`, `act_and_mul`/activation, `triton_poi_fused_*` mixed) | Must stay a small residual; if it grows, refine the regex above. |

## Application rules

1. Match top-to-bottom: first matching category wins. Order of precedence: communication → attention →
   norm → quantization → sampling → memory → gemm → uncategorized. (attention/norm before gemm so
   cutlass-attn and fused-rmsnorm kernels are not swallowed by the broad `gemm` patterns.)
2. The script's own `Category` column is advisory; the canonical bucket above is authoritative for
   `breakdown.md` aggregation.
3. `scheduler / CPU gap` is never a kernel-table row — report it from gap/overlap evidence only, with
   an explicit "(not captured in GPU-time kernel table)" tag.
4. RoPE (`mrope`) and elementwise activation (`act_and_mul`, `silu`) currently fall to `uncategorized`
   by design (kept out of `gemm`/`norm` to avoid inflating those buckets); revisit if a later case
   shows them above ~5%.
