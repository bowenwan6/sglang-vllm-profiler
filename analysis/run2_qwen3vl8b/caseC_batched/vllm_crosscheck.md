# Case C — vLLM Cross-check (corroboration / falsification only)

Run: `run2_qwen3vl8b` · Case C `caseC_batched` (512→128, c=16). Framework: **vLLM 0.21.0** (reference).
Mode: single-trace per window. Raw: `vllm_prefill_raw.txt`, `vllm_decode_raw.txt`.

> vLLM = reference only; no vLLM optimization recommendations. Attention comparisons carry **ceiling M**.

Both windows parsed. Single-trace → overlap table empty by design.

## vLLM prefill_like — kernel table (≥1%)

GEMM-dominated by the **same nvjet FP8 family**, but mapped to **torch.compile / inductor AOT regions**:

| Kernel | Category | Share | Python location | CPU op |
|---|---|---:|---|---|
| nvjet_192x192_64x4_2x1_coopB_TNN | gemm | 21.5% | `torch_aot_compile/.../inductor_cache/.../…:1395 call` | aten::mm |
| nvjet_128x272_64x4_2x1_coopA_TNT | gemm | 15.3% | `inductor_cache/.../call` | aten::mm |
| nvjet_128x136_64x6_1x2_TNT | gemm | 10.0% | `inductor_cache/.../call` | aten::mm |
| nvjet_192x144_64x5_1x2_coopB_TNT | gemm | 8.3% | `inductor_cache/.../call` | aten::mm |
| nvjet_192x96_64x5_1x2_TNT | gemm | 7.3% | `inductor_cache/.../call` | aten::mm |
| cutlass FlashAttnFwdSm90 (FA3) | attention | 5.1% | `flash_attn_interface.py:176` | _vllm_fa3_C::fwd |
| triton_poi_fused_mul_silu_slice_1 | uncategorized (act) | 2.7% | `inductor_cache/.../call` | (compiled) |
| triton_red_fused_fused_add_rms_norm | norm | ~3.3% | `inductor_cache/.../call` | (compiled) |
| reshape_and_cache_flash | quantization | 1.6% | `_custom_ops.py:2714` | reshape_and_cache_flash |

## vLLM decode_like — kernel table (decode section, ≥1%)

Same nvjet family via **`cudaGraphLaunch`** (`gpu_model_runner.py:3568 _model_forward`):
nvjet 25.7% + 13.9% + 8.2% + 5.7% + … ; FA3 10.2%; lm_head 4.4%; fused_add_rms_norm ~2.7%.

## Cross-check conclusions vs SGLang

1. **GEMM dominance is shared** (vLLM prefill nvjet ~80% across rows; vLLM decode ~57%+ nvjet + 10% FA3;
   SGLang 72.6% / 85.7%). Same nvjet FP8 family, same matmul role. → **Falsifies** "SGLang's GEMM
   kernels are intrinsically slower." PR #22392 is an *absolute* SGLang lead, not a gap-closer.
2. **Dispatch/compilation path differs — strongest Case C lead.** vLLM prefill GEMMs are inside
   **torch.compile / inductor AOT-compiled `call` regions** (`torch_aot_compile/.../inductor_cache/…`),
   and vLLM decode GEMMs run under **`cudaGraphLaunch`**. SGLang dispatches the same GEMMs **eagerly**
   via `aten::mm` at `unquant.py:138` (only a few `_forward_raw` rows are graph-captured). At c=16 the
   GEMMs are large, so raw compute is similar — but vLLM's compiled/graphed dispatch removes per-op
   launch + epilogue overhead and enables inductor fusions (silu/rmsnorm folded into the compiled
   region). → **Corroborates** that the 1.32× batched gap is dispatch/fusion-side, not kernel-speed.
3. **Batched memory management is SGLang-only but hidden.** SGLang's radix-cache/allocator copies
   (`allocator.py:159 free`, `radix_cache.py` match/cache) appear at ~3.3% but are 86–96% overlapped
   (low-roi); vLLM shows no equivalent above cutoff. Not a gap driver.
4. **Attention (ceiling M).** SGLang FlashInfer vs vLLM FA3; shares comparable (4–11%); not the driver.
