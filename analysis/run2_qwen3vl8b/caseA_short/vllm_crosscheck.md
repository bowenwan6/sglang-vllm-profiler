# Case A — vLLM Cross-check (corroboration / falsification only)

Run: `run2_qwen3vl8b` · Case A `caseA_short`. Framework: **vLLM 0.21.0** (reference baseline).
Mode: **single-trace** per window (`--no-profile-by-stage`). Raw: `vllm_prefill_raw.txt`, `vllm_decode_raw.txt`.

> vLLM is the reference. This file is used **only** to corroborate or falsify SGLang hypotheses.
> **No vLLM optimization recommendations** are produced. Any attention-attribution comparison carries
> **ceiling M** (SGLang FlashInfer vs vLLM FlashAttention-v3 — different backends).

- prefill_like: `traces/.../vllm/prefill_like/rank0.*.pt.trace.json.gz`
- decode_like: `traces/.../vllm/decode_like/rank0.*.pt.trace.json.gz` (window also contains an extend/prefill section)

Both windows parsed successfully. Single-trace mode → overlap table is intentionally empty ("use
mapping/formal for overlap"); kernel + fuse tables are valid.

## vLLM prefill_like — kernel table (≥1%)

| Kernel | Category | GPU time | Share | Launches | Python location | CPU op |
|---|---|---:|---:|---:|---|---|
| nvjet_sm90_tst_192x144_64x5_2x1_v_bz_coopB_TNT | gemm | 13.73 ms | 27.9% | 252 | `qwen3_vl.py:1511 forward` | cudaGraphLaunch |
| nvjet_sm90_tst_128x72_64x8_1x2_h_bz_splitK_TNT | gemm | 5.70 ms | 11.6% | 180 | `qwen3_vl.py:1511 forward` | cudaGraphLaunch |
| cutlass FlashAttnFwdSm90 (FA3) | attention | 3.68 ms | 7.5% | 288 | `flash_attn_interface.py:176 flash_attn_varlen_func` | _vllm_fa3_C::fwd |
| nvjet_sm90_tst_64x136_64x8_4x1_v_bz_TNT | gemm | 3.34 ms | 6.8% | 180 | `qwen3_vl.py:1511 forward` | cudaGraphLaunch |
| nvjet_sm90_tst_64x72_64x12_2x1_v_bz_TNT | gemm | 2.63 ms | 5.4% | 180 | `qwen3_vl.py:1511 forward` | cudaGraphLaunch |
| nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 2.41 ms | 4.9% | 8 | `utils.py:92 default_unquantized_gemm` | aten::mm |
| cutlass FlashAttnFwdCombine | attention | 1.69 ms | 3.4% | 288 | `flash_attn_interface.py:176` | _vllm_fa3_C::fwd |
| reshape_and_cache_flash | quantization | 0.78 ms | 1.6% | 288 | `_custom_ops.py:2714` | reshape_and_cache_flash |
| triton fused_add_rms_norm (×2) | norm | ~1.5 ms | ~3.1% | 288 | `qwen3_vl.py:1511 forward` | cudaGraphLaunch |

**Catalog (vLLM):** "vLLM-origin DSV3 router GEMM" (18.2%) + "vLLM fused residual add + RMSNorm" (3.0%).

## vLLM decode_like — kernel table (decode section, ≥1%)

(Long capture: 39 744 launches per kernel → large absolute ms; shares are what matter.)

| Kernel | Category | Share | Python location | CPU op |
|---|---|---:|---|---|
| nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT | gemm | 34.7% | `gpu_model_runner.py:3568 _model_forward` | cudaGraphLaunch |
| nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT | gemm | 19.8% | `_model_forward` | cudaGraphLaunch |
| nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 10.9% | `_model_forward` | cudaGraphLaunch |
| nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 7.7% | `_model_forward` | cudaGraphLaunch |
| cutlass FlashAttnFwdSm90 (FA3) | attention | 6.5% | `_model_forward` | cudaGraphLaunch |
| nvjet_sm90_tst_384x8_…TNT (lm_head) | gemm | 5.8% | `utils.py:92 default_unquantized_gemm` | aten::mm |

**Catalog (vLLM):** "vLLM-origin Attention + Quantization" (15.7%) + "vLLM fused residual add + RMSNorm" (2.9%).

## Cross-check conclusions vs SGLang

1. **GEMM dominance is shared, not a differentiator.** Both frameworks are dominated by the **same
   `nvjet_sm90_*` FP8 GEMM family** (vLLM prefill ~60%+ across nvjet rows; vLLM decode ~73%; SGLang
   81% / 72%) at the **same matmul sites** (linear layers + lm_head). → **Falsifies** "SGLang is slow
   because its GEMM kernels are slow": vLLM runs the identical nvjet GEMMs. The PR #22392 CUTLASS-FP8
   opportunity would help SGLang's *absolute* GEMM time but does **not** by itself explain the
   *cross-framework* TTFT gap, since vLLM pays the same nvjet cost.
2. **Launch path differs — strong lead.** Every vLLM GEMM/attention row is dispatched via
   **`cudaGraphLaunch`** (full CUDA-graph capture, incl. the prefill window), whereas SGLang's GEMMs
   dispatch via **`aten::mm`** (eager `cudaLaunchKernelExC`) in the mapping trace. For Case A's tiny
   128-token prefill (kernels are sub-millisecond), **per-kernel CPU launch overhead** plausibly
   dominates the 1.56× TTFT residual — consistent with Phase-1's "A→B prompt 16× but TTFT +4.9 ms"
   (compute-insensitive, dispatch-bound). → **Corroborates** the scheduler/dispatch-overhead focus.
3. **Attention backends differ (ceiling M).** SGLang FlashInfer `BatchPrefillWithPagedKVCache` vs vLLM
   cutlass `FlashAttnFwdSm90`/`FlashAttnFwdCombine` (FA3). Shares are comparable (~5–8%), so attention
   is **not** the gap driver, but any kernel-level attention comparison stays **≤ M**.
4. **Norm / kv-cache parity.** Both fuse residual-add+RMSNorm (~3%) and write kv-cache at ~1.5–1.6%.
   No cross-framework gap here.
