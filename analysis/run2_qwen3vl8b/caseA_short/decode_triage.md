# Case A — SGLang DECODE Triage

Run: `run2_qwen3vl8b` · Case A `caseA_short` (128→128, c=1, `--disable-overlap-schedule`) · stage **DECODE**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `decode_triage_raw.txt`.

- mapping: `traces/run2_qwen3vl8b/caseA_short/sglang_mapping/…-TP-0-DECODE.trace.json.gz` (graph-off)
- formal: `traces/run2_qwen3vl8b/caseA_short/sglang_formal/…-TP-0-DECODE.trace.json.gz` (graph-on)

Triage succeeded; both traces parsed; stage section `decode` rendered.

## Kernel table (≥1% GPU-time share)

| Kernel | Category | GPU time | Share | Launches | Python location | CPU op |
|---|---|---:|---:|---:|---|---|
| nvjet_sm90_tst_192x144_64x5_2x1_v_bz_coopB_TNT | gemm | 2.02 ms | 31.8% | 36 | `quantization/unquant.py:138 apply` | aten::mm |
| nvjet_sm90_tst_128x72_64x8_1x2_h_bz_splitK_TNT | gemm | 1.13 ms | 17.9% | 36 | `quantization/unquant.py:138 apply` | aten::mm |
| nvjet_sm90_tst_128x72_64x8_4x2_h_bz_TNT | gemm | 0.67 ms | 10.6% | 36 | `quantization/unquant.py:138 apply` | aten::mm |
| flashinfer PersistentVariableLengthMergeStatesKernel | attention | 0.47 ms | 7.5% | 36 | `flashinfer_backend.py:893 forward_decode` | cudaLaunchKernelExC |
| nvjet_sm90_tst_64x72_64x12_1x2_h_bz_TNT | gemm | 0.46 ms | 7.3% | 36 | `quantization/unquant.py:138 apply` | aten::mm |
| flashinfer BatchPrefillWithPagedKVCacheKernel | attention | 0.34 ms | 5.3% | 36 | `flashinfer_backend.py:779 forward_extend` | cudaLaunchKernelExC |
| nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 0.30 ms | 4.7% | 1 | `logits_processor.py:887 _compute_lm_head` | aten::mm |
| fused_add_rmsnorm (flashinfer cutlass) | norm | 0.23 ms | 3.6% | 72 | `kernel_api_logging.py:417 wrapper` | cudaLaunchKernelExC |
| cublasLt::splitKreduce_kernel | gemm | 0.14 ms | 2.3% | 36 | `quantization/unquant.py:138 apply` | aten::mm |
| fused_qknorm_warp | norm | 0.12 ms | 1.8% | 36 | `models/utils.py:416 apply_qk_norm` | sglang::fused_inplace_qknorm |
| elementwise direct_copy | memory | 0.11 ms | 1.7% | 36 | `flashinfer_backend.py:779 forward_extend` | aten::copy_ |
| act_and_mul_kernel | uncategorized (activation) | 0.10 ms | 1.6% | 36 | `kernel_api_logging.py:417 wrapper` | sglang::_run_activation_inplace |
| _triton_mrope_forward_fused | uncategorized (rope) | 0.09 ms | 1.4% | 36 | `rotary_embedding/triton_kernels.py:111` | cuLaunchKernelEx |

## Overlap-opportunity table (formal graph-on)

Top-4 GEMM rows all `headroom`, `excl 100% / hid 0%` (run exclusively), dep risk low for the coopB /
4x2 variants, high for the splitK variant. lm_head GEMM dep risk high.

## Fuse-opportunity table (catalog-backed)

| Pattern | Confidence | Related GPU time | Share | Candidate path |
|---|---|---:|---:|---|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | 4.59 ms | **72.4%** | `sgl-kernel/.../gemm.py`, `quantization/fp8_utils.py` |
| Fused QK RoPE reshape + KV cache write | Confirmed | 0.48 ms | 7.5% | `attention/utils.py` |
| Fused residual add + RMSNorm | Confirmed | 0.23 ms | 3.6% | `layers/layernorm.py` |
| In-place QK RMSNorm | Confirmed | 0.12 ms | 1.8% | `models/utils.py`, `jit_kernel/norm.py` |

## Read

DECODE is also **GEMM-bound**: nvjet FP8 GEMM family = **72.4%**, attention (MergeStates + decode-path
BatchPrefill) ~12.8%, norm ~5.4%. Same dominant kernel family and same `unquant.py:138` site as EXTEND;
same open PR #22392 catalog hit. Decode-stage GPU time per step is tiny (~6.3 ms total over 36 launches),
consistent with Phase-1 TPOT parity — the decode kernels themselves are not where the cross-framework
gap lives.
