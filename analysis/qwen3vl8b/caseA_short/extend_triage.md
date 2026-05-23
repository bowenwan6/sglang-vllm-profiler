# Case A — SGLang EXTEND (prefill) Triage

Run: `qwen3vl8b` · Case A `caseA_short` (128→128, c=1, `--disable-overlap-schedule`) · stage **EXTEND/prefill**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `extend_triage_raw.txt`.

- mapping: `traces/qwen3vl8b/caseA_short/sglang_extend_mapping/…-TP-0-EXTEND.trace.json.gz` (graph-off)
- formal: `traces/qwen3vl8b/caseA_short/sglang_extend_formal/…-TP-0-EXTEND.trace.json.gz` (graph-on)

Triage succeeded; both traces parsed; stage section `extend/prefill` rendered. No source changes, no re-collection.

## Kernel table (≥1% GPU-time share)

| Kernel | Category | GPU time | Share | Launches | Python location | CPU op |
|---|---|---:|---:|---:|---|---|
| nvjet_sm90_tst_192x8_64x8_2x1_v_bz_TNT | gemm | 17.63 ms | 35.7% | 360 | `quantization/unquant.py:138 apply` | aten::mm |
| nvjet_sm90_tst_128x8_64x12_4x1_v_bz_splitK_TNT | gemm | 9.98 ms | 20.2% | 360 | `quantization/unquant.py:138 apply` | aten::mm |
| nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT | gemm | 5.62 ms | 11.4% | 360 | `quantization/unquant.py:138 apply` | aten::mm |
| nvjet_sm90_tst_64x8_64x16_2x1_v_bz_splitK_TNT | gemm | 3.89 ms | 7.9% | 360 | `quantization/unquant.py:138 apply` | aten::mm |
| nvjet_sm90_tst_384x8_64x4_2x1_v_bz_TNT | gemm | 2.95 ms | 6.0% | 10 | `logits_processor.py:887 _compute_lm_head` | aten::mm |
| flashinfer BatchPrefillWithPagedKVCacheKernel | attention | 2.66 ms | 5.4% | 360 | `flashinfer_backend.py:779 forward_extend` | cudaLaunchKernelExC |
| fused_add_rmsnorm (flashinfer cutlass) | norm | 1.51 ms | 3.1% | 720 | `kernel_api_logging.py:417 wrapper` | cudaLaunchKernelExC |
| cublasLt::splitKreduce_kernel | gemm | 1.15 ms | 2.3% | 720 | `quantization/unquant.py:138 apply` | aten::mm |
| flashinfer PersistentVariableLengthMergeStatesKernel | attention | 0.93 ms | 1.9% | 360 | `flashinfer_backend.py:779 forward_extend` | cudaLaunchKernelExC |
| _triton_mrope_forward_fused | uncategorized (rope) | 0.75 ms | 1.5% | 360 | `rotary_embedding/triton_kernels.py:111` | cuLaunchKernelEx |

## Overlap-opportunity table (formal graph-on)

| Priority | Verdict | Kernel | Scope | Formal signal | Dep risk | Rec |
|---|---|---|---|---|---|---|
| P1 | headroom | nvjet_192x8_…TNT | `unquant.py:138 apply` | 17.6 ms, 35.8%, excl 100% / hid 0% | low | try fusion |
| P2 | headroom | nvjet_128x8_…splitK_TNT | `unquant.py:138 apply` | 10.0 ms, 20.3%, excl 100% / hid 0% | high | check deps |
| P1 | headroom | nvjet_64x8_64x16_4x1…TNT | `unquant.py:138 apply` | 5.6 ms, 11.4%, excl 100% / hid 0% | low | try fusion |
| P2 | headroom | nvjet_64x8_64x16_2x1_splitK | `unquant.py:138 apply` | 3.9 ms, 7.9%, excl 100% / hid 0% | high | check deps |
| P2 | headroom | nvjet_384x8_…TNT | `logits_processor.py:887` | 2.95 ms, 6.0%, excl 100% / hid 0% | high | check deps |

All top GEMM rows show `excl 100% / hid 0%` — they run **exclusively** (no kernel-level overlap hiding
them); headroom exists if they can be fused or overlapped, but dep risk is high for the splitK variants.

## Fuse-opportunity table (catalog-backed)

| Pattern | Confidence | Related GPU time | Share | Candidate path |
|---|---|---:|---:|---|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | 40.09 ms | **81.2%** | `sgl-kernel/.../gemm.py`, `quantization/fp8_utils.py` (open SGLang PR) |
| Fused QK RoPE reshape + KV cache write | Confirmed | 3.80 ms | 7.7% | `attention/utils.py` (already has fused path) |
| Fused residual add + RMSNorm | Confirmed | 1.51 ms | 3.1% | `layers/layernorm.py` (already fused) |

## Read

EXTEND is **GEMM-bound**: the `nvjet_sm90_*` FP8 matmul family (all mapping to
`quantization/unquant.py:138 apply` → `aten::mm`) takes **81.2%** of EXTEND GPU time. Attention
(FlashInfer BatchPrefill + MergeStates) is only ~7.3%. The single largest catalog hit is the **open
upstream SGLang PR #22392** that replaces the nvjet FP8 GEMM path with a CUTLASS scaled-MM path to
remove memset bubbles / extra copies — i.e. an existing-but-not-yet-merged path that should apply here.
