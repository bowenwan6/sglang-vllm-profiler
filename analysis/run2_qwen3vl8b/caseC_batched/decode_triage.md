# Case C — SGLang DECODE Triage

Run: `run2_qwen3vl8b` · Case C `caseC_batched` (512→128, c=16, default) · stage **DECODE**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `decode_triage_raw.txt`.

Triage succeeded; both traces parsed; `decode` section rendered.

## Kernel table (≥1% GPU-time share)

| Kernel | Category | GPU time | Share | Launches | Python location | CPU op |
|---|---|---:|---:|---:|---|---|
| nvjet_192x208_64x4_2x1_coopB_TNT | gemm | 76.90 ms | 41.2% | 36 | `unquant.py:138 apply` | aten::mm |
| nvjet_192x192_64x4_1x2_coopB_TNN | gemm | 51.83 ms | 27.8% | 72 | `unquant.py:138 apply` | aten::mm |
| nvjet_256x152_64x4_1x2_coopA_TNT | gemm | 19.72 ms | 10.6% | 36 | `unquant.py:138 apply` | aten::mm |
| flashinfer BatchPrefill (decode) | attention | 7.95 ms | 4.3% | 72 | `flashinfer_backend.py:779 forward_extend` | cudaLaunchKernelExC |
| nvjet_192x192_64x4_2x1_coopB_TNN | gemm | 5.54 ms | 3.0% | 36 | `unquant.py:138 apply` | aten::mm |
| act_and_mul_kernel | uncategorized (activation) | 5.14 ms | 2.8% | 72 | `kernel_api_logging.py:417` | sglang::_run_activation_inplace |
| fused_add_rmsnorm (cutlass) | norm | 4.97 ms | 2.7% | 144 | `kernel_api_logging.py:417` | cudaLaunchKernelExC |
| nvjet_128x144_64x6_1x2_TNT | gemm | 3.77 ms | 2.0% | 72 | `unquant.py:138 apply` | aten::mm |
| elementwise copy | memory | 3.27 ms | 1.8% | 72 | `flashinfer_backend.py:779/893` | aten::copy_ |
| fused_qknorm_warp | norm | 1.94 ms | 1.0% | 72 | `models/utils.py:416 apply_qk_norm` | sglang::fused_inplace_qknorm |

## Overlap-opportunity (formal graph-on)

Top-3 GEMM rows headroom, `excl 100% / hid 0%` (run exclusively), **dep risk high** (large batched
GEMMs, hard to overlap). `act_and_mul` also exclusive.

## Fuse-opportunity (catalog-backed)

| Pattern | Confidence | Related | Share |
|---|---|---:|---:|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | 159.86 ms | **85.7%** |
| Fused QK RoPE reshape + KV cache write | Confirmed | 10.36 ms | 5.6% |
| Fused residual add + RMSNorm | Confirmed | 4.97 ms | 2.7% |
| In-place QK RMSNorm | Confirmed | 1.94 ms | 1.0% |

## Read

Batched DECODE is **the most GEMM-bound stage seen so far: 85.7%** nvjet FP8 GEMM family. At c=16 the
decode GEMMs are large (single coopB kernel = 41.2% alone). Attention only 4.3%. This is consistent
with the batched-throughput path being matmul-dominated. PR #22392 catalog hit again. As with Case A,
the dominant kernels are shared with vLLM (see `vllm_crosscheck.md`), so absolute GEMM cost ≠ the
cross-framework differentiator.
