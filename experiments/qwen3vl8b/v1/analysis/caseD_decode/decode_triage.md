# Case D — SGLang DECODE Triage

Run: `qwen3vl8b` · Case D `caseD_decode` (512→512, c=16, default) · stage **DECODE**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `decode_triage_raw.txt`.
Lowest-priority case (decode-heavy sanity; gap 1.09×).

Triage succeeded; both traces parsed; `decode` section rendered.

## Kernel table (top, ≥1%)

| Kernel | Category | Share | Python location |
|---|---|---:|---|
| nvjet_192x208_64x4_1x2_coopB_TNT | gemm | 42.0% | `unquant.py:138 apply` |
| nvjet_128x248_64x4_2x1_coopA_TNT | gemm | 27.2% | `unquant.py:138` |
| nvjet_128x272_64x4_2x1_coopA_TNT | gemm | 10.4% | `unquant.py:138` |
| flashinfer BatchPrefill (decode) | attention | 4.2% | `flashinfer_backend.py:779` |
| nvjet_192x192_64x4_2x1_coopB_TNN | gemm | 2.9% | `unquant.py:138` |
| act_and_mul_kernel | uncategorized (activation) | 2.8% | `kernel_api_logging.py:417` |
| fused_add_rmsnorm | norm | 2.7% | `kernel_api_logging.py:417` |
| fused_qknorm_warp | norm | 1.0% | `models/utils.py:416` |

## Fuse-opportunity

| Pattern | Confidence | Share |
|---|---|---:|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | **85.7%** |

## Read

DECODE is **the most GEMM-bound stage** (nvjet FP8 85.7%, single coopB kernel 42.0%), identical in
structure to Case C decode (both c=16). Attention only 4.2%. This is the decode-heavy sanity check: it
confirms the same matmul-dominated decode path and the same PR #22392 catalog hit as every other case.
Combined with Phase-1 TPOT parity and the small 1.09× residual gap, **Case D adds no new gap signal** —
it corroborates that the decode kernels themselves are not the cross-framework differentiator.
