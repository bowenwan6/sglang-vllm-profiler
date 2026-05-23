# Case D — SGLang EXTEND (prefill) Triage

Run: `run2_qwen3vl8b` · Case D `caseD_decode` (512→512, c=16, default) · stage **EXTEND/prefill**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `extend_triage_raw.txt`.
Lowest-priority case (decode-heavy sanity; residual gap 1.09×).

Triage succeeded; both traces parsed; `extend/prefill` section rendered.

## Kernel table (top, ≥1%)

| Kernel | Category | Share | Python location |
|---|---|---:|---|
| nvjet_192x16_64x8_2x1_v_bz_TNT | gemm | 21.6% | `unquant.py:138 apply` (aten::mm) |
| nvjet_64x16_64x16_2x1_splitK_TNT | gemm | 10.7% | `unquant.py:138` |
| nvjet_192x8_64x8_2x1_v_bz_TNT | gemm | 10.5% | `unquant.py:138` |
| nvjet_64x16_64x16_4x1_v_bz_TNT | gemm | 6.8% | `unquant.py:138` |
| nvjet_128x8_64x12_4x1_splitK_TNT | gemm | 6.0% | `unquant.py:138` |
| nvjet_128x16_64x11_2x1_splitK_TNT | gemm | 5.9% | `model_runner.py:3191 _forward_raw` (cudaGraphLaunch) |
| flashinfer BatchPrefill (mask=1 / mask=0) | attention | 5.7% + 4.0% | `flashinfer_backend.py:779/893` |
| fused_add_rmsnorm | norm | 3.3% | `kernel_api_logging.py:417` |

## Fuse-opportunity

| Pattern | Confidence | Share |
|---|---|---:|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | **72.5%** |
| Fused QK RoPE reshape + KV cache write | Confirmed | ~11% |
| Fused residual add + RMSNorm | Confirmed | 3.3% |

## Read

EXTEND structure is **essentially identical to Case C** (both 512-token input, c=16): GEMM-bound nvjet
FP8 family 72.5%, attention ~9.7%, fused norm 3.3%. Same `unquant.py:138` site, same PR #22392 catalog
hit. No Case-D-specific prefill behavior.
