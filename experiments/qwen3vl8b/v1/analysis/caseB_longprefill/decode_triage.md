# Case B — SGLang DECODE Triage

Run: `qwen3vl8b` · Case B `caseB_longprefill` (2048→128, c=1, default) · stage **DECODE**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `decode_triage_raw.txt`.
**Confidence ceiling M** (Case B bimodal in both frameworks).

- mapping: `traces/.../caseB_longprefill/sglang_mapping/…-TP-0-DECODE.trace.json.gz` (gz verified OK)
- formal: `traces/.../caseB_longprefill/sglang_formal/…-TP-0-DECODE.trace.json.gz` (gz verified OK)

Triage succeeded; both traces parsed; `decode` section rendered.

## Kernel table (≥1% GPU-time share)

| Kernel | Category | GPU time | Share | Launches | Python location | CPU op |
|---|---|---:|---:|---:|---|---|
| nvjet_256x144_64x4_2x1_coopA_TNT | gemm | 19.45 ms | 40.0% | 36 | `unquant.py:138 apply` | aten::mm |
| nvjet_128x272_64x4_2x1_coopA_TNT | gemm | 18.22 ms | 37.5% | 108 | `unquant.py:138 apply` | aten::mm |
| flashinfer BatchPrefill (decode) | attention | 6.28 ms | 12.9% | 36 | `flashinfer_backend.py:779 forward_extend` | cudaLaunchKernelExC |
| act_and_mul_kernel | uncategorized (activation) | 1.28 ms | 2.6% | 36 | `kernel_api_logging.py:417` | sglang::_run_activation_inplace |
| fused_add_rmsnorm (cutlass) | norm | 1.09 ms | 2.2% | 72 | `kernel_api_logging.py:417` | cudaLaunchKernelExC |
| elementwise copy | memory | 0.62 ms | 1.3% | 36 | `flashinfer_backend.py:779` | aten::copy_ |

## Fuse-opportunity (catalog-backed)

| Pattern | Confidence | Related | Share |
|---|---|---:|---:|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | 38.02 ms | **78.3%** |
| Fused QK RoPE reshape + KV cache write | Confirmed | 6.82 ms | 14.1% |
| Fused residual add + RMSNorm | Confirmed | 1.09 ms | 2.2% |

## Read

DECODE again **GEMM-bound** (nvjet FP8 78.3%), attention 12.9% (higher relative share than Case A/C
because c=1 long-context decode has larger per-step attention work over the 2048-token KV). Same
`unquant.py:138` site, same PR #22392 catalog hit. Consistent with all other cases; nothing Case-B-specific
in the decode kernels beyond the ceiling-M caveat. All numbers ≤ M (bimodal).
