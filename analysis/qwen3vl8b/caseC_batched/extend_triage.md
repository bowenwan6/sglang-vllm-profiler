# Case C — SGLang EXTEND (prefill) Triage

Run: `qwen3vl8b` · Case C `caseC_batched` (512→128, c=16, default config) · stage **EXTEND/prefill**.
Mode: **two-trace** (mapping graph-off + formal graph-on). Framework: SGLang. Raw: `extend_triage_raw.txt`.

- mapping: `traces/qwen3vl8b/caseC_batched/sglang_extend_mapping/…-TP-0-EXTEND.trace.json.gz`
- formal: `traces/qwen3vl8b/caseC_batched/sglang_extend_formal/…-TP-0-EXTEND.trace.json.gz`

Triage succeeded; both traces parsed; `extend/prefill` section rendered.

## Kernel table (≥1% GPU-time share)

| Kernel | Category | GPU time | Share | Launches | Python location | CPU op |
|---|---|---:|---:|---:|---|---|
| nvjet_192x16_64x8_2x1_v_bz_TNT | gemm | 16.36 ms | 20.9% | 324 | `unquant.py:138 apply` | aten::mm |
| nvjet_192x8_64x8_2x1_v_bz_TNT | gemm | 8.83 ms | 11.3% | 180 | `unquant.py:138 apply` | aten::mm |
| nvjet_64x16_64x16_2x1_splitK_TNT | gemm | 7.55 ms | 9.7% | 468 | `unquant.py:138 apply` | aten::mm |
| nvjet_64x16_64x16_4x1_v_bz_TNT | gemm | 5.14 ms | 6.6% | 324 | `unquant.py:138 apply` | aten::mm |
| nvjet_128x8_64x12_4x1_splitK_TNT | gemm | 5.05 ms | 6.5% | 180 | `unquant.py:138 apply` | aten::mm |
| nvjet_128x16_64x11_2x1_splitK_TNT | gemm | 4.91 ms | 6.3% | 180 | `model_runner.py:3191 _forward_raw` | **cudaGraphLaunch** |
| flashinfer BatchPrefill (mask=1) | attention | 4.13 ms | 5.3% | 324 | `flashinfer_backend.py:779 forward_extend` | cudaLaunchKernelExC |
| flashinfer BatchPrefill (mask=0, decode) | attention | 3.37 ms | 4.3% | 180 | `flashinfer_backend.py:893 forward_decode` | cudaLaunchKernelExC |
| nvjet_64x8_64x16_4x1_v_bz_TNT | gemm | 2.81 ms | 3.6% | 180 | `unquant.py:138 apply` | aten::mm |
| nvjet_384x16_64x4_2x1 (lm_head) | gemm | 2.71 ms | 3.5% | 9 | `logits_processor.py:887` | aten::mm |
| fused_add_rmsnorm (cutlass) | norm | 2.54 ms | 3.2% | 1008 | `kernel_api_logging.py:417` | cudaLaunchKernelExC |
| cublasLt splitKreduce | gemm | 1.86 ms | 2.4% | 1008 | `unquant.py:138 apply` | aten::mm |
| flashinfer MergeStates | attention | 1.54 ms | 2.0% | 504 | `flashinfer_backend.py:779/893` | cudaLaunchKernelExC |
| **Memcpy DtoD** | memory | 1.41 ms | 1.8% | 197 | `mem_cache/allocator.py:159 free` (79%), `radix_cache.py:360 match_prefix` (17%) | aten::cat |
| **unrolled_elementwise copy** | memory | 1.21 ms | 1.5% | 172 | `radix_cache.py:440 cache_finished_req` (70%), `allocator.py:159 free` (27%) | aten::copy_ |
| _triton_mrope_forward_fused | uncategorized (rope) | 1.05 ms | 1.3% | 504 | `rotary_embedding/triton_kernels.py:111` | cuLaunchKernelEx |
| elementwise copy | memory | 0.81 ms | 1.0% | 324 | `flashinfer_backend.py:893/779` | aten::copy_ |

## Overlap-opportunity (formal graph-on)

Top GEMM rows headroom; the `_forward_raw` rows (`cudaGraphLaunch`) show `excl 92.8% / hid 7.2%` and
`97.1% / 2.9%` — i.e. graph-on slightly overlaps them. The two **radix-cache / allocator memory
kernels** are `low-roi-hidden` (excl 13.4% / **hid 86.6%** and excl 4.2% / **hid 95.8%**) → already
overlapped, **skip**.

## Fuse-opportunity (catalog-backed)

| Pattern | Confidence | Related | Share |
|---|---|---:|---:|
| **PR #22392 — CUTLASS FP8 scaled MM replacing nvjet** | Confirmed | 56.81 ms | **72.6%** |
| Fused QK RoPE reshape + KV cache write | Confirmed | 9.13 ms | 11.7% |
| Fused residual add + RMSNorm | Confirmed | 2.54 ms | 3.2% |

## Read

Batched EXTEND is **GEMM-bound** (nvjet FP8 family 72.6%, attention ~11.6%). New vs Case A: batched
c=16 surfaces **radix-cache / allocator memory-management kernels** (`allocator.py:159 free`,
`radix_cache.py:360 match_prefix`, `:440 cache_finished_req`) at ~3.3% combined — but they are
**mostly hidden/overlapped** (86–96%), so not a gap source. Some GEMMs are now graph-captured
(`_forward_raw` via `cudaGraphLaunch`) while most still dispatch eagerly (`unquant.py:138` → `aten::mm`).
