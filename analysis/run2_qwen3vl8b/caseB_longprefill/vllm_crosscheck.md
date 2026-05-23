# Case B — vLLM Cross-check (corroboration / falsification only)

Run: `run2_qwen3vl8b` · Case B `caseB_longprefill` (2048→128, c=1). Framework: **vLLM 0.21.0** (reference).
Mode: single-trace per window. Raw: `vllm_prefill_raw.txt`, `vllm_decode_raw.txt`.
**Confidence ceiling M** (Case B bimodal both frameworks; SGLang EXTEND trace unavailable).

> vLLM = reference only; no vLLM optimization recommendations. Both windows parsed.

## vLLM prefill_like — real 2048-prefill captured (≥1%)

| Kernel | Category | Share | Launches | CPU op |
|---|---|---:|---:|---|
| nvjet_128x272_64x4_2x1_coopA_TNT | gemm | 51.3% | 864 | **aten::mm (eager)** |
| nvjet_256x144_64x4_2x1_coopA_TNT | gemm | 26.8% | 180 | **aten::mm (eager)** |
| cutlass FlashAttnFwdSm90 (FA3) | attention | 6.9% | 288 | _vllm_fa3_C::fwd |
| nvjet_256x136_64x4_1x2_coopA_TNT | gemm | 4.9% | 108 | aten::mm |
| triton_poi_fused_mul_silu_slice_1 | uncategorized (act) | 2.9% | 288 | (compiled) |
| reshape_and_cache_flash | quantization | 1.6% | 288 | reshape_and_cache_flash |
| triton fused_add_rms_norm (×2) | norm | ~2.7% | 288 | (compiled) |

## vLLM decode_like — decode section (≥1%)

nvjet via `cudaGraphLaunch` (`_model_forward`): 34.1% + 19.4% + 10.7% + 7.6% + …; FA3 7.8% + combine 1.9%.

## Cross-check conclusions vs SGLang (all ≤ M)

1. **GEMM dominance is shared** (vLLM prefill nvjet ~83%; vLLM decode ~72% nvjet + 10% FA3; SGLang
   decode 78.3%). Same nvjet FP8 family. → **Falsifies** "SGLang GEMMs are intrinsically slower." PR
   #22392 = absolute SGLang lead, not a gap-closer.
2. **Prefill dispatch is EAGER in BOTH frameworks here — important Case-B-specific nuance.** Unlike Case
   A/C where vLLM prefill GEMMs ran under `cudaGraphLaunch` / inductor-compiled regions, vLLM's **2048-token
   prefill GEMMs dispatch via `aten::mm` (eager)** — long prefills are not graph-captured in vLLM either.
   So the "eager-dispatch / no-CUDA-graph" hypothesis (OBS-A1/OBS-C1) is **weaker for Case B prefill**:
   both frameworks pay eager launch overhead on the long prefill. The Phase-1 Case B gap (3.20×) likely
   owes more to bimodality + per-request fixed overhead than to a graph-coverage difference. (≤ M.)
3. **Decode path:** vLLM decode runs under CUDA graph (`cudaGraphLaunch`) while SGLang decode is eager
   (`aten::mm`) — the decode-side dispatch difference from OBS-A1/C1 still holds, but Case B is c=1
   short-decode so decode is not the gap driver (TTFT, not TPOT, is the gap).
4. **Attention (ceiling M, fairness-dependent):** SGLang FlashInfer vs vLLM FA3; shares 7–13%. Not the driver.
