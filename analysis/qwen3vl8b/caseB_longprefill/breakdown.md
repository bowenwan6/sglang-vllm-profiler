# Case B — Category Breakdown

Run: `qwen3vl8b` · Case B `caseB_longprefill` (2048→128, c=1). Buckets per `analysis/category_regex.md`.
**Confidence ceiling M** (bimodal). **SGLang EXTEND unavailable** (see `extend_triage.md`).

## SGLang GPU-time share by canonical category

| Category | EXTEND (prefill) | DECODE | Notes |
|---|---:|---:|---|
| gemm | **n/a (no trace)** | **~80%** | nvjet FP8 family; `unquant.py:138`. |
| attention | n/a | 12.9% | FlashInfer BatchPrefill (decode over 2048-token KV → higher share). Ceiling M. |
| norm | n/a | 2.2% | fused_add_rmsnorm. |
| memory | n/a | 1.3% | flashinfer copies. |
| quantization / communication / sampling | n/a | ~0 | bf16, TP=1, greedy. |
| scheduler / CPU gap | n/a | n/a | not in GPU-time table. |
| uncategorized | n/a | 2.6% | act_and_mul. |

**Largest cost category (DECODE): GEMM** ~80%, same as all other cases.

## vLLM reference (single-trace)

| Category | prefill_like | decode_like | Notes |
|---|---:|---:|---|
| gemm | ~83% (nvjet, **eager aten::mm**) | ~72% nvjet + ~10% FA3 (cudaGraphLaunch) | **prefill eager in vLLM too** |
| attention | 6.9% (FA3) | ~10% (FA3) | ceiling M |
| norm | ~2.7% | ~2.7% | |

## Interpretation (≤ M)

1. **GEMM dominates** the captured stages in both frameworks → not the differentiator; PR #22392 is an
   absolute SGLang lead only.
2. **Case-B-specific:** vLLM's long-prefill GEMMs are **eager (`aten::mm`)**, not graph/compiled — so the
   dispatch-overhead lead that explains Case A/C is **weaker here**. The Case B gap is more plausibly
   bimodality + c=1 fixed per-request overhead than graph-coverage. (Cannot confirm SGLang prefill
   dispatch directly — EXTEND trace missing.)
3. SGLang EXTEND breakdown is **unavailable**; prefill-stage category mix is inferred only from vLLM.
4. All Case B conclusions carry ceiling M.
