# Case A — Preliminary Observations (NOT final recommendations)

Run: `qwen3vl8b` · Case A `caseA_short` (128→128, c=1, `--disable-overlap-schedule`).
Residual TTFT gap from Phase 2: **SGLang 19.6 ms vs vLLM 12.6 ms ≈ 1.56×**.

> These are **draft** observations from the pilot triage, not validated hypotheses and not Phase-5
> recommendations. Confidence is a draft. Each carries the schema fields requested for Phase 4.

---

## OBS-A1 — Eager kernel-launch (no CUDA-graph on prefill) is the leading TTFT-gap suspect

- **stage:** EXTEND (prefill); also DECODE dispatch
- **kernel / op:** all `nvjet_sm90_*` FP8 GEMMs via `aten::mm` / `cudaLaunchKernelExC` (SGLang) vs the
  identical nvjet GEMMs via `cudaGraphLaunch` (vLLM)
- **source pointer:** SGLang `python/sglang/srt/layers/quantization/unquant.py:138 apply`; vLLM
  `vllm/model_executor/models/qwen3_vl.py:1511 forward` (graph-captured)
- **SGLang evidence:** EXTEND/DECODE both GEMM-bound (84% / 75%); top GEMM rows show `excl 100% / hid
  0%` (run exclusively, no overlap); mapping trace dispatches eagerly.
- **vLLM evidence:** every vLLM GEMM/attention row dispatched via `cudaGraphLaunch` — vLLM amortizes
  per-kernel launch overhead via captured graphs even on the prefill window; corroborates that the
  gap is dispatch-side, not compute-side. Phase-1 corroboration: A→B prompt ×16 but TTFT only +4.9 ms
  (compute-insensitive → dispatch-bound).
- **catalog status:** overlap-catalog (launch-overhead / CUDA-graph coverage); not a fuse pattern.
- **impact estimate:** H — most likely explains the bulk of the 1.56× residual on a 128-token prefill
  where kernels are sub-ms.
- **confidence (draft):** M — kernel-share tables cannot directly measure CPU launch gaps; needs gap
  analysis to confirm.
- **fairness dependence:** no (does not depend on FlashInfer-vs-FA3).
- **caveat:** GPU-time kernel table does not contain the `scheduler / CPU gap` category; the claim is
  inferred from dispatch path + Phase-1 scaling, not measured launch-gap time.
- **recommended Phase 5 / next action:** measure inter-kernel CPU launch gaps on the SGLang prefill
  window; check whether SGLang prefill runs under CUDA graph / piecewise-graph and whether enabling it
  closes the gap. (Validation only — not executed now.)

## OBS-A2 — nvjet FP8 GEMM family is the dominant GPU cost; PR #22392 CUTLASS-FP8 is an absolute-speed lead

- **stage:** EXTEND (81.2%) + DECODE (72.4%)
- **kernel / op:** `nvjet_sm90_tst_*` FP8 scaled matmuls, `aten::mm`
- **source pointer:** `python/sglang/srt/layers/quantization/unquant.py:138 apply`; candidate fused
  path `sgl-kernel/python/sgl_kernel/gemm.py`, `quantization/fp8_utils.py`
- **SGLang evidence:** fuse table "PR #22392 CUTLASS FP8 scaled MM replacing nvjet" Confirmed at 81.2%
  (EXTEND) / 72.4% (DECODE).
- **vLLM evidence:** vLLM runs the **same nvjet family** (decode ~73%) → this is **not** the
  cross-framework differentiator; PR #22392 improves SGLang absolute time but both frameworks pay nvjet.
- **catalog status:** existing open-upstream path (SGLang PR #22392), mainline-elsewhere/in-flight.
- **impact estimate:** M for absolute SGLang speed; **L** for closing the *vLLM gap* (vLLM pays it too).
- **confidence (draft):** H (catalog-confirmed, dominant share) for the cost attribution; L that it
  closes the cross-framework gap.
- **fairness dependence:** no.
- **caveat:** do not present as a gap-closer vs vLLM — it is a shared cost.
- **recommended Phase 5 / next action:** if PR #22392 is mergeable, A/B the CUTLASS-FP8 path for an
  absolute prefill speedup; track separately from the vLLM-gap question.

## OBS-A3 — Attention backend differs but is not the gap driver (ceiling M)

- **stage:** EXTEND + DECODE
- **kernel / op:** SGLang FlashInfer `BatchPrefillWithPagedKVCache` + `MergeStates` vs vLLM cutlass
  `FlashAttnFwdSm90` + `FlashAttnFwdCombine` (FA3)
- **source pointer:** SGLang `flashinfer_backend.py:779 forward_extend` / `:893 forward_decode`; vLLM
  `flash_attn_interface.py:176 flash_attn_varlen_func`
- **SGLang evidence:** attention ~7.3% (EXTEND) / 12.8% (DECODE).
- **vLLM evidence:** FA3 ~11% (prefill) / 6.5% (decode) — comparable share; not a large divergence.
- **catalog status:** n/a (backend difference, not a fuse/overlap miss).
- **impact estimate:** L.
- **confidence (draft):** M — capped by the FlashInfer-vs-FA3 backend difference.
- **fairness dependence:** **yes** — FlashInfer 0.6.11 vs FlashAttention v3; any kernel-level attention
  claim inherits ceiling M.
- **caveat:** different backends → not apples-to-apples at kernel level.
- **recommended Phase 5 / next action:** none for gap-closing; keep as a documented ceiling.
