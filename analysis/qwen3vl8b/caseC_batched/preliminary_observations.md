# Case C — Preliminary Observations (NOT final recommendations)

Run: `qwen3vl8b` · Case C `caseC_batched` (512→128, c=16, default).
Residual gap (Phase 2, W500 clean): **SGLang 249.1 ms vs vLLM 189.0 ms ≈ 1.32×** (CV 2.9%, gate PASS).

> Draft observations from triage; not validated, not Phase-5 recommendations.

---

## OBS-C1 — Eager `aten::mm` dispatch vs vLLM torch.compile/CUDA-graph is the leading batched-gap suspect

- **stage:** EXTEND (prefill) + DECODE
- **kernel / op:** identical `nvjet_sm90_*` FP8 GEMM family in both frameworks; SGLang dispatches via
  `aten::mm` eagerly, vLLM via inductor AOT `call` regions (prefill) + `cudaGraphLaunch` (decode)
- **source pointer:** SGLang `python/sglang/srt/layers/quantization/unquant.py:138 apply`; vLLM
  `torch_compile_cache/torch_aot_compile/.../inductor_cache/…:1395 call` and
  `vllm/v1/worker/gpu_model_runner.py:3568 _model_forward` (graph)
- **SGLang evidence:** GEMM 72.6% (EXTEND) / 85.7% (DECODE), nearly all eager `aten::mm`; only the few
  `_forward_raw` rows are graph-captured (excl 92–97%, slight overlap).
- **vLLM evidence:** same nvjet kernels but inside compiled/graphed regions → per-op launch + epilogue
  overhead amortized and silu/rmsnorm folded into the compiled region. Corroborates dispatch/fusion gap.
- **catalog status:** overlap-catalog (torch.compile / CUDA-graph coverage); vLLM-side compile-fusion
  reference (`vllm-torch-compile-fusions.md`).
- **impact estimate:** H — most plausible source of the stable 1.32× batched gap (compute is shared).
- **confidence (draft):** M — kernel tables show the dispatch path but not the launch-gap time directly.
- **fairness dependence:** no (not attention-backend dependent).
- **caveat:** SGLang vs vLLM differ in graph/compile strategy by design; "gap" here is a config/feature
  difference, not a kernel bug. `scheduler / CPU gap` not measured in GPU-time table.
- **recommended Phase 5 / next action:** measure SGLang batched-decode CPU launch gaps; test whether
  enabling/extending SGLang CUDA-graph or piecewise-graph coverage on the c=16 path narrows the gap.

## OBS-C2 — nvjet FP8 GEMM is the dominant cost (85.7% decode); PR #22392 = absolute lead, not gap-closer

- **stage:** EXTEND 72.6% / DECODE 85.7%
- **kernel / op:** `nvjet_sm90_*` FP8 matmuls, `aten::mm`, `unquant.py:138 apply`
- **SGLang evidence:** fuse table PR #22392 Confirmed at 72.6% / 85.7%.
- **vLLM evidence:** vLLM runs the same nvjet family → shared cost, not the differentiator.
- **catalog status:** open-upstream SGLang PR #22392 (CUTLASS FP8 replacing nvjet).
- **impact estimate:** M absolute; L for closing vLLM gap.
- **confidence (draft):** H for attribution; L that it closes the cross-framework gap.
- **fairness dependence:** no.
- **caveat:** do not present as a vLLM-gap fix.
- **recommended Phase 5 / next action:** track PR #22392 as an absolute-speed experiment, separate from the gap.

## OBS-C3 — Batched radix-cache / allocator memory ops are SGLang-specific but already overlapped

- **stage:** EXTEND
- **kernel / op:** Memcpy DtoD + unrolled_elementwise copy
- **source pointer:** `mem_cache/allocator.py:159 free`, `radix_cache.py:360 match_prefix`,
  `radix_cache.py:440 cache_finished_req`
- **SGLang evidence:** ~3.3% combined GPU time, but overlap table marks them `low-roi-hidden`
  (hid 86.6% / 95.8%) → already hidden behind compute.
- **vLLM evidence:** no equivalent above the 1% cutoff in the vLLM windows.
- **catalog status:** no fuse/overlap miss (already overlapped).
- **impact estimate:** L.
- **confidence (draft):** M.
- **fairness dependence:** no.
- **caveat:** these are batched-serving cache-management ops; visible only at c=16; not on the critical path.
- **recommended Phase 5 / next action:** none (documented as not-a-gap).

## OBS-C4 — Attention backend differs but is not the gap driver (ceiling M)

- **stage:** EXTEND + DECODE
- **kernel / op:** SGLang FlashInfer BatchPrefill/MergeStates vs vLLM FA3 cutlass
- **SGLang evidence:** attention ~11.6% (EXTEND) / 4.3% (DECODE).
- **vLLM evidence:** FA3 ~5% (prefill) / ~10% (decode) — comparable, no large divergence.
- **catalog status:** n/a.
- **impact estimate:** L. **confidence (draft):** M (capped). **fairness dependence:** **yes** (FlashInfer vs FA3).
- **caveat:** different backends → not apples-to-apples at kernel level.
- **recommended Phase 5 / next action:** none for gap-closing; documented ceiling.
