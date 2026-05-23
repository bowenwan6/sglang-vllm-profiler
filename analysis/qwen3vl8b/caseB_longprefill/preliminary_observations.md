# Case B — Preliminary Observations (NOT final recommendations)

Run: `qwen3vl8b` · Case B `caseB_longprefill` (2048→128, c=1, default).
Phase-1 gap 3.20×; Phase-2 W300 1.41× but **CV 68.4% (SGLang) / 85.9% (vLLM) — bimodal, ceiling M**.

> Draft observations. **Every Case B hypothesis is capped at confidence M** (bimodal both frameworks +
> SGLang EXTEND trace unavailable). Not validated; not Phase-5 recommendations.

---

## OBS-B1 — Case B prefill dispatch is eager in BOTH frameworks → graph-coverage is NOT the Case B driver

- **stage:** EXTEND (prefill) — SGLang side inferred (no SGLang EXTEND trace), vLLM side measured
- **kernel / op:** nvjet FP8 GEMMs; **vLLM long-prefill GEMMs dispatch via `aten::mm` (eager)**, not
  `cudaGraphLaunch` (unlike Case A/C decode)
- **source pointer:** vLLM prefill `aten::mm` (eager); SGLang prefill site unverifiable (trace missing)
  but DECODE confirms the same `unquant.py:138` GEMM family
- **SGLang evidence:** none on prefill (EXTEND unavailable); DECODE shows nvjet 80% / PR #22392.
- **vLLM evidence:** prefill nvjet 51.3% + 26.8% via eager `aten::mm`; FA3 6.9%. Both frameworks eager
  on the 2048-prefill → the dispatch-overhead lead (OBS-A1/C1) is **weakened** for Case B prefill.
- **catalog status:** n/a (this is a falsification of the graph-coverage hypothesis for Case B).
- **impact estimate:** informs ranking — Case B gap is likely bimodality + c=1 fixed overhead, not graph coverage.
- **confidence (draft):** M (capped; partly inferred).
- **fairness dependence:** no.
- **caveat:** SGLang prefill dispatch not directly observed; bimodal.
- **recommended Phase 5 / next action:** if Case B is pursued, first resolve bimodality (more warmup /
  isolate the two modes) before any kernel-level prefill claim; capture a real SGLang long-prefill EXTEND
  trace would require `--disable-radix-cache` + a profiler fix for the long-prefill stage window.

## OBS-B2 — nvjet FP8 GEMM dominates DECODE (78.3%); PR #22392 = absolute lead (shared cost)

- **stage:** DECODE
- **kernel / op:** nvjet FP8 family, `aten::mm`, `unquant.py:138 apply`
- **SGLang evidence:** fuse table PR #22392 Confirmed 78.3%.
- **vLLM evidence:** vLLM decode also nvjet-dominated (~72%) → shared cost, not the differentiator.
- **catalog status:** open-upstream PR #22392.
- **impact estimate:** M absolute; L for gap. **confidence (draft):** M (capped). **fairness dependence:** no.
- **caveat:** ceiling M (bimodal).
- **recommended Phase 5 / next action:** track PR #22392 as an absolute-speed experiment, separate from the gap.

## OBS-B3 — SGLang EXTEND trace unavailable (documented data gap, not a finding)

- **stage:** EXTEND
- **observation:** no usable SGLang prefill-stage trace after the original supplement (8-attempt formal
  miss + corrupt mapping) and 3 Phase-4 re-collect attempts (truncated EXTEND / valid-DECODE ×2). Root
  cause: prefix-cache + `--profile-by-stage` long-prefill window labeling. Full provenance in `extend_triage.md`.
- **SGLang evidence:** n/a. **vLLM evidence:** vLLM prefill captured (substitute reference).
- **catalog status:** n/a.
- **impact estimate:** low for kernel attribution (GEMM family + source site known from DECODE/other cases);
  high for prefill-stage *timing* (none for SGLang Case B).
- **confidence (draft):** n/a (data gap). **fairness dependence:** no.
- **caveat:** any Case B prefill-stage claim is vLLM-referenced only and ≤ M.
- **recommended Phase 5 / next action:** out of Phase-4 scope; needs profiler/source work to capture a
  genuine long-prefill EXTEND stage.
