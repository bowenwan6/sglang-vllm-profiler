# Phase 4 — Hypotheses (evidence-backed, ranked-input)

Run: `qwen3vl8b` · Qwen3-VL-8B-Instruct @ `0c351dd` · H200 · TP=1 · bf16 · greedy.
Source: per-case triage of Phase 3 traces (Cases A/C/B/D). **These are hypotheses with confidence
H/M/L, not validated conclusions.** Phase 5 validates the top ones. vLLM = reference baseline only;
no vLLM optimization recommendations. Catalog = `llm-torch-profiler-analysis` references.

## Cross-case summary (what every trace agrees on)

1. **Both frameworks are GEMM-bound by the same `nvjet_sm90_*` FP8 GEMM family** (SGLang 72–86% of
   GPU time per stage; vLLM similar). Same matmul role, same kernels. → GEMM cost is **shared**, so it
   is **not** the cross-framework differentiator.
2. **The differentiator is dispatch/compilation, not kernel speed.** SGLang dispatches GEMMs **eagerly**
   (`aten::mm` at `srt/layers/quantization/unquant.py:138 apply`). vLLM runs the same GEMMs inside
   **torch.compile / inductor AOT regions** (prefill, Cases C/D) and under **CUDA graph**
   (`cudaGraphLaunch`, decode, all cases). Exception: **long prefill (Case B) is eager in vLLM too**.
3. **The gap is a TTFT / first-token fixed-overhead effect, not a per-token decode deficit** — Phase-1
   TPOT parity + Case D (512-token decode) shrinking the gap to 1.09× both confirm this.
4. **Attention backend differs (FlashInfer vs FA3)** — comparable shares, never the driver; all
   attention-level claims carry **ceiling M**.

---

## H1 — SGLang eager kernel dispatch (no torch.compile / limited CUDA-graph) drives the TTFT gap

- **case / stage:** A (EXTEND+DECODE), C (EXTEND+DECODE), D (corroborating). **Not** Case B prefill.
- **observation:** identical nvjet GEMMs are eager `aten::mm` in SGLang but compiled/graphed in vLLM;
  SGLang top GEMM rows run `excl 100% / hid 0%` (no overlap). On short/batched prefills where kernels
  are sub-ms to small, per-op CPU launch + un-fused epilogues plausibly account for the residual gap.
- **kernel / op / source:** `nvjet_sm90_*` FP8 GEMMs · SGLang `srt/layers/quantization/unquant.py:138
  apply` (`aten::mm`) · vLLM `inductor_cache/…call` (prefill) + `gpu_model_runner.py:3568 _model_forward`
  (`cudaGraphLaunch`).
- **SGLang evidence:** A EXTEND GEMM 81%/eager; C EXTEND 73% (only `_forward_raw` rows graph-captured),
  C DECODE 85% eager; D mirrors C.
- **vLLM evidence:** all vLLM GEMM rows compiled/graphed (Cases A/C/D); corroborated by Phase-1
  "prompt ×16 → TTFT +4.9 ms" (compute-insensitive → dispatch-bound).
- **catalog status:** overlap-catalog (CUDA-graph / torch.compile coverage); `vllm-torch-compile-fusions.md`.
- **impact:** **H** (best explanation of the 1.56× Case A and 1.32× Case C residual gaps).
- **confidence:** **M** — kernel-share tables show the dispatch *path* but not the launch-gap *time*;
  needs CPU-gap measurement to confirm.
- **fairness dependence:** no.
- **caveats:** SGLang vs vLLM graph/compile strategy is a design/config difference, not a kernel bug;
  `scheduler / CPU gap` is not in the GPU-time kernel table; does not apply to Case B prefill (H4).
- **Phase 5 validation:** measure SGLang inter-kernel CPU launch gaps (prefill window, Cases A/C); test
  whether enabling/extending SGLang CUDA-graph / piecewise-graph (or torch.compile) coverage on these
  paths narrows the measured TTFT gap.

## H2 — nvjet FP8 GEMM is the dominant absolute cost; PR #22392 (CUTLASS FP8) is an absolute-speed lead, NOT a gap-closer

- **case / stage:** all cases, both stages.
- **observation:** the nvjet FP8 GEMM family is 72–86% of SGLang GPU time and is a Confirmed catalog
  match to open SGLang **PR #22392** (CUTLASS scaled-MM replacing nvjet, removing memset bubbles/copies).
- **kernel / op / source:** `nvjet_sm90_*` · `unquant.py:138 apply` → candidate `sgl-kernel/.../gemm.py`,
  `srt/layers/quantization/fp8_utils.py`.
- **SGLang evidence:** fuse table Confirmed: A 81%/72%, C 73%/86%, B 78% (decode), D 73%/86%.
- **vLLM evidence:** vLLM runs the **same nvjet family** → it is a shared cost; speeding it up helps
  SGLang absolute latency but does **not** by itself close the vLLM gap.
- **catalog status:** open-upstream (PR #22392).
- **impact:** **M** for absolute SGLang speed; **L** for closing the cross-framework gap.
- **confidence:** **H** for the cost attribution; **L** that it closes the gap.
- **fairness dependence:** no.
- **caveats:** must not be presented as a vLLM-gap fix.
- **Phase 5 validation:** if PR #22392 is mergeable, A/B the CUTLASS-FP8 path for an absolute prefill/decode
  speedup; track separately from the gap question.

## H3 — Attention backend (FlashInfer vs FA3) is not the gap driver (ceiling M)

- **case / stage:** all, EXTEND+DECODE.
- **observation:** SGLang FlashInfer `BatchPrefillWithPagedKVCache`/`MergeStates` vs vLLM cutlass
  `FlashAttnFwdSm90`/`Combine` (FA3); shares comparable (4–13%), no large divergence.
- **source:** SGLang `flashinfer_backend.py:779/893`; vLLM `flash_attn_interface.py:176`.
- **evidence:** A attn 7%/13%; C 11.6%/4.3%; D 9.7%/4.2%; vLLM FA3 5–13%.
- **catalog status:** n/a (backend difference).
- **impact:** **L**. **confidence:** **M** (capped). **fairness dependence:** **yes** (FlashInfer 0.6.11 vs FA3).
- **caveats:** not apples-to-apples at kernel level → documented ceiling, not a recommendation.
- **Phase 5 validation:** none for gap-closing; keep as a ceiling on any attention-level claim.

## H4 — Case B long-prefill gap is bimodality + c=1 fixed overhead, NOT graph coverage (ceiling M)

- **case / stage:** B, EXTEND (prefill).
- **observation:** vLLM's **2048-token prefill GEMMs are eager `aten::mm`** (not graph/compiled), unlike
  Case A/C — so the H1 dispatch lead is weak for Case B prefill (both eager). Case B is bimodal in both
  frameworks (Phase-2 CV 68%/86%).
- **source:** vLLM prefill `aten::mm`; SGLang prefill **unobserved** (EXTEND trace unavailable — see below).
- **evidence:** vLLM prefill nvjet 51%+27% eager; SGLang DECODE confirms same GEMM family.
- **catalog status:** n/a (falsifies H1 for Case B prefill).
- **impact:** ranking input — deprioritize Case B for kernel-level work. **confidence:** **M** (bimodal + inferred).
- **fairness dependence:** no.
- **caveats:** **SGLang Case B EXTEND trace is unavailable** (original gz corrupt; 3 Phase-4 re-collect
  attempts + prior 8 attempts failed — prefix-cache + `--profile-by-stage` long-prefill limit; full
  provenance in `caseB_longprefill/extend_triage.md`). Case B prefill-stage claims are vLLM-referenced only.
- **Phase 5 validation:** resolve Case B bimodality first (more warmup / mode isolation) before any
  kernel-level prefill claim; capturing a real SGLang long-prefill EXTEND needs `--disable-radix-cache`
  + a profiler fix (out of Phase-4 scope).

---

## Data gaps / ceilings carried into Phase 5

- **Case B SGLang EXTEND**: no usable trace (see H4 caveat). Prefill-stage timing for SGLang Case B is absent.
- **`scheduler / CPU gap`**: not measurable from the GPU-time kernel table; H1 needs a CPU-gap probe.
- **Attention ceiling M**: any FlashInfer-vs-FA3 kernel comparison.
- **Single representative trace per (framework, stage, case)** — not repeated reps; shares are stable but
  absolute times are one-window snapshots.
