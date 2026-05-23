# Case D — vLLM Cross-check (corroboration / falsification only)

Run: `run2_qwen3vl8b` · Case D `caseD_decode` (512→512, c=16). Framework: **vLLM 0.21.0** (reference).
Mode: single-trace per window. Raw: `vllm_prefill_raw.txt`, `vllm_decode_raw.txt`.

> vLLM = reference only; no vLLM recommendations. Attention comparisons carry **ceiling M**.

## vLLM prefill_like

Same nvjet FP8 family inside **torch.compile / inductor AOT `call` regions** (confirmed: rows split
`inductor_cache/…call` + `aten::mm`), FA3 attention present. Mirrors Case C prefill.

## vLLM decode_like — decode section (≥1%)

nvjet via **`cudaGraphLaunch`** (`gpu_model_runner.py:3568 _model_forward`): 29.2% + 15.7% + 9.3% + 6.4%
+ …; FA3 13.1%; lm_head 4.9%.

## Cross-check conclusions vs SGLang

1. **GEMM dominance shared** (SGLang decode 85.7% / EXTEND 72.5%; vLLM decode ~60%+ nvjet + 13% FA3,
   prefill nvjet-dominated). Same family → not the differentiator; PR #22392 = absolute lead only.
2. **Dispatch path differs as in Case C:** vLLM prefill GEMMs in inductor-compiled regions, decode under
   `cudaGraphLaunch`; SGLang eager `aten::mm`. Consistent with OBS-A1/OBS-C1 (dispatch/compile lead).
3. **Smallest gap (1.09×)** + decode-heavy: with 512 output tokens the run is dominated by many decode
   steps where TPOT is at parity, so the fixed TTFT dispatch overhead is amortized over a long decode →
   the relative gap shrinks. This **corroborates** that the SGLang gap is a *first-token / fixed-overhead*
   effect, not a per-token decode deficit. → strong sanity-check confirmation of the overall thesis.
4. Attention (ceiling M): FA3 13.1% in vLLM decode vs FlashInfer 4.2% in SGLang decode — share differs but
   both small; not the driver (and not apples-to-apples → M).
