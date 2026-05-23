# Case D — Preliminary Observations (NOT final recommendations)

Run: `run2_qwen3vl8b` · Case D `caseD_decode` (512→512, c=16, default).
Residual gap (Phase 2, W30): **SGLang 206.2 ms vs vLLM 189.7 ms ≈ 1.09×** (CV 3.3%, gate PASS).
Lowest-priority case — decode-heavy sanity check.

> Draft observations. Not validated; not Phase-5 recommendations.

---

## OBS-D1 — Smallest gap confirms the TTFT-fixed-overhead thesis (sanity check passes)

- **stage:** whole-run (decode-heavy)
- **observation:** Case D (512→**512** output) has the smallest residual gap (1.09×) of all cases. Its
  kernel profile is structurally identical to Case C (512 input, c=16): GEMM-bound (nvjet 72% extend /
  85% decode), attention small, PR #22392 catalog hit.
- **SGLang evidence:** decode 85.7% nvjet; extend 72.5% nvjet; same `unquant.py:138` site.
- **vLLM evidence:** same nvjet family; prefill in inductor-compiled regions, decode under cudaGraphLaunch.
- **catalog status:** PR #22392 (shared cost, not gap-closer).
- **impact estimate:** corroborating — the long 512-token decode amortizes the fixed first-token overhead,
  so the gap shrinks. Supports OBS-A1/C1 (the gap is a *first-token / dispatch fixed-overhead* effect, not
  a per-token decode deficit, since TPOT is at parity).
- **confidence (draft):** M.
- **fairness dependence:** no.
- **caveat:** none beyond the shared attention-backend ceiling M.
- **recommended Phase 5 / next action:** none — Case D is a sanity check; it adds no new actionable bottleneck.

## OBS-D2 — nvjet FP8 GEMM dominant (85.7% decode); PR #22392 absolute lead (shared)

- **stage:** EXTEND 72.5% / DECODE 85.7%
- **kernel / op:** nvjet FP8 family, `aten::mm`, `unquant.py:138`.
- **SGLang evidence:** fuse table PR #22392 Confirmed 72.5% / 85.7%.
- **vLLM evidence:** same family in vLLM → shared cost.
- **catalog status:** open-upstream PR #22392.
- **impact estimate:** M absolute; L for gap. **confidence (draft):** H attribution / L gap-closing.
- **fairness dependence:** no.
- **recommended Phase 5 / next action:** track PR #22392 as absolute-speed experiment, separate from the gap.

## OBS-D3 — Attention backend differs but is not the driver (ceiling M)

- vLLM decode FA3 13.1% vs SGLang FlashInfer 4.2% — both small; not the gap source. Fairness-dependent → M.
