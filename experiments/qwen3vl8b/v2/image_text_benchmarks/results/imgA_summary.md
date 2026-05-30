# IMG-A Benchmark Summary — image+text (c=1)

> Run: 2026-05-30 21:21 UTC  GPU=7  seed=1  n=400  warmup=30  reps=5  resolution=720p  range_ratio=1.0

> SGLang image headline baseline: `SGLANG_USE_CUDA_IPC_TRANSPORT=1` (IPC on).

> IPC benefit and PCG benefit reported separately.

> vLLM is anchor only — no causal inference.

> **Image+text conclusions are separate from text-only Issue #2 findings.**

> ⚠️ **STATUS: INVALID / INCOMPLETE.** This run was halted by a benchmark-generator
> correctness bug (see "Corrected diagnosis" below). No headline numbers are produced.
> A sanitized rerun supersedes this file.

## Headline numbers (TTFT p50, median of reps)

| variant | ipc | pcg | ttft_p50 median (ms) | CV% | tpot_p50 (ms) | out_tok/s | status |
|---|---|---|---|---|---|---|---|
| IMG_A_S0_ipc | on | off | FAIL | ?% | ? | ? | INVALID_FAILURES |

(S2_ipc_pcg, S0_ipc_repeat, V0_vllm, S0_noipc were never reached — the runner stops on first failure.)

## Bracket drift (S0_ipc vs S0_ipc_repeat)

⚠ One or both bracket variants failed — drift cannot be assessed.

## PCG benefit (Q2): S0_ipc vs S2_ipc_pcg

⚠ S0_ipc or S2_ipc_pcg failed — PCG benefit cannot be assessed.

## IPC benefit (Q3): S0_noipc vs S0_ipc

⚠ S0_ipc or S0_noipc failed — IPC benefit cannot be assessed.

## SGLang IPC baseline vs vLLM anchor (Q1)

⚠ S0_ipc or V0_vllm failed.

## Token composition (per request)

Vision tokens: 882/req  |  Text tokens: ~139/req  |  Resolution: 720p, image-count=1, range_ratio=1.0, seed=1

## Failure summary

❌ 1 variant with issues:
  - IMG_A_S0_ipc: status=INVALID_FAILURES (rep3: 2/400 failures, HTTP 400 `<|video_pad|>`)

## Failure diagnosis

**Error:** `"No data iterator found for token: <|video_pad|>"` (SGLang HTTP 400, `BadRequestError`).

**Pattern:** 2 failures at measured-window indices 44 and 185 in rep3 only. Rep1 and rep2 completed 400/400 with 0 failures.

> ⚠️ **SUPERSEDED HYPOTHESIS (do not rely on):** The original write-up attributed this
> to *server-side cache state accumulation* (RadixCache / image cache after ~800
> requests). **This is now known to be wrong.** It also incorrectly assumed all reps
> use the identical seeded prompt set; in fact each `bench_serving` subprocess
> regenerates its own random text, so reps do **not** share prompts. The corrected
> diagnosis is below.

### ✅ Corrected diagnosis — benchmark-generator special-token bug (not perf, not cache)

This is a **correctness bug in the SGLang benchmark generator**, not a server
cache/state or CUDA-IPC performance finding.

- `sglang.bench_serving --dataset-name image` synthesizes random text via
  `gen_mm_prompt` in `sglang/benchmark/datasets/common.py`.
- `image.py` passes only `processor.image_token_id` into `gen_mm_prompt`, which
  removes **only** `<|image_pad|>` (151655) from the random token pool.
- **`<|video_pad|>` (151656) and other multimodal control tokens
  (`<|vision_start|>`, `<|vision_end|>`, …) are NOT excluded.** The random text can
  therefore contain `<|video_pad|>`.
- Qwen3-VL's chat-template + multimodal preprocessor treat `<|video_pad|>` as a
  **video placeholder**. Since the request supplies an image but no video, the
  server returns HTTP 400 `No data iterator found for token: <|video_pad|>`.
- **Probability:** ~0.084% per 128-token prompt (empirically 0.079% over 21,500
  prompts). E[failures] ≈ 0.36 per 430-request rep; P(≥1 failure in 5 reps) ≈ 83%.
  This is why the failures appear intermittently, not from cache buildup.

This is directly analogous to the historical text-only issue where random benchmark
prompts had to be sanitized to avoid multimodal/control special tokens.

**Corrective action (chosen):** Sanitize the generated prompts so they contain no
multimodal special/control tokens. Implemented in this repo as a runtime
monkeypatch wrapper around `bench_serving` (no SGLang source modification). An
upstream fix to `gen_mm_prompt` is a **follow-up**, not a prerequisite to finishing
#4. See `../debug_video_pad/audit_notes.md` and `../debug_video_pad/debug_plan.md`.

## Partial data (informational only — NOT headline)

| rep | completed | failures | ttft_p50 (ms) | tpot_p50 (ms) |
|---|---|---|---|---|
| 1 | 400 | 0 | 87.06 | 5.20 |
| 2 | 400 | 0 | 61.81 | 5.20 |
| 3 (stopped) | 398 | **2** | 60.69 | 5.20 |

Note: rep1 TTFT (87ms) > rep2 (62ms) suggests 30 warmup requests may be insufficient
to fully settle the 720p image pipeline. Secondary finding for warmup tuning.

**Do NOT use these partial numbers as headline.** Headline requires the sanitized
rerun with 5 clean reps per variant.

## Recommendation

IMG-A incomplete due to the benchmark-generator special-token bug. Proceed with the
**sanitized** runner (`run_image_text_imgA_sanitized.py`). **Do not proceed to
IMG-B / IMG-C** until the sanitized IMG-A passes with 0 failures.
