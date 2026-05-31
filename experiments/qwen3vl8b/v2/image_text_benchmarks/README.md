# v2 / Round 2 — Image+text benchmarks (issue #4)

Clean Qwen3-VL **image+text** benchmarks with **CUDA IPC transport**. Tests whether the text-only Case-A
PCG finding (#2) transfers to the image path, and separates the **CUDA-IPC transport** benefit from the
**PCG** prefill-graph lever.

- **Issue:** #4 (parent #1) on `bowenwan6/sglang-vllm-profiler`. Builds on #2 (text-only, complete).
- **Status:** ⚠️ **BLOCKED — `video_pad` correctness blocker. Formal IMG-A/B/C paused.**
  Smoke passed (2026-05-30). IMG-A S0_ipc rep3 hit HTTP 400
  `"No data iterator found for token: <|video_pad|>"` at 2/400 requests. Root cause
  identified and validated: `gen_mm_prompt` in `sglang/benchmark/datasets/common.py` does not exclude
  `video_pad_id` from the random token pool. V1 payload audit PASS and V2 tiny serving repro PASS are recorded
  in [`debug_video_pad/validation_plan.md`](debug_video_pad/validation_plan.md). V3 sanitized smoke is pending.
  Debug plan at
  [`debug_video_pad/debug_plan.md`](debug_video_pad/debug_plan.md);
  audit at [`debug_video_pad/audit_notes.md`](debug_video_pad/audit_notes.md).
- **Protocol:** [`protocol.md`](protocol.md) — decision-complete, gated Phases 4.0–4.5.
- **Model:** `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` (same as v1/#2; verify in env snapshot before runs).

Key design points (see protocol for detail):

- **SGLang image headline runs must set `SGLANG_USE_CUDA_IPC_TRANSPORT=1`.** `S0_noipc` is the ablation.
- **Two separate levers:** CUDA IPC = image-feature *transport*; PCG (`--enforce-piecewise-cuda-graph`) =
  prefill *graph coverage* (testing lever, not a production fix). Never conflated.
- **Synthetic image dataset** (`--dataset-name image`): images generated inline (base64), reproducible via
  `--seed`; no external downloads, no large checked-in assets. Dataset identity = harness commit + image
  params + seed (see [`../../../../datasets/qwen3vl8b/image_text/README.md`](../../../../datasets/qwen3vl8b/image_text/README.md)).
- **Both frameworks benchmarked via `--backend sglang-oai-chat`** (the image dataset rejects `--backend
  vllm`; the chat request function POSTs `image_url` data URIs to any `/v1/chat/completions`).
- **Clean only** — no KAPI, no profiler. Servers serialized.

Workloads: **IMG-A** (1 img + short text, c=1), **IMG-B** (1 img + medium text, c=1), **IMG-C** (1 img,
c=16 batched); optional **IMG-D** multi-image.

## Phase 4.0 open items — RESOLVED ✅ (smoke 2026-05-30)

1. **vLLM image anchor** — ✅ confirmed working via `sglang-oai-chat`.
2. **Length pinning** — ✅ `--random-range-ratio 1.0` pins text length.
3. **IPC observability** — ✅ env var accepted; both IPC-on and off paths smoke-clean.

## Active blocker — video_pad correctness (Phase 4.1+)

`gen_mm_prompt` does not exclude `video_pad_id` (151656) from the Qwen3-VL
random token pool. ~0.084% of 128-token prompts contain `<|video_pad|>`.
Expected failures per 430-request batch ≈ 0.36; P(≥1 failure in 5 reps) ≈ 83%.

**Resolution path:** `debug_video_pad/validation_plan.md` — V1 payload audit PASS and V2 serving repro PASS
are complete; V3 sanitized smoke is the next gate. The blocker must be resolved (or the sanitized
workaround approved) before resuming formal IMG-A/B/C.

## Layout (created as phases proceed)

- `protocol.md` — this experiment's plan (exists).
- `run_image_text_smoke.py`, `run_image_text_imgA.py` — original runners (IMG-A invalidated by blocker).
- `bench_serving_sanitized.py`, `run_image_text_smoke_sanitized.py`, `run_image_text_imgA_sanitized.py` —
  sanitized prompt path for validation/resume after approval.
- `results/` — future per-variant `results.json`, `summary.md`, and `raw/` per-rep dumps (raw not committed
  unless approved).
- server logs → `logs/qwen3vl8b/v2/image_text_benchmarks/` (not committed unless approved).
