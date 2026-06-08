# v2 / Round 2 — Image+text benchmarks (issue #4)

Clean Qwen3-VL **image+text** benchmarks with **CUDA IPC transport**. Tests whether the text-only Case-A
PCG finding (#2) transfers to the image path, and separates the **CUDA-IPC transport** benefit from the
**PCG** prefill-graph lever.

- **Issue:** #4 (parent #1) on `bowenwan6/sglang-vllm-profiler`. Builds on #2 (text-only, complete).
- **Status:** ⚠️ **PARTIAL — generator unblocked, PCG path blocked by upstream
  capture-stream assertion.** The benchmark generator `<|video_pad|>` bug is **merged
  upstream** as commit `07f326c184 Fix multimodal synthetic benchmark prompt generation
  to exclude special tokens (#26864)`. Profiler runs source the fix from
  `/data/sglang-pr` on `main` (HEAD `62c505a196` after `git pull` on 2026-06-08),
  selected via `PYTHONPATH=/data/sglang-pr/python`. V1 audit + V2 serving repro both
  PASS (see [`debug_video_pad/validation_plan.md`](debug_video_pad/validation_plan.md));
  Stage 4.1 smoke PASS (see [`smoke_fixed/smoke_summary.md`](smoke_fixed/smoke_summary.md));
  Stage 4.2 IMG-A **PARTIAL**:

  - `IMG_A_S0_ipc` ✅ 5/5 reps, 2000 requests, 0 failures, TTFT p50 64.8 ms.
  - `IMG_A_S2_ipc_pcg` ❌ rep1 server crash with
    `AssertionError: PCG capture stream is not set` in
    `srt/compilation/cuda_piecewise_backend.py:171`.
  - `IMG_A_S0_ipc_repeat` / `IMG_A_V0_vllm` / `IMG_A_S0_noipc` skipped per protocol §9.

  See [`results_fixed/imgA_summary.md`](results_fixed/imgA_summary.md) for partial
  numbers. PCG crash debug closed (2026-06-08):
  [`debug_pcg_capture_stream/conclusion.md`](debug_pcg_capture_stream/conclusion.md)
  — E1 text-only + PCG `OK`; E2a image + IPC + PCG @ n=32 ASSERT; E3 image + **no
  IPC** + PCG @ n=32 ASSERT → fault is **VLM image + PCG**, IPC not required.
  Upstream auto-disables PCG for VLMs and `--enforce-piecewise-cuda-graph` is a
  "for testing" override; the assertion is a defensive guard. **PCG benefit (Q2)
  cannot be measured on this HEAD without an upstream change.** Next step is to
  resume IMG-A with the non-PCG variants only
  (`S0_ipc_repeat → V0_vllm → S0_noipc`) to recover Q1/Q3 + bracket drift.
  `S2_ipc_pcg` stays explicitly excluded. Recovery plan:
  [`fixed_generator_plan.md`](fixed_generator_plan.md) resumes at Stage 4.2's
  non-PCG variants.
  Original protocol unchanged at [`protocol.md`](protocol.md); see
  [`debug_video_pad/upstream_fix_plan.md`](debug_video_pad/upstream_fix_plan.md)
  for the merged-fix history.
- **Prior pre-fix partial IMG-A is INVALID for performance conclusions** — it ran
  only 3 of 5 reps of one of five variants under the buggy generator. Kept in
  [`results/imgA_summary.md`](results/imgA_summary.md) as historical record only.
- **Do NOT proceed to IMG-B / IMG-C** until IMG-A yields headline-quality data or
  the PCG path is explicitly excluded with a documented rationale.
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

## Resolved: video_pad correctness blocker

`gen_mm_prompt` previously excluded only `image_pad_id`, leaving `<|video_pad|>`
(151656) and other multimodal/control tokens in the Qwen3-VL random pool
(~0.084% per 128-token prompt → P(≥1 failure in 5 reps) ≈ 83%). The fix is **merged
upstream** as commit `07f326c184 Fix multimodal synthetic benchmark prompt generation
to exclude special tokens (#26864)`. It excludes `tokenizer.all_special_ids` from the
multimodal random pool. V1 audit PASS, V2 serving repro PASS. Profiler runs source
the fix from `/data/sglang-pr` on `main` (HEAD `62c505a196` after pull on 2026-06-08),
selected at runtime via `PYTHONPATH=/data/sglang-pr/python`. Recovery proceeds via
[`fixed_generator_plan.md`](fixed_generator_plan.md).

## Layout

- `protocol.md` — original #4 protocol (unchanged source of truth).
- `fixed_generator_plan.md` — **active** recovery plan for the fixed generator.
- `run_image_text_smoke.py`, `run_image_text_imgA.py` — original runners; partial
  IMG-A from these is invalidated.
- `bench_serving_sanitized.py`, `run_image_text_smoke_sanitized.py`,
  `run_image_text_imgA_sanitized.py` — sanitized monkeypatch fallback (only if
  the fixed clone is unavailable).
- `smoke_fixed/` — Stage 4.1 fixed-generator smoke outputs (future).
- `results_fixed/` — Stage 4.2+ fixed-generator IMG-A/B/C outputs (future);
  raw per-rep dumps under `results_fixed/raw/` not committed unless approved.
- `results/` — historical (invalidated partial IMG-A); not overwritten.
- `smoke/` — original 2026-05-30 smoke; not overwritten.
- server logs → `logs/qwen3vl8b/v2/image_text_benchmarks/{smoke_fixed,results_fixed}/`
  (not committed unless approved).
