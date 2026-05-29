# v2 / Round 2 — Image+text benchmarks (issue #4)

Clean Qwen3-VL **image+text** benchmarks with **CUDA IPC transport**. Tests whether the text-only Case-A
PCG finding (#2) transfers to the image path, and separates the **CUDA-IPC transport** benefit from the
**PCG** prefill-graph lever.

- **Issue:** #4 (parent #1) on `bowenwan6/sglang-vllm-profiler`. Builds on #2 (text-only, complete).
- **Status:** **protocol drafting / pending approval — no benchmark runs, no servers, no runner yet.**
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

## Open items (gate Phase 4.0, before any perf run)

1. **vLLM image anchor** — `sglang-oai-chat` against vLLM's chat endpoint with data-URI images is
   **unverified**; must smoke first.
2. **Length pinning** — confirm the `--random-range-ratio` value that fixes text length.
3. **IPC observability** — confirm `SGLANG_USE_CUDA_IPC_TRANSPORT=1` actually engages the transport path.

## Layout (created as phases proceed)

- `protocol.md` — this experiment's plan (exists).
- `run_image_text_benchmarks.py` — runner (**not implemented until Phase 4.0 confirms schema**).
- `results/` — future per-variant `results.json`, `summary.md`, and `raw/` per-rep dumps (raw not committed
  unless approved).
- server logs → `logs/qwen3vl8b/v2/image_text_benchmarks/` (not committed unless approved).
