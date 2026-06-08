# Stage 4.1 Fixed-Generator Smoke Summary — image+text

> Run: 2026-06-08 14:59 UTC  GPU=7  seed=1  num_prompts=2  warmup=1  resolution=720p  range_ratio=1.0

> **Purpose:** confirm the fixed-generator path is wired up end-to-end. NO performance conclusions.

## Overall verdict: ❌ FAILURES — do NOT proceed to IMG-A

## Fixed-SGLang provenance

- `/data/sglang-pr` HEAD SHA: `62c505a196fd5bc997f478b0a7c6403ce655a838`
- `/data/sglang-pr` branch: `main`
- Merged fix `07f326c184` in history: **True**
- `sglang.__file__`: `/data/sglang-pr/python/sglang/__init__.py`
- `sglang.benchmark.datasets.common.__file__`: `/data/sglang-pr/python/sglang/benchmark/datasets/common.py`
- Fix marker (`get_available_multimodal_text_tokens` in `gen_mm_prompt`): `FIX_OK`
- Fixed-path import gate: **True**

## Per-case results

| case | status | completed | failures | forbidden_token_err | vision_tok | text_tok | median_ttft_ms | output_non_empty |
|---|---|---|---|---|---|---|---|---|
| smoke_sglang_ipc | SERVER_NO_START | ? | ? | ? | ? | ? | ? | ? |

## Token composition (smoke, no perf weight)


## Sample outputs

**smoke_sglang_ipc** (status=SERVER_NO_START  ipc=True):

## Stop condition check

⚠️ STOP CONDITIONS TRIGGERED — do NOT proceed to IMG-A:
- smoke_sglang_ipc: SERVER_NO_START