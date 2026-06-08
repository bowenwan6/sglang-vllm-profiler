# Stage 4.1 Fixed-Generator Smoke Summary — image+text

> Run: 2026-06-08 15:25 UTC  GPU=7  seed=1  num_prompts=2  warmup=1  resolution=720p  range_ratio=1.0

> **Purpose:** confirm the fixed-generator path is wired up end-to-end. NO performance conclusions.

## Overall verdict: ✅ ALL PASS — fixed-generator path validated; safe to proceed to IMG-A

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
| smoke_sglang_ipc | OK | 2 | 0 | False | 1764 | 280 | 80.8823883999139 | True |
| smoke_sglang_noipc | OK | 2 | 0 | False | 1764 | 284 | 165.08410056121647 | True |
| smoke_vllm_anchor | OK | 2 | 0 | False | 1764 | 283 | 104.0576274972409 | True |

## Token composition (smoke, no perf weight)

- **smoke_sglang_ipc**: input=2044 (text=280 + vision=1764), output=64
- **smoke_sglang_noipc**: input=2048 (text=284 + vision=1764), output=64
- **smoke_vllm_anchor**: input=2047 (text=283 + vision=1764), output=64

## Sample outputs

**smoke_sglang_ipc** (status=OK  ipc=True):
  req1: `The text you've provided appears to be a random, nonsensical string of characters, symbols, and fragments — likely corru`
  req2: `The text you've provided appears to be a random string of characters, symbols, and words with no coherent meaning or str`
**smoke_sglang_noipc** (status=OK  ipc=False):
  req1: `The text you've provided appears to be a random, nonsensical string of characters, symbols, and words from multiple lang`
  req2: `The text you've provided appears to be a random, nonsensical string of characters, symbols, and words from multiple lang`
**smoke_vllm_anchor** (status=OK  ipc=False):
  req1: `It seems the text you've provided is either corrupted, encoded incorrectly, or contains a mix of random characters, symb`
  req2: `It seems the content you've provided is either corrupted, encoded incorrectly, or contains a mix of random characters, s`

## Stop condition check

✅ No stop conditions. Fixed-generator smoke pass — IMG-A is the next gated stage.