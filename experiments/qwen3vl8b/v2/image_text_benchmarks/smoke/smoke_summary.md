# Phase 4.0 Smoke Summary — image+text

> Run: 2026-05-30 20:56 UTC  GPU=7  seed=1  num_prompts=2  warmup=1  resolution=720p  range_ratio=1.0

> **Purpose:** schema / path verification only — NO performance conclusions.

## Overall verdict: ✅ ALL PASS — safe to proceed to IMG-A

## Per-case results

| case | status | completed | failures | vision_tok | text_tok | median_ttft_ms | output_non_empty |
|---|---|---|---|---|---|---|---|
| smoke_sglang_ipc | OK | 2 | 0 | 1764 | 279 | 204.94718599366024 | True |
| smoke_sglang_noipc | OK | 2 | 0 | 1764 | 279 | 124.87074144883081 | True |
| smoke_vllm_anchor | OK | 2 | 0 | 1764 | 285 | 99.81352143222466 | True |

## Phase 4.0 open items resolution

**1. vLLM image anchor** (`sglang-oai-chat` → vLLM `/v1/chat/completions` with data-URI images):
  ✅ RESOLVED — works. completed=2, failures=0, non-empty_output=True, vision_tok=1764

**2. Text length pinning** (`--random-range-ratio 1.0`):
  ✅ range_ratio=1.0 confirmed. total_text_tok=279 over 2 requests (avg 139.5 tok/req; includes chat-template overhead over 128 raw text tokens). Use `--random-range-ratio 1.0` in IMG-A/B to pin text length.

**3. CUDA IPC transport** (`SGLANG_USE_CUDA_IPC_TRANSPORT=1` observability):
  ✅ Both IPC-on and IPC-off paths ran cleanly. Env var accepted by SGLang server (no error). Direct engagement verification requires checking server log for IPC-related init lines (grep the server log for 'ipc' / 'transport'). IPC-on completed=2, noipc completed=2.

## Token composition (smoke, no perf weight)

- **smoke_sglang_ipc**: input=2043 (text=279 + vision=1764), output=64
- **smoke_sglang_noipc**: input=2043 (text=279 + vision=1764), output=64
- **smoke_vllm_anchor**: input=2049 (text=285 + vision=1764), output=64

## Sample outputs

**smoke_sglang_ipc** (status=OK  ipc=True):
  req1: `The text you've provided appears to be a random, nonsensical string of characters, symbols, and words — likely generated`
  req2: `It seems like your message is a mix of random characters, symbols, and fragmented words — possibly due to a corrupted in`
**smoke_sglang_noipc** (status=OK  ipc=False):
  req1: `The text you've provided appears to be a random, nonsensical string of characters, symbols, and words from multiple lang`
  req2: `The image you've provided appears to be corrupted or filled with noise — it's not a clear image, but rather a mosaic of `
**smoke_vllm_anchor** (status=OK  ipc=False):
  req1: `The text you've provided appears to be a random string of characters, symbols, and words — many of which are nonsensical`
  req2: `The image you've provided appears to be corrupted or entirely noise — it's filled with static, pixelation, and multicolo`

## Stop condition check

✅ No stop conditions. Smoke pass — implement IMG-A runner next.