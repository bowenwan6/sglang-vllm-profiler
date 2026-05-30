# Audit notes — `No data iterator found for token: <|video_pad|>`

> Read-only source audit. No experiments run. No SGLang source modified.
> Performed: 2026-05-30. GPU: 7. Failure context: IMG-A S0_ipc rep3.

---

## 1. Error location

**File:** `sglang/srt/multimodal/processors/base_processor.py:610`
```python
raise ValueError(f"No data iterator found for token: {text_part}")
```
Call stack: `legacy_load_mm_data` → `submit_data_loading_tasks`.

`submit_data_loading_tasks` iterates over `text_parts` (the input prompt split on
multimodal special-token regexes). For each `text_part` that `get_modality_of_token`
classifies as a known modality, it looks for the corresponding `data_iterator`. If
there is none, it raises the error.

For Qwen3-VL with an image-only request: `data_iterators` has only
`Modality.IMAGE`. If `text_parts` contains `<|video_pad|>`, the code looks for
`Modality.VIDEO` iterator → not found → error.

---

## 2. How `<|video_pad|>` ends up in the split prompt

### 2a. Token identities

| token | token ID | excluded from `gen_mm_prompt` pool? |
|---|---|---|
| `<\|image_pad\|>` | 151655 | ✅ yes (`image_pad_id` removed) |
| `<\|video_pad\|>` | 151656 | ❌ **no** |

`gen_mm_prompt` in `sglang/benchmark/datasets/common.py` only removes `image_pad_id`
from the random token pool. All other special tokens — including `video_pad_id=151656`
— remain in the pool.

### 2b. Pool size and per-request probability

```
pool size            : 151,668 tokens (all Qwen3-VL vocab minus image_pad)
P(video_pad in 1 slot): 1 / 151,668 ≈ 0.00066%
P(video_pad in 128-token prompt): 1 − (1 − 1/151668)^128 ≈ 0.084%  ≈ 1 in 1185 requests
Expected failures per 430 requests (30 warmup + 400 measured): 0.36
```

**Empirical confirmation (50 seeds × 430 prompts = 21,500 total):**
- 17 prompts contained `<|video_pad|>` → **0.079% per prompt** (matches theoretical 0.084%)
- P(at least 1 failure in 5 reps) ≈ 83%

The benchmark **cannot** reach 5 clean reps with the current `gen_mm_prompt` for
Qwen3-VL.

### 2c. Server-side path: why the error fires

When the encoded request arrives at `tokenizer_manager.py`, the flow is:
1. `serving_chat.py` tokenizes the chat message via Qwen3-VL chat template →
   `prompt_ids` (list of ints)
2. `process_mm_data_async(input_text=input_ids, ...)` is called
3. `load_mm_data` decodes `input_ids` back to string via
   `self._tokenizer.decode(prompt)`
4. If the decoded string contains `<|video_pad|>` as a token string, the
   `combined_regex` split captures it as a `text_part`
5. `get_modality_of_token` classifies it as `Modality.VIDEO` (video_token_regex
   built from `<|video_pad|>`)
6. `cnt[VIDEO]=1`, `n_video=0` → condition
   `cnt[Modality.VIDEO] != n_video` fires → **legacy path**
7. `legacy_load_mm_data` builds `data_iterators` without `Modality.VIDEO` →
   `submit_data_loading_tasks` raises the error

### 2d. Why rep1 / rep2 didn't fail

**The dataset is non-deterministic between subprocess invocations despite the same
`--seed`.** Verified by calling `sample_image_requests(seed=1)` three times in the
same process:

| run | prompt_lens[0:5] | vision_lens[0:5] |
|---|---|---|
| 1 | [1026, 1020, 1027, 1025, 1025] | [882, 882, 882, 882, 882] |
| 2 | [1024, 1021, 1022, 1024, 1024] | [882, 882, 882, 882, 882] |
| 3 | [1023, 1022, 1032, 1024, 1029] | [882, 882, 882, 882, 882] |

Vision tokens (882/request) are stable; text token counts vary. Possible causes:
Unicode tokenization artifacts when random token IDs are decoded and re-embedded in
the chat template, or processor-internal state changing between calls. The **exact
cause of non-determinism is itself an open question** (see D0 below) but does not
change the primary root cause.

Because each subprocess (rep) generates a **fresh** pseudo-random dataset, each rep
independently has a ~0.36 expected video_pad-containing prompt count. Rep1 and rep2
happened to draw 0; rep3 drew 2 (at measured-window indices 44 and 185).

---

## 3. RadixCache relevance

**Not directly the cause.** RadixCache caches KV by token-sequence prefix. For
image requests, all prompts share the same system-prompt prefix and the same
`<|image_pad|>×882` sequence (same token IDs regardless of image content). The
RadixCache hit at ~885 tokens (seen in server log: `#cached-token: 885`) is
therefore expected and benign. RadixCache does not introduce `<|video_pad|>` into
the prompt; it only caches/reuses KV for the shared prefix.

A `--disable-radix-cache` experiment (Stage D5) can confirm, but is low priority.

---

## 4. `SGLANG_USE_CUDA_IPC_TRANSPORT=1` relevance

**Unlikely the primary cause.** IPC transport controls how image feature tensors
move from the tokenizer worker to the scheduler (`base_processor.py` lines 1186–
1239, `MmItemMemoryPool`). It does not change which tokens appear in the prompt
string, and therefore does not change the `cnt[VIDEO]` comparison in `load_mm_data`.
The no-IPC control (Stage D4) will confirm.

---

## 5. Available server-side controls

| mechanism | flag / endpoint | confirmed available |
|---|---|---|
| RadixCache disable | `--disable-radix-cache` (server arg) | ✅ `server_args.py:677` |
| Cache flush | `POST /flush_cache` HTTP endpoint | ✅ `http_server.py:761` |
| Language-only mode | `--language-only` | ✅ `server_args.py:780` |

---

## 6. Raw data cross-check

| rep | idx 44 error | idx 185 error | input_lens[44] | input_lens[185] |
|---|---|---|---|---|
| 1 | (none) | (none) | 1023 | 1025 |
| 2 | (none) | (none) | 1023 | 1032 |
| 3 | `No data iterator for <\|video_pad\|>` | same | 1024 | 1027 |

`input_lens` differ between reps at the same index, confirming that rep3's
prompts at those positions are distinct from rep1/rep2. Note that `input_lens`
in the JSONL equals `DatasetRow.prompt_len` = text + vision tokens (≈ 139 text
+ 882 vision ≈ 1021). The video_pad token adds 1 token to the text count.

---

## 7. Root cause summary

**Primary cause:** `gen_mm_prompt` (`sglang/benchmark/datasets/common.py`) excludes
only `image_pad_id` from the random token pool. `video_pad_id` (151656) remains
in the pool and is picked occasionally (~0.084%/prompt for Qwen3-VL). The decoded
prompt string then contains `<|video_pad|>`, which the multimodal preprocessor
correctly classifies as a video token. Since no video data was provided, the
server returns HTTP 400.

**Contributing factor:** Dataset generation is not deterministic between subprocess
invocations (same seed ≠ same output). This makes failures appear intermittently
rather than systematically across reps.

**Not the cause:** Server state accumulation, RadixCache corruption, IPC transport,
or overlap schedule.

---

## 8. Fix path

**Upstream fix (preferred):** In `gen_mm_prompt`, exclude **all** multimodal
special tokens (all token IDs whose decoded string matches any multimodal token
regex), not just `image_pad_id`. Candidate PR: add `all_special_ids` exclusion
and/or explicitly exclude `video_pad_id`.

**Workaround without source modification:** Run a pre-check script that generates
the batch from a candidate seed and verifies 0 video_pad-containing prompts before
launching the server, trying successive seeds until clean. Since generation is
non-deterministic, a seed is no guarantee; the pre-check must verify at runtime.

See `debug_plan.md` for staged experiments.
