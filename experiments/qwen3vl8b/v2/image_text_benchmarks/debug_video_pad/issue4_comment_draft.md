# Draft — Issue #4 status comment (image+text + CUDA IPC)

> Draft for posting to GitHub issue #4. Not auto-posted. Plain status update.

## Summary

Phase 4.0 smoke **passed**; formal IMG-A was **blocked by a benchmark-generator
correctness bug**, not by a performance or server-state effect. Root cause is now
confirmed and a sanitized path is in place to continue #4.

## What passed (Phase 4.0 smoke, GPU 7, clean: no KAPI / no profiler)

- SGLang **with** CUDA IPC (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`): ✅ 0 failures
- SGLang **without** IPC: ✅ 0 failures
- vLLM anchor via `--backend sglang-oai-chat`: ✅ 0 failures
- Open items resolved: vLLM image anchor works; `--random-range-ratio 1.0` pins
  text length; IPC env var accepted on both paths.
- Token composition at 720p: ~882 vision tokens + ~139 text tokens per request.

## What blocked formal IMG-A

IMG-A halted at the first variant (`S0_ipc`) on rep 3 with 2/400 requests failing:

```
HTTP 400  No data iterator found for token: <|video_pad|>
```

### Root cause (confirmed, not a perf/cache finding)

`sglang.bench_serving --dataset-name image` builds random text via `gen_mm_prompt`
(`sglang/benchmark/datasets/common.py`). `image.py` passes only
`processor.image_token_id`, so the generator removes **only** `<|image_pad|>`
(151655) from the random token pool. Other multimodal/control special tokens —
notably `<|video_pad|>` (151656), plus `<|vision_start|>`, `<|vision_end|>`,
`<|vision_pad|>` — remain in the pool and occasionally land in the prompt text.

For an image-only request, Qwen3-VL's preprocessor treats `<|video_pad|>` as a
video placeholder; with no video data attached, the server returns HTTP 400.

This is the multimodal analogue of the earlier text-only issue where random
benchmark prompts had to avoid special tokens.

### Evidence (D0 payload audit, CPU-only, 50 seeds × 430 prompts = 21,500)

| generator | prompts with `<\|video_pad\|>` | prompts with any forbidden mm token |
|---|---|---|
| current `gen_mm_prompt` | 17 / 21,500 (0.079%) | 49 / 21,500 (0.228%) |
| sanitized (exclude all special ids) | 0 | **0** |

Expected ≈ 0.36 video-pad failures (≈ 0.98 any-forbidden) per 430-request rep ⇒
P(≥1 failure in 5 reps) is high. That matches the intermittent rep-3 failure.

The earlier "server-side cache accumulation" hypothesis was **wrong** and has been
superseded; reps do not share a prompt set (each `bench_serving` subprocess
regenerates its own random text).

## How we continue #4

We sanitize the generated prompts so they contain **no** multimodal special/control
tokens, via a runtime monkeypatch wrapper around `bench_serving` (it replaces the
benchmark's `gen_mm_prompt` with a version that excludes all special ids). **No
SGLang source is modified.** Workload shape is unchanged (720p, 1 image, c=1,
output 128; variants S0_ipc / S2_ipc_pcg / S0_ipc_repeat / V0_vllm / S0_noipc).

The CUDA-IPC and PCG questions for #4 are unaffected — this was purely a prompt
content problem.

## Follow-up (not required to finish #4)

Optional upstream fix for `gen_mm_prompt`: exclude all multimodal/special token ids
(e.g. `tokenizer.all_special_ids`) from the random pool, not just `image_pad_id`.
Optionally, the server could also return a clearer error when user text contains an
unsupported multimodal placeholder. We will finish #4 with the sanitized path and
leave the upstream PR/issue as a separate follow-up.
