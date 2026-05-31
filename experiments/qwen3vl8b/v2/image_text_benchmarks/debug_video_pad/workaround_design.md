# Workaround design — sanitize image+text benchmark prompts

> Goal: run formal IMG-A with prompts that contain **no** multimodal special/control
> tokens, **without modifying SGLang source**. Decision record for Step 4.

## Constraint

- Do **not** edit `/sgl-workspace/sglang` source.
- Prefer controlled/sanitized prompts produced inside this repo.
- Do **not** rely on seed-search alone unless the exact same prompt set is provably
  reused by every `bench_serving` subprocess.

## Option survey

### Option A — `bench_serving` custom/fixed image+text dataset (rejected)

`sglang.bench_serving --dataset-name image` is **synthesis-only**: `ImageDataset`
generates random text (`gen_mm_prompt`) and random pixels inline. There is no
`--dataset-path` hook for the image dataset to inject our own prompt strings or
image data. So we cannot feed sanitized prompts through a built-in file dataset
without code.

### Option B — seed search only (rejected as sole method)

Pick a seed whose generated batch has no forbidden token. **Rejected as the sole
mechanism** because generation is **not** reproducible across subprocess
invocations: calling `sample_image_requests(seed=1)` repeatedly yields different
`prompt_lens` (verified in audit). A seed that passes a pre-flight check in one
process is **not guaranteed** to produce the same prompts in the `bench_serving`
subprocess. So seed search cannot *prove* a clean run.

### Option C — runtime monkeypatch wrapper (CHOSEN)

Run `bench_serving` through a thin wrapper that **replaces the benchmark's
`gen_mm_prompt`** with a sanitized version before the dataset is generated. This
modifies only the in-memory function binding of the benchmark module **at runtime**;
no file under `/sgl-workspace/sglang` is changed.

Why it is correct and robust:

- `image.py` does `from ...common import gen_mm_prompt`, and `sample_image_requests`
  calls the **bare name** `gen_mm_prompt(...)`, resolved in the `image` module's
  globals at call time (during `get_dataset`, long after import).
- Therefore rebinding `sglang.benchmark.datasets.image.gen_mm_prompt` to a sanitized
  function makes **every** generated prompt sanitized — deterministically, for all
  requests, regardless of seed reproducibility.
- The wrapper then hands control to the real CLI via
  `runpy.run_module("sglang.bench_serving", run_name="__main__")`, so all argument
  parsing, dataset loading, request dispatch, and result writing are stock behavior.

## Sanitization rule

Replace `gen_mm_prompt(tokenizer, image_pad_id, token_num)` with a version that
excludes **all** special/control token ids from the random pool:

```python
exclude = set(tokenizer.all_special_ids)          # 14 ids 151643..151656 for Qwen3-VL
exclude |= set(tokenizer.get_added_vocab().values())  # remaining control tokens
pool = [t for t in tokenizer.get_vocab().values() if t not in exclude]
selected = random.choices(pool, k=token_num)
return tokenizer.decode(selected)
```

`all_special_ids` covers `<|image_pad|>`, `<|video_pad|>`, `<|vision_start|>`,
`<|vision_end|>`, `<|vision_pad|>`, `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`,
etc.; `get_added_vocab()` additionally covers `<tool_call>`, `<think>`, `<|fim_*|>`.

## Defense-in-depth: post-generation assertion

In addition to the sanitized generator, the runner performs a **special-token audit
pass**: a pre-flight that generates the batch through the patched path and asserts no
prompt contains any of `<|image_pad|>`, `<|video_pad|>`, `<|vision_start|>`,
`<|vision_end|>`, `<|vision_pad|>`. If any appears, the run aborts before launching a
server.

## What is NOT done

- No `/sgl-workspace/sglang` source edit (if that were ever the only path, stop and
  ask first — per Step 4 instruction).
- No change to image resolution / count / concurrency / output length / variant set.
- IPC and PCG levers are untouched; this only fixes prompt content.
