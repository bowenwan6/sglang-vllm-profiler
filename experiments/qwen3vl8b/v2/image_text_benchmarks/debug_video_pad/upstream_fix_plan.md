# Upstream fix plan — multimodal benchmark special-token filtering

> Status: local SGLang fix branch prepared and tested in `/data/sglang-pr`.
> Do not modify `/sgl-workspace/sglang`.

## 1. Problem

The SGLang synthetic image benchmark path can generate multimodal/control special tokens
inside random text prompts:

```bash
python -m sglang.bench_serving --dataset-name image
```

For Qwen3-VL, `gen_mm_prompt` currently removes `image_pad_id` from the random token
pool, but leaves other special ids such as `<|video_pad|>`, `<|vision_start|>`,
`<|vision_end|>`, and `<|vision_pad|>`. In an image-only request, a generated
`<|video_pad|>` is interpreted as a video placeholder without a video payload and the
server returns HTTP 400:

```text
No data iterator found for token: <|video_pad|>
```

## 2. Evidence

Profiler-side validation is complete:

- **V1 payload audit:** PASS. The stock generator emitted `<|video_pad|>` in
  8 / 12,900 prompts and emitted any forbidden token in 42 / 12,900 prompts. The
  sanitized generator emitted 0 / 12,900 forbidden tokens. Artifact:
  `results/D0_payload_audit.md`.
- **V2 serving repro:** PASS. An image-only request containing `<|video_pad|>` returned
  HTTP 400, while the safe-text control returned HTTP 200. Artifact:
  `results/V2_serving_repro.md`.

This is a benchmark-generator correctness bug, not an IPC, PCG, cache, or performance
result.

## 3. Working tree policy

Use a fresh clone:

```bash
cd /data
git clone git@github.com:bowenwan6/sglang.git sglang-pr
cd /data/sglang-pr
git remote add upstream https://github.com/sgl-project/sglang.git
git fetch upstream
git checkout -b fix/mm-benchmark-special-tokens upstream/main
```

If `/data/sglang-pr` already exists, verify it is clean and disposable before reuse. Do
not modify `/sgl-workspace/sglang`.

## 4. Fix design

Target file:

```text
python/sglang/benchmark/datasets/common.py
```

Keep the public call shape of `gen_mm_prompt(tokenizer, image_pad_id, token_num)` to avoid
touching callers. Internally, filter the random token pool by excluding:

1. `tokenizer.all_special_ids` (which includes `additional_special_tokens_ids` for
   Hugging Face tokenizers)
2. the existing `image_pad_id` argument, when present

The key behavior is: synthetic benchmark random text should be sampled from ordinary
text tokens only, not tokenizer control tokens.

Preferred implementation shape:

```python
def get_special_token_ids(tokenizer):
    special_token_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    return special_token_ids
```

Then use it inside `gen_mm_prompt` to filter `all_available_tokens`.

## 5. Tests

Target test file:

```text
test/registered/bench_fn/test_benchmark_datasets_api.py
```

Add a deterministic CPU test near `test_image_sampler`:

1. Create a lightweight tokenizer with ordinary tokens and added multimodal special
   tokens (`<|image_pad|>`, `<|video_pad|>`, `<|vision_start|>`, `<|vision_end|>`,
   `<|vision_pad|>`).
2. Patch `random.choices` so the test can inspect the token pool passed to sampling.
3. Call `gen_mm_prompt(tokenizer, image_pad_id, token_num)`.
4. Assert none of the special ids appear in the sampled token pool.
5. Assert ordinary token ids remain available.

This avoids probabilistic tests and does not require a model download or GPU.

## 6. Validation commands

Run targeted tests first:

```bash
cd /data/sglang-pr
python -m unittest \
  test.registered.bench_fn.test_benchmark_datasets_api.TestBenchmarkDatasetsAPI.test_gen_mm_prompt_excludes_special_tokens
```

Then run the full dataset API test file:

```bash
python -m unittest test.registered.bench_fn.test_benchmark_datasets_api
```

Optional profiler-side verification, using the patched clone on `PYTHONPATH`:

```bash
PYTHONPATH=/data/sglang-pr/python:$PYTHONPATH \
python /data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/debug_payload_audit.py
```

Expected patched result: 0 forbidden-token hits from the patched generator.

## 7. Commit and PR

Commit in the SGLang clone with the same attribution discipline:

```bash
git add python/sglang/benchmark/datasets/common.py \
  test/registered/bench_fn/test_benchmark_datasets_api.py
git commit -m "fix(benchmark): exclude special tokens from multimodal prompts"
git push origin fix/mm-benchmark-special-tokens
```

Suggested PR title:

```text
Fix multimodal synthetic benchmark prompt generation to exclude special tokens
```

PR summary should include:

- Problem: `gen_mm_prompt` only excluded `image_pad_id`.
- Impact: Qwen3-VL image benchmark can randomly generate `<|video_pad|>` and fail
  image-only requests.
- Fix: exclude tokenizer special/control token ids from multimodal synthetic random
  text.
- Tests: deterministic unit test plus local payload audit / serving repro from the
  profiler repo.

## 8. After PR branch validation

Once the branch is locally validated, return to the profiler repo and update issue #4.
After the upstream fix is merged or the local patched clone is selected for experiments,
resume:

1. sanitized or patched image smoke,
2. formal IMG-A,
3. then IMG-B/IMG-C only if IMG-A is clean.

## 9. Local implementation status

Prepared on 2026-05-31 in a clean clone:

```text
/data/sglang-pr
branch: fix/mm-benchmark-special-tokens
commit: e384fe215 fix(benchmark): exclude special tokens from multimodal prompts
```

Implementation:

- `python/sglang/benchmark/datasets/common.py`
  - added `get_available_multimodal_text_tokens`
  - `gen_mm_prompt` now filters `tokenizer.all_special_ids` plus the existing
    `image_pad_id` argument
- `test/registered/bench_fn/test_benchmark_datasets_api.py`
  - added `test_gen_mm_prompt_excludes_special_tokens`
  - deterministic test patches `random.choices` and asserts multimodal special ids are
    absent from the sampled token pool

Validation:

```text
PYTHONPATH=/data/sglang-pr/python python test/registered/bench_fn/test_benchmark_datasets_api.py TestBenchmarkDatasetsAPI.test_gen_mm_prompt_excludes_special_tokens
PASS

PYTHONPATH=/data/sglang-pr/python python test/registered/bench_fn/test_benchmark_datasets_api.py
PASS — 32 tests
```

Push/PR status:

- The initial intended fork remote `git@github.com:bowenwan6/sglang.git` was not
  accessible (`Repository not found`), so the working clone currently tracks
  `https://github.com/sgl-project/sglang.git`.
- Next step for PR: create or grant access to the `bowenwan6/sglang` fork, add it as a
  writable remote, push `fix/mm-benchmark-special-tokens`, then open the upstream PR.
