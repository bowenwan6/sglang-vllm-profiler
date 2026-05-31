#!/usr/bin/env python3
"""
Sanitized wrapper around `sglang.bench_serving` (NO SGLang source modification).

Replaces the benchmark's random-text generator `gen_mm_prompt` with a version that
excludes ALL special/control token ids from the random pool, then hands control to
the stock `sglang.bench_serving` CLI. This prevents the synthetic image dataset from
ever emitting multimodal/control tokens such as `<|video_pad|>` that cause
HTTP 400 `No data iterator found for token: <|video_pad|>`.

How it works:
  - `image.py` calls the bare name `gen_mm_prompt` resolved in its own module globals
    at runtime (during dataset generation, well after import).
  - We rebind `sglang.benchmark.datasets.image.gen_mm_prompt` to a sanitized function
    BEFORE the dataset is generated. This is an in-memory monkeypatch; no file under
    /sgl-workspace/sglang is changed.
  - We then run the real CLI via runpy so all arg parsing / dispatch / output writing
    is stock behavior.

Usage: identical to `python3 -m sglang.bench_serving <args>`, e.g.
    python3 bench_serving_sanitized.py --backend sglang-oai-chat --dataset-name image ...
"""
import random
import runpy
import sys

import sglang.benchmark.datasets.image as image_mod

# cache exclusion set per tokenizer object id
_EXCLUDE_CACHE = {}


def _excluded_ids(tokenizer):
    key = id(tokenizer)
    excl = _EXCLUDE_CACHE.get(key)
    if excl is None:
        excl = set()
        try:
            excl |= set(tokenizer.all_special_ids)
        except Exception:
            pass
        try:
            excl |= set(tokenizer.get_added_vocab().values())
        except Exception:
            pass
        _EXCLUDE_CACHE[key] = excl
    return excl


def sanitized_gen_mm_prompt(tokenizer, image_pad_id, token_num):
    """Drop-in replacement for sglang.benchmark.datasets.common.gen_mm_prompt that
    excludes ALL special/control token ids (not just image_pad_id)."""
    excl = _excluded_ids(tokenizer)
    pool = [t for t in tokenizer.get_vocab().values() if t not in excl]
    selected = random.choices(pool, k=token_num)
    return tokenizer.decode(selected)


def main():
    image_mod.gen_mm_prompt = sanitized_gen_mm_prompt
    sys.argv = ["sglang.bench_serving"] + sys.argv[1:]
    runpy.run_module("sglang.bench_serving", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
