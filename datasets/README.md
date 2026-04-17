# datasets/

Canonical `autobench` JSONL datasets. Produced by `python -m sglang.auto_benchmark convert` and validated by `... validate`. **Never regenerated mid-project** — byte identity across Phase 1 / 2 / 5 is required.

## Files

| File | Case | Prompt len | Output len | Concurrency | SHA-256 |
|---|---|---|---|---|---|
| `caseA_short.jsonl` | A: latency-bound short | 128 | 128 | 1 | (record on generation) |
| `caseB_long_prefill.jsonl` | B: latency-bound long-prefill | 2048 | 128 | 1 | (record) |
| `caseC_batched.jsonl` | C: batched throughput | 512 | 128 | 16 | (record) |
| `caseD_decode_heavy.jsonl` | D: decode-heavy | 512 | 512 | 16 | (record) |

## Generation command

```
python -m sglang.auto_benchmark convert \
  --input-format random --prompt-len <P> --output-len <O> \
  --num-prompts 400 --output datasets/<case>.jsonl
python -m sglang.auto_benchmark validate --input datasets/<case>.jsonl
sha256sum datasets/<case>.jsonl
```
