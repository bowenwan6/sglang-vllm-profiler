Adds vLLM's `--goodput` option to `bench_serving`: a run reports how many requests per second met their latency SLOs, and what share did.

## Motivation

`bench_serving` reports throughput and latency percentiles, but not how much of the load met a latency target; that has to be computed by hand from `--output-details`. Goodput, the load served within its SLOs, is the metric SLO-aware serving work optimises and reports (e.g. [JITServe](https://arxiv.org/abs/2504.20068), NSDI '26), and vLLM's benchmark already has the option. #5495 proposed it for the old `bench_serving.py` and was closed as stale after a request to rebase.

## Modifications

- `--goodput KEY:VALUE ...` with keys `ttft`, `tpot`, `e2el` in milliseconds, e.g. `--goodput ttft:2000 tpot:100`. A request is good if it succeeded and met every listed SLO.
- The run prints, and writes to the result file, request goodput (good requests per second), SLO attainment (good / sent) and the attainment of each SLO alone. Without the flag the output is unchanged.
- The scoring is a new module, `python/sglang/benchmark/goodput.py`; `serving.py` gains 40 lines of wiring.
- A docs section and two unit cases.

```
---------------------Goodput----------------------
SLOs (ms):                               ttft:2000 tpot:100 e2el:20000
Request goodput (req/s):                 28.37
SLO attainment (%):                      63.48
TTFT SLO attainment (%):                 100.00
TPOT SLO attainment (%):                 86.72
E2EL SLO attainment (%):                 76.49
```

## Accuracy Tests

No model or server code is touched. For the metric: unit cases with hand-computed values, and on a live server (`Qwen/Qwen3-8B`, one H200) the printed values equal an independent recount of the per-request records in 8 runs over the `sglang-oai-chat` and `sglang` backends.

Not changed here: a streaming request that the server aborts inside a 200 response is still recorded as successful by the benchmark (#40881), so it can count as good.

## Speed Tests and Profiling

No inference path is touched. An example of the output, a load sweep on this branch (ShareGPT, `--goodput ttft:2000 tpot:100 e2el:20000`, 50 s of arrivals per row):

| request rate (req/s) | output tok/s | goodput (req/s) | SLO attainment |
|---|---|---|---|
| 20.5 | 3684 | 18.89 | 100.0 % |
| 32.8 | 5516 | 28.74 | 99.8 % |
| 40.9 | 6488 | 33.20 | 99.6 % |
| 51.2 | 7480 | 35.55 | 93.0 % |
| 65.5 | 8785 | 28.37 | 63.5 % |
| 81.9 | 9688 | 9.23 | 18.8 % |

Output throughput keeps rising while goodput peaks near 50 req/s. One run per row; just past the peak a 50-second run is noisy (repeated runs near 65 req/s ranged from 37 to 72 %). The block above is the 65.5 req/s row.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
