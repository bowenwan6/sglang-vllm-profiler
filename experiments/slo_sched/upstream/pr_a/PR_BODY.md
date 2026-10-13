## Motivation

`bench_serving` reports throughput and latency percentiles, but not how much of the load met a latency target. The per-request data is already collected; today the share of requests that met a target has to be computed by hand from `--output-details`.

This adds the `--goodput` option of vLLM's serving benchmark, with the same spelling, so command lines carry over. #5495 proposed it for the old `bench_serving.py` and was closed for inactivity after a request to rebase; this is a new implementation on `sglang.benchmark.serving`.

## Modifications

- `--goodput KEY:VALUE [KEY:VALUE ...]` with keys `ttft`, `tpot`, `e2el` and values in milliseconds, any subset. An unknown key, a value that is not a positive finite number and a repeated key are rejected before any request is sent.
- A request is *good* if it succeeded and met every listed SLO, so a failed request is a miss. TPOT is the one the benchmark already reports, `(latency - ttft) / (output_len - 1)`; a single-token response has a TPOT of 0.
- New output, printed and written to the result file: request goodput (good requests per second), SLO attainment (good / sent) and one attainment per listed SLO, which shows which SLO binds.
- The scoring is a new module, `python/sglang/benchmark/goodput.py` (pure functions and one `msgspec.Struct`). `serving.py` gains 36 lines that parse the option, call the module and report the result; `calculate_metrics` and `BenchmarkMetrics` are not touched.
- Without the flag there is no Goodput block and no new field in the result file.
- One section in `docs/docs/developer_guide/bench_serving.mdx`, and three unit cases in `test/registered/unit/bench/test_bench_serving_goodput.py`.

```
---------------------Goodput----------------------
SLOs (ms):                               ttft:2000 tpot:100 e2el:20000
Request goodput (req/s):                 22.70
SLO attainment (%):                      52.22
TTFT SLO attainment (%):                 99.78
TPOT SLO attainment (%):                 74.43
E2EL SLO attainment (%):                 72.36
```

This is a summary of per-request latencies the benchmark already has, not a new measurement. What the existing lines cannot give is the joint figure: in the block above the three SLOs are met by 99.8, 74.4 and 72.4 % of the requests one at a time and by 52.2 % together.

## Accuracy Tests

No model or server code is touched. For the metric itself:

- **Unit cases** with hand-computed values: joint and per-SLO attainment over a failed and a single-token request, a subset of SLOs, the TPOT definition, invalid entries, no flag. To check that the cases guard something, 15 small regressions were applied to `goodput.py` one at a time (`any` for `all`, attainment over completed instead of sent requests, TPOT divided by `output_len`, an unknown key accepted, ...); each one turns the file red.
- **Live server** (`Qwen/Qwen3-8B`, one H200, this branch): in 8 runs with the flag (the `sglang-oai-chat` backend at fixed concurrency and at six request rates, and the `sglang` backend) the result fields, the printed block and a recount from `--output-details` agree. A run without the flag prints no block and writes no field.

One thing the metric inherits from the benchmark: a streaming request that the server aborts inside a 200 response (a waiting timeout, for example) is recorded as successful by the chat and native request functions, so it counts in request throughput and can count as good. That is independent of this PR; #40881 proposed a fix for the chat path and was closed without review.

## Speed Tests and Profiling

No inference path is touched. An example of what the option shows, a load sweep on this branch: `Qwen/Qwen3-8B`, one H200, `sglang-oai-chat` backend, ShareGPT, `--goodput ttft:2000 tpot:100 e2el:20000`, 50 s of arrivals per row.

| request rate (req/s) | output tok/s | goodput (req/s) | SLO attainment |
|---|---|---|---|
| 20.3 | 3636 | 18.60 | 100.0 % |
| 32.4 | 5474 | 28.38 | 99.8 % |
| 40.5 | 6762 | 34.85 | 99.7 % |
| 50.7 | 7586 | 38.30 | 98.5 % |
| 64.8 | 8552 | 22.70 | 52.2 % |
| 81.0 | 9664 | 8.41 | 17.2 % |

Output throughput is highest in the last row; goodput peaks near 50 req/s. This is one run per row, and just past the peak a 50-second run is noisy: three runs at about 64 req/s with an earlier revision of this patch gave 36.9, 69.6 and 71.8 % attainment, so that region is better read as a range. The block shown above is the 64.8 req/s row.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).

## Review and Merge Process

1. Ping Merge Oncalls to start the process. See the [PR Merge Process](https://github.com/sgl-project/sglang/blob/main/.github/MAINTAINER.md#pull-request-merge-process).
2. Get approvals from [CODEOWNERS](https://github.com/sgl-project/sglang/blob/main/.github/CODEOWNERS) and other reviewers.
3. Trigger CI tests with [comments](https://docs.sglang.io/developer_guide/contribution_guide.html#how-to-trigger-ci-tests) or contact authorized users to do so.
   - Common commands include `/tag-and-rerun-ci`, `/tag-run-ci-label`, `/rerun-failed-ci`
4. After green CI and required approvals, ask Merge Oncalls or people with Write permission to merge the PR.
