Adds `sglang:num_good_requests_total`: with `--goodput-slo` the server counts the requests that met their latency SLOs, so request goodput and SLO attainment can be read from `/metrics`.

## Motivation

The server exports TTFT, TPOT and end-to-end latency as separate histograms. They cannot say how many requests met all of their SLOs at once, and a single SLO can only be read off when it happens to be a bucket boundary. That share (SLO attainment) and the rate of requests inside their SLOs (goodput) are what SLO-aware serving work optimises and reports (e.g. [JITServe](https://arxiv.org/abs/2504.20068), NSDI '26), and what an operator alerts or scales on. #43967 adds the same definition to `bench_serving`; this is the server side and does not depend on it.

## Modifications

- `--goodput-slo '{"ttft": 500, "tpot": 50}'`: latency SLOs in milliseconds, a JSON dict with keys from `ttft`, `tpot` and `e2el`. Unknown keys and non-positive values are rejected at startup.
- `sglang:num_good_requests_total`, labelled like `sglang:num_requests_total`. A request counts if it finished with `stop` or `length` and met every given SLO, judged on the values the latency histograms record. Aborted requests and PD prefill workers do not count; a request without a request-level TPOT meets the `tpot` SLO.
- Without the flag nothing is counted and the counter has no samples.
- Docs (`production_metrics.mdx`, `server_arguments.mdx`), unit cases in the existing test files, and one check in `test/registered/observability/test_metrics.py`.

```promql
# request goodput (req/s), and SLO attainment among finished requests
sum by (model_name) (rate(sglang:num_good_requests_total[5m]))
sum by (model_name) (rate(sglang:num_good_requests_total[5m]))
  / sum by (model_name) (rate(sglang:num_requests_total[5m]))
```

## Accuracy Tests

No model or scheduling code is touched. For the counter, on a live server (`Qwen/Qwen3-8B`, one H200): its growth on `/metrics` equals a client-side recount from the latencies each response reports in `meta_info`, in all 12 load phases (14,800 native `/generate` requests, half of them streaming) over five servers: three SLOs, one SLO, a queue limit that rejects most of a burst, no flag, and two tokenizer workers.

`TestEnableMetrics.test_metrics_1gpu` passes with the new check and fails when the collector is not given the SLOs.

## Speed Tests and Profiling

No inference path is touched. `observe_one_finished_request` takes 16.5 µs without the flag and 19.3 µs with three SLOs (median of 5 runs of 200k calls); it runs once per finished request, outside the scheduler.

What the counter adds, with `--goodput-slo '{"ttft": 1000, "tpot": 40, "e2el": 8000}'` under closed-loop load:

| concurrency | requests | met TTFT | met TPOT | met E2E | met all (the counter) |
|---|---|---|---|---|---|
| 8 | 300 | 100 % | 100 % | 100 % | 100 % |
| 64 | 900 | 100 % | 99.9 % | 100 % | 99.9 % |
| 256 | 1500 | 96.8 % | 94.7 % | 100 % | 91.7 % |
| 1024 | 2500 | 64.8 % | 14.5 % | 46.7 % | 11.8 % |

The per-SLO columns are from the client-side recount. One run per row.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [x] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
