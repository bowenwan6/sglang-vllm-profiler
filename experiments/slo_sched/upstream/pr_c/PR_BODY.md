Adds `sglang:num_good_requests_total`: with `--goodput-slo` the server counts the requests that met their latency SLOs, so request goodput and SLO attainment can be read from `/metrics`.

## Motivation

The server exports TTFT, TPOT and end-to-end latency as separate histograms. They cannot say how many requests met all of their SLOs at once, and a single SLO can be read off only if it is a bucket boundary. That share (SLO attainment) and the rate of requests inside their SLOs (goodput) are what SLO-aware serving work optimises and reports (e.g. [JITServe](https://arxiv.org/abs/2504.20068), NSDI '26), and what an operator alerts or scales on. #43967 added the same definition to `bench_serving --goodput`; this is the server-side counterpart.

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

No model or scheduling code is touched. For the counter, on a live server:

- GPU: one NVIDIA H200
- Base: upstream `fc404bbbb3`
- Model: `Qwen/Qwen3-8B`, default arguments plus `--enable-metrics` and the flags below
- Requests: native `/generate` with random prompts, half of them streaming, 1 to 200 output tokens, closed loop at concurrency 8 to 1024

After each load phase, the counter's growth on `/metrics` must equal a client-side recount from the latencies each response reports in `meta_info`:

| Server | Phases | Requests | Counter equals recount |
|---|---|---|---|
| `--goodput-slo '{"ttft": 1000, "tpot": 40, "e2el": 8000}'` | 4 | 5,200 | 4 of 4 |
| `--goodput-slo '{"tpot": 20}'` | 3 | 3,200 | 3 of 3 |
| one lenient SLO, `--max-running-requests 32 --max-queued-requests 16` | 1 | 600 | 1 of 1 |
| no `--goodput-slo` (the counter has no samples) | 1 | 900 | 1 of 1 |
| the first one with `--tokenizer-worker-num 2` | 3 | 4,900 | 3 of 3 |

On the queue-limit server 536 of the 600 requests were rejected, and `sglang:num_requests_total` grew by 64: a request rejected before it runs is in neither counter, as for the latency histograms. The docs say so and name `sglang:http_requests_total` as the denominator that includes them.

`TestEnableMetrics.test_metrics_1gpu` passes with the new check and fails when the collector is not given the SLOs.

Not run: PD disaggregation (the prefill exclusion has a unit case), the Ray metrics backend, embedding requests.

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
