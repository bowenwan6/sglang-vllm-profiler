# PR-A — draft description (not opened; Bowen reviews it together with PR-B, then opens it)

- Branch: `bowenwan6/sglang` `pr/bench-goodput` @ `fd40750d7b` — one commit on `upstream/main`
  @ `b524de2de6` (2026-10-06). Patch: [`0001-bench-goodput.patch`](0001-bench-goodput.patch).
- Same diff as `d132f6739e`, the commit session P1 measured at `734cf3cf3b`; `benchmark/serving.py`
  did not change upstream between the two bases.
- Checked: 8 unit cases, `ruff`, and the full pre-commit suite in a real SGLang environment at
  `734cf3cf3b` (P1 step 0); the 8 cases again on the rebased file through the Mac stub harness.
  Not yet re-run in a real environment on the rebased commit.
- Evidence and limits: [`../../results/pra_usage.md`](../../results/pra_usage.md). The load sweep below
  carries the repeats of session P2 (2026-10-06): three runs at each of the four highest rates.

Everything below the line is the text for GitHub.

---

**Title:** `[Bench] Report goodput and SLO attainment in bench_serving`

## Motivation

`bench_serving` reports throughput and latency percentiles, but not how much of the served load met a
latency target. The per-request data is already collected; the share of requests that met a target has
to be computed by hand from `--output-details`.

This adds the `--goodput` option that vLLM's `benchmark_serving.py` has (same spelling, so command
lines carry over). #5495 proposed it for the old `bench_serving.py` and was closed for inactivity after
a request to rebase; this is a fresh implementation on `sglang.benchmark.serving`.

## Modifications

- `--goodput KEY:VALUE [KEY:VALUE ...]` with keys `ttft`, `tpot`, `e2el` and values in milliseconds,
  any subset. Unknown keys, missing or non-positive values and duplicate keys are rejected.
- A request is *good* if it succeeded and met every listed SLO. Failed requests count as misses.
- New output, printed and written to the result JSON: request goodput (good requests per second), SLO
  attainment (good / sent), and one attainment per listed SLO, which shows which SLO binds.
- Without the flag the console output and the result fields are unchanged (covered by a test).
- One section in `docs/docs/developer_guide/bench_serving.mdx`, and
  `test/registered/unit/bench/test_bench_serving_goodput.py` (8 cases).

```
---------------------Goodput----------------------
SLOs (ms):                               ttft:2000 tpot:100 e2el:20000
Request goodput (req/s):                 30.87
SLO attainment (%):                      69.64
TTFT SLO attainment (%):                 100.00
TPOT SLO attainment (%):                 91.11
E2EL SLO attainment (%):                 78.53
```

It is a summary of the per-request latencies the benchmark already has, not a new measurement. What
the existing lines cannot give is the joint figure: in the block above the three SLOs are met by 100,
91 and 79 % of the requests one at a time and by 70 % together.

## Accuracy Tests

No model or server code is touched.

For the metric itself: the unit cases use hand-computed values (failed requests, a one-token response,
each SLO alone and together, the no-flag path). On 23 runs against a live server the printed number of
good requests equals an independent recount from `--output-details` exactly.

One thing the metric inherits: a streaming request that the server aborts inside a 200 response (a
waiting timeout, for example) is currently recorded as a successful one by the chat and native request
functions, so it is counted in request throughput and can be counted as good. #40881 fixes the chat
path.

## Speed Tests and Profiling

No inference path is touched. An example of what the option shows, on a load sweep: `Qwen/Qwen3-8B`,
one H200, ShareGPT, `--goodput ttft:2000 tpot:100 e2el:20000`, 50 s of requests per row.

| request rate (req/s) | runs | output tok/s | goodput (req/s) | SLO attainment |
|---|---|---|---|---|
| 20.1 | 1 | 3586 | 18.30 | 100.0 % |
| 32.2 | 1 | 5412 | 28.13 | 99.8 % |
| 40.2 | 3 | 6984 | 35.76 (35.08–36.69) | 99.8–100.0 % |
| 50.3 | 3 | 7727 | 37.56 (35.34–39.50) | 93.4–96.5 % |
| 64.4 | 3 | 8808 | 26.73 (16.53–32.80) | 36.9–71.8 % |
| 80.4 | 3 | 9668 | 9.02 (7.59–11.34) | 16.0–23.1 % |

Means, with the range over the runs in brackets. Output throughput is highest in the last row; goodput
peaks around 40–50 req/s. Just past that point a 50-second run is noisy: the three runs at 64.4 req/s
gave 36.9, 69.6 and 71.8 % attainment. The block shown above is the 69.6 % run.

## Checklist

- [x] Format your code according to the Format code with pre-commit.
- [x] Add unit tests according to the Run and add unit tests.
- [x] Update documentation according to Write documentations.
- [ ] Provide accuracy and speed benchmark results (not applicable: benchmark client only; an example
      run is above).
- [x] Follow the SGLang code style guidance.
