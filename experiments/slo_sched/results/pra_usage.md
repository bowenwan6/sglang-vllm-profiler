# `--goodput` in `sglang.bench_serving` — how to use it, and what session P1 showed

> Session P1, 2026-10-05 02:12–03:19 UTC, one H200 on a RADIX node, `Qwen/Qwen3-8B`, SGLang at
> `bowenwan6/sglang` `feat/bench-goodput` @ `d132f6739e` (`upstream/main` `734cf3cf3b` + PR-A),
> torch 2.14.1+cu130, flashinfer 0.7.0.post1. Tasks and thresholds: [`../PLAN.md`](../PLAN.md) §1.3
> and its amendment. Tables: [`pra/p1_report.md`](pra/p1_report.md), produced by
> `scripts/p1_report.py` from [`pra/summary.jsonl`](pra/summary.jsonl). Raw per-request files stay in
> `results/raw/p1/` (ignored). One run per cell unless a repeat is stated; σ comes from three runs of
> one cell.

## 1. The command

```bash
python -m sglang.bench_serving --backend sglang-oai-chat --host 127.0.0.1 --port 30000 \
  --model Qwen/Qwen3-8B --dataset-name sharegpt --num-prompts 3218 --request-rate 64.4 \
  --goodput ttft:2000 tpot:100 e2el:20000 --output-details --output-file run.jsonl
```

Keys are `ttft`, `tpot`, `e2el`, values in milliseconds, any subset. A request is good if it succeeded
and met every listed SLO. The run prints one more block and adds four fields to the result JSON
(`goodput_slos_ms`, `request_goodput`, `slo_attainment`, `slo_attainment_by_metric`):

```
Request throughput (req/s):              44.33
Output token throughput (tok/s):         8710.19
---------------------Goodput----------------------
SLOs (ms):                               ttft:2000 tpot:100 e2el:20000
Request goodput (req/s):                 30.87
SLO attainment (%):                      69.64
TTFT SLO attainment (%):                 100.00
TPOT SLO attainment (%):                 91.11
E2EL SLO attainment (%):                 78.53
```

How to read it: goodput is the part of the request throughput that was useful; attainment is the share
of *sent* requests that were good; the per-SLO lines say which SLO is binding (here end-to-end latency,
then TPOT; TTFT is not the problem).

Choosing SLO values: with `--output-details` the result keeps every request's TTFT, inter-token times
and output length, so another SLO set can be evaluated afterwards without a rerun
(`scripts/p1_report.py --alt ttft:2000,tpot:50,e2el:10000`). Start with the product's numbers; if no
request misses them at the highest load of interest, they do not bind and goodput equals throughput.

## 2. What it is for — four uses, measured

SLOs `ttft:2000 tpot:100 e2el:20000` (the JITServe paper's values) unless stated. Load is relative to
c0 = 40.2 req/s, the request throughput of a closed-loop probe at concurrency 256.

### 2.1 Finding the SLO-compliant capacity (T1)

| load | sent | output tok/s | goodput (req/s) | attainment | per SLO (ttft / tpot / e2el) |
|---|---|---|---|---|---|
| 0.5 × c0 | 1006 | 3586 | 18.30 | 100.0 % | 100 / 100 / 100 |
| 0.8 × c0 | 1609 | 5412 | 28.13 | 99.8 % | 100 / 100 / 100 |
| 1.0 × c0 | 2011 | 6783 | 35.08 | 99.8 % | 100 / 100 / 100 |
| 1.25 × c0 | 2514 | 7400 | 35.34 | 93.4 % | 100 / 100 / 94 |
| 1.6 × c0 | 3218 | 8710 | 30.87 | 69.6 % | 100 / 91 / 79 |
| 2.0 × c0 | 4022 | 9244 | 7.59 | 16.2 % | 85 / 34 / 52 |

Output throughput rises at every step and is highest at 2.0 × c0. Goodput peaks around 1.0–1.25 × c0
and is 79 % lower at 2.0 × c0. Throughput alone would call the last row the best one. Three runs at
1.25 × c0 gave 35.34, 39.50 and 37.83 req/s (σ = 5.6 % of the mean).

The same six runs under a tighter set (`tpot:50 e2el:10000`), evaluated offline: 18.25, 27.55, 33.86,
23.59, 7.43, 2.47 req/s — the peak moves to 1.0 × c0.

### 2.2 Sizing the batch cap (T2), at 1.25 × c0

| `--max-running-requests` | output tok/s | goodput (req/s) | attainment | per SLO | mean TTFT |
|---|---|---|---|---|---|
| 32 | 4010 | 0.86 | 4.2 % | 4 / 100 / 30 | 32.8 s |
| 128 | 7349 | 14.86 | 39.5 % | 40 / 100 / 100 | 3.1 s |
| 512 | 7418 | 35.27 | 93.0 % | 100 / 99 / 94 | 57 ms |

Caps 128 and 512 differ by 1 % in output throughput and by 2.4× in goodput. The per-SLO line says why:
at 128 the requests queue and miss TTFT; at 512 they run and a few miss the end-to-end bound.

### 2.3 Comparing queue policies under mixed prompt lengths (T3)

Two benchmark processes at once, each with its own SLOs: short (256-token prompts, 24 req/s,
`ttft:1000 tpot:100`) and long (12k-token prompts, 2 req/s, `e2el:30000`).

| policy | client | goodput (req/s) | attainment | per SLO | mean TTFT |
|---|---|---|---|---|---|
| `fcfs` | short | 2.73 | 12.7 % | ttft 50, tpot 41 | 1796 ms |
| `fcfs` | long | 0.97 | 56.0 % | e2el 56 | 1933 ms |
| `hrrn` | short | 6.61 | 30.7 % | ttft 94, tpot 32 | 519 ms |
| `hrrn` | long | 0.95 | 55.0 % | e2el 55 | 2263 ms |

`hrrn` fixes the short requests' TTFT (50 % → 94 % on time) at no cost to the long ones. Attainment
still stops at 31 %, and the per-SLO line shows the reason is TPOT: decoding stalls while 12k-token
prefills run. A TTFT table alone would not show that the remaining problem is a different one.

### 2.4 Tuning load shedding (T4), at 1.5 × c0 with `--max-running-requests 128`

With aborted requests counted as failed (see §3.1):

| `SGLANG_REQ_WAITING_TIMEOUT` | aborted by the server | output tok/s | goodput (req/s) | attainment |
|---|---|---|---|---|
| off | 0 | 7362 | 7.37 | 19.8 % |
| 2 s | 735 of 3017 | 7085 | 22.95 | 48.1 % |
| 10 s | 401 of 3017 | 7162 | 8.88 | 21.1 % |

Shedding costs 4 % of output throughput and triples goodput when the bound matches the TTFT objective;
a 10 s bound is almost as bad as none. One global value has to be chosen for every request — the case
PR-B addresses.

## 3. Two problems this session exposed

### 3.1 The benchmark counted server-aborted streaming requests as successes (fixed on the branch)

The stock numbers for the two T4 cells with a timeout were inflated: the 2 s cell read 47.69 req/s
and 9427 output tok/s, more than the server reached anywhere in the T1 sweep (46.78 req/s, 9244
tok/s). Cause: when the waiting timeout fires on a queued *streaming* request the server answers HTTP
200 and puts the error inside the stream ([captures](pra/abort_responses/), taken on the node):

| endpoint | what the client receives |
|---|---|
| chat or completions, streaming | 200, then `data: {"error": {"message": "Request waiting timeout reached.", "type": "SERVICE_UNAVAILABLE", "code": 503}}` |
| native `/generate`, streaming | 200, then a chunk with empty text and `finish_reason: {"type": "abort", "status_code": 503, …}` |
| any endpoint, non-streaming | 503 with the same message |

The chat and native streaming paths of `bench_serving` skipped that event and recorded a completed
request with the *requested* output length. In the 2 s cell 735 requests (24 %) were phantom successes
and 25 % of the reported output tokens were never generated; with `--goodput` 567 of them were also
counted as good, so goodput read 31.91 instead of 22.95 req/s. This is what made the offline
recomputation disagree with the printed value in exactly these two cells (A1.5).

Fix: `9eb681bef6` on `feat/bench-goodput` — an in-stream error makes the request a failed one carrying
the server's message, on the chat, completions and native paths; regression test
`test_bench_serving_stream_error.py` fails on the old code in all three and passes on the new. It has
run only through the Mac stub harness so far, not in a real SGLang environment. It is a separate commit
so it can be offered as its own PR; `--goodput` should not be merged without it.

### 3.2 The first sampled request on a fresh server stalls the scheduler for about 70 s

On this node build, the first request that samples with top-k / top-p (the default for a chat request
without `temperature`, from Qwen3's `generation_config.json`) was scheduled 70 s after it arrived, and
every other request waited behind it; later sampled requests were immediate. Greedy requests are not
affected, which is why the benchmark (temperature 0) never sees it. Not investigated further; it may be
a cold just-in-time kernel build specific to this installation. Consequence for our tests: send only
`temperature: 0` requests, or warm the sampling path first, before timing anything. No waiting-timeout
abort was delivered during the stall in three attempts, so a timeout test must not start inside it.

## 4. Acceptance (PLAN.md §1.3)

| | result |
|---|---|
| Step 0, real environment | unit tests 10 passed (8 goodput cases + the existing file), `ruff format`, `ruff check`, full pre-commit on the three files: all pass |
| A1.5 printed = recomputed | 17 of 19 cells identical; the two that differ are the T4 timeout cells, explained and fixed in §3.1 |
| A1.6 knee | **PASS**: 79 % below the peak at 2.0 × c0 with output throughput at its own peak |
| A1.7 a different decision than throughput or mean TTFT | **FAIL as written**: in T2, T3 and T4 the configuration that is best by throughput (or mean TTFT) is also best by goodput. What goodput added was magnitude and cause — 2.4× where throughput shows 1 % (T2), and the binding SLO (T3) — not a different winner |

By the plan's rule PR-A goes ahead: the knee is the stated use. The description must not claim that
goodput changes which configuration wins on these tasks.

## 5. Limits

One model, one GPU, one seed per cell except the T1 repeat; σ = 5.6 % comes from three runs of a single
cell and is applied to the others. T3's rates were not calibrated, so both classes are overloaded.
The node was assigned for 67 minutes. The extension (1 credit) was requested on a 70-minute estimate;
the tasks took 34 minutes and ended 23 minutes inside the first hour, and the follow-up captures of
§3 used 7 minutes of the second.
