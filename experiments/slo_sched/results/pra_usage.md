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
1.25 × c0 gave 35.34, 39.50 and 37.83 req/s (σ = 5.6 % of the mean; attainment σ = 1.7 %, output
throughput σ = 3.8 %).

The latency columns the benchmark already prints show the same degradation (mean TPOT 15 → 155 ms,
p99 end-to-end 13 → 67 s), so the knee is not hidden without goodput. What they do not give is the
share of requests that met the target: at 1.6 × c0 the three SLOs are met by 100, 91 and 79 % of
requests one at a time and by 69.6 % together; at 2.0 × c0 by 85, 34 and 52 % and by 16.2 % together.
The joint figure cannot be read off per-metric percentiles.

The same six runs under a tighter set (`tpot:50 e2el:10000`), evaluated offline: 18.25, 27.55, 33.86,
23.59, 7.43, 2.47 req/s — the peak moves to 1.0 × c0.

### 2.2 Sizing the batch cap (T2), at 1.25 × c0

| `--max-running-requests` | output tok/s | goodput (req/s) | attainment | per SLO | mean TTFT |
|---|---|---|---|---|---|
| 32 | 4010 | 0.86 | 4.2 % | 4 / 100 / 30 | 32.8 s |
| 128 | 7349 | 14.86 | 39.5 % | 40 / 100 / 100 | 3.1 s |
| 512 | 7418 | 35.27 | 93.0 % | 100 / 99 / 94 | 57 ms |

Caps 128 and 512 differ by 1 % in output throughput and by 2.4× in goodput. The per-SLO line says why:
at 128 the requests queue and miss TTFT; at 512 they run and a few miss the end-to-end bound. Mean TTFT
(3.1 s against 57 ms) points at the same cap, so goodput did not change the choice here.

The cap trades first-token time against streaming speed at the same throughput: cap 128 has mean TPOT
15 ms (p99 26 ms) and p99 end-to-end 16 s, cap 512 has 36 ms (p99 92 ms) and 32 s. Which side matters
is the SLO's decision. The same three runs under two other SLO sets, **chosen after seeing the data**
(`p1_report.py --alt`):

| SLO set | cap 32 | cap 128 | cap 512 |
|---|---|---|---|
| `e2el:20000` (a deadline, no first-token target) | 6.05 req/s, 29.5 % | 37.43 req/s, 99.6 % | 35.54 req/s, 93.7 % |
| `ttft:10000 tpot:30` (fast streaming, patient first token) | 3.40 req/s, 16.6 % | 37.45 req/s, 99.6 % | 14.82 req/s, 39.1 % |

Under the second set cap 128 is 2.5× ahead while throughput (1 % apart) and mean TTFT both point at
cap 512; under the first the lead is 5 % in req/s, inside the noise. These are illustrations of how the
answer follows the SLO, not planned comparisons, and they do not count toward A1.7.

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

Output throughput is the same in all three (within 3.8 %, which is one σ of throughput), so throughput
does not rank the settings; goodput separates them 3.1×. A 10 s bound is almost as bad as none. Without
a bound 82 % of the generated tokens went to requests that missed an SLO; with the 2 s bound, 37 %. The latency columns improve too (mean TTFT of the served requests 8.6 s, 1.6 s, 6.1 s), but
they are computed over the served requests only and so cannot show what the shedding cost: attainment
counts the 735 refused requests as misses.

The 2 s bound is equal to the TTFT objective, which is too late by the prefill time: half of the served
requests waited almost the full two seconds (median TTFT 1975 ms, p99 2091 ms) and 825 of them were
served with a TTFT just above the objective. A bound a little below the objective would do better; not
run.
One global value has to be chosen for every request — the case PR-B addresses.

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
`test_bench_serving_stream_error.py` fails on the old code in all three and passes on the new. Per path
on the old code: chat and native (`--backend sglang`, the default) record a success; completions
already records a failure, because the error event has no `choices` and the parser raises, but with a
traceback as its message. It has run only through the Mac stub harness so far, not in a real SGLang
environment.

**Upstream already has an open PR for the chat path** (found 10-05, after P1; the stage-2 search missed
it): sgl-project/sglang#40881, opened 2026-09-23, +4 lines in `serving.py` and three loopback tests,
no review yet. It describes the same symptom (an error-only stream reported as `success=True` with the
requested output length). It does not cover the native path, where the abort arrives as a
`finish_reason` and not as an `error` key. Consequences: our fix is not opened as a competing PR; what
we can add is the native path and a real-server reproduction (735 of 3017 requests under a 2 s waiting
timeout). `--goodput` does not depend on either: without a fix it is inflated by aborted requests in
exactly the way request and token throughput already are.

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
| A1.7 a different decision than throughput (T2, T4) or mean TTFT (T3) | **Met in T4 once aborted requests are counted as failed; not met on the benchmark's stock output, and not in T2 or T3.** T4 corrected: best by output throughput is nominally "no timeout" (7362 tok/s; the three settings are within 3.8 %, one σ of throughput), best by goodput is the 2 s bound (22.95 against 7.37 req/s, a 68 % gap against a 16.7 % threshold). T2 and T3: same winner; goodput added magnitude and cause — 2.4× where throughput shows 1 % (T2), and the binding SLO (T3) |

Correction (10-05, later the same day): the first version of this table said A1.7 failed in all three
tasks. That verdict judged T4 on the stock numbers, which §3.1 shows are inflated by the aborted
requests; `p1_report.py` now states A1.7 for both accountings. The honest reading of T4 is "throughput
cannot tell the three settings apart, goodput can", not "throughput picks the wrong one": the
throughput ranking is inside the noise.

By the plan's rule PR-A goes ahead. What its description can claim: the knee (T1) and the per-SLO
breakdown (T2, T3); T4 belongs with the abort-accounting fix and with PR-B. What it must not claim:
that goodput contradicts the latency columns. In every task a reader of mean TTFT and TPOT would pick
the same configuration. Goodput is a summary of the per-request latencies against a target — the joint
share that met it, with failed requests counted as misses — and not new information.

## 5. Limits

One model, one GPU, one seed per cell except the T1 repeat; σ = 5.6 % comes from three runs of a single
cell and is applied to the others. T3's rates were not calibrated, so both classes are overloaded.
Every open-loop cell sends requests for 50 s. Above saturation the queue grows for as long as requests
arrive, so attainment there depends on the run length (the 16 % at 2.0 × c0 is a 50-second number); the
position of the knee does not. Goodput in req/s divides by the run's duration, which includes the drain
after the last arrival; attainment is the steadier number. The corrected T4 numbers are an offline
re-count of the recorded responses, not a run of the fixed benchmark.
The node was assigned for 67 minutes. The extension (1 credit) was requested on a 70-minute estimate;
the tasks took 34 minutes and ended 23 minutes inside the first hour, and the follow-up captures of
§3 used 7 minutes of the second.

## 6. Addendum — session P2 (2026-10-06): repeats, and the fix run on a server

Same node type, model and base commit; the benchmark is the branch with both commits (the goodput
option and the in-stream-abort fix). Tables: [`prb/p2_report.md`](prb/p2_report.md).

**The fix in a real environment.** `test_bench_serving_stream_error.py` passes there (it had run only
through the Mac stub harness), and T4 was run again with the fixed benchmark, three seeds per setting:

| `SGLANG_REQ_WAITING_TIMEOUT` | failed requests | output tok/s | goodput (req/s) | attainment | printed = recomputed |
|---|---|---|---|---|---|
| off | 0 | 7743 (346) | 7.33 (0.80) | 18.7 % (2.7) | yes |
| 2 s | 732 (16) | 7493 (398) | 24.75 (1.65) | 48.7 % (1.5) | yes |
| 10 s | 376 (29) | 7608 (386) | 8.36 (0.84) | 19.0 % (2.4) | yes |

Mean and standard deviation over the three seeds. This replaces the offline re-count of §2.4 (7.37,
22.95 and 8.88 req/s from one run each) with measured runs and agrees with it. The aborted requests
are now reported as failed, and A1.5 holds in every cell. For A1.7 the reading of §4 stands with error
bars: output throughput differs by 3 % between "off" and 2 s, less than one standard deviation, and
goodput differs 3.4×.

**T1 with repeats.** Two more seeds at 1.0, 1.6 and 2.0 × c0 (1.25 × c0 already had three runs):

| load | runs | output tok/s | goodput (req/s) | attainment |
|---|---|---|---|---|
| 1.0 × c0 | 3 | 6984 | 35.76 (35.08–36.69) | 99.8–100.0 % |
| 1.25 × c0 | 3 | 7727 | 37.56 (35.34–39.50) | 93.4–96.5 % |
| 1.6 × c0 | 3 | 8808 | 26.73 (16.53–32.80) | 36.9–71.8 % |
| 2.0 × c0 | 3 | 9668 | 9.02 (7.59–11.34) | 16.0–23.1 % |

The knee is where it was. What the repeats add is the spread just past it: at 1.6 × c0 one seed gave
36.9 % where the other two gave 69.6 and 71.8 %. A single 50-second run in that region is not a
measurement of attainment; the PR description gives ranges.
