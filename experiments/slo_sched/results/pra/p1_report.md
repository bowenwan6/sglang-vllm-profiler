## A1.5 — printed vs recomputed good-request count

19 cells, 2 differ.
| cell | printed | recomputed | sent |
|---|---|---|---|
| t4_wt2 | 2019 | 2187 | 3017 |
| t4_wt10 | 877 | 1039 | 3017 |

## T1 — load sweep

Capacity probe (closed loop, concurrency 256): 40.22 req/s, 7724 output tok/s, mean TPOT 34.7 ms.

| × c0 | rate (req/s) | sent | req thr | out tok/s | goodput (req/s) | attain % | per SLO % | mean TTFT ms | p99 TTFT ms | mean TPOT ms | p99 E2E ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 20.11 | 1006 | 18.30 | 3586 | 18.30 | 100.0 | ttft 100 tpot 100 e2el 100 | 32 | 64 | 8.6 | 6875 |
| 0.8 | 32.18 | 1609 | 28.20 | 5412 | 28.13 | 99.8 | ttft 100 tpot 100 e2el 100 | 38 | 132 | 13.0 | 11200 |
| 1 | 40.22 | 2011 | 35.15 | 6783 | 35.08 | 99.8 | ttft 100 tpot 100 e2el 100 | 36 | 130 | 14.9 | 13290 |
| 1.25 | 50.28 | 2514 | 37.84 | 7400 | 35.34 | 93.4 | ttft 100 tpot 100 e2el 94 | 52 | 191 | 35.5 | 31832 |
| 1.6 | 64.36 | 3218 | 44.33 | 8710 | 30.87 | 69.6 | ttft 100 tpot 91 e2el 79 | 90 | 366 | 65.9 | 46590 |
| 2 | 80.44 | 4022 | 46.78 | 9244 | 7.59 | 16.2 | ttft 85 tpot 34 e2el 52 | 726 | 4233 | 154.9 | 67070 |

Noise at 1.25 × c0 over 3 runs: goodput 35.34, 39.50, 37.83 req/s (σ = 5.6 % of the mean); attainment 93.4, 95.5, 96.5 % (σ = 1.7 %); output throughput 7400, 7953, 7829 tok/s (σ = 3.8 %).

**A1.6**: goodput peaks at 1.25 × c0 (35.34 req/s); at 2 × c0 it is 79 % below the peak while output throughput is 100 % of its own peak → PASS.

## T2 — batch cap at 1.25 × c0

| max running | req thr | out tok/s | goodput (req/s) | attain % | per SLO % | mean TTFT ms | mean TPOT ms | p99 E2E ms | peak concurrency |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 20.51 | 4010 | 0.86 | 4.2 | ttft 4 tpot 100 e2el 30 | 32805 | 7.8 | 69234 | 1461 |
| 128 | 37.58 | 7349 | 14.86 | 39.5 | ttft 40 tpot 100 e2el 100 | 3098 | 15.4 | 16236 | 519 |
| 512 | 37.94 | 7418 | 35.27 | 93.0 | ttft 100 tpot 99 e2el 94 | 57 | 36.4 | 32441 | 558 |

T2: best by output_throughput = cap 512, best by goodput = cap 512, goodput gap 0.0 % (threshold 16.7 %) → same decision.

## T3 — queue policy, short and long clients at once

| policy | client | rate | sent | goodput (req/s) | attain % | per SLO % | mean TTFT ms | p99 TTFT ms | p99 E2E ms |
|---|---|---|---|---|---|---|---|---|---|
| fcfs | short | 24.13 | 1207 | 2.73 | 12.7 | ttft 50 tpot 41 | 1796 | 5955 | 22188 |
| fcfs | long | 2.00 | 100 | 0.97 | 56.0 | e2el 56 | 1933 | 6104 | 37305 |
| hrrn | short | 24.13 | 1207 | 6.61 | 30.7 | ttft 94 tpot 32 | 519 | 1378 | 22264 |
| hrrn | long | 2.00 | 100 | 0.95 | 55.0 | e2el 55 | 2263 | 7337 | 37751 |

T3 (both clients pooled): best by mean_ttft_ms = hrrn, best by goodput = hrrn, goodput gap 0.0 % (threshold 16.7 %) → same decision.

## T4 — global waiting timeout at 1.5 × c0, max running 128

| waiting timeout (s) | sent | failed (503) | req thr | out tok/s | goodput (req/s) | attain % | per SLO % | mean TTFT ms | p99 E2E ms |
|---|---|---|---|---|---|---|---|---|---|
| off | 3017 | 0 | 37.24 | 7362 | 7.37 | 19.8 | ttft 20 tpot 100 e2el 88 | 8603 | 25841 |
| 2 | 3017 | 0 | 47.69 | 9427 | 31.91 | 66.9 | ttft 73 tpot 94 e2el 100 | 1227 | 14009 |
| 10 | 3017 | 0 | 42.01 | 8304 | 12.21 | 29.1 | ttft 35 tpot 95 e2el 98 | 5326 | 20889 |

T4: best by output_throughput = timeout 2, best by goodput = timeout 2, goodput gap 0.0 % (threshold 16.7 %) → same decision.

T4 with server-aborted responses counted as failed (what the fixed benchmark reports):

| waiting timeout (s) | aborted by the server | req thr | out tok/s | goodput (req/s) | attain % | phantom output tokens in the stock report % |
|---|---|---|---|---|---|---|
| off | 0 | 37.24 | 7362 | 7.37 | 19.8 | 0 |
| 2 | 735 | 36.07 | 7085 | 22.95 | 48.1 | 25 |
| 10 | 401 | 36.42 | 7162 | 8.88 | 21.1 | 14 |

T4, server-aborted requests counted as failed: best by output_throughput = timeout off, best by goodput = timeout 2, goodput gap 67.9 % (threshold 16.7 %) → decision differs.
The two differ by 3.8 % in output_throughput; σ of output throughput is 3.8 %, so that ranking is inside the noise.

**A1.7**, on the benchmark's stock output: decision differs in none → FAIL.

**A1.7**, with server-aborted requests counted as failed: decision differs in T4 → PASS.

## T1 re-evaluated offline under ttft:2000,tpot:50,e2el:10000

| × c0 | goodput (req/s) | attain % |
|---|---|---|
| 0.5 | 18.25 | 99.7 |
| 0.8 | 27.55 | 97.7 |
| 1 | 33.86 | 96.3 |
| 1.25 | 23.59 | 62.3 |
| 1.6 | 7.43 | 16.7 |
| 2 | 2.47 | 5.3 |

## T2 re-evaluated offline under ttft:2000,tpot:50,e2el:10000

| max running | goodput (req/s) | attain % |
|---|---|---|
| 32 | 0.84 | 4.1 |
| 128 | 13.96 | 37.2 |
| 512 | 22.35 | 58.9 |

## T1 re-evaluated offline under e2el:20000

| × c0 | goodput (req/s) | attain % |
|---|---|---|
| 0.5 | 18.30 | 100.0 |
| 0.8 | 28.16 | 99.9 |
| 1 | 35.08 | 99.8 |
| 1.25 | 35.48 | 93.8 |
| 1.6 | 34.81 | 78.5 |
| 2 | 24.42 | 52.2 |

## T2 re-evaluated offline under e2el:20000

| max running | goodput (req/s) | attain % |
|---|---|---|
| 32 | 6.05 | 29.5 |
| 128 | 37.43 | 99.6 |
| 512 | 35.54 | 93.7 |

## T1 re-evaluated offline under ttft:10000,tpot:30

| × c0 | goodput (req/s) | attain % |
|---|---|---|
| 0.5 | 18.28 | 99.9 |
| 0.8 | 27.95 | 99.1 |
| 1 | 34.80 | 99.0 |
| 1.25 | 15.07 | 39.8 |
| 1.6 | 5.50 | 12.4 |
| 2 | 1.34 | 2.9 |

## T2 re-evaluated offline under ttft:10000,tpot:30

| max running | goodput (req/s) | attain % |
|---|---|---|
| 32 | 3.40 | 16.6 |
| 128 | 37.45 | 99.6 |
| 512 | 14.82 | 39.1 |
