# S1 — PR track: a goodput benchmark (PR-A) and a per-request waiting timeout (PR-B)

> **Status: written 2026-10-05. Stage 2 (research) is done; nothing has run on a GPU.**
> Branch `exp/slo-prs`. This is the active plan. Thresholds in §2–§4 are fixed before any server run;
> afterwards this file gets dated "Outcome" sections only.
> Background (paper facts, SGLang source facts, the full client-emulated ladder):
> [`BACKGROUND.md`](BACKGROUND.md). PR-B research: [`PRB_RESEARCH.md`](PRB_RESEARCH.md).
> **Every server session is announced with its plan first and starts only after Bowen's approval.**

## 0. What this track does

JITServe as a whole will not be ported (BACKGROUND.md §2). Two pieces are small enough to upstream and
useful without the rest:

| | What | Why it can land |
|---|---|---|
| **PR-A** | `--goodput` in `sglang.benchmark.serving`: requests per second that meet stated SLOs, and the share that do | vLLM's benchmark has had it since 2024-10 (vllm#9338); SGLang's attempt (#5495) got "could you rebase the changes? Then we can merge it" and was closed for inactivity |
| **PR-B** | A per-request waiting timeout: a queued request is dropped once it has waited longer than its own bound | SGLang has only one global bound; the scan that enforces it was hardened a month ago (#37143); Triton has the per-request form |

Deadline-aware ordering (PR-C in earlier notes) is **out of the active scope**; it stays in
BACKGROUND.md as a possible later study.

Order: stage 1 uses PR-A on real tasks, stage 2 researches PR-B, stage 3 builds and measures PR-B with
PR-A's metric. Code lives in the SGLang fork on two branches cut from `upstream/main`
(`feat/bench-goodput`, `feat/req-waiting-timeout`); this repo holds plans, the mixed-workload client,
results and reports.

## 1. Stage 1 — PR-A: build it, use it, find out what it is good for

### 1.1 Design
- `--goodput KEY:VALUE [KEY:VALUE ...]`, keys `ttft`, `tpot`, `e2el`, values in milliseconds — the
  spelling of vLLM's `benchmark_serving.py`, so existing command lines carry over.
- A request is **good** if it succeeded and met every listed SLO.
- New numbers, printed and written to the result JSON: `request_goodput` (good requests per second),
  `slo_attainment` (good / sent; failed requests are misses) and one attainment per listed SLO, which
  shows which SLO binds.
- No flag, no change: default output and JSON are byte-identical to today's.
- Shape: optional fields on `BenchmarkMetrics`, one argument to `calculate_metrics`, CLI parsing with
  validation, a unit test under `test/registered/unit/bench/`. Target ≤ 200 changed lines.

### 1.2 Local validation (Mac, no server)
| # | Accept |
|---|---|
| A1.1 | Unit test: hand-computed goodput, attainment and per-SLO attainment on synthetic `RequestFuncOutput`s, including failed requests, a missing `tpot` (one-token output) and the no-flag path. |
| A1.2 | Invalid input is rejected with a clear message: unknown key, missing value, non-positive value, duplicate key. |
| A1.3 | With no `--goodput`, the result dictionary has exactly the keys it has on `upstream/main`. |
| A1.4 | The repository's pre-commit hooks pass on the changed files. |

### 1.3 Server tasks — "is it useful, and how is it used" (session P1, ≈ 2 h, needs approval)
Model `Qwen/Qwen3-8B` on one H200, the patched benchmark, `--output-details` kept for every run.
SLO set for all tasks unless stated: `ttft:2000 tpot:100 e2el:20000` (the JITServe paper's values).

| id | Task | What goodput should show that the existing numbers do not |
|---|---|---|
| T1 | Load sweep: one server, ShareGPT, request rate at six points from light to 2× the knee | output throughput keeps rising or flattens while goodput peaks and falls; the peak is the SLO-compliant capacity |
| T2 | Batch cap: `--max-running-requests` ∈ {32, 128, 512} at a rate above the knee | the cap with the highest throughput is not the one with the highest goodput |
| T3 | Queue policy under mixed prompt lengths (short chat + long documents): `fcfs`, `lpm`, `hrrn` | policies ranked by mean TTFT versus ranked by goodput |
| T4 | Global waiting timeout at overload: off, 2 s, 10 s | throughput barely moves, goodput does — and one value cannot suit both SLOs, which is PR-B's case |

Noise: T1's point nearest the knee is run three times; σ is the standard deviation of its goodput.

| # | Accept ("useful") |
|---|---|
| A1.5 | Recomputing goodput offline from each run's `--output-details` reproduces the printed value exactly. |
| A1.6 | T1 shows a knee: at the highest rate goodput is ≥ 30 % below its peak while output throughput is within 10 % of its own peak. |
| A1.7 | In at least one of T2, T3, T4 the configuration that is best by output throughput (T2, T4) or by mean TTFT (T3) is not the one that is best by goodput, with a goodput gap ≥ max(5 %, 3 σ). |

If A1.6 and A1.7 both fail, the metric adds nothing to percentiles on these tasks: PR-A is still
offered as parity with vLLM, and the write-up says plainly that no decision changed. A1.5 failing
blocks the PR.

Output: `results/pra_usage.md` — how to choose SLO values, the commands, how to read the block, the
T1–T4 tables — and the per-run summaries under `results/pra/`.

### 1.4 Opening PR-A
Unit tests and hooks green; A1.5 holds on recorded runs; the PR body carries the T1 curve and one
T2–T4 example; Bowen reviews and opens it. It does not wait for stage 3.

## 2. Stage 2 — PR-B research (done 2026-10-05)

[`PRB_RESEARCH.md`](PRB_RESEARCH.md). Summary: SGLang has global waiting and running timeouts only;
nobody has proposed a per-request form in sglang or vllm; Triton's queue policy is the precedent; the
field follows the path `priority` takes and is enforced in the existing rank-0 scan, under 150 changed
lines; two use cases where one global value cannot do the job. **Verdict: go to implementation.**

## 3. Stage 3 — PR-B: patch, debug, benchmark

### 3.1 Patch
Request field `waiting_timeout` (seconds); effective bound = the smaller of it and
`SGLANG_REQ_WAITING_TIMEOUT` when both are set; same clock and same abort response as the global
timeout (503 for non-streaming, HTTP 200 with an in-stream error for streaming); enforced in
`_poll_timeout_aborts`. Not in the first PR: the gRPC proto and the Rust front end, the running
timeout, a header form. The detailed, source-grounded version of this stage is
[`PRB_PLAN.md`](PRB_PLAN.md); where the two differ, that file wins.

### 3.2 Debug ladder
| step | where | Accept |
|---|---|---|
| D1 unit tests | a real SGLang environment (the node, CPU only — the Mac cannot import the package) | the cases of `PRB_PLAN.md` §5.3 beside `test_scheduler_timeouts.py`; the existing file still passes |
| D2 dummy-weight server | node, session P2 | upstream's mock-model flags, complete list in `PRB_PLAN.md` §7.3: a request with `waiting_timeout=1` behind a full batch is aborted after ≈ 1 s (503, or the in-stream error when streaming) and produces no token; one without the field is served. Greedy requests only (P1: the first sampled request stalls a fresh server) |
| D3 real model | node, session P2 | the same on Qwen3-8B, plus U3 below |

### 3.3 Benchmark
Client: `scripts/slo_client.py` in this repo (open loop, seeded, per-request class, SLO and
`waiting_timeout`; the stock benchmark sends one body to every request) with `scripts/slo_metrics.py`,
whose goodput must agree with PR-A's on a single-class run. Server: `--max-running-requests B*`, B\*
chosen in P1 from T2. Arms: `none` (no timeout) · `global` (the best of {2, 5, 10, 20} s for that use
case) · `per-request`. Same request list and arrival times on every arm; two seeds.

| id | Use case | Load |
|---|---|---|
| U1 | Surge, two classes at 50/50: interactive (TTFT ≤ 2 s, on the 100 ms token schedule) and batch (finish within 20–40 s) | 60 s cycle: 45 s at 0.6 λ\*, 15 s at 2.0 λ\*; three cycles |
| U2 | Callers with budgets: each request is useless after a budget drawn from {3, 10, 30} s | steady 1.3 λ\* |
| U3 | Control: field absent; field present with a bound that never binds | steady 0.8 λ\* |
| U4 | Neutral: one class, one SLO | steady 1.3 λ\* |

λ\* is the T1 knee. SLO attainment and its σ are computed as in BACKGROUND.md §3.4; "wasted tokens" are
prompt and output tokens the server processed for requests that then missed their SLO.

| # | Accept |
|---|---|
| A3.1 | U1: attainment(`per-request`) − attainment(`global`) ≥ max(5 pp, 3 σ). |
| A3.2 | U2: wasted tokens at least halved against `global`, and attainment ≥ + 5 pp. |
| A3.3 | U3: both variants within A/A noise of the unpatched server (± 2 % throughput, ± 2 pp attainment). |
| A3.4 | U4 is reported whatever it shows; the expectation is no difference from `global`. |
| A3.5 | Completed-token throughput with the field in use ≥ 98 % of `global` on U1. |

**Stop rule.** A3.1 and A3.2 both fail → no PR-B: the result is written up as "a tuned global timeout
is enough on this workload". One of them holds → PR-B is offered for that use case only. A3.3 failing
blocks the PR until fixed.

### 3.4 Opening PR-B
D1–D3 and A3.3 hold, at least one of A3.1 / A3.2 holds; the PR body carries the U1 / U2 table, the U3
control and the U4 result; Bowen reviews and opens it, after PR-A or together with it.

## 4. Server sessions

An assignment is at most 1 h plus two 1 h extensions and the home directory is wiped between
assignments, so each session rebuilds with `tools/radix/setup_node.sh` (current main needs
`FLASHINFER_VER=0.7.0.post1`) and installs the fork branch at a recorded SHA.

| session | content | estimate | cap |
|---|---|---|---|
| P1 | T1–T4 with PR-A | 2 h | 3 h |
| P2 | D2, D3, U3; B\* and λ\* re-checked | 1.5 h | 3 h |
| P3 | U1, U2, U4 (three arms, two seeds) | 2.5 h | 3 h |

Estimate ≈ 6 GPU-hours, cap 9. Before each session its exact plan (commands, cells, time budget,
stop conditions) is posted to Bowen; nothing is assigned before the answer. Operations as in Q3: tmux
on the node, driven from the Mac, no credentials or agents on the node, `127.0.0.1` only, results
synced every two minutes, release at the end.

## 5. Repository layout and commit rules

```
experiments/slo_sched/
  PLAN.md            this file
  BACKGROUND.md      paper and source facts, the full ladder (optional later study)
  PRB_RESEARCH.md    stage 2
  RUNBOOK.md         written before P1
  scripts/           slo_client.py, slo_metrics.py, runners, sync
  results/           summaries and reports (tracked); results/raw/ ignored
```

Commits are small and frequent, pushed as they are made, authored as Bowen, Conventional Commits, no
co-author or assistant attribution anywhere (this repo's convention; the same holds in the fork).
Nothing is posted upstream — issue, PR or comment — except by Bowen.

## 6. Risks

| risk | mitigation |
|---|---|
| The paper's SLO values do not bind on an H200 with an 8B model | T1 finds the knee empirically; if no rate in the sweep misses an SLO, the `tpot` and `e2el` values are halved once and the sweep is repeated |
| "Favourable use cases" read as cherry-picking | U3 and U4 are part of the design and of the PR body |
| The per-request bound and a client disconnect overlap in U2 | U2's clients keep the connection open, which is the case the disconnect path cannot see |
| Fork branch drifts from `upstream/main` | each branch is rebased before its PR is opened; sessions record the SHA they ran |
| Two open PRs plus #33726 | upstream's idle-PR cap is five; PR-A is opened first and kept moving |

## Amendment — 2026-10-05, before any P1 data

Written while the node was still installing its environment; no cell had run.

| id | change | reason |
|---|---|---|
| M1 | T3 compares `fcfs` and `hrrn` only; `lpm` is dropped | T3's prompts are random text with no shared prefixes, so `lpm` would order the queue like `fcfs` |
| M2 | T3 is two concurrent clients, each with its own SLO set: short (256-token prompts, 128 out, 0.6 × c0, `ttft:1000 tpot:100`) and long (12k-token prompts, 256 out, 2 req/s, `e2el:30000`) | the stock benchmark sends one workload per process; two processes are also how a mixed deployment would be measured |
| M3 | Load is defined from a closed-loop capacity probe c0 (concurrency 256, 1500 prompts) instead of a knee search: T1 sweeps {0.5, 0.8, 1.0, 1.25, 1.6, 2.0} × c0, T2 runs at 1.25 × c0, T4 at 1.5 × c0 with `--max-running-requests 128` | one probe costs a minute; a bisection would cost a quarter of the session |
| M4 | A1.5 compares good-request **counts**; the offline value uses `latency = ttft + sum(itl)` | the benchmark's own latency is taken at the last stream chunk, which can be a few milliseconds after the last token, so a borderline request can differ; any difference is reported per cell |
| M5 | The "halve the SLOs and rerun" fallback of §6 becomes an offline re-evaluation | `--output-details` keeps per-request timings, so another SLO set needs no rerun |

## Outcome — stage 1, local part (2026-10-05)

PR-A is implemented on the fork: `bowenwan6/sglang`, branch `feat/bench-goodput` @ `d132f6739e`, one
commit on `upstream/main` @ `734cf3cf3b`: +134 lines in `python/sglang/benchmark/serving.py`, a
127-line unit test (`test/registered/unit/bench/test_bench_serving_goodput.py`) and 20 lines in
`docs/docs/developer_guide/bench_serving.mdx`.

- **A1.1–A1.3: pass, with a caveat.** The unit file's eight cases pass on the Mac, but through a stub
  harness that loads `serving.py` without the rest of the package, because the Mac's Python does not
  have SGLang's pinned dependencies. The first run inside a real SGLang environment is step 0 of P1.
- **A1.4: partly.** `ruff format`, `ruff check` (v0.15.1), `isort` 7.0.0 and `codespell` 2.4.1 are clean
  on the changed files; the complete pre-commit suite has not been run.
- The flag parses end to end (`cli_main`), and the printed block and result fields were exercised on
  real metric objects.
- Against §1.1: the per-SLO attainments are one dictionary field, `slo_attainment_by_metric`, and the
  SLO values are echoed as `goodput_slos_ms`. Two cases were removed from the unit file to meet
  upstream's unit-test admission rule (a duplicate and a print-format mirror).
- Open: the P1 server tasks (A1.5–A1.7).

## Outcome — session P1 (2026-10-05, 02:12–03:19 UTC)

Full account: [`results/pra_usage.md`](results/pra_usage.md); tables:
[`results/pra/p1_report.md`](results/pra/p1_report.md). One H200, `Qwen/Qwen3-8B`, fork branch at
`d132f6739e`, 19 cells, every cell with `--output-details`.

- **Step 0 (real environment): pass.** 10 unit tests, `ruff`, and the full pre-commit suite on the three
  files. A1.1–A1.4 now hold without the Mac caveat.
- **A1.5: 17 of 19 cells identical.** The two that differ are the T4 cells with a waiting timeout, and the
  cause is a benchmark bug, not a rounding issue: a streaming request the server aborts arrives as HTTP
  200 with an in-stream error, and the chat and native paths recorded it as a completed request with the
  requested output length (735 phantom successes of 3017 and 25 % phantom output tokens in the 2 s
  cell). Fixed in `9eb681bef6` on the same branch with a regression test that fails on the old code;
  that commit has run only through the Mac stub harness.
- **A1.6: PASS.** Output throughput rises to 9244 tok/s at 2.0 × c0 while goodput falls from 35.3 to 7.6
  req/s (−79 %). Noise from three runs at 1.25 × c0: σ = 5.6 %.
- **A1.7: met in T4 once aborted requests are counted as failed; not met on the stock output, and not
  in T2 or T3** (corrected later on 10-05, see below). In T2 and T3 the winner by throughput (or mean
  TTFT) is also the winner by goodput; goodput added magnitude and cause instead: caps 128 and 512 are
  1 % apart in throughput and 2.4× apart in goodput; under `hrrn` the per-SLO line shows TPOT, not TTFT,
  is what still fails.
- **T4 with the fix applied offline:** no timeout 7.37 req/s, 2 s 22.95, 10 s 8.88, with output throughput
  at 7362, 7085 and 7162 tok/s — the three are within 3.8 %, one σ of throughput, so throughput does not
  rank them and goodput separates them 3.1×. This is T4's stated expectation ("throughput barely moves,
  goodput does"). A global bound helps only when it matches the tightest objective, which is PR-B's
  motivation.
- **Side finding:** the first sampled (top-k / top-p) request on a fresh server was scheduled about 70 s
  late and blocked everything behind it; greedy requests are unaffected. Sessions P2 and P3 must use
  `temperature: 0` or warm the sampler first.
- **Deviations:** the event monitor on the Mac delivered nothing for 30 minutes (the run itself was
  healthy); `scripts/p1_sync.sh` failed on a quoting error and the sync was done with the same `rsync`
  command by hand; five short diagnostic servers were run after the tasks to capture the abort
  responses. One extension was requested (1 credit) and turned out unnecessary for the tasks.

**Consequences.** PR-A is two commits: the goodput option and the in-stream-error fix, which can be
offered separately but must land first or together. Its description claims the knee and the per-SLO
breakdown, not a changed winner. Before it is opened: one run of the fix commit's unit file in a real
environment (first step of P2).

### Corrections to this outcome — 2026-10-05, after a second reading of the data

Made while answering "did the experiment show the metric is useful"; no new run.

1. **A1.7 was judged on the wrong T4 numbers.** `p1_report.py` compared the T4 settings on the
   benchmark's stock output, which the abort-accounting bug inflates (the 2 s cell read 9427 tok/s).
   On the corrected accounting the nominal winner by output throughput is "no timeout" and the winner
   by goodput is the 2 s bound, with a 68 % goodput gap against a 16.7 % threshold: A1.7 is met in T4 by
   its letter. The throughput ranking is inside the noise (3.8 % apart, σ = 3.8 %), so the supported
   statement is that throughput cannot separate the settings, not that it picks the wrong one. The
   script now prints A1.7 for both accountings and the throughput σ.
2. **The in-stream-error fix overlaps an open upstream PR.** sgl-project/sglang#40881 (2026-09-23, no
   review by 10-05) fixes the chat path the same way; the stage-2 prior-art search did not look for
   benchmark fixes and missed it. Checked on the old code with the stub harness: chat and native (the
   default backend) record a success, completions already records a failure with a traceback as its
   message. So what remains ours is the native path plus the real-server reproduction. The fix is not
   to be opened as a competing PR, and "must land first or together" above no longer holds: `--goodput`
   is independent of the fix (aborted requests inflate it exactly as they already inflate throughput).
3. **What the evidence does and does not show.** Shown: the numbers are computed correctly; the knee
   (T1); throughput blind to a 3× difference in useful work (T4, offline re-count, one run per setting).
   Not shown, and not showable: a case where goodput contradicts the latency columns — it is a summary
   of the same per-request latencies. Its additions are the joint share (69.6 % where the three SLOs
   are met by 100, 91 and 79 % one at a time) and failed requests counted as misses.
4. **New limits recorded in `results/pra_usage.md` §5:** every cell sends for 50 s, so attainment above
   saturation depends on run length; goodput in req/s is noisier than attainment (σ 5.6 % against
   1.7 %).

Post hoc, three recorded T2 runs under other SLO sets (`pra_usage.md` §2.2): with `ttft:10000 tpot:30`
cap 128 beats cap 512 2.5× while throughput and mean TTFT point at cap 512. Chosen after seeing the
data; an illustration, not a test.

**Revised next steps.** (a) PR-A is the `--goodput` commit alone, rebased, with T1 and the T2 per-SLO
table in its description. (b) The fix: Bowen decides between commenting on #40881 with the native-path
gap and our reproduction, or waiting for it and following up with the native path. (c) P2, if approved,
also reruns T4 three times with the fixed benchmark, so that T4's numbers come from a run and carry
their own σ.

## Amendment — 2026-10-06, before any PR-B data

Written after re-reading stage 3 against a queue model; no PR-B code had run and no node was assigned.

**What was wrong.** [`scripts/prb_sim.py`](scripts/prb_sim.py) (tables:
[`results/prb/sim.md`](results/prb/sim.md)) models the server as 128 slots and a waiting queue. With
the default first-come-first-served order a per-request bound cannot beat a global bound tuned to the
tightest class: total attainment 75.2 % against 77.1 % in U1 as first planned, and 66.6 % against
77.6 % in steady overload. Capacity is the limit either way; under FCFS the requests that are allowed
to wait stay in the queue, age to its head and take the slots, so all the shedding moves to the
impatient class — and total attainment even falls when the patient class is the expensive one. A3.1
and A3.2 as written could not have held. The field does win where the queue is ordered by priority:
low-priority requests are made to wait on purpose, and no single bound suits both them and the
requests that must not wait.

| id | change | reason |
|---|---|---|
| N1 | **U1 is now "chat bursts above capacity, steady batch, priority to chat".** Server: `--enable-priority-scheduling --disable-priority-preemption --max-running-requests 128`. Chat (`priority` 1): 0.3 × c_chat, rising to 1.5 × c_chat for the last 20 s of every 60 s. Batch (`priority` 0): 0.5 × c_batch throughout. Three cycles (180 s). c_chat and c_batch are the closed-loop capacities of each class alone, measured in P2 | the case in which one bound must fail one class. Model: per-request 83 % against 75 % for the best global value; batch 98 against 67; chat equal |
| N2 | **U2 is now "steady chat and a batch burst, FCFS", and is a reported result, not a gate.** Chat 0.4 × c_chat throughout; batch 2.0 × c_batch for 15 s of every 60 s; default scheduling flags | what the field does not do has to be in the PR text. Model: a tie in total (72.9 against 72.4), chat lower (56 against 87), batch higher (100 against 49) |
| N3 | Arms: `none`; `global` ∈ {1.5, 5, 20, 30} s; `per-request` (chat 1.5 s, batch 30 s); `per-request, chat only`. The global grid contains both per-request bounds | the first grid {2, 5, 10, 20} lacked the chat bound, which would have handed the per-request arm an edge it had not earned; P1 showed that a bound equal to the 2 s objective is too late by the prefill time |
| N4 | Classes: chat = prompt of about 256 tokens, 128 output tokens, `ttft:2000 tpot:100`; batch = prompt of about 1024 tokens, 256 output tokens, `e2el:40000`. Greedy, `ignore_eos`, no two prompts alike, the same seeded request list on every arm | the work per request is fixed, so arms are paired and the radix cache cannot help one arm |
| N5 | **U4 is now an equivalence check**: one class, every request carrying `waiting_timeout` 1.5 on a server without the global knob, against `SGLANG_REQ_WAITING_TIMEOUT=1.5` and no field | same clock and same rule should shed the same requests |
| N6 | Optional arm in U1 if time allows, `hang-up`: no field; the client closes the connection when no first token has arrived after the class bound | the alternative available today (PRB_PLAN.md §13 S9) |
| N7 | Acceptance for stage 3 is replaced by the table below | |
| N8 | D2 uses plain `--load-format dummy`; the token-oracle and KV-canary flags are dropped | they check KV integrity, which this patch does not touch, and add ways for the ladder to fail |
| N9 | Sessions: **P2** = D1, D2, D3, the client checked against `bench_serving --goodput`, c_chat and c_batch, U3, U4, the PR-A extras (repeats of three T1 points; T4 three times with the fixed benchmark), a one-seed pilot of U1. **P3** = U1 (7 arms × 3 seeds), U2 (5 arms × 2 seeds). Bowen's allowance for 2026-10-06: 10 GPU-hours, one to two hours per assignment; each session's plan is still posted before the node is assigned | |

| # | Accept (replaces A3.1–A3.5) |
|---|---|
| A3.1′ | U1: batch attainment(`per-request`) − batch attainment(`global` 1.5) ≥ max(10 pp, 3 σ); chat attainment(`per-request`) ≥ chat attainment(`global` 1.5) − max(3 pp, 3 σ); total attainment(`per-request`) − the best global total ≥ max(3 pp, 3 σ). σ is pooled over the three seeds of those two arms |
| A3.2′ | U2 is reported whatever it shows. If `per-request` beats the best global total by more than 3 σ there, the model is wrong and the cause is found before any claim is made |
| A3.3 | U3, on fresh server processes: the patched build with the field absent, and with the field present but loose (3600 s), is within the A/A spread of the unpatched build (or ± 3 %) on mean and p99 TTFT, mean TPOT and output throughput |
| A3.4′ | U4: attainment and the number of refused requests agree within max(2 pp, 3 σ) between the two forms |

A3.5 is dropped: completed-token throughput is not comparable between arms that serve different
mixes of the two classes.

**Stop rule.** A3.1′ fails → PR-B is not offered with a performance claim; Bowen decides whether it is
offered as an API change at all. A3.3 failing blocks the PR until fixed.

## Outcome — stage 3 (sessions P2 and P3, 2026-10-06)

Full account: [`results/prb_results.md`](results/prb_results.md); tables:
[`results/prb/report.md`](results/prb/report.md). `Qwen/Qwen3-8B`, H200, SGLang `734cf3cf3b` plus the
patch. P2: 05:17–07:13 UTC, one GPU, one extension. P3: 12:44–14:29 UTC, two GPUs (Bowen asked for a
tensor-parallel run), one extension.

- **D1–D3: pass.** Unit files in a real environment on the pin and on current `upstream/main` (with the
  full pre-commit suite); the ladder on dummy weights, on the real model, with the global bound and
  against the unpatched build, on one GPU and with `--tp-size 2`. A 1 s bound is enforced at
  1.00–1.02 s on all three endpoints.
- **A3.1′: pass.** U1, 19 runs of about 10 000 requests: per-request 82.6 % in total (chat 79.0, batch
  100.0) against 77.0 % for the best global bound, 1.5 s (78.9, 67.5); the other global values and no
  bound give 35–42 %. Bounding only the chat class does as well (82.5 %); a client that hangs up
  instead reaches 75.0 %.
- **A3.2′: reported.** U2 under FCFS: per-request 71.1 % against 78.5 % for the global 1.5 s bound —
  batch 100.0 against 52.7 %, chat 60.8 against 87.7 %. No gain, as the model said; a loss of 7.4 pp
  in total because the batch class is 3.4 times as expensive as chat here.
- **A3.3: pass**, on the first version and on the final build. **A3.4′: pass** (82.4 / 80.5 % as a field,
  82.2 / 80.3 % as the global knob).
- The queue model's predictions for U1 were within 2 pp of the measurements on every arm.

**Deviations from the amendment.**

| What | Why |
|---|---|
| The scan was rewritten during P2 (`4e04090272`): a request is dropped once it has outstayed either bound | measured on the node, the first version cost servers that use only the global bound 56 % more per scan; the rewrite costs 3 %. Same rule (150 combinations compared), the ladder passed again, and the pilot's per-request run repeated on it gave 82.4 against 82.5 % |
| Nine U1 runs ran in P2's assignment, on the rewritten scan loaded from a second source tree | the hour was already paid for; the tree each server imports is checked before it starts |
| P3 ran on a two-GPU assignment; a tensor-parallel ladder and two load runs were added | Bowen's request. With the usual batch cap two GPUs absorbed the load and nothing was refused, so two more runs with `--max-running-requests 32` were added: 2059 and 2034 of 5127 requests refused, no hang |
| U1b added (chat bursts at 2.0 of capacity; five runs) | to see whether U1 depends on the burst height. Not in the amendment; decided before its data. Chat equal, batch +32.7 pp, total +4.5 pp |
| U3 repeated on the final build | the first U3 ran on the first version of the scan |
| U2's global 30 s run was skipped by a reordering of the runner's phases and run last | my error in the stop condition; same server flags, same seed |
| P2's background unit sweep started 25 minutes late | it died under `set -u` when conda activated; restarted by hand, fixed in `prb_node.sh` |
| Seeds: three for the two arms of A3.1′, two for the others, one for "no bound" in U2 and U1b | time |

**Not done.** PD mode, the msgpack IPC path, the Rust front end and `sgl-model-gateway`, more than two
GPUs. A second model.

**Consequences.** PR-B's description (`upstream/pr_b/PR_DESCRIPTION.md`) claims the priority use case
and states the FCFS result as what the field does not do. Nothing is opened: Bowen reviews PR-A and
PR-B together.
