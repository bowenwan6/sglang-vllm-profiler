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
`SGLANG_REQ_WAITING_TIMEOUT` when both are set; same clock, same 503 response as the global timeout;
enforced in `_poll_timeout_aborts`. Not in the first PR: PD mode (waits on #34457), gRPC and the Rust
front end, the running timeout, a header form.

### 3.2 Debug ladder
| step | where | Accept |
|---|---|---|
| D1 unit tests | Mac | new cases beside `test_scheduler_timeouts.py`: per-request bound fires, the smaller-of-two rule, field absent = today's behaviour; the existing file still passes |
| D2 dummy-weight server | node, session P2 | upstream's mock-model flags (`--load-format dummy --sampling-backend token_oracle`, Qwen3-0.6B): a request with `waiting_timeout=1` behind a full batch returns 503 after ≈ 1 s and produces no token; one without the field is served |
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
