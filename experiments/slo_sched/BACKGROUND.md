# S1 background — JITServe and SGLang

> **Status: written 2026-10-04; trimmed on 2026-10-22 to the paper and source facts.** The track's plan
> and outcomes are in [`PLAN.md`](PLAN.md). This file keeps what its PRs built on: the paper and its
> artifact (§1) and SGLang as of early October (§2).
> Paper: [arXiv 2504.20068v3](https://arxiv.org/abs/2504.20068) (NSDI '26).
> Artifact: [`UIUC-MLSys/JITServe`](https://github.com/UIUC-MLSys/JITServe) @ `84584a11` (Apache-2.0).

## 0. The question, in one paragraph

JITServe reports 1.4–6.3× more "service goodput" than existing schedulers on mixed-SLO traffic by
giving every request just enough serving bandwidth to meet its own SLO. Nothing of it exists in SGLang
(zero hits for "JITServe" in upstream issues and PRs; no per-request deadline in the engine; the word
"goodput" does not occur in the tree). Porting it whole is not realistic (§2.3). The PRs of this track
took the two smallest pieces: a goodput figure for the benchmark and for the server (PR-A, PR-C), and a
per-request waiting bound (PR-B, dropped after measurement).

## 1. What the paper does and claims (read from the paper text, its figures and the artifact source)

| Fact | Where |
|---|---|
| Three request types. Latency-sensitive: token *i* counts if delivered by `TTFT_SLO + i·TBT_SLO`. Deadline-sensitive: all tokens (input + output) count if the request finishes by its deadline, else zero. Compound: all sub-request tokens count if the last call finishes by the end-to-end deadline | §3 |
| Priority of a request = `goodput(r) / t_gen(r)`, `t_gen` = remaining-length upper bound × per-token time; a small additive δ per frame prevents starvation | §4.2 |
| Batch of fixed size B per frame: keep requests with priority ≥ p × (B-th highest), sort by input length, take the size-B window with the largest priority sum (GMAX) | §4.2, Alg. 1 |
| Decisions change only at frame boundaries, Δ = 50 decode steps (≈ 300 ms); preemption only when the projected gain exceeds `stall × token rate`; reported overhead < 1 % | §4.2 |
| Length upper bound: quantile regression forest on the prompt, re-run every 50 generated tokens; 7 ms per prediction. The artifact asks for the 0.95 quantile | §4.1; `request_analyzer/prediction.py` |
| Admission control: a request not scheduled within `waiting_time` (5 s in the example) is **dropped** | §5 |
| Built on vLLM, ≈ 2,800 lines. The artifact vendors a vLLM snapshot (`fa0f12170`) and subclasses the **V0** scheduler (`SequenceGroup`, waiting/swapped/running queues, `PreemptionMode.SWAP`/`RECOMPUTE`) | §5; `scheduler/slo_scheduler.py` |
| Evaluation: 16 × A100; Llama-3.1-8B, Qwen2.5-14B, Qwen3-30B-A3B, Llama-3.1-70B; arrivals from a Microsoft trace scaled to the cluster; > 10 K requests over ≥ 1 h; five runs; SLOs TTFT ≈ 2 s, TBT ≈ 100 ms, deadline 20 s, compound 20 s × stages; mix 1:1:1 | §6.1 |
| Artifact metric details the paper does not state: token goodput weights a decode token **8×** a prefill token; every request's SLO is the base SLO × (`collection_id % 4 + 1`); a latency request "meets" its SLO only if TTFT and **all** TBTs do | `benchmark/benchmark_scheduler.py` |
| Headline: 1.4–6.3× goodput, or 28.5–83.2 % fewer resources at equal goodput; within 3–9 % of an oracle; 96–98 % of Sarathi-Serve's throughput | §6 |

Numbers read from the figures (bar labels are exact, line plots ± 0.1 k):

| Figure | Reading |
|---|---|
| Fig. 17, request goodput (req/s) | oracle 3.23 · JITServe 3.17 · without the Request Analyzer (mean length instead) 2.91 · without GMAX (SJF on the estimates) 2.70 · Sarathi-Serve 1.35 |
| Fig. 17, token goodput (tok/s) | 7808 · 7637 · 6893 · 6080 |
| Fig. 15, Llama-3.1-8B, token goodput at 4.0 → 6.0 RPS | JITServe 7.5 k → 7.7 k · LTR 7.1 k → 6.2 k · Sarathi-Serve 6.7 k → 3.8 k · vLLM 5.7 k → 3.3 k |
| Fig. 20, token-goodput gain by mix | latency-only 1.72× · deadline-only 1.76× · latency/deadline 66/33 (no compound) 1.67× · 33/66 (no compound) 1.81× · 1:1:1 2.08× · compound-only 1.19× |
| Fig. 21, JITServe over EDF | request goodput ≈ 2.4× (8B), ≈ 2.2× (70B); token goodput ≈ 10× and ≈ 4× |

Four readings of the evaluation:

1. **Most of the gain over FCFS-type baselines needs neither GMAX nor a good predictor.** "Without GMAX"
   is already 2.0× Sarathi-Serve; the analyzer is worth 8–10 %, GMAX 15–20 %, perfect information 2 %.
2. **The gain is an overload phenomenon.** At the lightest load shown JITServe leads the best baseline
   by ≈ 5 %; the multiples come from baselines collapsing as load rises.
3. **EDF without shedding collapses** (Fig. 21, and Appendix D.1 proves it is not competitive).
4. **Compound requests are not needed** for the effect: two-class mixes gain 1.7–1.8×.

**Re-read on 2026-10-13 (paper text of arXiv v3 and the artifact @ HEAD), after PR-B was dropped.**

- `waiting_time` is a **per-request API parameter**, next to `deadline`, `target_ttft` and `target_tbt`
  (§5: `client.responses.create(model, input, deadline=None, target_tbt=0.2, target_ttft=5,
  waiting_time=5)`); "requests unscheduled beyond it are dropped". PR-B was exactly this parameter.
- The paper text has no rule that drops a request the moment it is predicted to miss. What is computed
  ahead of time is the priority: in the artifact (`jitserve/scheduler/policy.py`, `SLOPolicy`) the
  reward of a request is multiplied by `(deadline / predicted_finish) ** penalty_factor`, the ratio
  clamped to [1e-6, 1], and priority = reward / remaining time. A request predicted to finish late
  sinks in the order, is not scheduled, and then falls to `waiting_time`. `slo_scheduler.py` also aborts
  every pending request when nothing has run for 2000 iterations (`_stop_dull_scheduling`).
- Predicted finish = time already served + remaining length upper bound × measured time per scheduling
  step; priorities are cached and recomputed every `schedule_interval` = 50 scheduler calls.
- §6.2: JITServe reaches 96–98 % of Sarathi-Serve's throughput. It trades a little throughput for
  goodput under load; nothing in it makes the engine faster.
- §4.2, Fig. 8: the batch-composition half of GMAX rests on one observation made on vLLM, that a batch
  with heterogeneous input lengths decodes slower per token. Not tested on SGLang.

## 2. Where SGLang is

### 2.1 Mechanism facts (read from source, `sgl-project/sglang` `upstream/main` @ `50be533d09`, 2026-10-01)

| Fact | Where |
|---|---|
| The waiting queue is re-sorted once per prefill admission pass by `SchedulePolicy.calc_priority`; policies: `fcfs`, `lpm`, `dfs-weight`, `hrrn`, `shortest-prefill-first`, `lof`, `random`, `routing-key` | `srt/managers/schedule_policy.py` |
| Per-request `priority: int` exists on chat, completion, responses, embedding and classify requests. With `--enable-priority-scheduling` and `fcfs` the queue sorts by `(priority, arrival)`; `--schedule-low-priority-values-first` flips the direction; priority scheduling requires `fcfs` or `lof` | `entrypoints/openai/protocol.py`; `_sort_by_priority_and_fcfs`; `arg_groups/validation_hook.py:200` |
| Preemption happens only when the running batch is full and a waiting request outranks a running one by more than `--priority-scheduling-preemption-threshold` (default 10); by the loop's exit condition it takes one victim per admitted request when memory is not the limit (as read; A0b.5 checks it) | `PrefillAdder.preempt_to_schedule` |
| A preempted or retracted request loses its KV: `release_kv_cache(req, tree_cache, is_insert=False)`, with `# TODO (csy): for preempted requests, we may want to insert into the tree`; it is re-queued and re-prefills prompt + generated tokens | `managers/schedule_batch.py` `release_req` |
| Every running request decodes on every step; prefill runs first whenever a prefill batch can be formed. There is no per-step subset of the running batch | `Scheduler.get_next_batch_to_run` |
| Load shedding exists only globally: `SGLANG_REQ_WAITING_TIMEOUT` / `SGLANG_REQ_RUNNING_TIMEOUT` (seconds, default −1 = off), answered with 503 | `srt/environ.py:637`; `Scheduler._poll_timeout_aborts` |
| Wall-clock decisions are taken on rank 0 only and broadcast: "every rank must drop the same requests in the same iteration, or the extend-vs-decode decision splits and the collectives hang" | `Scheduler.ingest_requests`, `_poll_timeout_aborts` |
| A queued request whose client disconnects is noticed only by a poll every `SGLANG_REQUEST_STATE_WAIT_TIMEOUT` (default 4 s). `POST /abort_request {"rid": …}` removes a request, queued or running, at once; the client may set `rid` on chat and completion requests | `tokenizer_manager.py:190,1862`; `http_server.py` `abort_request`; `Scheduler.abort_request`; `protocol.py:968` |
| `sglang.benchmark.serving` reports TTFT / TPOT / ITL percentiles; it has no goodput or SLO attainment and its `--extra-request-body` is one value per run, so it cannot send a per-request priority or SLO | `python/sglang/benchmark/serving.py` |
| Qwen3-VL carries scheduler-level overrides (`prefill_decode_interval = 22`, a `max_running_requests` adjustment); no file under `arg_groups/model_overrides/` targets dense Qwen3 | `model_overrides/qwen3_vl.py:42,127` |

**Errata to the table above (2026-10-05, from `PRB_PLAN.md` and session P1).** `/abort_request` matches
rids by prefix, so client-chosen rids must be prefix-free. Load shedding has a second
global knob besides the waiting timeout: `--max-queued-requests`. The abort a waiting timeout produces
is a 503 only for non-streaming requests; a streaming request gets HTTP 200 with the error inside the
stream, so a client must classify by payload. The first sampled request on a fresh server stalled the
scheduler for about 70 s on the node; greedy requests avoid it.

### 2.2 Where upstream is heading (GitHub, as of 2026-10-04)

- **SLO and admission policy are being built in the router, not the engine.** `experimental/sgl-router`
  got SLO-ordered bucket selection on 09-22 ([#40292](https://github.com/sgl-project/sglang/pull/40292),
  headers `x-sgl-ttft-slo-ms`, `x-sgl-tps-slo`; the targets order buckets of engines and are not
  forwarded to the engine). The same author opened per-group admission, pressure signals and
  admission-aware power-of-two on 10-03 (#42428, #42429, #42432) and merged ≈ 15 router PRs in the ten
  days before.
- **Engine queue policies get merged when small, opt-in and backed by production numbers.** HRRN
  ([#32911](https://github.com/sgl-project/sglang/pull/32911), +220 lines) took 40 days;
  `shortest-prefill-first` (#40024, +281 lines) merged in one day with no review thread. Paper-derived
  or larger ones stall: UniBoost (#35637, ICML '26, +1338 lines) has no review after six weeks; #13507
  has three approvals and has been open since 2025-11; #34011 (token-aware admission cap) since 08-07;
  a bounded-lookahead PR (#42456) was closed the day it was opened.
- **PD-Multiplexing** is the engine-side latency work: it protects ITL while prefilling
  ([#10813](https://github.com/sgl-project/sglang/issues/10813), evaluated against a 60 ms ITL target;
  several PRs on 10-03/04). It does not look at per-request SLOs.
- **Priority scheduling's own roadmap** ([#13526](https://github.com/sgl-project/sglang/issues/13526)
  and [#8743](https://github.com/sgl-project/sglang/issues/8743)) was auto-closed with "Analyze
  preemption cost" and "Profile the impact of preemption" unchecked and priority-aware batching at
  "design WIP".
- **A goodput metric for the benchmark** was offered once
  ([#5495](https://github.com/sgl-project/sglang/pull/5495)); a maintainer answered "could you rebase
  the changes? Then we can merge it", and the PR was closed for inactivity on 08-20.
- Upstream closes idle PRs automatically and caps idle PRs per author at five.

### 2.3 JITServe mechanism → SGLang today

| JITServe mechanism | SGLang today |
|---|---|
| Per-request SLO in the API | integer `priority` only |
| `waiting_time` drop | one global timeout |
| Goodput-density ordering with a slot-bound batch | `--max-running-requests` + priority sort |
| Preemption at frame boundaries, cost-aware, KV swapped | preempt on arrival of a higher priority, KV dropped |
| Length upper bound, refined online | none |
| Just-in-time pacing (less than one slot per request) | none |
| Input-length grouping, compound requests, power-of-K replicas | none / router territory |
