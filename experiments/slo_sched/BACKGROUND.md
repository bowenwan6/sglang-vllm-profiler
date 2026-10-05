# S1 background — is JITServe-style SLO-aware scheduling worth bringing to SGLang?

> **Status: written 2026-10-04, not run. Since 2026-10-05 this is the background document, not the
> active plan.** The active plan is [`PLAN.md`](PLAN.md): two upstream PRs, a goodput benchmark (PR-A)
> and a per-request waiting timeout (PR-B). This file keeps what the PR track builds on — the paper
> and source facts (§1–§2) and the full client-emulated ladder with its hypotheses (§3–§8), which
> stays available as a later, optional study and is where deadline-aware ordering (PR-C) would be
> tested. Nothing below has touched a GPU. Branch `exp/slo-prs`.
> Paper: [arXiv 2504.20068v3](https://arxiv.org/abs/2504.20068) (NSDI '26).
> Artifact: [`UIUC-MLSys/JITServe`](https://github.com/UIUC-MLSys/JITServe) @ `84584a11` (Apache-2.0).

## 0. The question, in one paragraph

JITServe reports 1.4–6.3× more "service goodput" than existing schedulers on mixed-SLO traffic by
giving every request just enough serving bandwidth to meet its own SLO. Nothing of it exists in SGLang
(zero hits for "JITServe" in upstream issues and PRs; no per-request deadline in the engine; the word
"goodput" does not occur in the tree). Porting it whole is not realistic (§2.3). S1 asks a narrower,
testable question: **on SGLang, under transient and sustained overload with two SLO classes, how much
SLO attainment is lost today, how much of it comes back with knobs that already exist, and which single
engine primitive — per-request shedding, deadline-aware ordering, length information, or decode
pacing — carries the rest.** Each primitive gets its own measured increment and its own go/no-go, so
the outcome is either a small, evidence-backed upstream proposal or a recorded negative result.

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

Four readings that shape this plan:

1. **Most of the gain over FCFS-type baselines needs neither GMAX nor a good predictor.** "Without GMAX"
   is already 2.0× Sarathi-Serve; the analyzer is worth 8–10 %, GMAX 15–20 %, perfect information 2 %.
2. **The gain is an overload phenomenon.** At the lightest load shown JITServe leads the best baseline
   by ≈ 5 %; the multiples come from baselines collapsing as load rises.
3. **EDF without shedding collapses** (Fig. 21, and Appendix D.1 proves it is not competitive).
4. **Compound requests are not needed** for the effect: two-class mixes gain 1.7–1.8×.

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
rids by prefix, so the client-chosen rids of the ladder must be prefix-free. Load shedding has a second
global knob besides the waiting timeout: `--max-queued-requests`. The abort a waiting timeout produces
is a 503 only for non-streaming requests; a streaming request gets HTTP 200 with the error inside the
stream, so a client must classify by payload. The first sampled request on a fresh server stalled the
scheduler for about 70 s on the node; the ladder's requests are greedy, which avoids it.

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
  "design WIP". L1 measures the first two.
- **A goodput metric for the benchmark** was offered once
  ([#5495](https://github.com/sgl-project/sglang/pull/5495)); a maintainer answered "could you rebase
  the changes? Then we can merge it", and the PR was closed for inactivity on 08-20.
- Upstream closes idle PRs automatically and caps idle PRs per author at five.

### 2.3 JITServe mechanism → SGLang today → how this plan tests it

| JITServe mechanism | SGLang today | In this plan |
|---|---|---|
| Per-request SLO in the API | integer `priority` only | emulated: the client turns class, deadline and length into a priority key (L1); a real field is primitive P1/P2 (L2) |
| `waiting_time` drop | one global timeout | `fcfs+gdrop` uses the global knob; per-request shedding is emulated by the client calling `/abort_request` at the request's own latest start time (L1); server-side it is P1 |
| Goodput-density ordering with a slot-bound batch | `--max-running-requests` + priority sort | emulated with static keys fixed at arrival (L1); dynamic re-evaluation is not expressible and is noted as a limit |
| Preemption at frame boundaries, cost-aware, KV swapped | preempt on arrival of a higher priority, KV dropped | preemptive vs non-preemptive arm; preemption count and re-prefilled tokens measured (L1) |
| Length upper bound, refined online | none | oracle length vs class-mean length, the same contrast as Fig. 17's analyzer ablation (L1); a predictor is out of scope |
| Just-in-time pacing (less than one slot per request) | none | bounded analytically from the measured step-time curve (H9); no prototype in this plan |
| Input-length grouping, compound requests, power-of-K replicas | none / router territory | **out of scope**, with reasons in §3.5 |

## 3. Scope, workload, SLOs, metrics

### 3.1 Stack
RADIX H200 node, `tools/radix/setup_node.sh` with `SGLANG_REF` = the `upstream/main` SHA current when
L0b starts (recorded; `check` re-greps the ten facts of §2.1 and stops on any mismatch). Model
`Qwen/Qwen3-8B`, served from its snapshot path with `HF_HUB_OFFLINE=1`. Reasons: dense and public (the
paper's Llama-3.1-8B is gated, and the node allows no token); no model-specific scheduler override,
unlike Qwen3-VL; same size class as the paper's smallest model. No engine patch in L0–L1. If no log line
or counter shows preemptions at the pin, a measurement-only patch adds one line per preemption (never
upstreamed, the Q3 rule).

### 3.2 Workload
Two classes, 50 % each, drawn with a fixed seed; the **same request list and arrival times on every arm**.
- **L (streaming chat)**: TTFT ≤ 2 s and token *i* by `arrival + 2 s + i · 100 ms`.
- **D (deadline)**: finished by `arrival + D_i`, `D_i ∈ {10, 20, 40} s` equiprobable. Deadlines differ
  per request because with one common deadline EDF inside the class is FCFS and the test is void (the
  artifact also spreads SLOs 1–4×).
- Prompts: ShareGPT first turns ≤ 1024 tokens from the lab's file (public; the artifact's LMSYS-derived
  trace comes from a gated dataset and is not used). Output length `L_i` ~ lognormal with median 225 and
  P95 1024 (the paper's Table 2 chatbot row), clipped to [16, 1536]; `ignore_eos` with
  `max_tokens = L_i`, greedy, so the true length is known to the client and identical across arms.
- If more than 15 % of D requests cannot meet their deadline alone at B\* (measured in L0b), all
  deadlines are multiplied by 1.5, once. This rule is fixed now.

### 3.3 Scenarios (rates relative to the knee λ\* of §4, L0b)
| id | arrival process | scored window | role |
|---|---|---|---|
| S1 | Poisson, 0.8 λ\* | 150 s | no-regression check |
| **S2** | 60 s cycle: 45 s at 0.6 λ\* then 15 s at 2.0 λ\* (mean 0.95 λ\*), 3 cycles | 180 s | **primary**: transient overload |
| S3 | Poisson, 1.5 λ\* | 150 s | sustained overload, the paper's RPS axis |

20 s warm-up at 0.6 λ\* before each window (not scored). After the window arrivals stop; a scored request
still unfinished 45 s later counts as missed.

### 3.4 Metrics (per cell)
- **Primary: SLO attainment** = scored requests that met their SLO / scored arrivals; overall and per
  class. Dropped, aborted and cancelled requests are misses.
- Token goodput by the paper's text (L: tokens on schedule; D: input + output if met), and the
  artifact's 8×-decode variant as a secondary column. Request goodput in req/s.
- Completed output tokens/s, TTFT / ITL / E2E percentiles per class, cancellations, 503s, preemptions,
  re-prefilled tokens, retractions.
- σ_run: the run-to-run standard deviation of attainment, from L0b. An increment "counts" only if it is
  ≥ max(5 pp, 3 σ_run).

### 3.5 What this is and is not
One dense model on one GPU, TP = 1, text only, two request classes, client-emulated policies with keys
fixed at arrival. It can show that a primitive pays on SGLang; it cannot show that the full JITServe
does not. Out of scope and why: **compound requests** (need client-side call chains and the
pattern-graph matcher; Fig. 20 shows the effect without them); **the QRF predictor** (Fig. 17 bounds
its value at ≈ 10 %; L1 measures the same bound with an oracle); **input-length grouping** (a
vLLM-kernel observation, untested here); **multi-replica power-of-K** (router policy); PD
disaggregation, VLMs, multi-GPU (one TP = 2 smoke in L2 only).

## 4. Layers, acceptance criteria, stop rules

A layer starts only when the previous one is accepted. "Accept" lists must all hold.

### L0a — harness and simulator (Mac, 0 GPU-h)
Builds: `slo_client.py` (open-loop, seeded, per-request `rid`, priority and abort time, per-token
timestamps), `slo_metrics.py` (offline calculator), a mock OpenAI-compatible SSE server with a
slot-bound priority queue and an abort endpoint, `slo_sim.py` (discrete-event simulator of the L1
arms), the runner, `RUNBOOK.md`.

| # | Accept |
|---|---|
| A0a.1 | The calculator reproduces a hand-computed six-request fixture exactly (attainment, both token goodputs, per class). |
| A0a.2 | Against the mock server the client holds the arrival schedule within ± 20 ms for ≥ 99 % of requests at twice the highest planned rate, and counted tokens equal requested tokens for every request. |
| A0a.3 | The simulator reproduces the analytical mean wait of an M/M/c queue within 5 % on three configurations, and conserves requests (arrived = finished + dropped + in flight). |
| A0a.4 | The simulator's predicted attainment for every arm × scenario is written to a tracked `results/predictions.json` **before a node is assigned** (parameters τ, prefill rate and λ\* enter as symbols and are filled from L0b without touching the model). |
| A0a.5 | A dry run of the full L0b + L1 runner against the mock server completes and every cell passes the VERIFIED checks of §6. |

Stop: none. This layer costs no credits.

### L0b — calibration on the node (≈ 1.5 GPU-h, cap 2.5 h)
Measures: step time τ(B) and throughput X(B) = B/τ(B) for B ∈ {16, 32, 64, 128, 256, 512}
(closed loop, 512 output tokens); prefill rate; **B\*** = the largest grid B with ITL p99 ≤ 50 ms and
`T_prefill + τ(B)·L_P90 ≤ 20 s`; the **knee λ\*** = the highest steady rate at which `fcfs` at B\*
attains ≥ 95 % (bisection); the global timeout W ∈ {2, 5, 10} s that is best for `fcfs+gdrop` on S2;
σ_run from three repeats of that cell.

| # | Accept |
|---|---|
| A0b.1 | `check` passes: pinned SHA, §2.1 facts, model snapshot complete, resolved server config dumped. |
| A0b.2 | Live token accounting: per-chunk tokens sum to `usage.completion_tokens` = `L_i` for ≥ 99.5 % of completed requests. |
| A0b.3 | Client fidelity on the node: schedule within ± 20 ms for ≥ 99 % of requests at 2.0 λ\*, client CPU ≤ 70 % of one core (else the client is sharded and this is re-checked). |
| A0b.4 | The knee is bracketed: ≥ 95 % at λ\*, < 95 % at 1.15 λ\*. |
| A0b.5 | Engagement, forced test: with B\* slots held by D requests an arriving L request is served ahead of older D requests, and exactly one D stream stalls on the preemptive arm and none on `fcfs`; `/abort_request` removes a queued request (it never produces a token and frees no slot late); the global timeout answers 503. |
| A0b.6 | σ_run ≤ 3 pp. If larger, S2 gets a third seed (+ 0.6 GPU-h) and the 3 σ_run rule stands. |
| A0b.7 | Workload sanity: on S1 `fcfs` attains ≥ 95 %; the share of D requests infeasible alone is ≤ 15 % (else the ×1.5 rule of §3.2). |

Stop: the knee cannot be bracketed, or the client cannot hold the schedule after sharding, or
A0b.5 fails (then the finding is that the existing knob does not behave as its code reads, and that
is reported instead of L1).

### L1 — the zero-engine-change ladder (≈ 3.2 GPU-h over two node sessions, cap 5 h)
Seven main arms × (S1 one seed, S2 two seeds, S3 one seed) + two side arms and two
threshold-sensitivity cells on S2 = 32 cells. Arms and hypotheses in §5. Three increments are computed
on **S2**, each tied to one engine primitive:

| increment | definition | primitive it argues for |
|---|---|---|
| G_shed | `prio2+shed` − max(`fcfs+gdrop`, `prio2`) | P1 — a per-request waiting deadline (server-side shedding) |
| G_key | `edf+shed` − `prio2+shed` | P2 — deadline-aware ordering of the waiting queue |
| G_len | `llf-oracle+shed` − `edf+shed` | P3 — length information (upper bound on any predictor) |

| # | Accept (the layer is valid) |
|---|---|
| A1.1 | Every cell VERIFIED (§6); no cell flagged MEM-BOUND (zero retractions at B\*). |
| A1.2 | H7 holds: on S1 every arm is within ± 2 pp attainment and ± 3 % throughput of `fcfs`. |
| A1.3 | The scenario stresses the server: `fcfs` on S2 < 85 %. At ≥ 85 % the surge peak is raised to 2.5 λ\* once and S2 is re-run (+ 0.5 GPU-h); a second failure ends L1 as "no headroom on this stack". Between 65 % and 85 % H1 is recorded as not supported and the ladder is still evaluated. |

An increment **counts** if it is ≥ max(5 pp, 3 σ_run) on S2 (mean of the two seeds).

| decision | rule |
|---|---|
| **GO to L2 (P1 + P2)** | G_shed and G_key both count, and the L-class attainment of `edf+shed` is not more than 3 pp below `prio2`'s |
| **Partial GO** | exactly one of G_shed, G_key counts: only its primitive goes to L2 |
| P3 enters L2 | only if G_len > 5 pp on S2 or S3 (H4 predicts it will not) |
| **STOP** | neither counts on S2, and best arm − max(`fcfs+gdrop`, `prio2`) < 5 pp on S3: the existing knobs already capture the gain. Written up as a negative result; L3 shrinks to the benchmark PR |
| anything else | no automatic GO: the result is reported and the next step is Bowen's call |

### L2 — engine prototype of the primitives that passed (Mac + ≤ 3 GPU-h; + 1 for TP = 2)
P1: per-request waiting deadline enforced in `_poll_timeout_aborts` (rank 0, broadcast). P2: an
opt-in `--schedule-policy` value whose key is computed once from the request's SLO fields, so every
rank sorts identical keys. P4 (pacing) and P5 (keep preempted KV in the radix tree) are design notes
only, written if H9 or H6 says they matter.

| # | Accept |
|---|---|
| A2.1 | Equivalence: the native primitive reproduces its emulated arm's S2 attainment within ± 3 pp over two seeds. |
| A2.2 | Default path untouched: with the new flags off, S1 `fcfs` is within A/A noise (± 2 %) of unpatched; existing `test/registered/unit/managers/` tests pass. |
| A2.3 | Cost: completed-token throughput on S1 with the flags on ≥ 98 % of `fcfs`. |
| A2.4 | New unit tests cover ordering, the deadline abort and the off-by-default path. |
| A2.5 | TP = 2 smoke (two GPUs, 15 min, needs its own OK): S2 completes with the primitives on, no hang, no rank divergence. |
| A2.6 | Upstream's in-repo code rules respected (`msgspec.Struct` for new containers, no defensive `getattr`, no in-place `ScheduleBatch` mutation, the large-class style for `Scheduler` edits); ≤ 300 changed lines per PR excluding tests. |

Stop: A2.1 fails by more than 5 pp (the emulation was not measuring what the primitive does — report
the gap), or A2.2 cannot be met.

### L3 — upstream (0 GPU-h; every post is Bowen's)
| step | Accept |
|---|---|
| PR-A: goodput and SLO attainment in `sglang.benchmark.serving` | output equals `slo_metrics.py` on three recorded runs; unit test; ≤ 200 lines; references #5495 |
| RFC issue | carries the L1 table, simulator-vs-measured, the preemption cost numbers, and proposes **only** primitives that passed their rule; reuses the router's header names; reviewed and filed by Bowen |
| PR-B (P1), PR-C (P2) | opened only after a code owner of `srt/managers` or the router author engages with the RFC; A2.1–A2.6 attached |

Stop: no maintainer response to the RFC within 21 days → stop at PR-A + RFC and keep the patch in the
fork. Never more than two open SGLang PRs from this track; PR #33726 keeps priority.

## 5. Arms and hypotheses (pre-registered)

All arms: `--max-running-requests B*`, metrics endpoint off, everything else default. Priority arms
add `--enable-priority-scheduling --schedule-low-priority-values-first
--priority-scheduling-preemption-threshold 10`; keys are 100 ms ticks since run start, lower first.
ŝ_i = `T_prefill(input_i) + L·τ(B*)`.

| arm | server | priority key | client `/abort_request` if no first token by | information used |
|---|---|---|---|---|
| `fcfs` | default | — | — | none |
| `fcfs+gdrop` | `SGLANG_REQ_WAITING_TIMEOUT=W` (tuned in L0b) | — | — | none |
| `prio2` | priority | L → 0, D → 1000 | — | class |
| `edf` | priority | arrival + (2 s for L, D_i for D) | — | class, deadline |
| `prio2+shed` | priority | as `prio2` | L: arrival + 2 s; D: arrival + D_i − ŝ(L̄) | class, deadline for shedding |
| `edf+shed` | priority | as `edf` | same | class, deadline |
| `llf-oracle+shed` | priority | L: as `edf`; D: arrival + D_i − ŝ(L_i) | L: arrival + 2 s; D: arrival + D_i − ŝ(L_i) | + true length |
| side: `hrrn` | `--schedule-policy hrrn` | — | — | prompt length |
| side: `prio2-nopreempt` | priority + `--disable-priority-preemption` | as `prio2` | — | class |

L̄ is the class mean output length; using it instead of `L_i` is the paper's "without Request
Analyzer" ablation. The threshold of 10 ticks (1 s) is arbitrary; one extra `edf+shed` S2 cell each at
0 and 50 records the sensitivity. Side arms and sensitivity cells use the request list of S2's first
seed.

Priors (a fluid approximation of S2 with a single 20 s deadline; **not** the test — A0a.4 replaces them
with the simulator's numbers before the run): `fcfs` ≈ 45–50 %, `fcfs+gdrop` ≈ 75 %, `prio2` ≈ 60–65 %
(L protected, D starved through the surge and then served too late), an ideal deadline-aware policy
≈ 90 %.

| H | Statement | Supported if | Why it matters |
|---|---|---|---|
| H1 | SGLang's default loses most SLOs in a transient overload it could absorb on average | S1 ≥ 95 % and `fcfs` S2 ≤ 65 % | without headroom there is nothing to propose |
| H2 | Shedding is the first-order lever | `fcfs+gdrop` − `fcfs` ≥ 15 pp on S2 | separates "drop late work" from "order work" |
| H3 (primary) | Per-request shedding plus deadline ordering beats every existing knob | G_shed + G_key ≥ 10 pp on S2 | the two increments behind L1's GO decision |
| H4 | Per-request length knowledge adds little | G_len ≤ 5 pp on S2 and S3 | Fig. 17: oracle + 2 %, no analyzer − 8 %; decides whether a predictor is ever worth building |
| H5 | EDF without shedding fails under sustained overload | `edf` ≤ `fcfs+gdrop` − 10 pp on S3 | Fig. 21; tells the RFC not to propose a bare EDF policy |
| H6 | Preemption pays for itself | `prio2` − `prio2-nopreempt` ≥ 5 pp L-class attainment on S2 at ≥ 95 % of its completed-token throughput; preemptions and re-prefilled tokens reported | upstream's open "analyze preemption cost" item; decides P5 |
| H7 | Nothing regresses at low load | S1: every arm within ± 2 pp and ± 3 % throughput of `fcfs` | the condition every upstream reviewer asks for |
| H8 | A slot-queue simulator calibrated with τ(B\*) and the prefill rate predicts the ladder | \|predicted − measured\| ≤ 7 pp on S2 for ≥ 6 of 7 main arms | a validated simulator prices later what-ifs without GPU time; a miss locates what SGLang does that a slot queue does not |
| H9 | Decode pacing has little room on this stack | X_max / X(B\*) − 1 < 15 %, X from the L0b curve | pacing cannot exceed saturated throughput; below 15 % P4 is dropped without a prototype |

*Falsification that would change the track:* H1 false after the one permitted re-scope ends S1 as "no
headroom"; H3 false with H2 true reduces the proposal to P1; H4 false (G_len > 10 pp) puts a length
bound back on the table; H9 false (≥ 15 %) makes pacing the subject of a separate plan.

## 6. Design details

**Blocking.** One server per (server configuration × block); five configurations (`fcfs`, `fcfs+gdrop`,
priority, `hrrn`, priority-no-preempt). The five priority-key arms share one server and differ only
in what the client sends. Between cells: `/flush_cache`, then wait until the server reports zero
running and queued requests. Arm order inside a block is shuffled with a recorded seed; S2's two seeds
are two blocks with different request lists.

**VERIFIED cell (no number is quotable without it).** Resolved server config equals the pin; sent =
finished + aborted + cancelled; arrival schedule within tolerance; `fcfs` arms show admission order =
arrival order (< 1 % inversions) and zero preemptions; priority arms show admission order following
the key and, on S2/S3 preemptive arms, at least one preemption; every request the client aborted has
zero tokens; the log scan (from the post-launch offset) finds no OOM, traceback or retraction. A cell
with retractions is flagged MEM-BOUND and not used.

**Backups and operations.** As Q3: tmux on the node, driven from the Mac, no credentials or agents on
the node, `127.0.0.1` only; `scripts/sync_from_node.sh` every 2 minutes; per-cell summaries tracked,
raw per-request files under `results/raw/` (git-ignored); STOP file at cell granularity; release only
after the final sync shows nothing pending.

## 7. Budget and decision tree

An assignment is at most 1 h plus two 1 h extensions, and the home directory is wiped between
assignments (`context/decisions.md`, 2026-09-29). Node work is therefore cut into sessions of ≤ 3 h.
Each session starts with `setup_node.sh` and the model download (≈ 12 min); sessions B and C add a
5-minute drift check — τ(B\*) within ± 5 % of session A's value, else λ\* is re-bracketed (+ 15 min).

| session | content | GPU time | gate |
|---|---|---|---|
| — | L0a on the Mac | 0 | A0a.1–5 |
| A | L0b: setup 12 min, τ curve 12, knee 17, W 17, noise 8, engagement 10, slack 15 | ≈ 1.5 h, cap 2.5 h | A0b.1–7, else stop |
| B | L1 decision block: S1, S2 both seeds, side and sensitivity cells = 25 cells × 4 min, 10 server starts, slack | ≈ 2.4 h, cap 3 h | A1.1–3, then GO / partial / STOP |
| C | L1 remainder: S3 (7 cells); also holds a third S2 seed (A0b.6) or the A1.3 re-scope if needed | ≈ 0.8 h, cap 2 h | H5 and the S3 half of the STOP rule |
| later | L2 validation, + 1 h on two GPUs for A2.5 | ≤ 3 h + 1 h | A2.1–6 |
| — | L3 | 0 | maintainer engagement within 21 days |
| **A + B + C** | | **≈ 4.7 h, hard cap 7.5 h** | |

Extensions have cost 1 credit per GPU-hour; what an assignment itself costs is still unknown. Every
assignment and every extension needs Bowen's explicit OK (68 credits left on 2026-09-29).

## 8. Risks

| risk | mitigation |
|---|---|
| The paper's SLOs are loose for an H200 and nothing binds | B\* is set by the deadline class, load is defined from the measured knee, A1.3 allows one pre-registered re-scope |
| Sustained-overload numbers depend on the window length | windows are fixed and equal across arms; S2 (bounded backlog) is the primary scenario; S3 is reported with its horizon |
| The Python client becomes the bottleneck at ≈ 7 k streamed tokens/s | A0b.3; shard the client by request id |
| Static keys under-represent JITServe's dynamic priorities | stated in §3.5; a STOP verdict reads "existing knobs suffice for this workload", not "JITServe has no value" |
| A client-side abort is not what a server-side deadline would do | `/abort_request` acts at once, unlike a disconnect (4 s poll); A2.1 tests the equivalence |
| `ignore_eos` text is not natural output | lengths, not content, drive scheduling; noted as a limit |
| Priority values as Prometheus labels | metrics endpoint off in every arm; evidence from logs and the client |
| Assignment expires mid-run | sync every 2 min, resume at cell granularity |

## 9. Decisions taken by default — change before L0a if you disagree

1. Model `Qwen/Qwen3-8B` rather than the paper's Qwen2.5-14B (also public) or the lab's Qwen3-VL-8B.
2. Two classes at 50/50, SLO values from the paper, deadlines {10, 20, 40} s.
3. S2 (surge) as the primary scenario rather than the paper's sustained RPS sweep.
4. L0b + L1 budget ≈ 4.7 GPU-h in three node sessions, with a 7.5 h cap.
5. Track name S1, directory `experiments/slo_sched/`, `plan.md` §13.

## 10. Checklist for the reviewer

- [ ] Every number in §4–§5 is written before any run; priors are labelled as priors.
- [ ] Each increment (G_shed, G_key, G_len) isolates one variable and maps to one engine primitive.
- [ ] Baselines are tuned (B\*, W) before the comparison, by rules fixed here.
- [ ] Engagement is verifiable per cell on both sides (ordering and preemption present where expected, absent on `fcfs`).
- [ ] There is a decision point after ≈ 1.5 GPU-h (L0b) and a STOP rule in L1 that can end the track;
      every node session fits the 3 h assignment limit.
- [ ] Nothing in L0–L1 patches the engine; nothing is posted upstream before L3 and Bowen's review.
- [ ] Node rules hold; nothing on the node is the only copy for more than 2 minutes.
