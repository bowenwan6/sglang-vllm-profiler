# PR-B — draft description (not opened; Bowen reviews it together with PR-A, then opens it)

- Branch: `bowenwan6/sglang` `pr/req-waiting-timeout` @ `6baf9e5d7e` — two commits on `upstream/main`
  @ `b524de2de6` (2026-10-06), to be squashed on merge. Patch: [`0001-req-waiting-timeout.patch`](0001-req-waiting-timeout.patch).
- Checked on that head, in a real SGLang environment: the four touched unit files and the full
  pre-commit suite (16 hooks, 13 files). The server runs below used the same change on `734cf3cf3b`.
- Evidence, per-class tables and limits: [`../../results/prb_results.md`](../../results/prb_results.md).
- Open questions for the maintainers are listed at the end; they are not part of the text for GitHub.

Everything between the two lines is the text for GitHub.

---

**Title:** `[Feature] Per-request waiting timeout (waiting_timeout)`

## Motivation

`SGLANG_REQ_WAITING_TIMEOUT` drops a request that has waited in the scheduler queue longer than a
bound. It is one value for the whole server, but requests differ in how long waiting is still useful:
a chat turn with no first token after two seconds has failed, a batch job can wait half a minute.

With `--enable-priority-scheduling` the difference is built in, because low-priority requests are made
to wait on purpose. A bound short enough for the interactive requests then drops the low-priority ones
while they wait, and a bound long enough for those lets interactive requests be served after they have
stopped being useful.

This PR lets a request carry its own bound. A client can already hang up when it stops waiting; the
field differs in that it bounds the waiting only (a client cannot tell a queued request from a running
one), the answer is an explicit 503 a caller or a router can act on, and it takes effect at the next
scheduler step.

## Modifications

- New optional request field `waiting_timeout` (seconds, positive and finite) on `/generate`,
  `/v1/completions` and `/v1/chat/completions`.
- Enforced by the existing scan, `Scheduler._poll_timeout_aborts`, on the leader rank, with the
  existing abort: HTTP 503 `Request waiting timeout reached.` for a non-streaming request, the same
  error inside the stream for a streaming one. A request is dropped once it has outstayed either its
  own bound or the global one, so it can shorten the operator's bound but not extend it.
- A server that never receives the field keeps today's scan: the queue is walked only if the global
  bound is set or a request has carried one (a sticky flag on the scheduler).
- Validation in `GenerateReqInput._validate_inputs`, which every entry path goes through (including
  `/vertex_generate`, which has no pydantic step), and `Field(gt=0, allow_inf_nan=False)` on the two
  OpenAI models.
- `TokenizedGenerateReqInput` gets the field as a defaulted tail field, so the wire order the Rust
  server mirrors is unchanged.
- One row in `sampling_params.mdx`, one clause in the `SGLANG_REQ_WAITING_TIMEOUT` row, 7 unit cases.

Not in this PR: `/v1/responses`, embeddings / score / rerank, requests that use `session_params`, the
Engine API, a header form, the gRPC proto and the Rust front end, PD-specific queues, the running
timeout, metrics.

## Accuracy Tests

No model output changes.

Functional check on a live server (`Qwen/Qwen3-8B`, one H200, and again on two with `--tp-size 2`).
The server runs with `--max-running-requests 1` and a long request holds the slot:

- a request with `waiting_timeout: 1` is refused after 1.00–1.02 s on all three endpoints — 503 when
  not streaming, the error event when streaming — and produces no token;
- the request holding the slot finishes untouched; a request with a 3600 s bound and one with none
  wait and are served;
- with `SGLANG_REQ_WAITING_TIMEOUT=2`, requests asking for 30 s, 0.5 s and nothing are refused at
  2.0 s, 0.5 s and 2.0 s;
- `0`, `-1` and `"abc"` are answered with 400;
- a server without this change ignores the field.

Under load on two GPUs (`--tp-size 2 --max-running-requests 32`, 85 req/s for 60 s) 2059 of 5127
requests were refused by their bound, every other request was served and the server stayed healthy;
the same load with the global knob instead refused 2034.

## Speed Tests and Profiling

`Qwen/Qwen3-8B`, one H200, `--max-running-requests 128`. Two request classes with fixed work: chat
(256-token prompt, 128 output tokens; objective: first token within 2 s, 100 ms per token) and batch
(1024-token prompt, 256 output tokens; objective: done within 40 s). Attainment is requests that met
their objective / requests sent.

**Not used, nothing changes.** Time of one scan over a queue of real `Req` objects:

| | 1000 queued | 10 000 queued |
|---|---|---|
| main, no global bound | 1.2 µs | 1.2 µs |
| this PR, no bound of either kind seen | 1.2 µs | 1.2 µs |
| main, global bound set | 58 µs | 1.05 ms |
| this PR, global bound set, no request bound | 60 µs | 1.00 ms |
| this PR, every queued request carries a bound | 98 µs | 1.62 ms |

End to end at 0.8 of capacity, a server with this change and no field in the requests matches one
without it: 100 % attainment in both, mean TTFT 27.3 against 27.4 ms (chat) and 43.1 against 43.1 ms (batch),
5237 against 5236 output tok/s.

**The same bound in both forms gives the same result.** Chat alone at 1.3 × capacity: 82.4 and 80.5 %
attainment with `waiting_timeout: 1.5` on every request, 82.2 and 80.3 % with
`SGLANG_REQ_WAITING_TIMEOUT=1.5` (two seeds; 903 / 1006 against 913 / 1019 requests refused).

**Where it helps: priority scheduling.** `--enable-priority-scheduling --disable-priority-preemption`;
chat has the higher priority and arrives at 0.3 of its capacity with bursts to 1.5 for 20 s of every
minute; batch arrives at 0.5 of its capacity. 180 s, about 10 000 requests per run, the same request
list on every row.

| waiting bound | runs | chat | batch | all |
|---|---|---|---|---|
| none | 2 | 31.5 % | 50.3 % | 34.7 % |
| global 1.5 s | 3 | 78.9 % | 67.5 % | 77.0 % |
| global 5 s | 2 | 37.4 % | 66.2 % | 42.3 % |
| global 20 s | 2 | 31.7 % | 75.3 % | 39.1 % |
| global 30 s | 2 | 32.3 % | 81.0 % | 40.6 % |
| **per request: chat 1.5 s, batch 30 s** | 3 | **79.0 %** | **100.0 %** | **82.6 %** |
| per request: chat 1.5 s only | 2 | 78.9 % | 100.0 % | 82.5 % |
| none; the chat client hangs up after 1.5 s | 2 | 69.9 % | 100.0 % | 75.0 % |

A global bound short enough for chat drops a third of the batch requests, which are only waiting
behind the bursts; every longer one lets stale chat requests be served. With the field, chat is where
the 1.5 s global bound puts it and no batch request is lost. The batch class needs no bound of its
own, so nothing has to be configured on the server. With heavier bursts (2.0 of the chat capacity) the
picture is the same: 69.6 against 65.1 % in total, batch 100.0 against 67.3 %.

**Where it does not help: first come, first served.** With the default order (chat at 0.4 of its
capacity, batch bursts at 2.0 of its capacity for 15 s of every minute) the requests that may wait
keep their place in the queue and take the slots, so the shedding moves to the chat class:

| waiting bound | runs | chat | batch | all |
|---|---|---|---|---|
| none | 1 | 38.2 % | 100.0 % | 54.3 % |
| global 1.5 s | 2 | 87.7 % | 52.7 % | 78.5 % |
| global 30 s | 1 | 38.3 % | 100.0 % | 54.3 % |
| per request: chat 1.5 s, batch 30 s | 2 | 60.8 % | 100.0 % | 71.1 % |

Total attainment is lower than with a global bound tuned to the chat objective. The field is for
deployments that order their queue, or for callers that want an explicit refusal; it does not raise
attainment under FCFS.

## Checklist

- [x] Format your code according to the Format code with pre-commit.
- [x] Add unit tests according to the Run and add unit tests.
- [x] Update documentation according to Write documentations.
- [x] Provide accuracy and speed benchmark results (above).
- [x] Follow the SGLang code style guidance.

---

## Questions a reviewer is likely to raise (for Bowen, not for the PR text)

| Question | Position taken in the patch | Alternative |
|---|---|---|
| Name and unit | `waiting_timeout`, seconds, mirroring the environment variable | `waiting_timeout_s`; note that `kv_mgr.waiting_timeout` already names an unrelated PD transfer knob |
| Smaller of the two bounds, or override | a request cannot extend the operator's bound | an override form, as Triton's `allow_timeout_override` |
| Status code | 503 with the existing message; only 400 / 503 / 500 become HTTP errors in the tokenizer manager | — |
| The sticky flag | keeps the default path at today's cost | always walk the queue: 5 lines and one test fewer, about 0.08 µs per queued request per step on every server |
| The clock | time in the current stay in the waiting queue, as for the global bound; a preempted or retracted request starts a new stay | first admission only |
| PD mode | no PD-specific code; the bound covers `waiting_queue` as the global one does; not exercised | apply the request bound in unified mode only |
| Sessions | `session_params` requests do not carry the field (a turn dropped from the queue would have to clear the session's in-flight state) | follow-up |
| Open PRs on the same code | #34457 (PD waiting timeout; rewrites part of the scan and of `test_scheduler_timeouts.py`; stale since 08-18) and #37260 (retracted requests and the queued-request limit) | rebase when either moves |
| A header form, a metric | not included | `x-override-waiting-timeout` is one table entry; a counter for timeout aborts does not exist for the global bound either |
