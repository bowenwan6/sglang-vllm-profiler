# PR-B research — a per-request waiting timeout for SGLang

> Read-only research, 2026-10-05. Source: `sgl-project/sglang` `upstream/main` @ `734cf3cf3b`
> (2026-10-04), the GitHub tracker of sglang and vllm, Triton's documentation. No GPU, no code yet.
> Verdict in §6. The benchmark design of §5 is pre-registered in [`PLAN.md`](PLAN.md) stage 3.

## 1. What exists in SGLang today

| Fact | Where |
|---|---|
| Two **global** timeouts, set by environment variable, default off (−1): `SGLANG_REQ_WAITING_TIMEOUT` (seconds in the waiting queue) and `SGLANG_REQ_RUNNING_TIMEOUT` (seconds in the running batch). A hit aborts the request with 503 and "Request waiting timeout reached." | `srt/environ.py:639`; `Scheduler._poll_timeout_aborts` |
| They began as `SGLANG_QUEUED_TIMEOUT_MS` / `SGLANG_FORWARD_TIMEOUT_MS` and were renamed on 2026-02-13 by a maintainer (hnyls2002) | [#18766](https://github.com/sgl-project/sglang/pull/18766) |
| They are used in production: "GLM NVFP4, B200 TP4, 60k-300k token prompts, `SGLANG_REQ_WAITING_TIMEOUT=45`". That report led to the rank-consistent rewrite: the scan runs on rank 0 only and its aborts are broadcast, because per-rank clocks split the queue and hung the collectives | [#37143](https://github.com/sgl-project/sglang/pull/37143), merged 2026-09-08, reviewed by hnyls2002 |
| In PD-disaggregation mode neither timeout is enforced; a PR that adds it has been open since 08-11 | [#34457](https://github.com/sgl-project/sglang/pull/34457) |
| Neither variable is mentioned anywhere under `docs/` | `git grep` on the pin |
| No generation request carries a timeout or deadline. Only `FlushCacheReqInput.timeout_s` and `OpenSessionReqInput.timeout` exist | `srt/managers/io_struct.py` |
| A queued request whose client disconnects is found by a poll every `SGLANG_REQUEST_STATE_WAIT_TIMEOUT` (default 4 s); `POST /abort_request` with the request's `rid` removes it at once | `tokenizer_manager.py:190,1876`; `environ.py:1494`; `http_server.py` |
| The engine already reads per-request headers for routing (`x-smg-routing-key`, `x-data-parallel-rank`) and a closed PR proposed `x-sglang-request-priority` | `entrypoints/openai/serving_base.py:259,271`; #19808 |
| The router has a global proxy timeout (`--request-timeout-secs`, default 300) and SLO headers used for bucket selection; nothing is forwarded to the engine as a deadline | `experimental/sgl-router/src/config/cli.rs:143`; #40292 |

## 2. Has anyone done it?

- **SGLang:** no. Searches of issues and PRs for "request timeout", "per-request timeout", "waiting
  timeout", "deadline" and "ttl" return the items above and nothing that proposes a per-request
  waiting timeout or deadline.
- **vLLM:** nothing found under the same queries. (Its benchmark has had `--goodput` since 2024-10,
  [vllm#9338](https://github.com/vllm-project/vllm/pull/9338) — relevant to PR-A.)
- **Triton Inference Server** is the direct precedent: the dynamic batcher's queue policy has
  `default_timeout_microseconds`, `timeout_action` (reject or delay) and `allow_timeout_override`, and
  a request may carry its own timeout
  ([batcher docs](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/batcher.html)).
- **gRPC** carries a per-call deadline to the server as a matter of course.
- **JITServe** exposes `waiting_time` per request and drops what is not scheduled within it
  (BACKGROUND.md §1).

So the idea is established elsewhere and absent here; it is not a duplicate of open upstream work.

## 3. Feasibility

A per-request field follows the path `priority` already takes. Touch points on the pin:

| Step | Where `priority` passes today |
|---|---|
| OpenAI request models | `protocol.py`: `CompletionRequest`, `ChatCompletionRequest`, `ResponsesRequest` |
| Request → `GenerateReqInput` | `serving_chat.py:1311`, `serving_completions.py:138`, `serving_responses.py:579` |
| `GenerateReqInput` and its batch split, `TokenizedGenerateReqInput` | `io_struct.py` |
| Tokenizer manager → scheduler | `tokenizer_manager.py:1535,1567` |
| `Req` construction | `Scheduler.handle_generate_request`, `schedule_batch.py` |
| Enforcement | `_poll_timeout_aborts`: one comparison per queued request, already rank 0 only |

Estimate: about ten one-line plumbing edits, ten lines in the scan, a unit test next to
`test/registered/unit/managers/test_scheduler_timeouts.py`, a docs paragraph. Under 150 changed lines
without tests. No new mechanism: the abort path, the status code and the rank broadcast are the ones
#37143 hardened.

Design choices to settle (working choice first; the rest go into the PR description as alternatives):

| Question | Working choice | Why |
|---|---|---|
| Name and unit | body field `waiting_timeout`, seconds, float > 0 | mirrors `SGLANG_REQ_WAITING_TIMEOUT` |
| Interaction with the global value | the smaller of the two when both are set | a client can tighten the operator's bound, never extend it |
| Clock | the one the global timeout uses (`wait_queue_entry_time`) | no new time source, same behaviour after a retraction |
| Response | identical to the global timeout (503, same message) | clients already handle it |
| Header form | not in the first PR | the router's `x-sgl-*-slo` naming is still moving; propose in the description |
| PD mode, gRPC proto, running timeout | out of scope | PD waits on #34457; the others are follow-ups |

Risks: the Rust front end and gRPC path build requests outside `protocol.py` (field silently ignored
there until added); embedding and classify requests share the scan (field left out of them in v1).

## 4. What it is for

1. **Mixed traffic on one engine.** Interactive requests are worthless after a couple of seconds in
   the queue; batch and agent-tool requests can wait tens of seconds. One global value is wrong for
   one of them: tight drops batch work that would have met its deadline, loose serves interactive
   requests nobody is waiting for. The production report in #37143 (45 s for 60k–300k-token prompts)
   is a value no interactive request could share.
2. **Requests whose caller has a budget.** A tool call with a 5 s budget that is still queued at 5 s
   will be computed and then discarded by the caller. A gateway that keeps the connection open never
   triggers the disconnect path; when the client does disconnect, the queue keeps the request for up
   to 4 s.
3. **A primitive the router can use.** The router already parses per-request SLO headers; an engine
   field gives it something to forward.

## 5. How to show it is useful

Arms on the same server build and the same request list: no timeout · the best single global value
(tuned over {2, 5, 10, 20} s) · per-request `waiting_timeout`. Metrics from PR-A's goodput plus tokens
spent on requests that missed their SLO.

| id | Use case | Expectation |
|---|---|---|
| U1 | Surge with two classes: interactive (TTFT ≤ 2 s) and batch (finish within 20–40 s) | per-request beats the best global value: the global arm either drops batch work or serves late interactive work |
| U2 | Callers with budgets: every request has a budget drawn from {3, 10, 30} s and is useless after it | per-request cuts the tokens computed for already-expired requests and raises goodput |
| U3 | Control: field absent; field present but never binding | no difference from the unpatched server |
| U4 | Neutral case: one class, one SLO, sustained overload | per-request equals the tuned global value; no gain is expected and that is reported |

Thresholds and the stop rule are in PLAN.md stage 3.

## 6. Verdict

**Go to implementation.** It is feasible in a small diff on top of a path a maintainer hardened a
month ago, nobody has proposed it, a precedent exists in Triton, and there are two use cases in which
a global timeout cannot do the same job. Whether it earns a PR is decided by U1 and U2, not here.
