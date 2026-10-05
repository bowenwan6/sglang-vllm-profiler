# PR-B plan — a per-request waiting timeout (`waiting_timeout`)

> **Status: written 2026-10-05. Source study only: nothing is implemented, no server and no test has
> run.** Source pin: `sgl-project/sglang` `upstream/main` @ `734cf3cf3b` (2026-10-04), read in the
> worktree `sglang-slo` (that commit plus `d132f6739e`, which touches only the benchmark, its test and
> its doc page). Files outside the sparse worktree (`rust/`, `proto/`, `docs/`, `.github/`,
> `test/run_suite.py`, `test/registered/scheduler/`) were read from the same commit with `git show`.
> This file details stages 3.1–3.2 of [`PLAN.md`](PLAN.md) and builds on
> [`PRB_RESEARCH.md`](PRB_RESEARCH.md); §9 lists where the source contradicts them, §10 what could not
> be checked. Upstream's in-repo contributor rules are applied by name: `unit-test-admission`,
> `general-code-style`, `no-getattr-defensive`, `no-dataclasses`, `comment-style`,
> `modify-component-must-read` and the `large-class-style` guide.
>
> **Conventions.** A path that does not start with `python/`, `test/`, `docs/`, `rust/`, `proto/` or
> `.github/` is relative to `python/sglang/srt/`; after its first mention a file is cited by base name
> (`scheduler.py` alone is `managers/scheduler.py`, `protocol.py` and the `serving_*.py` files are in
> `entrypoints/openai/`, test files are under `test/registered/unit/`). Line numbers hold at the pin
> only; re-grep after every rebase. *Leader rank* = the rank with `pp_rank == 0`, `attn_tp_rank == 0`,
> `attn_cp_rank == 0` (one per attention-DP group).

## 0. The change in one table

| | First PR | See |
|---|---|---|
| Field | `waiting_timeout`, seconds, `Optional[float]`, `None` = absent | §1.1, §8 Q1 |
| Accepted on | native `/generate`, `/v1/completions`, `/v1/chat/completions`; inherited by `/invocations`, `/vertex_generate` and the gRPC OpenAI pass-through | §1.1, §1.2 |
| Rule | effective bound = the smaller of the request's value and `SGLANG_REQ_WAITING_TIMEOUT`; a global value ≤ 0 means "off", not "zero seconds" | §2.6 |
| Clock | `time.perf_counter()` against `req.time_stats.wait_queue_entry_time`: the **current stay** in `waiting_queue`, not time since arrival | §2.4, §2.5 |
| Enforced in | `Scheduler._poll_timeout_aborts`, on the leader rank only; no other code compares the field with a clock | §2.2 |
| Response | the existing abort path, unchanged: HTTP 503 for non-streaming requests, **HTTP 200 with an error event for streaming ones** | §3 |
| Validation | positive and finite; declared on the two pydantic models and checked once in `GenerateReqInput._validate_inputs` for every entry path | §4 |
| Default path | a server that never receives the field keeps today's O(1) scan (sticky gate) | §2.7 |
| Not included | `/v1/responses`, embeddings / classify / score / rerank, sessions, the Engine API, a header form, the gRPC proto, the Rust front end, PD-specific queues, the running timeout, new metrics | §1.3, §1.4 |
| Size | ≈ 39 added / 5 removed source lines in 7 files, ≈ 6 doc lines, ≈ 125 lines of unit tests | §7.2 |

## 1. Request path

### 1.1 Hops that need an edit

A generation request travels: HTTP body → pydantic model or `GenerateReqInput` → `TokenizerManager`
→ `TokenizedGenerateReqInput` (IPC) → `Scheduler.handle_generate_request` → `Req` → `waiting_queue`.

| # | Hop | Where | The `priority` line today | Line to add |
|---|---|---|---|---|
| 1 | Completions model (pydantic) | `entrypoints/openai/protocol.py:413-414`, class at 351 | `# Priority for the request` / `priority: Optional[int] = None` | after 414: a one-line comment and `waiting_timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)` |
| 2 | Chat model (pydantic) | `protocol.py:973-974`, class at 858 | same two lines | after 974: the same two lines |
| 3 | Completions → internal | `entrypoints/openai/serving_completions.py:138`, inside `GenerateReqInput(` at 116 | `priority=request.priority,` | `waiting_timeout=request.waiting_timeout,` |
| 4 | Chat → internal | `entrypoints/openai/serving_chat.py:1311`, inside `GenerateReqInput(` at 1283 | `priority=request.priority,` | `waiting_timeout=request.waiting_timeout,` |
| 5 | Native input (stdlib dataclass) | `managers/io_struct.py:319-320`, class at 177 | `priority: Optional[int] = None` | **append** after `kv_hints` (374-376), before `regenerate_rid` (378): two comment lines and `waiting_timeout: Optional[float] = None` |
| 6 | Validation | `io_struct.py:440-467` (`_validate_inputs`, called at 427) | — (`priority` has none) | the range check of §4 |
| 7 | Batch and `n > 1` split | `io_struct.py:1015` in `__getitem__` (938; `sub = GenerateReqInput(` at 944) | `priority=self.priority,` | `waiting_timeout=self.waiting_timeout,` |
| 8 | Tokenized request (msgspec) | `io_struct.py:1109-1110`, class at 1041 | `priority: Optional[int] = None` | **append after line 1153**, not beside `priority`: two comment lines and `waiting_timeout: Optional[float] = None` |
| 9 | Tokenizer manager | `managers/tokenizer_manager.py:1535`, inside `TokenizedGenerateReqInput(` at 1504 (`_create_tokenized_object`, 1461) | `priority=obj.priority,` | `waiting_timeout=obj.waiting_timeout,` |
| 10 | Scheduler → `Req` | `managers/scheduler.py:2812`, inside `Req(` at 2780 (`handle_generate_request`, 2755) | `priority=recv_req.priority,` | `waiting_timeout=recv_req.waiting_timeout,` |
| 11 | `Req` signature | `managers/schedule_batch.py:1043` (`__init__` at 1008) | `priority: Optional[int] = None,` | **append** after `cache_salt` (1056): `waiting_timeout: Optional[float] = None,` |
| 12 | `Req` attribute | `schedule_batch.py:1156` | `self.priority = priority` | a one-line comment and `self.waiting_timeout = waiting_timeout` |
| 13 | Gate state (new) | `scheduler.py:1301`, end of `init_running_status` (1268) | — | a one-line comment and `self._has_req_waiting_timeout = False` |
| 14 | Gate arming (new) | `scheduler.py:2827`, after `req.tokenizer = self.tokenizer` | — | `if req.waiting_timeout is not None:` / `self._has_req_waiting_timeout = True` |
| 15 | Enforcement | `scheduler.py:3390-3394` | — | §2.6 |

Exact text of the additions that carry a comment (every line fits the 88-column formatter):

```python
# protocol.py, after 414 and again after 974
    # Seconds the request may wait in the scheduler queue before a 503 abort.
    waiting_timeout: Optional[float] = Field(default=None, gt=0, allow_inf_nan=False)

# io_struct.py, GenerateReqInput, after 376
    # Seconds the request may stay in the waiting queue before a 503 abort;
    # SGLANG_REQ_WAITING_TIMEOUT still caps it. None means no bound of its own.
    waiting_timeout: Optional[float] = None

# io_struct.py, TokenizedGenerateReqInput, after 1153
    # See GenerateReqInput.waiting_timeout. Appended as a defaulted tail field,
    # so the Rust server's shorter arrays decode it as None.
    waiting_timeout: Optional[float] = None

# schedule_batch.py, after 1156
        # Seconds. Wall-clock comparisons against it must stay on the leader rank.
        self.waiting_timeout = waiting_timeout

# scheduler.py, after 1301
        # Sticky; servers that never see waiting_timeout skip the per-step queue walk.
        self._has_req_waiting_timeout = False
```

Why three rows deviate from "next to `priority`":

- **Row 8 is a wire-order constraint.** `TokenizedGenerateReqInput` inherits `array_like=True`
  (`io_struct.py:86`), so field order is wire order. The Rust server mirrors it positionally and emits a
  prefix that stops at `disagg_prefill_dp_rank` (`rust/sglang-server/src/message/io_struct.rs:14-56`,
  "inserting a field anywhere but the end shifts every later field on the wire"); shorter arrays decode
  with defaulted tails. The Python side says the same at `io_struct.py:1143-1153` ("Keep tail fields
  append-only"), and `test/registered/unit/managers/test_io_struct.py:51-76` compares the two
  declarations. `priority` (1110) already sits behind the Rust prefix, so an insertion beside it would
  not break that test, but it would break the append-only rule the last three fields follow.
- **Rows 5 and 11 keep positional signatures stable.** `GenerateReqInput` fields are positional except
  `rid` (180) and `http_worker_ipc` (309); `Req.__init__` parameters are positional-capable too. The
  two newest dataclass fields were appended (`cache_salt` 366, `kv_hints` 374).
- Row 7 needs no `_normalize_*` step: like `priority`, the field is one scalar for the whole batch.
  There is no analogue of `_set_default_priority` (`tokenizer_manager.py:868, 3855-3862`).

### 1.2 Entry points that inherit the field with no edit of their own

| Entry | Why |
|---|---|
| `/invocations` | takes `ChatCompletionRequest` and calls the chat handler (`entrypoints/http_server.py:2103-2110`) |
| `/vertex_generate` | splats `parameters` into `GenerateReqInput(...)` (`http_server.py:2132-2136`) with no pydantic step — one reason row 6 exists |
| gRPC OpenAI pass-through | `OpenAIRequest.json_body` (`proto/sglang/runtime/v1/sglang.proto:331`) becomes a `ChatCompletionRequest` / `CompletionRequest` (`entrypoints/grpc_bridge.py:713-732`) |
| Radix-native sessions (`session_id`) | they take the normal `Req(...)` route (`scheduler.py:2768-2772, 2780`) |
| List prompts, `n > 1` | every batch path indexes `obj[i]`, i.e. row 7 (`tokenizer_manager.py:1606, 1611, 1988, 2004, 2022`); a single prompt with `n > 1` is turned into a batch first (`io_struct.py:540-547`) |

### 1.3 Deliberately left out

| What | Where `priority` passes today | Why not in the first PR |
|---|---|---|
| Embeddings, classify | `protocol.py:1330, 1358`; `serving_embedding.py:176`; `serving_classify.py:74`; `io_struct.py:1233, 1374, 1407, 1445`; `tokenizer_manager.py:1567`; `scheduler.py:3452` | prefill-only work; a second dataclass with two `__getitem__` branches (10 more one-line edits at the places listed). They still share the queue and the scan: their `Req.waiting_timeout` is `None`, so the global bound applies as today |
| Score, rerank, transcription, the Ollama and `/v1/messages` front ends | internal builders: `managers/tokenizer_manager_score_mixin.py:1080`, `serving_rerank.py:433`, `serving_transcription.py:99, 338`, `entrypoints/openai/streaming_asr.py:178`, `entrypoints/ollama/serving.py:90, 208`; the `/v1/messages` front end converts to a `ChatCompletionRequest` built from its own model | their request models carry no `priority` either (e.g. `ScoringRequest`, `protocol.py:1384-1417`); nothing to mirror |
| `/v1/responses` | `protocol.py:1840`; `serving_responses.py:579, 2703-2774` | on this path `priority` never reaches the engine (§9 R3), so there is no working template; the built-in tool loop re-submits under the same rid (2739-2754) and background mode exists (568). If maintainers want it: three lines (model field, 561, 2746) |
| OpenAI Batch API | — | no `/v1/batches` route exists at the pin; `BatchRequest` (`protocol.py:288`) is referenced by no route |
| Sessions (`session_params`) | `session/session_controller.py:326` in `Session.create_req` (204-350) | `create_req` marks a streaming session in flight (344) and registers a node for the other kind (347-348). The flag is cleared only by `finish_req` (354) and `abort_req` (367); the queue-pop branch of `Scheduler.abort_request` (`scheduler.py:5313-5331`) calls neither, while the door-reject helper has to do it by hand (`scheduler.py:3244-3251`, "a session left in-flight rejects every later request"). A turn dropped from the queue would lock its session. The global timeout has this exposure already (read, not reproduced); the first PR does not widen it |
| Engine API | `entrypoints/engine.py:460/504, 575/619`; `entrypoints/EngineBase.py:37`; `entrypoints/http_server_engine.py:134/148` | 7 more lines in 3 files; explicit keyword arguments, so an unsupported `waiting_timeout=` raises instead of being dropped |
| Header form | `entrypoints/request_headers.py:18` (`x-override-priority`) | §8 Q4 |
| PD rebootstrap payload | `schedule_batch.py:2115` | an engine-internal recompute request; PD |
| Error stubs | `scheduler.py:2894-2901` (session not found), `disaggregation/encoder/receiver.py:2512-2550` (EPD abort stub, `priority` at 2537) | they only carry an error back |

### 1.4 Paths that build requests outside these classes

| Path | What happens to `waiting_timeout` |
|---|---|
| Embedded Rust server (`SGLANG_RUST_SERVER`, `environ.py:1865`; `scheduler.py:2272-2301`) | **silently ignored.** The Rust side ignores unported body fields, `priority` among them (`rust/sglang-server/src/message/request.rs:1079-1091`), and emits arrays that end at `disagg_prefill_dp_rank` (`rust/sglang-server/src/message/io_struct.rs:46-55`); the tail field decodes as `None` |
| gRPC native generate | **cannot be sent.** `TextGenerateRequest` / `GenerateRequest` carry `priority` (`sglang.proto:106, 134`) but no timeout; the Python side is `GenerateReqInput(**req_dict)` (`grpc_bridge.py:287`). Needs a proto field and Rust conversion. The Rust server's own gRPC front rejects `priority` as unsupported (`rust/sglang-server/src/grpc/convert.rs:72-81`) |
| PD prefill and decode servers | **not a separate path.** Both build the `Req` in `handle_generate_request` (`disagg_mode=self.disaggregation_mode`, `scheduler.py:2807`) and both event loops run the scan (§2.2). The field would be enforced once the request is in `waiting_queue`; only the PD queues of §2.3 are outside. See §8 Q7 |
| EPD encoder wait | requests waiting for embeddings sit in `mm_receiver.waiting_list` (`disaggregation/encoder/receiver.py:1898`) before the scheduler dispatches them (`managers/scheduler_components/request_receiver.py:227-245`) |
| Sessions | `Session.create_req` builds its own `Req` (§1.3) |
| Internal builders | warm-up (`entrypoints/warmup.py:81, 114, 150`), health check (`http_server.py:735-740`): always `None` |

Old servers and typos are a third silent case: FastAPI's dataclass parsing and pydantic's default both
drop unknown body fields (checked on stand-in models with pydantic 2.10.3; the Rust comment at
`request.rs:1086-1087` states the same). A client cannot tell from the response whether the field was
understood — only the behaviour shows it (§7.3 D2, control run).

## 2. Enforcement

### 2.1 What the scan does today

`scheduler.py:3383-3405`:

```python
    def _poll_timeout_aborts(self) -> List[AbortReq]:
        """Emit aborts only; every rank must drop the same requests in the
        same iteration, or the extend-vs-decode decision splits and the
        collectives hang.
        """
        aborts: List[AbortReq] = []

        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) > 0:
            deadline = time.perf_counter() - timeout_s
            for req in self.waiting_queue:
                entry_time = req.time_stats.wait_queue_entry_time
                if 0 < entry_time < deadline:
                    aborts.append(
                        AbortReq(
                            rid=req.rid,
                            abort_message="Request waiting timeout reached.",
                            finished_reason={
                                "type": "abort",
                                "status_code": HTTPStatus.SERVICE_UNAVAILABLE,
                                "message": "Request waiting timeout reached.",
                            },
                        )
                    )
```

- It only **emits**. Removal happens when the `AbortReq` is dispatched to `abort_request`
  (`scheduler.py:5303`) on every rank.
- The loop over `waiting_queue` exists **only when the global value is > 0** (3390). With both
  defaults the whole scan is two environment reads per scheduler step (3390, 3407).
- The running-timeout half (3407-3434) is untouched by this plan.

### 2.2 Where it is called, the rank gate, the broadcast

| Step | Evidence |
|---|---|
| One caller | `ingest_requests` (`scheduler.py:2052-2089`); the scan is at 2070 |
| Every event loop calls it | `scheduler.py:1902, 1946`; `disaggregation/prefill.py:696, 782`; `disaggregation/decode.py:2930, 2977`; `managers/scheduler_pp_mixin.py:136, 280, 431`; `multiplex/multiplexing_mixin.py:116`; `hardware_backend/mlx/scheduler_mixin.py:215` |
| Gate | `scheduler.py:2064-2069`: `not self._deferred_input_requests and pp_rank == 0 and attn_tp_rank == 0 and attn_cp_rank == 0`. Not "rank 0": with attention DP there is one leader per DP group, each scanning its own queue. The first clause skips the scan while inputs are held back at a pause (PD prefill overlap loop, `prefill.py:782-784`) |
| Broadcast | the aborts go into `recv_requests(local_reqs=...)` (`request_receiver.py:80-110`). Without DP they are appended to the pulled requests and broadcast over the TP group (201-211); with DP they ride the work channel "scoped to the ranks sharing one waiting queue" (166-179); later PP stages receive the list from the previous stage (146-158; `scheduler_pp_mixin.py:136`) |
| Dispatch | `process_input_requests` (`scheduler.py:2092`) → dispatcher (2114; `(AbortReq, self.abort_request)` at 1718) → `abort_request` (5303) |
| Order in one pass | pulled requests first, then the timeout aborts (`[*recv_reqs, *local_reqs]`, `request_receiver.py:203`; `work_reqs.extend(local_reqs)`, 174) |
| Skipped polls | with `--scheduler-recv-interval` > 1 (default 1, `arg_groups/fields/schedule.py:198-201`) a skipped poll returns `[]` before the broadcast (`request_receiver.py:92-94`); the aborts are recomputed on the next step — the scan is stateless |
| Pause | the scan runs before the paused check (`scheduler.py:1902-1905, 1946-1949`), so queued requests can time out during `/pause_generation` |

**Consequence for PR-B.** The new field reaches every rank inside the broadcast
`TokenizedGenerateReqInput`, but only the leader compares it with a clock, and the decision travels as
an `AbortReq` exactly like today's. No new synchronisation is needed, on one condition: **nothing
outside `_poll_timeout_aborts` may compare `req.waiting_timeout` with a clock** — not
`_add_request_to_queue`, not the queue policy, not the prefill adder; those run on every rank with
rank-local clocks. Resolution is one scheduler step: a request past its bound is aborted at the next
`ingest_requests`, before admission is attempted in that step.

### 2.3 Where a not-yet-running request can be, and whether the scan sees it

| Holder | Where | Seen by the scan | Own bound |
|---|---|---|---|
| `waiting_queue` | `scheduler.py:1279` | yes | the global timeout |
| Grammar queue | `constrained/grammar_manager.py:34`; filled at 198-201 (`scheduler.py:3116-3118`), drained at `scheduler.py:3845-3848` | no; the entry time is still 0 | `SGLANG_GRAMMAR_MAX_POLL_ITERATIONS` × poll interval (`grammar_manager.py:252-258`; `environ.py:407-408`) |
| Chunked prefill in progress | `self.chunked_req` (`scheduler.py:3934-3942, 4084-4090`); the request left the queue at 4079 | no — it is in flight | the running timeout |
| dLLM manager | `dllm/mixin/scheduler.py:222-232` moves requests out of `waiting_queue` into `DllmManager.waiting_queue` / `staging_queue` (430-431) | only while still in the scheduler's queue | — |
| PD prefill: bootstrap, in-flight | `scheduler.py:3284-3289, 5371-5388` | no | `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` on the KV transfer (`environ.py:711`) |
| PD decode: prealloc, transfer, retracted | `scheduler.py:3290-3297, 5390-5427` | no | same |
| EPD encoder wait | `disaggregation/encoder/receiver.py:1898` | no | — |
| Tokenizer manager during a pause | `tokenizer_manager.py:902-903` | no (not yet at the scheduler) | — |

Requests that wait in `waiting_queue` because of a LoRA constraint (`scheduler.py:3963-3964`) or a
storage prefetch (3997-4003) are in the queue and are seen.

### 2.4 What `wait_queue_entry_time` means

Defined at `observability/req_time_stats.py:627` ("get by time.perf_counter()", 626); `0.0` means "not
stamped". The setter (741-760) stamps on the first call and, on any later call, runs
`set_retract_time` and **overwrites**: the value is always the time of the latest entry into the queue.

| Event | Call site | Effect on the clock |
|---|---|---|
| First entry (unified mode) | `scheduler.py:3282` in `_add_request_to_queue` (3268) | starts |
| Grammar ready | `scheduler.py:3848` → 3282 | starts here: compile time is not counted |
| Priority preemption | `scheduler.py:4080-4082` → 3282 | **restarts** |
| Retraction (KV pool full) | `scheduler.py:4271-4272` → 3282 | **restarts** |
| Pause with retract | `scheduler.py:5514-5522` → 3282 | restarts, and the scan keeps running during the pause |
| PD prefill: bootstrap done, optimistic-prefill yield | `disaggregation/prefill.py:537, 550, 1735` | starts / restarts |
| PD decode: KV transferred; retracted | `disaggregation/decode.py:2584`; `scheduler.py:3297` | starts / restarts |
| PD prefill retry | `reset_prefill_retry_time` (729-739) | back to 0 |
| Between prefill chunks | — | nothing: the request is not in the queue |

So both the global bound and the new one mean "seconds in the current stay in `waiting_queue`". A
request that already streamed tokens, is preempted or retracted, and then waits longer than its bound
is aborted mid-stream. `queue_duration_s` (663-665) accumulates across stays but the scan does not use
it. Decision in §8 Q6.

### 2.5 Sentinels and the clock

- **Global value.** `EnvFloat(-1)` (`environ.py:639`); unset → −1; an unparsable value → warning and the
  default (`environ.py:81-86`). The scan requires `> 0`, so −1, 0 and negatives are "off"
  (`test/registered/unit/managers/test_scheduler_timeouts.py:132-135` pins 0).
- **Request value.** `None` = absent; otherwise positive and finite (§4).
- **Clock.** Stamp and scan both use `time.perf_counter()` (`req_time_stats.py:742`;
  `scheduler.py:3391`) in the same process, the leader rank. The request value is a duration: no
  conversion, no cross-process rebasing. Two things to avoid: storing it on `SchedulerReqTimeStats`
  (keys ending in `time` are rebased across processes, `req_time_stats.py:366-372`), and
  `time.monotonic()` (`lora/lora_drainer.py:88-91` subtracts this stamp from `time.monotonic()`, a
  different clock — not a pattern to copy).

### 2.6 The exact change

`scheduler.py:3390-3394` (five lines) become ten; the `aborts.append(AbortReq(...))` block below keeps
its indentation and content.

```diff
-        if (timeout_s := envs.SGLANG_REQ_WAITING_TIMEOUT.get()) > 0:
-            deadline = time.perf_counter() - timeout_s
-            for req in self.waiting_queue:
-                entry_time = req.time_stats.wait_queue_entry_time
-                if 0 < entry_time < deadline:
+        global_timeout_s = envs.SGLANG_REQ_WAITING_TIMEOUT.get()
+        if global_timeout_s > 0 or self._has_req_waiting_timeout:
+            now = time.perf_counter()
+            for req in self.waiting_queue:
+                timeout_s = req.waiting_timeout
+                # The global bound caps the request's own; <= 0 means it is off.
+                if timeout_s is None or 0 < global_timeout_s < timeout_s:
+                    timeout_s = global_timeout_s
+                entry_time = req.time_stats.wait_queue_entry_time
+                if timeout_s > 0 and 0 < entry_time < now - timeout_s:
                     aborts.append(
```

| Request value | Global value | `timeout_s` used | Fires when the stay exceeds |
|---|---|---|---|
| `None` | ≤ 0 | global (≤ 0) | never — today's behaviour |
| `None` | g > 0 | g | g — today's behaviour, same float arithmetic (`now - g`) |
| r | ≤ 0 | r | r |
| r | g > 0, g < r | g | g: a request cannot extend the operator's bound |
| r | g > 0, g ≥ r | r | r |

Invariant relied on: `Req.waiting_timeout` is `None` or positive and finite (§4). The trap this shape
avoids: `min(req.waiting_timeout, global)` with the default −1 gives −1 and would abort every request
that carries the field at once.

Rule compliance: `Req.__init__` always sets the attribute, so the scan reads it directly
(`no-getattr-defensive`); `Scheduler.__init__` is untouched and the one new piece of state is a
one-line assignment inside the existing `init_running_status` helper (`large-class-style` §2.2);
`scheduler.py` is not on the frozen list (§1.2 of that guide names `model_runner.py` only); no
`ScheduleBatch` field is touched; no environment variable is added; every new comment is one or two
ASCII lines stating a unit, a sentinel or the cross-rank constraint (`comment-style`).

### 2.7 Cost of the walk, and the gate

Without a gate, a per-request field forces the walk on every step of every deployment, because any
queued request might carry a bound. Timings of the loop on stand-in objects (150-entry `__dict__`,
best of 9, CPython 3.12.4, Mac; not the real `Req`):

| State | 1 000 queued | 10 000 queued |
|---|---|---|
| Today, global off | 0.05 µs | 0.05 µs |
| Today, global on | 47 µs | 0.53 ms |
| Proposed, gate closed (field never seen, global off) | 0.05 µs | 0.05 µs |
| Proposed, gate open, global off, no queued request carries a bound | 44 µs | 0.46 ms |
| Proposed, global on, no queued request carries a bound | 69 µs | 0.75 ms |
| Proposed, global on, every queued request carries a bound | 88 µs | 0.94 ms |

Reading: ungated, the default path gains ≈ 44 ns per queued request per scheduler step; existing users
of the global timeout pay ≈ 22 ns more per queued request for the extra attribute read. Writing the
rule as a helper function instead costs 58 / 90 / 156 µs per 1 000 in the last three rows.

**Decision: gate with a sticky flag** (`_has_req_waiting_timeout`, rows 13-14 of §1.1). A server that
never sees the field does exactly what it does today; after the first request that carries it, the
server behaves like one with the global timeout on. The flag is set on every rank (requests are
broadcast) and read on the leader only; no collective depends on it. The simpler alternative — no
flag, always walk — is §8 Q5.

## 3. Abort response path

Flow for a request dropped from the queue: scan emits `AbortReq(rid, abort_message, finished_reason)`
→ `abort_request` pops it before any forward ("Abort method 1 ... requests that have not started
anything", `scheduler.py:5321-5324`) and forwards the reason (5329-5331; `_make_abort_req`, 5981-5992)
→ `SenderWrapper.send_output` (`managers/scheduler_components/output_sender.py:12-29`) → tokenizer
`handle_loop` (`tokenizer_manager.py:2354-2367`) → `_handle_abort_req` (3429-3494) builds
`{"text", "output_ids", "meta_info": {"id", "finish_reason", "weight_version", "e2e_latency",
"completion_tokens"}}` → `_handle_abort_finish_reason` (1798-1829): non-streaming raises
`fastapi.HTTPException(status_code, detail=message)` (1823-1828), streaming returns the dict (1829).

| Endpoint | Stream | HTTP status | What the client reads | Produced at |
|---|---|---|---|---|
| `/generate` | no | **503** | `{"object":"error","message":"Request waiting timeout reached.","type":"503","param":null,"code":503}` | `http_server.py:556-609` (handler for `HTTPException`, body at 603-609) |
| `/generate` | yes | **200** | one `data:` line: `{"text":"","output_ids":[],"meta_info":{"id":…,"finish_reason":{"type":"abort","status_code":503,"message":"Request waiting timeout reached."},…,"completion_tokens":0}}`, then `data: [DONE]` | `http_server.py:919-951` |
| `/v1/chat/completions`, `/v1/completions` | no | **503** | the same error object as `/generate` | `serving_base.py:109-112, 195-211` |
| `/v1/chat/completions`, `/v1/completions` | yes | **200** | `data: {"error":{"object":"error","message":"Request waiting timeout reached.","type":"SERVICE_UNAVAILABLE","param":null,"code":503}}`, a usage chunk if requested, `data: [DONE]`. No choice chunk, hence no OpenAI `finish_reason` | `serving_chat.py:2066-2084`, `serving_completions.py:354-370`; `serving_base.py:213-227` |

**"503 + 'Request waiting timeout reached.'" is confirmed for non-streaming requests only.** For
streaming, the handlers start the generator before sending headers (`serving_chat.py:1930-1936`,
`serving_completions.py:206-210`), but the first item is the error event, not an exception, so the
response is a 200 stream whose first line is the error. A client that classifies by status code will
count a dropped streaming request as a success.

Further facts:

- No token is computed for a request dropped this way; nothing reaches a batch.
- The tokenizer manager turns only 400, 503 and 500 into HTTP errors (`tokenizer_manager.py:1808-1814`).
  Any other code (408, 429, 504) would come back as a normal response with an abort finish reason —
  the source-level reason to keep 503 (§8 Q3).
- Batched and `n > 1` requests are all-or-nothing when not streaming: the first 503 fails the HTTP
  request and the cleanup aborts the dispatched siblings (`tokenizer_manager.py:922-931, 3692-3716`).
- Already covered upstream, no new test needed: the 503 mapping
  (`test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py:274-299, 1069-1117`) and the
  end-to-end global timeout (`test/registered/scheduler/test_scheduler_control.py:309-325` with
  `python/sglang/test/kits/abort_timeout_kit.py:56-94`).
- **Unverified:** the streaming branches test `isinstance(status_code, HTTPStatus)`
  (`serving_chat.py:2073-2075`, `serving_completions.py:360-362`). That holds under the default pickle
  IPC (`SGLANG_USE_PICKLE_IPC = EnvBool(True)`, `environ.py:367`). With msgpack IPC the reason is typed
  `Dict[str, Optional[Union[str, int, List[int]]]]` (`io_struct.py:1490`) and would arrive as a plain
  int, sending the request down the "graceful abort" branch (a final chunk with `finish_reason:
  "abort"` instead of the error event). Reasoned from the types; msgspec was not available to run.

## 4. Validation

| Class | Kind | Validation it gets |
|---|---|---|
| `CompletionRequest`, `ChatCompletionRequest` (`protocol.py:351, 858`) | pydantic `BaseModel` (via `PDRoutingFields`, 334) | FastAPI body validation; `RequestValidationError` → 400 (`http_server.py:613-670`) |
| `GenerateReqInput` (`io_struct.py:176-177`) | stdlib `@dataclass` (existing, grandfathered by `no-dataclasses`) | pydantic only when FastAPI parses `/generate` (`http_server.py:909-915`); plain construction on every other path; `_validate_inputs` (440) on **every** path through `normalize_batch_and_arguments` (402-438), called at `tokenizer_manager.py:867` before any request state exists (890) |
| `TokenizedGenerateReqInput` (`io_struct.py:1041`) | `msgspec.Struct` with `tag=True, kw_only=True, array_like=True` (86) | none on construction |
| `Req` (`schedule_batch.py:1005`) | plain class | none |

How comparable optional numbers are validated today: `temperature: float = Field(default=1.0, gt=0,
allow_inf_nan=False)` (`protocol.py:1405, 1548`); `Annotated[float, Field(ge=0.0, le=0.99,
allow_inf_nan=False)]` (841); `field_validator("max_tokens")` (424-429); cross-field checks raising
`ValueError` inside `_validate_inputs` (`io_struct.py:452-467`) and at 429-430.

**Two layers, one authority.**

1. *Authority, all paths* — appended to `_validate_inputs` after line 467:

   ```python
           if self.waiting_timeout is not None and not (
               isinstance(self.waiting_timeout, (int, float))
               and 0 < self.waiting_timeout < float("inf")
           ):
               raise ValueError(
                   "waiting_timeout should be a positive, finite number of seconds."
               )
   ```

   It covers the paths pydantic never sees: `/vertex_generate`, the gRPC bridge, a future header
   override or Engine argument. NaN fails both comparisons; a string fails the `isinstance`.
2. *Schema, OpenAI models* — `Field(default=None, gt=0, allow_inf_nan=False)` on rows 1-2: the house
   pattern, an OpenAPI entry, and a 400 before any template or tokenizer work.

Behaviour of layer 2, checked on a stand-in model with pydantic 2.10.3: `0`, `-1`, `NaN`, `Infinity`,
`1e400`, `"abc"`, `[1]` are rejected; `"3"` becomes 3.0 and `true` becomes 1.0 (lax mode); `null` and
absence give `None`. A plain `Optional[float]` dataclass field without layer 1 accepts `0`, `-1` and
`inf`.

Where a rejection surfaces: OpenAI endpoints → 400 (`http_server.py:613-670`; for `ValueError`,
`serving_base.py:113-118`, `serving_chat.py:1933-1936, 2291-2292`); `/generate` non-streaming → 400
(`http_server.py:958-960, 2145-2149`); `/generate` streaming → HTTP 200 with an error event whose
`code` is 400 (`http_server.py:927-944`), as for every other normalisation error on that path.

## 5. Tests

### 5.1 How the existing unit tests build a scheduler without a GPU

`test/registered/unit/managers/test_scheduler_timeouts.py` (registered `base-a-test-cpu`, line 34):

- `maybe_stub_sgl_kernel()` runs before `scheduler` is imported (25-28); on CPU it installs stub
  modules for `sgl_kernel.*` (`test/registered/unit/README.md`).
- `_scheduler()` (66-78) calls `Scheduler.__new__(Scheduler)` and sets only what the method under test
  reads: `waiting_queue`, `running_batch`, `last_batch`, `result_queue`,
  `enable_continuous_input_polling`, three cache flags, `ipc_channels` and `beam_coordinator` as mocks.
- `_FakeReq` (37-53) stands in for `Req`: `rid`, `cache_request_handle`, `to_finish`, `finished()`, and
  `time_stats = SimpleNamespace(wait_queue_entry_time=…, forward_entry_time=…, trace_ctx=MagicMock())`.
- Time is real `time.perf_counter()` with offsets (`now - 10`), not a patched clock; the knobs are set
  with `envs.SGLANG_REQ_WAITING_TIMEOUT.override(value)` (118, 129, 134).
- The running-timeout class publishes a topology (`enter_scope(self, published_topology())`, 139-140)
  because `_collect_inflight_batches` reads `get_parallel().pp_size` (`scheduler.py:5294`); the waiting
  cases need none. One case drives the real `abort_request` (169-187).

Abort tests next to it: `test_tokenizer_manager_rid_cleanup.py` builds a `TokenizerManager` the same
way (`__new__`, 116-151) and covers the abort payload (376-420) and the status mapping (274-299,
1069-1117); `test_scheduler_sampling_mask_validation.py:19-72` drives `handle_generate_request` on a
`__new__` scheduler.

### 5.2 Changes the PR forces on existing tests

- `_FakeReq` and `_req()` gain `waiting_timeout=None`; `_scheduler()` gains
  `has_req_waiting_timeout=False`. Without the first the three existing waiting cases raise
  `AttributeError` — the price of reading the attribute directly, which the rules require.
- No other unit test calls the real scan: the PD and MLX tests replace `_poll_timeout_aborts` or
  `ingest_requests` (`test/registered/unit/disaggregation/test_prefill_result_polling.py:215, 460`;
  `test/registered/unit/managers/test_disagg_idle_step_counters.py:494`). The two tests that call the
  real `handle_generate_request` pass a `MagicMock` request, which supplies the attribute
  (`test_scheduler_sampling_mask_validation.py:43-49`;
  `test/registered/unit/dllm/test_gemma4_uniform_lifecycle.py:392-397`).
- "Field absent = today's behaviour" needs **no new case**: the three existing waiting cases (112-135)
  are that guard and must stay green unmodified apart from the fixture.

### 5.3 New cases

| Case | File, class | Set-up | Asserts | A future diff that turns it red | Category |
|---|---|---|---|---|---|
| `test_effective_bound_is_the_smaller_of_request_and_global` | `test_scheduler_timeouts.py`, `TestWaitingTimeout` | one fake request that has waited 10 s, scan armed; six `subTest` rows (request, global, expected): (1, 100, abort), (100, 1, abort), (100, 1000, none), (1, −1, abort), (100, −1, none), (`None`, −1, none) | emitted rids; `status_code == 503` on the aborting rows | row 1: "global only" or `max`; row 2: the request overriding the operator's bound; row 3: a predicate that fires whenever the field is present; row 4: restoring `if global > 0:` around the loop; row 5: `min(req, global)` with the −1 sentinel — every request carrying the field dies at once on a default server; row 6: naive defaulting once the scan is armed | derived property (boundary and sentinel math) |
| `test_queue_is_not_walked_until_a_request_carries_a_bound` | same class | `waiting_queue` is a list subclass whose `__iter__` raises; flag off; global −1 | the scan returns `[]` | removing the gate: no functional test notices an O(queue) walk on the default path | derived property with a silent failure mode |
| `test_generate_request_carries_its_bound_and_arms_the_scan` | `test_scheduler_timeouts.py`, new class | `__new__` scheduler as in `test_scheduler_sampling_mask_validation.py:24-42`, unified mode, a real `TokenizedGenerateReqInput` (built as in `test_io_struct.py:115-127`, plus `bootstrap_port` so line 2765 needs no context) with `waiting_timeout=1.5`; `_add_request_to_queue` and `init_req_max_new_tokens` mocked; `mm_input_error` passed so the method returns at `scheduler.py:2912-2920`; topology published for `set_finish_with_abort` (`schedule_batch.py:2151`) | the `Req` handed to the queue has `waiting_timeout == 1.5` and the flag is `True`; without the field: `None` and `False` | dropping the keyword at 2812 or the arming lines: the field would be ignored with the global knob off and every other test would still pass | critical-path bookkeeping |
| `test_getitem_preserves_waiting_timeout` | `test_io_struct.py`, `TestGenerateReqInputNormalization` | two prompts, `sampling_params={"n": 2}`, `waiting_timeout=1.5` | all four sub-requests carry 1.5 | dropping the `__getitem__` line: batched and `n > 1` requests lose the bound while single ones keep it. Precedents: `test_io_struct.py:1104-1115, 1270-1278, 1306-1320` | critical-path bookkeeping |
| `test_waiting_timeout_must_be_positive_and_finite` | same class | `subTest` over `0`, `-1.0`, `inf`, `nan`, `"2"` | `normalize_batch_and_arguments()` raises `ValueError` | weakening the check: `0` would abort at once, `inf` and `nan` never fire, and `/vertex_generate` has no other guard | validation boundary |
| `test_waiting_timeout_reaches_internal_request` | `test/registered/unit/entrypoints/openai/test_serving_completions.py`, `ServingCompletionTestCase` (fixture at 69-91, model: 98-108) | `CompletionRequest(..., waiting_timeout=1.5)` | `internal.waiting_timeout == 1.5` | dropping the keyword at `serving_completions.py:138` | critical-path bookkeeping |
| the same for chat | `test/registered/unit/entrypoints/openai/test_serving_chat.py` (pattern: 693-737) | `ChatCompletionRequest(..., waiting_timeout=1.5)`, `_process_messages` patched | `adapted.waiting_timeout == 1.5` | dropping the keyword at `serving_chat.py:1311`; a separate line, a separate failure | critical-path bookkeeping |

Optional eighth case, for the tokenizer hop (`tokenizer_manager.py:1535`): build the manager with
`_make_tokenizer_manager` (`test_tokenizer_manager_rid_cleanup.py:116-151`), call
`_create_tokenized_object` on a normalised `GenerateReqInput` and assert the field. Keep it only if
the fixture stays under ≈ 20 lines; otherwise the GPU case of §5.6 covers that hop.

The third and the optional case were designed by reading the code paths only; they need one run in a
real environment before they are trusted.

### 5.4 Cases considered and rejected under the admission rule

| Idea | Why not |
|---|---|
| "Field absent behaves as before" | duplicates the three existing cases |
| pydantic rejects `waiting_timeout=0` on the models | tests pydantic's `gt=0`; the tested authority is layer 1 of §4 |
| msgpack round trip of the new tail field | an `Optional[float]` round trip tests msgspec; the Rust lockstep test (`test_io_struct.py:51-76`) already guards the prefix |
| 503 mapping in the tokenizer manager | covered (`test_tokenizer_manager_rid_cleanup.py:274-299, 1069-1117`) |
| "`Req` stores the attribute" | tautology |
| "a re-queued request gets a fresh clock" | with a fake request this is a number compared with itself; the restamp belongs to `set_wait_queue_entry_time`, exercised at `test/registered/unit/observability/test_req_time_stats.py:184-266` |

### 5.5 How they run

```bash
# local, from the repository root, in an environment with SGLang's dependencies
python3 test/registered/unit/managers/test_scheduler_timeouts.py -v
pytest test/registered/unit/managers/test_scheduler_timeouts.py \
       test/registered/unit/managers/test_io_struct.py \
       test/registered/unit/entrypoints/openai/test_serving_completions.py \
       test/registered/unit/entrypoints/openai/test_serving_chat.py -v

# CI form (one file per process)
cd test && python3 run_suite.py --hw cpu --suite base-a-test-cpu
```

CI: job `base-a-test-cpu` (`.github/workflows/pr-test.yml:327-335`) on `ubuntu-latest`, Python 3.10,
installed with `uv pip install -e "python[dev]" --index-strategy unsafe-best-match --prerelease allow`,
then `run_suite.py --hw cpu --suite base-a-test-cpu` with partition flags
(`.github/workflows/_pr-test-stage-cpu.yml:53, 85, 121, 159`). Three of the four files are in that
suite; `test_io_struct.py` is registered for `stage-b-test-cpu-intel`, CUDA `base-b` / `1-gpu-small`
and an AMD suite (45-47), so its two new cases run there. The README's changed-line coverage check
(`diff-cover … --fail-under=60`) is a local recommendation; no workflow runs it at the pin.

### 5.6 Optional GPU case

A `RequestWaitingTimeoutMixin` in `python/sglang/test/kits/abort_timeout_kit.py` and one class in
`test/registered/scheduler/test_scheduler_control.py` beside `TestAbortWithWaitingTimeout` (309-325):
server with `--max-running-requests=1` and **no** global knob; start one long request, wait until it
runs, send a second with `"waiting_timeout": 0.001`; assert the second is an error with `code == 503`
and the first completes. It is the only test that fails if any single hop of §1.1 is dropped. Cost:
one more server start on `base-b` / `1-gpu-small` (the file is registered with `est_time=421`).
Offered to the maintainers, not assumed (§8 Q11).

## 6. Edge cases and failure modes

| # | Case | Evidence | Decision |
|---|---|---|---|
| E1 | Batch and `n > 1`: one scalar for all sub-requests | `io_struct.py:938-1038`; all batch paths use `obj[i]` (§1.2) | add row 7; guarded by a test |
| E2 | `n > 1` sends a prefix-cache warm-up request first; it carries the bound too | `tokenizer_manager.py:2028-2048` (copy of the tokenized object, `max_new_tokens = 0`) | accept: the bound applies per queue entry, so total queue wait can reach twice the value. If the warm-up times out: non-streaming → 503 for the whole request; streaming → the abort is discarded at 2048 and the samples are sent anyway (2061-2080). Same as the global timeout; documented |
| E3 | Retracted or preempted request re-enters the queue | §2.4; the re-entry also passes the queue-limit check (`scheduler.py:3277`) | the clock restarts; a streaming request can be dropped mid-stream. Same clock as the global bound (§8 Q6) |
| E4 | Waiting on grammar compilation | §2.3 | not counted, unchanged; the grammar queue keeps its own bound (§8 Q9) |
| E5 | `--max-queued-requests` | `_abort_on_queued_limit` (`scheduler.py:3337-3381`), checked on every entry (3277); "ignored when using disaggregation-mode" (`arg_groups/fields/schedule.py:35-38`) | independent. Shorter stays free slots sooner; because new requests are dispatched before the aborts of the same pass (§2.2), a slot freed by a timeout is usable one step later |
| E6 | Priority scheduling | eviction at 3348-3367 picks by `(priority, wait_queue_entry_time)`; preemption re-queues at 4080-4082 | no interaction beyond E3. If the queue limit evicts a request the scan also selected in the same pass, the later abort finds nothing and the tokenizer ignores an unknown rid (`tokenizer_manager.py:3437-3444`) |
| E7 | Multi-rank consistency (TP, attention DP, PP) | §2.2 | inherited, provided the field is compared only in the scan; a TP = 2 repeat is an optional part of D3 |
| E8 | Metrics | scheduler-side timeout aborts are not counted: `sglang:num_aborted_requests_total` is incremented only in `TokenizerManager.abort_request` (`tokenizer_manager.py:2158-2162`; `observability/metrics_collector.py:1701-1705, 1927-1928`); the queue-time histogram is observed at first forward only (`req_time_stats.py:769-773`) | no metric in the first PR; a counter is a follow-up (§8 Q10). **Never a label**: the value is unbounded (compare the `priority` label, `tokenizer_manager.py:3086-3089`) |
| E9 | msgspec and IPC order | §1.1 notes | tail append; lockstep test stays green |
| E10 | Abort by rid **prefix** | every abort branch matches `req.rid.startswith(recv_req.rid)`, including running requests, which then get the timeout message and 503 (`scheduler.py:5316, 5374, 5385, 5393, 5401, 5416, 5430-5444`; `grammar_manager.py:115`). A string rid on a batch expands to `f"{rid}_{i}"` (`io_struct.py:743`), so `X_1` is a prefix of `X_10`…`X_19` | pre-existing for the global timeout and for `/abort_request`; out of scope, named in the PR body. Server-generated rids are fixed-width (`io_struct.py:554, 741`). Any benchmark client must use prefix-free rids |
| E11 | Old servers, the Rust front end, typos | §1.4 | documented; D2 has a control run on the unpatched build |
| E12 | Engine paused | §2.2; a retract-mode pause re-queues running requests (`scheduler.py:5514-5522`) | bounds keep expiring during a pause, as today; documented |
| E13 | Error stubs wait in the queue too | `set_finish_with_abort` then `_add_request_to_queue` (e.g. `scheduler.py:3040-3043`) | such a request can receive the timeout 503 instead of its own 400; negligible, same as today |
| E14 | Sessions | §1.3 | excluded |
| E15 | PD mode | §1.4 | no PD-specific code; behaves as the global timeout does there; not exercised (§8 Q7) |
| E16 | Resolution | one scan per `ingest_requests`; §2.2 | the abort arrives at bound + at most one scheduler step (more with `--scheduler-recv-interval` > 1) |
| E17 | PD prefill continuous input polling calls `ingest_requests` in a tight loop | `disaggregation/prefill.py:730-747, 782-840` | one more reason for the gate of §2.7 |
| E18 | Integer values, booleans | pydantic coerces to float; plain construction keeps an `int` | harmless: the comparison is numeric |

## 7. Implementation

### 7.1 Steps in order

| # | Step | Check |
|---|---|---|
| 0 | Branch `feat/req-waiting-timeout` from `upstream/main`, SHA recorded; an environment that can import `sglang.srt.managers.scheduler` (§10) | `python3 test/registered/unit/managers/test_scheduler_timeouts.py` green before any edit |
| 1 | `Req`: signature and attribute (rows 11-12) | — |
| 2 | Test fixture (§5.2), then the two scan cases, then the scan and the gate state (rows 13, 15) | matrix rows 1 and 4 fail on the unmodified scan (it knows only the global value); all six pass after; a `min()` shortcut would fail rows 5 and 6; the three old waiting cases unchanged |
| 3 | `io_struct.py`: field, validation, `__getitem__`, tail field (rows 5-8) with their two cases | `test_io_struct.py` green, including the Rust lockstep test |
| 4 | Tokenizer manager and scheduler keywords, gate arming (rows 9, 10, 14) with the intake case | green |
| 5 | Models and conversions (rows 1-4) with the two conversion cases | green |
| 6 | Docs: one row in the `/generate` table of `docs/docs/basic_usage/sampling_params.mdx` (after `routed_experts_start_len`, row at 113); one clause in the `SGLANG_REQ_WAITING_TIMEOUT` row of `docs/docs/references/environment_variables.mdx:72-74` | — |
| 7 | Pre-commit on the changed files (isort 7.0.0, ruff and ruff-format v0.15.1, codespell 2.4.1, the `check-registered-tests` hooks) | clean |
| 8 | D2, D3 (§7.3) | thresholds below |

### 7.2 Estimated diff

| File | + | − |
|---|---|---|
| `entrypoints/openai/protocol.py` | 4 | 0 |
| `entrypoints/openai/serving_chat.py` | 1 | 0 |
| `entrypoints/openai/serving_completions.py` | 1 | 0 |
| `managers/io_struct.py` (3 field, 7 validation, 1 split, 3 tail field) | 14 | 0 |
| `managers/tokenizer_manager.py` | 1 | 0 |
| `managers/schedule_batch.py` | 3 | 0 |
| `managers/scheduler.py` (2 state, 1 keyword, 2 arming, 10 scan) | 15 | 5 |
| **Source** | **39** | **5** |
| Docs, 2 files | 6 | 1 |
| `test_scheduler_timeouts.py` | ≈ 75 | 3 |
| `test_io_struct.py` | ≈ 28 | 0 |
| `test_serving_completions.py`, `test_serving_chat.py` | ≈ 22 | 0 |
| **Unit tests** | **≈ 125** | **3** |
| **Total** | **≈ 170** | **≈ 9** |
| Optional: tokenizer-hop case | ≈ 22 | |
| Optional: GPU case (kit mixin and class) | ≈ 60 | |

### 7.3 Debug ladder

**D1 — unit tests.** Commands in §5.5. Accept: the new cases pass; the three existing waiting cases
pass with the fixture change only; `test_io_struct.py` passes including
`test_rust_tokenized_generate_schema_stays_in_lockstep`. Two practical conditions: the Mac cannot run
them as it stands (§9 P3, §10), and the lockstep case reads
`rust/sglang-server/src/message/io_struct.rs` from the repository root
(`test/registered/unit/managers/test_io_struct.py:53-57`), so the checkout must contain `rust/` — the
sparse `sglang-slo` worktree does not.

**D2 — dummy-weight server (node).** Launch, assembled from
`python/sglang/test/mock_model/utils.py:17-59` and `python/sglang/test/test_utils.py:698-709`; not run:

```bash
SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=1 \
SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT=1 \
sglang serve --model-path Qwen/Qwen3-0.6B \
  --load-format dummy \
  --sampling-backend token_oracle \
  --cuda-graph-backend-prefill=disabled \
  --kv-canary raise \
  --max-running-requests 1 \
  --host 127.0.0.1 --port 30000
```

`token_oracle` is an accepted `--sampling-backend` value only while
`SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE` is set (`server_args.py:345-353`). The config and tokenizer of
`Qwen/Qwen3-0.6B` must be in the local Hugging Face cache; whether `--load-format dummy` also wants
the weight files present was not checked. If the canary objects to an abort path, repeat with
`--kv-canary log` and report that separately.

```bash
B=http://127.0.0.1:30000; J='Content-Type: application/json'

# R0  idle server, no field -> 200
curl -s -o r0.json -w '%{http_code} %{time_total}\n' $B/generate -H "$J" \
  -d '{"text":"hi","sampling_params":{"max_new_tokens":16,"ignore_eos":true}}'

# R1  occupy the only slot
curl -s -o r1.json $B/generate -H "$J" \
  -d '{"rid":"hold-1","text":"hold","sampling_params":{"max_new_tokens":8192,"ignore_eos":true}}' &
sleep 2

# R2  bounded, non-streaming -> 503 after about 1 s
curl -s -o r2.json -w '%{http_code} %{time_total}\n' $B/generate -H "$J" \
  -d '{"rid":"bounded-1","text":"hi","waiting_timeout":1.0,"sampling_params":{"max_new_tokens":16,"ignore_eos":true}}'

# R3  bounded, streaming -> HTTP 200, one data line with finish_reason.type "abort", status_code 503
curl -sN -w '\n%{http_code} %{time_total}\n' $B/generate -H "$J" \
  -d '{"rid":"bounded-2","text":"hi","stream":true,"waiting_timeout":1.0,"sampling_params":{"max_new_tokens":16,"ignore_eos":true}}'

# R4  chat, non-streaming -> 503      R5  chat, streaming -> HTTP 200 and data: {"error":{...,"code":503}}
curl -s -o r4.json -w '%{http_code} %{time_total}\n' $B/v1/chat/completions -H "$J" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"waiting_timeout":1.0}'
curl -sN -w '\n%{http_code} %{time_total}\n' $B/v1/chat/completions -H "$J" \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"hi"}],"max_tokens":16,"stream":true,"waiting_timeout":1.0}'

# R6  completions, non-streaming -> 503
curl -s -o r6.json -w '%{http_code} %{time_total}\n' $B/v1/completions -H "$J" \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"hi","max_tokens":16,"waiting_timeout":1.0}'

# R7  a bound that does not bind, R8  no bound: both wait for the slot, then 200
curl -s -o r7.json -w '%{http_code} %{time_total}\n' $B/generate -H "$J" \
  -d '{"rid":"loose-1","text":"hi","waiting_timeout":3600,"sampling_params":{"max_new_tokens":16,"ignore_eos":true}}' &
curl -s -o r8.json -w '%{http_code} %{time_total}\n' $B/generate -H "$J" \
  -d '{"rid":"free-1","text":"hi","sampling_params":{"max_new_tokens":16,"ignore_eos":true}}' &
wait

# R9  validation -> 400 each
for v in 0 -1 '"abc"'; do
  curl -s -o /dev/null -w "$v -> %{http_code}\n" $B/generate -H "$J" -d "{\"text\":\"hi\",\"waiting_timeout\":$v}"
done
```

Second run with `SGLANG_REQ_WAITING_TIMEOUT=2` added to the environment, R1 again, then three bounded
requests: `waiting_timeout` 30, 0.5 and absent. Control run: R1 and R2 against the unpatched build at
the recorded upstream SHA.

| # | Accept |
|---|---|
| D2.1 | R2, R4, R6: HTTP 503, message exactly `Request waiting timeout reached.`, 1.0 s ≤ `time_total` ≤ 1.5 s |
| D2.2 | R3, R5: HTTP 200; the stream carries the 503 payload of §3 and no token; ≤ 1.5 s |
| D2.3 | R1 is undisturbed: `meta_info.completion_tokens == 8192`, `finish_reason.type == "length"` |
| D2.4 | R7, R8: HTTP 200 with 16 completion tokens; R0: 200 |
| D2.5 | R9: 400 three times |
| D2.6 | Global run: 503 at 2.0–2.5 s (30), at 0.5–1.0 s (0.5), at 2.0–2.5 s (absent) |
| D2.7 | Control: R2 on the unpatched build returns 200 after R1 finishes — the field is ignored there |
| D2.8 | Server log: no traceback, no `kv_canary violation:` |

**D3 — real model (node).** The same R0–R9 on `Qwen/Qwen3-8B` launched without the mock flags and
variables (`sglang serve --model-path Qwen/Qwen3-8B --max-running-requests 1 --host 127.0.0.1 --port
30000`), same thresholds; then U3 of PLAN.md §3.3 at B\*. Read
`docs/cookbook/autoregressive/Qwen/Qwen3.mdx` first and record every flag that departs from it
(`--max-running-requests 1` is a test setting). Optional, needs its own approval: R1 and R2 at TP = 2
— accept: same 503, no hang (the broadcast path of §2.2).

## 8. Open questions and risks

### 8.1 Questions for the maintainers

| # | Question | Recommendation | Basis |
|---|---|---|---|
| Q1 | Name and unit | `waiting_timeout`, seconds, body field | mirrors `SGLANG_REQ_WAITING_TIMEOUT`. Two caveats to state: the name already denotes an unrelated PD transfer knob (`kv_mgr.waiting_timeout`, `disaggregation/common/conn.py:441`; `disaggregation/fake/conn.py:57`), and `comment-style` prefers the unit in the name — alternative `waiting_timeout_s`. Both conventions exist (`timeout_s`, `io_struct.py:1755`; `timeout`, 2316) |
| Q2 | Smaller-of-two or override | smaller of the two | a client can tighten the operator's bound, never extend it; the override form is the alternative (Triton's `allow_timeout_override`, as cited in PRB_RESEARCH.md §2, not re-checked here) |
| Q3 | Status code and message | 503 and the same message | only 400 / 503 / 500 become HTTP errors (`tokenizer_manager.py:1808-1814`); the existing end-to-end kit asserts `code == 503` (`python/sglang/test/kits/abort_timeout_kit.py:89-91`) |
| Q4 | Header form | not in the first PR | the tree already has `x-override-*` headers behind `SGLANG_ENABLE_REQUEST_HEADER_OVERRIDES` (`entrypoints/request_headers.py:10-33`; `environ.py:341`), applied on `/generate` and chat only (`http_server.py:917-918`; `serving_chat.py:1327-1331`). A later form is one table entry, `"x-override-waiting-timeout": ("waiting_timeout", float)`; layer 1 of §4 already validates it |
| Q5 | Gate or unconditional walk | gate | §2.7; dropping it saves 5 lines and one test and costs ≈ 44 ns per queued request per step everywhere |
| Q6 | Bound per stay, or first admission only | per stay | "same clock as the global timeout" (§2.4). A first-admission variant would test `req.retraction_count == 0` (`schedule_batch.py:1343-1344, 1974-1977`) |
| Q7 | PD mode | no PD-specific code; say in the PR body that the bound covers `waiting_queue` only, as the global one does, and that PD was not exercised | §1.4, §9 R2. Fallback if asked: apply the request bound in unified mode only |
| Q8 | `/v1/responses` | leave out | §1.3 |
| Q9 | Should grammar compile time count | no | unchanged semantics; the grammar queue has its own bound |
| Q10 | A counter for timeout aborts | follow-up | E8: timeout aborts are invisible in the metrics today, global ones included |
| Q11 | The GPU case of §5.6 | offer it | it is the only whole-path guard; it costs one server start |
| Q12 | Engine API argument | follow-up | §1.3 |

### 8.2 Risks

| Risk | Evidence | Mitigation |
|---|---|---|
| A naive `min` with the −1 sentinel aborts every request that carries the field | `environ.py:639`; `scheduler.py:3390` | the shape of §2.6; matrix row 5 |
| Streaming clients never see a 503 status; U1's interactive class is streaming | §3 | the benchmark client classifies by payload; the PR body and the docs row say so |
| A queue walk on the default path | §2.7 | gate and its test |
| A timeout abort hits other requests whose rid it prefixes | E10 | pre-existing; prefix-free rids in every benchmark; named in the PR body |
| PD behaviour is unexercised | §1.4 | Q7 |
| The field is silently ignored by older servers, the Rust front end, a typo | §1.4 | D2.7; docs |
| Unit tests cannot run on the Mac | §10 | a CI-like environment, or step 0 of the node session |
| `scheduler.py` churn; the open PD-timeout PR (#34457, not read) may touch the same function | 15 small hunks | rebase before opening; re-grep every line of §1.1 |
| The existing test file needs a fixture edit | §5.2 | part of the PR, called out in its description |

## 9. Discrepancies found

**PRB_RESEARCH.md**

| # | Claim | What the source shows |
|---|---|---|
| R1 | §1: "Neither variable is mentioned anywhere under `docs/`" | Both are documented at the pin: `docs/docs/references/environment_variables.mdx:72` (`SGLANG_REQ_WAITING_TIMEOUT`, "Timeout (in seconds) for requests waiting in the queue before being scheduled") and `:77`. The rename commit itself (`e6f7a372ef`, #18766, committed 2026-02-12) carried the two doc rows |
| R2 | §1: "In PD-disaggregation mode neither timeout is enforced" | Every PD event loop calls `ingest_requests` and with it the scan (`disaggregation/prefill.py:696, 782`; `disaggregation/decode.py:2930, 2977`; `scheduler_pp_mixin.py:280, 431`), since #38389 (`b23d835048`, 2026-09-07) replaced the bare `recv_requests` calls there. Both timeouts apply in PD mode to requests in `waiting_queue` and in flight. What is not covered are the bootstrap, prealloc and transfer queues |
| R3 | §3: `serving_responses.py:579` listed under "Request → `GenerateReqInput`" | Line 579 passes `priority` to `_generate_with_builtin_tools`, not to `GenerateReqInput` (533-570 has no `priority`). Inside (2695-2774) the value is only rebound (2707, 2774) and never applied to the rebuilt request (2739-2754). On `/v1/responses` the field does not reach the engine. `ResponsesRequest.priority` is also `int = 0`, not optional (`protocol.py:1840`) |
| R4 | §3: `tokenizer_manager.py:1535,1567` | 1567 is the embedding object; only 1535 concerns generation |
| R5 | §3: `Req` construction "`Scheduler.handle_generate_request`, `schedule_batch.py`" | incomplete: `priority` also enters a `Req` in `Session.create_req` (`session/session_controller.py:326`), `handle_embedding_request` (`scheduler.py:3452`) and the EPD stub (`disaggregation/encoder/receiver.py:2537`); it is replayed in `build_rebootstrap_payload` (`schedule_batch.py:2115`) and exposed by the Engine API (`entrypoints/engine.py:460, 575`) and the gRPC proto (`sglang.proto:106, 134`) |
| R6 | §3: enforcement is "one comparison per queued request, already rank 0 only" | (a) the gate is the leader of each attention-DP group at PP stage 0, and the scan is skipped while inputs are deferred (`scheduler.py:2064-2069`); (b) today there is **no** loop when the global value is off (3390), so the field adds a queue walk to the default path unless gated (§2.7) |
| R7 | §3 risks: the gRPC path builds requests outside `protocol.py`, "field silently ignored there" | three different cases (§1.4): the embedded Rust server ignores it silently; native gRPC generate has no proto field, so it cannot be sent; the gRPC OpenAI pass-through goes through the pydantic models and **would honour** it |
| R8 | §1: per-request headers are `x-smg-routing-key`, `x-data-parallel-rank`, plus a closed PR for a priority header | line references are right (`serving_base.py:259, 271`), but a header form for `priority` exists at the pin: `x-override-priority` (`entrypoints/request_headers.py:18`) |
| R9 | §3: "Response: identical to the global timeout (503, same message)" | identical, yes; but the global timeout itself answers 503 only to non-streaming requests (§3) |
| R10 | §3: "under 150 changed lines without tests" | not contradicted: ≈ 44 changed source lines (§7.2) |

**PLAN.md**

| # | Claim | What the source shows |
|---|---|---|
| P1 | §3.1 "same 503 response"; D2 "returns 503 after ≈ 1 s" | true for non-streaming requests; streaming ones get HTTP 200 with an error event. U1's interactive class streams, so `slo_client.py` must classify by payload |
| P2 | §3.1 "Not in the first PR: PD mode" | cannot be had passively: the field lands on the `Req` in PD servers too and the scan runs there (R2). It needs the explicit statement of Q7 or an explicit condition |
| P3 | §3.2 D1 runs on the Mac | the file imports `sglang.srt.managers.scheduler`; none of the Mac's five Python interpreters has `msgspec`, and the default one also lacks `orjson`, `fastapi`, `uvloop`, `setproctitle`, `pybase64`, `xgrammar`. PR-A's single-module stub harness does not extend to importing `Scheduler` |
| P4 | §3.2 D1 "field absent = today's behaviour" as a new case; "the existing file still passes" | the first duplicates three existing cases and would not be admissible; the second holds only after the fixture edit of §5.2 |
| P5 | §3.2 D2 flags "`--load-format dummy --sampling-backend token_oracle`" | incomplete: `token_oracle` is rejected unless `SGLANG_KV_CANARY_ENABLE_TOKEN_ORACLE=1` (`server_args.py:345-353`); upstream's helper also passes `--cuda-graph-backend-prefill=disabled`, `--kv-canary raise` and sets `SGLANG_KV_CANARY_ENABLE_WRITE_INPUT_ASSERT` (`python/sglang/test/mock_model/utils.py:19-59`) |
| P6 | §2 "the existing rank-0 scan" | see R6 (a) |

**BACKGROUND.md §2.1** (its own pin is `50be533d09`; its line numbers were checked against that commit)

| # | Claim | What the source shows |
|---|---|---|
| B1 | `srt/environ.py:637` | right at its pin; 639 at `734cf3cf3b`. Between the two pins `ingest_requests` gained `stop_at_pause` and the deferred-input clause of the gate |
| B2 | "Wall-clock decisions are taken on rank 0 only" | leader of each attention-DP group (R6 a) |
| B3 | "`POST /abort_request {"rid": …}` removes a request" | the match is a **prefix** match (E10). The client-emulated ladder aborts by client-chosen rid: its rids must be prefix-free, or aborting `r1` also aborts `r10`…`r19` |
| B4 | "Load shedding exists only globally: [the two timeouts] … answered with 503" | `--max-queued-requests` is a second global shedding knob (`scheduler.py:3337-3381`); "503" holds for non-streaming requests |
| B5 | Policy list | the choices also include `priority` (`arg_groups/fields/schedule.py:86-96`) |
| B6 | `tokenizer_manager.py:190,1862` | 1862 is the `try:` of the wait loop; the timeout is at 1867-1868 and the abort at 1876-1877, at both pins |

**Found on the way (not claims of those documents)**

- `/v1/responses` drops `priority` (R3): dead parameter, a candidate for a separate upstream report.
- A streaming-session turn aborted in the queue appears to leave the session in flight (§1.3; read,
  not reproduced). It affects the global timeout and `/abort_request` today.

## 10. Not verified

| Item | Why | Where it matters |
|---|---|---|
| Any runtime behaviour | no server and no SGLang test was run | all of §7.3 |
| The new test cases | written from the code paths; no environment can import the scheduler on the Mac | §5.3; first run is D1 |
| msgspec behaviour of the tail field (field order, decoding of shorter arrays) | msgspec is not installed locally; inferred from the Rust mirror, its tests (`rust/sglang-server/src/message/io_struct.rs:160-185`, "we emit 32 … Trailing defaulted fields are omitted") and the lockstep test | §1.1 row 8 |
| The streaming error event under msgpack IPC | reasoned from types | §3 |
| The session in-flight flag after a queue abort | read, not run | §1.3 |
| pydantic behaviour at the version SGLang pins | checked with 2.10.3 on stand-in models only | §4 |
| Whether `--kv-canary raise` tolerates aborted requests, and how long 8192 oracle tokens hold the slot | not run | D2 |
| What the open PD-timeout PR (#34457) changes | no network access in this study | Q7, §9 R2 |
| The scan timings on the real `Req` | measured on a stand-in | §2.7 |
| Whether `uv pip install -e "python[dev]"` resolves on macOS | not attempted | D1 on the Mac |

## 11. Added after session P1 (2026-10-05) — measured, not read

Session P1 ran the global waiting timeout on a real server (`Qwen/Qwen3-8B`, one H200, the pin plus
PR-A). What it settles for this plan:

| Topic | Measured | Where |
|---|---|---|
| §3, abort response | With `SGLANG_REQ_WAITING_TIMEOUT=2` and both slots busy, a queued request was answered after 2.02 s: chat and completions **streaming** → HTTP 200, `data: {"error": {"message": "Request waiting timeout reached.", "type": "SERVICE_UNAVAILABLE", "code": 503}}`, then `[DONE]` (chat adds a zero-usage chunk when `include_usage` is set); native `/generate` streaming → HTTP 200 and one chunk with `finish_reason: {"type": "abort", "status_code": 503, …}`; every **non-streaming** form → HTTP 503 with `{"object": "error", "message": "Request waiting timeout reached.", "type": "503", "code": 503}` | `results/pra/abort_responses/` |
| §7.3, forming a full batch | `--max-running-requests 2` with two greedy 3000-token requests held both slots for well over the 2 s bound; the queued requests were aborted as expected | `results/pra/abort_responses/capture.sh` |
| §7.3, request settings | The first request that samples with top-k / top-p on a fresh server was scheduled about 70 s late and blocked every request behind it, and no timeout abort was delivered during that stall. D2 and D3 requests must set `temperature: 0`, or the ladder must first send one sampled request and wait for its answer | `results/pra_usage.md` §3.2 |
| Clients | The stock benchmark counted an in-stream abort as a completed request with the requested output length. Fixed on the PR-A branch (`9eb681bef6`); `slo_client.py` must classify a response by its payload, not by the HTTP status | `results/pra_usage.md` §3.1 |
| §5.5, where D1 runs | The Mac cannot import the package (it has transformers 4.51 against the pinned 5.17, and no `msgspec`). In P1 the node ran the registered CPU unit files in the real environment in 13 s; D1 is the first step of P2 | `results/pra/step0_summary.txt` |

An independent check of this plan against the source was started on 2026-10-05; its findings are
applied in a dated correction block below when it reports.
