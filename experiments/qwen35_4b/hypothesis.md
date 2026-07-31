# Hypothesis and Acceptance Criteria — Qwen3.5-4B BCG DeepStack

> Keeps the "what is known" vs "what is suspected" vs "what would
> satisfy us" boundaries explicit so future readers do not have to
> reconstruct them from mixed prose.

## 1. Established facts (source-level only)

Every item here is directly verifiable in `source_audit.md` against
upstream SGLang `main` at frozen SHA
`89f4a80c1f5e71c1c960df120f1e03b43dfd3c1d`.

- **[F1]** `Qwen/Qwen3.5-4B` reports architectures
  `["Qwen3_5ForConditionalGeneration"]` (HF revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`).
- **[F2]** `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MoeForConditionalGeneration` are on the multimodal
  **breakable-CUDA-graph** (BCG) allowlist
  (`configs/model_config.py:1845-1848`,
  `multimodal_breakable_cuda_graph_supported_model_archs`). They are
  **not** on the piecewise-CUDA-graph (PCG / `tc_piecewise`) allowlist
  at lines `1836-1841`, which contains only `Cohere2VisionForConditionalGeneration`,
  `KimiK25ForConditionalGeneration`, `MiniMaxM3SparseForCausalLM`,
  `MiniMaxM3SparseForConditionalGeneration`.
- **[F3]** `Qwen3_5ForCausalLM.forward` (`models/qwen3_5.py` line
  `1408-1478`) accepts `input_deepstack_embeds` and in-place adds it
  to the layer-body output on layers 0–2 when non-`None` and
  non-empty. Downstream layers (3 … final) receive that modified
  hidden state as input; the DeepStack effect can therefore propagate
  through all later layers via the residual stream, though it is only
  injected at 0–2.
- **[F4]** `general_mm_embed_routine` (`managers/mm_utils.py`
  lines `1108-1140` for the allocator, `1247-1373` for the routing)
  allocates `input_deepstack_embeds` as a per-call
  `torch.zeros(...)` and scatters per-modality DeepStack tiles into
  it. The Python tensor object is fresh per call; its
  `.data_ptr()` is not stable by contract, but the CUDA caching
  allocator may reuse the same underlying address, so pointer
  equality across calls is not by itself informative.
- **[F5]** The prefill BCG buffer registry
  (`model_executor/cuda_graph_buffer_registry.py` line `867-877`)
  registers an `input_embeds` slot for multimodal models but **no
  `input_deepstack_embeds` slot**.
- **[F6]** `general_mm_embed_routine` copies `input_embeds` into
  `forward_batch.input_embeds` (the static slot) at
  `mm_utils.py:1361-1363`; **no analogous copy exists for
  `input_deepstack_embeds`**.
- **[F7]** The BCG replay closure `replay_layer_forward`
  (`model_executor/runner/prefill_cuda_graph_runner.py` line
  `1498-1519`) copies **only** `input_embeds` into the stable slot
  before `.replay()`. `input_deepstack_embeds` is not copied and is
  not forwarded into `.replay()`. The BCG backend's `.replay()`
  itself (`runner_backend/breakable_cuda_graph_backend.py:241-248`)
  ignores `**kwargs` and just replays the captured graph.
- **[F8]** `run_dummy_multimodal_deepstack_forward`
  (`prefill_cuda_graph_runner.py:662-725`) is called **only** from
  `tc_piecewise_cuda_graph_backend._run_compile_pass`
  (`runner_backend/tc_piecewise_cuda_graph_backend.py:214-216`),
  after `torch.compile` install, at `capture_num_tokens[-1]` only. It
  is a **Dynamo shape-stability warmup for TC piecewise**, not a BCG
  slot. BCG has **no** DeepStack warmup and its capture
  (`_run_forward`, line 606-649) drives the LM without the
  `input_deepstack_embeds` kwarg — so the branch inside
  `Qwen3_5ForCausalLM.forward` that fires the DeepStack `add_` is
  **cold** at BCG capture time.
- **[F9]** `can_run_graph` (`prefill_cuda_graph_runner.py:1004-1061`)
  rejects batches whose `forward_batch.input_embeds` or
  `forward_batch.replace_embeds` is not None. In the scheduler
  (`managers/schedule_batch.py:2233-2401`), `batch.input_embeds` is
  populated only when the request carries an API-level
  `req.input_embeds`; normal multimodal image requests leave it
  `None` and set `batch.multimodal_inputs` instead. So the
  "embed-carrying rejection" gate does **not** target normal image
  requests.
- **[F10]** No registered upstream test asserts DeepStack-active BCG
  replay correctness for any Qwen3.5 / Qwen3-VL model at the audit
  SHA.
- **[F11]** PR #30872 (merged 2026-07-28) added the `input_embeds`
  slot, added Qwen3.5 to the BCG allowlist, and introduced the
  `replay_layer_forward` copy of `input_embeds`. PR #30868 (merged
  2026-07-19) added `run_dummy_multimodal_deepstack_forward` on the
  TC piecewise warmup path. Neither PR added a stable
  `input_deepstack_embeds` slot or a per-request copy on the BCG
  code path.

## 2. Source-level observations that are *not yet* runtime evidence

These follow from §1 as reasonable readings but the runtime path may
override them; they must not be quoted as bugs.

- **[O1]** Because [F5], [F6], [F7], and [F8] all hold, the BCG-captured
  graph for Qwen3.5-4B is expected to contain no DeepStack `add_`
  kernels at all (the branch was cold at capture, and no slot exists
  for it), so a replay would omit DeepStack from the layer-body
  computation regardless of the pointer the live request carries.
- **[O2]** Because [F9] holds, image requests are not filtered out by
  the `input_embeds is not None` gate. They pass into
  `PrefillCudaGraphRunner.execute` (if BCG is otherwise enabled), and
  `general_mm_embed_routine` populates DeepStack from inside
  `model.forward` — after `can_run_graph` has already returned.
- **[O3]** Because [F3] gates DeepStack on `input_deepstack_embeds is
  not None and numel() > 0`, an otherwise-correct eager reference
  will differ from a BCG replay that silently omitted the branch by
  exactly the "DeepStack-zeroed" signature: identical layer 0–2 body
  outputs modulo the missing `add_(input_deepstack_embeds[:, sep :
  sep + hidden_size])` contribution, then attenuated / amplified
  downstream via the residual stream.
- **[O4]** Because BCG uses raw `torch.cuda.Graph` capture with no
  Dynamo involvement (unlike TC piecewise), a shape mismatch or a
  new tensor-valued branch cannot trigger a "recompile" or "eager
  fallback" the way it can under TC piecewise. Failure modes
  observable under BCG are (a) captured graph is missing kernels
  (silent divergence), (b) captured graph reads from freed memory
  (illegal memory / segfault), or (c) the runner refuses to serve
  the request (e.g., through a code path this audit did not find).

## 3. Runtime hypotheses (unverified — this is the point of §7)

Enumerated so the validation plan can predeclare what it must
disprove or confirm.

- **[H_A]** On `Qwen/Qwen3.5-4B` under BCG (default breakable prefill
  backend), an image request's BCG replay runs *without* the current-
  request DeepStack contribution, and the resulting hidden states /
  logits / tokens differ measurably from the eager path.
- **[H_B]** BCG runs for the image request but correctness is
  preserved because some code path (this audit missed) copies live
  DeepStack values into a stable buffer before each replay, or a
  runtime dispatch we did not find performs the DeepStack contribution
  outside the captured graph.
- **[H_C]** Image requests do not exercise the BCG replay path on
  this server configuration — a filter we did not see routes them to
  eager. Correctness is preserved trivially; the BCG performance
  premise is simply not being tested for image requests. This is a
  feature-gap outcome, not a correctness bug.
- **[H_D]** BCG replay attempts to run but produces a hard failure at
  inference time (illegal memory access, assertion, segfault). This
  would surface as a request failure or a server crash, not silent
  divergence.

Exactly one of [H_A], [H_B], [H_C], [H_D] is expected to hold. The
validation plan must distinguish among them with direct evidence
(execution-path telemetry from instrumentation, plus paired eager and
BCG outputs), not by elimination.

## 4. Unverified assumptions to be checked at run time

- **[A1]** Current upstream SGLang `main` at the frozen SHA builds
  and serves `Qwen/Qwen3.5-4B` end-to-end on our hardware with the
  environment in `provenance.md` §3.
- **[A2]** BCG prefill is actually enabled for `Qwen3.5-4B` in the
  server configuration under test (verified by inspecting server
  logs and the branch-local instrumentation's per-request
  execute-path record, not by absence of a crash).
- **[A3]** Our correctness fixture (`fixtures/image_bands.png`,
  deterministic prompt) produces well-formed multimodal input on
  Qwen3.5-4B's processor and produces
  `input_deepstack_embeds.abs().sum() > 0`. If DeepStack is trivially
  zero on this fixture the whole test is a no-op and the run is
  classified `INFRA_FAILURE` or `AMBIGUOUS`.
- **[A4]** Comparing hidden states / logits between BCG and eager
  paths on the same input is achievable via a small
  instrumentation patch limited to this branch (implemented in
  Step 2). SGLang's own `return_logprob` and `return_hidden_states`
  hooks may or may not be usable under BCG (return_logprob under BCG
  falls back to an eager tail per `can_run_graph:1038`); the runner
  must handle both cases.
- **[A5]** A deliberate "eager with DeepStack zeroed" ablation (see
  the diagnostic in §6 below) is achievable by an instrumentation
  hook that replaces the live `input_deepstack_embeds` with
  `torch.zeros_like(input_deepstack_embeds)` before the LM forward.

## 5. Acceptance criteria — machine verdict (predeclared)

The validation plan must produce a verdict of exactly one of the
following. **These verdict labels are the source of truth**; the
implementation (`scripts/verdict.py`) and any tooling must use these
exact strings.

- **`PASS_BCG_CORRECT`** — The image request demonstrably replays
  BCG (execute-path telemetry shows `_execute_body_capture` +
  `replay_layer_forward` for the image request) AND the resulting
  greedy tokens / logits match the eager reference within the
  eager-vs-eager noise envelope AND the branch-local instrumentation
  confirms DeepStack was non-zero at request time. Supports [H_B].
- **`FEATURE_GAP_EAGER_FALLBACK`** — The image request does *not*
  enter BCG replay (execute-path telemetry shows the eager runner
  handled it, despite BCG being enabled). Correctness is preserved
  trivially. The BCG performance premise for image requests is
  unverified. Supports [H_C]. **This is not "PASS" in the strong
  sense; it is a documented gap.**
- **`FAIL_BCG_DEEPSTACK`** — The image request demonstrably replays
  BCG AND one of the following is true: (a) live DeepStack is
  missing / stale in the captured graph (the runner-side instrument
  confirms `input_deepstack_embeds` was passed but the resulting
  hidden states match an "eager with DeepStack zeroed" ablation more
  closely than they match "eager normal"), or (b) BCG replay produces
  matched greedy-token / hidden-state divergence from eager that
  exceeds the eager-vs-eager noise envelope with the zero-DeepStack
  signature. Supports [H_A] (or [H_D] if the failure is a crash).
- **`AMBIGUOUS`** — Divergence exists but cannot be cleanly
  attributed (bf16 noise envelope swallows the signal; instrumentation
  disagrees with itself; DeepStack was zero on the fixture; matched
  control failed).
- **`INFRA_FAILURE`** — Environment cannot be brought to the frozen
  provenance, GPU-0 shared-tenancy makes the measurement unreliable,
  server fails to bring up, the runner aborts on a foreign-PID
  guard, or provenance-preflight fails on a hard pin.

**An eager fallback (`FEATURE_GAP_EAGER_FALLBACK`) is never labelled
"bug closed" or "full PASS".** It is a real outcome the validation
plan is designed to distinguish and it must be reported as such. Any
post-hoc rewrite of these tiers requires an explicit "Amendment N"
block in this file.

## 6. Predeclared diagnostic ablation

The validation plan must produce a **three-way comparison** on the
same fixture and the same server-side cache state, so the machine
verdict has an unambiguous attribution:

1. **`eager_normal`** — eager runner, DeepStack computed normally by
   `general_mm_embed_routine` and passed to the LM.
2. **`eager_zero_deepstack`** — eager runner with a branch-local
   instrumentation hook that replaces `input_deepstack_embeds` with
   `torch.zeros_like(input_deepstack_embeds)` **immediately before
   the LM forward call**, so the LM's `is not None and numel() > 0`
   guard is still True and the `add_` branch still runs, but the
   contribution is exactly zero. This is a diagnostic ablation to
   isolate the "no-DeepStack" signature; it is not production
   behaviour and must be explicitly labelled as such in the results.
3. **`bcg_normal`** — BCG runner, DeepStack computed normally.

Interpretation rule (predeclared, before observing any result):

- If `bcg_normal` output ≈ `eager_normal` output (within the
  eager-vs-eager noise envelope), and instrumentation confirms BCG
  replay was used → verdict is `PASS_BCG_CORRECT`.
- If `bcg_normal` output ≠ `eager_normal` output beyond the noise
  envelope, AND `bcg_normal` output ≈ `eager_zero_deepstack` output
  (within noise) → verdict is `FAIL_BCG_DEEPSTACK` with strong
  attribution: the BCG path is behaving like DeepStack was zeroed.
- If `bcg_normal` diverges from both `eager_normal` and
  `eager_zero_deepstack` → verdict is `AMBIGUOUS` (the divergence
  is real but does not match the DeepStack-missing signature; some
  other mechanism is at play).
- If instrumentation shows every image request went through the
  eager runner and not through BCG replay → verdict is
  `FEATURE_GAP_EAGER_FALLBACK` regardless of the other two arms.
