# Hypothesis and Acceptance Criteria — Qwen3.5-4B BCG DeepStack

> Keeps the "what is known" vs "what is suspected" vs "what would
> satisfy us" boundaries explicit so future readers do not have to
> reconstruct them from mixed prose.

## 1. Established facts (source-level only)

Every item here is directly verifiable in `source_audit.md`.

- **[F1]** `Qwen/Qwen3.5-4B` reports architectures
  `["Qwen3_5ForConditionalGeneration"]` (HF revision
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`).
- **[F2]** `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MoeForConditionalGeneration` are on the multimodal
  prefill-BCG allowlist in current SGLang `main`
  (`configs/model_config.py` line `1836-1847`).
- **[F3]** `Qwen3_5ForCausalLM.forward` (`models/qwen3_5.py` line
  `1408-1465`) accepts `input_deepstack_embeds` and in-place adds it
  to `hidden_states` on layers 0–2 when non-`None` and non-empty.
- **[F4]** `general_mm_embed_routine` (`managers/mm_utils.py` line
  `1108-1140`) allocates `input_deepstack_embeds` as a **fresh
  per-request `torch.zeros(...)`** and scatters per-modality DeepStack
  tiles into it. Address is not stable across requests.
- **[F5]** The prefill BCG buffer registry
  (`model_executor/cuda_graph_buffer_registry.py` line `867-877`)
  registers an `input_embeds` slot for multimodal models but **no
  `input_deepstack_embeds` slot**.
- **[F6]** The BCG replay closure `replay_layer_forward`
  (`model_executor/runner/prefill_cuda_graph_runner.py` line
  `1498-1519`) copies **only** `input_embeds` into the stable slot
  before `.replay()`. `input_deepstack_embeds` is not copied and is
  not forwarded into `.replay()`.
- **[F7]** `run_dummy_multimodal_deepstack_forward` (same file line
  `662-725`) exists and is called during warmup, but only for
  `capture_num_tokens[-1]`, using a local `torch.zeros(...)` that goes
  out of scope after the call. It is a Dynamo warmup, not a stable
  slot registration.
- **[F8]** No registered upstream test asserts DeepStack-active BCG
  replay correctness for any Qwen3.5 / Qwen3-VL model.
- **[F9]** PR #30872 (merged 2026-07-28) added the `input_embeds` slot
  and turned on Qwen3.5 in the BCG allowlist. PR #30868 (merged
  2026-07-19) added `run_dummy_multimodal_deepstack_forward`. Neither
  PR added a stable `input_deepstack_embeds` slot or a per-request copy.

## 2. Source-level observations that are *not yet* runtime evidence

These follow from §1 as reasonable readings but the runtime path may
override them; they must not be quoted as bugs.

- **[O1]** Because [F5] and [F6] hold, the BCG-captured graph's
  DeepStack contribution is expected to depend on whatever address the
  warmup tensor from [F7] left in the captured pointer table.
- **[O2]** Because [F7] only warms `capture_num_tokens[-1]`, the
  DeepStack branch for other captured shapes may either not exist in
  the captured graph, or, if it does, be captured on some other
  freshly-allocated pointer. Either way, replays on those shapes look
  fragile from source.
- **[O3]** Because [F3] gates DeepStack on `is not None and numel() >
  0`, a captured graph that never saw the tensor-valued branch would
  simply skip the `add_`, silently omitting DeepStack.
- **[O4]** No source path visible to this audit copies live
  DeepStack values into a stable buffer before each replay. So the
  captured graph appears to read a *stale* pointer — either zero
  (if the warmup buffer memory is still resident) or freed memory
  (if it has been reclaimed).

## 3. Runtime hypotheses (unverified — this is the point of §7)

Enumerated so the validation plan can predeclare what it must
disprove or confirm.

- **[H_A]** On `Qwen/Qwen3.5-4B` under BCG (`breakable` prefill backend
  or `--enforce-piecewise-cuda-graph`), an image request's BCG replay
  runs *without* the current-request DeepStack contribution, and the
  resulting hidden states / logits / tokens differ measurably from the
  eager path.
- **[H_B]** BCG replay of an image request runs *correctly* (contrary
  to [H_A]) because the runtime silently disables BCG for image
  requests (e.g., `can_run_graph` returns `False` when
  `forward_batch.mm_inputs is not None` or an equivalent gate), which
  would preserve correctness at the cost of the BCG performance
  premise.
- **[H_C]** BCG replay attempts to use DeepStack but Dynamo triggers a
  recompile / assertion at the first tensor-valued DeepStack shape
  not covered by the single-shape warmup in [F7]. This would surface
  as a crash or a `Falling back to eager execution` log, similar to
  the historical Qwen3-VL R1 recompile.
- **[H_D]** Some code path this audit missed already stabilises
  DeepStack for BCG replay (e.g., a runtime path in
  `general_mm_embed_routine` that writes DeepStack into a captured
  buffer, or a per-model override in `Qwen3.5ForConditionalGeneration`),
  and no defect exists.

Exactly one of these is expected to hold. The validation plan must
distinguish among them with direct evidence, not by elimination.

## 4. Unverified assumptions to be checked at run time

- **[A1]** Current upstream SGLang `main` builds and serves
  `Qwen/Qwen3.5-4B` end-to-end on our hardware with the frozen
  environment in `provenance.md` §3.
- **[A2]** BCG prefill can actually be enabled for `Qwen3.5-4B` from
  the CLI without silent fallback (verified by inspecting server logs
  and `can_run_graph` decisions per request, not by absence of a
  crash).
- **[A3]** vLLM (or another well-tested reference) can serve
  `Qwen/Qwen3.5-4B` and produce deterministic greedy output usable as
  a correctness reference. If not, an eager SGLang run replaces it.
- **[A4]** Our correctness fixture (`fixtures/*.png`, deterministic
  prompts) produces well-formed multimodal input on Qwen3.5-4B's
  processor — non-trivial because Qwen3.5 uses its own visual token
  scheme.
- **[A5]** Comparing hidden states / logits between BCG and eager
  paths on the same input is achievable via SGLang's existing hooks
  (return_logprob, return_hidden_states, or a small instrumentation
  patch limited to this branch); the validation plan must confirm
  which hooks are available.

## 5. Acceptance criteria (predeclared, before observing any result)

The validation plan must produce a verdict of exactly one of:

- **`PASS`** — Runtime evidence supports [H_D] or [H_B] with clean
  attribution: either DeepStack is preserved end-to-end, or BCG is
  provably not exercised (in which case the perf premise is not being
  claimed and correctness is trivially safe). Either outcome closes
  the investigation without an upstream defect claim.
- **`FAIL`** — Runtime evidence supports [H_A] or [H_C] with clean
  attribution: DeepStack is omitted / stale under BCG replay and
  either (a) outputs diverge from the eager reference beyond a
  matched-noise envelope, or (b) BCG replay assert-fails / recompiles
  at inference time. In this case the profiler-repo issue upgrades to
  a candidate upstream SGLang issue with a minimal repro.
- **`AMBIGUOUS`** — Divergence exists but cannot be cleanly attributed
  (e.g., insufficient reference determinism, bf16 noise floor
  swallowing the signal, environment drift). Recorded as-is; upstream
  filing is deferred pending a stronger repro.
- **`INFRA_FAILURE`** — Environment cannot be brought to the frozen
  provenance, or GPU shared-tenancy noise makes the measurement
  unreliable. Recorded as-is; the run is not counted against the
  hypothesis in either direction.

**Verdict is declared before the results are observed.** Any post-hoc
softening of these tiers (as happened in Qwen3-VL R6's original
machine-pass verdict) is forbidden without an explicit written
amendment recorded in this file.
