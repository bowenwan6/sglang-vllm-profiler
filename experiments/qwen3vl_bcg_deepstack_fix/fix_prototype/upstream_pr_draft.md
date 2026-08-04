# Upstream PR draft — Copy input_deepstack_embeds into a BCG replay slot

**Status.** Draft only — NOT for submission. This document is a design
artefact that establishes what the upstream PR would look like once
the six pre-modification requirements are all met (see
[`README.md`](./README.md)). No PR has been opened. No source has
been modified in `/data/sglang-fork` or in the pinned scratchpad
checkout.

**Target.** `sgl-project/sglang` `main` at
`e76d0acdc923d992bbda20d4b2bc51db9ac314a7` or newer (rebase to
tip-of-main at submission time).

---

## Proposed PR title

```
fix(cuda_graph): copy input_deepstack_embeds into a BCG replay slot
```

## Proposed PR body

### What this changes

Extend the existing `input_embeds` slot-and-copy pattern in the
prefill breakable CUDA graph (BCG) runner to `input_deepstack_embeds`.
Three sites (registry, capture-pass, replay-pass) — a symmetric
mirror of the pattern already landed for `input_embeds`.

### Why

`replay_layer_forward` in `prefill_cuda_graph_runner.py` currently
reads `layer_kwargs["input_embeds"]` and copies it into a stable slot
before `.replay()`. It does not do the same for
`input_deepstack_embeds` even though (a)
`general_mm_embed_routine` in `mm_utils.py` synthesises and passes it
per-request for DeepStack-carrying models (Qwen3-VL family) and (b)
the LM's `forward(...)` signature accepts it explicitly and applies
it to hidden_states at target layer indices.

The result on any DeepStack-carrying model that is (or becomes)
allowlisted for BCG prefill is that the captured graph is built with
the DeepStack `add_` branch cold and no consumer for the live tensor
— the DeepStack contribution is silently dropped and outputs diverge
from eager.

Today only `Qwen3_5ForConditionalGeneration` and its MoE variant
are on the BCG allowlist
(`sglang.srt.configs.model_config.multimodal_breakable_cuda_graph_supported_model_archs`),
and every shipped `Qwen/Qwen3.5-*` release carries
`deepstack_visual_indexes = []`, so no shipped configuration
currently triggers the failure. **This is a latent regression** that
would activate the moment a Qwen3.5 release ships with populated
DeepStack, or Qwen3-VL is added to the BCG allowlist. Landing this
fix defensively removes that trap.

### How

Three-site mirror of the landed `input_embeds` slot-and-copy:

1. **`cuda_graph_buffer_registry.build_prefill_registry`** — accept
   `num_deepstack_embeddings: int = 0`; when
   `is_multimodal and register_input_embeds and num_deepstack_embeddings > 0`,
   append a `GraphSlot("input_deepstack_embeds", (mt, hidden *
   num_ds), embed_dtype, axis="tokens", padding=ZERO,
   copy_from_fb=False)` alongside the existing `input_embeds` slot.
2. **`PrefillCudaGraphRunner.__init__`** — pass
   `num_deepstack_embeddings=getattr(self.model_runner.model,
   "num_deepstack_embeddings", 0)` to the builder. Precedent:
   `run_dummy_multimodal_deepstack_forward` already reads the
   attribute this way at `prefill_cuda_graph_runner.py:704`.
3. **`PrefillCudaGraphRunner._run_forward`** (BCG branch) — when
   the slot exists, pass `input_deepstack_embeds=<slot>` to
   `layer_model.forward` at capture time so Dynamo/capture traces
   the `add_` branch into the graph.
4. **`PrefillCudaGraphRunner._execute_body_capture.replay_layer_forward`**
   — after the existing `input_embeds` copy, mirror-copy
   `layer_kwargs["input_deepstack_embeds"]` into the slot,
   guarded on `numel() > 0`.

The fix is data-driven — models without DeepStack (`num_deepstack_embeddings == 0`)
see zero allocation and zero copy overhead. No public API breaks; no
new arguments to `_run_forward`, no new fields on `ForwardBatch`;
the DeepStack tensor keeps riding `layer_kwargs`.

### What this does not change

* **BCG allowlist.** Adding Qwen3-VL to
  `multimodal_breakable_cuda_graph_supported_model_archs` is an
  orthogonal policy decision and is out of scope for this PR.
* **TC piecewise path.** `run_dummy_multimodal_deepstack_forward` at
  `prefill_cuda_graph_runner.py:687-750` continues to serve TC
  piecewise unchanged.
* **`ForwardBatch` schema.** No new field.
* **`can_run_graph` / `can_replay_locally`.** Neither gate needs
  changes — an image request with populated DeepStack already passes
  today; the fix ensures the tensor survives replay.

### Repro (attach reviewer-runnable script)

Requires: any DeepStack-carrying model on the BCG allowlist. The
simplest way to reach it is a test-only monkey-patch that appends
`"Qwen3VLForConditionalGeneration"` to the allowlist at import time,
then a small image prefill against `Qwen/Qwen3-VL-2B-Instruct` (or
larger). Under pre-fix `main`:

* `bcg_normal` tokens are bit-identical to `bcg_zero_deepstack` and
  both track `eager_zero_deepstack`;
* `eager_zero_deepstack` diverges from `eager_normal` at the first
  non-boilerplate token (typical `l1_max_abs ≈ 1.14` on Qwen3-VL-8B
  with a fixed image fixture).

Under this PR:

* `bcg_normal` bit-matches `eager_normal` within the eager-vs-eager
  bf16 noise floor (`l1_max_abs ≤ 0.1`);
* `bcg_zero_deepstack` still matches `eager_zero_deepstack` (the
  ablation arm is unchanged — the fix does not affect the zero path).

### Test plan

Included in this PR (see file changes):

* Unit tests for slot registration (present iff
  `num_deepstack_embeddings > 0`; absent for `is_multimodal=False`;
  correct dtype/shape/axis/padding).
* Unit tests for `replay_layer_forward` DeepStack copy path (copy
  happens when slot present and kwarg populated; skipped for
  `None`, `numel() == 0`, or missing slot).
* Unit tests for capture-pass DeepStack kwarg presence (passed iff
  the slot exists).
* Integration test — bcg-vs-eager token equivalence on a small
  DeepStack-carrying model (test-only monkey-patch adds
  `Qwen3VLForConditionalGeneration` to the BCG allowlist for the
  test process).
* Integration test — no regression on a currently-shipping
  empty-DeepStack model (Qwen3.5-4B or similar) — bcg == eager
  pre-fix and post-fix.

Skeletons proposed by the reviewer in
[`regression_tests_skeleton.py`](./regression_tests_skeleton.py).

### Compatibility

* **No public API changes.** `build_prefill_registry` gets one new
  keyword argument with a safe default of `0`; every existing caller
  is unaffected.
* **No downstream behavior change** for models without DeepStack. The
  slot allocation gate makes the fix a pure no-op for
  Cohere2Vision, KimiK25, MiniMaxM3, Qwen3.5 (empty DeepStack today),
  and all text-only models.
* **No new heavy dependency.** Uses existing `GraphSlot`,
  `PaddingPolicy.ZERO`, `slice_for`, and `has_slot` APIs.

### Performance impact

Additive per-BCG-replay cost on DeepStack-carrying models: one
`torch.Tensor.copy_` of shape
`(num_tokens, hidden × num_deepstack_embeddings)` in the embed
dtype. On Qwen3-VL-8B at the largest bucket
(`num_tokens = 4096`, `hidden = 4096`, `num_ds = 3`, `bf16`) this is
`96 MiB` per replay copy — comparable in size to the existing
`input_embeds` slot copy and dominated by the surrounding
`.replay()` cost. Zero cost on models without DeepStack.

### Risk / concerns

* **Attribute-name convention.** `num_deepstack_embeddings` is the
  attribute already established by `qwen3_vl.py:1302` and read by
  `run_dummy_multimodal_deepstack_forward`. Any future model that
  wires DeepStack under a different attribute name would silently
  degrade to the current broken state; a matching test on the new
  model would catch this. Mitigated by keeping the attribute name
  aligned with the existing TC piecewise dummy.
* **Slot lifetime.** `input_deepstack_embeds` uses the same
  `slice_for(bs=1, num_tokens)` semantics as `input_embeds`; the
  slot is bs=1 (BCG capture is a single-request replay) and sized
  to `max_num_tokens × hidden × num_ds` up front. Follows the
  established pattern; no new lifecycle to reason about.
* **Speculative decoding / target_verify.** Verify paths route
  through `runner_backend/breakable_cuda_graph_backend.py`; the
  `.replay()` signature ignores kwargs, so this change does not
  affect their contract.

### References

* Symmetric pattern for `input_embeds` on current main:
  `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
  L867–877 (slot); `.../prefill_cuda_graph_runner.py`
  L660–668 (capture-pass); L1610–1628 (replay-pass copy).
* Related merged PRs:
  - #30868 (2026-07-19) — TC piecewise Dynamo warmup for DeepStack.
  - #30872 (2026-07-28) — introduced the `input_embeds` slot &
    copy pattern that this PR mirrors for `input_deepstack_embeds`.

---

## Evidence bundle referenced from the PR

Reviewers can be pointed at these documents for the source-level
audit and evidence:

* [`r1_upstream_audit.md`](../r1_upstream_audit.md) — full
  current-main audit of the three co-omission sites, with file:line
  citations.
* [`plan.md`](../plan.md) — sub-track plan revised to reflect
  current-main evidence.
* Historical live-fire evidence on Qwen3-VL under a monkey-patched
  BCG allowlist:
  `experiments/qwen35_4b/results/attempt_gpu1_20260801T115524Z/verdict.md`
  and its `verdict.json` — 4-arm 2×2 recorded on 2026-08-01 that
  produces the exact `FAIL_BCG_DEEPSTACK` signature this PR fixes.
* [`fix_prototype/patch.diff`](./patch.diff) — the actual diff
  proposed to upstream (verified via `git apply --check` against
  upstream `main @ e76d0acdc9`).
* [`fix_prototype/patch_notes.md`](./patch_notes.md) — annotated
  per-hunk rationale + risk register.
* [`fix_prototype/regression_tests_skeleton.py`](./regression_tests_skeleton.py)
  — test skeleton with fail-pre-fix / pass-post-fix conditions.

---

## Pre-submission checklist

Do **not** submit until every box is ticked:

- [ ] Six pre-modification requirements all met (see
      `fix_prototype/README.md`). Currently 4/6 (audit, fix
      design, regression tests, and clean `git apply --check`);
      requirements (1) reproducible current-main failure, (2)
      DeepStack-non-empty proof at LM entry, (3) BCG-replay
      specificity proof remain blocked on the shared CUDA/driver
      mismatch.
- [ ] R2 (baseline reproduction on the pinned SHA under the
      monkey-patch) reproduces Attempt 03's `FAIL_BCG_DEEPSTACK`
      signature bit-identically.
- [ ] Fix applied on a fresh clone of upstream `main` at
      submission-time HEAD (rebase from the shallow clone; do not
      apply to `/data/sglang-fork`).
- [ ] All regression tests written and passing on the fixed clone.
- [ ] `pytest test/srt/` clean on the fixed clone (or explicit
      justification for any pre-existing failure).
- [ ] BCG-vs-eager latency regression bench run on Qwen3-VL-2B or
      similar; results attached (≤ 5 % / ≤ 1 ms delta acceptable).
- [ ] Peak GPU memory delta measured; results attached (≤ 5 % / ≤
      100 MiB acceptable).
- [ ] User approval to open the PR.
