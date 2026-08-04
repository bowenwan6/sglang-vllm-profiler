# Patch notes — F-DeepStack fix prototype (annotated)

## Hunk 1 — `cuda_graph_buffer_registry.py`

**Change.** Add a `num_deepstack_embeddings` parameter to
`build_prefill_registry` (default `0`). Inside the
`is_multimodal and register_input_embeds` block, if
`num_deepstack_embeddings > 0`, append a new `GraphSlot(
"input_deepstack_embeds", ...)` with shape `(mt, hidden_size *
num_deepstack_embeddings)`, `axis="tokens"`, `PaddingPolicy.ZERO`,
`copy_from_fb=False`.

**Rationale.**

* Shape mirror of `input_embeds`. Different width because DeepStack
  packs `num_ds` per-layer contributions along the feature axis; the
  Qwen3-VL LM at `qwen3_vl.py:1136` unpacks
  `input_deepstack_embeds[:, sep : sep + hidden_size]`.
* `axis="tokens"` matches how `input_embeds` is registered — capture
  writes / reads at the padded-token boundary, slice_for handles the
  padding zero-fill.
* `padding_policy=PaddingPolicy.ZERO` matches `input_embeds`. Padded
  tokens read a zero contribution.
* `copy_from_fb=False` matches `input_embeds`. There is no
  `ForwardBatch.input_deepstack_embeds` field; the tensor rides
  `layer_kwargs`, copied in by `replay_layer_forward` (hunk 3).
* Zero-cost for models without DeepStack. Every currently-shipping
  `Qwen/Qwen3.5-*` has `num_deepstack_embeddings = 0` and pays no
  allocation.

## Hunk 2 — `prefill_cuda_graph_runner.py` `__init__`

**Change.** At the `build_prefill_registry(...)` call site (line
~314), pass
`num_deepstack_embeddings=getattr(self.model_runner.model, "num_deepstack_embeddings", 0)`.

**Rationale.**

* Attribute access is defensive — non-DeepStack models (Qwen3.5,
  Cohere2Vision, KimiK25, MiniMaxM3) do not define the attribute;
  `getattr` returns `0` and the slot is skipped.
* Precedent: `run_dummy_multimodal_deepstack_forward` already reads
  the attribute this way at `prefill_cuda_graph_runner.py:704`. Same
  contract, same fallback.

## Hunk 3 — `prefill_cuda_graph_runner.py` `_run_forward`

**Change.** In the BCG capture branch of `_run_forward`, if
`self.buffer_registry.has_slot("input_deepstack_embeds")`, grab
`slice_for(1, num_tokens)` and pass it as a kwarg to
`self.layer_model.forward(...)`. Otherwise fall through to the
existing 4-positional-arg call.

**Rationale.**

* The captured graph MUST contain the DeepStack `add_` kernels for
  the fix to work. That requires the kwarg to be present at capture
  time. Absent this hunk, hunk 1's slot is filled at replay time but
  the captured graph has no consumer.
* Using the slot buffer (not a fresh `torch.zeros`) means capture and
  replay read from the same `data_ptr`; the CUDA graph's kernel
  arguments stay valid.
* Zero content at capture: the slot's `PaddingPolicy.ZERO` initialisation
  produces a zero tensor for the first capture pass. The
  DeepStack `add_` traces as a no-op numerically at capture time
  (adds 0 to hidden_states) but the kernel node is recorded and the
  operand pointer is captured for the slot. Replay overwrites the
  slot with the live tensor and the `add_` becomes non-trivial.

## Hunk 4 — `prefill_cuda_graph_runner.py` `replay_layer_forward`

**Change.** After the existing `input_embeds` copy block, add a
symmetric block: if
`self.buffer_registry.has_slot("input_deepstack_embeds")`, read
`layer_kwargs.get("input_deepstack_embeds")`, guard on
`de is not None and de.numel() > 0`, and copy into the slot.

**Rationale.**

* Mirrors hunk 1's slot with hunk 3's capture. Live DeepStack from
  `general_mm_embed_routine` reaches `replay_layer_forward` via
  `layer_kwargs` (per the LM's `forward(..., input_deepstack_embeds=...)`
  signature and the fact that the outer `model.forward` passes it
  in as a kwarg).
* No positional-fallback path (unlike `input_embeds` at line 1621-1622)
  because `input_deepstack_embeds` is exclusively a kwarg in the LM
  signature: `qwen3_vl.py:1145` has it as `input_deepstack_embeds:
  Optional[torch.Tensor] = None` after 3 leading positional args.
* `numel() > 0` guard is defence-in-depth. Under normal routing the
  tensor is either populated (image request) or `None` (text-only).
  The guard also makes hunk 1's slot allocation redundant when the
  model happens to pass an empty non-None tensor — safety-net.

## Overall design invariants

* **No new abstractions.** The fix uses existing `GraphSlot`,
  `PaddingPolicy.ZERO`, `slice_for`, `copy_`, and `has_slot` APIs.
* **Zero-cost for non-DeepStack.** Slot allocation gate on
  `num_deepstack_embeddings > 0`.
* **No API breaks.** `build_prefill_registry` gets a new kwarg with a
  safe default; existing callers unchanged.
* **Symmetric with an already-landed pattern.** The three sites
  (registry, capture, replay) map 1:1 to how current-main handles
  `input_embeds`.
* **No BCG allowlist change.** Correctness fix only; the policy
  decision to add Qwen3-VL to the BCG allowlist is orthogonal and
  should not be bundled.

## What could go wrong

* **Slot pointer stability across bucket recompiles.** `input_embeds`
  handles this today — the slot is allocated once at
  `build_prefill_registry` and sliced per bucket via `slice_for(1,
  num_tokens)`. Following the same pattern preserves the stability
  contract.
* **Model attribute drift.** If a new model exposes DeepStack under a
  different attribute name (e.g., `num_deepstack_layers`), the fix
  silently degenerates to the current broken state. Mitigated by
  keeping the attribute-name convention consistent with the existing
  TC piecewise dummy at `prefill_cuda_graph_runner.py:704`.
* **`num_deepstack_embeddings` set but never non-zero at inference.**
  A model that declares DeepStack but never populates it (unusual)
  would allocate the slot for nothing. Waste is bounded by
  `max_num_tokens × hidden × num_ds × dtype_size` and matches
  `input_embeds`'s always-on cost.
* **Interaction with speculative decoding target_verify.** Verify
  paths route through
  `runner_backend/breakable_cuda_graph_backend.py`; the `.replay()`
  signature ignores kwargs, so hunk 4's addition does not conflict.
  The captured graph will have DeepStack traced regardless of
  whether target_verify uses the same shape bucket.

## What must be tested before this ever lands (see
`regression_tests_skeleton.py`)

* Slot allocated iff `num_deepstack_embeddings > 0`.
* Slot NOT allocated for Qwen3.5 (empty DeepStack).
* Capture-pass includes the DeepStack `add_` kernel(s) in the
  captured graph (nsys kernel-count diff).
* Replay copy actually reaches the slot (data_ptr check +
  post-replay comparison).
* BCG normal == eager normal within bf16 tolerance on a Qwen3-VL
  image request (end-to-end regression).
* Text-only requests unaffected.
* Mixed-batch requests correct.
* Bucket-independent: correct across the padded-bucket sweep.
