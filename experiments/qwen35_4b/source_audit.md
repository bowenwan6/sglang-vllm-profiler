# Source Audit — Qwen3.5-4B multimodal prefill BCG DeepStack

> **Scope.** Source-level reading of current upstream SGLang `main` and
> the HuggingFace `Qwen/Qwen3.5-4B` model card. Records *what the code
> does* — not what happens at runtime. Nothing in this file constitutes
> evidence of a runtime defect. Runtime evidence is `validation_plan.md`'s
> job.

## 1. Provenance of this audit

- **Upstream SGLang `main` HEAD read:**
  `5f9b0db18c787cf56ed9bbaf255f083f26c6ebc2` (2026-07-31; verified via
  `GET /repos/sgl-project/sglang/commits/main`, subject `Fix async
  loading of RunAI-streamed tensors (#32896)`).
- **Files fetched via `raw.githubusercontent.com/sgl-project/sglang/main/...`**
  and cached under
  `<scratchpad>/sglang_snapshot/*` during the audit; **not committed**
  to this repository (per artifact rules). Any citation in this file
  refers to upstream `main` at the SHA above; re-verification is the
  reader's responsibility if upstream `main` has moved since.
- **HF `Qwen/Qwen3.5-4B` metadata:** `sha =
  851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `pipeline_tag =
  image-text-to-text`, `config.architectures =
  ["Qwen3_5ForConditionalGeneration"]`, `config.model_type =
  "qwen3_5"`, `gated = false` (verified via `GET
  /api/models/Qwen/Qwen3.5-4B`).

## 2. Qwen3.5 architecture registration

- `python/sglang/srt/configs/model_config.py`
  - line `1793`: `Qwen3_5ForConditionalGeneration` and
    `Qwen3_5MoeForConditionalGeneration` in
    `multimodal_model_architectures`.
  - line `1836`: `multimodal_piecewise_cuda_graph_supported_model_archs`
    is the BCG / piecewise-CUDA-graph allowlist; it includes
    `Qwen3_5ForConditionalGeneration` and
    `Qwen3_5MoeForConditionalGeneration` (lines `1846-1847`).
  - `is_multimodal_piecewise_cuda_graph_supported` (line `1908`) checks
    membership in that list.
  - `is_multimodal_breakable_cuda_graph_supported` (line `1916`) — a
    separate accessor for the breakable variant.
- `python/sglang/srt/models/qwen3_5.py`
  - line `1771`: `class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`
    with `language_model_cls=Qwen3_5ForCausalLM`.
  - line `1928`: `class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration)`
    with `language_model_cls=Qwen3_5MoeForCausalLM`.
  - line `2319`: `EntryClass = [Qwen3_5MoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration]` — how the loader picks them up.

**Implication.** `Qwen/Qwen3.5-4B` reports
`architectures=["Qwen3_5ForConditionalGeneration"]`, so on current
upstream `main` with `--enable-multimodal` (the multimodal default),
`is_multimodal_piecewise_cuda_graph_supported` returns `True` and BCG /
piecewise capture is *allowed*. Whether it *runs* depends on the
prefill backend selection at server-startup time; that is what the
validation plan must observe with an inference-time flag audit.

## 3. Qwen3.5 language model DeepStack data flow

`python/sglang/srt/models/qwen3_5.py:1408-1465` — `Qwen3_5ForCausalLM.forward`:

```python
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: Optional[torch.Tensor] = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,
) -> Union[torch.Tensor, PPProxyTensors]:
    ...
    for layer_idx in range(self.start_layer, self.end_layer):
        layer = self.layers[layer_idx]
        ...
        hidden_states, residual = layer(...)

        # Process deepstack embeddings if provided
        if (
            input_deepstack_embeds is not None
            and input_deepstack_embeds.numel() > 0
            and layer_idx < 3
        ):
            sep = self.hidden_size * layer_idx
            hidden_states.add_(
                input_deepstack_embeds[:, sep : sep + self.hidden_size]
            )
```

Observations:

- The DeepStack contribution is only added for `layer_idx < 3` (first
  three decoder layers).
- Contribution is an in-place `add_` on `hidden_states` after the layer
  call. Skipping it silently produces a numerically different but not
  crash-inducing result.
- The gate is a **control-flow guard** on `input_deepstack_embeds is
  None` (via `is not None and numel() > 0`). Dynamo specialises on this
  guard, which is the source of the R1 recompile seen in the older
  Qwen3-VL sub-track (§4 in `plan.md`).

`Qwen3_5MoeForCausalLM` (line `1560`) subclasses `Qwen3_5ForCausalLM`
without overriding `forward`, so it inherits the same DeepStack path.

## 4. Multimodal wrapper DeepStack data flow

`python/sglang/srt/models/qwen3_vl.py`:

- line `1212`: `class Qwen3VLForConditionalGeneration(nn.Module):` — the
  base wrapper Qwen3.5 inherits.
- line `1301`: `self.deepstack_visual_indexes =
  config.vision_config.deepstack_visual_indexes` (per-model list of
  vision-block outputs that feed DeepStack).
- line `1302`: `self.num_deepstack_embeddings =
  len(self.deepstack_visual_indexes)`.
- line `1303`: `self.use_deepstack = {Modality.IMAGE: True,
  Modality.VIDEO: True}` — DeepStack is on for image and video by
  default.
- line `1416`: `hidden_states = general_mm_embed_routine(...,
  language_model=self.model, ..., use_deepstack=self.use_deepstack, ...)`.

`python/sglang/srt/managers/mm_utils.py`:

- line `1006`: `def embed_mm_inputs(...)`. Line `1090-1094` calls
  `multimodal_model.separate_deepstack_embeds(embedding)` to split each
  vision embedding into `(image_embed, deepstack_embed)`.
- lines `1108-1124`: when `use_deepstack` is truthy, allocates a fresh
  per-call `input_deepstack_embeds = torch.zeros(...)` with shape
  `(num_tokens, hidden_size * num_deepstack_embeddings)`, then scatters
  per-modality DeepStack tiles into it. Stored in `other_info`.
- line `1140`: `return input_embeds, other_info`.

The `other_info` dict is unpacked further upstream into `kwargs` passed
into `language_model.forward(...)`. So `input_deepstack_embeds` reaches
the LM as a kwarg whose **data pointer is fresh per request**, unless
some upstream layer has copied it into a stable buffer.

## 5. Prefill BCG capture / replay data flow

`python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`:

### 5.1 What has a stable slot

- The multimodal `input_embeds` slot is registered by
  `cuda_graph_buffer_registry.build_prefill_registry` at
  `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py:867-877`
  when `is_multimodal=True` and `register_input_embeds=True`. The
  registry allocates a stable device buffer indexed as `"input_embeds"`.
- **No slot named `input_deepstack_embeds` is ever registered.**
  (`grep -n input_deepstack cuda_graph_buffer_registry.py` returns
  nothing; the string appears only in `prefill_cuda_graph_runner.py`
  and the two model files.)

### 5.2 What the replay closure copies

`prefill_cuda_graph_runner.py:1484-1519` — the `_execute_body_capture`
closure that runs on the BCG / Full-CG replay path:

```python
def replay_layer_forward(*args, **layer_kwargs):
    # The captured body graph reads activations from the static
    # input_embeds slot. The outer model.forward (run eagerly)
    # passes the live embeddings into layer_model.forward as the
    # 4th positional arg (or input_embeds kwarg): for multimodal
    # batches these are the composed text+vision embeds, for
    # text-only batches they are get_input_embeddings()(input_ids).
    # Copy them into the slot before replay so the graph sees the
    # current request's embeddings (mirrors main's BCG closure).
    if self.buffer_registry.has_slot("input_embeds"):
        ie = layer_kwargs.get("input_embeds")
        if ie is None and ie_idx is not None and len(args) > ie_idx:
            ie = args[ie_idx]
        if ie is not None:
            self.buffer_registry.get_slot("input_embeds").slice_for(
                1, static_num_tokens
            )[: ie.shape[0]].copy_(ie)
    hs = self.backend.replay(shape_key, static_forward_batch, **kwargs)
    ...
```

Observations:

- The closure copies **only** `input_embeds` into the `input_embeds`
  slot.
- `input_deepstack_embeds` is present in `layer_kwargs` (routed there by
  `general_mm_embed_routine`) but the closure does not touch it.
- The `**kwargs` handed to `self.backend.replay(...)` is the **outer**
  `kwargs` from `_execute_body_capture`, not the `layer_kwargs` seen
  by the closure — so `input_deepstack_embeds` is not forwarded into
  `.replay()` either.
- The captured CUDA graph (for BCG) reads from whatever memory address
  it captured. Since `input_deepstack_embeds` never got a stable slot,
  the captured address is either the DeepStack warmup tensor (allocated
  as a local `torch.zeros(...)` inside
  `run_dummy_multimodal_deepstack_forward`) or — if capture did not
  see the tensor-valued branch — no capture site exists.

### 5.3 The only existing DeepStack accommodation

`prefill_cuda_graph_runner.py:662-725` —
`run_dummy_multimodal_deepstack_forward`:

- Called from the `_toggle_multi_platform_ops` context, once, at
  `capture_num_tokens[-1]` only (per PR #30868's diff — the change
  reads `cuda_graph_runner.run_dummy_multimodal_deepstack_forward(
  inner_model, cuda_graph_runner.capture_num_tokens[-1])`).
- Allocates a **local**
  `deepstack_embeds = torch.zeros((num_tokens, hidden_size * num_deepstack), ...)`,
  marks dim 0 dynamic via `torch._dynamo.maybe_mark_dynamic`, and calls
  `language_model.forward(..., input_deepstack_embeds=deepstack_embeds)`.
- Purpose (per its docstring): "Warm the tensor-valued deepstack branch
  before serving requests. … leaving this branch cold makes the first
  image request synchronously recompile the language model."
- **What it does not do:** it does not register a stable `input_deepstack_embeds`
  slot, does not persist the local tensor beyond its own scope, and
  only warms one shape (`capture_num_tokens[-1]`).

## 6. Existing tests

Registered upstream tests reachable via GitHub code search for
`is_multimodal_piecewise_cuda_graph_supported`, `deepstack cuda_graph`,
and `input_deepstack_embeds`:

- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py`
  — from PR #30872. Covers wrapper resolution and the `input_embeds`
  slot helper. Does not exercise the DeepStack path.
- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py`
  — from PR #30868. Covers the mrope helper and raw `cu_seqlens` fallback.
  Does not exercise DeepStack correctness.
- `test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py`
  — asserts the allowlist itself.
- `test/registered/unit/multimodal/test_vit_cuda_graph_runner.py`
  — ViT graph runner unit; upstream of the LM path this audit is about.

**No test currently asserts that BCG-captured, BCG-replayed
`hidden_states` for a multimodal batch equals the eager-path
`hidden_states` for the same batch when DeepStack is active.** That is
the gap the validation plan proposes to close, in a `Qwen3.5-4B`-scoped
form.

## 7. Related PRs and issues

- **PR #30872 — Enable multimodal prefill BCG for VL and audio models**
  (MERGED 2026-07-28, merge SHA `c9947b087bf9`). Adds the
  `input_embeds` static slot / copy in BCG capture and replay, orders
  decode-graph capture before prefill-graph capture, and adds
  `Qwen3_5*ForConditionalGeneration` to the multimodal BCG allowlist.
  **PR diff contains no `input_deepstack_embeds` slot or copy.**
- **PR #30868 — fix: fix vlm cuda graph shape stability** (MERGED
  2026-07-19, merge SHA `d4801be44773`). Introduces
  `run_dummy_multimodal_deepstack_forward` (single-shape warmup only)
  and a defensive eager fallback for replacement backends missing a
  capture stream. This is a **shape-stability** fix, not a
  capture-replay-slot fix; the two are easy to conflate but address
  different failure modes.
- **Issue #27212 — "Cannot serve text only checkpoints of
  Qwen3_5ForConditionalGeneration that set language_model_only"**
  (open). Adjacent but not this hypothesis; it is about text-only
  serving of a Qwen3.5 wrapper checkpoint.
- **Issue #21327 — "[Bug] IndexError in `embed_mm_inputs` when images
  and videos coexist with deepstack"** (closed). Historical DeepStack
  bug at the mm-input assembly layer; different failure mode.

No open upstream issue explicitly claims a DeepStack replay-side gap in
BCG on Qwen3.5 as of the audit SHA. That is the point of `plan.md` §7:
before opening any upstream issue, get runtime evidence one way or the
other.

## 8. Summary — what is established vs unverified

Established (source-only):

- Qwen3.5 architectures are in the BCG allowlist on current upstream.
- The language-model forward reads `input_deepstack_embeds` and
  contributes it to layers 0–2 via an in-place `add_`, gated on
  `is not None and numel() > 0`.
- `general_mm_embed_routine` produces `input_deepstack_embeds` as a
  fresh tensor per request.
- BCG capture / replay stabilises `input_embeds` via a registered slot
  and per-request copy; no such slot or copy exists for
  `input_deepstack_embeds`.
- The single DeepStack accommodation
  (`run_dummy_multimodal_deepstack_forward`, PR #30868) is a Dynamo
  warmup only, tied to `capture_num_tokens[-1]`.
- No registered test asserts DeepStack-active BCG replay correctness.

Unverified (needs runtime evidence):

- Whether the captured graph, at replay, actually reads from the
  warmup DeepStack tensor address (silently producing zero contribution
  or a stale value) or triggers a Dynamo recompile / assertion.
- Whether the `breakable` prefill backend on Qwen3.5-4B silently falls
  back to eager on the first image request (which would preserve
  correctness at the cost of BCG's perf premise).
- Whether the numeric divergence — if any — is large enough to matter
  operationally (e.g., changes the greedy token stream) or is buried
  in bf16 noise.
