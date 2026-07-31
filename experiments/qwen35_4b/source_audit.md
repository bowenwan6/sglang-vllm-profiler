# Source Audit — Qwen3.5-4B multimodal prefill BCG DeepStack

> **Scope.** Source-level reading of current upstream SGLang `main` and
> the HuggingFace `Qwen/Qwen3.5-4B` model card. Records *what the code
> does* — not what happens at runtime. Nothing in this file constitutes
> evidence of a runtime defect. Runtime evidence is `validation_plan.md`'s
> job.

## 1. Provenance of this audit

- **Upstream SGLang `main` HEAD read (rebaselined 2026-07-31):**
  `89f4a80c1f5e71c1c960df120f1e03b43dfd3c1d` (subject: `Support
  fastsafetensors no-GDS loading and page-cache release (#31859)`,
  verified via `GET /repos/sgl-project/sglang/commits/main`). This SHA
  supersedes the earlier `5f9b0db1…` audit anchor; every line number
  and citation below refers to this SHA.
- **Files fetched via `raw.githubusercontent.com/sgl-project/sglang/89f4a80c…/…`**
  and cached under `<scratchpad>/sglang_snapshot/*` during the audit;
  **not committed** to this repository (per artifact rules). A
  companion isolated `git clone` of upstream `main` at the same SHA
  lives at `<scratchpad>/sglang_checkout/sglang/` and is the checkout
  the Step 2 runner sources via `PYTHONPATH`.
- **HF `Qwen/Qwen3.5-4B` metadata:** `sha =
  851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, `pipeline_tag =
  image-text-to-text`, `config.architectures =
  ["Qwen3_5ForConditionalGeneration"]`, `config.model_type =
  "qwen3_5"`, `gated = false` (verified via `GET
  /api/models/Qwen/Qwen3.5-4B`).

## 2. Qwen3.5 architecture registration — BCG vs PCG lists are DISTINCT

**This is the most important correction to a prior draft that conflated
the two lists.** SGLang keeps them separate and Qwen3.5 is on only one.

- `python/sglang/srt/configs/model_config.py`:
  - Lines `1836-1841` — `multimodal_piecewise_cuda_graph_supported_model_archs`
    (the **PCG / `tc_piecewise` / torch.compile-based** prefill allowlist):

    ```python
    multimodal_piecewise_cuda_graph_supported_model_archs = [
        "Cohere2VisionForConditionalGeneration",
        "KimiK25ForConditionalGeneration",
        "MiniMaxM3SparseForCausalLM",
        "MiniMaxM3SparseForConditionalGeneration",
    ]
    ```

    **Qwen3.5 is NOT on this list.** So `--enforce-piecewise-cuda-graph`
    is not the correct handle for Qwen3.5 multimodal graph capture; it
    was on this list for the *historical* Qwen3-VL sub-track only.
  - Lines `1843-1848` — `multimodal_breakable_cuda_graph_supported_model_archs`
    (the **BCG / breakable-CUDA-graph** allowlist):

    ```python
    # Multimodal archs whose LM prefill is validated under breakable CUDA graph;
    # embed-carrying batches are rejected at replay (can_run_graph) and run eager.
    multimodal_breakable_cuda_graph_supported_model_archs = [
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ]
    ```

    **Qwen3.5 is on this list.** Enabling BCG for it does not require
    any `--enforce-piecewise-cuda-graph` flag; it is inherited from
    the default breakable prefill backend.
  - Line `1908` — `is_multimodal_piecewise_cuda_graph_supported`
    checks membership in the PCG list.
  - Line `1916` — `is_multimodal_breakable_cuda_graph_supported`
    checks membership in the BCG list.
  - Lines `473-477` — `ModelConfig` computes both booleans and stores
    them on itself: `self.is_multimodal_piecewise_cuda_graph_supported`
    and `self.is_multimodal_breakable_cuda_graph_supported`. They are
    independently consulted downstream — one does not gate the other.
- `python/sglang/srt/models/qwen3_5.py`:
  - line `1771`: `class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`
    with `language_model_cls=Qwen3_5ForCausalLM`.
  - line `1928`: `class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration)`
    with `language_model_cls=Qwen3_5MoeForCausalLM`.
  - line `2319`: `EntryClass = [Qwen3_5MoeForConditionalGeneration,
    Qwen3_5ForConditionalGeneration]` — how the loader picks them up.

**Implication (revised).** `Qwen/Qwen3.5-4B` reports
`architectures=["Qwen3_5ForConditionalGeneration"]`, so on current
upstream `main` with `--enable-multimodal` (the multimodal default):

- BCG (`is_multimodal_breakable_cuda_graph_supported`) returns
  **True**. This is the code path that must be exercised in the
  validation plan.
- PCG (`is_multimodal_piecewise_cuda_graph_supported`) returns
  **False**. Runtime experiments must NOT set
  `--enforce-piecewise-cuda-graph` as a "BCG control": the flag
  either has no effect on Qwen3.5 or forces a code path outside the
  BCG allowlist — either way it does not sit "in the same family" as
  BCG and should not be relied on for comparison. (This corrects an
  earlier `hypothesis.md` and `validation_plan.md` design that used
  `--enforce-piecewise-cuda-graph` as `C2 = bcg_enforce`.)

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

- The DeepStack contribution is added only for `layer_idx < 3` (first
  three decoder layers).
- The contribution is an in-place `add_` on the layer-body output.
  When it is missing, layers 0–2 emit a numerically different (but
  not crash-inducing) `hidden_states` value that then flows through
  layers 3 … final via the residual stream; downstream layers see
  changed inputs and can amplify or attenuate the discrepancy.
  **The observable effect at the final layer is therefore not
  restricted to layers 0–2.** The validation plan must not assume the
  divergence stays local to those layers.
- The gate is a **control-flow guard** on `input_deepstack_embeds is
  None` (via `is not None and numel() > 0`). Under TC piecewise (which
  uses torch.compile), the guard triggers a Dynamo recompile; under
  BCG (raw CUDA graph capture), it changes which kernels get recorded
  into the captured graph and there is no Dynamo involvement at all.
  The two failure modes are not the same.

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
- lines `1108-1124`: when `use_deepstack` is truthy, allocates a
  per-call `input_deepstack_embeds = torch.zeros(...)` with shape
  `(num_tokens, hidden_size * num_deepstack_embeddings)`, then scatters
  per-modality DeepStack tiles into it. Stored in `other_info`.
- line `1140`: `return input_embeds, other_info`.
- lines `1247-1373`: `general_mm_embed_routine` unpacks
  `other_info["input_deepstack_embeds"]` into `kwargs`, then calls
  `language_model(..., input_embeds=..., **kwargs)`.
- lines `1361-1363` — **`input_embeds` is copy_'d into a stable slot**
  when `forward_batch.input_embeds is not None`:

  ```python
  if forward_batch.input_embeds is not None:
      forward_batch.input_embeds.copy_(input_embeds)
      input_embeds = forward_batch.input_embeds
  ```

  This is how the composed text+vision embedding ends up in the BCG
  static slot at replay time. **There is no analogous copy for
  `input_deepstack_embeds` anywhere in the routine.**

### 4.1 Pointer wording — corrected

Precise phrasing to use in this and downstream documents:

- `general_mm_embed_routine` allocates `input_deepstack_embeds` via a
  fresh `torch.zeros(...)` call on every invocation. That produces a
  **fresh `torch.Tensor` object** per call — a fresh Python identity.
- The **`.data_ptr()` of that fresh tensor is not stable by contract**:
  no code path copies it into a registered buffer or pins it.
- The observed `.data_ptr()` value across calls is not guaranteed to
  differ, however, because PyTorch's CUDA caching allocator may
  reuse the same underlying address for repeated same-size,
  same-lifetime allocations. So a runtime observation like "the
  pointer looks stable" is not evidence that the tensor is stable —
  it is evidence that the allocator handed back the same slab; the
  BCG capture bridge still has no contract on it.
- Any runtime observation of pointer equality/inequality is diagnostic
  only; it is not a substitute for observing the captured-graph
  DeepStack read/write path itself.

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
- `input_deepstack_embeds` is present in `layer_kwargs` (routed there
  by `general_mm_embed_routine`) but the closure does not touch it.
- The `**kwargs` handed to `self.backend.replay(...)` is the **outer**
  `kwargs` from `_execute_body_capture`, not the `layer_kwargs` seen
  by the closure — so `input_deepstack_embeds` is not forwarded into
  `.replay()` either.
- The BCG backend's `replay(...)` (`runner_backend/breakable_cuda_graph_backend.py:241-248`)
  simply calls `self._graphs[shape_key].replay()`; it does not consume
  `**kwargs` at all. The captured CUDA graph reads and writes only the
  addresses recorded at capture time.

### 5.3 The only existing DeepStack accommodation — scoped to TC PIECEWISE, not BCG

`prefill_cuda_graph_runner.py:662-725` —
`run_dummy_multimodal_deepstack_forward`:

```python
def run_dummy_multimodal_deepstack_forward(
    self, language_model: torch.nn.Module, num_tokens: int
) -> bool:
    """Warm the tensor-valued deepstack branch before serving requests.
    The regular PCG dummy is text-only. Qwen3-VL only provides
    ``input_deepstack_embeds`` after visual encoding, so leaving this
    branch cold makes the first image request synchronously recompile the
    language model. The model/signature checks keep this a no-op for
    non-deepstack architectures."""
    ...
    deepstack_embeds = torch.zeros(
        (num_tokens, hidden_size * num_deepstack),
        dtype=self.model_runner.dtype,
        device=self.device,
    )
    torch._dynamo.maybe_mark_dynamic(deepstack_embeds, 0)
    ...
    language_model.forward(
        fb.input_ids, ..., input_deepstack_embeds=deepstack_embeds,
    )
```

**Callers of this function** (verified by grep across upstream `main`
at the frozen SHA):

- `python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py:214-216`
  — inside `_run_compile_pass`, called **only after** `install_compile`
  (i.e., after `torch.compile` has been applied to the language
  model), and after the per-shape Dynamo warmup loop. This is the
  **TC piecewise** (torch.compile-based) prefill backend, not BCG.

**No other caller exists** anywhere in the SGLang tree at the frozen
SHA. In particular:

- `runner_backend/breakable_cuda_graph_backend.py` never calls
  `run_dummy_multimodal_deepstack_forward` and never allocates any
  DeepStack tensor for warmup.
- The BCG prefill capture path (`capture_one_shape` →
  `capture_prepare` → `_run_forward`) drives the LM via
  `layer_model.forward(input_ids, positions, forward_batch,
  forward_batch.input_embeds)` — four positional args, no
  `input_deepstack_embeds` kwarg. So the CUDA graph captured for BCG
  is captured under the `input_deepstack_embeds is None` branch, and
  the DeepStack `add_` kernels are simply not recorded into the graph.

**Implication.** The single DeepStack accommodation upstream is a
**Dynamo shape-stability warmup for TC piecewise** (PR #30868). It
does not touch BCG, does not register any slot, and its local
`torch.zeros(...)` allocation goes out of scope when the function
returns. The BCG bridge for `input_deepstack_embeds` is not just
"undersized" — it is **absent by construction** on the BCG code path.

### 5.4 Separation of PR #30868 vs PR #30872

The two PRs are frequently conflated. They do not overlap:

- **PR #30868 (merged 2026-07-19)** — introduces
  `run_dummy_multimodal_deepstack_forward` and a defensive eager
  fallback in the tc_piecewise backend for the historical
  "PCG capture stream is not set" assertion (the Qwen3-VL R1 failure
  mode). This is a **PCG / Dynamo warmup** change. It does not touch
  BCG, and it does not add any capture / replay slot.
- **PR #30872 (merged 2026-07-28)** — enables the multimodal
  prefill BCG allowlist (adds Qwen3.5 to
  `multimodal_breakable_cuda_graph_supported_model_archs`), registers
  the `input_embeds` static slot in the buffer registry, and adds the
  `replay_layer_forward` closure that copies live `input_embeds` into
  that slot before `.replay()`. This is a **BCG replay bridge**
  change. It handles `input_embeds` and nothing else.

The DeepStack replay path is therefore neither PR's scope. This
audit hypothesises there is a gap between them; §7 must prove or
disprove it at run time.

## 6. `can_run_graph` and the "embed-carrying batches are rejected at replay" comment

`prefill_cuda_graph_runner.py:1004-1061` — `can_run_graph`. Relevant
early rejections:

```python
if forward_batch.input_embeds is not None:
    return False
if forward_batch.replace_embeds is not None:
    return False
```

The comment on `multimodal_breakable_cuda_graph_supported_model_archs`
(§2) states "embed-carrying batches are rejected at replay
(can_run_graph) and run eager." Reconciling this with the code:

- `forward_batch.input_embeds` is **set** only when the request
  arrives with an API-level `input_embeds` parameter (see
  `managers/schedule_batch.py:2233-2401` — `input_embeds` is
  populated from `req.input_embeds`, i.e., a per-request pre-computed
  embedding provided by the caller).
- For a normal multimodal image request, the client sends
  `input_ids` + image data. The scheduler sets
  `forward_batch.mm_inputs` (via `batch.multimodal_inputs`) but leaves
  `forward_batch.input_embeds = None`. The composed text+vision
  embedding is built later, inside `general_mm_embed_routine`, when
  `model.forward` runs.
- Therefore the `can_run_graph` gate `input_embeds is not None →
  False` fires for **API-`input_embeds`** requests, not for normal
  image requests. Normal image requests are **not** filtered out here
  by construction.

This is a critical point. The prior draft implicitly assumed the
comment covered normal image requests; it does not. The runtime
plan must verify empirically which path image requests actually
take.

## 7. Existing tests

Registered upstream tests reachable via GitHub code search for
`is_multimodal_piecewise_cuda_graph_supported`, `deepstack cuda_graph`,
and `input_deepstack_embeds`:

- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py`
  — from PR #30872. Covers wrapper resolution and the `input_embeds`
  slot helper. Does not exercise the DeepStack path.
- `test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py`
  — from PR #30868. Covers the mrope helper and raw `cu_seqlens`
  fallback. Does not exercise DeepStack correctness.
- `test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py`
  — asserts the allowlist itself.
- `test/registered/unit/multimodal/test_vit_cuda_graph_runner.py`
  — ViT graph runner unit; upstream of the LM path this audit is about.

**No test currently asserts that BCG-captured, BCG-replayed
`hidden_states` for a multimodal batch equals the eager-path
`hidden_states` for the same batch when DeepStack is active.** That is
the gap the validation plan proposes to close, in a `Qwen3.5-4B`-scoped
form.

## 8. Related PRs and issues

- **PR #30872 — Enable multimodal prefill BCG for VL and audio models**
  (MERGED 2026-07-28, merge SHA `c9947b087bf9`). Adds the
  `input_embeds` static slot / copy in BCG capture and replay, orders
  decode-graph capture before prefill-graph capture, and adds
  `Qwen3_5*ForConditionalGeneration` to
  `multimodal_breakable_cuda_graph_supported_model_archs`.
  **PR diff contains no `input_deepstack_embeds` slot or copy.**
- **PR #30868 — fix: fix vlm cuda graph shape stability** (MERGED
  2026-07-19, merge SHA `d4801be44773`). Introduces
  `run_dummy_multimodal_deepstack_forward` (single-shape warmup only,
  called from `tc_piecewise_cuda_graph_backend._run_compile_pass`) and
  a defensive eager fallback for replacement backends missing a
  capture stream. This is a **TC-piecewise / Dynamo shape-stability**
  fix, not a BCG capture-replay-slot fix.
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

## 9. Summary — what is established vs unverified

Established (source-only):

- Qwen3.5 architectures are on the **BCG (breakable)** allowlist, not
  the PCG allowlist.
- The language-model forward reads `input_deepstack_embeds` and
  contributes it to layers 0–2 via an in-place `add_`, gated on
  `is not None and numel() > 0`. Downstream layers see the
  contribution through the residual stream.
- `general_mm_embed_routine` allocates `input_deepstack_embeds` as a
  fresh tensor object per request; the underlying pointer is not
  guaranteed stable by contract but the CUDA allocator may return the
  same address.
- `general_mm_embed_routine` copies `input_embeds` into the
  registered static slot when one exists; there is no such copy for
  `input_deepstack_embeds`.
- BCG capture / replay stabilises `input_embeds` via a registered slot
  and per-request copy; no such slot or copy exists for
  `input_deepstack_embeds`.
- The single DeepStack accommodation
  (`run_dummy_multimodal_deepstack_forward`, PR #30868) is called
  **only** from `tc_piecewise_cuda_graph_backend`. BCG has no
  DeepStack warmup at all, and its capture forwards the LM without
  the `input_deepstack_embeds` kwarg.
- `can_run_graph`'s `input_embeds is not None` filter targets
  API-level `input_embeds` requests, not multimodal image requests.
- No registered test asserts DeepStack-active BCG replay correctness.

Unverified (needs runtime evidence):

- Whether image requests on Qwen3.5-4B under BCG in fact enter the
  BCG execute path, or whether some other runtime filter (that this
  audit missed) routes them to eager.
- If BCG replay does run, whether the captured graph silently omits
  the DeepStack `add_` (producing a divergence that looks like a
  zero-DeepStack ablation), or whether some other code path pins
  DeepStack.
- Whether the numeric divergence, if any, changes the greedy token
  stream or is buried in bf16 noise on the chosen prompt.
