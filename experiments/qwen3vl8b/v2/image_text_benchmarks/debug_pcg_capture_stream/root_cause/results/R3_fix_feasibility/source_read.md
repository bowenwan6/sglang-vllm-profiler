# R3.A — Source read of the warmup + multimodal forward paths

> Purpose: characterise whether fix shape (X) defensive CUDA fallback or
> fix shape (Y) broaden warmup capture is the right minimal change,
> grounded in the actual upstream code instead of guesswork.

## 1. Warmup driver

`/data/sglang-fork/python/sglang/srt/model_executor/runner_backend/tc_piecewise_cuda_graph_backend.py`

```
def _run_compile_pass(self):
    # `enable_tc_piecewise_cuda_graph()` activates the PCG path on
    # the model.
    with enable_tc_piecewise_cuda_graph():
        ...
        cuda_graph_runner._run_dummy_forward(num_tokens=capture_num_tokens[0])
        ...
        self.install_compile(language_model.model, ...)
        with enable_torch_compile_warmup():
            # Iterates every captured shape, running one dummy forward
            # per shape. This is the 'Compiling num tokens' loop seen in
            # server logs.
            for num_tokens in reversed(capture_num_tokens):
                cuda_graph_runner._run_dummy_forward(num_tokens=num_tokens)
```

Followed by:

```
@contextmanager
def capture_session(self, stream):
    self._capture_stream = stream
    with self.replay_session():
        # THIS is the only place set_pcg_capture_stream is ever called.
        with set_pcg_capture_stream(stream):
            yield
```

Key fact: `set_pcg_capture_stream(stream)` is only set inside
`capture_session()`. After capture finishes, the stream goes back to
None for the rest of the process lifetime. The defensive assertion in
`cuda_piecewise_backend.py:171` checks `get_pcg_capture_stream()`; any
`CUDAPiecewiseBackend.__call__` invoked outside `capture_session` sees
`None`.

## 2. Dummy ForwardBatch construction

`prefill_cuda_graph_runner.py:530 capture_prepare`

```
def capture_prepare(self, num_tokens: int):
    ...
    forward_batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=1,
        input_ids=_slot("input_ids"),
        input_embeds=(
            _slot("input_embeds") if registry.has_slot("input_embeds") else None
        ),
        ...
        # NB: NO mm_inputs.
        ...
    )
    return forward_batch, attn_backend
```

`mm_inputs` is never set on the dummy batch. `ForwardBatch.contains_mm_inputs()`
relies on the `mm_inputs` field; with it left at default, the predicate
returns False for every warmup pass.

## 3. Multimodal embed routine

`/data/sglang-fork/python/sglang/srt/managers/mm_utils.py:1023 general_mm_embed_routine`

```
def general_mm_embed_routine(
    input_ids, forward_batch, language_model,
    multimodal_model=None, ..., use_deepstack={}, **kwargs):
    ...
    if (
        not forward_batch.forward_mode.is_decode()
        and not forward_batch.forward_mode.is_target_verify()
        and forward_batch.contains_mm_inputs()   # <-- gate
    ):
        ...
        if use_deepstack:
            kwargs["input_deepstack_embeds"] = other_info["input_deepstack_embeds"]
    ...
    return language_model.forward(
        input_ids, positions, forward_batch, **kwargs,
    )
```

`input_deepstack_embeds` only enters `kwargs` when the multimodal gate
fires. The gate requires `forward_batch.contains_mm_inputs()` — i.e.
real mm_inputs on the batch. Warmup never has them; therefore
`Qwen3LLMModel.forward` always receives `input_deepstack_embeds=None`
during warmup. Dynamo specializes on that None, captures every
piecewise CUDA graph for the None branch, and never sees the
non-None branch until live image traffic arrives.

## 4. Model forward shape signature

`qwen3_vl.py:1136 Qwen3LLMModel.forward`

```
def forward(
    self,
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    forward_batch: ForwardBatch,
    input_embeds: torch.Tensor = None,
    pp_proxy_tensors: Optional[PPProxyTensors] = None,
    input_deepstack_embeds: Optional[torch.Tensor] = None,   # the divergent axis
):
    ...
    for layer_idx, layer in enumerate(...):
        deepstack_embeds = self.get_deepstack_embeds(
            layer_idx - 1, input_deepstack_embeds,   # qwen3_vl.py:1129 — the Dynamo guard
        )
        hidden_states, residual = layer(
            positions, hidden_states, forward_batch, residual,
            post_residual_addition=deepstack_embeds,
        )
```

`get_deepstack_embeds` does `if input_deepstack_embeds is None: return None`.
Dynamo captures that `is None` check as a guard. R1 traced it exactly:
the [0/3] recompile fail-reason is `input_deepstack_embeds is None`
at `qwen3_vl.py:1129`.

## 5. Fix-shape feasibility against this source

### (X) Defensive CUDA fallback
- Change site: `cuda_piecewise_backend.py:162-172`.
- Patch size: ~1 line — drop the `_is_hip and` guard so the fallback
  applies to CUDA too.
- Reasoning: at the failing call site `entry.runnable` is bound to
  `self.compiled_graph_for_general_shape` (the inductor-compiled
  general-shape function), which is callable in eager. Falling back
  to it instead of asserting preserves correctness; loses cudagraph
  speedup only for the recompiled frame.
- Upstream cost: trivial. Matches existing HIP precedent.
- Verdict: **safe minimum**. Does not unblock the image+text PCG
  benefit measurement (Issue #4 Q2), but ensures no hard crash.

### (Y) Broaden warmup capture
- Change sites: at minimum `capture_prepare` (insert a synthetic
  mm-inputs branch) and/or `general_mm_embed_routine` (allow the
  multimodal branch when a warmup flag is set). More invasive: a
  model-specific warmup hook on `Qwen3VLForConditionalGeneration`.
- The synthetic deepstack-embeds tensor needs to match a real one:
  shape `[num_tokens, hidden_size * num_deepstack_embeddings]`,
  dtype `bfloat16`, device `cuda:0`. From the model config:
  `hidden_size=4096`, `num_deepstack_embeddings = len(deepstack_visual_indexes)`.
  A `torch.zeros(num_tokens, 4096 * num_deepstack_embeddings, ...)`
  warmup would exercise the multimodal Dynamo frame without
  needing real image features.
- Risk: a synthetic forward path during warmup must not corrupt the
  captured FX state for the *text* compile frame. The two frames are
  Dynamo-distinct, so this should be safe in principle — but needs to
  be validated experimentally before we trust it.
- Upstream cost: non-trivial. The change is model-specific (deepstack
  is a Qwen3-VL concept) or warmup-hook-generic.
- Verdict: **the right structural fix** for image+text PCG benefit,
  but a bigger surface to land. Should follow (X), not replace it.

### (Z) Per-model PCG opt-in
- Wire `Qwen3VLForConditionalGeneration` into
  `ModelConfig.is_multimodal_piecewise_cuda_graph_supported`
  (`server_args.py:3145-3146`). This already exists as the right
  selective-enablement hook.
- Only meaningful as a positive declaration after (Y) is in place.
  Without (Y), the right value is False — matching the current
  upstream auto-disable.

## 6. R3 next step

R3.B: apply (X) as a minimal one-line patch and re-run the E2a recipe.
Expected: assertion does not fire; image+text run completes (without
per-shape cudagraph acceleration on the recompiled frame). Confirms our
bug-class identification and gives a measurable upstream-ready safety
patch.

R4 commits the (X) patch (after explicit user approval). R5 drafts the
upstream issue with the R1+R2 evidence + the proposed (X) patch
attached, and recommends (Y) as a follow-up.

If R3.B does *not* eliminate the assertion, the bug class is wider than
just the multimodal recompile path and we re-enter R3 with a fresh
hypothesis.
