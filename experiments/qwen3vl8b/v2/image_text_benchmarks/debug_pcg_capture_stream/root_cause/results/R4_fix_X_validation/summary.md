# R4 — Fix-X validation + Y prototype attempt

> Three sub-stages: **R4.A** production-shape sanity for (X), **R4.B**
> n=400 stretch for (X), **R4.C** (Y) prototype validation. R4.A + R4.B
> PASS (the (X) safety patch holds at scale and under production
> shape). R4.C is a **documented FAILURE**: the naive (Y) prototype
> bypasses `set_attention_metadata_context()` and crashes the
> torch.compile pass with a data-dependent assertion. The lesson
> shapes R5's upstream design for (Y).

## R4.A — Production-shape sanity (X)

Same E2a recipe as R3.B (image 720p c=1 n=32 warmup=30 PCG on IPC on
GPU 0), `SGLANG_DEBUG_PCG_CALL_TRACE` unset.

| | |
|---|---|
| Classification | **`OK_FALLBACK_TAKEN`** |
| AssertionError lines | 0 |
| Fallback warning lines | 1 (idempotent) |
| `[PCG_DEBUG]` lines | 0 (gate unset, as expected) |
| Successful requests | 32 / 32 |
| Mean TTFT | 106.72 ms |
| Median TTFT | 106.25 ms |
| P99 TTFT | 125.15 ms |
| Mean TPOT | 5.18 ms |

Within 1–2 ms of R3.B (median 103.05 / mean 104.81 ms). The (X) fix
works without the diagnostic gate.

## R4.B — n=400 stretch (X)

Same as R4.A but `--num-prompts 400` to match Stage 4.2's original
IMG_A_S2_ipc_pcg recipe exactly (single rep).

| | |
|---|---|
| Classification | **`OK_FALLBACK_TAKEN`** |
| Wall-clock duration | 316.29 s (~5 min) |
| Successful requests | 400 / 400 |
| Mean TTFT | 131.44 ms |
| Median TTFT | 104.62 ms |
| P90 TTFT | 129.11 ms |
| P99 TTFT | 598.34 ms (some tail) |
| Mean TPOT | 5.19 ms |
| Output token throughput | 161.87 tok/s |

Median TTFT consistent with R3.B / R4.A at ~104 ms; mean a bit
higher (131.44 ms) and a long-tail P99 (598 ms) consistent with image
preprocessing variance at higher request volume. **(X) holds at
scale.**

## R4.C — (Y) prototype validation — **FAILURE (documented)**

Patch: `../../patches/R4_fix_Y_prototype_deepstack_warmup.patch`
(fork commit `31cc8752f`, retained as instructive failure).

The prototype added two hooks:

1. `Qwen3VLForConditionalGeneration.pcg_warmup_multimodal_branch(
   input_ids, positions, forward_batch)` — synthesizes a
   `torch.zeros([num_tokens, hidden_size * num_deepstack_embeddings])`
   tensor and calls `self.model(input_ids, positions, forward_batch,
   input_embeds=embed, input_deepstack_embeds=zeros)` directly.
2. `TcPiecewiseCudaGraphBackend._run_compile_pass` — after the
   regular text-only `Compile-num-tokens` loop, runs a second
   `Compile MM num tokens` loop calling the hook if present.

### Outcome

| | |
|---|---|
| Classification | **`SERVER_ASSERTION_OTHER`** |
| Compile MM num tokens loop iterations | 1 of 58 (crashed on first) |
| `compiling_num_tokens_text_lines` | text-only loop completed (58 / 58) |

### Why it failed

The MM warmup loop's first iteration calls
`pcg_warmup_multimodal_branch(...)`, which calls `self.model(...)`
directly. That call bypasses the wrapper SGLang normally places
around `model_runner.model.forward(...)`:
`set_attention_metadata_context(self.model_runner, ...)`.

Inside the now-tracing Dynamo graph, the per-layer `self_attn.forward`
calls `get_attn_backend() → get_forward_context()`, which has an
assertion `assert _current is not None`. Because the thread-local
forward context was never pushed, the assertion fires *inside a
torch.compile-traced region*. Dynamo's reaction:

```
torch._dynamo.exc.Unsupported: Data-dependent assertion failed
    (cannot compile partial graph)
```

`fullgraph=True` is set by `install_torch_compiled`, so the compile
aborts and the server scheduler shuts down. Stack trace from
`raw/server.log`:

```
File ".../tc_piecewise_cuda_graph_backend.py:229" in _run_compile_pass
    mm_warmup(fb.input_ids, fb.positions, fb)
File ".../qwen3_vl.py:1361" in pcg_warmup_multimodal_branch
    return self.model(...)
File ".../compilation/compile.py:197" in trampoline
    return compiled_callable(*args, **kwargs)
File ".../torch/_dynamo/eval_frame.py:1034" in compile_wrapper
    raise e.with_traceback(None) from e.__cause__  # User compiler error
File ".../qwen3_vl.py:1175" in forward
    hidden_states, residual = layer(...)
File ".../qwen3.py:309" in self_attn.forward
    attn_output = self.attn(q, k, v, forward_batch, save_kv_cache=save_kv_cache)
File ".../layers/radix_attention.py:145" in forward
    return get_attn_backend().forward(...)
File ".../model_executor/forward_context.py:59" in get_forward_context
    assert _current is not None, ...
```

### What this tells us about the cleaner (Y) design

1. The multimodal warmup forward **must go through the same
   `model_runner.model.forward(...)` path** the regular text-only
   warmup uses, so `set_attention_metadata_context()` wraps the call.
2. That path routes through `general_mm_embed_routine`, which gates
   the `input_deepstack_embeds` kwarg on `contains_mm_inputs()`. So
   the cleaner (Y) needs one of:
   - A thread-local "force-multimodal-warmup" flag that
     `general_mm_embed_routine` reads. When set + `use_deepstack`, it
     synthesizes `kwargs["input_deepstack_embeds"]` even without real
     mm_inputs and then calls `language_model.forward(...)` as usual.
   - Or a `warmup_with_mm_inputs` kwarg on
     `Qwen3VLForConditionalGeneration.forward` that triggers the
     synthesize-then-route path.
   - Or a model method `pcg_warmup_forward(forward_batch, num_tokens)`
     that internally invokes `set_attention_metadata_context()` AND
     synthesizes the deepstack tensor AND calls `self.model.forward`.
3. The shape of synthesized `input_deepstack_embeds` (zeros of
   `[num_tokens, hidden_size * num_deepstack_embeddings]`) is
   correct — what failed was the forward-context plumbing, not the
   tensor synthesis.

R5's upstream draft will recommend (X) for immediate merge and
include the (Y) design sketch above as a follow-up issue that
upstream maintainers can scope and land separately.

## Summary table

| Sub-stage | Classification | Verdict |
|---|---|---|
| R4.A production sanity (X) | `OK_FALLBACK_TAKEN` | **PASS** — (X) works without diagnostic gate |
| R4.B n=400 stretch (X) | `OK_FALLBACK_TAKEN` | **PASS** — (X) holds at scale (400/400) |
| R4.C (Y) prototype | `SERVER_ASSERTION_OTHER` | **FAILURE (documented)** — naive prototype crashes; informs cleaner upstream design |

R5 next: distill R1+R2+R3+R4 into an upstream-ready draft. R5 produces
the draft only; user decides filing.
