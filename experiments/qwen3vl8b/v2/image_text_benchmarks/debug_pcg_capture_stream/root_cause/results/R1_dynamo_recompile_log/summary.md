# R1 — Dynamo recompile reason (env-vars only)

> Goal: identify which guard fires the runtime Dynamo recompile that
> leaves a piecewise CUDA-graph submodule without a capture stream and
> trips the assertion at
> `cuda_piecewise_backend.py:171`.
>
> Outcome: **root cause class identified** — multimodal control-flow
> recompile on `input_deepstack_embeds is None` (qwen3_vl.py:1129).
> SGLang's piecewise CUDA-graph warmup captures graphs while
> `input_deepstack_embeds = None` (text-only synthetic warmup at server
> startup), so the captured callables don't cover the multimodal branch.
> When the first image request arrives at runtime, Dynamo recompiles
> `Qwen3LLMModel.forward`; the new compile lands piecewise submodules
> that have no associated capture stream → assertion fires.

## 1. Run identity

| | |
|---|---|
| Recipe | image 720p, 1 image, c=1, n=32, warmup=30, output_len=128 |
| Server flags | `--enforce-piecewise-cuda-graph`, `--attention-backend flashinfer`, IPC on |
| Env vars | `TORCH_LOGS=recompiles_verbose,dynamic,guards,graph_breaks`, `TORCHDYNAMO_VERBOSE=1`, `CUDA_VISIBLE_DEVICES=0`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1` |
| GPU | 0 (H200), 0 MiB before/after |
| sglang | 0.0.0.dev1+gda802ddca (`/sgl-workspace/sglang`) |
| model | `Qwen/Qwen3-VL-8B-Instruct` snapshot `0c351dd` |
| classification | `PCG_CAPTURE_STREAM_ASSERT` (matches prior debug's E2a) |
| raw log | `raw/server.log` (44092 lines, ~16 MB; NOT committed) |
| trimmed | `recompile_excerpt.log` (133 lines; committed) |

## 2. Recompile cascade

`Qwen3LLMModel.forward` (`qwen3_vl.py:1136`) recompiles **four times**
between server start and the assertion:

| Frame | Time | Trigger | Interpretation |
|---|---|---|---|
| `[0/1]` | 21:47:23 | `not ___dict_contains(...)` guard on `G['__import_sgl_kernel_dot_elementwise'].is_arch_support_pdl.__closure__[1].cell_contents` (`sgl_kernel/utils.py:50`) | Benign — `is_arch_support_pdl` memoizes its result; the first call mutates the cached dict, invalidating Dynamo's guard. Recompile happens once at first execution. |
| `[0/2]` | 21:47:29 | `tensor 'positions' size mismatch at index 1. expected 8192, actual 7680` | Iteration through SGLang's startup "Compiling num tokens" loop — token-count axis changes. Goes through many sizes (1024, 7680, 8192, …, 256, 16, …). Each size after the first is recompiled into the same frame. |
| **`[0/3]`** | **21:47:49** | **`input_deepstack_embeds is None` guard failed** at `qwen3_vl.py:1129` (`get_deepstack_embeds`) **+** `positions` size mismatch | **First image prompt arrives during bench warmup.** Multimodal embeds are now present, so the `is None` specialization is invalid → recompile. **This is the multimodal control-flow recompile that breaks PCG**. |
| `[0/4]` | 21:48:04 | `input_embeds` size mismatch 80 → 1024 + `input_deepstack_embeds is None` still fails + `positions` size mismatch | Different image request shape — another multimodal recompile on top of [0/3]. Same root cause, different shape signature. |

After [0/4], ~9 s of mixed prefill batches succeed
(`cuda graph: True` on lines 43954–43988). Then the assertion fires at
21:48:13 in `submod_0` (the layer-0 piecewise submodule of the freshly
recompiled forward graph). Stack:

```
File "/sgl-workspace/sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py", line 171, in __call__
    stream is not None
AssertionError: PCG capture stream is not set, please check if runtime recompilation happened
File "<eval_with_key>.669", line 298, in forward
    submod_0 = self.submod_0(l_input_embeds_, s47, ...,
                                              l_self_modules_layers_modules_0_layer_communicator_input_layernorm_parameters_weight_, ...)
File "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_vl.py", line 1136, in forward
```

## 3. Why this is the root cause (not just a symptom)

- **Text-only PCG works** ([prior debug E1](../../E1_text_autobench_PCG_control_summary.md)
  was `OK`; in this R1 trace the warmup `Compiling num tokens` loop
  iterates 58+ token-count graphs successfully under `cuda graph: True`).
  → The PCG infrastructure itself is sound when the model's Dynamo
  specialization is stable.
- **Image+text PCG fails**, and Dynamo's own log shows the failing
  guard is `input_deepstack_embeds is None`. → The instability is in the
  multimodal control flow, not in shape coverage.
- The startup-time warmup (`Compiling num tokens`) runs with synthetic
  text-only inputs (no `mm_inputs`), so the captured PCG callables all
  specialize on `input_deepstack_embeds = None`. The first image
  request must therefore recompile.
- The PCG capture step only attaches a capture stream to the
  *originally captured* callable. Dynamo's recompiled callable is a
  *different* `submod_N` instance, so it has no capture stream — the
  assertion at `cuda_piecewise_backend.py:171` is exactly the
  defensive guard for this case.

## 4. What this rules out (vs prior debug hypotheses)

- **Not** an unbounded token-count case ("Dynamo may silently recompile
  … whose token count exceeds the captured range" — comment at
  `cuda_piecewise_backend.py:156-161`). Failing token counts in [0/3] /
  [0/4] are 80 and 1024 — well inside the captured 1..8192 range.
- **Not** an IPC-specific path. (Prior E3 already showed IPC is not
  required; this R1 reconfirms — the recompile is in `qwen3_vl.forward`,
  which doesn't touch IPC.)
- **Not** a shape-axis bug per se. Shape mismatches DO appear in [0/2],
  [0/3], [0/4]'s guard-failure lists, but the *defining* axis is
  `input_deepstack_embeds is None` — that's the one that flips
  between warmup and runtime.

## 5. Implications for the fix shapes

| Shape | Verdict |
|---|---|
| **(X) defensive CUDA fallback** at `cuda_piecewise_backend.py:163-169` (mirror HIP path) | Works as a safety net but does not actually let image+text use PCG; degrades to eager and silently loses any PCG benefit. Acceptable as a "must not crash" patch but not a measurement-enabling fix. |
| **(Y) broaden warmup capture** — synthesize a deepstack-embeds-present forward during the `Compiling num tokens` warmup so the captured PCG callables cover both `input_deepstack_embeds is None` and `input_deepstack_embeds = tensor` branches | **Direct fix.** Removes the recompile trigger entirely. Captures the multimodal path during warmup. Allows the image+text PCG benefit (Issue #4 Q2) to actually be measurable. Right shape if it's feasible to construct a dummy deepstack-embeds tensor at warmup time. |
| **(Z) per-model PCG opt-in** via `is_multimodal_piecewise_cuda_graph_supported` | Only useful as a positive declaration ("this model supports image+text PCG") *after* (Y) is in place. Without (Y), the right declaration is "not supported" (matching the current upstream auto-disable). |

**R2 + R3** should validate (Y) is feasible: confirm that the SGLang
warmup driver can be augmented to feed a synthesized multimodal batch
through `forward()` once, alongside the existing num-tokens sweep.

## 6. Open questions for R2

R2 instruments `cuda_piecewise_backend.__call__` per-call to confirm:

1. The `submod_0` instance that asserts is **a different Python id**
   than the warmup-captured `submod_0`. (Expected: yes — that's the
   sense of "recompile invalidates the capture".)
2. The recompiled forward never re-enters the `compile_piecewise_graph`
   path that would attach a capture stream. (Expected: yes — capture
   only happens during the explicit warmup driver, not at inference
   time.)
3. There is no observable difference in the failing call's input
   tensors that would invalidate (Y) — e.g., the deepstack-embeds path
   doesn't depend on host-side python state that the warmup can't
   reproduce.

R2 will surface those facts before R3 commits to (Y).
