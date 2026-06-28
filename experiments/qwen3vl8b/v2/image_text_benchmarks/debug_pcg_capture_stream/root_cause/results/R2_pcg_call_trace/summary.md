# R2 — Source-level cuda_piecewise call-trace

> Goal: confirm at the call-site that R1's control-flow recompile
> hypothesis is what produces the missing capture stream. Specifically:
> the `CUDAPiecewiseBackend` instance that asserts is a *new* Python
> object — created by Dynamo's runtime recompile — and is never
> reached by `PiecewiseCudaGraphRunner.capture()`.

## 1. Run identity

| | |
|---|---|
| Recipe | image 720p, 1 image, c=1, n=32, warmup=30, output_len=128 |
| Env | `SGLANG_DEBUG_PCG_CALL_TRACE=1`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, `CUDA_VISIBLE_DEVICES=0` (TORCH_LOGS dropped — already captured in R1) |
| sglang | `/data/sglang-fork` branch `fix/pcg-vlm-deepstack-warmup` HEAD `2167b5f4d` (off upstream `da802ddca`); selected via `PYTHONPATH=/data/sglang-fork/python` |
| GPU | 0 (H200), 0 MiB before/after |
| classification | `PCG_CAPTURE_STREAM_ASSERT` (same as R1) |
| `[PCG_DEBUG]` lines emitted | 17 542 |
| raw log | `raw/server.log` (NOT committed) |
| trimmed | `pcg_call_trace_excerpt.log` (82 lines, committed) |
| patch reference | `../../patches/R2_piecewise_call_logging.patch` |

## 2. Smoking-gun line

```
[PCG_DEBUG] ASSERTION ABOUT TO FIRE id=0x702f42eba060 layer_idx=0
            runtime_shape=1024 entry.compiled=False
            entry.num_finished_warmup=1
```

The asserting `CUDAPiecewiseBackend` is **`id=0x702f42eba060`**, layer 0
of a Dynamo compile frame whose `sym_shape_indices=[1, 4, 9, 10]`.

That signature does **not** match any instance from the startup PCG
warmup phase, where layer 0 had `sym_shape_indices=[1, 8]`. Different
sym-shape indices ⇒ different fx graph ⇒ different Dynamo frame ⇒
**different Python object**. Confirmed by id and by the four-symbol
shape signature (`[1, 4, 9, 10]`) which doesn't appear at all during
the in-warmup phase.

## 3. Full history of `id=0x702f42eba060`

(line numbers refer to the full 17 542-line raw trace under `raw/server.log`.)

| # | line | event | runtime_shape | num_finished_warmup | observation |
|---|---|---|---|---|---|
| 1 | 17169 | `call enter`, `first_run_finished=False` | n/a | n/a | This is Dynamo's initial trace of the new compile frame — not a user call. |
| 2 | 17206 | `call enter`, `first_run_finished=True` | 8192 | 0 → 1 | First real call after recompile; goes through "first warmup pass" branch — does NOT capture. |
| 3 | 17317 | `call enter` | 1024 | 0 → 1 | First call at this shape; "first warmup pass" branch again. |
| 4 | 17428 | `call enter` | 4 | 0 → 1 | First call at decode shape; "first warmup pass" branch again. |
| 5 | **17539** | `call enter` | 1024 | 1 | **Second call at shape 1024.** `num_finished_warmup ≥ 1` ⇒ skips the warmup-pass branch ⇒ goes to capture path. |
| 5a | 17541 | `about to capture; stream=None` | 1024 | 1 | `get_pcg_capture_stream()` returns `None` — the capture-phase thread-local stream is unset because we're not inside `PiecewiseCudaGraphRunner.capture()`. |
| 5b | 17542 | **`ASSERTION ABOUT TO FIRE`** | 1024 | 1 | The assertion at `cuda_piecewise_backend.py:171` fires. |

## 4. What this confirms

1. **The asserting instance is brand new.** It was not present during
   the startup PCG capture phase. (Distinct Python id; distinct
   sym_shape_indices.) ⇒ R1's "Dynamo recompile creates new
   CUDAPiecewiseBackend instances" hypothesis is correct.
2. **The new instance never gets a capture stream.** The
   `set_pcg_capture_stream()` call only happens inside the explicit
   `PiecewiseCudaGraphRunner.capture()` phase at server startup. By the
   time the recompiled instance reaches `__call__` for the second time
   at any given shape, that phase is long over and
   `get_pcg_capture_stream()` returns `None`. The assertion is
   structurally unreachable from `__call__`'s perspective — it can only
   ever succeed for the original startup-frame instances.
3. **The capture path on the new instance has a trivial soft warmup**
   (`num_finished_warmup < 1` branch returns once without capturing),
   which is meant to absorb the very first call. But by the second
   call the code commits to capture and immediately requires a stream.
   For a recompiled instance no such stream is ever set. So the very
   second call at any single shape is fatal.
4. **`_is_hip` is `False` on this H200.** The existing HIP fallback at
   `cuda_piecewise_backend.py:163-169` doesn't help; CUDA always
   asserts.

## 5. Inputs and outputs for the fix decision

R1 identified the control-flow trigger (`input_deepstack_embeds is
None` guard failing on first image). R2 has now also established that
the *mechanism* is structural — a new CUDAPiecewiseBackend instance per
Dynamo frame with no path to acquire a capture stream at inference
time.

This makes the fix-shape comparison concrete:

- **(X) defensive fallback** at `cuda_piecewise_backend.py:163-169`,
  extended from HIP to CUDA. **Works** — the failing call has
  `entry.compiled = False` so `entry.runnable` is bound to
  `compiled_graph_for_general_shape` (the inductor-compiled general-shape
  graph), which is callable in eager. Falling back to it instead of
  asserting preserves correctness, just with no cudagraph speedup for
  the recompiled frame. Safe; **no PCG benefit on the image path**.
- **(Y) broaden warmup capture** by running a forward at warmup with
  `input_deepstack_embeds` synthesized to a non-None tensor.
  Constructing such a tensor is non-trivial (requires shape matching
  the multimodal embedding pipeline). If done, the multimodal compile
  frame gets PCG-captured during warmup ⇒ no inference-time recompile
  ⇒ no assertion ⇒ full image+text PCG benefit measurable. Bigger
  change, larger surface area; the upstream auto-disable on VLMs
  suggests this hasn't been attempted before.
- **(Z) per-model opt-in.** Only meaningful as a "yes I support
  multimodal PCG" declaration after (Y) is in place; pointless by
  itself.

R3 will commit to one of X, Y, Z based on a feasibility probe for (Y).
If the warmup driver can't easily synthesize a deepstack-embeds tensor,
(X) becomes the right short-term shape — it ensures #4 benchmarks
don't crash and exposes a *measurable* image+text PCG-ON baseline (even
if eager) for comparison against the text-only PCG-ON result from #2.

## 6. Notes for the upstream issue (R5)

The R2 evidence directly answers two follow-up questions a maintainer
would raise on an upstream issue:

1. *"Is the assertion reachable on CUDA at all?"* — Yes, deterministically
   on Qwen3-VL + `--enforce-piecewise-cuda-graph` + the first image
   request. Two clean reproduction recipes already exist (E2a from prior
   debug, R1+R2 from this sub-track).
2. *"Why doesn't the existing soft-warmup branch swallow this?"* —
   It only absorbs the *first* call at each shape. For a recompiled
   frame, no real PCG capture phase ever runs over it, so the second
   call at any shape hits the missing-stream check. The soft warmup is
   not a recovery path, it's a "skip the very first warmup call"
   optimization.
