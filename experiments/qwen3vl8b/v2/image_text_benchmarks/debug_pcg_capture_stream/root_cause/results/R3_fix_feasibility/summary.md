# R3 — Fix-shape feasibility (X) validation

> Outcome: **PASS** (`classification: OK_FALLBACK_TAKEN`). The (X)
> minimum-safety patch — extending the existing HIP eager fallback in
> `CUDAPiecewiseBackend.__call__` to also cover CUDA — removes the
> deterministic `PCG capture stream is not set` assertion crash on
> Qwen3-VL + `--enforce-piecewise-cuda-graph` and lets the n=32
> warmup-30 image+text bench complete cleanly.

## 1. Run identity

| | |
|---|---|
| Recipe | image 720p, 1 image, c=1, n=32, warmup=30, output_len=128 |
| Server flags | `--enforce-piecewise-cuda-graph`, `--attention-backend flashinfer`, IPC on |
| Env vars | `SGLANG_DEBUG_PCG_CALL_TRACE=1`, `CUDA_VISIBLE_DEVICES=0`, `SGLANG_USE_CUDA_IPC_TRANSPORT=1` |
| sglang | `/data/sglang-fork` branch `fix/pcg-vlm-deepstack-warmup` HEAD `8a2dcb33a` (off upstream `da802ddca`) via `PYTHONPATH=/data/sglang-fork/python` |
| GPU | 0 (H200), 0 MiB before/after |
| Patch reference | `../../patches/R3_fix_X_cuda_eager_fallback.patch` |
| Classification | **`OK_FALLBACK_TAKEN`** |

## 2. What the (X) patch does

In `cuda_piecewise_backend.py:__call__`, the missing-capture-stream
case used to be guarded by `if _is_hip and stream is None: ... return`.
On CUDA the assertion at the next line was reached instead. The (X)
patch drops the `_is_hip and` clause:

```diff
-            if _is_hip and stream is None:
-                print_warning_once(...)
-                return entry.runnable(*args)
-            assert (
-                stream is not None
-            ), "PCG capture stream is not set, please check if runtime recompilation happened"
+            if stream is None:
+                print_warning_once(
+                    "PCG capture stream is not set; likely a Dynamo runtime "
+                    "recompilation. Falling back to eager execution for this "
+                    "subgraph."
+                )
+                return entry.runnable(*args)
```

`entry.runnable` is bound to `compiled_graph_for_general_shape` (the
inductor-compiled general-shape function), which is callable in
eager. The recompiled multimodal frame loses cudagraph speedup but
correctness and stability are preserved.

## 3. Server-log evidence

| | |
|---|---|
| `PCG capture stream is not set, please check if runtime recompilation happened` | 0 occurrences |
| `AssertionError` | 0 occurrences |
| `Traceback (most recent call last)` | 0 occurrences |
| `Falling back to eager execution for this subgraph` | 1 occurrence (line 17631), idempotent via `print_warning_once` |
| `[PCG_DEBUG] about to capture; stream=None` | seen exactly at the moment the new fallback fires |
| Total `[PCG_DEBUG]` lines in raw/server.log | ~17 700+ (consistent with R2 scale) |

## 4. Bench client summary

Bench client `python3 -m sglang.benchmark.serving --dataset-name image
--image-resolution 720p --num-prompts 32 --warmup-requests 30
--max-concurrency 1 --random-input-len 128 --random-output-len 128
--random-range-ratio 1.0`:

| Metric | Value |
|---|---|
| Successful requests | **32 / 32** |
| Benchmark duration | 24.38 s |
| Mean TTFT | **104.81 ms** |
| Median TTFT | 103.05 ms |
| P99 TTFT | 124.93 ms |
| Mean TPOT (excl. 1st) | 5.17 ms |
| Output token throughput | 168.03 tok/s |
| Total token throughput | 1 513.26 tok/s |
| Mean E2E latency | 761.06 ms |

(Full table in `bench_summary.txt`.)

## 5. Cross-comparison vs Stage 4.2 IMG_A baselines

| Variant | Reps × n | TTFT p50 | Notes |
|---|---|---|---|
| `IMG_A_S0_ipc` (no PCG) | 5 × 400 | 64.8 ms | Stage 4.2 headline-quality (n=400 × 5 reps) |
| `IMG_A_S2_ipc_pcg` (PCG, *no fix*) | aborted | — | crashed deterministically: `PCG_CAPTURE_STREAM_ASSERT` |
| **R3.B `IMG_A_S2_ipc_pcg` (PCG, with (X) fix)** | 1 × 32 | **103.05 ms** | single n=32 sample (not headline-quality); 0 failures, fallback fired |

Two observations from the R3.B number (with the caveat that 1 × 32 is
*not* a headline-quality measurement — variance bounds are unknown):

1. **Image+text PCG-ON with the (X) fix is *slower* than PCG-OFF on
   this same recipe** (~103 vs 65 ms TTFT). That is consistent with R2's
   structural picture: the recompiled multimodal frame loses cudagraph
   replay benefit and falls back to the inductor general-shape graph.
   PCG-ON only helps on the warmup-captured *text* compile frame.
2. So **(X) is correct for safety but does not deliver the Issue #4 Q2
   "PCG benefit on image+text" measurement.** That benefit requires
   capturing the multimodal Dynamo frame at warmup — which is what
   fix shape (Y) addresses.

This is the same conclusion R3.A's source-read reached, now backed by a
real run.

## 6. Decision for R4

R4 commits **(X)** as the upstream-suitable minimum-safety patch and
documents **(Y)** as the follow-up for actually unlocking the image+text
PCG benefit. R4 plan:

1. **Hold (X)** at `fix/pcg-vlm-deepstack-warmup` HEAD `8a2dcb33a`.
   Already pushed to `bowenwan6/sglang`.
2. **Re-run** the recipe with `SGLANG_DEBUG_PCG_CALL_TRACE` unset to
   confirm the fix path works without the diagnostic gate (production
   shape).
3. **Stretch validation**: run a longer recipe (n=400, warmup=30,
   matching prior IMG_A_S2_ipc_pcg) to confirm no degradation under
   load. Optional — only if R4 wants a headline-quality datapoint.

R5 then drafts the upstream issue using R1+R2+R3 evidence and recommends
(X) for merge. (Y) is left as future work scoped to Issue #5's selective
PCG defaults track.

## 7. What this does NOT do

- Does not enable Issue #4 Q2 "PCG benefit on image+text" measurement.
  That stays blocked behind fix (Y).
- Does not change the upstream auto-disable for VLMs at
  `server_args.py:3145-3146`. `is_multimodal_piecewise_cuda_graph_supported`
  defaults are unchanged — Qwen3-VL still auto-disables PCG; the
  override flag still works (and now degrades gracefully instead of
  crashing).
- Does not touch the `--disable-piecewise-cuda-graph` hint that the
  existing in-source error message recommends. That hint stays valid
  as a workaround for any operator who wants no PCG at all.
