# Finding: the PCG capture-stream eager fallback is not a rare transient

**Status:** draft for a **new** upstream issue. Deliberately *not* folded into
issue #4 — [`plan.md`](../../../plan.md) §11.5 criterion 5 says a residual finding
opens its own issue rather than expanding #4's scope.

**Found by:** issue #4 v3 phase-1 engagement smoke, 2026-09-04.
**Stack:** `upstream/main` @ `ff1285cc28` (+ PR #33726, + local counters — see
[`manifest.md`](manifest.md) §7). **Model:** Qwen3-VL-8B-Instruct, GPU H200.

## Summary

On Qwen3-VL image serving under `--cuda-graph-backend-prefill tc_piecewise`, the
piecewise backend's capture-stream fallback fired **75 200 times across 81 638
graph-eligible calls (92.11%)** on 2 distinct shapes, in a 2030-request benchmark.
It is announced by a `print_warning_once`, so the log shows it **once**.

The affected subgraphs run eager for the rest of the process: past the first
fallback the rate is **99.95%** (75 200 of 75 237 calls). The arm's latency is
therefore an eager number wearing a `tc_piecewise` label — and it carried the
**lowest coefficient of variation of any arm in our bracket (0.7%)**, because an
arm that is consistently eager is consistently eager.

## Mechanism

`python/sglang/srt/compilation/cuda_piecewise_backend.py`, in the
`entry.cudagraph is None` branch:

```python
stream = get_pcg_capture_stream()
if stream is None:
    print_warning_once(
        "PCG capture stream is not set. This can be a Dynamo runtime "
        "recompilation or an optional VLM branch pre-warmed outside "
        "CUDA graph capture; falling back to eager execution for this "
        "subgraph."
    )
    return entry.runnable(*args)
```

Two properties combine badly:

1. **The branch returns without capturing.** `entry.cudagraph` stays `None`, so
   the *next* call on that shape re-enters the same branch. The code comment's
   reassurance — "subsequent matching shapes still use their captured graphs" —
   holds only for shapes captured at startup, not for the shape that just fell
   back. A shape that misses the capture stream once is eager permanently.
2. **`print_warning_once` is `@functools.lru_cache(None)`** on the message string
   (`utils/common.py:2796`). The count is capped at 1 by construction, so the log
   cannot distinguish one fallback from six hundred.

## Why it is hard to notice

Every other signal reports a healthy piecewise arm:

| signal | reported |
|---|---|
| `/server_info` → `cuda_graph_config.prefill.backend` | `tc_piecewise` |
| `Capture target prefill CUDA graph begin. backend=` | `tc_piecewise` |
| scheduler per-prefill line, all 21 benchmark batches | `cuda graph: True` |
| TTFT p50 | 144.8 ms — unremarkable next to the other arms |

The scheduler's indicator is per **batch**; the fallback is per **subgraph**. A
batch can be marked as running under a CUDA graph while a subgraph inside it
executes eagerly, so the batch-level flag cannot see this at all.

## Reproduction and measurement

Serve Qwen3-VL-8B with `--cuda-graph-backend-prefill tc_piecewise --disable-radix-cache`,
then send image+text requests (1×720p PNG + ~128 text tokens, greedy, c=1). Send
enough of them: onset is at graph-eligible call ~6400, so a short run misses most
of it.

The magnitude is only visible with counters. Ours are measurement-only and are
not proposed as the fix:

```
PCG_STATS periodic eligible=1000  eager_fallback=0     eager_shapes=0
PCG_STATS periodic eligible=6000  eager_fallback=0     eager_shapes=0
PCG_STATS fallback eligible=6402  eager_fallback=1     eager_shapes=1   ← onset
PCG_STATS fallback eligible=81438 eager_fallback=75000 eager_shapes=2
PCG_STATS fallback eligible=81638 eager_fallback=75200 eager_shapes=2
```

**Onset is deterministic in call count; the observed share is not.** Two
independent runs both put the first fallback at graph-eligible call **6402**. A
short run therefore under-reports it badly: a 20-request smoke ended 636 calls
past onset and scored 8.53%, while a 2030-request benchmark ran to 81 638 calls
and scored **92.11%**. Past onset the rate is **99.95%** (75 200 of 75 237). A
short clean-looking run is not evidence the path is healthy.

## Suggested directions

Offered as options, not a preferred design — the maintainers own this call.

1. **Make it observable.** Replace the warn-once with a counter plus a periodic
   or shutdown summary. This is the minimum: the current form actively hides
   magnitude, and a benchmark cannot be trusted without it.
2. **Make it recoverable.** Attempt capture on a later call once a capture stream
   is available, instead of leaving the entry permanently uncaptured.
3. **Make it loud when it matters.** A serving-time fallback that persists is a
   different event from a warmup-time one; the persistent case is worth an error
   or a metric rather than a suppressed warning.

## Impact on our own work

`A2_tcp` is reported `UNVERIFIED` in issue #4's bracket, so issue #4's **Q2**
("does #2's tc_piecewise win transfer to the image path?") cannot be answered on
current upstream. That is recorded as the arm's exact failure per §11.5
criterion 4, not worked around.
