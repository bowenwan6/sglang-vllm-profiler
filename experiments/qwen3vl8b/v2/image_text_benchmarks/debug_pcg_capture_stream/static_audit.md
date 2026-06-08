# Static audit — PCG capture-stream assertion (no GPU)

> Read-only audit of `/data/sglang-pr` (upstream main, HEAD `62c505a196`, contains
> merged generator fix `07f326c184`) and the Stage 4.2 IMG-A run artifacts. No
> servers started, no benchmarks run, no SGLang source modified. Files inspected
> are all under `/data/sglang-pr/python/sglang/srt/...` and
> `/data/sglang-vllm-profiler/...`.

## TL;DR

- The crash is a **defensive assertion** in
  `srt/compilation/cuda_piecewise_backend.py:170-172` that fires when
  `get_pcg_capture_stream()` returns `None` at the moment a per-shape
  `torch.cuda.graph(...)` capture is attempted.
- Upstream SGLang **auto-disables PCG for multimodal models** as a deliberate
  safety: `_handle_piecewise_cuda_graph()` in `srt/server_args.py:1375-1376`
  sets `self.disable_piecewise_cuda_graph = True` whenever
  `self.get_model_config().is_multimodal` is true (Qwen3-VL qualifies via
  `Qwen3VLForConditionalGeneration ∈ multimodal_model_archs`).
- **`--enforce-piecewise-cuda-graph` is explicitly an override**: lines
  1342-1346 return early from `_handle_piecewise_cuda_graph()` before any of the
  auto-disable rules run, with the inline comment
  *"Skip auto-disable when enforce flag is set (for testing)"*. So forcing PCG
  on a VLM bypasses every safety listed below and walks straight into the
  assertion.
- Token-budget mismatch is **not** the cause here. The runtime token count for
  Qwen3-VL c=1 image+text prefill is ≈ 1024 (882 vision + 142 text), and 1024
  is in the captured size list (`Capture cuda graph num tokens [4, 8, …, 960,
  1024, 1280, …, 8192]` from the S2 server log). The Dynamo "recompile because
  token count exceeded captured range" pattern documented in the source
  comments does not match this trace.
- More likely: the **multimodal embed path** (`general_mm_embed_routine →
  language_model.forward`) triggers a Dynamo recompile branch whose execution
  reaches the captured size entry from outside the
  `PiecewiseCudaGraphRunner.capture()` `set_pcg_capture_stream(...)` window,
  so `_pcg_capture_stream` is None at assertion time.
- Static evidence is sufficient to classify this as **expected unsupported
  behavior under the documented `--enforce-piecewise-cuda-graph` override on
  VLMs**, not a regression of the generator fix `07f326c184`.

---

## 1. Where is `set_pcg_capture_stream()` normally set?

Defined in `srt/compilation/piecewise_context_manager.py`:

```python
# line 18 — module-level global
_pcg_capture_stream = None

# line 29-30 — read accessor
def get_pcg_capture_stream():
    return _pcg_capture_stream

# line 58-63 — context manager that sets / resets the global
@contextmanager
def set_pcg_capture_stream(stream: torch.cuda.Stream):
    global _pcg_capture_stream
    _pcg_capture_stream = stream
    yield
    _pcg_capture_stream = None
```

Called exactly once in the normal flow, in
`srt/model_executor/piecewise_cuda_graph_runner.py:481-513`
(`PiecewiseCudaGraphRunner.capture()`):

```python
def capture(self) -> None:
    with (
        freeze_gc(...),
        graph_capture() as graph_capture_context,
    ):
        stream = graph_capture_context.stream
        with set_pcg_capture_stream(stream):
            # … for each num_tokens in self.capture_num_tokens (reversed):
            self.capture_one_batch_size(num_tokens)
```

So `_pcg_capture_stream` is non-None **only** during the dedicated per-shape
capture loop driven by `PiecewiseCudaGraphRunner.capture()` at server warmup.
Everywhere else (request-handling serving path), it is `None`.

## 2. Why does `cuda_piecewise_backend.py:171` expect it?

`CUDAPiecewiseBackend.__call__` (`cuda_piecewise_backend.py:112-225`) is the
entry point for one compiled subgraph of the piecewise-compiled model forward.
The branch that fires the assertion is the **cudagraph capture branch** for a
shape that lives in `concrete_size_entries` but has no captured graph yet:

```python
# lines 151-172 (paraphrased)
if entry.cudagraph is None:
    if entry.num_finished_warmup < 1:
        entry.num_finished_warmup += 1
        return entry.runnable(*args)           # warmup pass: no capture yet

    # capture path
    stream = get_pcg_capture_stream()
    if _is_hip and stream is None:
        print_warning_once(
            "PCG capture stream is not set; likely a Dynamo runtime "
            "recompilation. Falling back to eager execution for this subgraph."
        )
        return entry.runnable(*args)            # HIP-only safety fallback
    assert (
        stream is not None
    ), "PCG capture stream is not set, please check if runtime recompilation happened"
    # … torch.cuda.graph(cudagraph, pool=self.graph_pool, stream=stream) …
```

The stream is required by `torch.cuda.graph(..., stream=stream)` (line 192 in
the source) to record the CUDA graph against the dedicated capture stream
allocated by `graph_capture()` in `PiecewiseCudaGraphRunner.capture()`.

The HIP-only fallback (lines 163-169) is documented inline:

> During normal capture (PiecewiseCudaGraphRunner.capture()),
> `set_pcg_capture_stream()` guarantees a valid stream. However, Dynamo may
> silently recompile on HIP/MLA serving batches whose token count exceeds the
> captured range. The replacement backend has no capture stream; fall back
> there instead of crashing while preserving the original assertion on other
> platforms.

The assertion is therefore intentionally **strict on CUDA**: any path that
reaches this branch outside the dedicated capture phase is treated as a bug
or unsupported scenario, with a deliberate decision **not** to silently fall
back on CUDA.

## 3. Initial capture, replay, or runtime recompilation?

- **Not replay.** Replay takes the `entry.cudagraph.replay()` branch at line
  224, which does not consult the capture stream.
- **Not the warmup pass.** Line 152-154 handles `num_finished_warmup < 1` by
  short-circuiting back to `entry.runnable(*args)` without touching the
  stream.
- **Most consistent with "out-of-capture-phase capture attempt."** The
  assertion fires when:
  - the runtime shape (`args[sym_shape_indices[0]]`) is in
    `concrete_size_entries` (i.e. it matches one of the configured PCG
    capture sizes), and
  - `entry.cudagraph` is still `None` (not yet captured for this shape), and
  - `num_finished_warmup >= 1` (warmup is done so the code wants to actually
    capture), and
  - we are **not** inside `PiecewiseCudaGraphRunner.capture()`'s
    `with set_pcg_capture_stream(stream):` block, so the global is `None`.

  The textbook trigger is a Dynamo runtime recompile that produces a fresh
  subgraph after the regular warmup-and-capture phase has ended. The inline
  comment names this case explicitly ("Dynamo may silently recompile …").

## 4. What exact model path calls into the failing backend?

Full traceback head from
`logs/qwen3vl8b/v2/image_text_benchmarks/results_fixed/IMG_A_S2_ipc_pcg_server.log`:

```text
File "/data/sglang-pr/python/sglang/srt/models/qwen3_vl.py",
     line 1277, in forward
    hidden_states = general_mm_embed_routine(

File "/data/sglang-pr/python/sglang/srt/managers/mm_utils.py",
     line 1138, in general_mm_embed_routine
    hidden_states = language_model(

File "/usr/local/lib/python3.12/dist-packages/torch/nn/modules/module.py",
     line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)

File "/data/sglang-pr/python/sglang/srt/compilation/compile.py",
     line 195, in trampoline
    return compiled_callable(*args, **kwargs)

File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py",
     line 1024, in compile_wrapper
    return fn(*args, **kwargs)

File "/data/sglang-pr/python/sglang/srt/models/qwen3_vl.py",
     line 1001, in forward
    def forward(

File "/usr/local/lib/python3.12/dist-packages/torch/_dynamo/eval_frame.py",
     line 472, in __call__
    return super().__call__(*args, **kwargs)

…

File "<eval_with_key>.669", line 298, in forward
    submod_0 = self.submod_0(l_input_embeds_, s47, …)

File "/data/sglang-pr/python/sglang/srt/compilation/cuda_piecewise_backend.py",
     line 171, in __call__
    stream is not None
AssertionError: PCG capture stream is not set, please check if runtime
                recompilation happened
```

Path summary:

```
Qwen3VLForConditionalGeneration.forward (qwen3_vl.py:1277)
  → general_mm_embed_routine (mm_utils.py:1138)
    → language_model(...)  (the inner Qwen3 LM)
      → torch.compile-wrapped forward (qwen3_vl.py:1001 + eval_frame.compile_wrapper)
        → FX graph forward (<eval_with_key>.669:298)
          → submod_0  ← this submod's backend is CUDAPiecewiseBackend
            → assertion at cuda_piecewise_backend.py:171
```

So the failing piecewise subgraph is the first language-model attention block
inside the FX-decomposed VLM forward, called from the multimodal embed
routine.

## 5. PCG auto-disable / enforce policy (the decisive bit)

From `srt/server_args.py:1342-1412` (`_handle_piecewise_cuda_graph`):

```python
def _handle_piecewise_cuda_graph(self):
    # Skip auto-disable when enforce flag is set (for testing)
    if self.enforce_piecewise_cuda_graph:
        self.disable_piecewise_cuda_graph = False
        return                                          # ← returns BEFORE checks

    # Disable piecewise cuda graph with following conditions:
    # 1. Disable Model Arch
    if self.get_model_config().is_piecewise_cuda_graph_disabled_model: …
    # 2. DP attention
    # 3. Torch compile
    # 4. Pipeline parallelism
    # 5. Non-CUDA hardware …
    # 6. MoE A2A backend
    # 7. LoRA
    # 8. Multimodal / VLM models                        ← our case
    if self.get_model_config().is_multimodal:
        self.disable_piecewise_cuda_graph = True
    # 9. GGUF / 10. DLLM / 11. CPU offload / …
```

`Qwen3VLForConditionalGeneration` is registered in
`srt/configs/model_config.py:1549` inside `multimodal_model_archs`, so
`ModelConfig.is_multimodal = True` for our snapshot. Without
`--enforce-piecewise-cuda-graph`, rule 8 would set
`disable_piecewise_cuda_graph = True`. With the enforce flag, the early
`return` at line 1346 bypasses **every** rule (1-19 above), including the
multimodal rule.

The S2 server log confirms the runtime state:

```text
disable_piecewise_cuda_graph=False,
enforce_piecewise_cuda_graph=True,
…
piecewise_cuda_graph_tokens=[4, 8, …, 960, 1024, 1280, …, 8192]
```

## 6. Token-budget hypothesis: NOT the cause

The inline comment at `cuda_piecewise_backend.py:156-161` warns about Dynamo
recompiling when "token count exceeds the captured range". But our runtime
prefill batch is ≈ 1024 tokens (882 vision + 142 text + chat-template overhead),
and 1024 is explicitly inside `piecewise_cuda_graph_tokens` (see server log
line 24 above). The PCG capture list ranges from 4 up to 8192. So the
"exceeds the captured range" condition is not met for this workload.

A more likely cause is a Dynamo recompile triggered by **VLM-specific control
flow** in `general_mm_embed_routine → language_model.forward`: branches that
depend on whether `pixel_values` is present, on mm tensor shape variance, on
mrope_positions presence, etc. Any such structural recompile happens at
serving time, well after `PiecewiseCudaGraphRunner.capture()` has exited its
`set_pcg_capture_stream(...)` block, so `_pcg_capture_stream` is `None` and
the CUDA path hits the assertion.

This hypothesis is consistent with — and unifies — the inline source comments,
the upstream auto-disable rule for VLMs (#8), and the observed traceback.

## 7. Evidence that this is not generator-related

- The fixed-generator import gate passed for both S0_ipc and S2_ipc_pcg
  preflights: `sglang.__file__ = /data/sglang-pr/python/sglang/__init__.py`,
  `sglang.benchmark.datasets.common.__file__ =
  /data/sglang-pr/python/sglang/benchmark/datasets/common.py`,
  `FIX_OK` marker present, merged commit `07f326c184` ancestor of HEAD.
- `IMG_A_S0_ipc` (same fixed-generator path, same dataset, same IPC env,
  PCG OFF) completed 5/5 reps with 0 failures, 2000 served requests, no
  forbidden-token errors. The bench client and server both ran the patched
  `gen_mm_prompt` over thousands of generated prompts.
- The traceback originates inside
  `sglang.srt.compilation.cuda_piecewise_backend`, not the dataset / harness /
  request-construction path.
- The crash is reproducible from server warmup state alone — no traffic from
  the bench client touches the generator before the crash; the request that
  triggers prefill is the first served request and it fails inside the model
  forward, not in the prompt construction.

## 8. What remains unknown (motivates D1-D4)

- **D1 (sanity):** Does the assertion reproduce deterministically on a tiny
  workload (e.g. 2 prompts) with the same `enforce-pcg + IPC + image` combo?
  Almost certainly yes given the static evidence, but a 2-prompt repro is
  cheap and removes any "rep-3 cache state" doubt.
- **D2 (IPC vs PCG):** Does the assertion still fire with `enforce-pcg` and
  the image dataset but **without** `SGLANG_USE_CUDA_IPC_TRANSPORT=1`? The
  capture-stream code path does not look at IPC, so we expect "yes, crashes
  the same way" — confirming that IPC is not a contributing factor and the
  fault is purely VLM+PCG.
- **D3 (PCG OFF positive control):** No-PCG + IPC + image — should pass (we
  already have S0_ipc as a 5-rep clean baseline; D3 just confirms the result
  is repeatable in a tiny variant of the debug runner).
- **D4 (text-only + PCG on upstream main):** Does
  `enforce-pcg + text-only random dataset` (Case-A-like) crash too on the new
  upstream main? Static evidence says no — the multimodal embed path is the
  trigger. But this directly distinguishes (a) "VLM+PCG specifically
  unsupported" from (b) "broader upstream PCG regression". A pass on D4
  closes off any thought of a wide PCG regression.

D5 (older `/sgl-workspace/sglang` text+PCG comparison) and D6 (lower
resolution / fewer tokens) are only needed if D1-D4 leave the picture
unclear.

## 9. Tentative conclusion (subject to D1-D4 confirmation)

The crash is **expected unsupported behavior**, not a bug introduced by the
generator fix:

- Upstream SGLang deliberately auto-disables PCG for multimodal models because
  the multimodal forward path triggers Dynamo recompilation patterns that the
  PCG infrastructure cannot safely handle.
- `--enforce-piecewise-cuda-graph` is a documented override "for testing"
  that intentionally bypasses every auto-disable rule.
- The assertion at `cuda_piecewise_backend.py:170-172` is a defensive guard.
  Its purpose is to surface, on CUDA, exactly the misuse pattern we just
  ran into.
- On AMD HIP, the same code path falls back to eager execution instead of
  asserting. On CUDA, the explicit decision is to fail loudly.

Probable follow-ups (subject to D1-D4):

- **Upstream issue (informational)**: it would be reasonable to file an
  upstream SGLang issue asking either (a) for `--enforce-piecewise-cuda-graph`
  to print a loud warning that it is unsafe on multimodal models, or (b) for
  the HIP fallback at `cuda_piecewise_backend.py:163-169` to be extended to
  CUDA so the override degrades gracefully instead of crashing. We should
  not propose a PR until we have a minimal fix and tested it locally.
- **Issue #4 implication**: PCG cannot be reported on the Qwen3-VL image+text
  path under the current upstream main without source changes. The #4 IMG-A
  bracket can run S0_ipc / S0_ipc_repeat / V0_vllm / S0_noipc (no PCG) for
  bracket drift + IPC + vLLM anchor; the **PCG benefit (Q2) finding from
  Issue #2 (text-only Case A) cannot be transferred to image+text in this
  upstream main**.
- **Issue #5 implication**: this strengthens the case that "selective /
  default-on PCG PR" should be conservatively scoped — PCG should remain
  auto-disabled for multimodal models, and any selective enable should be
  per-workload-shape rather than per-model.
