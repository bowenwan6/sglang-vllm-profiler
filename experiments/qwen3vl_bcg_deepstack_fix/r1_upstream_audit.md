# R1 — Current-main upstream audit (2026-08-04)

**Status.** CPU-only research; no source touched, no GPU run. Audits
`sgl-project/sglang` at HEAD `e76d0acdc923d992bbda20d4b2bc51db9ac314a7`
(2026-08-04 17:17:58Z, PR #33346 merge) against the pinned SGLang SHA
`58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` (2026-07-31, `plan.md` §7.2)
and the historical fork HEAD `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`
(`/data/sglang-fork`, `plan.md` §4.1). All line numbers below refer to
the current-main files under `<scratchpad>/upstream_main/`.

## 0. Preservation invariants (verified)

* `/data/sglang-fork` HEAD unchanged (`986c89e69c…`).
* Frozen scratchpad checkout at `58974ca16c…` untouched (not read this
  pass; upstream shallow clone lives in a separate directory).
* Working tree unchanged apart from the two long-standing protected
  files noted in `plan.md` §7.6.
* Harness self-test `experiments/qwen35_4b/scripts/test_instrumentation.py`
  passes on CPU — pre-hook lifecycle, zero-mode substitution, Qwen3-VL
  toy generalisation, and the BCG allowlist monkey-patch against the
  real `sglang.srt.configs.model_config` all pass.

## 1. Overall drift since the pinned SHA

* Pinned `58974ca16c…` (2026-07-31) → current `e76d0acdc9…` (2026-08-04):
  4 days of upstream drift.
* `prefill_cuda_graph_runner.py` grew from **994 lines** (fork/pinned)
  to **1756 lines** (current main): +762 lines (+76 %). Substantial
  reorganisation of the BCG code path.
* `run_dummy_multimodal_deepstack_forward` was previously scoped to TC
  piecewise; still scoped to TC piecewise on current main (unchanged in
  role but relocated from ~662-725 to 687-750).

## 2. Answers to the five verification questions

### 2.1 Is Qwen3-VL eligible for prefill BCG on current main?

**No.** Verified at
`python/sglang/srt/configs/model_config.py:1839-1842`:

```python
multimodal_breakable_cuda_graph_supported_model_archs = [
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForConditionalGeneration",
]
```

The BCG allowlist is unchanged from the pinned SHA. Qwen3-VL
(`Qwen3VLForConditionalGeneration`, `Qwen3VLMoeForConditionalGeneration`)
is **not** present. `server_args.py:4508` still disables prefill
BCG for any multimodal arch not on this allowlist. Qwen3-VL therefore
runs eager for prefill on shipped upstream today, exactly as at the
pinned SHA — the failure mode remains a **latent regression** unless
a monkey-patch or a future upstream allowlist addition unlocks it.

### 2.2 Does non-empty `input_deepstack_embeds` reach the LM entry?

**Yes, when routed.** `general_mm_embed_routine` in
`python/sglang/srt/managers/mm_utils.py:1122-1145` and `:1203-1245`
allocates `input_deepstack_embeds = torch.zeros(N, hidden × num_ds)`
per multimodal call, scatters the visual tiles into it, and writes
`other_info["input_deepstack_embeds"]`; the routine downstream
passes `input_deepstack_embeds=<tensor>` as a kwarg to the LM's
`.forward(...)`. The pattern is unchanged from the pinned SHA.

`Qwen3LLMModel.forward` at `python/sglang/srt/models/qwen3_vl.py:1106+`
still accepts `input_deepstack_embeds` and applies it at
layer_idx `[8, 16, 24]` (Qwen3-VL-8B) via
`hidden_states.add_(input_deepstack_embeds[:, sep : sep + hidden_size])`.
Attempt 03 previously confirmed the tensor is populated with
`nonzero_frac ≈ 0.98` and reaches the LM under a monkey-patched
allowlist — that evidence remains applicable because none of the
routing above has moved.

### 2.3 Does BCG have a stable replay slot for `input_deepstack_embeds`?

**No.** Verified at
`python/sglang/srt/model_executor/cuda_graph_buffer_registry.py:855-877`:

```python
if is_multimodal:
    slots.append(GraphSlot("mrope_positions", ...))
    if register_input_embeds:
        slots.append(GraphSlot(
            "input_embeds",
            lambda _bs2, mt: (mt, hidden_size),
            embed_dtype,
            axis="tokens",
            padding_policy=PaddingPolicy.ZERO,
            copy_from_fb=False,
        ))
```

Only `input_embeds` (and `mrope_positions`) is registered for the
multimodal case. There is no `input_deepstack_embeds` slot anywhere
in `cuda_graph_buffer_registry.py` or in the prefill registry
builder. This is the symmetric absence to the `input_embeds` slot
that PR #30872 introduced.

### 2.4 Where is the value omitted, ignored, or replaced?

**Three co-omissions on current main.** All three must be closed
together for the value to survive BCG capture + replay:

1. **Slot registration** —
   `cuda_graph_buffer_registry.py:867-877` registers
   `"input_embeds"` for multimodal but no
   `"input_deepstack_embeds"` slot.
2. **Capture-pass `_run_forward`** —
   `prefill_cuda_graph_runner.py:660-668`:
   ```python
   if self._uses_eager_prefill_tail():
       positions = self._get_layer_model_positions(forward_batch)
       return self.layer_model.forward(
           forward_batch.input_ids,
           positions,
           forward_batch,
           forward_batch.input_embeds,
       )
   ```
   Four positional arguments; no `input_deepstack_embeds` kwarg.
   The captured graph is built with the DeepStack `add_` branch
   cold and never traces the `add_` kernels for the DeepStack
   layers (0–2 in Qwen3.5, `[8, 16, 24]` in Qwen3-VL-8B, etc.).
3. **Replay bridge `replay_layer_forward`** —
   `prefill_cuda_graph_runner.py:1610-1628`:
   ```python
   def replay_layer_forward(*args, **layer_kwargs):
       if self.buffer_registry.has_slot("input_embeds"):
           ie = layer_kwargs.get("input_embeds")
           if ie is None and ie_idx is not None and len(args) > ie_idx:
               ie = args[ie_idx]
           if ie is not None:
               self.buffer_registry.get_slot("input_embeds").slice_for(
                   1, static_num_tokens
               )[: ie.shape[0]].copy_(ie)
       hs = self.backend.replay(shape_key, static_forward_batch, **kwargs)
       return _slice_output_rows(hs, raw_num_tokens) if full_path else hs
   ```
   Reads `layer_kwargs["input_embeds"]` and copies it into the
   registered slot. Does **not** read `layer_kwargs["input_deepstack_embeds"]`
   and has no slot to copy it into anyway. `**kwargs` bound to the
   enclosing `_execute_body_capture` is passed to `.replay()`
   which ignores kwargs at
   `runner_backend/breakable_cuda_graph_backend.py:242-250`.

None of `can_replay_locally`
(`prefill_cuda_graph_runner.py:1021-1088`) or `can_run_graph`
(`prefill_cuda_graph_runner.py:1090-1132`) gate on
`input_deepstack_embeds` presence or value, so an image request
with a populated tensor passes both eligibility checks and enters
BCG replay unimpeded — the tensor is simply dropped along the way.

### 2.5 Can the existing `input_embeds` replay design generalize safely?

**Yes, cleanly.** The `input_embeds` handling on current main is a
three-site pattern (slot registration + capture-pass source +
replay-pass copy) that is now well-established after the pinned →
main window. The equivalent trio for `input_deepstack_embeds` is:

1. **Slot registration** (mirror of `cuda_graph_buffer_registry.py:867-877`)
   — add a conditional `GraphSlot("input_deepstack_embeds",
   lambda _bs2, mt: (mt, hidden_size * num_deepstack_embeddings),
   embed_dtype, axis="tokens", padding_policy=PaddingPolicy.ZERO,
   copy_from_fb=False)` gated on
   `getattr(model, "num_deepstack_embeddings", 0) > 0`. When the
   model configures no DeepStack layers (every shipped
   `Qwen/Qwen3.5-*` today), no slot is allocated and no memory or
   copy overhead is introduced. This preserves Qwen3.5 behaviour
   verbatim.
2. **Capture-pass source** (mirror of `_run_forward:660-668`) — pass
   the slot buffer through as `input_deepstack_embeds=<slot>` at
   capture time. The captured graph then traces the DeepStack
   `add_` at the target layer indices. This mirrors what
   `run_dummy_multimodal_deepstack_forward` does today for
   TC piecewise (`prefill_cuda_graph_runner.py:687-750`); the BCG
   capture pass needs the same treatment.
3. **Replay-pass copy** (mirror of `replay_layer_forward:1619-1626`)
   — after the current `input_embeds` copy, add a symmetric
   `if self.buffer_registry.has_slot("input_deepstack_embeds"):`
   block that pulls the live tensor from `layer_kwargs`, checks
   `.numel() > 0`, and copies into the slot before `.replay()`.

The pattern **generalises without new abstractions** — it is the
same pattern reviewers have already approved for `input_embeds`.
Correctness for the currently-empty case (Qwen3.5) is preserved by
the `num_deepstack_embeddings > 0` gate on slot allocation.

## 3. What has changed vs the pinned-SHA plan

The prior plan (`experiments/qwen3vl_bcg_deepstack_fix/plan.md` §4)
proposed three fix approaches. In light of current-main evidence:

* **Approach 4.A (register-slot-and-copy)** remains the recommended
  design and becomes **strictly cleaner** to implement because the
  `input_embeds` symmetric pattern is now visible in the code
  reviewers will diff against. The fix is a 3-site copy of that
  pattern.
* **Approach 4.B (numel-guard + eager fallback)** remains as a
  defence-in-depth guard around the slot copy, unchanged in
  reasoning.
* **Approach 4.C (dummy-trace at capture only)** was already ranked
  as insufficient standalone; current main confirms this — the
  TC piecewise `run_dummy_multimodal_deepstack_forward` exists in
  parallel with the `input_embeds` slot-and-copy for BCG. BCG needs
  both, and the slot is doing the load-bearing work.

The `input_embeds`-half of the bug in the pinned/fork snapshot has
been **fixed upstream between the pin and current main**. The
`input_deepstack_embeds`-half remains. Filing a PR against current
main is therefore a smaller, cleaner delta than a PR against the
pinned SHA would have been — the reviewers already accepted the
pattern.

## 4. Harness compatibility summary

| Harness dependency | Current-main location | Status |
|---|---|---|
| `multimodal_breakable_cuda_graph_supported_model_archs` import path | `sglang.srt.configs.model_config` (line 1839) | Stable; monkey-patch symbol path unchanged |
| `is_multimodal_breakable_cuda_graph_supported` accessor | `model_config.py:1910` | Stable; attribute-based patching still works |
| `Qwen3LLMModel` class name (pre-hook target) | `qwen3_vl.py:1106` (was `:1104` on fork) | Class name unchanged; hook still matches |
| `Qwen3_5ForCausalLM.forward(input_deepstack_embeds=...)` | `qwen3_5.py:1415` | Kwarg signature unchanged; numel guard at :1450-1451 |
| `general_mm_embed_routine` DeepStack synthesis | `mm_utils.py:1122-1245` | Structure unchanged; still writes `other_info["input_deepstack_embeds"]` |
| `Qwen3VLForConditionalGeneration.num_deepstack_embeddings` | `qwen3_vl.py:1302` | Attribute present; used by TC piecewise dummy |
| Existing byte-pinned image fixture | `experiments/qwen35_4b/scripts/fixtures/` | Verifiable; hash pinned |
| `test_instrumentation.py` on CPU | This repo | Passes 2026-08-04 |

**All harness pieces are compatible with current main.** No script
edits are required to point the harness at a current-main clone; only
the `--frozen-sglang` path and the SHA pin would need updating (both
CLI-configurable).

## 5. Blocker (unchanged from §9)

The nvidia driver `595.71.05` upgrade on 2026-08-04 12:53 UTC
prevents torch `2.11.0+cu130` from initialising CUDA
(`cuInit(0) → 803`). All GPU reproduction rungs (R2 onward) remain
paused pending an environment fix. Every CPU-only rung (R0, R1)
has now been completed successfully.

## 6. Consolidated verdict for the sub-track

The current-main audit **strengthens** the sub-track's premise:

1. The `input_deepstack_embeds`-under-BCG bug is present on current
   main, not just on the pinned SHA. Filing an upstream PR against
   current main is the correct target.
2. The failure mode is latent (Qwen3-VL not on the BCG allowlist) —
   any upstream PR should either (a) land the slot+copy fix as a
   defensive future-proofing change, or (b) land it together with
   adding Qwen3-VL to the BCG allowlist.
3. The fix design is now provably symmetric with an existing landed
   pattern (`input_embeds` slot-and-copy). The PR delta is bounded
   and reviewable.
4. The harness is fully compatible with current main; no script
   rewrite is required.
5. The R2+ reproduction rungs remain blocked on the shared driver
   issue but can proceed on any host with a compatible driver /
   torch combination.

## 7. Immediate next steps (recommended)

1. **Update the sub-track plan** (`plan.md` §10 and
   `experiments/qwen3vl_bcg_deepstack_fix/plan.md`) with the current-
   main audit findings and the sharper fix delta.
2. **Prepare (do not apply) the current-main fix patch** as a
   scratchpad file: the three-site slot-and-copy for
   `input_deepstack_embeds`. CPU-only preparation is safe; the
   patch stays out of `/data/sglang-fork` and out of the frozen
   scratchpad checkout.
3. **Write a paired regression-test skeleton** (unit + integration)
   that would exercise the fix.
4. **When CUDA returns** — run R2 baseline reproduction to
   demonstrate the bug on a fresh current-main clone under the
   monkey-patch; then R3 unnecessary because R1 already answered
   the upstream-current question at the source level; then R4-R8
   only if the R2 signature is not identical to Attempt 03.
