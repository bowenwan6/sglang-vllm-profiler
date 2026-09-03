# Issue #9 — Conclusion: Qwen3.5 DeepStack propagation under multimodal prefill BCG

> Closing write-up for [profiler-repo issue #9](https://github.com/bowenwan6/sglang-vllm-profiler/issues/9),
> closed **2026-09-03** as `completed`. Structured after the SGLang
> [pull-request template](https://github.com/sgl-project/sglang/blob/main/.github/pull_request_template.md)
> so it can be pasted onto the tracker or reused as upstream-facing prose.
>
> **Verdict: `NOT_APPLICABLE_QWEN35`** for the question as posed, **plus** a
> confirmed and fixed defect on a different model — upstream
> [sgl-project/sglang#33726](https://github.com/sgl-project/sglang/pull/33726).

## Motivation

`Qwen/Qwen3.5-*` is registered as multimodal and sits on SGLang's **breakable
CUDA graph (BCG)** prefill allowlist. Its language model receives per-request
`input_deepstack_embeds`, added into `hidden_states` in the early decoder
layers.

A source-level read of the BCG replay bridge showed an asymmetry: the bridge
stabilises `input_embeds` — registered buffer slot plus a per-request copy into
that slot — but **no slot is registered for `input_deepstack_embeds`, and the
replay closure never forwards it.** The single upstream DeepStack
accommodation at the time (`run_dummy_multimodal_deepstack_forward`, PR #30868)
is a Dynamo shape-stability warmup scoped to the `tc_piecewise` backend and is
**not** called on the BCG capture path.

If that reading were right, a replayed BCG prefill would silently drop the
DeepStack contribution — wrong numerics with no error, no warning, and no
crash. The issue asked whether that actually happens on Qwen3.5.

## Modifications

Nothing was changed in SGLang for the Qwen3.5 question itself; it resolved as
not applicable. The work that *did* ship came out of the reproduction:

| Change | Where |
|---|---|
| Register a stable CUDA-graph slot for `input_deepstack_embeds`; refresh it per replay | `srt/model_executor/runner/prefill_cuda_graph_runner.py`, `runner_utils/buffers.py`, `cuda_graph_buffer_registry.py` |
| `supports_bcg_deepstack_replay` capability opt-in | `srt/models/qwen3_vl.py` |
| Fail closed — validate before any write; raise on a non-empty tensor that does not fit the slot (`None`/empty still clear it as genuine absence) | same runner |
| Qwen3-VL / Qwen3-VL-MoE added to `multimodal_breakable_cuda_graph_supported_model_archs` | `srt/configs/model_config.py` (added by the maintainer) |
| 8 tests / 11 subtests + server-args allowlist coverage | `test/registered/unit/…` |

Upstream PR: **+293 / −12 across 7 files**, approved by
[@JustinTong0323](https://github.com/JustinTong0323), fail-closed review from
[@charliechenye](https://github.com/charliechenye) addressed in `f2596d5e99`.

## Accuracy Tests

The diagnostic is a predeclared four-arm matrix on one fixture and one
server-side cache state, so a verdict has unambiguous attribution:

| Arm | Meaning |
|---|---|
| `eager_normal` | eager runner, DeepStack computed and passed normally |
| `eager_zero_deepstack` | eager, DeepStack replaced by `zeros_like` immediately before the LM forward — the `numel() > 0` guard still passes, so the `add_` still runs but contributes exactly zero |
| `bcg_normal` | BCG runner, DeepStack normal |
| `bcg_zero_deepstack` | BCG runner, zeroed |

A drop under replay has a specific signature: `bcg_normal` diverges from
`eager_normal` **and** matches `bcg_zero` **and** `bcg_zero` matches
`eager_zero`. Anything less is `AMBIGUOUS`.

### Why Qwen3.5 is `NOT_APPLICABLE`

`harness_gpu1_20260801T062833Z` (GPU 1, frozen SGLang `58974ca16…`) — machine
verdict `HARNESS_NOT_DIAGNOSTIC`.

The instrumentation was proven working first, so the null result is a fact
about the model and not about the harness:

- `register_forward_pre_hook(..., with_kwargs=True)` fires **111 times per
  arm** on real prefills, attributed to `Qwen3_5ForCausalLM` (Attempt 01 had
  fired 0 times — that attempt was discarded as a harness defect).
- The multimodal request is well-formed: `<|vision_start|><|image_pad|><|vision_end|>`,
  placeholder count 1, image count 1, **zero** `More image data items provided
  than corresponding tokens found in the prompt` warnings.
- The image is genuinely consumed: greedy output is
  `" The image features three vertical stripes in red, green, and blue."`,
  matching the byte-pinned `image_bands.png` fixture.

With the harness cleared, the model closes the question:

**Every publicly released `Qwen/Qwen3.5-*` checkpoint — 0.8B, 2B, 4B, 9B, 27B,
35B-A3B — ships `vision_config.deepstack_visual_indexes = []`.** Therefore:

1. `self.deepstack_visual_indexes = []` → `num_deepstack_embeddings = 0`.
2. In `general_mm_embed_routine`, `deepstack_embedding_shape` becomes
   `input_embeds.shape[:-1] + (hidden_size * 0,)` → the tensor is allocated
   with shape `(N, 0)`, `numel = 0`.
3. `Qwen3_5ForCausalLM.forward`'s DeepStack branch is guarded by
   `is not None and numel() > 0` → **trivially skipped**.

Runtime instrumentation agrees: `numel = 0`, `nonzero_frac = 0.0`, and
`eager_normal == eager_zero_deepstack` — the ablation has no signal to move
because there is no DeepStack contribution to remove.

**The harness will not fabricate the input.** `hypothesis.md` Amendment 5
forbids overriding `deepstack_visual_indexes`, hand-editing the served
`config.json`, or injecting a synthetic tensor. Forcing the branch would
measure a configuration no user can serve, so the honest verdict is
`NOT_APPLICABLE_QWEN35` — **not** `PASS` (nothing was exercised) and **not**
`FAIL` (nothing broke).

### The defect is real — on Qwen3-VL

`Qwen/Qwen3-VL-8B-Instruct` ships `deepstack_visual_indexes = [8, 16, 24]`
(width 12288), so it does exercise the path. Reproduced under a
profiler-owned, test-only BCG-allowlist patch:

| Milestone | Tree | Model | Verdict |
|---|---|---|---|
| R2 / M2 | unpatched upstream | Qwen3-VL-8B | **`FAIL_BCG_DEEPSTACK`** — full zero-DeepStack signature |
| M4 / M4b / M4c | patched | Qwen3-VL-8B | **`PASS_BCG_CORRECT`** — `bcg_normal` matches `eager_normal` and diverges from `bcg_zero` |
| M7 | patched | Qwen3-VL-8B | isolation: pass |
| M8 | patched | **Qwen3.5-4B** | `NOT_APPLICABLE_QWEN35` — no-regression gate: 4/4 requests served, `input_deepstack_embeds present=False` on every LM entry, `deepstack_replay_width = 0` → no slot registered |
| M9 | pre-merge | Qwen3-VL-30B-A3B (MoE) | PASS |
| M10 | post-merge | Qwen3-VL-4B (dense) | PASS |
| CPU unit suite | post-merge | — | 193 passed, 30 subtests |

M8 is the two-way close: the fix is inert on Qwen3.5 for exactly the reason
Qwen3.5 could not exhibit the bug.

Evidence: [`experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md`](../qwen3vl_bcg_deepstack_fix/upstream_handoff.md).

## Speed Tests and Profiling

None, deliberately. This is a **correctness** track — the failure mode is
silently wrong numerics at unchanged speed, so a latency number would neither
confirm nor refute it. All performance questions stay in issues #2/#3/#4/#5.

## Residual risk

1. **Every run in this track used a container ~771 commits behind upstream**,
   requiring `LD_PRELOAD` of the host `libcuda` and
   `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`. Both arms of every comparison
   share the confound, so each contrast is internally controlled — but no run
   here is production-representative. A confirming run on a current devbox was
   requested by the maintainer and remains **unrun**.
2. **MoE × post-merge is untested.** M9 covers MoE on the pre-merge tree, M10
   covers dense on the post-merge tree; the fourth cell is empty.
3. **Qwen3.5's own BCG behaviour is only gated, not characterised.** M8 shows
   the DeepStack path is inert; it says nothing about other Qwen3.5 BCG
   surfaces. That is what the separate GDN sub-track pursued
   (`PASS_BCG_GDN_NOTABLE_GAP`, [`gdn/final_report.md`](gdn/final_report.md)).
4. **Latent-regression scope was never swept.** The same missing-slot pattern
   could affect any other per-request tensor a multimodal LM forward receives.
   Only `input_deepstack_embeds` was audited.

## What can be done next

| | Item | Cost |
|---|---|---|
| 1 | **Land #33726.** Currently `mergeable: true`; the one red lane (`extra-a-test-1-gpu-small`) failed in 11 s — an infrastructure abort, not a test failure, and an AMD lane outside a CUDA-only change's blast radius. Re-run it, then ping merge oncall. | none |
| 2 | **Confirming smoke on a current devbox** — closes residual risk 1, the one thing a reviewer explicitly asked for. | small, GPU |
| 3 | **MoE × post-merge smoke** — fills the empty cell in residual risk 2. | small, GPU |
| 4 | `assertRaises(RuntimeError)` → `assertRaisesRegex` so the fail-closed guard's message is pinned (reviewer item 6, optional). | one line |
| 5 | **Sweep for sibling latent regressions** — audit every other per-request tensor crossing the BCG replay boundary for the same missing-slot pattern. Highest-value follow-up; a genuinely new work item, not #9. | medium |
| 6 | **Re-test DeepStack under BCG when a Qwen3.5 checkpoint ships a non-empty `deepstack_visual_indexes`.** The verdict is scoped to what is released today and expires the moment that changes. | conditional |

**Do not conflate BCG with PCG.** This track is the breakable CUDA graph
prefill path. Issue #4's capture-stream sub-track and issue #5's policy
question are about `tc_piecewise` (PCG). They are different backends; a result
on one is not a result on the other.
