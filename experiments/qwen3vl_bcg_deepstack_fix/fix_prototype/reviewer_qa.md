# Reviewer Q&A — evidence-based answers

Concise answers to the 20 questions a strict SGLang maintainer would
likely ask. Every claim references the actual source, tests, or
measured evidence.

---

**Q1. Why is this not handled by the existing `input_embeds` slot?**

`input_embeds` is the composed text+vision embeddings the LM's
transformer stack starts from — shape `(num_tokens, hidden_size)`.
DeepStack is a per-layer *additive* residual contribution at
target layer indices (Qwen3-VL-8B: `[8, 16, 24]`), packed on the
feature axis with shape `(num_tokens, hidden_size × num_deepstack_embeddings)`.
It cannot be smuggled through `input_embeds` because (a) it has a
different shape, (b) it must be applied at specific layers via
`hidden_states.add_(input_deepstack_embeds[:, sep:sep+hidden])`
(qwen3_vl.py:1136), not at the LM entry.

**Q2. Why does DeepStack need a separate stable input?**

BCG capture records kernel arguments by pointer. If DeepStack came
through a fresh per-request tensor, replay would reuse the captured
pointer — reading whatever was at that address at replay time. A
stable slot (like `input_embeds` already has) is the minimum
requirement for the captured graph to see current-request data.

**Q3. Why is a model capability flag preferable to checking the
runtime argument?**

Because a model could expose `num_deepstack_embeddings` as a
config detail without actually wiring it through `layer_model.forward`.
An explicit opt-in on the model class declares "I supply this
kwarg AND expect the runner to route it through a stable slot".
The pattern matches other SGLang capability flags — `supports_lora`
on `gemma3_causal.py:736`, `supports_torch_tp` on
`torch_native_llama.py:400`, `supports_fused_context_kv` on
`dflash.py:338` — all read by the runner via
`getattr(model, "supports_X", False)`.

**Q4. Why is the flag on Qwen3-VL rather than a generic multimodal
base class?**

There is no shared multimodal base class in SGLang that both
Qwen3-VL and its dense/MoE variants inherit from. The other BCG-
allowlisted archs (Cohere2Vision, KimiK25, MiniMaxM3) each
inherit from `nn.Module` directly. Placing the flag on
`Qwen3VLForConditionalGeneration` scopes it to precisely the
family that uses DeepStack; Qwen3-VL-Moe inherits it via
`Qwen3VLMoeForConditionalGeneration(Qwen3VLForConditionalGeneration)`.

**Q5. Why does Qwen3.5 inherit the flag if its current configuration
has no DeepStack?**

`Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`
so the flag propagates by construction. The runtime gate on
`num_deepstack_embeddings > 0` short-circuits allocation: every
shipped `Qwen/Qwen3.5-*` release ships
`deepstack_visual_indexes = []`, so `num_deepstack_embeddings = 0`,
so `deepstack_replay_width = 0`, so no slot is registered and no
buffer is allocated. The gate ensures zero cost today AND
auto-enables the fix should a future Qwen3.5 release populate
DeepStack — explicit `False` overrides would need manual updates.

**Q6. Why clear the buffer instead of simply skipping replay or
falling back to eager?**

Skipping replay or eager-fallback would defeat BCG entirely on
image requests. The clear-buffer approach lets BCG continue
serving while guaranteeing the captured `add_` reads a well-
defined value. Clearing is O(bucket-size) tensor zero — a
memset comparable to `input_embeds`' copy already present in the
same closure — not a graph rebuild.

**Q7. What happens when DeepStack shape changes between requests?**

Shape is `(num_tokens, hidden × num_ds)`. `hidden` and `num_ds`
are model constants set at load; `num_tokens` varies per request.
The slot is sized to `(max_num_tokens, hidden × num_ds)`. The
copy uses `slot[:de.shape[0]].copy_(de)` — automatically handles
the token dimension. The feature dimension is fixed per model;
`de.shape[1:] == slot.shape[1:]` guard in `replay_layer_forward`
catches any framework bug that changes it mid-run.

**Q8. Could zeroing the slot hide a framework bug instead of
surfacing it?**

For the intended production case (text-only request →
`input_deepstack_embeds is None`) zeroing is CORRECT behavior,
not a bug hider. For the failure case (shape/dtype/device
mismatch on an image request) zeroing degrades to a text-only-
equivalent output — worse than eager but not a crash and not
a data-corrupting outcome. An `assert` here would crash the
serving request; the current design keeps the server up while
producing a semantically defined output. Both choices are
defensible; the current one prioritizes availability. If
reviewers prefer stricter surfacing, replacing the else-branch
`slot.zero_()` with `assert False` (or a `RuntimeError`) is a
one-line change.

**Q9. Does this support video and multi-image requests?**

Yes — `general_mm_embed_routine` at `mm_utils.py:1108-1245`
allocates a single `(N_tokens, hidden × num_ds)` DeepStack tensor
that spans the request's total token count regardless of how many
images or videos contribute to the visual encoding. The runner
doesn't care about the source composition; it copies the flat
tensor into the slot.

**Q10. Is the extra buffer safe under tensor parallelism?**

Yes — `PrefillCudaGraphRunner` is instantiated per rank
(`model_runner.py` allocates one runner per worker), so the
buffer is per-rank. TP shards the feature axis across ranks;
`hidden_size` in this runner is already the rank-local hidden
dim, so `deepstack_replay_width = hidden × num_ds` is also
rank-local. No cross-rank buffer sharing.

**Q11. Can concurrent requests overwrite the same stable slot?**

No — the BCG runner processes prefills serially per rank; the
outer scheduler serialises prefill invocations on a single
runner. The slot is written inside `replay_layer_forward` and
read by the immediately-following `backend.replay(...)`. There
is no overlap window during which a second request could
overwrite the slot before the first replay consumes it.

**Q12. Why are 82 source lines (post-review-simplification) necessary?**

Four source hunks each carry irreducible content:
* `qwen3_vl.py`: +5 (flag declaration + 3-line comment).
* `buffers.py`: +11 (dataclass field + conditional allocation +
  kwarg passthrough).
* `cuda_graph_buffer_registry.py`: +14 (new kwarg + conditional
  slot).
* `prefill_cuda_graph_runner.py`: +50 (width computation, kwarg
  wiring in `__init__`, capture-pass kwargs unpacking, replay-
  side copy block).

The replay-side copy is the largest single hunk (~20 lines) and
carries the correctness-critical guards. Comments in each hunk
were trimmed during pre-PR review — nothing further to prune
without sacrificing readability.

**Q13. Can the patch be reduced to fewer files?**

Each file plays a distinct role:
* `qwen3_vl.py` — model capability declaration.
* `buffers.py` — dataclass field (registry adopts fields by
  attribute name; the registry-side `getattr(source, slot.name)`
  contract requires the field to exist on the source).
* `cuda_graph_buffer_registry.py` — slot registration.
* `prefill_cuda_graph_runner.py` — orchestration.

Collapsing any two would require rewriting an unrelated
abstraction. The dataclass field on `PrefillInputBuffers` is
necessary because `build_prefill_registry` currently raises
`ValueError: source is missing buffer 'input_deepstack_embeds'`
when adopting via `source=self.buffers` — the alternative would
be to widen the registry API to allow selective self-allocation,
which is a bigger change to shared infrastructure.

**Q14. How do we know the output difference is not normal BF16
nondeterminism?**

The eager-repeat noise floor is measured in each 4-arm run as
the `eager_repeat_noise` comparison (repeating an image request
under `eager_normal`): `prefix=15/True/l1_max_abs=0.000` in
every run. The pre-fix `bcg_normal_vs_eager_normal` divergence
is `l1_max_abs = 1.154` on scored tokens beyond the 7-token
common prefix — three orders of magnitude above the noise floor.
Post-fix `bcg_normal_vs_eager_normal` is `l1_max_abs = 0.071`,
comparable to the `bcg_zero_vs_eager_zero` control noise of
`0.066` and 15+ orders below the pre-fix divergence.

**Q15. Why is the RGB-stripe fixture sufficient for a correctness
regression?**

The RGB-stripe fixture is used **only** to trigger a deterministic
image-processing path — three saturated primary colors produce
distinct, high-confidence DeepStack contributions that cause a
clear divergence between the with-DeepStack and without-DeepStack
paths at the first non-boilerplate token. The correctness claim
is a **semantic equivalence** claim: *before the fix, BCG normal
reproduces zero-DeepStack behavior; after the fix, BCG normal
reproduces eager-normal behavior.* Nothing about visual quality
is claimed.

**Q16. Why are Qwen2.5-VL and text-only models unaffected?**

`Qwen2_5_VLForConditionalGeneration(nn.Module)` does not inherit
from `Qwen3VLForConditionalGeneration`. `Qwen3ForCausalLM(nn.Module)`
same. `getattr(model, "supports_bcg_deepstack_replay", False)`
returns `False` for both, so `deepstack_replay_width = 0`, so
no dataclass field, no slot, no capture kwarg, no replay copy.
Their `PrefillInputBuffers.create()` and `build_prefill_registry`
calls take the exact same code paths they took before the PR.
Regression protection: `test_qwen2_5_vl_does_not_declare_capability`
and `test_text_only_qwen3_does_not_declare_capability` (unit tests).

**Q17. What measurable memory or latency overhead does Qwen3-VL
gain?**

* **Memory**: one `(max_num_tokens, hidden × num_ds)` `bf16`
  buffer. On Qwen3-VL-8B at `max_num_tokens = 8192`:
  `8192 × 4096 × 3 × 2 = 192 MiB`. This is comparable to the
  existing `input_embeds` slot (`8192 × 4096 × 2 = 64 MiB` for
  the same bucket), so a ~3x increase for a feature that
  correctly needs 3x the width.
* **Latency**: one `slot[:de.shape[0]].copy_(de)` per replay,
  plus `slot[de.shape[0]:].zero_()` on the padded tail. The
  copy is bounded by the tensor size (~96 MiB for a 4096-token
  image on Qwen3-VL-8B); observed added replay latency in the
  M4b run was within noise of the pre-fix measurement (both
  under `bcg_normal_text` timings; not separately isolated in
  the current harness but bounded by memory bandwidth).

**Q18. Why should this be merged rather than disabling Qwen3-VL BCG?**

Qwen3-VL is **not currently on** the BCG allowlist upstream, so
"disabling" is the status quo. This PR is a defensive change
that lands the correctness infrastructure so that, if/when
Qwen3-VL is added to the allowlist (or a Qwen3.5 release ships
with populated DeepStack), the code path is already correct.
It does not force Qwen3-VL onto the allowlist — that is a
separate policy decision.

**Q19. Does the fix remain correct if upstream changes the number
of DeepStack layers?**

Yes. `num_deepstack_embeddings` is derived from
`self.visual.deepstack_visual_indexes` at model init
(qwen3_vl.py:1314). The runner reads it via `getattr`, computes
`deepstack_replay_width = hidden × num_ds`, allocates the buffer
+ slot at that width, and the LM's `add_` reads
`input_deepstack_embeds[:, sep:sep+hidden]` per layer index —
so any positive `num_ds` value works. The unit test
`test_slot_shape_and_dtype_match_contract` fires with `num_ds = 3`
(width 192 with hidden 64) to demonstrate the width computation.

**Q20. What protects against stale values in padded portions of
the slot?**

Two protections layered:
1. The valid-input branch explicitly zeros the padded tail:
   `if de.shape[0] < static_num_tokens: slot[de.shape[0]:].zero_()`.
2. The invalid-input branch zeros the whole slice:
   `else: slot.zero_()`.

For gate #4 (bucket reuse across image → text-only), an in-
process runtime test in `<scratchpad>/f_deepstack_fix_v2/…`
exercises A(image)→B(text-only, `de=None`)→C(smaller image)→
D(shape-mismatch)→E(dtype-mismatch); slot state is always
either `[copy][zero-pad]` or `[zero]*full_slice` — never carrying
stale content across requests. All 5 assertions pass.
