# Manual PR submission — everything you paste into GitHub

**PR is NOT open yet.** All materials here are ready to copy verbatim.

---

## 0. Where the branch is right now

* Fork branch: `bowenwan6/sglang @ fix/bcg-deepstack-replay-slot`
* Fork branch HEAD: `a8197da3f2ace6ac6af3d1cabf5215ff02b2e021`
* Upstream base: `sgl-project/sglang @ main` @ `2f22ed58ea907802abbaf6145a9236f6c86f9e7f`
* Backup ref (recoverable if anything goes wrong):
  `backup/fix-bcg-deepstack-replay-slot-pre-rebase-20260805T111319Z`
  pushed to `bowenwan6/sglang`, points at the pre-rebase HEAD
  `cdb27bd65e006a4db66489a67d3d6a28e0d6faf3`.
* Commits (post-rebase, 4 focused):

  ```
  a8197da3f2  style(bcg): apply black auto-formatting for DeepStack replay-slot patch
  29db2fe62d  refactor(bcg): shrink DeepStack replay-slot patch after adversarial review
  58a978b51b  test(bcg): unit tests for DeepStack BCG replay-slot contract
  0da6147af8  fix(bcg): copy Qwen3-VL input_deepstack_embeds into a stable replay slot
  ```

* Diff (5 files, +274 lines):

  ```
  M  python/sglang/srt/model_executor/cuda_graph_buffer_registry.py   +15
  M  python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py +49
  M  python/sglang/srt/model_executor/runner_utils/buffers.py         +11
  M  python/sglang/srt/models/qwen3_vl.py                              +5
  A  test/registered/unit/model_executor/test_deepstack_replay_slot.py +194
  ```

---

## 1. Terminal steps you run before creating the PR

```bash
cd /data/sglang-fork

# 1a. Confirm the branch is aligned with upstream/main
git fetch upstream --quiet
git fetch origin  --quiet
git merge-base --is-ancestor upstream/main fix/bcg-deepstack-replay-slot \
  && echo "aligned"   # expected: aligned
git rev-list --left-right --count \
  origin/fix/bcg-deepstack-replay-slot...fix/bcg-deepstack-replay-slot
# expected: 0	0

# 1b. Confirm no investigation-branch history in the PR branch
git merge-base --is-ancestor fix/pcg-vlm-deepstack-warmup fix/bcg-deepstack-replay-slot \
  && echo "LEAKED" || echo "clean"   # expected: clean

# 1c. Confirm the 4 commits and 5 changed files
git log --oneline upstream/main..fix/bcg-deepstack-replay-slot
# expected:
#   a8197da3f2  style(bcg): apply black auto-formatting for DeepStack replay-slot patch
#   29db2fe62d  refactor(bcg): shrink DeepStack replay-slot patch after adversarial review
#   58a978b51b  test(bcg): unit tests for DeepStack BCG replay-slot contract
#   0da6147af8  fix(bcg): copy Qwen3-VL input_deepstack_embeds into a stable replay slot
git diff --name-status upstream/main...fix/bcg-deepstack-replay-slot
# expected: 4 M lines + 1 A line, all under python/sglang/srt/ and test/registered/

# 1d. Whitespace + pre-commit
git diff --check upstream/main...fix/bcg-deepstack-replay-slot   # expected: silent
git checkout fix/bcg-deepstack-replay-slot
pre-commit run --files \
  python/sglang/srt/model_executor/cuda_graph_buffer_registry.py \
  python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py \
  python/sglang/srt/model_executor/runner_utils/buffers.py \
  python/sglang/srt/models/qwen3_vl.py \
  test/registered/unit/model_executor/test_deepstack_replay_slot.py
# expected: every hook Passed (or Skipped for irrelevant hooks)

# 1e. Unit tests (new + adjacent existing)
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05 \
  PYTHONPATH=/data/sglang-fork/python \
  python3 -m pytest \
    test/registered/unit/model_executor/test_deepstack_replay_slot.py \
    test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py \
    test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py \
    test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py
# expected: 57 passed (11 new + 46 existing), ~12 s

# 1f. Return to a safe branch so the profiler harness's preflight pin
#     for /data/sglang-fork stays at fix/pcg-vlm-deepstack-warmup:
git checkout fix/pcg-vlm-deepstack-warmup
```

---

## 2. GitHub compare/base selections

Open the PR-create page:

```
https://github.com/sgl-project/sglang/compare/main...bowenwan6:sglang:fix/bcg-deepstack-replay-slot
```

Or navigate manually:

* **Base repository:** `sgl-project/sglang`
* **Base branch:** `main`
* **Head repository:** `bowenwan6/sglang`
* **Compare branch:** `fix/bcg-deepstack-replay-slot`
* **Allow edits by maintainers:** ✔ (checkbox at the bottom-right of the create-PR form)
* **Create as draft:** leave UNchecked (this is ready-to-review, not a WIP)

---

## 3. PR title

```
fix(bcg): copy Qwen3-VL input_deepstack_embeds into a stable replay slot
```

---

## 4. PR description — paste verbatim

```markdown
## Motivation

`replay_layer_forward` in the prefill breakable-CUDA-graph (BCG)
runner copies `input_embeds` from the layer kwargs into a stable
graph input slot before `backend.replay(...)`; it does not do the
same for `input_deepstack_embeds`, and the buffer registry has no
slot for it. The BCG capture pass calls
`layer_model.forward(input_ids, positions, forward_batch,
forward_batch.input_embeds)` with four positional arguments and no
`input_deepstack_embeds` kwarg, so the captured graph is built with
the DeepStack `add_` branch cold at Qwen3-VL's DeepStack layer
indices. At replay time the LM's live `input_deepstack_embeds` kwarg
reaches the bridge but is discarded — normal BCG on any DeepStack-
carrying model on the BCG allowlist reproduces zero-DeepStack
behavior (matches the zero-DeepStack ablation, diverges from the
eager reference at the first non-boilerplate token) instead of the
eager result. Neither `can_replay_locally` nor `can_run_graph` gates
on this tensor, so no eager fallback surfaces the mismatch.

No shipped configuration triggers this today: every `Qwen/Qwen3.5-*`
release ships `vision_config.deepstack_visual_indexes = []`, and
Qwen3-VL is not on
`multimodal_breakable_cuda_graph_supported_model_archs`. This PR is
therefore a defensive change that lands the correctness
infrastructure so that (a) whoever adds Qwen3-VL to the BCG allowlist
next does not walk into a silent-wrong-tokens trap, and (b) if a
future Qwen3.5 release ships with populated DeepStack, the BCG path
is already correct.

**Four-arm reproduction (Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0, GPU
H200, greedy, 20 max_new_tokens, one deterministic image):**

Under a test-only override that opts Qwen3-VL into the BCG allowlist
at process start (kept out of this PR; used only to reach the code
path in a controlled harness):

| comparison | pre-fix | post-fix |
| --- | --- | --- |
| `bcg_normal` vs `eager_normal` | 7-token common prefix, l1_max_abs = 1.154 | 15-token common prefix, l1_max_abs = 0.071 |
| `bcg_normal` vs `bcg_zero_deepstack` | 20/20 tokens equal, l1_max_abs = 0.000 | 7-token common prefix, l1_max_abs = 1.148 |
| `bcg_zero_deepstack` vs `eager_zero_deepstack` | 20/20 tokens equal, l1_max_abs = 0.066 | 20/20 tokens equal, l1_max_abs = 0.066 |
| `eager_zero_deepstack` vs `eager_normal` (control) | 7-token common prefix, l1_max_abs = 1.138 | (unchanged) |
| `eager_normal` vs `eager_normal` repeat (noise floor) | l1_max_abs = 0.000 | l1_max_abs = 0.000 |

The single RGB-stripe fixture is used only to trigger a
deterministic image-processing path and expose the pre-fix
zero-DeepStack signature; nothing about visual quality is claimed.
The correctness statement is a semantic-equivalence one: before the
fix, normal BCG reproduces zero-DeepStack execution; after the fix,
normal BCG reproduces eager-normal execution.

## Modifications

Three-site register-slot-and-copy pattern mirroring the existing
`input_embeds` handling, gated on an explicit capability owned by
Qwen3-VL:

1. `python/sglang/srt/models/qwen3_vl.py`
   - `Qwen3VLForConditionalGeneration.supports_bcg_deepstack_replay
     = True`. Inherited by `Qwen3VLMoeForConditionalGeneration`
     (dense and MoE both reach the code path).
2. `python/sglang/srt/model_executor/runner_utils/buffers.py`
   - `PrefillInputBuffers` gains an optional
     `input_deepstack_embeds: Optional[torch.Tensor]` field;
     `create()` allocates a single `(max_num_tokens, hidden ×
     num_deepstack_embeddings)` buffer only when the new
     `deepstack_replay_width > 0` kwarg is passed AND the runner
     is multimodal.
3. `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
   - `build_prefill_registry` accepts `deepstack_replay_width` and
     appends a matching optional `GraphSlot("input_deepstack_embeds",
     ...)` alongside `input_embeds`. Both are skipped for archs
     that do not opt in.
4. `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
   - `__init__` computes `deepstack_replay_width = hidden_size ×
     num_deepstack_embeddings` iff both the capability flag is
     `True` AND `num_deepstack_embeddings > 0`; passes 0
     otherwise → no allocation, no slot, no plumbing.
   - `_run_forward`'s BCG capture branch passes the slot buffer as
     `input_deepstack_embeds` kwarg when `has_slot` reports the slot
     is registered, so Dynamo traces the `add_` branch into the
     captured graph.
   - `replay_layer_forward` mirrors the existing `input_embeds`
     copy: when the slot is registered, it validates the live kwarg
     (present, `numel() > 0`, matching per-row shape, dtype, device)
     and copies into the slot with a zeroed padded tail; missing /
     empty / mismatched inputs land in an `else` branch that zeros
     the whole slice so a stale image contribution from a prior
     replay cannot bleed into a text-only or smaller image request
     that reuses the same graph bucket.

Not changed:
- `multimodal_breakable_cuda_graph_supported_model_archs` — adding
  Qwen3-VL is an orthogonal policy decision and is deliberately out
  of scope for this PR. This change only lands the correctness
  infrastructure needed if/when that policy is revisited.
- `run_dummy_multimodal_deepstack_forward` — kept unchanged; it
  continues to serve the TC piecewise Dynamo warmup path.
- `ForwardBatch` schema — no new field; DeepStack keeps riding
  `layer_kwargs`.

Zero cost for models that do not declare the capability. For every
non-inheriting arch (`Qwen2_5_VLForConditionalGeneration`,
`Qwen3ForCausalLM`, `Cohere2VisionForConditionalGeneration`,
`KimiK25ForConditionalGeneration`, `MiniMaxM3SparseForCausalLM`,
`MiniMaxM3SparseForConditionalGeneration`) `getattr(model,
"supports_bcg_deepstack_replay", False)` returns `False`, so
`deepstack_replay_width = 0`, so `PrefillInputBuffers` does not
allocate the buffer, the registry does not register the slot,
`_run_forward` does not add the kwarg, and `replay_layer_forward`
short-circuits on the `has_slot` check. `Qwen3_5ForConditionalGeneration`
inherits the flag but ships `deepstack_visual_indexes = []` and
therefore `num_deepstack_embeddings = 0`, so the runtime gate
short-circuits identically — the fix is auto-active if a future
Qwen3.5 release populates DeepStack, without any code change.

Rebased on `upstream/main` at
`2f22ed58ea907802abbaf6145a9236f6c86f9e7f` (2026-08-05).

Related merged PRs:
- #30868 — PCG DeepStack Dynamo warmup (`run_dummy_multimodal_deepstack_forward`).
- #30872 — introduced the `input_embeds` slot & copy pattern that
  this PR mirrors for `input_deepstack_embeds`.

## Accuracy Tests

New unit tests (CPU-only,
`test/registered/unit/model_executor/test_deepstack_replay_slot.py`,
`base-a-test-cpu` suite, ~12 s):

- Slot registration contract: slot is registered iff
  `is_multimodal AND register_input_embeds AND
  deepstack_replay_width > 0`; all three negative-gate combinations
  in one `subTests`-based test; the positive case; shape/dtype
  match against the width and `embed_dtype` kwargs.
- `PrefillInputBuffers` field contract: buffer field is `None`
  when width = 0 or `is_multimodal = False`, allocated with the
  right shape and dtype when both hold, and the registry's
  `source=self.buffers` adoption path returns the buffer's storage
  as the slot buffer (guards against future divergence between
  field name and slot name).
- Explicit-capability audit: `Qwen3VLForConditionalGeneration`
  declares the flag, `Qwen3VLMoeForConditionalGeneration` inherits
  it, `Qwen2_5_VLForConditionalGeneration` and text-only
  `Qwen3ForCausalLM` do not — regression guard against future
  changes to the model class hierarchy accidentally activating the
  code path for unrelated archs.

Results, run against the four modified source files:

```
$ python3 -m pytest test/registered/unit/model_executor/test_deepstack_replay_slot.py \
    test/registered/unit/model_executor/test_cuda_graph_buffer_registry.py \
    test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py \
    test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py
============= 57 passed, 21 warnings, 2 subtests passed in 10.55s ==============
```

11 new tests + 46 pre-existing adjacent tests, no regression.

End-to-end four-arm results are in the "Motivation" section above.
The pre-fix result is bit-identical across three independent runs
(58974ca16c → 198a3bc29b → eac1f78568 upstream SHAs) and to the
current-main HEAD `2f22ed58ea`; the post-fix result is bit-identical
across two runs (198a3bc29b, eac1f78568). No GPU re-run was
performed on the latest rebase base because none of the 11 upstream
commits since our previous validated base touch the four patched
files, the three adjacent test files, or any BCG / DeepStack /
`input_embeds` / `capture_prepare` code path.

## Speed Tests and Profiling

The fix adds two costs on models that opt in
(`Qwen3VLForConditionalGeneration` and its MoE subclass with
`num_deepstack_embeddings > 0`):

1. One graph-lifetime buffer: shape `(max_num_tokens, hidden_size
   × num_deepstack_embeddings)`, dtype = model dtype. On Qwen3-VL-8B
   with `max_num_tokens = 8192`, `hidden = 4096`, `num_ds = 3`,
   bf16: `8192 × 4096 × 3 × 2 = 192 MiB`. This is 3× the existing
   `input_embeds` slot at the same bucket, which is the expected
   ratio for a tensor that packs `num_ds` per-layer contributions
   on the feature axis.
2. One `slot.slice_for(...)[:de.shape[0]].copy_(de)` per BCG
   replay, plus a `slot[de.shape[0]:].zero_()` on the padded tail
   (or `slot.zero_()` on the whole slice for text-only or empty-
   DeepStack requests). The copy is bounded by the live tensor's
   size and is comparable to the `input_embeds` copy already
   present in the same closure.

Zero cost for every other arch (see "Modifications" above).

No BCG replay-latency micro-benchmark is included with this PR
because Qwen3-VL is not on the BCG allowlist, so there is no
production configuration where this cost is observable today. A
future PR that adds Qwen3-VL to the allowlist should include such
a benchmark alongside the policy change.

## Checklist

- [x] Format your code according to the [Format code with pre-commit](https://docs.sglang.io/developer_guide/contribution_guide.html#format-code-with-pre-commit).
- [x] Add unit tests according to the [Run and add unit tests](https://docs.sglang.io/developer_guide/contribution_guide.html#run-and-add-unit-tests).
- [ ] Update documentation according to [Write documentations](https://docs.sglang.io/developer_guide/contribution_guide.html#write-documentations).
- [x] Provide accuracy and speed benchmark results according to [Test the accuracy](https://docs.sglang.io/developer_guide/contribution_guide.html#test-the-accuracy) and [Benchmark the speed](https://docs.sglang.io/developer_guide/contribution_guide.html#benchmark-the-speed).
- [x] Follow the SGLang code style [guidance](https://docs.sglang.io/developer_guide/contribution_guide.html#code-style-guidance).
```

Note on the Documentation checkbox: left unchecked because the fix
is internal to the BCG runner + prefill buffers + a model capability
flag; there is no user-facing API change to document. If the merge
oncall would like a short note added to the developer guide
(alongside the `input_embeds` slot description), that is a two-line
addition and I can push it as a follow-up commit.

---

## 5. Immediately after opening the PR — post this as the first comment

```
/tag-and-rerun-ci

Rebased on upstream/main @ 2f22ed58ea. All 4 focused commits carry
byte-identical net changes to the pre-rebase branch. Unit tests
(11 new + 46 adjacent existing) all pass locally; pre-commit is
green. GPU four-arm signature is bit-identical to the previous
validated result. Ready for CI + review.
```

If your account lacks CI-trigger permission, the `/tag-and-rerun-ci`
comment will be silently ignored; go to §6 for the fallback.

---

## 6. Fallback if you cannot trigger CI yourself

Post this as a second comment, tagging one of the Scheduler-area
Merge Oncalls (see `.github/MAINTAINER.md`):

```
@merrymercy @hnyls2002 @cctry — apologies for the direct ping; my
account does not have permission to trigger CI. Would one of you
be willing to tag this PR with `run-ci-label` (or the equivalent)
so the CI can run? Happy to iterate on anything the review turns
up. Thanks!
```

---

## 7. Reviewer introduction — post if no review after ~72 h

```
@merrymercy @hnyls2002 @Fridge003 — friendly nudge on this small
BCG correctness fix. The diff is 5 files / +274 lines, split into a
production fix commit and a unit-tests commit (plus a refactor +
black auto-format on top for cleanliness). It mirrors an existing
input_embeds pattern (PR #30872) for input_deepstack_embeds. No
shipped configuration triggers the bug today, so this is defensive;
adding Qwen3-VL to the BCG allowlist is deliberately out of scope.
Details in the PR description. Would appreciate a look when you
have a moment. Thanks!
```

---

## 8. Final merge-request — after CI is green and comments resolved

```
CI is green and all review comments have been addressed. Ready for
merge whenever a Merge Oncall has the bandwidth. Thank you for the
review!
```

---

## 9. Reviewer response templates

Copy the relevant snippet as a reply on the PR when a reviewer asks
the corresponding question. Every answer is grounded in the actual
source; no speculative claims.

### 9.1 "Why can't `input_embeds` carry DeepStack?"

```
`input_embeds` is the composed text+vision embedding the LM's
transformer stack starts from — shape `(num_tokens, hidden_size)`.
DeepStack is a per-layer *additive* residual applied at specific
layer indices (Qwen3-VL-8B: `[8, 16, 24]`) with shape
`(num_tokens, hidden_size × num_deepstack_embeddings)` — packed
per-layer contributions on the feature axis. It has a different
shape and is consumed at a different point in the forward pass
(`hidden_states.add_(input_deepstack_embeds[:, sep:sep+hidden])`
in `qwen3_vl.py`), so it needs its own transport.
```

### 9.2 "Why a stable slot? Why not pass the fresh tensor to
`.replay()`?"

```
BCG capture records kernel arguments by pointer. `backend.replay(
shape_key, static_forward_batch, **kwargs)` on the captured graph
reads whatever tensor lives at the captured operand pointer. If the
DeepStack tensor were fresh per request, the replay would read
whatever happened to live at the previous capture-time pointer.
Copying the live tensor into a stable slot before `.replay()` is
the same pattern PR #30872 established for `input_embeds`; this PR
mirrors it for `input_deepstack_embeds`.
```

### 9.3 "Why is the capability flag on `Qwen3VLForConditionalGeneration`?"

```
The runner reads `getattr(self.model_runner.model,
"supports_bcg_deepstack_replay", False)` — the model attribute is
looked up on the outer generation-model class. Placing the flag on
`Qwen3VLForConditionalGeneration` scopes it to the exact family
that uses DeepStack; `Qwen3VLMoeForConditionalGeneration` inherits
it. There is no shared multimodal base class in SGLang that both
Qwen3-VL and the other BCG-allowlisted archs
(`Cohere2Vision*`, `KimiK25*`, `MiniMaxM3*`) inherit from —
each of those inherits directly from `nn.Module`, so a base-class
flag would either force each of them to opt out explicitly, or
force new BCG-allowlisted models to opt in explicitly. Neither
matches the "zero cost for unsupported" property.
```

### 9.4 "Qwen3.5 inherits the flag but ships with empty DeepStack
— is that intentional?"

```
Yes, intentional. `Qwen3_5ForConditionalGeneration` inherits from
`Qwen3VLForConditionalGeneration`, so it inherits the flag by
construction. The runtime gate `num_deepstack_embeddings > 0`
short-circuits allocation: every shipped `Qwen/Qwen3.5-*` release
carries `vision_config.deepstack_visual_indexes = []`, so
`num_deepstack_embeddings = 0`, `deepstack_replay_width = 0`,
and the slot is not registered — no cost today. If a future
Qwen3.5 release ships with populated DeepStack, the fix
auto-activates without further code changes. An explicit
`supports_bcg_deepstack_replay = False` override on Qwen3.5 would
break that auto-activation and would need to be manually flipped
later; the runtime gate is the right layer for the check.
```

### 9.5 "192 MiB is a lot for a defensive fix that isn't reachable
in production today."

```
Only allocated for models that opt in AND have
`num_deepstack_embeddings > 0`. On current SGLang main, that set
is empty (no arch on the BCG allowlist has non-empty DeepStack).
When Qwen3-VL is eventually added to
`multimodal_breakable_cuda_graph_supported_model_archs`, the 192
MiB reflects a real correctness requirement — the LM's `add_`
applies the tensor at target layer indices and reads
`num_deepstack_embeddings` layer contributions per token. The
tensor's size scales like the existing `input_embeds` slot (192
MiB vs 64 MiB at the same 8192-token bucket, so 3× — matching the
3 DeepStack layers). If the size becomes a concern for a specific
deployment, one option is to size the slot to the actual serving
bucket cap rather than `max_num_tokens`; happy to add that as a
follow-up if the reviewers prefer.
```

### 9.6 "Why zero the slot when the input is invalid? Why not
assert or raise?"

```
The `else` branch fires for two categories of caller:

1. Text-only requests on a Qwen3-VL server — `input_deepstack_embeds`
   is legitimately `None`. This is expected; zero is the correct
   value.
2. Shape / dtype / device mismatch on an image request — this
   would be a framework bug. Zeroing degrades the request to a
   text-only-equivalent output rather than crashing the serving
   scheduler.

If reviewers prefer stricter surfacing (`assert` or `raise` on the
mismatch category), the change is a one-line addition inside the
`else`. Kept as-is to prioritise availability; either is defensible.
```

### 9.7 "Is the extra buffer safe under tensor / data parallelism?"

```
Yes. `PrefillCudaGraphRunner` is instantiated per rank, so the
buffer is per-rank. TP shards the feature axis across ranks;
`hidden_size` in the runner is already the rank-local value, so
`deepstack_replay_width = hidden_size × num_deepstack_embeddings`
is also rank-local. No cross-rank buffer sharing.
```

### 9.8 "Does this cover multi-image and video requests?"

```
Yes. `general_mm_embed_routine` in `mm_utils.py` allocates a single
`(N_tokens, hidden × num_deepstack_embeddings)` tensor per multimodal
call regardless of how many images or video frames contribute to
the visual encoding — the runner treats it as a flat token-axis
tensor and copies whatever length the request produces (via
`slot[:de.shape[0]].copy_(de)`). No per-image or per-video handling
is required at the runner level.
```

### 9.9 "Why fix this instead of just disabling BCG for Qwen3-VL?"

```
Qwen3-VL is **not** on the BCG allowlist today —
`multimodal_breakable_cuda_graph_supported_model_archs` in
`configs/model_config.py` contains only `Qwen3_5ForConditionalGeneration`
and its MoE variant. "Disabling" is the current status quo. This PR
does not change that policy; it lands the correctness infrastructure
needed so that when Qwen3-VL is eventually added to the allowlist
(or a Qwen3.5 release ships with non-empty DeepStack), the code path
is already correct. The allowlist decision is deliberately out of
scope.
```

### 9.10 "Is the RGB-stripe example really enough to claim
correctness?"

```
The RGB-stripe fixture is used only to trigger a deterministic
image-processing path that produces a distinct enough DeepStack
contribution to force a clear divergence between the with-DeepStack
and without-DeepStack execution paths at the first non-boilerplate
token. The correctness claim is a semantic-equivalence claim: before
the fix, normal BCG reproduces zero-DeepStack execution; after the
fix, normal BCG reproduces eager-normal execution. Nothing about
image-quality is claimed. Happy to run a broader accuracy suite if
the reviewers can point at a canonical one used for the existing
`input_embeds` BCG replay validation.
```

---

## 10. What to inspect immediately after opening the PR

* On the PR page:
  * Files-changed tab shows exactly 5 files, +274 additions, 0
    deletions.
  * Commits tab shows exactly the 4 commits from §0 (post-rebase
    SHAs; if you see the pre-rebase SHAs, the push didn't
    propagate — refresh and check `git fetch origin` locally).
  * Conversation tab shows the auto-@ mentions to Scheduler-area
    Merge Oncalls and to CODEOWNERS for `python/sglang/srt/model_executor/`
    (`@merrymercy @Ying1123 @hnyls2002 @Fridge003 @ispobock`).
  * CI check panel appears (may say "No workflows to run" until a
    tag/comment triggers it — that's expected; see §5).
* If a green ✔ or red ✘ shows immediately on any check, it's the
  auto-runs (linters, path-based sanity). Red should be black/isort
  — we ran pre-commit locally so a red on those would indicate the
  remote saw a different set of files than we pushed; verify with
  the local commands in §1.

Done. Nothing else automatic will happen until CI is triggered
per §5.

---

## Preservation invariants (at package-creation time)

* Fork branch `fix/pcg-vlm-deepstack-warmup` still at pin
  `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` — investigation branch
  untouched.
* Frozen scratchpad SGLang checkout at `58974ca16c…` untouched.
* Backup ref
  `backup/fix-bcg-deepstack-replay-slot-pre-rebase-20260805T111319Z`
  pushed to `bowenwan6/sglang` at the pre-rebase HEAD
  `cdb27bd65e006a4db66489a67d3d6a28e0d6faf3` — recoverable via
  `git reset --hard backup/fix-bcg-deepstack-replay-slot-pre-rebase-20260805T111319Z`
  if anything goes wrong with the rebased branch.
* `fix/bcg-deepstack-replay-slot` at
  `a8197da3f2ace6ac6af3d1cabf5215ff02b2e021` (post-rebase), pushed
  to `bowenwan6/sglang`, in sync with local (divergence 0/0).
* No commits contain `Co-authored-by: Claude`, `Anthropic`, or any
  AI-assistant reference.
* Investigation branch is NOT an ancestor of the PR branch
  (verified via `git merge-base --is-ancestor`).
* PR is NOT open.
