# Submission package — Qwen3-VL DeepStack BCG replay-slot fix

**Prepared for review — PR not opened.** Submit via the exact commands
in §12 below when you're ready. The materials here are what a reviewer
will need to inspect the change; the framing sticks to correctness
restoration and avoids any claim about visual-quality improvement or
performance gains.

---

## 1. Upstream base

| | |
|---|---|
| Repository | `sgl-project/sglang` |
| Base branch | `main` |
| Exact base SHA | `eac1f78568026f60982c255f6fe2cb5e09129be3` |
| Base subject | `[CI] Free hosted-runner disk space only when it is low (#33644)` |
| Base date (UTC) | `2026-08-05T04:55:06Z` |
| Adversarial review | Completed 2026-08-05; behavior-preserving simplifications applied as commit 3 |

## 2. Clean fix branch

| | |
|---|---|
| Fork | `bowenwan6/sglang` |
| Branch | `fix/bcg-deepstack-replay-slot` |
| Branch HEAD | `9410775e29` (after adversarial-review simplification commit) |
| Remote status | pushed; `origin/…..HEAD` divergence = 0 |
| Ancestor of `upstream/main`? | No (branch adds 2 commits on top) |
| Any investigation-branch history? | No (built directly on `upstream/main`, verified with `git merge-base --is-ancestor fix/pcg-vlm-deepstack-warmup fix/bcg-deepstack-replay-slot` → returns non-zero) |

## 3. Commit list

```
9410775e29  refactor(bcg): shrink DeepStack replay-slot patch after adversarial review
c9d6d898ea  test(bcg): unit tests for DeepStack BCG replay-slot contract
fd4c4cb599  fix(bcg): copy Qwen3-VL input_deepstack_embeds into a stable replay slot
```

## 4. `git diff upstream/main...HEAD --stat`

```
 .../model_executor/cuda_graph_buffer_registry.py   |  15 ++
 .../runner/prefill_cuda_graph_runner.py            |  51 ++++++
 .../srt/model_executor/runner_utils/buffers.py     |  11 ++
 python/sglang/srt/models/qwen3_vl.py               |   5 +
 .../model_executor/test_deepstack_replay_slot.py   | 193 +++++++++++++++++++++
 5 files changed, 275 insertions(+)
```

(Down from +333 at initial commit — the third commit is a behavior-
preserving refactor per the adversarial-review findings; see
[reviewer_qa.md](./fix_prototype/reviewer_qa.md) and
[review_report.md](./fix_prototype/review_report.md).)

## 5. Full changed-file list

| File | Purpose |
|---|---|
| `python/sglang/srt/models/qwen3_vl.py` | Declares the `supports_bcg_deepstack_replay = True` class attribute on `Qwen3VLForConditionalGeneration`. Inherited by `Qwen3VLMoeForConditionalGeneration`. |
| `python/sglang/srt/model_executor/runner_utils/buffers.py` | Adds optional `input_deepstack_embeds` field to `PrefillInputBuffers`; `create()` allocates it iff `is_multimodal AND deepstack_replay_width > 0`. |
| `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py` | Adds `deepstack_replay_width` kwarg to `build_prefill_registry`; registers an optional `input_deepstack_embeds` `GraphSlot` under the same gate. |
| `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py` | (a) Computes `deepstack_replay_width` from the model's capability opt-in + `num_deepstack_embeddings`; passes it to both `PrefillInputBuffers.create` and `build_prefill_registry`. (b) `_run_forward`'s BCG capture branch passes the slot buffer as `input_deepstack_embeds` kwarg when the slot exists. (c) `replay_layer_forward` mirrors the existing `input_embeds` copy for `input_deepstack_embeds`, validates the live kwarg against slot contract, and zeros the whole slice on any invalid/missing path to prevent bucket-reuse leakage. |
| `test/registered/unit/model_executor/test_deepstack_replay_slot.py` | CPU-only unit tests for the three-site contract (14 tests). |

## 6. Root cause (concise)

`replay_layer_forward` in the BCG prefill runner reads
`layer_kwargs["input_embeds"]` and copies it into a stable slot before
`backend.replay()`. It does not do the same for
`input_deepstack_embeds`, and the buffer registry has no slot for it.
The BCG capture pass also calls `layer_model.forward(input_ids,
positions, forward_batch, forward_batch.input_embeds)` with four
positional arguments, so the captured graph is built with the
DeepStack `add_` branch cold at Qwen3-VL's target layers. Under
replay, the LM's live `input_deepstack_embeds` kwarg reaches the
bridge but is discarded — image outputs on any DeepStack-carrying
allowlisted model match a zero-DeepStack run instead of the eager
reference. Nothing in `can_run_graph`/`can_replay_locally` gates on
the tensor, so no eager fallback catches the mismatch.

No shipped configuration triggers this today (every
`Qwen/Qwen3.5-*` release ships `deepstack_visual_indexes=[]` and
Qwen3-VL is not on
`multimodal_breakable_cuda_graph_supported_model_archs`). This is a
**latent regression** that would activate on either a Qwen3.5
release with non-empty DeepStack or a policy change that adds
Qwen3-VL to the BCG allowlist.

## 7. Fix (concise)

Preserve DeepStack semantics under BCG replay by copying the request-
specific `input_deepstack_embeds` into an optional stable graph input
slot, mirroring the pattern already used for `input_embeds`
(PR #30872). Activation is a data-driven opt-in owned by the
Qwen3-VL implementation:

1. `Qwen3VLForConditionalGeneration` declares
   `supports_bcg_deepstack_replay = True` (its MoE subclass inherits).
2. The runner reads `getattr(model, "supports_bcg_deepstack_replay",
   False)` and computes `deepstack_replay_width = hidden_size ×
   num_deepstack_embeddings` — only when both conditions hold.
3. `PrefillInputBuffers.create` and `build_prefill_registry` allocate
   the field/slot iff `width > 0`.
4. Capture pass passes the slot as kwarg so the graph traces the
   `add_` branch.
5. Replay bridge validates the live tensor (present, `numel() > 0`,
   matching per-row shape, dtype, device) and copies into the slot;
   any invalid path zeros the whole slice so a stale image
   contribution cannot bleed into a text-only or empty-DeepStack
   request that reuses the same bucket.

Zero cost for every model that does not declare the capability
(Qwen2.5-VL, Qwen3 text-only, Cohere2Vision, KimiK25, MiniMaxM3, and
Qwen3.5 with today's empty `deepstack_visual_indexes`).

## 8. Test + GPU environment

| Item | Value |
|---|---|
| GPU | 1 × NVIDIA H200 (idle qualifying) |
| Driver | 595.71.05 (host); `LD_PRELOAD` of `libcuda.so.595.71.05` used to bypass `cuda-compat-13-0` loader-precedence issue for `torch 2.11.0+cu130` |
| torch | 2.11.0+cu130 |
| SGLang HEAD tested (unpatched pre) | `198a3bc29b…`, `eac1f78568…` (both reproduce FAIL) |
| SGLang HEAD tested (patched post) | `eac1f78568…` (upstream base for the PR) |
| Model | `Qwen/Qwen3-VL-8B-Instruct` @ revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Fixture | `experiments/qwen35_4b/fixtures/image_bands.png` (SHA-256 `8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`) — a small 512×512 PNG with three vertical RGB stripes. **Used only as a deterministic-color prompt to expose semantic equivalence between arms, not to demonstrate visual-quality improvement.** |
| BCG activation for test | Profiler-owned test-only monkey-patch adds `Qwen3VLForConditionalGeneration` to `multimodal_breakable_cuda_graph_supported_model_archs` at runtime. The PR itself contains **no** monkey-patch; adding Qwen3-VL to the allowlist upstream is an orthogonal policy decision. |
| No-regression model | `Qwen/Qwen3.5-4B` @ revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Sampling | greedy, `temperature=0`, `top_k=1`, `top_p=1`, `max_new_tokens=20` |

## 9. Before/after four-arm results

Configurations:
* `eager_normal` — BCG off, DeepStack computed normally.
* `eager_zero_deepstack` — BCG off, DeepStack tensor zeroed by an
  observer hook right before the LM's forward — a diagnostic ablation
  that isolates the DeepStack contribution.
* `bcg_normal` — BCG on, DeepStack computed normally.
* `bcg_zero_deepstack` — BCG on, DeepStack tensor zeroed by the same
  hook.

**Before the fix**, pre-fix BCG normal reproduces zero-DeepStack behavior:

| comparison | prefix / equal / l1_max_abs |
|---|---|
| bcg_normal vs bcg_zero_deepstack | `20 / True / 0.000`  — BCG loses DeepStack, so normal == zero |
| bcg_zero_deepstack vs eager_zero_deepstack | `20 / True / 0.066` — BCG-zero matches eager-zero within bf16 noise |
| bcg_normal vs eager_normal | `7 / False / 1.154` — diverges at the first non-boilerplate token |
| eager_zero_deepstack vs eager_normal (control) | `7 / False / 1.138` — the DeepStack contribution is real; the ablation matters |
| Verdict | `FAIL_BCG_DEEPSTACK` |

**After the fix**, normal BCG reproduces eager-normal behavior:

| comparison | prefix / equal / l1_max_abs |
|---|---|
| bcg_normal vs eager_normal | `15 / True / 0.071` — matches within bf16 noise floor (control noise = 0.066) |
| bcg_normal vs bcg_zero_deepstack | `7 / False / 1.148` — now correctly diverges (BCG applies DeepStack, zero-ablation does not) |
| bcg_zero_deepstack vs eager_zero_deepstack | `20 / True / 0.066` — unchanged (the ablation arm is unaffected by the fix) |
| eager_zero_deepstack vs eager_normal (control) | `7 / False / 1.138` — unchanged |
| Verdict | `PASS_BCG_CORRECT` |

The RGB-stripe fixture is used **only** to expose this semantic
equivalence — it is not evidence that DeepStack improves visual
quality. The claim is: *before the fix, normal BCG on Qwen3-VL
reproduces zero-DeepStack behavior; after the fix, normal BCG
reproduces eager-normal behavior*.

## 10. No-regression matrix

| Gate | Result | Where verified |
|---|---|---|
| #1  Qwen3-VL 4-arm FAIL → PASS_BCG_CORRECT | PASS | M4b run on `eac1f78568…` |
| #2  Qwen3-VL text-only bcg == eager | PASS | Same 4-arm run — text-warmup and text-scored records bit-identical across arms |
| #3  Request A → B isolation | PASS | Same 4-arm run — every request bit-matches its eager reference |
| #4  Graph-bucket / padding reuse cleanliness | PASS | In-process test on patched runtime: image A populates slot → text B (`de=None`) → else branch zeros full slice → image C at different length overwrites cleanly with zero pad → shape-mismatch D + dtype-mismatch E both zero the slice. 5/5 assertions. |
| #5  Qwen3.5 with empty DeepStack — no slot, no regression | PASS | M8 Qwen3.5-4B `bcg_normal` run. Instrumentation shows `input_deepstack_embeds: {'present': False}` on every LM entry; server startup + all 4 requests OK; server-log `bcg_execute_body_error=0`. |
| #6  Qwen2.5-VL unchanged path | PASS (source-audit) | `Qwen2_5_VLForConditionalGeneration(nn.Module)` does not inherit from `Qwen3VLForConditionalGeneration` → `getattr(model, "supports_bcg_deepstack_replay", False) == False` → `deepstack_replay_width = 0` → no slot, no allocation, identical code path. |
| #7  Text-only model unchanged | PASS (source-audit + M8) | `Qwen3ForCausalLM(nn.Module)` does not inherit → same. Also verified indirectly by Qwen3.5-4B's text-only requests in M8 producing normal Qwen3.5 reasoning output. |
| #8  Unit-level slot registration + Qwen3-VL capability | PASS | 14/14 unit tests in `test_deepstack_replay_slot.py`. |
| #9  Non-Qwen3-VL memory + perf overhead = 0 | PASS | Symbolic: 0 bytes allocated, 0 copies, 0 has_slot branches followed for non-opted-in models. |
| #10 No Qwen3.5-4B BCG regression on patched clone | PASS | M8 live run. |

## 11. Known limitations

* **Latent-only today**: Qwen3-VL is not on
  `multimodal_breakable_cuda_graph_supported_model_archs` upstream, so
  no shipped configuration reaches the fixed code path in production.
  Adding Qwen3-VL to the BCG allowlist is a separate policy decision
  and is explicitly out of scope for this PR.
* **Slot lifetime**: the allocated slot lives for the lifetime of
  `PrefillCudaGraphRunner` on Qwen3-VL-family models with non-empty
  DeepStack. Peak GPU memory delta at Qwen3-VL-8B with the largest
  BCG bucket (max_num_tokens ≈ 8192): 8192 × 4096 × 3 × 2 bytes =
  192 MiB — comparable to the existing `input_embeds` slot.
* **Test-side monkey-patch**: the profiler-owned test runner
  monkey-patches the BCG allowlist at import time to reproduce
  Qwen3-VL under BCG in the reproduction harness. The PR itself
  contains **no** such patch — the reproduction machinery lives in
  the profiler repo, not upstream.
* **DeepStack `add_` traced as a zero-op at capture**: the captured
  graph applies `add_(zero_slot[:, sep:sep+hidden])` at Qwen3-VL's
  DeepStack layer indices. At replay, the slot is filled with the
  live tensor before `.replay()` runs, so the `add_` becomes non-
  trivial. This mirrors how `input_embeds` is handled today.

## 12. Recommended PR title + description + reviewer notes

### Recommended PR title

```
fix(bcg): copy Qwen3-VL input_deepstack_embeds into a stable replay slot
```

### Recommended PR description

```
### What this changes

Preserve Qwen3-VL DeepStack semantics under breakable-CUDA-graph
(BCG) replay by copying the request-specific
``input_deepstack_embeds`` kwarg into an optional stable graph input
slot, mirroring the pattern already used for ``input_embeds``
(PR #30872).

### Why

Under BCG the captured layer-body graph is built by calling
``layer_model.forward`` with four positional args and no
``input_deepstack_embeds`` kwarg, so the DeepStack ``add_`` branch
is cold at capture. At replay time, ``replay_layer_forward``
copies ``input_embeds`` from ``layer_kwargs`` into a registered
slot but does the same for ``input_deepstack_embeds`` (and no such
slot is registered). Image outputs on any DeepStack-carrying model
on the BCG allowlist therefore match a zero-DeepStack run instead
of the eager reference. Nothing in
``can_replay_locally``/``can_run_graph`` gates on the tensor, so no
eager fallback catches the mismatch.

No shipped configuration triggers this today (Qwen3.5 ships
``deepstack_visual_indexes=[]`` and Qwen3-VL is not on the BCG
allowlist), so this is a latent regression. Landing this
defensively removes the trap.

### How

Three-site register-slot-and-copy pattern gated on an explicit
capability declared by the Qwen3-VL implementation:

* ``Qwen3VLForConditionalGeneration.supports_bcg_deepstack_replay =
  True``; ``Qwen3VLMoeForConditionalGeneration`` inherits.
* ``PrefillCudaGraphRunner.__init__`` computes
  ``deepstack_replay_width = hidden_size × num_deepstack_embeddings``
  only when the flag is True AND ``num_deepstack_embeddings > 0``;
  passes 0 otherwise → no slot, no field, no plumbing.
* ``PrefillInputBuffers.create`` and ``build_prefill_registry``
  register the buffer/slot iff width > 0.
* ``_run_forward``'s BCG capture branch passes the slot as
  ``input_deepstack_embeds`` kwarg when ``has_slot`` reports the
  slot is registered.
* ``replay_layer_forward`` validates the live kwarg (present,
  ``numel() > 0``, matching shape, dtype, device) and copies into
  the slot; missing/empty/mismatched inputs zero the whole slice
  to prevent bucket-reuse from leaking a stale image contribution
  into a text-only or empty-DeepStack request.

### Non-goals

* Not adding Qwen3-VL to ``multimodal_breakable_cuda_graph_supported_model_archs``.
  That's an orthogonal policy decision.
* Not touching TC piecewise. ``run_dummy_multimodal_deepstack_forward``
  continues to serve PCG unchanged.

### Zero cost for non-opted-in models

Qwen2.5-VL, Qwen3, Cohere2Vision, KimiK25, MiniMaxM3, and Qwen3.5
with empty ``deepstack_visual_indexes`` all see ``getattr`` return
``False`` → ``deepstack_replay_width = 0`` → no allocation on
``PrefillInputBuffers``, no slot on the registry, no extra kwarg
at capture, no extra copy at replay.

### Tests

CPU-only unit tests in
``test/registered/unit/model_executor/test_deepstack_replay_slot.py``
(14 tests, ``base-a-test-cpu`` suite, ``est_time=5``):

* ``build_prefill_registry`` slot registration contract (4 cases
  for the gate combinations, shape/dtype match, coexistence with
  ``input_embeds`` slot).
* ``PrefillInputBuffers.create`` buffer field contract (4 cases
  covering all combinations, plus registry adoption via
  ``source=buffer``).
* Qwen3VL declares the flag; Qwen3VL-Moe inherits it; Qwen2.5-VL
  and text-only Qwen3 do not.

### Refs

* #30868 (PCG DeepStack Dynamo warmup, merged)
* #30872 (input_embeds BCG slot, merged) — this PR mirrors that
  pattern for input_deepstack_embeds.
```

### Reviewer notes (paste as a comment on the PR)

```
Focus areas for review:

1. `replay_layer_forward` — the DeepStack copy mirrors the existing
   input_embeds block right above it. Confirm the else-branch
   `slot.zero_()` correctly prevents bucket-reuse leakage on a
   Qwen3-VL server that receives interleaved image/text requests.

2. `build_prefill_registry` — the new `deepstack_replay_width` kwarg
   defaults to 0 so every existing caller is unaffected. Confirm the
   `if width > 0` gate is inside the `if register_input_embeds` gate,
   so the eager-extend path (which passes `register_input_embeds=False`)
   also skips the DeepStack slot.

3. `PrefillInputBuffers` — the new dataclass field is required
   (no default), so any code that constructs the dataclass directly
   would break. Grep for `PrefillInputBuffers(` — the only call in
   the tree is `.create()` (updated in this PR); if you have out-of-
   tree callers, please flag.

4. `Qwen3VLForConditionalGeneration` — the new class attribute is a
   simple bool declaration; the reason for storing it on the class
   (rather than the instance) is that the runner reads it via
   getattr before instance-level attributes are meaningful.

5. Unit tests are CPU-only and run in ~14 s under
   `pytest test/registered/unit/model_executor/test_deepstack_replay_slot.py`.
```

## 13. Suggested framing (from operator brief, preserved verbatim)

> Preserve Qwen3-VL DeepStack semantics under BCG replay by copying
> request-specific DeepStack embeddings into an optional stable graph
> input slot.

## 14. Exact commands to inspect + submit

Local inspection (in your `/data/sglang-fork`):

```bash
cd /data/sglang-fork

# Fetch and confirm base
git fetch upstream --quiet
git log fix/bcg-deepstack-replay-slot --oneline upstream/main..HEAD
git diff upstream/main...fix/bcg-deepstack-replay-slot --stat
git diff upstream/main...fix/bcg-deepstack-replay-slot -- python/sglang/srt/models/qwen3_vl.py
git diff upstream/main...fix/bcg-deepstack-replay-slot -- python/sglang/srt/model_executor/

# Confirm nothing from the investigation branch leaked in
git merge-base --is-ancestor fix/pcg-vlm-deepstack-warmup fix/bcg-deepstack-replay-slot \
  && echo "LEAKED" || echo "CLEAN"

# Verify the unit tests pass on the fix branch
git checkout fix/bcg-deepstack-replay-slot
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05 \
  PYTHONPATH=/data/sglang-fork/python \
  python3 -m pytest test/registered/unit/model_executor/test_deepstack_replay_slot.py -v

# Return to a safe branch afterwards
git checkout fix/pcg-vlm-deepstack-warmup
```

Confirm the remote is in sync:

```bash
cd /data/sglang-fork
git fetch origin --quiet
git rev-list --count origin/fix/bcg-deepstack-replay-slot..HEAD   # expect 0
git rev-list --count HEAD..origin/fix/bcg-deepstack-replay-slot   # expect 0
```

Submit the PR (via GitHub web UI or `gh`):

```bash
# via GitHub web (simplest):
# open https://github.com/bowenwan6/sglang/pull/new/fix/bcg-deepstack-replay-slot
# base repository: sgl-project/sglang, base branch: main
# compare: bowenwan6/sglang, branch: fix/bcg-deepstack-replay-slot
# title + body: use §12 above verbatim

# or via gh (if installed):
gh pr create \
  --repo sgl-project/sglang \
  --base main \
  --head bowenwan6:fix/bcg-deepstack-replay-slot \
  --title "fix(bcg): copy Qwen3-VL input_deepstack_embeds into a stable replay slot" \
  --body-file /path/to/pr_body.md
```

Where `pr_body.md` is the description in §12.

---

## Preservation invariants (at package-preparation time)

* `/data/sglang-fork`:
  * `main` fast-forwarded to `eac1f78568…` (per your earlier
    directive).
  * `fix/pcg-vlm-deepstack-warmup` pin still `986c89e69c…`
    (investigation branch, untouched).
  * `fix/bcg-deepstack-replay-slot` — the clean PR branch, 2 commits
    ahead of upstream/main; pushed to `origin`.
* Frozen scratchpad checkout at `58974ca16c…` untouched.
* Protected DeepStack artefacts (R5C `audit_report.md` M state,
  R6.3 orphan dir) untouched, unstaged.
* No commits amended, squashed, reset, force-pushed, or deleted.
* No `Co-authored-by: Claude`, no Anthropic mentions in the fix
  commits.
* No upstream PR opened.
