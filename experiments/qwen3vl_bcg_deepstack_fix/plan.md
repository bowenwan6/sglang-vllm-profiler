# Qwen3-VL BCG DeepStack — reproduction and fix plan

**Status.** Planning document only. No code changes, no upstream source
modifications, no substantial GPU experiments authorised at this
stage. Awaiting review.

**Branch.** `debug/qwen3vl-bcg-deepstack-fix`, cut from
`debug/qwen35-4b-gdn-prefill-bcg` HEAD `b8c0f45` (which is a strict
superset of `debug/qwen35-4b-bcg-deepstack` HEAD `d29b4a6`).

**Preservation invariants** (unchanged and load-bearing here):

* `/data/sglang-fork` HEAD `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`
  is read-only historical evidence; no writes in this pass.
* Frozen SGLang scratchpad checkout at HEAD `58974ca16c…` (Qwen3.5
  sub-track pin) is preserved read-only.
* All DeepStack attempt directories under
  `experiments/qwen35_4b/results/` (`infracheck_gpu7_20260801T012122Z`,
  `attempt_gpu7_20260801T013522Z`, `harness_gpu1_20260801T062833Z`,
  `attempt_gpu1_20260801T115524Z`) stay verbatim as historical evidence.
* Protected DeepStack artefacts under
  `experiments/qwen3vl8b/v2/…/R5C_correctness_audit/audit_report.md`
  (uncommitted M state) and
  `…/R6.3_image_and_sweep/attempt_gpu2_partial_orphaned_20260729T094128Z/`
  (uncommitted orphan dir) remain under user control.
* GPU allowlist per current authorisation: `{0..7}` (per
  `experiments/qwen35_4b/gdn/validation_plan.md` Amendment 1); foreign-
  PID guard active; PGID-scoped cleanup; never signal foreign PIDs.
* Author identity: Bowen Wang; commits do not reference Claude,
  Anthropic, or AI assistants.

---

## 1. Objective

Convert the source-level `replay_layer_forward` diagnosis (audited at
[`plan.md`](../../plan.md) §7.3(4)) and the live-fire Attempt 03
verdict `FAIL_BCG_DEEPSTACK` (`experiments/qwen35_4b/results/attempt_gpu1_20260801T115524Z/`)
into an upstream-ready fix:

1. **Reproduce** the same `FAIL_BCG_DEEPSTACK` signature on **current
   upstream SGLang `main`**, not only on the pinned `58974ca16c…` +
   `986c89e69c…` combination used in the Qwen3.5 sub-track.
2. **Prove DeepStack is non-empty at LM entry** on the reproduction
   arm (i.e., the input path works — the bug is in the replay bridge,
   not in embedding routing).
3. **Distinguish four regimes cleanly**: eager+DeepStack,
   BCG+DeepStack, eager+zero-DeepStack (diagnostic ablation),
   BCG+zero-DeepStack (diagnostic ablation).
4. **Identify the exact omission** in the replay/buffer path
   (register-slot or `layer_kwargs` forwarding).
5. **Design a general, production-safe fix** that:
   - is not conditioned on the model class name;
   - does not regress the currently-empty-DeepStack case (Qwen3.5 today);
   - matches the existing pattern that PR #30872 established for
     `input_embeds` in the same code path;
   - is scoped enough to file as a small, reviewable upstream PR.
6. **Validate correctness, request isolation, graph-bucket behaviour,
   and performance** under the fix on a real image workload.
7. **Add regression tests** that both catch the failure mode (missing
   DeepStack under BCG replay) and prevent latent recurrence
   (numel-guard + `add_` outcome check).

---

## 2. Prior evidence — what stays valid and what must be revalidated

### 2.1 Evidence that remains valid

* **Source-level diagnosis** (audited [`plan.md`](../../plan.md)
  §7.3(4)): `replay_layer_forward` copies only `input_embeds`; no
  `input_deepstack_embeds` slot is registered in
  `cuda_graph_buffer_registry`; `layer_kwargs` is not forwarded into
  `.replay(...)`. Verified independently against
  `/data/sglang-fork` HEAD `986c89e69c…` at
  `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:923-929`
  in this pass — the function body is:

  ```python
  def replay_layer_forward(*args, **layer_kwargs):
      return self.backend.replay(
          shape_key, static_forward_batch, **kwargs
      )
  ```

  where `**kwargs` binds to the enclosing outer-forward kwargs (no
  DeepStack), and `**layer_kwargs` (which *does* contain
  `input_deepstack_embeds`) is silently discarded.

* **Attempt 03 verdict** `FAIL_BCG_DEEPSTACK` at
  [`attempt_gpu1_20260801T115524Z/verdict.md`](../qwen35_4b/results/attempt_gpu1_20260801T115524Z/verdict.md).
  Signature: `bcg_normal_vs_bcg_zero` is bit-identical (l1_max_abs
  = 0.0), `bcg_zero_vs_eager_zero` is within bf16 noise
  (l1_max_abs = 0.066), `bcg_normal_vs_eager_normal` diverges at
  the first non-boilerplate token (7/15 common prefix, l1_max_abs =
  1.15), `eager_repeat_noise` is 0.0. This is the textbook
  "zero-DeepStack signature under BCG" pattern.

* **DeepStack is genuinely non-empty at LM entry on Qwen3-VL**:
  Attempt 03's normal-arm hook records `shape=[896, 12288]` (i.e.
  `[N_tokens, hidden_size × num_deepstack_embeddings]` with
  `hidden=4096`, `num_ds=3`), `nonzero_frac ≈ 0.98`, `abs_sum
  150085.25`, `sha256_16 54263d67…`. The zero-arm hook verifies
  substitution: `nonzero_frac → 0.0` in `bcg_zero` and `eager_zero`.

* **Cross-arch DeepStack + BCG audit** at
  [`latent_bug_analysis.md`](../qwen35_4b/latent_bug_analysis.md)
  §2: no shipped `Qwen/Qwen3.5-*` release populates DeepStack;
  Qwen3-VL populates 3 layers `[8, 16, 24]` on 8B and `[5, 11, 17]`
  on 4B / 2B; Qwen3-VL is **not** on the BCG allowlist on upstream
  `main @ 58974ca16` — Attempt 03 reached BCG only via a
  profiler-owned test-only monkey-patch
  (`experiments/qwen35_4b/scripts/bcg_allowlist_patch.py`).

* **Harness** under `experiments/qwen35_4b/scripts/`:
  `server_launcher.py`, `bcg_allowlist_patch.py`, `client.py`,
  `instrumentation.py`, `verdict.py`, `runner.sh`,
  `bootstrap/sitecustomize.py`, `fixtures/` (byte-pinned image
  SHA-256 `8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`).
  All CPU-testable pieces are covered by
  `test_instrumentation.py`.

### 2.2 Evidence that must be revalidated

* **Current upstream SGLang state** (post the frozen `58974ca16c…`
  pin, 2026-07-31). Nothing in this project has queried GitHub since
  then. The specific questions:
  - Has `replay_layer_forward` been amended to copy
    `input_deepstack_embeds` or forward `layer_kwargs`?
  - Has `Qwen3VLForConditionalGeneration` (or its MoE variant) been
    added to `multimodal_breakable_cuda_graph_supported_model_archs`?
  - Has any new register-slot or numel-guard shipped?
  - Has any related PR merged since PR #30868 (2026-07-19) and PR
    #30872 (2026-07-28) that changes the DeepStack code path?
* **Post-driver-upgrade CUDA state**. Driver `595.71.05` (installed
  2026-08-04 12:53 UTC, `/proc/driver/nvidia/version` mtime) is
  outside torch `2.11.0+cu130`'s supported range. Any GPU
  reproduction is blocked until this is resolved (matches the §9
  L2Norm sub-track blocker; both tracks share this constraint).
* **Instrumentation robustness against upstream churn**. The
  Attempt 03 hook targets `Qwen3LLMModel` by class name via
  `register_forward_pre_hook`; if upstream renames the LM class or
  moves DeepStack routing, the hook must be updated.
* **Fixture placeholder alignment against current tokenizer**. If
  Qwen3-VL's tokenizer / processor revision has moved, the
  `<|vision_start|><|image_pad|><|vision_end|>` placeholder must be
  re-verified.

---

## 3. Reproduction ladder

Ordered least-cost to most-cost. Each rung has an explicit expected
outcome, a hard-fail condition, and a decision rule for whether to
climb to the next rung.

Every GPU-touching step is **paused until the CUDA/driver mismatch
(§4) is resolved**. R0, R1, and R5-design are executable now.

### R0 — environment + preflight check (CPU-only, executable now)

**Do.** Verify: (a) `git status` clean apart from preserved
protected files; (b) frozen `/data/sglang-fork` HEAD unchanged; (c)
`experiments/qwen35_4b/scripts/test_instrumentation.py` still passes
on CPU; (d) the byte-pinned fixture SHA-256 matches.
**Expected.** All pass.
**Hard fail.** Any drift in preservation invariants → stop, do not
proceed.
**Advance rule.** All pass → R1.

### R1 — upstream state audit (CPU-only, executable now)

**Do.** Query GitHub (or a fresh SGLang clone under scratchpad) for
current upstream `main` HEAD and diff the following files against
the pinned `58974ca16c…`:

* `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`
* `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
* `python/sglang/srt/model_executor/runner_backend/breakable_cuda_graph_backend.py`
* `python/sglang/srt/configs/model_config.py`
  (BCG allowlist lines ~1845-1848, PCG allowlist lines ~1836-1841)
* `python/sglang/srt/models/qwen3_5.py`
* `python/sglang/srt/models/qwen3_vl.py`
* `python/sglang/srt/managers/mm_utils.py` (DeepStack synthesis
  around lines 1108-1373)
* Any new file matching `**/*deepstack*.py`.

**Expected.** Small diff; the `replay_layer_forward` bug is likely
still present because the audit at `plan.md` §7 was recent and no
matching upstream PR is on record.
**Alternate outcomes and their consequences.**
* (a) *Bug still present, no meaningful upstream change* → proceed
  to R2 as designed.
* (b) *`replay_layer_forward` fixed upstream* → close this sub-track
  as "already fixed upstream", file a note in this plan.md and stop.
  Do not proceed with a duplicate fix.
* (c) *Qwen3-VL added to BCG allowlist but `replay_layer_forward`
  unchanged* → the bug is now reachable in shipped production; treat
  as increased urgency and proceed to R2 without the monkey-patch.
* (d) *Some other fix landed (e.g., numel-guard eager fallback)* →
  characterise it in the plan, decide whether a stronger fix
  (register-slot-and-copy) still adds value.

**Advance rule.** Outcomes (a) and (c) → R2. (b) → stop and close.
(d) → decide before advancing.

### R2 — baseline reproduction on the pinned SHA (GPU, blocked on CUDA)

**Do.** Re-run the Attempt 03 2×2 (four arms: `eager_normal`,
`eager_zero_deepstack`, `bcg_normal`, `bcg_zero_deepstack`) on
`Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0` against the same pinned
SGLang checkout `58974ca16c…` + fork `986c89e69c…` combination
Attempt 03 used, GPU 1 (or the current authorised alternate), same
byte-pinned image fixture. Same profiler-owned test-only monkey-
patch (`bcg_allowlist_patch.py`) to opt Qwen3-VL into BCG. Same
verdict schema.

**Expected outcome (positive).** Verdict `FAIL_BCG_DEEPSTACK`, with
signature bit-identical to Attempt 03 within eager-vs-eager bf16
noise (l1_max_abs `bcg_normal_vs_bcg_zero` = 0.0;
`bcg_normal_vs_eager_normal` ≥ ~1.0 on scored tokens beyond the
common prefix).

**Hard fail.**
* Verdict differs from `FAIL_BCG_DEEPSTACK` (e.g., `PASS_BCG_CORRECT`,
  `FEATURE_GAP_EAGER_FALLBACK`, or new `AMBIGUOUS`) → the bug is not
  stably reproducible; investigate why before designing the fix.
* GPU or preflight failure → `INFRA_FAILURE`, retry after cleanup.

**Advance rule.** `FAIL_BCG_DEEPSTACK` with the same signature → R3.
Otherwise, diagnose before continuing.

### R3 — upstream-current reproduction (GPU, conditional on R1)

**Do.** If R1 shows meaningful upstream changes in the DeepStack
code path (outcomes c or d from R1), cut a fresh SGLang clone at
current upstream `main`, wire the profiler runner to import from it
via `PYTHONPATH`, verify the pin check succeeds, and rerun the 2×2.

**Expected.** Same `FAIL_BCG_DEEPSTACK` signature as R2 modulo any
new upstream defence. If a numel-guard eager fallback has landed
upstream, expect the arm to `FEATURE_GAP_EAGER_FALLBACK` at the
image request instead of divergence.

**Advance rule.** Signature holds → R4. Signature changed →
characterise and update the fix design before R4.

### R4 — non-empty DeepStack proof (re-verification, GPU-cheap)

**Do.** Under R2 (or R3), re-verify the pre-hook records
`nonzero_frac > 0.5` and `abs_sum > 0` at LM entry on the
`bcg_normal` arm. This is already established by Attempt 03; the
re-verification catches upstream churn that might break routing.

**Expected.** `nonzero_frac ≈ 0.98`, `numel = N_tokens × 12288`,
finite, non-zero.

**Hard fail.** DeepStack not populated at LM entry → the failure is
in mm_utils / embedding routing, not in replay; re-scope.

**Advance rule.** Pass → R5.

### R5 — direct evidence that the captured graph lacks DeepStack kernels

**Do.** Use `nsys profile` on the `bcg_normal` and `eager_normal`
arms; extract kernel-count and kernel-name diffs. Confirm that the
BCG-captured graph contains no additional `add_` kernels
corresponding to the DeepStack contribution (or use a distinctive
kernel name pattern that DeepStack `add_` produces).

**Expected.** Either:
* the BCG graph is bit-for-bit the same graph as the "no-DeepStack"
  eager graph (strong evidence the capture happened with a cold
  DeepStack branch), or
* small residual kernels appear but produce zero output (`add_` on
  a zero-slot).

**Hard fail.** BCG graph *does* contain DeepStack `add_` kernels but
they still fail to affect outputs → the bug is in the buffer
lifecycle, not in the graph. Re-scope.

**Advance rule.** Any of the two positive signatures → R6.

### R6 — zero-DeepStack ablation signature re-verification

**Do.** Confirm `bcg_zero_vs_eager_zero` matches within eager noise
and `eager_zero_vs_eager_normal` diverges at the first non-
boilerplate token by the same magnitude as Attempt 03 (l1_max_abs
~1.14). This is the diagnostic ablation that isolates the effect to
DeepStack specifically, not to some other BCG artefact.

**Expected.** Pattern matches Attempt 03 within noise.

**Hard fail.** Pattern does not match (e.g., BCG diverges from
`eager_zero_deepstack` too) → some other BCG bridge issue is
present besides DeepStack; scope out.

**Advance rule.** Pattern matches → R7.

### R7 — sensitivity to graph-bucket / shape

**Do.** Vary the prefill bucket size — repeat R2 with images that
push `padded_num_tokens` to the buckets we care about (e.g. 128,
512, 896, 1024, 2048). Verify the FAIL signature persists across
buckets and is not a single-bucket fluke.

**Expected.** FAIL signature is bucket-independent.

**Hard fail.** Some buckets pass, others fail → the bug is bucket-
conditional; re-scope.

**Advance rule.** Bucket-independent → R8.

### R8 — request-order isolation

**Do.** Interleave request order: `text_only → image → text_only →
image_batched_with_text`. Verify:

* Each request's output is deterministic across repeats within an
  arm (already established by earlier sub-tracks at
  `experiments/qwen35_4b/results/gdn_firsttoken_gpu6…`; re-verify
  under this harness).
* The DeepStack drop happens *only* on the image request, not on
  text-only (which has empty DeepStack and should not be affected).
* Batched requests with mixed content route correctly.

**Expected.** Text-only unaffected; image-only shows FAIL;
mixed-batch shows partial FAIL (correctness of the text portion
preserved).

**Hard fail.** Text-only requests show output divergence → the bug
scope is wider than DeepStack (either mm_utils routing or a
scheduling issue); re-scope up.

**Advance rule.** Pattern matches → design the fix (§4).

---

## 4. Fix design — general, production-safe, not model-name-specific

Three approaches ranked. All three would need to be applied to a
clean branch off current upstream `main`, not to `/data/sglang-fork`
(preservation invariant).

### 4.A **Register-slot-and-copy** (recommended)

Extends the existing `input_embeds` pattern (PR #30872) to
`input_deepstack_embeds`:

1. In `cuda_graph_buffer_registry.py`, register an
   `input_deepstack_embeds` slot alongside the existing
   `input_embeds` slot when:
   - `is_multimodal and register_input_embeds` is True (same guard
     as `input_embeds`), and
   - warmup observed at least one call with
     `input_deepstack_embeds.numel() > 0` (so Qwen3.5-style empty
     configs skip the allocation).
   Slot shape: same `[max_padded_tokens, hidden_size *
   num_deepstack_embeddings]` template used by `mm_utils.py:1108-1140`;
   dtype and device match model.
2. In `prefill_cuda_graph_runner.py` `_execute_body_capture`
   (`replay_layer_forward` at line 923-929 in the fork snapshot),
   forward `layer_kwargs` into `.replay(...)` rather than the outer
   `kwargs`. Copy live `input_deepstack_embeds` into the registered
   slot before the `.replay(...)` call, mirroring the existing
   `input_embeds` copy. Handle the `numel == 0` case as a no-op
   (skip the copy).
3. In the BCG capture pass (`_run_forward` at
   `prefill_cuda_graph_runner.py:606-649`), route the LM's forward
   with `input_deepstack_embeds=<slot buffer>` (pre-filled zero if
   the warmup did not observe a non-empty tensor). This guarantees
   the DeepStack `add_` branch is traced into the captured graph.
4. Preserve behaviour on models with empty DeepStack: `mm_utils.py`
   allocates `(N, 0)` — the `numel() > 0` guard in
   `Qwen3_5ForCausalLM.forward` (and Qwen3-VL's LM) already trivially
   skips the `add_`; the slot allocation policy above must also
   detect this and not allocate a zero-width slot.

**Why this is production-safe.** No new gate keyed on model name.
The slot allocation is data-driven from the warmup observation.
Correct for all currently-shipping models: Qwen3.5-* (empty
DeepStack) sees no allocation and no extra copy; Qwen3-VL populates
the slot at replay time. Mirrors the existing pattern reviewers
already approved for `input_embeds` in PR #30872, so review should
be short.

**Cost.** One extra `copy_` per BCG replay of size ≤ `padded_tokens
× hidden_size × num_deepstack_embeddings` (e.g. 4096 × 4096 × 3 bf16
≈ 96 MiB on the largest bucket) — comparable to the existing
`input_embeds` copy. Overhead is measurable, not dominant.

### 4.B **Numel guard + eager fallback** (minimal defensive)

Add to `can_run_graph` (or the enclosing BCG dispatch): if
`input_deepstack_embeds` is present in the LM's `layer_kwargs` and
`numel() > 0`, route the request to the eager path instead of BCG
replay.

**Correct?** Yes — silently correct. **Loses.** BCG performance on
image requests (defeats the purpose of adding Qwen3-VL to the BCG
allowlist).

**Why keep it as a fallback design.** If 4.A's buffer-registry
lifecycle is too complex to land in the current window, 4.B is a
one-check-two-line change that guarantees correctness while a
proper fix is discussed. Also useful as a defensive check to be
kept alongside 4.A — a numel guard downstream of the copy protects
against future churn that removes the slot allocation.

### 4.C **Extend BCG capture to dummy-trace the DeepStack branch**
(alternative to 4.A's capture-pass change)

Similar to `run_dummy_multimodal_deepstack_forward` in
`tc_piecewise_cuda_graph_backend`, allocate a small nonzero DeepStack
tensor during BCG warmup so Dynamo / capture traces both branches
into the graph. At replay, live DeepStack is copied into the slot
and applied; when live DeepStack is empty, the slot is filled with
zeros and the `add_` is a no-op.

**Trade-off vs 4.A.** 4.C alone (without the slot) still leaves the
replay-side copy unaddressed; the DeepStack tensor must live
somewhere the graph knows about. So 4.C is really a *subset* of 4.A
— you always need the slot; 4.A includes the capture-pass tracing
that 4.C standalone would provide. Prefer 4.A.

### Recommended fix pick: **4.A + 4.B numel guard as defence-in-depth**

* Land 4.A as the primary correctness fix.
* Keep a `numel() > 0` guard around the slot copy so a shape-mismatch
  bug becomes a clear failure instead of a silent zero.

### What we will **not** do

* Rename or hard-code model class names in the fix. The fix must
  key on the presence of `input_deepstack_embeds` in `layer_kwargs`,
  not on `model.__class__.__name__`.
* Skip capture-time tracing. Any fix that only copies data at replay
  time but leaves the captured graph without a DeepStack `add_`
  node cannot possibly work.
* Modify `/data/sglang-fork`. All fix work happens on a fresh
  upstream-`main` checkout in a scratchpad worktree; the fork stays
  read-only historical evidence.
* File the upstream PR before R2 and R3 both pass at the current
  upstream SHA (would be filing against a non-reproducible failure).

---

## 5. Validation plan (post-fix, GPU)

Each gate is a hard blocker on the next.

* **Gate C-1: bit-exact correctness.** With the fix applied, the
  Attempt 03 2×2 rerun must produce
  - `bcg_normal_vs_eager_normal`: common_prefix_len == full length,
    l1_max_abs ≤ eager-vs-eager noise floor (≤ 0.1);
  - `bcg_zero_vs_eager_zero`: unchanged from pre-fix (should still
    match, since zero-DeepStack has no `add_` to fix);
  - No regression on `bcg_normal_vs_bcg_zero` — under the fix these
    should now *diverge* (DeepStack is now applied under BCG), so
    the equality that was the FAIL signature now becomes an
    inequality expected to match `eager_normal_vs_eager_zero`.
* **Gate C-2: request-order isolation.** Interleaved
  text→image→text: text arms bit-identical to pre-fix; image arm
  matches eager reference; no leakage between requests.
* **Gate C-3: graph-bucket equivalence.** Across a bucket sweep
  (e.g., 256, 512, 896, 1024, 2048), image responses match eager
  reference at each bucket. This proves the slot lifecycle is
  bucket-shape-safe.
* **Gate C-4: empty-DeepStack regression.** Same 2×2 on a Qwen3.5
  target (empty DeepStack). Expected: `bcg_normal` == `eager_normal`
  == `bcg_zero_deepstack` == `eager_zero_deepstack` — the fix must
  not alter behaviour when there is no DeepStack to copy.
* **Gate P-1: BCG replay latency.** Compare BCG replay wall-clock
  latency at each bucket, pre-fix vs post-fix. Acceptable regression
  ≤ 5 % or ≤ 1 ms per prefill (whichever is larger). If regression
  is above threshold, revisit the slot lifecycle for redundant
  copies.
* **Gate P-2: peak GPU memory.** Registered slot adds up to
  `max_padded_tokens × hidden × num_ds × dtype_size` bytes.
  Acceptable if within `≤ 5 %` of pre-fix peak. Otherwise, revisit
  slot lifetime (e.g. tie it to bucket size rather than
  `max_padded_tokens`).

---

## 6. Regression tests suitable for upstream review

* **Unit test** in
  `test/srt/model_executor/test_cuda_graph_buffer_registry.py`
  (new): assert `input_deepstack_embeds` slot is registered iff
  `is_multimodal and register_input_embeds and warmup_saw_deepstack`
  is True; assert slot dtype/device/shape match the LM's expected
  DeepStack tensor.
* **Unit test** in
  `test/srt/model_executor/test_prefill_cuda_graph_runner.py`
  (new): assert `replay_layer_forward` forwards
  `layer_kwargs["input_deepstack_embeds"]` (when non-empty) into a
  copy targeting the registered slot before `.replay(...)`.
* **Integration test** — small VLM (Qwen3-VL-2B or 4B, whichever is
  cheapest to load in CI) with a byte-pinned image fixture; assert
  `bcg_normal_output == eager_normal_output` within bf16 tolerance.
  This test *catches the regression* — pre-fix it fails, post-fix
  it passes.
* **Mixed-batch test** — image + text-only in one prefill; assert
  both requests produce correct outputs and the image request's
  DeepStack contribution does not leak into the text request.
* **Zero-DeepStack test** — Qwen3.5-* (empty DeepStack config);
  assert BCG replay works exactly as pre-fix on this model (no
  regression on models that never used DeepStack).

Tests must fail deterministically pre-fix, not on flake. This is
established: Attempt 03 already shows the failure is bit-stable
across repeats and arms.

---

## 7. Major risks and uncertainties

* **CUDA driver mismatch (shared with §9 sub-track).** No GPU work
  is executable until the driver / torch mismatch is resolved. R0,
  R1, and §4-design work is all executable now.
* **Upstream may have partially fixed this.** R1 is the check. If
  outcome (b) fires, this sub-track closes as "already upstream".
* **Buffer-registry lifecycle complexity.** The registry currently
  assumes fixed slots per bucket. DeepStack tensor shape depends on
  `num_tokens × hidden × num_ds`. Slot allocation policy needs to
  handle the case where `num_ds` is model-dependent and unknown
  until first non-empty observation. Mitigated by warmup-driven
  allocation.
* **Slot-lifetime races**. If two prefills interleave with different
  bucket sizes, the slot must be sized for the largest bucket and
  written per-replay. The existing `input_embeds` slot already
  handles this — reuse the same pattern.
* **Model-side changes.** If a future Qwen3-VL variant changes
  `num_deepstack_embeddings` at runtime, the slot allocation may
  need to be re-sized. Acceptable since warmup re-runs on new
  buckets already.
* **Attribution risk on R3.** If the current upstream SHA differs
  meaningfully, small unrelated latency deltas may confound the
  correctness signal. Mitigated by scoring on token equality
  first, latency second.
* **Fixture drift.** The Attempt 03 fixture used Qwen3-VL-8B's
  `<|vision_start|><|image_pad|><|vision_end|>` placeholder; if
  upstream Qwen3-VL processor changes the required placeholder,
  the fixture warning will re-appear. R0 catches this before R2.

---

## 8. Non-goals

* **No source patch to `/data/sglang-fork`.** Preservation invariant.
* **No PR filed** before R2 (and R3 if R1 requires it) both pass.
* **No performance headline.** BCG-vs-eager performance under the
  fix is a follow-on; this pass validates correctness.
* **No merge into `main` of this profiler repo without user review.**
* **No modifying pinned SGLang SHA (`58974ca16c…`).** A new upstream
  reproduction environment lives in a fresh scratchpad clone; the
  existing pin is preserved for evidence continuity.

---

## 9. Immediate next steps (§ track)

Executed only after this plan is reviewed and explicit approval is
given for each step.

1. **N1 (docs, this document)** — file this plan; update
   `plan.md` §10 to reference it; commit `docs(qwen3vl): plan
   BCG DeepStack fix — reproduction ladder and fix design`.
2. **N2 (R0 + R1, CPU-only)** — run the preflight and the upstream
   audit; report findings; commit `docs(qwen3vl): R0/R1 upstream
   audit — <outcome>`.
3. **N3 (CUDA / driver resolution)** — wait for the shared
   environment blocker to lift (see §9 in `plan.md`). No workaround
   attempted from this branch.
4. **N4 (R2 baseline reproduction)** — reproduce Attempt 03's FAIL
   on the pinned SHA to confirm stability; commit
   `test(qwen3vl): reproduce baseline FAIL_BCG_DEEPSTACK on pinned SHA`.
5. **N5 (R3–R8 ladder)** — climb the reproduction ladder; each rung
   commits + pushes.
6. **N6 (fix prototype on a fresh upstream checkout)** — apply 4.A
   + 4.B in a scratchpad worktree; run the validation gates.
7. **N7 (upstream PR preparation)** — rebase onto current upstream
   `main`, write PR description with reproduction evidence,
   regression tests, and correctness gates. **Do not open the PR
   until user review.**

---

## 10. References

* [`plan.md`](../../plan.md) §7 (Qwen3.5-4B BCG DeepStack sub-track
  close-out), §9 (L2Norm sub-track deferred with the same driver
  blocker), §10 (this sub-track index).
* [`experiments/qwen35_4b/latent_bug_analysis.md`](../qwen35_4b/latent_bug_analysis.md)
  — cross-arch audit + Attempt 03 outcome.
* [`experiments/qwen35_4b/results/attempt_gpu1_20260801T115524Z/verdict.md`](../qwen35_4b/results/attempt_gpu1_20260801T115524Z/verdict.md)
  — live-fire `FAIL_BCG_DEEPSTACK` evidence.
* [`experiments/qwen35_4b/hypothesis.md`](../qwen35_4b/hypothesis.md)
  — verdict schema and Amendment 5 closure.
* [`experiments/qwen35_4b/validation_plan.md`](../qwen35_4b/validation_plan.md)
  — GPU allowlist + Amendments 1-5.
* `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:923-929`
  (source of the bug on `/data/sglang-fork`).
* `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py`
  (register-slot pattern for `input_embeds` from PR #30872).
* `python/sglang/srt/managers/mm_utils.py:1108-1373` (DeepStack
  synthesis).
* SGLang PR #30868 (2026-07-19, PCG Dynamo warmup, merged).
* SGLang PR #30872 (2026-07-28, `input_embeds` slot for BCG, merged).
