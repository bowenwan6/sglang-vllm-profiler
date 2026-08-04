# F-DeepStack: current-main upstream fix prototype

**Scratchpad only.** Not applied to any tree. No source edits under
`/data/sglang-fork` or the frozen pinned checkout. Prepared for review
before any real patch is authorised.

**Base.** `sgl-project/sglang` @ `e76d0acdc923d992bbda20d4b2bc51db9ac314a7`
(shallow clone under `<scratchpad>/upstream_main/`).

**Design.** Three-site register-slot-and-copy for
`input_deepstack_embeds`, mirroring the `input_embeds` pattern already
in place on current main. Gated on
`getattr(model, "num_deepstack_embeddings", 0) > 0` so models with no
DeepStack layers (every shipped `Qwen/Qwen3.5-*` today) see zero
allocation and zero copy overhead.

## Files

* `patch.diff` — unified diff over the three files.
* `patch_notes.md` — annotated rationale for each hunk.
* `regression_tests_skeleton.py` — proposed test skeleton (unit +
  integration outline) that must fail pre-fix and pass post-fix.

## What this prototype does NOT do

* Add `Qwen3-VL` to the BCG allowlist. That is a separate, orthogonal
  policy decision. The fix is defensive: it lands the correctness
  infrastructure so that (a) whoever adds Qwen3-VL to the allowlist
  next does not walk into the same silent-drop trap, and (b) the
  latent regression cannot become a shipped regression.
* Touch the TC piecewise dummy `run_dummy_multimodal_deepstack_forward`
  which already handles the PCG side.
* Change any public API. No new arguments to `_run_forward`, no new
  fields on `ForwardBatch`; the DeepStack tensor keeps riding
  `layer_kwargs`.

## Preconditions before applying to a real branch

Per the operator brief the following must all be true before any
SGLang source is modified:

1. A **reproducible current-main failure** — R2 rerun on a fresh
   current-main clone under the monkey-patch, matching Attempt 03's
   `FAIL_BCG_DEEPSTACK` signature. **Blocked on driver.**
2. **Proof DeepStack is non-empty at LM entry** — already established
   by Attempt 03 (`nonzero_frac ≈ 0.98`); re-verify on R2.
3. **Proof the issue is specific to BCG replay** — the Attempt 03
   4-arm 2×2 already isolates it (bcg_normal == bcg_zero_deepstack;
   eager_normal ≠ eager_zero_deepstack); re-verify on R2.
4. **A current-main root cause** — completed in
   `r1_upstream_audit.md` §2.4.
5. **A production-safe fix design** — this document.
6. **A regression-test plan** — see `regression_tests_skeleton.py`.

Requirements 1-3 remain blocked on the shared CUDA/driver mismatch.
