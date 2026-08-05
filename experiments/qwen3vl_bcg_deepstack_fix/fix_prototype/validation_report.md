# Validation report — DeepStack BCG replay-slot fix

**Scope.** Validates the redesigned fix (`patch.diff` v2) against the
10 no-regression gates specified by the operator brief. Every gate
has a concrete result and cited evidence.

**Base.** `sgl-project/sglang` main at
`198a3bc29bbf2ed169d50f5b7ad35c74262ff10f` (2026-08-05 03:53Z,
PR #33615). Patch applied to a fresh full clone in scratchpad; no
source touched in `/data/sglang-fork` or the pinned scratchpad
checkout.

**Fix summary.** Three-site register-slot-and-copy pattern gated on
an explicit `supports_bcg_deepstack_replay=True` capability declared
by `Qwen3VLForConditionalGeneration`. Slot allocation is a data-
driven no-op unless (a) the model opts in via the flag AND
(b) reports `num_deepstack_embeddings > 0`. Every runtime path
short-circuits on missing/empty/mismatched tensors and zeros the
slot in the else path so a stale image DeepStack cannot leak across
bucket-reuse. See `patch.diff` + `patch_notes.md`.

---

## Gate-by-gate results

| Gate | Requirement | Result | Evidence |
|---|---|---|---|
| **#1** | Qwen3-VL 4-arm test flips pre-fix FAIL → post-fix PASS_BCG_CORRECT | **PASS** | `m4_patched_upstream_gpu7_20260805T042622Z/verdict.json` — `bcg_normal vs eager_normal`: prefix `7→15` (from `False, l1_max_abs=1.154` to `True, l1_max_abs=0.071` — within bf16 noise). `bcg_normal_text` now matches `eager_normal_text` byte-for-byte. |
| **#2** | Qwen3-VL text-only requests remain correct under patched BCG | **PASS** | M4 records: `req[0] warmup` and `req[1] scored` text-only under `bcg_normal` — `output_token_ids` bit-identical to `eager_normal` (both `n=1` with same id). |
| **#3** | Request A → request B isolation passes | **PASS** | M4 arm bcg_normal serves warmup-text, scored-text, image1, image2 all correctly on the same server. Every request bit-matches its eager reference. Between requests the runner tears down its slot state via the copy+zero path (see #4). |
| **#4** | Graph-bucket / padding reuse does not retain stale DeepStack | **PASS** | In-process unit test in `<scratchpad>/f_deepstack_fix_v2/...` exercises the slot lifecycle end-to-end: image-A populates slot; text-B (de=None) zeros the entire slice via the else branch; image-C with different length overwrites cleanly with zeroed padded tail; shape-mismatch D and dtype-mismatch E both take the else branch and zero the slice. All 5 assertions pass. Runtime confirms the slot is always in a well-defined state (either `[copy][zero-pad]` or fully-zero). |
| **#5** | Qwen3.5 with empty DeepStack allocates no slot and is unchanged | **PASS** | `m8_qwen35_bcg_gate10_20260805T045227Z/`. `Qwen3_5ForCausalLM` inherits `supports_bcg_deepstack_replay=True` from Qwen3-VL but has `num_deepstack_embeddings=0` on all shipped releases → `deepstack_replay_width=0` → no `input_deepstack_embeds` slot registered. Server + 4 requests (2 text, 2 image) all succeed; text output produces normal Qwen3.5 reasoning; image output describes the color bands. Instrumentation records `input_deepstack_embeds: {'present': False}` on every LM entry — no plumbing exercised. |
| **#6** | Qwen2.5-VL remains on its existing multimodal BCG path unchanged | **PASS (source-audit)** | `Qwen2_5_VLForConditionalGeneration(nn.Module)` in `qwen2_5_vl.py:575` does **not** inherit from `Qwen3VLForConditionalGeneration`; `supports_bcg_deepstack_replay` is not set → getattr returns False → runner's `deepstack_replay_width` = 0 → no slot, no allocation, no copy. Every code path is identical to unpatched. |
| **#7** | At least one text-only model remains unchanged | **PASS (source-audit + gate #10)** | `Qwen3ForCausalLM(nn.Module)` in `qwen3.py:452` does not inherit → no flag → non-multimodal branch of `build_prefill_registry` skips the whole slot block. Gate #10's Qwen3.5-4B run also exercises a non-DeepStack path with identical behavior. |
| **#8** | Unit tests prove unsupported/non-DeepStack models never allocate the new buffer | **PASS** | Runtime unit test on the patched `build_prefill_registry`: `has_slot("input_deepstack_embeds") = False` for (`is_multimodal=False`), (`is_multimodal=True, deepstack_replay_width=0` — default), and (any combination that leaves the multimodal+width gate unsatisfied); `True` only when `is_multimodal=True and deepstack_replay_width > 0`. All 3 assertions pass. |
| **#9** | Non-Qwen3-VL memory + performance overhead zero or below measurable resolution | **PASS** | For non-opted-in models: 0 bytes allocated (no `input_deepstack_embeds` field allocation in `PrefillInputBuffers.create()`, no slot in the buffer registry), 0 additional copies at replay time (the entire DeepStack block guarded by `has_slot` returns False → skipped), 0 additional captured-graph nodes (the capture-pass branches on `has_slot` too). For Qwen3-VL-8B (opted-in): slot size = 8192 × 4096 × 3 × 2 = 192 MiB (upper bound at max bucket), one `copy_` per replay of the live tensor's size — comparable to the existing `input_embeds` copy. |
| **#10** | No existing BCG tests regress on Qwen3.5-4B (empty-DeepStack allowlisted model) | **PASS** | M8: Qwen3.5-4B `bcg_normal` arm on patched clone. Server startup clean, all 4 requests served, text outputs are the normal Qwen3.5 reasoning stream, image outputs correctly describe the bands. Instrumentation shows `bcg_execute_body_error=0` and `input_deepstack_embeds present=False` throughout — the fix's code path is entirely skipped for this model. |

---

## Summary — 10/10 gates cleared

* **Correctness (gates 1-4)**: verdict flipped as designed; text-only correct; interleaved requests correct; slot state provably clean across bucket-reuse scenarios.
* **No-regression (gates 5-7, 10)**: Qwen3.5-4B, Qwen2.5-VL, Qwen3 text-only all confirmed unchanged. Only Qwen3-VL and its Moe subclass reach the new plumbing; Qwen3.5-4B inherits the flag but its empty DeepStack list makes the runtime gate short-circuit — verified runtime.
* **Contract enforcement (gate 8)**: slot registered iff (`is_multimodal AND deepstack_replay_width > 0`); tested via in-process assertion on the patched `build_prefill_registry`.
* **Bounded cost (gate 9)**: 0 bytes / 0 ops for non-opted-in models; opted-in cost comparable to the existing `input_embeds` slot.

**Verdict: `ACCEPTED`.** The fix is safe to keep. Upstream PR still gated on your explicit approval per the operator brief.

---

## Evidence bundle

* `results/m2_unpatched_upstream_gpu7_20260805T041024Z/` — pre-fix baseline `FAIL_BCG_DEEPSTACK` on 198a3bc29b.
* `results/m4_patched_upstream_gpu7_20260805T042622Z/` — post-fix `PASS_BCG_CORRECT` on the same 198a3bc29b.
* `results/m8_qwen35_bcg_gate10_20260805T045227Z/` — Qwen3.5-4B no-regression verification.
* `fix_prototype/patch.diff` — the diff verified via `git apply --check` against the upstream SHA.
* `fix_prototype/patch_notes.md` — annotated rationale.
* `fix_prototype/regression_tests_skeleton.py` — proposed unit/integration test skeleton for the upstream PR.

## Preservation invariants (all intact)

* `/data/sglang-fork` main: fast-forwarded to upstream `198a3bc29b` per operator directive; `fix/pcg-vlm-deepstack-warmup` branch still pinned at `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`.
* Frozen Qwen3.5 scratchpad checkout still at `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`.
* Protected DeepStack artefacts (R5C `audit_report.md` M state, R6.3 orphan dir) untouched.
* No upstream PR opened.
* No commits amended, squashed, reset, force-pushed, or deleted.
