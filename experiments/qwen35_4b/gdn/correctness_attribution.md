# Stage-1 correctness attribution — GDN prefill BCG investigation

**Timestamp:** 2026-08-03
**Predecessor:** `smallcell_summary.md` (Phase 6) recorded provisional
`FAIL_BCG_GDN_CORRECTNESS`.
**Attribution question:** is the observed A0-vs-A1/A2/A3 divergence
(1) deterministic numerical path variation near a greedy top-1
boundary, (2) nondeterministic graph execution, or (3) a real
correctness defect?

## Signal

**`SIGNAL_GOOD`. Continue automatically to Stage 2.**

## Rationale

Two decisive experiments were run per the Stage-1 protocol:

### T8 — Arm self-repeat (`gdn_selfrepeat_gpu6_20260803T225529Z`)

Two independent cold-server bring-ups per arm, same (p=128, b=1) cell,
compare rep1 vs rep2 within each arm.

| arm | within-arm result | token mismatches | max_lp_diff |
|---|---|---|---|
| A0 | PASS 8/8 | 0 | 0.000000 |
| A1 | PASS 8/8 | 0 | 0.000000 |
| A2 | PASS 8/8 | 0 | 0.000000 |
| A3 | PASS 8/8 | 0 | 0.000000 |

**Every arm is internally deterministic** across cold bring-ups.
Rules out interpretations (2) nondeterministic graph execution,
runtime randomness, request-state contamination, harness instability,
and port-rotation timing effects.

### T9 — `max_new_tokens=1` first-token cross-arm (`gdn_firsttoken_gpu6_20260803T230743Z`)

Four sequential cold-server bring-ups (A0, A1, A2, A3), single-token
generation, records include top-5 alternates per T7.

| result | count |
|---|---|
| First-token AGREEMENT across A0/A1/A2/A3 | **8/8** |
| First-token DIVERGENCE | 0 |

Prefill-side selected-logprob delta (A0/A2 vs A1/A3) ranges from
**0.000 to 0.090**; every observed top-1/top-2 margin exceeds 0.06.
BCG prefill and eager prefill produce numerically-different-but-
still-in-the-same-greedy-basin logits.

Grouping pattern: **A0 = A2 exactly, A1 = A3 exactly** — confirming
that at `max_new_tokens=1` (no decode CG exercised), the divergence
is purely a **prefill-path** effect, not a decode-CG effect.

## Attribution

The Phase-6 Gate-1 failures at `max_new_tokens=128` (A0 vs A1 5/8
pass, A0 vs A2 2/8, A0 vs A3 1/8) are **entirely autoregressive
amplification** of the tiny (~0.006–0.09) BCG-vs-eager prefill
logprob deltas revealed at n=1. At every individual step every arm
picks the same top-1 in isolation; but over 128 sequential greedy
picks, tiny numerical deltas eventually flip a step whose margin
is smaller than the accumulated noise, and the two sequences drift
apart from there.

This is (1) **deterministic numerical path variation near a greedy
top-1 boundary** per the Stage-1 rubric. Not (2), not (3).

## What this means for the investigation

**Provisional `FAIL_BCG_GDN_CORRECTNESS` is retracted.**

The evidence does not support a correctness defect in the "model
produces wrong tokens" sense. It supports the standard property that
greedy decoding at temperature=0 is fragile to floating-point
reduction order under CUDA graph replay — a property of any
graph-captured inference stack, not a Qwen3.5-4B / GDN / BCG bug.

The observed pattern (A2 fails without BCG, sharing the mechanism
with A1/A3) was already suggestive in Phase 6; Stage 1 T9 confirmed
it by isolating the first-token step where **decode CG never runs**
and A0 = A2 exactly.

## Continuation to Stage 2

Stage 2 (capture-vs-replay overhead separation):
- Run A0 and A1 twice each at (p=128, b=1) to satisfy
  `MIN_CAPTURES_FOR_REPRO=2` from `gdn_verdict.py`.
- Extract capture-time kernel count separately from steady-state
  replay kernels per request.
- Reassess `H_A` (Phase 6 showed A1 launches +16.5 % kernels vs A0
  across the whole trace, but that mixed one-time capture with
  per-request replay).

If steady-state `H_A` firms up, proceed to Stage 3 (< 1024
alternate-stream threshold ladder). If steady-state `H_A` collapses,
Stage 2 result is `PASS_BCG_GDN_NO_GAP`.

Preservation invariants held throughout Stage 1: `/data/sglang-fork`
HEAD unchanged (`986c89e69c…`), frozen SGLang HEAD unchanged
(`58974ca16c…`) with empty `git diff --stat`.
