# Qwen3.5-4B — BCG DeepStack Investigation

> **Investigation, not a confirmed bug.** This directory tracks a
> source-level suspicion on current upstream SGLang. Nothing here asserts
> a runtime failure until runtime evidence supports it. No GPU work is
> authorised in this phase.

- **Branch:** `debug/qwen35-4b-bcg-deepstack` (based on `main` = `a803285`).
- **Model target:** `Qwen/Qwen3.5-4B` (HF, `sha=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  `config.architectures=["Qwen3_5ForConditionalGeneration"]`,
  `model_type=qwen3_5`, ungated).
- **Upstream anchor:** SGLang `main` @
  `5f9b0db18c787cf56ed9bbaf255f083f26c6ebc2` (2026-07-31).
- **Plan-level context:** `plan.md` §7.

## What this investigation is about

`Qwen/Qwen3.5-4B` is registered on current SGLang as multimodal and
prefill-BCG-supported (see the source citations in `plan.md` §7.3). Its
language model receives per-request `input_deepstack_embeds` that are
added to `hidden_states` in layers 0–2. The BCG replay bridge visibly
stabilises `input_embeds` (registered slot + per-request copy into that
slot) but does not obviously stabilise `input_deepstack_embeds` — no slot
is registered, and the replay closure only forwards `input_embeds`.

The runtime question is whether an image request under BCG (a) works
correctly anyway, (b) silently drops the DeepStack contribution and
returns divergent outputs, or (c) triggers a Dynamo recompile / assertion
that keeps correctness but defeats the perf premise. The four possible
outcomes are enumerated in `plan.md` §7.5.

## Layout (to be filled in later parts)

This README is the anchor placed in Part 1. Later parts fill in:

- `source_audit.md` — deep source-level audit of upstream `main`, with
  exact line numbers and captured file provenance.
- `provenance.md` — pinned upstream / model / driver / library SHAs and
  the exact commands used to reproduce them.
- `validation_plan.md` — the CPU-planned validation design able to
  distinguish outcomes (1)–(4).
- `hypothesis.md` — clearly separated established facts vs runtime
  hypotheses vs unverified assumptions, with acceptance criteria.
- `results/` — reserved for future validation attempts (empty until GPU
  is authorised).
- `scripts/` — reserved for future runner / analysis skeletons (CPU-only
  scaffolding lands in Part 5).

Populated in the commits `docs(qwen35): organize BCG DeepStack validation
records`, `docs(qwen35): plan BCG DeepStack validation`, and
`feat(qwen35): prepare BCG DeepStack validation`.

## Historical context (read-only)

The Qwen3-VL-8B PCG capture-stream investigation on
`debug/v2-imgA-pcg-capture-stream-fix` (§4 in `plan.md`) touches
adjacent code but is a **different** problem and a different reproduction
target. That branch and its `experiments/qwen3vl8b/v2/…/root_cause/` tree
remain historical evidence; nothing under §7 rewrites, reorders, or
supersedes it. The historical fork at `/data/sglang-fork` (HEAD
`986c89e69`) is not touched by this investigation.

## Reporting rules

- Preserve exact SHAs, commands, and external PR links.
- Never claim "confirmed upstream bug" until runtime reproduction is
  in hand.
- Never launch GPU work without an explicitly authorised GPU ID from
  the user.
- Commit convention: `docs(qwen35): …`, `feat(qwen35): …`,
  `test(qwen35): …`, `fix(qwen35): …` per `CLAUDE.md`.
