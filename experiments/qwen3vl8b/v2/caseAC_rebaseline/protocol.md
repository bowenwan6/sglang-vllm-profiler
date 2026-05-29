# Protocol — v2 Case A/C rebaseline on production-default overlap (issue #2)

> **PLACEHOLDER — not yet locked.** Fill in and get approval before executing any run.
> Foundational v2 experiment: every later v2 claim (#3/#4/#5) rests on this production-default baseline.

## Goal

Re-establish the Qwen3-VL clean headline baseline with the **production-default overlap schedule (ON)**,
and answer: **does forcing prefill piecewise CUDA graph still reduce TTFT against the overlap-ON
baseline?** (v1's −39% Case-A win was measured only in the overlap-OFF regime.)

## Variants

| id | framework | flags | role |
|---|---|---|---|
| S0 | SGLang | default (overlap-ON) | **headline production baseline** |
| S2 | SGLang | default + `--enforce-piecewise-cuda-graph` | PCG testing lever |
| V0 | vLLM | default | clean cross-framework anchor |
| S0-abl | SGLang | `--disable-overlap-schedule` | **optional** ablation only (v1 baseline) |

## Scope

- Case A: 128→128, c=1
- Case C: 512→128, c=16
- **Clean only — no KAPI logging, no profiler** (Instrumentation Policy, plan.md §0).
- Greedy: temperature=0, top_p=1.

## To record (per run)

TTFT (p50/p95/p99), TPOT, CV, failures, warmup, reps, GPU id, exact flags, dataset sha, framework
versions, torch/CUDA. Use an S0→S2→S0 bracket (or interleaved for Case C) to bound drift.

## Acceptance (from issue #2)

- Main tables explicitly label overlap/default status.
- `--disable-overlap-schedule` moved out of headline wording (ablation only).
- Report states clearly whether PCG still helps vs the production-default SGLang baseline.

## Open decisions (resolve before locking)

- GPU id (serialized; confirm a free GPU at run time).
- warmup/reps per case (reuse v1: A = w30/reps3; C = w500 interleaved?).
- Whether to include the S0-abl ablation this round or defer.
