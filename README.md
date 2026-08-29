<div align="center">

# SGLang vs vLLM — Latency Profiling Lab

<p>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/badge/SGLang-sgl--project-blue?logo=github&logoColor=white" />
  </a>
  &nbsp;
  <a href="https://github.com/vllm-project/vllm">
    <img src="https://img.shields.io/badge/vLLM-vllm--project-blueviolet?logo=github&logoColor=white" />
  </a>
</p>

Phase-gated profiling of SGLang against vLLM on **Qwen3-VL-8B-Instruct** to locate *where* SGLang's
latency gap comes from — not a generic benchmark ranking.

*Single H200 · TP=1 · bfloat16 · greedy · text-only path*

</div>

## Overview

This repo holds three tracks:

1. **`qwen3vl8b`** — the original TTFT-gap investigation on Qwen3-VL-8B-Instruct, asking a focused
   question: **where does SGLang's time-to-first-token (TTFT) gap versus vLLM come from?** The goal
   is not to declare a winner but to attribute the gap to a specific stage (prefill vs decode),
   kernel family, and system path, and to turn that into ranked, evidence-backed hypotheses for
   optimization. Organised as a phase-gated pipeline (Phase 0 → 5): prove the two servers are
   comparable, establish a baseline, shape / de-noise the workloads, collect torch-profiler traces,
   triage them, and validate the top hypothesis. See §Directory Layout below and `plan.md` §1–§6.
   **Text-only Case A/C is complete; the image+text arm (#4) is the active profiling priority.**
2. **`qwen35_4b`** — correctness-first sub-track, **concluded**. Two questions were asked and
   answered: the DeepStack gap on `Qwen/Qwen3.5-4B` is `NOT_APPLICABLE_QWEN35` (every shipped
   Qwen3.5 checkpoint has an empty DeepStack index list), and the GDN study returned
   `PASS_BCG_GDN_NOTABLE_GAP` (+13.6 % launches, ≤2 % wall-clock, no correctness bug). See
   [`experiments/qwen35_4b/README.md`](experiments/qwen35_4b/README.md) and `plan.md` §7.
   The roadmap's actual **SGLang-vs-vLLM Qwen3.5 transfer comparison (#3) has not been run** —
   these two studies answer different questions and are not a substitute.
3. **`qwen3vl_bcg_deepstack_fix`** — spun out of #9. A real, live-fire correctness bug: a
   BCG-replayed Qwen3-VL image prefill dropped its DeepStack contribution. Fixed, validated
   `FAIL → PASS`, and upstreamed as
   **[sgl-project/sglang#33726](https://github.com/sgl-project/sglang/pull/33726)** (open, approved,
   mergeable). Current state:
   [`upstream_handoff.md`](experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md).

## Main Findings

1. **Clean Case A exposes an actionable TTFT gap — confirmed on the production default (v2 #2).** In an
   uninstrumented benchmark on **SGLang default (overlap-ON)**, Case A (128→128, c=1) TTFT is **21.94 ms**
   vs vLLM's **13.12 ms**, while TPOT is unchanged — i.e. the issue is on the first-token / prefill side.
   (v1 measured this on the `--disable-overlap-schedule` baseline at ~19.2 ms, which *understated* the
   production gap; that flag is now an ablation only.)
2. **GPU kernel speed is not the differentiator.** Both frameworks spend 72–86% of GPU time in the
   *same* `nvjet_sm90_*` FP8 GEMM family — GEMM is a **shared absolute cost**, not what explains the
   Case A TTFT gap.
3. **The cause is SGLang's VLM prefill graph coverage; a clean intervention validates it on the
   production default.** For Qwen3-VL, SGLang disables prefill piecewise CUDA graph (VLM auto-disable).
   Forcing it on (`--enforce-piecewise-cuda-graph`) drops Case A TTFT **21.94 → 14.04 ms (−36%)**, TPOT
   unchanged, 0 failures — **reaching the vLLM TTFT range**. The v2 #2 production-default rebaseline shows
   this is **not** an artifact of the overlap-OFF baseline. (Testing lever, not a production fix.)
4. **Case C defines the boundary (confirmed on the production default).** At c=16 batched (clean), the
   same intervention yields **no material TTFT gap and no Case-A-like improvement** (SGLang default
   204.8 ms, +PCG 230.6 ms, vLLM 215.7 ms; batched CV ~14–15%). The fix is workload-shape-dependent →
   favor **selective enablement** (low-concurrency, text-only, stable shapes), not a global VLM force-on.

## Experiment Setup

| Item | Value |
|---|---|
| Model | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (sha256-verified) |
| Hardware | single **H200**, servers run serialized (never co-resident) |
| SGLang | `0.0.0.dev1+g0c8049d9b` (system python3) |
| vLLM | `0.21.0` (conda env `/opt/miniconda3/envs/profiling`) |
| torch / CUDA | `2.11.0+cu130` / CUDA `13.0` (aligned across both frameworks) |
| Precision / TP | bfloat16 / TP=1 |
| Sampling | greedy (`temperature=0`, `top_p=1`) |

> Attention backends are **not** aligned (SGLang FlashInfer vs vLLM FlashAttention v3) — a *measured*
> variable, so any attention-kernel-level conclusion carries **confidence ceiling M**.

## Workloads

| Case | Shape | Concurrency | Purpose |
|---|---|---|---|
| **A** `caseA_short` | 128 → 128 | 1 | short latency; cleanest fixed-overhead case |
| **B** `caseB_longprefill` | 2048 → 128 | 1 | long prefill; chunk/prefill behavior (bimodal → ceiling M) |
| **C** `caseC_batched` | 512 → 128 | 16 | batched serving; concurrency path |
| **D** `caseD_decode` | 512 → 512 | 16 | decode-heavy sanity check |

Clean validation focuses on **Case A** (the actionable gap) and **Case C** (the batched boundary).

## Phase Status

| Phase | Purpose | Status |
|---|---|---|
| 0 — Equivalence | weights/tokenizer/greedy-output parity | ✅ Complete |
| 1 — Baseline | establish gap; isolate TTFT vs TPOT | ✅ Complete |
| 2 — Shaping / Variance gate | lock profilable cases | ✅ Complete |
| 3 — Profiling / Trace collection | SGLang + vLLM stage traces | ✅ Complete |
| 4 — Triage | per-case kernel/overlap/fuse + hypotheses | ✅ Complete |
| 5 — Validation | clean Case A/C validation | ✅ Complete for scoped A/C clean validation |
| v2 #2 — Default-overlap rebaseline | production-default overlap-ON Case A/C baseline + PCG re-test | ✅ Complete / PASS (`experiments/qwen3vl8b/v2/caseAC_rebaseline/results/`) |

### Round-2 track status (as of 2026-08-29)

| Track | Issue | Status |
|---|---|---|
| Default-overlap rebaseline | [#2](https://github.com/bowenwan6/sglang-vllm-profiler/issues/2) | ✅ Complete / PASS — closed |
| Qwen3.5 DeepStack question | [#9](https://github.com/bowenwan6/sglang-vllm-profiler/issues/9) | ✅ Answered `NOT_APPLICABLE_QWEN35` — **ready to close on the tracker** |
| Qwen3-VL BCG DeepStack fix | (spun out of #9) | ✅ Fixed + validated; upstream PR [#33726](https://github.com/sgl-project/sglang/pull/33726) open, approved, mergeable |
| Qwen3-VL image+text + CUDA IPC | [#4](https://github.com/bowenwan6/sglang-vllm-profiler/issues/4) | ⚠️ **PARTIAL — the active profiling priority.** Only `IMG_A_S0_ipc` completed (5/5 reps, 2 000 requests, TTFT p50 64.8 ms). PCG arm crashed; repeat / vLLM / no-IPC controls unrun. |
| Qwen3.5 SGLang-vs-vLLM transfer | [#3](https://github.com/bowenwan6/sglang-vllm-profiler/issues/3) | ❌ **Not run.** The DeepStack and GDN studies answer different questions. |
| Selective / default-on graph policy | [#5](https://github.com/bowenwan6/sglang-vllm-profiler/issues/5) | ❌ Blocked on #4. Must now distinguish **PCG** from **BCG** — they are different backends. |

## Directory Layout

Every data directory has one `qwen3vl8b/` subtree (the single experiment):

| Path | Contents |
|---|---|
| `experiments/qwen3vl8b/` | per-phase research artifacts: `phase0/`…`phase4/` (summaries, `raw/`, `metadata/`, `scripts/`), `env_snapshot.md`, `README.md`, `phase3/caseB_trace_issue.md` |
| `datasets/qwen3vl8b/` | canonical autobench JSONL (`caseA..D.jsonl`) — never regenerate mid-project |
| `traces/qwen3vl8b/` | raw torch-profiler traces (**Git LFS**): per case `sglang_{mapping,formal}/` (DECODE), `sglang_extend_{mapping,formal}/` (EXTEND), `vllm/{prefill_like,decode_like}/` |
| `analysis/qwen3vl8b/` | triage outputs: per-case `{extend,decode}_triage.md`, `breakdown.md`, `vllm_crosscheck.md`, `preliminary_observations.md`; global `hypotheses.md`, `ranked_recommendations.md`; `category_regex.md` |
| `reports/qwen3vl8b/` | human-facing reports: `01_current_status_report.md`, `03_profiling_analysis.md` |
| `logs/qwen3vl8b/` | infrastructure side-effects (server stderr, kernel-API trails) — consult on failure only |
| `configs/qwen3vl8b/` | reserved for Phase 5 sweep configs |
| `plan.md` | **active v2 source of truth** (short; current mainline + Round 2 roadmap). Full v1 plan archived at `experiments/qwen3vl8b/v1_archive_plan.md` |
| `experiments/qwen3vl8b/v2/` | Round 2 (v2) experiments: `caseAC_rebaseline/` (#2, ✅ complete) and `image_text_benchmarks/` (#4, ⚠️ partial — the active priority) |
| `experiments/qwen35_4b/` | **Qwen3.5-4B correctness sub-track — concluded.** DeepStack verdict `NOT_APPLICABLE_QWEN35`; GDN verdict `PASS_BCG_GDN_NOTABLE_GAP` (`gdn/final_report.md`). Unrelated to the Qwen3-VL-8B PCG capture-stream sub-track under `experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/`. |
| `experiments/qwen3vl_bcg_deepstack_fix/` | **Qwen3-VL BCG DeepStack replay-slot fix** — upstream PR [#33726](https://github.com/sgl-project/sglang/pull/33726). Start at [`upstream_handoff.md`](experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md); `results/m*/` hold the milestone evidence (M10 = post-merge smoke). The two `*submission*.md` files are superseded historical snapshots. |

## How To Read This Repo

0. **`plan.md`** — active v2 mainline + Round 2 roadmap (start here for *current* direction); v1 detail in `experiments/qwen3vl8b/v1_archive_plan.md`.
1. **`reports/qwen3vl8b/01_current_status_report.md`** — the narrative status + key findings (v1).
2. **`reports/qwen3vl8b/03_profiling_analysis.md`** — detailed Phase 4 per-case triage analysis.
3. **`analysis/qwen3vl8b/hypotheses.md`** — structured hypotheses (H1–H4) with evidence + confidence.
4. **`analysis/qwen3vl8b/ranked_recommendations.md`** — what to validate first, and why.
5. **Phase summaries** — `experiments/qwen3vl8b/phase{1,2,3}/summary.md` for baseline / shaping / trace inventory.
6. **Raw artifacts** — only when auditing (see Artifact Policy).

## Artifact Policy

- **Raw provenance is not edited.** Benchmark raw JSON (`experiments/qwen3vl8b/*/raw/`), trace metadata
  JSON (`experiments/qwen3vl8b/phase3/metadata/`), and triage tool output (`analysis/qwen3vl8b/**/*_raw.txt`)
  are append-only records of what was collected; their embedded paths/timestamps are historical and
  should not be hand-edited.
- **Traces are Git LFS.** Everything under `traces/qwen3vl8b/` (`*.gz`) and the kernel-API `*.log`
  files are stored via Git LFS.
- **Processed/deliverable docs** (summaries, `analysis/**` markdown, reports, `plan.md`, this README)
  are hand-edited and reviewed.

## Side Quests / Methodological Notes

1. **Measurement hygiene (KAPI logging).** Early exploratory SGLang runs enabled
   `SGLANG_KERNEL_API_LOGLEVEL=1`, which inflates latency; the early four-workload ratios are kept only
   as instrumentation-confounded exploratory provenance, not clean evidence. See
   [`experiments/qwen3vl8b/methodology_correction.md`](experiments/qwen3vl8b/methodology_correction.md).
2. **Case C warmup/variance.** A W500 side investigation surfaced batched warmup/variance sensitivity
   and motivated the clean interleaved rerun; its older cross-framework number is not the final result —
   the clean Case C conclusion (no material gap / no Case-A-like benefit) stands.
3. **Case B trace limitation.** Case B's SGLang EXTEND trace is unavailable, so Case B is excluded from
   the clean headline; this does not affect the Case A finding or the Case C boundary result. (Attention
   backend FlashInfer vs FA3 also carries a confidence ceiling on attention-kernel claims.)

## Next Step

The correctness detour is done: the Qwen3-VL BCG DeepStack bug is fixed and sitting in an approved,
mergeable upstream PR. **The profiling mainline resumes at #4.** Full execution plan and acceptance
gates in [`reports/2026-08-28_profiling_resumption_audit.md`](reports/2026-08-28_profiling_resumption_audit.md).

1. **#4 — finish IMG-A (active, next GPU work).** Only the `S0_ipc` arm ran. Pin a fresh environment
   manifest, run a small current-upstream image+PCG smoke to see whether the capture-stream assertion
   still reproduces, then complete the bracket `S0_ipc_repeat → V0_vllm → S0_noipc` even if PCG stays
   excluded. **Do not start IMG-B/C** until IMG-A has drift, framework-anchor, and IPC controls.
   Image+text conclusions stay separate from the text-only (#2) findings, and the **CUDA-IPC transport**
   benefit stays separate from the **PCG** prefill-graph lever.
2. **#3 — the real Qwen3.5 transfer check** (parallel, after a common environment pin): clean Case A/C,
   SGLang default vs the supported graph lever vs a vLLM anchor. Note the old Qwen3-VL PCG lever may
   not be valid for Qwen3.5 — its supported route is BCG unless a source audit proves otherwise.
3. **#5 — graph-enablement policy** (after #4): build the backend × modality × load matrix, and decide
   PCG vs BCG explicitly. **BCG must not silently replace the PCG arm** — different backends.
4. **Tracker hygiene (no GPU):** close #9 as `NOT_APPLICABLE_QWEN35` with a cross-link to the
   Qwen3-VL fix evidence; post a refreshed checklist on #1.
