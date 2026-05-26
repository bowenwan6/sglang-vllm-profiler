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

This repo holds one experiment, **`qwen3vl8b`**, that asks a focused question: **where does SGLang's
time-to-first-token (TTFT) gap versus vLLM come from?** The goal is not to declare a winner but to
attribute the gap to a specific stage (prefill vs decode), kernel family, and system path, and to
turn that into ranked, evidence-backed hypotheses for optimization.

The work is organized as a phase-gated pipeline (Phase 0 → 5): prove the two servers are comparable,
establish a baseline, shape/​de-noise the workloads, collect torch-profiler traces, triage them, and
(next) validate the top hypothesis.

## Main Findings

1. **Clean Case A exposes an actionable TTFT gap.** In an uninstrumented benchmark, SGLang's Case A
   (128→128, c=1; selected baseline `--disable-overlap-schedule`) TTFT is ~**19.2 ms** vs vLLM's
   **13–14 ms**, while TPOT is unchanged — i.e. the issue is on the first-token / prefill side.
2. **GPU kernel speed is not the differentiator.** Both frameworks spend 72–86% of GPU time in the
   *same* `nvjet_sm90_*` FP8 GEMM family — GEMM is a **shared absolute cost**, not what explains the
   Case A TTFT gap.
3. **The cause is SGLang's VLM prefill graph coverage; a clean intervention validates it.** For
   Qwen3-VL, SGLang disables prefill piecewise CUDA graph (VLM auto-disable). Forcing it on
   (`--enforce-piecewise-cuda-graph`) drops Case A TTFT to **11.7–13.4 ms**, TPOT unchanged, 0 failures
   — **reaching the vLLM TTFT range**. Prefill piecewise graph coverage is a **validated contributor**
   to Case A TTFT. (Testing lever, not a production fix; S2 CV ~10–12% → no claim of stable superiority.)
4. **Case C defines the boundary.** At c=16 batched (clean), the same intervention yields **no material
   TTFT gap and no Case-A-like median improvement** (SGLang ≈ vLLM ≈ 190 ms). The fix is
   workload-shape-dependent → favor **selective enablement** (low-concurrency, text-only, stable
   shapes), not a global VLM force-on.

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
| `plan.md` | the execution plan — single source of truth for methodology |

## How To Read This Repo

1. **`reports/qwen3vl8b/01_current_status_report.md`** — start here; the narrative status + key findings.
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

1. **Production-safe selective graph enablement** for low-concurrency / text-only / shape-stable VLM
   requests (the Case-A locus) — design work, no source changes here, and no global VLM force-on.
2. *(Optional)* broader clean cross-framework benchmarking (e.g. Case B/D) if a four-workload headline
   is later needed.
3. *(Optional)* H2 absolute-speed track (`nvjet → CUTLASS-FP8`, SGLang PR #22392), separate from the
   latency-gap question. See `analysis/qwen3vl8b/ranked_recommendations.md`.
