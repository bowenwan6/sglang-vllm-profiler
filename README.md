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

1. **Clean Case A exposes an actionable TTFT gap.** In an uninstrumented benchmark, SGLang's default
   Case A (128→128, c=1) TTFT is ~**19.2 ms** vs vLLM's **13–14 ms**.
2. **Profiling points away from GEMM speed and toward prefill graph coverage.** Both frameworks spend
   72–86% of GPU time in the *same* `nvjet_sm90_*` FP8 GEMM family (so it isn't a slow-SGLang-GEMM
   problem); config + source audit shows SGLang **disables prefill piecewise CUDA graph by default for
   this VLM** (Qwen3-VL multimodal auto-disable), leaving prefill on an eager dispatch path.
3. **A clean controlled intervention materially reduces Case A TTFT.** Forcing prefill piecewise CUDA
   graph (`--enforce-piecewise-cuda-graph`) drops Case A TTFT to **11.7–13.4 ms** with **TPOT
   unchanged**, 0 failures, into the vLLM TTFT range — validating graph coverage as a real, actionable
   Case-A contributor. (Testing lever, not a production fix; S2 CV ~10–12%, so stable superiority over
   vLLM is not claimed.)
4. **Clean Case C is a boundary result.** At c=16 batched, the same intervention yields **no material
   median TTFT gap and no Case-A-like benefit** (SGLang ≈ vLLM ≈ 190 ms). The optimization is
   workload-shape-dependent → favor **selective enablement** (low-concurrency, text-only, stable
   shapes), not a global VLM force-on.
5. **Methodological note (provenance).** Early Phase 1/2 four-workload ratios (A 4.89× / B 3.20× /
   C 1.32× / D 1.33×) were collected with **SGLang-only KAPI logging** and are retained only as
   instrumentation-confounded exploratory provenance; Case B/D clean re-baselining is pending. See
   [`experiments/qwen3vl8b/methodology_correction.md`](experiments/qwen3vl8b/methodology_correction.md).

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

Early Phase-1 SGLang/vLLM TTFT p50 ratios (A 4.89× · B 3.20× · C 1.32× · D 1.33×) are
**instrumentation-confounded exploratory measurements** (SGLang-only KAPI logging), not clean results —
see the Main Findings banner and `methodology_correction.md`.

## Phase Status

| Phase | Purpose | Status |
|---|---|---|
| 0 — Equivalence | weights/tokenizer/greedy-output parity | ✅ PASS |
| 1 — Baseline | establish gap; isolate TTFT vs TPOT | ✅ complete — ⚠️ **KAPI-confounded** (provenance only) |
| 2 — Shaping / Variance gate | lock profilable cases (incl. Case C W500 probe) | ✅ complete — ⚠️ Case C W500 **KAPI-confounded** |
| 3 — Profiling / Trace collection | SGLang DECODE+EXTEND, vLLM prefill/decode | ✅ complete (Case B SGLang EXTEND unavailable — caveat) |
| 4 — Triage | per-case kernel/overlap/fuse + hypotheses | ✅ complete |
| 5 — Validation | clean H1 validation | 🔄 in progress — Case A clean H1 supported; Case C clean correction done; B/D clean baseline pending |

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

## Current Caveats

- **Attention backend mismatch** (FlashInfer vs FA3) → all attention-kernel-level findings carry **ceiling M**.
- **Case B** is bimodal in both frameworks, and its **SGLang EXTEND trace is unavailable** (corrupt +
  un-recapturable under the profiler's long-prefill stage mechanism — see
  `experiments/qwen3vl8b/phase3/caseB_trace_issue.md`). All Case B conclusions carry **ceiling M**.
- **Early four-workload ratios are instrumentation-confounded** (SGLang-only KAPI logging) — exploratory
  discovery signals, not clean conclusions. See `experiments/qwen3vl8b/methodology_correction.md`.
- **H1 is validated for clean Case A only** (materially lower TTFT, TPOT unchanged); it is a
  testing-lever result, **not a production fix**, and **not established for Case C** (no Case-A-like
  benefit under clean test).
- **Case B / Case D** have no clean cross-framework baseline yet → not part of any headline.

## Next Step

1. **Clean (no-KAPI, no-profiler) baseline for Case B and Case D** before any four-workload TTFT-ratio
   headline.
2. **Production-safe design discussion** for low-concurrency / text-only VLM prefill graph enablement
   (the Case-A locus) — without the `--enforce-piecewise-cuda-graph` testing lever and **without
   claiming a global VLM fix**. (Docs only; no source changes this round.)
3. H2 (`nvjet → CUTLASS-FP8`, SGLang PR #22392) as a separate **absolute-speed** track. See
   `analysis/qwen3vl8b/ranked_recommendations.md`.
