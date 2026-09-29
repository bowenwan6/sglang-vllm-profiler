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

*Single H200 · TP=1 · bfloat16 · greedy · text-only and image+text paths*

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
   **Text-only Case A/C (#2) and the image+text arm (#4, round 3) are complete.** The next question,
   whether a CUDA graph pays inside the vision encoder (Q3), is pre-registered on branch
   [`exp/q3-vit-graph`](https://github.com/bowenwan6/sglang-vllm-profiler/tree/exp/q3-vit-graph/experiments/qwen3vl8b/q3_vit_graph).
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
   **[sgl-project/sglang#33726](https://github.com/sgl-project/sglang/pull/33726)** (open and
   approved; it needs current upstream `main` merged in before it can land). Current state:
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
>
> This is the round-1 stack. Round 3 pins its own in
> [`experiments/qwen3vl8b/v3_issue4/manifest.md`](experiments/qwen3vl8b/v3_issue4/manifest.md).

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

### Track status (as of 2026-09-29)

| Track | Issue | Status |
|---|---|---|
| Default-overlap rebaseline | [#2](https://github.com/bowenwan6/sglang-vllm-profiler/issues/2) | ✅ Complete / PASS — closed |
| Qwen3.5 DeepStack question | [#9](https://github.com/bowenwan6/sglang-vllm-profiler/issues/9) | ✅ **Closed 2026-09-03** — verdict `NOT_APPLICABLE_QWEN35` ([conclusion](experiments/qwen35_4b/issue9_conclusion.md)) |
| Qwen3-VL BCG DeepStack fix | (spun out of #9) | ✅ Fixed + validated; upstream PR [#33726](https://github.com/sgl-project/sglang/pull/33726) open and approved. It needs current upstream `main` merged in (the last check showed a conflict in `prefill_cuda_graph_runner.py`), then a fresh smoke run |
| Qwen3-VL image+text + CUDA IPC | [#4](https://github.com/bowenwan6/sglang-vllm-profiler/issues/4) | ✅ **Measured and reported** — [`issue4_v3_report.pdf`](experiments/qwen3vl8b/v3_issue4/issue4_v3_report.pdf) (7 pp). **Transport `cuda_ipc` is worth −28.2% of TTFT** and is not the default. The prefill graph pays **−16.3% at 256×256 and −14.0% at 360p**, nothing measurable at 720p+; what it recovers in *ms* is set by prefill token count almost regardless of composition, the *percentage* by composition. On a mixed stream the net stays positive to a **43–59% image share** on TTFT and further on e2e. Against a **real 1M-request production size distribution**, though, only **15.2% of requests** land where a material win was measured ([`workload_realism.md`](experiments/qwen3vl8b/v3_issue4/workload_realism.md)) — the deliverable is the curve, not a threshold. SGLang is **36.9% faster than vLLM** on the image path, reversing #2's text-only result. |
| Qwen3.5 SGLang-vs-vLLM transfer | [#3](https://github.com/bowenwan6/sglang-vllm-profiler/issues/3) | ❌ **Not run.** The DeepStack and GDN studies answer different questions. |
| Selective / default-on graph policy | [#5](https://github.com/bowenwan6/sglang-vllm-profiler/issues/5) | 🟡 **Partly answered by #4 v3.** The backend sweep exists: BCG pays below ~250 visual tokens and is neutral above; PCG is **unmeasurable on current upstream** (92% eager fallback, [`pcg_eager_fallback_finding.md`](experiments/qwen3vl8b/v3_issue4/pcg_eager_fallback_finding.md)). What #5 still needs is the concurrency axis. |
| ViT CUDA graph (Q3) | — | 🟡 **Pre-registered 2026-09-29** on branch [`exp/q3-vit-graph`](https://github.com/bowenwan6/sglang-vllm-profiler/tree/exp/q3-vit-graph/experiments/qwen3vl8b/q3_vit_graph): does capturing the vision encoder in a CUDA graph pay, and can the gain be predicted from an eager trace alone? |

## Directory Layout

| Path | Contents |
|---|---|
| `plan.md` | **Research source of truth**: current mainline, roadmap and the record of every sub-track (§1–§12). |
| `datasets/qwen3vl8b/` | Canonical text workloads `caseA..D.jsonl`; their sha256 is pinned in each run's metadata. Never regenerated mid-project. |
| `tools/radix/setup_node.sh` | Rebuilds the profiling environment on an SGLang community (RADIX) GPU node. |
| `experiments/qwen3vl8b/v1/` | **Round 1 — Phases 0–5.** `phase0/`…`phase5/` (summaries, scripts, run metadata), `analysis/` (Phase 4 triage, Phase 5 launch-gap analysis), `traces/` (Phase 3 torch-profiler traces, Git LFS), `logs/` (server and orchestrator logs, Git LFS), `reports/` (v1 narrative reports), `v1_archive_plan.md` (the full v1 plan). |
| `experiments/qwen3vl8b/v2/` | **Round 2.** `caseAC_rebaseline/` (#2, production-default rebaseline, ✅) and `image_text_benchmarks/` (first #4 attempt, superseded by round 3; its `debug_pcg_capture_stream/root_cause/` holds the PCG capture-stream root cause, `plan.md` §4). |
| `experiments/qwen3vl8b/v3_issue4/` | **Round 3 — #4 image+text + CUDA IPC (✅ 2026-09-06).** `manifest.md` (frozen stack), `progress.md` (step log), `issue4_v3_report.pdf`, `imgA_report.md`, `imgR_report.md`, `q1_report.md`, `q2_report.md`, `workload_realism.md`, `pcg_eager_fallback_finding.md`, `figures/`, `scripts/`. |
| `experiments/qwen35_4b/` | **Qwen3.5-4B correctness sub-track — concluded.** DeepStack verdict `NOT_APPLICABLE_QWEN35` ([`issue9_conclusion.md`](experiments/qwen35_4b/issue9_conclusion.md)); GDN verdict `PASS_BCG_GDN_NOTABLE_GAP` ([`gdn/final_report.md`](experiments/qwen35_4b/gdn/final_report.md)). |
| `experiments/qwen3vl_bcg_deepstack_fix/` | **Qwen3-VL BCG DeepStack replay-slot fix** — upstream PR [#33726](https://github.com/sgl-project/sglang/pull/33726). Start at [`upstream_handoff.md`](experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md); `results/` holds the r2 baseline and the m* milestone evidence (m10 = post-merge smoke). The two `*submission*.md` files are superseded snapshots. |

## How To Read This Repo

0. **`plan.md`** — current direction and the record of every track. Start here.
1. **Main Findings and Track status above** — the one-page summary.
2. **[`experiments/qwen3vl8b/v3_issue4/issue4_v3_report.pdf`](experiments/qwen3vl8b/v3_issue4/issue4_v3_report.pdf)**
   — the image+text (#4) report; `q1_report.md`, `q2_report.md` and `workload_realism.md` next to it
   carry the follow-on questions.
3. **[`experiments/qwen3vl8b/v2/caseAC_rebaseline/`](experiments/qwen3vl8b/v2/caseAC_rebaseline/)** — the
   production-default text-only numbers behind Main Findings 1, 3 and 4.
4. **Round 1** — [`v1/reports/03_profiling_analysis.md`](experiments/qwen3vl8b/v1/reports/03_profiling_analysis.md)
   (Phase 4 triage), [`v1/analysis/hypotheses.md`](experiments/qwen3vl8b/v1/analysis/hypotheses.md) and
   [`ranked_recommendations.md`](experiments/qwen3vl8b/v1/analysis/ranked_recommendations.md), and the
   phase summaries `experiments/qwen3vl8b/v1/phase{1,2,3}/summary.md` and `v1/phase5/*/summary.md`.
   [`v1/reports/01_current_status_report.md`](experiments/qwen3vl8b/v1/reports/01_current_status_report.md)
   is the v1-era status report (overlap-OFF baseline).
5. **Raw artifacts** — only when auditing (see Artifact Policy).

## Artifact Policy

- **Raw provenance is not edited.** Run metadata (`experiments/qwen3vl8b/v1/phase*/raw/*_meta.json`,
  `v1/phase3/metadata/`) and triage tool output (`v1/analysis/**/*_raw.txt`) are append-only records;
  their embedded paths and timestamps are historical and predate the 2026-09-29 move into `v1/`.
- **Raw outputs stay out of git.** Per-request bench JSON, server logs and profiler dumps are not
  committed (`**/raw/` is ignored); commit summaries and aggregate results instead.
- **Removed from the tree on 2026-09-29, restorable from tag
  [`archive/pre-cleanup-2026-09-29`](https://github.com/bowenwan6/sglang-vllm-profiler/tree/archive/pre-cleanup-2026-09-29):** round 1's per-request bench JSON (phases 1, 2 and 5; 146 files,
  502 MiB) and about 1 GiB of kernel-API debug logs. The reported numbers live in the per-run
  `summary.md` / `results.json` files, which stay. To restore, e.g., Phase 5's raw files (the tag uses the
  pre-move paths):
  ```bash
  git archive archive/pre-cleanup-2026-09-29 experiments/qwen3vl8b/phase5 | tar -x -C /tmp/restore
  ```
- **Git LFS** stores `*.gz` (torch-profiler traces) and `*.log`.
- **Processed and deliverable docs** (summaries, analysis markdown, reports, `plan.md`, this README) are
  hand-edited and reviewed.

## Side Quests / Methodological Notes

1. **Measurement hygiene (KAPI logging).** Early exploratory SGLang runs enabled
   `SGLANG_KERNEL_API_LOGLEVEL=1`, which inflates latency; the early four-workload ratios are kept only
   as instrumentation-confounded exploratory provenance, not clean evidence. See
   [`experiments/qwen3vl8b/v1/methodology_correction.md`](experiments/qwen3vl8b/v1/methodology_correction.md).
2. **Case C warmup/variance.** A W500 side investigation surfaced batched warmup/variance sensitivity
   and motivated the clean interleaved rerun; its older cross-framework number is not the final result —
   the clean Case C conclusion (no material gap / no Case-A-like benefit) stands.
3. **Case B trace limitation.** Case B's SGLang EXTEND trace is unavailable, so Case B is excluded from
   the clean headline; this does not affect the Case A finding or the Case C boundary result. (Attention
   backend FlashInfer vs FA3 also carries a confidence ceiling on attention-kernel claims.)

## Next Step

- **Q3 — ViT CUDA graph:** pre-registered 2026-09-29 on branch
  [`exp/q3-vit-graph`](https://github.com/bowenwan6/sglang-vllm-profiler/tree/exp/q3-vit-graph/experiments/qwen3vl8b/q3_vit_graph) (`plan.md` §12.6 on that branch); runs on an SGLang community GPU node.
- **PR #33726:** merge current upstream `main` into the PR branch, re-run the dense and MoE smokes, and
  report the result on the PR.
- **#5 — graph-enablement policy:** #4 v3 supplied the backend sweep; the concurrency (load) axis is still
  missing. Decide PCG vs BCG explicitly — **BCG must not silently replace the PCG arm** (different backends).
- **#3 — the Qwen3.5 transfer check** (after a common environment pin): clean Case A/C, SGLang default vs
  the supported graph lever vs a vLLM anchor. The old Qwen3-VL PCG lever may not be valid for Qwen3.5 —
  its supported route is BCG unless a source audit proves otherwise.
- **Tracker hygiene (no GPU):** post a refreshed checklist on #1 and re-scope #5.

> ⚠️ **Read [`plan.md` §3.5](plan.md) before running graph experiments.** Upstream restructured the
> CUDA-graph flags: `--enforce-piecewise-cuda-graph` is now a deprecated alias for
> `--cuda-graph-backend-prefill=tc_piecewise`, **breakable (BCG) is the default prefill backend
> on CUDA**, and PR #33726 adds Qwen3-VL to the breakable allowlist — so Qwen3-VL's default flips
> from *no prefill graph* to *BCG-on* when it merges. The two-arm `default` vs `+PCG` design no
> longer spans the space.
