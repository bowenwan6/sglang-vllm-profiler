# SGLang vs vLLM Profiling — Active Plan (v2)

> **Active v2 source of truth.** Short by design: it states the current mainline and the v2 roadmap.
> The full v1 (Phase 0–5) plan is archived at `experiments/qwen3vl8b/v1_archive_plan.md`.
> Experiment: `qwen3vl8b` · `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` · single H200 · TP=1 · bf16 · greedy.

---

## 1. Current Mainline (v1 finding + v2 #2 production-default confirmation)

- **Case A TTFT gap is real on the production default.** v2 #2 (clean, GPU 1, 0 failures) on **SGLang
  default = overlap-ON**: Case A `128→128, c=1` SGLang TTFT **21.94 ms** vs vLLM **13.12 ms**; **TPOT
  unchanged** → gap is on the first-token / prefill side, not decode.
- **PCG still helps under the production default.** `--enforce-piecewise-cuda-graph` drops Case A TTFT
  **21.94 → 14.04 ms (−36%)**, TPOT flat (5.47 ms), 0 failures → into the vLLM band. The v1 finding is
  **not an artifact of the overlap-OFF baseline.** Still a **testing lever, not a production fix**.
- **`--disable-overlap-schedule` is ablation-only.** v2 ran it for v1 comparability: 19.07 ms TTFT —
  *lower* TTFT than overlap-ON but worse TPOT (5.87 vs 5.47) and throughput (167 vs 179 tok/s). So v1's
  no-overlap baseline **understated** the production TTFT gap (the default→vLLM gap 21.9 vs 13.1 is
  *larger* than v1's overlap-OFF gap 19.1 vs 13.1).
- **Cause direction.** SGLang detects Qwen3-VL as multimodal/VLM and **auto-disables the prefill/extend
  piecewise CUDA graph**, so low-concurrency prefill pays per-launch dispatch overhead that vLLM
  (graph/compile-covered) does not.
- **Case C boundary confirmed on the production default.** v2 #2 interleaved `512→128, c=16`: SGLang
  default pooled **204.8 ms**, +PCG **230.6 ms**, vLLM **215.7 ms** (batched CV ~14–15%) → **no
  material gap and no Case-A-like PCG benefit**. The effect is workload-shape-dependent.
- **GEMM is shared cost.** Both frameworks spend 72–86% of GPU time in the same `nvjet_sm90_*` FP8 GEMM
  family → GEMM is a shared absolute cost, **not** the SGLang↔vLLM differentiator.

## 2. What Must NOT Be Used As Headline

- **Phase 1 four-case ratios (4.89× / 3.20× / 1.32× / 1.33×)** and **Phase 2 Case C W500** → KAPI-
  confounded exploratory provenance only (see `experiments/qwen3vl8b/methodology_correction.md`).
- **`--disable-overlap-schedule`** → *ablation only*, **not** the production-default headline baseline.
  (v2 #2 fixed this: the headline is now SGLang default overlap-ON.)
- **`--enforce-piecewise-cuda-graph`** → validation/testing lever, **not** production behavior.
- **Case B** → SGLang EXTEND trace unavailable → excluded from any headline.

## 3. Why v2 Exists

The v1 finding is sound but its Case-A baseline ran with `--disable-overlap-schedule` (overlap-OFF),
which is not the production default. To turn this into an upstream SGLang recommendation, v2 must:

1. Re-run a **production-default overlap-ON** baseline and re-test whether PCG still helps (#2).
2. Add **image+text** workloads, with `SGLANG_USE_CUDA_IPC_TRANSPORT=1` on SGLang image runs (#4).
3. Replicate on the **Qwen3.5** VL model (the model's name is "Qwen3.5"; there is no `-VL` suffix) (#3).
4. Design **selective/default-on PCG** for safe Qwen3-VL cases — not a global force-on of the lever (#5).

## 4. v2 Roadmap / Issues #1–#5

Source: GitHub issues #1–#5 on `bowenwan6/sglang-vllm-profiler` (@JustinTong0323, 2026-05-27).
Dependency order: **#2 → {#4, #3 parallel} → #5 → report restructure**.

| # | Title | Priority | Goal | Status |
|---|---|---|---|---|
| 1 | Tracking: next-round follow-ups | meta | Umbrella; final deliverable separates baseline / ablation / Qwen3.5 / image+text / PR proposal | open (tracking) |
| **2** | **Default-overlap Qwen3-VL rebaseline** | **P0 (foundational)** | Production-default overlap-ON Case A/C baseline; does PCG still help? | **✅ COMPLETE / PASS** (results under `v2/caseAC_rebaseline/results/`) |
| **4** | **Qwen3-VL image+text + CUDA IPC** | **P1 — UNBLOCKED / recovery plan drafted** | Image+text behavior + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`; separate from text-only conclusions | Generator `<\|video_pad\|>` bug fixed in `/data/sglang-pr` (`fix/mm-benchmark-special-tokens` @ `78e6c03e2`). **V1 audit PASS + V2 serving repro PASS** (`debug_video_pad/validation_plan.md`). **Fixed-generator recovery plan** at `v2/image_text_benchmarks/fixed_generator_plan.md` (gated Stages 4.1 smoke → 4.2 IMG-A → 4.3 IMG-B/C). Prior partial IMG-A is invalid for perf conclusions. |
| 3 | Qwen3.5 VL-model profiling | P1 | Same clean methodology on Qwen3.5; does the PCG finding transfer? | next candidate (parallel/after #2; transfer check) |
| 5 | Selective/default-on PCG PR plan | P2 | Minimum safe exception in VLM auto-disable + guards + fallback | planned (needs #4) |

## 5. Immediate Next Step

**Issue #2 is COMPLETE** (clean run, GPU 1, 0 failures; results under
`experiments/qwen3vl8b/v2/caseAC_rebaseline/results/`).

**Issue #4 is UNBLOCKED.** The benchmark-generator `<|video_pad|>` bug is fixed in the
local SGLang clone at `/data/sglang-pr` (`fix/mm-benchmark-special-tokens` @ `78e6c03e2`).
V1 payload audit and V2 tiny serving repro both PASS
(`experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/validation_plan.md`).

The prior partial IMG-A is **invalid for performance conclusions** (3/5 reps of one
of five variants, with 2 failures in rep 3 under the buggy generator). It is kept as
historical record only.

**Next step for #4:** execute the fixed-generator recovery plan at
`experiments/qwen3vl8b/v2/image_text_benchmarks/fixed_generator_plan.md`. Gated
stages: 4.1 fixed-generator smoke → 4.2 IMG-A formal → 4.3 IMG-B/C decision (only if
IMG-A is clean). The fixed code is selected via
`PYTHONPATH=/data/sglang-pr/python` — `/sgl-workspace/sglang` is not modified.
Each result records `sglang.__file__` and SGLang commit SHA as provenance.

Then: **#3** (Qwen3.5 transfer check, parallel/after) → **#5** (selective/default-on PCG PR, needs #4's
image evidence).

## 6. Artifact Rules

- v2 results go **only** under `experiments/qwen3vl8b/v2/...` and `logs/qwen3vl8b/v2/...`. Never overwrite
  v1 Phase 0–5 artifacts.
- v1 raw JSON, traces, logs, scripts, and SGLang source are **not modified**.
- **Clean headline runs forbid KAPI/profiler:** never set `SGLANG_KERNEL_API_LOGLEVEL` /
  `SGLANG_KERNEL_API_LOGDEST`; no profiler. Servers run serialized (never co-resident).
- Every run records: GPU id, exact flags, framework versions, model snapshot, dataset sha256, warmup/
  reps/num-prompts, failures/error rate, and the KAPI/profiler-disabled confirmation.
- **Raw per-rep dumps and server logs are generated but NOT committed** unless explicitly approved
  (committed deliverables = summaries + aggregate `case*_results.json`). Raw lives in `results/raw/`,
  server logs in `logs/qwen3vl8b/v2/...`.
