# SGLang vs vLLM Profiling — Active Plan (v2)

> **Active v2 source of truth.** Short by design: it states the current mainline and the v2 roadmap.
> The full v1 (Phase 0–5) plan is archived at `experiments/qwen3vl8b/v1_archive_plan.md`.
> Experiment: `qwen3vl8b` · `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` · single H200 · TP=1 · bf16 · greedy.

---

## 1. Current Mainline / Validated v1 Finding

- **Case A TTFT gap is real.** Low-concurrency / short-latency `128→128, c=1`: clean (no-KAPI,
  no-profiler) SGLang TTFT ~**19.2 ms** vs vLLM **13–14 ms**; **TPOT unchanged** → the gap is on the
  first-token / prefill side, not decode.
- **Cause direction.** SGLang detects Qwen3-VL as multimodal/VLM and **auto-disables the prefill/extend
  piecewise CUDA graph**, so low-concurrency prefill pays per-launch dispatch overhead that vLLM
  (graph/compile-covered) does not.
- **Clean intervention.** Forcing the graph on (`--enforce-piecewise-cuda-graph`) drops Case A TTFT to
  **11.7–13.4 ms**, TPOT unchanged, 0 failures → reaches the vLLM TTFT range. **It is a testing lever,
  not a production fix** (S2 CV ~10–12% → no stable-superiority claim).
- **Case C boundary (clean).** At `512→128, c=16` batched, SGLang ≈ vLLM ≈ **~190 ms** — **no material
  TTFT gap and no Case-A-like median improvement**. The effect is workload-shape-dependent.
- **GEMM is shared cost.** Both frameworks spend 72–86% of GPU time in the same `nvjet_sm90_*` FP8 GEMM
  family → GEMM is a shared absolute cost, **not** the SGLang↔vLLM differentiator.

## 2. What Must NOT Be Used As Headline

- **Phase 1 four-case ratios (4.89× / 3.20× / 1.32× / 1.33×)** and **Phase 2 Case C W500** → KAPI-
  confounded exploratory provenance only (see `experiments/qwen3vl8b/methodology_correction.md`).
- **`--disable-overlap-schedule`** → v1 *selected baseline / ablation*, **not** the production-default
  headline baseline. (This is exactly what v2 #2 fixes.)
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
| **2** | **Default-overlap Qwen3-VL rebaseline** | **P0 (foundational)** | Production-default overlap-ON Case A/C baseline; does PCG still help? | **NEXT — protocol drafting** |
| 3 | Qwen3.5 VL-model profiling | P1 | Same clean methodology on Qwen3.5; does the PCG finding transfer? | planned (after #2) |
| 4 | Qwen3-VL image+text + CUDA IPC | P1 | Image+text behavior; separate from text-only conclusions | planned (after #2) |
| 5 | Selective/default-on PCG PR plan | P2 | Minimum safe exception in VLM auto-disable + guards + fallback | planned (needs #2, #4) |

## 5. Immediate Next Step

**Issue #2 protocol** — `experiments/qwen3vl8b/v2/caseAC_rebaseline/protocol.md`.
Status: **protocol drafting / pending approval before execution.** No server, no benchmark, no runs until
approved.

## 6. Artifact Rules

- v2 results go **only** under `experiments/qwen3vl8b/v2/...` and `logs/qwen3vl8b/v2/...`. Never overwrite
  v1 Phase 0–5 artifacts.
- v1 raw JSON, traces, logs, scripts, and SGLang source are **not modified**.
- **Clean headline runs forbid KAPI/profiler:** never set `SGLANG_KERNEL_API_LOGLEVEL` /
  `SGLANG_KERNEL_API_LOGDEST`; no profiler. Servers run serialized (never co-resident).
- Every run records: GPU id, exact flags, framework versions, model snapshot, dataset sha256, warmup/
  reps/num-prompts, failures/error rate, and the KAPI/profiler-disabled confirmation.
