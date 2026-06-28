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
| **4** | **Qwen3-VL image+text + CUDA IPC** | **P1 — PCG path closed; non-PCG IMG-A resume pending** | Image+text behavior + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`; separate from text-only conclusions | Generator `<\|video_pad\|>` bug merged upstream as `07f326c184` (#26864). Profiler uses `/data/sglang-pr` on `main` (HEAD `62c505a196`). Stage 4.1 smoke ✅. Stage 4.2 IMG-A: S0_ipc 5/5 reps clean (TTFT p50 64.8 ms); S2_ipc_pcg deterministically crashes with `AssertionError: PCG capture stream is not set`. PCG debug (`debug_pcg_capture_stream/conclusion.md`) shows E1 text-only PCG OK, E2a image+IPC+PCG+n=32 ASSERT, E3 image+noIPC+PCG+n=32 ASSERT → **fault is VLM image + PCG**, **IPC not required**. Upstream auto-disables PCG for VLMs (`server_args.py:1374-1376`); `--enforce-piecewise-cuda-graph` is a "for testing" override and crashes deterministically on Qwen3-VL. **PCG benefit (Q2) cannot be measured on this HEAD without an upstream change.** Next: file informational upstream SGLang issue with the n=32 minimal repro; resume IMG-A with non-PCG variants only (`S0_ipc_repeat → V0_vllm → S0_noipc`) to recover Q1/Q3 + bracket drift. No SGLang PR. |
| 3 | Qwen3.5 VL-model profiling | P1 | Same clean methodology on Qwen3.5; does the PCG finding transfer? | next candidate (parallel/after #2; transfer check) |
| 5 | Selective/default-on PCG PR plan | P2 | Minimum safe exception in VLM auto-disable + guards + fallback | planned (needs #4) |

## 5. Immediate Next Step

**Issue #2 is COMPLETE** (clean run, GPU 1, 0 failures; results under
`experiments/qwen3vl8b/v2/caseAC_rebaseline/results/`).

**Issue #4 is PARTIAL — generator unblocked, PCG path blocked by upstream
SGLang capture-stream assertion.** The benchmark-generator `<|video_pad|>` bug is
merged upstream as `07f326c184` (#26864); profiler runs use `/data/sglang-pr` on
`main` (HEAD `62c505a196`, 2026-06-08). V1 audit + V2 serving repro both PASS;
Stage 4.1 fixed-generator smoke PASS; Stage 4.2 IMG-A:

- **`IMG_A_S0_ipc`** ✅ 5/5 reps, 2000 requests, 0 failures, TTFT p50 64.8 ms
  (87.2/65.1/63.7/63.6/64.8 across reps), TPOT 5.23 ms, throughput 175 tok/s,
  no forbidden-token errors. **Single clean datapoint — not yet a headline** (no
  bracket counterpart, no anchor).
- **`IMG_A_S2_ipc_pcg`** ❌ rep1 server crash with
  `AssertionError: PCG capture stream is not set, please check if runtime
  recompilation happened` in `srt/compilation/cuda_piecewise_backend.py:171`,
  triggered on first prefill of Qwen3-VL with `--enforce-piecewise-cuda-graph`
  + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`. Not generator-related; fix gate stayed
  green.
- **`IMG_A_S0_ipc_repeat` / `IMG_A_V0_vllm` / `IMG_A_S0_noipc`** — skipped per
  protocol §9 stop condition.

The prior pre-fix partial IMG-A is **invalid for performance conclusions** and
kept as historical record only.

PCG debug closed (`v2/image_text_benchmarks/debug_pcg_capture_stream/conclusion.md`):
**E1** text-only + PCG `OK` (upstream main PCG not broadly regressed);
**E2a** image + IPC + PCG @ n=32 `PCG_CAPTURE_STREAM_ASSERT` (Stage 4.2 crash
deterministically reproduced at minimal cost ~30 s GPU);
**E3** image + **no IPC** + PCG @ n=32 also `PCG_CAPTURE_STREAM_ASSERT` with
identical signature → **IPC is not a required trigger**. The fault is **VLM
image path + PCG**. Upstream auto-disables PCG for VLMs
(`server_args.py:1374-1376`) and `--enforce-piecewise-cuda-graph` is a
"for testing" override that bypasses the safety; the assertion at
`cuda_piecewise_backend.py:170-172` is a defensive guard. The generator fix
gate stayed green throughout — this is not a generator bug.

**Next step for #4 (outside this debug branch):**
- File an informational upstream SGLang issue with the E2a n=32 minimal
  repro recipe — ask for a graceful CUDA fallback (mirroring the existing
  HIP fallback at `cuda_piecewise_backend.py:163-169`) or a loud warning on
  VLM + `--enforce-piecewise-cuda-graph`. **No SGLang PR** at this stage.
- Resume IMG-A with the **non-PCG** variants only
  (`S0_ipc_repeat → V0_vllm → S0_noipc`) to recover bracket drift, IPC
  benefit (Q3), and vLLM anchor (Q1). `S2_ipc_pcg` stays explicitly
  excluded with rationale: PCG is upstream-auto-disabled for VLMs and the
  override required to force it crashes deterministically on this HEAD.
  PCG benefit (Q2) for image+text is **not measurable on this upstream
  HEAD without an upstream change**, and the Case-A text-only PCG finding
  from #2 cannot be transferred to image+text within #4 here.
- Do **not** proceed to IMG-B / IMG-C until IMG-A yields headline-quality
  data on the non-PCG variants.
Fixed-generator recovery plan at `v2/image_text_benchmarks/fixed_generator_plan.md`
remains the source of truth for the IMG-A resume.

Then: **#3** (Qwen3.5 transfer check, parallel/after) → **#5** (selective/default-on PCG PR, needs #4's
image evidence).

## 5a. Sub-track — PCG capture-stream root-cause (active, branch `debug/v2-imgA-pcg-capture-stream-fix`)

Server was rebuilt on 2026-06-28; environment re-set up from scratch (system sglang
at `/sgl-workspace/sglang`, profiling conda env at `/opt/miniconda3/envs/profiling`,
Qwen3-VL-8B-Instruct snapshot `0c351dd` re-downloaded). On the rebuilt env the bug
**reproduces deterministically** on system sglang HEAD `da802dd` (newer than the
prior debug's HEAD `62c505a196`), GPU 0:

- Repro recipe unchanged from prior E2a: image 720p, 1 image, c=1, n=32, warmup=30,
  output_len=128, IPC on, PCG on, ≈80 s wall-clock from launch to assertion.
- Crash now observable as fired at the **warmup→bench-phase boundary**: ~30 warmup
  POSTs complete successfully with `cuda graph: True`, then the very next forward
  hits the assertion at `cuda_piecewise_backend.py:171` in
  `qwen3_vl.forward` → fx `submod_0` (layer-0 piecewise submodule). The failing
  call is the first prefix-cache-hit prefill (`#new-seq: 1, #new-token: 1,
  #cached-token: 1020`).
- **New upstream surface observation:** the VLM PCG auto-disable is now gated by
  a per-model knob `ModelConfig.is_multimodal_piecewise_cuda_graph_supported`
  (`server_args.py:3145-3146`). Same selective-enablement shape Issue #5 plans,
  so the fix surface has shifted under us.

Sub-track plan (R0 → R5) lives at
[`v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/README.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/README.md).

- **R0** — record plan + findings to date (this update).
- **R1** — Dynamo guard instrumentation via `TORCH_LOGS=recompiles_verbose,dynamic,guards`
  to capture the exact recompile reason.
- **R2** — source-level instrumentation in `cuda_piecewise_backend.py` (patch file
  under `root_cause/patches/`, applied + reverted around the run).
- **R3** — ranked hypotheses + minimal differential experiments (one axis at a time).
- **R4** — fix proposal (X defensive fallback / Y broaden warmup capture / Z per-model
  opt-in) + validation via E2a PASS + stretch IMG-A S2_ipc_pcg run.
- **R5** — upstream issue/PR draft (filing is user-triggered, not automatic).

Sub-track does **not** touch v1 Phase 0–5 artifacts, does **not** restart
#4 IMG-A non-PCG resume in parallel (that remains queued), and does **not**
modify `--enforce-piecewise-cuda-graph` defaults or the
`is_multimodal_piecewise_cuda_graph_supported` table without explicit approval
(that is Issue #5's scope). All sglang source modifications are kept as
revertable `.patch` files under `root_cause/patches/`.

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
