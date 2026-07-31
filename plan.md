# SGLang vs vLLM Profiling — Active Plan (v2)

> **v2 source of truth.** States the current mainline, the v2 roadmap, and the
> audited outcome of the Qwen3-VL PCG capture-stream investigation.
> The full v1 (Phase 0–5) plan is archived at
> `experiments/qwen3vl8b/v1_archive_plan.md`.
> Experiment: `qwen3vl8b` · `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` · single H200 ·
> TP=1 · bf16 · greedy.

---

## 1. Current mainline

- **Case A TTFT gap is real on the production default.** v2 #2 (clean, GPU 1,
  0 failures) on SGLang default = overlap-ON: Case A `128→128, c=1` SGLang
  TTFT **21.94 ms** vs vLLM **13.12 ms**; TPOT unchanged → gap is on the
  first-token / prefill side, not decode.
- **PCG still helps under the production default.** `--enforce-piecewise-cuda-graph`
  drops Case A TTFT **21.94 → 14.04 ms (−36 %)**, TPOT flat (5.47 ms),
  0 failures → into the vLLM band. The v1 finding is **not an artifact of
  the overlap-OFF baseline**. Still a testing lever, not a production fix.
- **`--disable-overlap-schedule` is ablation-only.** v2 ran it for v1
  comparability: 19.07 ms TTFT — *lower* TTFT than overlap-ON but worse TPOT
  (5.87 vs 5.47) and throughput (167 vs 179 tok/s). v1's no-overlap baseline
  **understated** the production TTFT gap.
- **Cause direction.** SGLang detects Qwen3-VL as multimodal / VLM and
  auto-disables the prefill / extend piecewise CUDA graph, so low-concurrency
  prefill pays per-launch dispatch overhead that vLLM (graph / compile-covered)
  does not.
- **Case C boundary confirmed on the production default.** v2 #2 interleaved
  `512→128, c=16`: SGLang default pooled **204.8 ms**, +PCG **230.6 ms**, vLLM
  **215.7 ms** (batched CV ~14–15 %) → no material gap and no Case-A-like PCG
  benefit. The effect is workload-shape-dependent.
- **GEMM is shared cost.** Both frameworks spend 72–86 % of GPU time in the
  same `nvjet_sm90_*` FP8 GEMM family → GEMM is a shared absolute cost, not
  the SGLang↔vLLM differentiator.
- **Qwen3-VL PCG capture-stream investigation concluded** (2026-07-29,
  Issue #4 sub-track): ✅ `CORRECTNESS_AND_SAFETY_FIX_PASS` +
  ⚠️ `PERFORMANCE_VALUE_PROMISING_BUT_NOT_FINAL` — audited details in §4.
  Upstream **SGLang PR #30868** (merged 2026-07-19) addresses the same root
  cause; the local fork branch is likely superseded and is not slated for a
  standalone upstream PR unchanged.

## 2. What must NOT be used as headline

- **Phase 1 four-case ratios** (4.89× / 3.20× / 1.32× / 1.33×) and
  **Phase 2 Case C W500** → KAPI-confounded exploratory provenance only
  (see `experiments/qwen3vl8b/methodology_correction.md`).
- **`--disable-overlap-schedule`** → ablation only, not the production-default
  headline baseline. (v2 #2 fixed this: the headline is now SGLang default
  overlap-ON.)
- **`--enforce-piecewise-cuda-graph`** → validation / testing lever, not
  production behavior.
- **Case B** → SGLang EXTEND trace unavailable → excluded from any headline.
- **Fork-PCG image+text performance dominance** (Issue #4 sub-track) →
  **not established**; only correctness + safety and text-only non-regression
  are established. Do not cite the 22 %-margin cell or the "R6.4 strictly
  dominant" reading — both retracted under provenance audit (see §4).

## 3. v2 roadmap — Issues #1–#5

Source: GitHub issues #1–#5 on `bowenwan6/sglang-vllm-profiler`
(@JustinTong0323, 2026-05-27). Dependency order:
**#2 → {#4, #3 parallel} → #5 → report restructure**.

| # | Title | Priority | Status |
|---|---|---|---|
| 1 | Tracking: next-round follow-ups | meta | open (tracking) |
| **2** | **Default-overlap Qwen3-VL rebaseline** | **P0** | ✅ COMPLETE / PASS (results under `v2/caseAC_rebaseline/results/`) |
| **4** | **Qwen3-VL image+text + CUDA IPC** | **P1** | Sub-track investigation ✅ concluded 2026-07-29 (see §4). Root cause fixed on fork; upstream SGLang PR #30868 (merged 2026-07-19) addresses the same root cause. Non-PCG IMG-A resume (`S0_ipc_repeat → V0_vllm → S0_noipc`) remains queued under `v2/image_text_benchmarks/fixed_generator_plan.md`. |
| 3 | Qwen3.5 VL-model profiling | P1 | **active sub-track (correctness first, not perf transfer):** Qwen3.5-4B BCG DeepStack investigation on branch `debug/qwen35-4b-bcg-deepstack` — see §7. Perf transfer check remains queued behind correctness gate. |
| 5 | Selective / default-on PCG PR plan | P2 | planned (needs #4) |

## 4. Sub-track — Qwen3-VL PCG capture-stream investigation

Active branch: `debug/v2-imgA-pcg-capture-stream-fix` (merged to `main`
2026-07-29 as PR #7). Full per-phase writeups under
[`root_cause/README.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/README.md);
sglang source edits kept as revertable `.patch` files under
`root_cause/patches/`; raw per-run logs stay under `results/<R-id>/raw/` and
are **not committed**.

### 4.1 Setup + provenance (frozen 2026-06-28 / 2026-07-28)

Server rebuilt 2026-06-28; environment re-set up from scratch. Source of truth
for sglang edits is the user fork
`git@github.com:bowenwan6/sglang.git`, cloned to `/data/sglang-fork`, branch
`fix/pcg-vlm-deepstack-warmup` started from upstream `da802ddca` so patched
python files stay binary-compatible with the installed `sgl_kernel`. Runs
source the fork via `PYTHONPATH=/data/sglang-fork/python`.

| Item | Value |
|---|---|
| Stock SGLang SHA | `da802ddcafe55e25b3e1db86b1e0444afc3e05bc` |
| Final fork SHA | `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` |
| Model snapshot | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| System python | `python 3.12.3` · torch 2.11.0+cu130 · flashinfer 0.6.12 · sgl_kernel 0.4.4 |
| Profiling env | `/opt/miniconda3/envs/profiling` — torch 2.11.0+cu130, vLLM 0.21.0 |
| Text dataset | `datasets/qwen3vl8b/caseA_short.jsonl` (SHA-256 `fab4917772…`) |
| Runtime libcuda | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` (host driver 595.71.05, fixes `cuda-compat-13-0` loader precedence) |

Full frozen provenance + verification commands live in
[`R6.0_provenance.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.0_provenance.md).
Numbers from any earlier HEAD are historical reference only.

### 4.2 R1 → R5 outcome (root-cause + fix implementation)

- **R1 (Dynamo tracing).** Recompile trigger identified: `Qwen3LLMModel.forward`
  specialises on the `input_deepstack_embeds is None` guard at
  `qwen3_vl.py:1129`; multimodal control-flow recompile, not a shape
  recompile. PCG warmup only ever feeds the `None` branch, so the first real
  image request forces a new fx graph whose piecewise submodules have no
  capture stream.
- **R2 (source-level trace).** Confirmed the asserting `CUDAPiecewiseBackend`
  is a distinct Python object from the warmup-frame instance;
  `set_pcg_capture_stream()` runs only inside `capture_session()` and never
  re-runs at inference. The assertion at `cuda_piecewise_backend.py:172` is
  structurally unreachable from a recompiled instance.
- **R3 → R4 (fix shapes).**
  - **(X) defensive CUDA eager fallback** at
    `cuda_piecewise_backend.py:163-169` — validated safe (n=32 / n=400) but
    **rejected for upstream**: converts a crash into a silent ~38 ms TTFT
    regression vs the PCG-off baseline. Kept as fork history + patch only.
  - **(Y) broaden warmup capture** — clean implementation on fork HEAD
    `986c89e69`: thread-local `force_warmup_deepstack_embeds` gate synthesises
    zero deepstack embeds during PCG warmup so Dynamo traces both branches;
    mirrored compile-pass and capture-pass hooks; model-attached static
    deepstack buffer for capture / replay address stability.
- **R5 (clean-Y verification).** Assertion / eager fallback / inference-time
  recompile all eliminated on fork under `--enforce-piecewise-cuda-graph`
  (n=32 R5.A, n=400 R5.B). The original R5 image-only TTFT gate (p50 clearly
  below the 64.8 ms PCG-off baseline) **FAILED as stated** —
  fork-PCG image+text ≈ 102–104 ms — because Qwen3-VL image prefill is
  vision-tower-dominated (~40 ms eager either way), leaving too small a
  PCG-covered LM fraction for graph-launch savings to overcome capture
  overhead. This is a workload property; the fix's value was re-framed and
  formally validated under **R6** (§4.3), not by silently rewriting R5's
  gate. R5.C correctness audit reported OUTPUTS_DIFFER; matched controls
  were deferred to R6.1.

### 4.3 R6 — fix-value validation (audited 2026-07-29)

**Audited conclusion:** ✅ `CORRECTNESS_AND_SAFETY_FIX_PASS` +
⚠️ `PERFORMANCE_VALUE_PROMISING_BUT_NOT_FINAL`. Overall mixed-workload
**performance dominance is not established**. An earlier machine-generated
"R6 PASS" (via R6.4 `STRICTLY_DOMINANT` + R6.5 3/3 agreement) is retracted:
R6.4's bootstrap 95 % CI on `p*` is unidentifiable, and the R6.5 machine
PASS combined stale ratio_0p5 / ratio_0p8 artifacts predating the
attempt_gpu2 launch context. Full audited evidence chain in
[`R6_FINAL_CONCLUSION.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6_FINAL_CONCLUSION.md);
per-phase status in
[`R6_fix_value_validation/README.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/README.md).

**Three independent claims the R6 protocol tests:**

1. **Correctness / safety.** Fork clean-Y preserves correctness on
   mixed-modality workloads, or residual output divergence is demonstrably
   attributable to normal PCG-vs-eager bf16 noise via matched control.
2. **Retained PCG benefit.** On text-only Case A against a Qwen3-VL server,
   fork-PCG delivers the same mean TTFT as stock-PCG (fix does not regress
   the text path); both remain clearly below stock-default. The fix
   *preserves* the text-only PCG speedup on a server that must also accept
   image traffic without crashing — it does not *create* it.
3. **Mixed-modality operational safety.** Interleaved text → image → text
   traffic on the same fork-PCG server yields 0 request failures / 0
   capture-stream assertions / 0 eager fallbacks / 0 inference-time
   recompiles of `qwen3_vl.forward`. Stock has no equivalent — stock
   crashes on the first image under `--enforce-piecewise-cuda-graph`.

**Per-phase outcome** (one-line summary; full row detail in the R6 README):

| Phase | Audited outcome |
|---|---|
| **R6.1** | ✅ PASS — `SAFETY_SUPERIORITY_PASS` (Amendment B, attempt 04) + `CORRECTNESS_PASS` (Amendment A, attempt 03). Stock reproduces the exact historical `AssertionError: PCG capture stream is not set` at second-same-shape post-recompile call; fork completes the identical bench (30 warmup + 32 measured) with 0 assertions / 0 fallbacks / 0 post-ready recompiles / 32 of 32 completed. |
| **R6.2** | ✅ `PASS_WITH_CAVEAT — TEXT_NON_REGRESSION_SUPPORTED` under Amendment C. Original machine FAIL on 3.050 % drift preserved verbatim. `fork_pcg / stock_pcg = 0.9617` supports non-regression; treat as within-noise-equivalent, not a proven 3.8 % speedup. All CVs ≤ 5.91 %, 5 of 5 × 400 of 400 every variant, 0 safety anomalies. |
| **R6.3** | R6.3c mixed safety ✅ PASS (0/0/0/0). R6.3a/b performance results promising but exploratory; cleanest per-cell signal is `cell_t512_r360p_c1` at ~8.6 % mean improvement with CV ~4 % on each side. Higher-margin cells (e.g. `cell_t512_r360p_c4` at ratio 0.7806) have per-variant CVs of 27–33 % and are exploratory only — the prior "22 % headline" framing is retracted. Loss regime cleanly isolated to `t2048_*` (long text). |
| **R6.4** | ⚠️ AMBIGUOUS. Point estimate `p* = C / (G + C) = −3.91` (outside `[0, 1]`) from small-n shared-GPU inputs. Bootstrap 95 % CI [−12.39, +15.44] — statistically unidentifiable. The earlier `STRICTLY_DOMINANT` framing is **retracted**. |
| **R6.5** | ❌ `INVALID_MIXED_PROVENANCE / AMBIGUOUS`. attempt_gpu2 machine PASS preserved verbatim but rejected on audit: `prelaunch_utc = 12:27:18Z`, yet ratio_0p5 (`12:17:32Z / 12:19:27Z`) and ratio_0p8 (`12:21:09Z / 12:23:54Z`) predate the launch by 6–10 min and are stale artifacts from attempt_gpu4. `R6_5_verdict.py` did not enforce launch-ID / timestamp checks. attempt_gpu4 independently AMBIGUOUS with only ratio_0p5 clean (fork/stock = 1.005, tied). **R6.5 does not validate mixed-workload dominance.** |

**Protocol amendments** (chronological; preserved verbatim on-branch):

- **Amendment A** (2026-07-28, authoritative for R6.1 attempts 03+) —
  three-tier verdict shape, cache-matched cold-cache repeats, phase-scoped
  recompile markers, direct stock-PCG image negative control.
  [`R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md).
- **Amendment B** (2026-07-28, authoritative for attempts 04+) — replaces
  Amendment A's 3-prompt negative control with the exact historical R1 / E2a
  repeated-shape recipe (720p × 32 requests, `--random-input-len 128
  --random-range-ratio 1.0 --num-prompts 32 --warmup-requests 30`,
  `--max-concurrency 1`), identical for stock-PCG and fork-PCG. All other
  Amendment A rules remain in force.
  [`R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md).
- **Amendment C** (2026-07-29, applies to R6.2 and any shared-GPU drift
  bracket in R6.3–R6.5) — reclassifies the drift bracket as a shared-GPU
  nuisance-control (`≤ 3 %` clean, `3–5 %` `PASS_WITH_CAVEAT`, `> 5 %`
  rerun / AMBIGUOUS). Fork-vs-stock non-regression (`≤ 1.05`), per-variant CV
  (`≤ 6 %`), and every safety hard-FAIL condition remain unchanged.
  [`R6.2_text_only_caseA/protocol_amendment_C_shared_gpu_drift_gate.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.2_text_only_caseA/protocol_amendment_C_shared_gpu_drift_gate.md).

**Pre-declared R6 verdict framework** (kept for provenance; audit narrowed it):

- **PASS** ← R6.1 = PASS AND R6.2 within thresholds (Amendment C: `PASS` or
  `PASS_WITH_CAVEAT`) AND R6.3c 0 failures / 0 assertions / 0 recompiles /
  0 fallbacks AND (R6.3b ≥ 1 winning cell OR R6.4 `p*` in operator-realistic
  range ≤ 0.95).
- **FAIL** ← R6.1 = FAIL, or R6.2 fork-PCG regresses stock-PCG beyond
  threshold, or R6.3c surfaces any failure / assertion / recompile /
  fallback.
- **R7_REQUIRED** ← R6.1 = AMBIGUOUS, or R6.3b + R6.4 shows no
  operator-realistic winning workload and `p*` > 0.95.

Under this framework a machine "R6 PASS" was reached; the 2026-07-29 audit
found the "≥ 1 winning cell" trigger fired with statistically thin per-cell
CVs and that R6.4 / R6.5 dominance claims were unsupported (details above).
A well-attributed performance headline requires a clean isolated
mixed-workload rerun with launch-context enforcement in the verdict script
and a proper model-serving-ready readiness check — deferred; **not** part of
the current merge.

**R6 out of scope:**

- Changing `is_multimodal_piecewise_cuda_graph_supported` defaults — that is
  Issue #5's scope.
- Filing an upstream SGLang PR from the local fork branch unchanged — the
  local branch is likely superseded by upstream PR #30868; any residual
  filing would need to be re-verified against current upstream first.
- Retroactively rewriting the R5.C `audit_report.md` — R5.C stands as
  written; R6.1 supersedes it as the correctness authority. Local
  uncommitted edit to `R5C_correctness_audit/audit_report.md` is preserved
  under user control.

## 5. Immediate next steps

- **Issue #3 — new sub-track (highest priority in #3).** A new
  `Qwen3.5-4B BCG DeepStack` investigation opens on branch
  `debug/qwen35-4b-bcg-deepstack` — see §7. This runs **before** the
  perf-transfer check because a suspected correctness gap on
  `Qwen3_5ForConditionalGeneration` under multimodal prefill BCG must be
  proven or disproven first; a perf comparison against an uncertified
  correctness base would be meaningless.
- **Issue #3 (perf transfer, deferred).** Apply the v2 methodology and
  test whether the Case A PCG finding transfers — gated on §7's verdict.
- **Issue #5** — Selective / default-on PCG PR plan: needs #4's image
  evidence, which is now audited and available in R6.
- **Upstream verification** — a separate CPU-only + fresh-run task to
  reproduce the R6.1 Amendment B repeated-shape crash on current upstream
  SGLang (post PR #30868). If the assertion no longer reproduces, close the
  Issue #4 sub-track as "already fixed upstream"; if some residual case
  reproduces, file a smaller PR scoped to that residual case. Do **not**
  upstream the old fork branch unchanged.
- **IMG-A non-PCG resume** — `S0_ipc_repeat → V0_vllm → S0_noipc` remains
  queued under
  [`fixed_generator_plan.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/fixed_generator_plan.md);
  orthogonal to the PCG sub-track.

## 6. Commit cadence and artifact rules

**Commit types** (per `CLAUDE.md` + Conventional Commits; author = Bowen Wang;
no `Co-Authored-By` trailers):

- `docs(v2): …` — plan and status revisions.
- `feat(v2): …` — new runners, generators, analysis tooling.
- `test(v2): …` — recorded experiment results, **including recorded
  failures** (a failed R6.1 or R6.2 result is a `test` commit, not a `fix`).
- `perf(v2): …` — perf implementation changes only; never for merely
  reporting perf numbers.
- `fix(v2): …` — profiler repo bugfixes.

Every runner spec, every recorded experiment, and any high-level conclusion
land as independent focused commits, pushed immediately. SGLang fork edits
commit + push in `/data/sglang-fork` only; never mix fork edits into
profiler commits.

**Artifact rules:**

- v2 results go only under `experiments/qwen3vl8b/v2/…` and
  `logs/qwen3vl8b/v2/…`. Never overwrite v1 Phase 0–5 artifacts.
- v1 raw JSON, traces, logs, scripts, and SGLang source are not modified.
- Clean headline runs forbid KAPI / profiler: never set
  `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST`; no profiler.
  Servers run serialised (never co-resident).
- Every run records: GPU id, exact flags, framework versions, model
  snapshot, dataset sha256, warmup / reps / num-prompts, failure / error
  rate, and the KAPI / profiler-disabled confirmation.
- Raw per-rep dumps and server logs are generated but not committed unless
  explicitly approved (committed deliverables = summaries + aggregate
  `case*_results.json`).

## 7. Sub-track — Qwen3.5-4B BCG DeepStack investigation (active)

> **Investigation, not confirmed upstream bug.** The runtime hypothesis
> stated below is a source-reading of current upstream SGLang, not a
> reproduced runtime failure. Nothing in §7 asserts that outputs are wrong
> on real hardware; that is what §7's validation plan is designed to prove
> or disprove. No GPU work is authorised until the plan lands and a GPU is
> explicitly approved.

Active branch: `debug/qwen35-4b-bcg-deepstack` (based on `main` = `a803285`).
Investigation anchor:
[`experiments/qwen35_4b/README.md`](experiments/qwen35_4b/README.md)
with `source_audit.md`, `provenance.md`, `hypothesis.md`,
`validation_plan.md`, and reserved `results/` / `scripts/` subdirs.

### 7.1 Why Qwen3.5-4B (target choice)

`Qwen/Qwen3.5-4B` (HF, `sha=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
`config.architectures=["Qwen3_5ForConditionalGeneration"]`,
`model_type=qwen3_5`, ungated, `pipeline_tag=image-text-to-text`) is the
lowest-cost reproduction target that (a) is a **natively multimodal**
Qwen3.5 model, (b) is registered on current SGLang as multimodal +
prefill-BCG-supported, and (c) inherits the Qwen3-VL DeepStack wrapper
from which the hypothesis is drawn. It replaces the earlier Qwen3-VL-8B
plan for BCG DeepStack validation because 4B is much cheaper to
serve and does not change any of the source-level control paths under test.

### 7.2 Upstream provenance frozen at investigation start

Values verified against `sgl-project/sglang` on 2026-07-31 via GitHub API.

| Item | Value |
|---|---|
| Upstream SGLang `main` HEAD | `5f9b0db18c787cf56ed9bbaf255f083f26c6ebc2` (subject: `Fix async loading of RunAI-streamed tensors (#32896)`) |
| PR #30872 (`Enable multimodal prefill BCG for VL and audio models`) | **MERGED** 2026-07-28T22:47:40Z — merge commit `c9947b087bf9d3d16b5198234ba4c39b68bb79e9`. Enables prefill BCG for multimodal VL / audio wrappers; adds `input_embeds` static slot; decode-first capture ordering. Contains **no `input_deepstack_embeds` slot or copy** (verified in PR diff). |
| PR #30868 (`fix: fix vlm cuda graph shape stability`) | **MERGED** 2026-07-19T14:35:51Z. Introduces `run_dummy_multimodal_deepstack_forward` on `PrefillCudaGraphRunner` to warm the tensor-valued DeepStack branch at `capture_num_tokens[-1]` only, plus a defensive eager fallback if a replacement backend is instantiated without a capture stream. Related but **not equivalent** to the §7 hypothesis — §7 is about capture / replay data-flow for `input_deepstack_embeds`, not Dynamo shape stability. |
| Local mirror `/sgl-workspace/sglang` | `da802dd` — a **stale** older HEAD; not the source of truth for this audit. Upstream files read via raw.githubusercontent for the audit under `experiments/qwen35_4b/`. |
| Historical local fork `/data/sglang-fork` | `986c89e69` — untouched by §7; read-only reference only. |

### 7.3 Established facts (source-level, on upstream `main` @ `5f9b0db1`)

Verified by inspection of files under
`python/sglang/srt/{models,model_executor,managers,configs}`. All line
numbers refer to raw upstream files captured 2026-07-31.

1. **Qwen3.5 architectures live in the BCG allowlist.** In
   `python/sglang/srt/configs/model_config.py:1836`,
   `multimodal_piecewise_cuda_graph_supported_model_archs` includes
   `"Qwen3_5ForConditionalGeneration"` and
   `"Qwen3_5MoeForConditionalGeneration"` (lines `1846-1847`).
   `is_multimodal_piecewise_cuda_graph_supported` (line `1908`) returns
   True for these architectures when `enable_multimodal` is set.
2. **Qwen3.5 inherits the Qwen3-VL multimodal wrapper.**
   `python/sglang/srt/models/qwen3_5.py:1771`:
   `class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`
   with `language_model_cls=Qwen3_5ForCausalLM`. The MoE variant
   (line `1928`) does the same. The `Qwen3_5ForCausalLM.forward`
   signature (line `1408`) accepts
   `input_deepstack_embeds: Optional[torch.Tensor] = None`, and the loop
   at line `1448-1458` adds
   `input_deepstack_embeds[:, sep : sep + hidden_size]` to `hidden_states`
   for `layer_idx < 3` when the tensor is non-`None` and non-empty.
3. **`general_mm_embed_routine` synthesises DeepStack live per request.**
   `python/sglang/srt/managers/mm_utils.py:1108-1140` builds
   `input_deepstack_embeds` in `embed_mm_inputs` as a fresh
   `torch.zeros(deepstack_embedding_shape, device=input_embeds.device, dtype=input_embeds.dtype)`
   allocated per call, then scatters per-modality DeepStack tiles into it
   and returns it in `other_info`. The pointer is fresh per request.
4. **BCG replay copies `input_embeds` into a stable slot; there is no
   equivalent for `input_deepstack_embeds`.**
   - `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py:867-877`
     registers an `input_embeds` slot when `is_multimodal and
     register_input_embeds=True`; **no slot named `input_deepstack_embeds`
     is ever registered**.
   - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`,
     `_execute_body_capture` closure `replay_layer_forward` (lines
     `1498-1519`) copies **only** the live `input_embeds` argument into the
     registry's `input_embeds` slot before calling
     `self.backend.replay(shape_key, static_forward_batch, **kwargs)`.
     The outer `**kwargs` is the tail-forward kwargs from `_execute_body_capture`,
     not `layer_kwargs`, so `input_deepstack_embeds` (passed by
     `general_mm_embed_routine` as a `layer_kwargs` entry) is **not**
     forwarded into `.replay()` and **not** copied into any stable slot.
5. **The one existing DeepStack accommodation is a Dynamo warmup only,
   not a capture / replay slot.**
   `run_dummy_multimodal_deepstack_forward` at
   `prefill_cuda_graph_runner.py:662-725` calls
   `language_model.forward(..., input_deepstack_embeds=deepstack_embeds)`
   during warmup with a locally-allocated `torch.zeros(...)` tensor. This
   traces the tensor-valued DeepStack branch through Dynamo so the
   `input_deepstack_embeds is None` guard does not force an
   inference-time recompile, but the tensor has no stable-address slot
   in the buffer registry and its allocation is dropped when the warmup
   function returns.
6. **Existing tests do not cover the DeepStack replay path.** Registered
   tests referencing BCG / piecewise + deepstack are limited to
   `test/registered/unit/model_executor/test_prefill_cuda_graph_runner_helpers.py`
   (mrope helper unit, added by PR #30868),
   `test/registered/unit/model_executor/test_prefill_cuda_graph_runner.py`
   (wrapper resolution + `input_embeds` slot helper, from PR #30872), and
   `test/registered/unit/configs/test_multimodal_piecewise_cuda_graph.py`
   (the allowlist itself). No test verifies that the captured graph's
   DeepStack contribution matches the eager result on a real image
   request.

### 7.4 Runtime hypothesis (unverified)

Given the source-level observations in §7.3, the working hypothesis is:

> On current SGLang `main` (`5f9b0db1`), when
> `Qwen/Qwen3.5-4B` serves an image request under
> `--enforce-piecewise-cuda-graph` (or the default `breakable` prefill
> backend), the BCG-captured layer-body graph replays without the live
> `input_deepstack_embeds` tensor pointer being copied into any stable
> slot. Consequences may include (a) silent output divergence versus the
> eager path because layers 0–2 miss their DeepStack contribution, or
> (b) an inference-time Dynamo recompile on the first tensor-valued
> DeepStack shape not covered by the single-shape warmup, or (c) a hard
> assertion / illegal-memory failure if the warmup tensor was freed.

This is the hypothesis the §7 validation plan must be able to prove or
disprove. It is **not** a confirmed bug and must not be quoted as one
until runtime evidence supports it.

### 7.5 Possible outcomes the validation must distinguish

1. Image request uses BCG and DeepStack is preserved correctly.
2. Request falls back to eager, so correctness is preserved but
   multimodal prefill BCG is not actually exercised.
3. Request uses BCG but DeepStack is omitted / stale, producing a
   correctness defect.
4. The suspected source-level gap is handled elsewhere in the code base
   and no defect exists.

### 7.6 Immediate next steps (§7 track)

Sequenced, no GPU work in any of these:

1. Done — organised investigation area under
   `experiments/qwen35_4b/` (`docs(qwen35): organize BCG DeepStack validation records`).
2. Done — validation plan with predeclared verdicts, evidence layers,
   configurations, fixtures, and confounder controls landed at
   [`experiments/qwen35_4b/validation_plan.md`](experiments/qwen35_4b/validation_plan.md)
   (`docs(qwen35): plan BCG DeepStack validation`).
3. Open the tracking issue in this profiler repo (not upstream)
   linking PR #30872, the source-level observations, and the
   validation plan — commit `docs(qwen35): link BCG DeepStack investigation issue`.
4. Land CPU-only scaffolding (deterministic fixture, provenance
   checks, GPU-refusing runner skeleton, PGID-scoped safeguards) —
   commit `feat(qwen35): prepare BCG DeepStack validation`.
5. Stop. Wait for user to authorise a specific GPU ID before any
   GPU-touching work.

### 7.7 §7 out of scope

- Any GPU work (server launch, weight download, `nvidia-smi` memory
  allocation) until the user authorises a specific GPU ID.
- Filing an upstream SGLang issue or PR — deferred until runtime
  reproduction (or definitive disproof) is in hand.
- Rewriting or repurposing the historical Qwen3-VL PCG evidence under
  §4 or the `debug/v2-imgA-pcg-capture-stream-fix` branch. §7 links
  historically where useful but treats §4 as read-only.
- Editing anything under `/data/sglang-fork`. That fork is preserved
  read-only as historical evidence at `986c89e69`.
