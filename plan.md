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
| System python | `python 3.12.3` · torch 2.11.0+cu130 · flashinfer 0.6.12 · sgl_kernel 0.4.5 |
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
Tracking issue: [profiler-repo issue #9](https://github.com/bowenwan6/sglang-vllm-profiler/issues/9)
(sub-track of #3; no upstream SGLang issue filed at investigation
start — deferred until runtime evidence is in hand).
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

### 7.2 Upstream provenance (rebaselined 2026-07-31)

Values verified against `sgl-project/sglang` on 2026-07-31 via GitHub API.

| Item | Value |
|---|---|
| **Executed local SGLang checkout (HARD PIN)** | isolated `git clone` under `<scratchpad>/sglang_checkout/sglang/`, HEAD pinned to `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`. The runner sources it via `PYTHONPATH` and verifies `sglang.__file__` resolves inside it. |
| Upstream SGLang `main` HEAD at rebaseline | `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` (2026-08-01 refresh, subject: `[perf] Assemble flat prompt top logprobs scheduler-side as numpy arrays (#32223)`). `/data/sglang-fork` main was also fast-forwarded to this SHA on the same date; the historical `fix/pcg-vlm-deepstack-warmup` branch stays at `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`. Later remote-main movement is informational only, not a hard failure — see `provenance.md` §1. |
| PR #30872 (`Enable multimodal prefill BCG for VL and audio models`) | **MERGED** 2026-07-28T22:47:40Z — merge commit `c9947b087bf9d3d16b5198234ba4c39b68bb79e9`. Adds Qwen3.5 to `multimodal_breakable_cuda_graph_supported_model_archs` (the **BCG** allowlist), registers the `input_embeds` static slot, and adds the `replay_layer_forward` per-request copy of `input_embeds`. **Contains no `input_deepstack_embeds` slot or copy on the BCG code path.** |
| PR #30868 (`fix: fix vlm cuda graph shape stability`) | **MERGED** 2026-07-19T14:35:51Z. Introduces `run_dummy_multimodal_deepstack_forward` and a defensive eager fallback, **both scoped to `tc_piecewise_cuda_graph_backend`**. This is a Dynamo shape-stability warmup for **TC piecewise / PCG**, not a BCG capture / replay slot. |
| Local mirror `/sgl-workspace/sglang` | `da802dd` — stale older HEAD; the installed sglang at that path is **not** the runner's source of truth. Runners override via `PYTHONPATH` to the frozen checkout and assert `sglang.__file__` resolves inside it. |
| Historical local fork `/data/sglang-fork` | `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` — untouched by §7; read-only reference only. The runner sanity-checks that this HEAD is unchanged before and after every attempt. |

### 7.3 Established facts (source-level, on upstream `main` @ `58974ca1`)

Verified by inspection of files under
`python/sglang/srt/{models,model_executor,managers,configs}`. All line
numbers refer to raw upstream files captured 2026-07-31.

1. **BCG and PCG allowlists are DISTINCT; Qwen3.5 is on BCG only.**
   In `python/sglang/srt/configs/model_config.py`:
   - Lines `1836-1841` — `multimodal_piecewise_cuda_graph_supported_model_archs`
     (the **PCG / `tc_piecewise` / torch.compile-based** allowlist)
     contains **only** `Cohere2VisionForConditionalGeneration`,
     `KimiK25ForConditionalGeneration`, `MiniMaxM3SparseForCausalLM`,
     `MiniMaxM3SparseForConditionalGeneration` — **not** Qwen3.5.
   - Lines `1845-1848` — `multimodal_breakable_cuda_graph_supported_model_archs`
     (the **BCG** allowlist) contains `Qwen3_5ForConditionalGeneration`
     and `Qwen3_5MoeForConditionalGeneration`. The in-source comment
     is: "embed-carrying batches are rejected at replay
     (can_run_graph) and run eager." (See §7.3(6) below for what
     "embed-carrying" actually gates.)
   - `is_multimodal_piecewise_cuda_graph_supported` (line `1908`)
     and `is_multimodal_breakable_cuda_graph_supported` (line `1916`)
     are two independent accessors. `ModelConfig` computes both.
     **`--enforce-piecewise-cuda-graph` is not a valid BCG control
     for Qwen3.5**; the validation plan does not use it.
2. **Qwen3.5 inherits the Qwen3-VL multimodal wrapper.**
   `python/sglang/srt/models/qwen3_5.py:1771`:
   `class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration)`
   with `language_model_cls=Qwen3_5ForCausalLM`. The MoE variant
   (line `1928`) does the same. `Qwen3_5ForCausalLM.forward`
   (line `1408-1478`) accepts
   `input_deepstack_embeds: Optional[torch.Tensor] = None`, and the
   loop at line `1448-1458` `add_`s
   `input_deepstack_embeds[:, sep : sep + hidden_size]` to
   `hidden_states` for `layer_idx < 3` when the tensor is non-`None`
   and non-empty. The DeepStack contribution is *injected* at
   layers 0–2 but propagates through later layers via the residual
   stream — the observable effect is not restricted to layers 0–2.
3. **`general_mm_embed_routine` synthesises DeepStack per request.**
   `python/sglang/srt/managers/mm_utils.py:1108-1140` allocates
   `input_deepstack_embeds` as a per-call `torch.zeros(...)`
   (`(num_tokens, hidden_size * num_deepstack_embeddings)`), scatters
   per-modality tiles into it, and stores it in `other_info`. Lines
   `1247-1373` route the routine and unpack
   `other_info["input_deepstack_embeds"]` into the LM's `kwargs`.
   Lines `1361-1363` copy `input_embeds` into a stable slot when
   one exists; **no analogous copy exists for
   `input_deepstack_embeds`**. The Python tensor is fresh per call;
   its `.data_ptr()` is not stable by contract but the CUDA caching
   allocator may reuse the same address, so pointer equality alone
   is not diagnostic.
4. **BCG replay copies `input_embeds` into a stable slot; there is
   no equivalent for `input_deepstack_embeds`.**
   - `python/sglang/srt/model_executor/cuda_graph_buffer_registry.py:867-877`
     registers an `input_embeds` slot when `is_multimodal and
     register_input_embeds=True`; **no slot named
     `input_deepstack_embeds` is ever registered**.
   - `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py`,
     `_execute_body_capture` closure `replay_layer_forward` (lines
     `1498-1519`) copies **only** the live `input_embeds` argument
     into the registry's `input_embeds` slot before calling
     `self.backend.replay(shape_key, static_forward_batch, **kwargs)`.
     `**kwargs` here is the **outer** tail-forward kwargs, not
     `layer_kwargs`; `input_deepstack_embeds` (routed as a
     `layer_kwargs` entry) is not forwarded into `.replay()` and
     not copied anywhere. The BCG backend's `.replay(...)`
     (`runner_backend/breakable_cuda_graph_backend.py:241-248`)
     replays the captured graph and ignores `**kwargs`.
5. **The one existing DeepStack accommodation is TC piecewise only,
   not BCG.** `run_dummy_multimodal_deepstack_forward` at
   `prefill_cuda_graph_runner.py:662-725` is a Dynamo shape-stability
   warmup that allocates a local `torch.zeros(...)` and traces the
   tensor-valued DeepStack branch. Its **only** caller is
   `tc_piecewise_cuda_graph_backend._run_compile_pass`
   (`runner_backend/tc_piecewise_cuda_graph_backend.py:214-216`),
   after `torch.compile` is installed. **The BCG capture path never
   invokes it.** BCG's `_run_forward` (`prefill_cuda_graph_runner.py:606-649`)
   drives `layer_model.forward(input_ids, positions, forward_batch,
   forward_batch.input_embeds)` — four positional args, no
   `input_deepstack_embeds` kwarg. The DeepStack `add_` branch is
   therefore **cold at BCG capture time**, and the captured graph
   contains no DeepStack kernels at all. This is a distinct concern
   from PR #30868's PCG Dynamo warmup and PR #30872's BCG replay
   bridge; the two PRs together do not close it.
6. **`can_run_graph`'s `input_embeds is not None` gate targets
   API-`input_embeds`, not multimodal image requests.**
   `prefill_cuda_graph_runner.py:1015-1016` returns `False` when
   `forward_batch.input_embeds is not None`. In `managers/schedule_batch.py:2233-2401`,
   `batch.input_embeds` is populated only when the request carries
   an API-level `req.input_embeds`; normal multimodal image requests
   leave it `None` and set `batch.multimodal_inputs` instead. So
   the "embed-carrying rejection" comment (§7.3(1)) applies to
   API-provided embeddings, not to image requests. Image requests
   are not filtered here by construction.
7. **Existing tests do not cover the DeepStack BCG replay path.**
   Registered tests referencing BCG / piecewise + deepstack are
   limited to the wrapper-resolution / helper units from PRs #30872
   and #30868, plus the allowlist assertion; no test verifies that
   the captured BCG graph's DeepStack contribution matches the eager
   result on a real image request.

### 7.4 Runtime hypothesis (unverified)

Given §7.3, the working hypothesis is:

> On current upstream SGLang `main` at
> `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`, when
> `Qwen/Qwen3.5-4B` serves an image request under the default
> breakable prefill backend, the BCG-captured layer-body graph has
> no DeepStack `add_` kernels (the branch was cold at capture) and
> no stable slot for `input_deepstack_embeds`. Consequences may be
> (a) silent output divergence versus the eager path with a
> "DeepStack-zeroed" signature, (b) illegal-memory / assertion at
> replay, (c) a runtime filter this audit did not find routes image
> requests to eager (feature gap; correctness preserved but BCG not
> exercised for images), or (d) some code path pins DeepStack that
> the audit missed and correctness holds. The validation plan must
> distinguish these with direct evidence, not by elimination.

This is the hypothesis §7's validation plan proves or disproves.
It is **not** a confirmed bug and must not be quoted as one until
runtime evidence supports it.

### 7.5 Machine verdict shape (predeclared)

The validation plan must emit exactly one of the following. **An
eager fallback is never "bug closed" or full PASS.**

- **`PASS_BCG_CORRECT`** — Image request demonstrably replays BCG
  and DeepStack-active results match the eager reference within the
  eager-vs-eager noise envelope.
- **`FEATURE_GAP_EAGER_FALLBACK`** — Correctness is preserved
  because the image request runs eager, but multimodal BCG
  support/performance is not demonstrated. Documented as a feature
  gap.
- **`FAIL_BCG_DEEPSTACK`** — Image request demonstrably replays BCG
  and live DeepStack is missing, stale, or produces a matched
  correctness divergence (zero-DeepStack signature), or the BCG
  replay raises an assertion / illegal memory access at inference
  time.
- **`AMBIGUOUS`** — Divergence exists but cannot be cleanly
  attributed.
- **`INFRA_FAILURE`** — Environment / GPU / preflight failure;
  neutral outcome, does not count for or against any hypothesis.

Predeclared **diagnostic ablation**: on top of `eager_normal` and
`bcg_normal`, the runner also collects `eager_zero_deepstack`
(eager run with a branch-local instrumentation hook that zeros
`input_deepstack_embeds` immediately before the LM forward). If
`bcg_normal` tracks `eager_zero_deepstack` rather than
`eager_normal`, that is strong attribution evidence for
`FAIL_BCG_DEEPSTACK`. This is a diagnostic ablation, not production
behavior.

### 7.6 Immediate next steps (§7 track)

Executed in one continuous pass; each step commits+pushes on
success.

1. **Step 1 (docs)** — corrected the BCG-vs-PCG facts, the
   caller/scope of `run_dummy_multimodal_deepstack_forward`, the
   pointer wording, the verdict labels, the simplified
   correctness/path protocol, the diagnostic ablation, and the
   top-level README. Commit `docs(qwen35): correct BCG DeepStack
   validation protocol`.
2. **Step 2 (live runner)** — implemented the profiler-owned
   instrumentation patch, the live runner, the live client, and
   the real verdict inference. All CPU-only tests pass. Commit
   `feat(qwen35): add live BCG DeepStack validation runner`.
3. **Step 3 (authorised-GPU acquisition)** — read-only query the
   authorised GPU (allowlist `{0, 1, 7}` per `validation_plan.md`
   Amendments 1 and 2, 2026-08-01); qualifies only when zero compute
   processes, memory ≤ 500 MiB, utilisation ≤ 5 %; the
   10-continuous-minute idle requirement applies unless the operator
   explicitly waives it AND the target GPU is currently qualifying;
   never signal a foreign PID.
4. **Step 4 (INFRA_CHECK) — PASS (2026-08-01, GPU 7).** Attempt
   `experiments/qwen35_4b/results/infracheck_gpu7_20260801T012122Z/`
   brought Qwen3.5-4B up on the authorised alternate GPU 7 under the
   frozen SGLang checkout `58974ca16…`: every hard pin matched
   (`Qwen3_5ForConditionalGeneration` @
   `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, imported sglang
   `INSIDE_FROZEN`, `/data/sglang-fork` still `986c89e69…`,
   `sglang-kernel==0.4.5` clearing the prior 0.4.4 blocker),
   `Multimodal data loading enabled with 16 worker threads` +
   `Using fa3 as multimodal attention backend`, prefill backend
   `breakable` captured all 58 shape buckets, warmup exercised BCG
   (`cuda graph: True`), the server reported `The server is fired up
   and ready to roll!` at 129 s, teardown signalled only own PGID,
   GPU 7 memory returned from 111,760 MiB to 4 MiB, and the 11
   foreign compute processes on other GPUs were unchanged pre-vs-post.
   One caveat carried into Step 5 — SGLang uses
   `mp.set_start_method('spawn', force=True)` for scheduler /
   model-worker subprocesses, so the branch instrumentation installed
   in the launcher parent does **not** propagate to workers; the
   authoritative per-batch BCG-vs-eager signal for Step 5 is SGLang's
   own `cuda graph: True/False` server-log line, and the
   `eager_zero_deepstack` diagnostic ablation behaves identically to
   `eager_normal` under this constraint (weakens attribution for the
   zero-DeepStack signature path only). Commit
   `test(qwen35): verify Qwen3.5 BCG infrastructure`.
5. **Step 5 (validation) — AMBIGUOUS (2026-08-01, GPU 7).** Attempt
   `experiments/qwen35_4b/results/attempt_gpu7_20260801T013522Z/`
   ran the three predeclared configs. Instrumentation propagation
   into SGLang spawn workers was unblocked by
   `experiments/qwen35_4b/scripts/bootstrap/sitecustomize.py`, so
   `bcg_execute_body_enter`, `model_runner_forward_enter`, and
   `general_mm_embed_routine_enter/exit` events fired inside the
   scheduler / worker subprocesses as intended. `bcg_normal` served
   both scored image prefills via BCG replay
   (`bcg_execute_body_enter` with `contains_mm_inputs=true`,
   `shape_key size=16`, `cuda graph: True`, no
   `bcg_execute_body_error`), and greedy text was bit-identical to
   `eager_normal` for every scored request. That rules out both
   `FEATURE_GAP_EAGER_FALLBACK` (BCG was not bypassed) and
   `FAIL_BCG_DEEPSTACK` (no divergence, no crash). However,
   `PASS_BCG_CORRECT` requires positive evidence that
   `input_deepstack_embeds.nonzero_frac > 0` at LM forward time, and
   this is not available: the branch instrumentation's
   `language_model.__call__` interceptor writes to the instance
   `__dict__` but `nn.Module` resolves `__call__` on the class, so
   `lm_forward_input_deepstack` never fires and `QWEN35_ZERO_DEEPSTACK=1`
   is a no-op — the `eager_zero_deepstack` ablation degenerates to
   `eager_normal`. A fixture caveat compounds this: the client's
   prompt used `<image>` rather than Qwen VL's expected
   `<|vision_start|><|image_pad|><|vision_end|>`, producing the
   "More image data items provided than corresponding tokens found
   in the prompt" warning on every image prefill. Under
   `validation_plan.md` §6, both conditions ("ablation arm was
   corrupted", fixture may not exercise DeepStack) trigger
   `AMBIGUOUS`. No upstream correctness bug is demonstrated; the
   source-level suspicion is neither confirmed nor refuted. No fix
   was implemented and no upstream issue was opened, per §7.7. GPU 7
   returned clean (4 MiB / 0 % / 0 apps), the 11 foreign compute
   processes on other GPUs are unchanged pre-vs-post, and
   `/data/sglang-fork` HEAD is still `986c89e69…`. Commit
   `test(qwen35): record Qwen3.5 BCG DeepStack verdict`.

6. **Step 6 (harness repair, CPU-only) — landed 2026-08-01.** Under
   `validation_plan.md` Amendment 2, the two Attempt-01 flaws are
   fixed on-branch without touching a GPU. The DeepStack observer
   moves to `language_model.register_forward_pre_hook(hook,
   with_kwargs=True)` scoped to one `general_mm_embed_routine` call
   and removed in `finally`; it records shape/dtype/numel/finite/
   nonzero_frac/abs_sum/sq_sum/SHA-256-16/data_ptr before
   modification, and in zero mode records a second summary proving
   the replacement is really zero. The client emits
   `<|vision_start|><|image_pad|><|vision_end|>` verbatim, records
   the rendered prompt / placeholder count / image count, and hard-
   fails any mismatch. `verdict.py` requires the full 2×2
   (`eager_normal`, `eager_zero_deepstack`, `bcg_normal`,
   `bcg_zero_deepstack`), valid image/placeholder alignment,
   nonzero DeepStack in normal arms, verified zero replacement in
   ablation arms, BCG replay confirmed in BCG arms, and
   ablation sensitivity (`eager_normal ≠ eager_zero_deepstack`
   beyond the eager-repeat noise floor) before returning any
   non-`AMBIGUOUS` verdict. The runner accepts
   `--config bcg_zero_deepstack`. The GPU allowlist widens to
   `{0, 1, 7}` — GPU 1 was extended by the operator on 2026-08-01
   as a standing addition. Proved by
   `scripts/test_instrumentation.py` on CPU (normal-mode hook
   fires on a nonzero tensor without changing output; zero-mode
   hook fires, verified-zero substitution changes output; no hook
   accumulation across repeated calls). Commit
   `fix(qwen35): repair DeepStack instrumentation and image input`.
7. **Step 7 (harness GPU validation) — HARNESS_NOT_DIAGNOSTIC
   (2026-08-01, GPU 1).** Attempt
   `experiments/qwen35_4b/results/harness_gpu1_20260801T062833Z/`
   ran `eager_normal` + `eager_zero_deepstack` on GPU 1 under the
   repaired harness. The `nn.Module.register_forward_pre_hook`
   interceptor fired 111 times per arm on real
   `Qwen3_5ForCausalLM` prefills; the corrected
   `<|vision_start|><|image_pad|><|vision_end|>` placeholder
   produced zero "More image data items…" warnings (down from 2
   per arm in Attempt 01); image data was really consumed
   (greedy output describes the fixture's colours). However,
   `Qwen/Qwen3.5-4B`'s `vision_config.deepstack_visual_indexes = []`
   (verified against every publicly released `Qwen/Qwen3.5-*`
   size: 0.8B / 2B / 4B / 9B / 27B / 35B-A3B), so
   `num_deepstack_embeddings = 0`, `input_deepstack_embeds` is
   allocated with `shape=(N, 0)` / `numel = 0`, and
   `Qwen3_5ForCausalLM.forward`'s DeepStack `add_` branch is
   trivially skipped by its `numel() > 0` guard. Runtime
   instrumentation confirms `nonzero_frac = 0.0` on every image
   request; the zero-substitution guard correctly skips the
   empty tensor; `eager_normal` == `eager_zero_deepstack` bit-for-
   bit. Per the brief's Step 2 fail-path rule and
   `validation_plan.md` Amendment 3, Step 8 (scored 2×2 rerun) is
   **skipped by design**. The source-level BCG DeepStack suspicion
   (F5, F6, F7, F8) remains **not testable** against any
   publicly-released `Qwen/Qwen3.5-*` checkpoint at the pinned
   SGLang SHA. GPU 1 returned clean (0 MiB / 0 % / 0 compute
   apps pre and post); `/data/sglang-fork` HEAD unchanged. Commit
   `test(qwen35): validate DeepStack measurement harness`.
8. **Step 8 (scored 2×2 rerun) — skipped by design.** Would run
   `eager_normal` + `eager_zero_deepstack` + `bcg_normal` +
   `bcg_zero_deepstack` if Step 7 had passed. Skipped because the
   Step 7 evidence establishes the ablation is trivially non-
   diagnostic on this model target.

Follow-up (queued, not on this branch): to test the source-level
BCG DeepStack suspicion at runtime, rebaseline the investigation
onto a checkpoint whose config ships a non-empty
`deepstack_visual_indexes` list (e.g. a Qwen3-VL model — that has
its own PCG investigation on
`debug/v2-imgA-pcg-capture-stream-fix`).

### 7.7 §7 out of scope

- Filing an upstream SGLang issue or PR — deferred until runtime
  reproduction (or definitive disproof) is in hand.
- Implementing a fix — the whole plan is diagnostic; a fix is a
  separate follow-up gated on the verdict.
- Rewriting or repurposing the historical Qwen3-VL PCG evidence
  under §4 or the `debug/v2-imgA-pcg-capture-stream-fix` branch.
  §7 links historically where useful but treats §4 as read-only.
- Editing anything under `/data/sglang-fork`. That fork is
  preserved read-only as historical evidence at `986c89e69`.
- Using any GPU outside the authorised allowlist `{0, 1, 7}` (see
  `experiments/qwen35_4b/validation_plan.md` Amendment 1 and
  Amendment 2).
