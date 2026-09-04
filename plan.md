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

Status as of **2026-08-29** (verified against the live GitHub API):

| # | Title | Priority | Status |
|---|---|---|---|
| 1 | Tracking: next-round follow-ups | meta | open — post refreshed checklist, close last |
| **2** | **Default-overlap Qwen3-VL rebaseline** | **P0** | ✅ COMPLETE / PASS (results under `v2/caseAC_rebaseline/results/`) — closed |
| 9 | Qwen3.5 DeepStack under multimodal prefill BCG | P1 | ✅ **CLOSED 2026-09-03** (`completed`) — verdict `NOT_APPLICABLE_QWEN35`; conclusion write-up in [`experiments/qwen35_4b/issue9_conclusion.md`](experiments/qwen35_4b/issue9_conclusion.md). Spun out the real Qwen3-VL bug → upstream PR [#33726](https://github.com/sgl-project/sglang/pull/33726), open/approved/mergeable (see `experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md`). |
| **4** | **Qwen3-VL image+text + CUDA IPC** | **P0 — active** | ⚠️ PARTIAL. Only `IMG_A_S0_ipc` completed (5/5 reps, 2 000 requests, TTFT p50 64.8 ms); `S2_ipc_pcg` crashed on the capture-stream assertion; `S0_ipc_repeat` / `V0_vllm` / `S0_noipc` unrun. Capture-stream sub-track ✅ concluded 2026-07-29 (§4). Resume plan: `v2/image_text_benchmarks/fixed_generator_plan.md`. |
| 3 | Qwen3.5 SGLang-vs-vLLM transfer check | P1 | ❌ **NOT RUN.** The DeepStack (§7) and GDN studies answered different correctness/mechanism questions; neither is the roadmap's cross-framework Case-A/Case-C comparison. Needs a freshly pinned, version-aligned environment. |
| 5 | Selective / default-on graph-enablement policy | P2 | ❌ blocked on #4. **Must be re-scoped to distinguish PCG from BCG** — the fork now carries a BCG allowlist while this issue was written about PCG. Do not silently substitute one for the other. |

## 3.5 Upstream drift audit (2026-09-03) — CUDA-graph flags were restructured

Verified by reading `upstream/main` @ `2da5802bfa`. This changes how #4 and #5
must be run and written; it does **not** invalidate #2's numbers.

**1. The PCG flag was renamed, not removed.** `--enforce-piecewise-cuda-graph`
is now a *deprecated alias* for `--cuda-graph-backend-prefill=tc_piecewise`
(`server_args.py:3988`). The old flag still works and still emits a
deprecation notice. `tc_piecewise` remains a first-class backend with its own
compile path — it is **not** slated for removal.

Prefill backends are now `full | breakable | tc_piecewise | disabled`, settable
per phase via `--cuda-graph-backend-{decode,prefill}` or the
`--cuda-graph-config` JSON. Other renames touching our scripts:
`--disable-cuda-graph` → `--cuda-graph-backend-{decode,prefill}=disabled`;
`--piecewise-cuda-graph-tokens` → `--cuda-graph-bs-prefill`;
`--enable-breakable-cuda-graph` → `--cuda-graph-backend-prefill=breakable`.

**2. BCG is now the default prefill backend on CUDA.**
`default_prefill_backend()` returns `Backend.BREAKABLE` on CUDA and
`TC_PIECEWISE` elsewhere (`cuda_graph_config.py:112`). The v1/v2-era premise
"SGLang ships no prefill graph by default" is obsolete **for text models**.

**3. For Qwen3-VL the auto-disable still fires — today.** There are now two
multimodal opt-in allowlists in `configs/model_config.py`:
`multimodal_piecewise_cuda_graph_supported_model_archs` (Kimi K2.5, MiniMax M3
Sparse) and `multimodal_breakable_cuda_graph_supported_model_archs`. **Qwen3-VL
is on neither in upstream `main`.** So the resolution walks: default
`breakable` → `disable_breakable_cudagraph_if_incompatible` → rule
`"multimodal model"` fires → prefill backend `disabled`. **The #2 root cause is
intact on current upstream**, and #4's baseline premise still holds.

**4. But PR #33726 flips exactly that — for Qwen3-VL specifically.** Our own PR
adds `Qwen3VLForConditionalGeneration` and `Qwen3VLMoeForConditionalGeneration`
to the *breakable* allowlist. On merge, Qwen3-VL's default prefill backend
becomes **BCG-on**, not disabled. Consequences:

- **#4's `S0` baseline is about to move.** A default-arm number taken before
  the merge measures a default that will not exist after it. Either run #4
  against the post-merge default, or pin the pre-merge SHA and label the arm as
  historical — do not mix.
- **#5 is largely pre-empted, and by BCG rather than PCG.** #5 was written to
  prototype selective/default-on *piecewise* for Qwen3-VL; the arch-allowlist
  mechanism it would have proposed already exists and is what #33726 uses. #5
  should be re-scoped to "which backend should Qwen3-VL default to, and on what
  evidence" — a three-way question (`breakable` / `tc_piecewise` / `disabled`),
  not the original binary.

**5. Explicitly selecting a backend still bypasses every auto-disable rule.**
`apply_cuda_graph_compatibility` returns early when `(prefill, backend)` is in
`server_args._cuda_graph_config_locked` (`cuda_graph_hook.py:115`), and the
parser locks any key set from a non-default source. So the PCG arm is still
runnable on Qwen3-VL, and the "explicit flag beats the cascade" contract that
#2 relied on is preserved by design rather than by accident.

### Consequence for the #4 arm matrix

The two-arm SGLang design (`default` vs `+PCG`) no longer spans the space. The
minimum honest matrix is now:

| arm | flag | measures |
|---|---|---|
| `S0` | none | the real production default — **whichever backend that resolves to on the pinned SHA; record it, never assume** |
| `S1` | `--cuda-graph-backend-prefill=disabled` | true no-prefill-graph floor |
| `S2` | `--cuda-graph-backend-prefill=tc_piecewise` | the #2 PCG lever |
| `S3` | `--cuda-graph-backend-prefill=breakable` | BCG, the incoming default |

Every arm must log the **resolved** backend at startup. Under the new
resolution cascade an unsupported request is silently downgraded to
`disabled`, so a flag being accepted is not evidence the backend engaged.


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

> Rewritten 2026-08-29. The §7 correctness detour is finished and its fix is
> upstream; the profiling mainline resumes at #4. Detailed work packages and
> acceptance gates:
> [`reports/2026-08-28_profiling_resumption_audit.md`](reports/2026-08-28_profiling_resumption_audit.md).

1. **Tracker hygiene (half day, no GPU).** Close #9 as `NOT_APPLICABLE_QWEN35`,
   stating that no shipped Qwen3.5 checkpoint exercises DeepStack and
   cross-linking the separate Qwen3-VL FAIL→PASS evidence. Post a refreshed
   checklist on #1. Amend #5 to name PCG and BCG as distinct backends.

2. **Close out the BCG handoff (engineering lane, low effort).** The fix is in
   an approved, mergeable PR. What remains is watching CI on `c31e6fe315`,
   optionally tightening `assertRaises` → `assertRaisesRegex`, and — if a
   current devbox becomes available — repeating the smoke on a
   production-representative stack. Tracked in
   [`upstream_handoff.md`](experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md).
   Do **not** treat this as closing #4 or #5: BCG ≠ PCG.

3. **#4 — the next GPU work. Detailed plan: [§11](#11-issue-4-execution-plan-v3-drafted-2026-09-03-not-started).**
   Superseded 2026-09-03: the upstream audits (§3.5 and §11.1) found that all
   four of #4's levers now have silent-degradation paths — the deprecated IPC
   env, the IPC pool's CPU fallback, the PCG capture-stream assertion demoted to
   a warning, and the moving default backend. The two-arm `default` vs `+PCG`
   bracket would still run and would measure the wrong thing. v3 replaces it
   with 6 SGLang arms + a vLLM anchor, gated behind a per-arm **engagement
   verifier**; the previous `IMG_A_S0_ipc` number is retired as historical.

4. **#3 — the actual Qwen3.5 transfer study (1–2 GPU days, parallelisable).**
   Confirm both frameworks can serve the same checkpoint and API semantics
   first. Repeat Phase-0 parity, then clean text-only Case A (128→128, c=1) and
   Case C (512→128, c=16) with SGLang default, the canonical supported graph
   intervention, and a vLLM anchor. Do not assume the Qwen3-VL PCG lever is
   valid for Qwen3.5 — its supported route is BCG unless a source audit says
   otherwise. The GDN report is an appendix, not a substitute.

5. **#5 — decide policy from the matrix (after #4).** Build backend × modality ×
   load explicitly, then implement the *smallest* policy the evidence supports.
   If benefit is confined to stable text c=1, choose selective enablement with
   observable hit/miss/fallback reasons rather than a global force-on.

6. **#1 — restructure the final report and close the umbrella** once #3, #4, #5
   and #9 have final dispositions. Optional B/D baselines and GDN L2Norm fusion
   must not block closure.

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

> **Reframe after Attempt 02 (2026-08-01):** the Qwen3.5 target was
> shown to be unable to exercise DeepStack at all — every shipped
> `Qwen/Qwen3.5-*` release carries
> `vision_config.deepstack_visual_indexes = []`, so `input_deepstack_embeds`
> is empty and the DeepStack `add_` branch is trivially skipped. A cross-arch
> audit ([`experiments/qwen35_4b/latent_bug_analysis.md`](experiments/qwen35_4b/latent_bug_analysis.md))
> shows the intersection of "on BCG allowlist" and "actually populates
> DeepStack" is empty on current upstream. The source-level suspicion
> remains valid but is a **latent regression** rather than a live-production
> bug. Attempt 03 will retarget the repaired harness to
> `Qwen/Qwen3-VL-8B-Instruct` under a profiler-owned test-only
> monkey-patch that adds Qwen3-VL to the BCG allowlist at runtime, to
> convert the latent hypothesis into live-fire evidence. No source edit
> to the frozen SGLang checkout; no upstream fix or issue filed yet.

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
- **`NOT_APPLICABLE_QWEN35`** (closure verdict, added 2026-08-03) —
  The primary target does not exercise the code path under test on
  any shipped release, and the harness will not fabricate the input
  by editing the checkpoint. See
  [`experiments/qwen35_4b/hypothesis.md`](experiments/qwen35_4b/hypothesis.md)
  §5 and Amendment 5 for the full criteria and preservation rules.

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
9. **Step 9 (Attempt 03 retarget — Qwen3-VL under monkey-patched
   BCG) — `FAIL_BCG_DEEPSTACK` (2026-08-01, GPU 1).** Attempt
   `experiments/qwen35_4b/results/attempt_gpu1_20260801T115524Z/`
   ran the full 4-arm 2×2 (`eager_normal`, `eager_zero_deepstack`,
   `bcg_normal`, `bcg_zero_deepstack`) against
   `Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0` under the profiler-owned
   test-only monkey-patch (`scripts/bcg_allowlist_patch.py`, opt-in
   via `QWEN35_PATCH_BCG_ALLOWLIST=1` or `--patch-bcg-allowlist`).
   Pre-state allowlist `["Qwen3_5ForConditionalGeneration",
   "Qwen3_5MoeForConditionalGeneration"]`; post-state adds
   `Qwen3VLForConditionalGeneration` and
   `Qwen3VLMoeForConditionalGeneration`. Frozen SGLang source at
   `58974ca16` unchanged (`git diff --stat` empty). All arms
   served the 893-token scored image prefill; both BCG arms served
   it with `cuda graph: True` and zero `bcg_execute_body_error`.
   DeepStack tensor observed at the LM entry (`module_class =
   Qwen3LLMModel`, `module_class_recognised = true`) with
   `shape=[896, 12288]` (= `[N, hidden_size * 3]` for text
   hidden=4096) and `nonzero_frac ≈ 0.98` in the two normal arms;
   the zero-substitution guard verified `nonzero_frac → 0.0` /
   `abs_sum → 0.0` in the two zero arms. **Live-fire verdict:
   `bcg_normal` is bit-identical to `bcg_zero_deepstack` (20/20
   tokens equal, mean logprob diff 0.0) and both track
   `eager_zero_deepstack`, while `eager_zero_deepstack` diverges
   from `eager_normal` at the very first non-boilerplate token
   (7/15 common prefix, l1_max_abs_diff = 1.14)**. This is the
   predicted `FAIL_BCG_DEEPSTACK` signature: SGLang's
   `replay_layer_forward` bridge silently drops the DeepStack
   contribution under BCG replay. The source-level suspicion in
   `experiments/qwen35_4b/latent_bug_analysis.md` § 2 is confirmed
   live-fire, with the caveat that the reproduction depends on the
   runtime monkey-patch — no shipped upstream configuration
   currently reaches this code path. GPU 1 returned to qualifying
   after cleanup; `/data/sglang-fork` HEAD unchanged. Commits
   `feat(qwen35): retarget harness to Qwen3-VL under monkey-patched BCG`
   (CPU scaffolding + validation_plan.md Amendment 4) and
   `test(qwen35): rerun 2x2 with Qwen3-VL under patched BCG`
   (GPU 2×2). Follow-up filing decision — defensive upstream note
   about `replay_layer_forward` copying `input_deepstack_embeds` in
   parallel with `input_embeds` — is out of scope for this pass
   per the brief.

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

### 7.8 Sub-track closure (2026-08-03) — verdict `NOT_APPLICABLE_QWEN35`

The Qwen3.5-4B DeepStack sub-track closes with verdict
**`NOT_APPLICABLE_QWEN35`** (see
[`experiments/qwen35_4b/hypothesis.md`](experiments/qwen35_4b/hypothesis.md)
§5 and Amendment 5). The Qwen3.5 target does not exercise the code
path under test on any shipped release, and the harness will not
fabricate the input by editing the checkpoint. Attempts 01, 02, 03
are preserved verbatim; Attempt 03's `FAIL_BCG_DEEPSTACK` on
Qwen3-VL under a profiler-owned test-only monkey-patch stands as an
exhibit of the latent regression on a different model, **not** as the
closing verdict for Qwen3.5. Investigation continues on a distinct
Qwen3.5 code path — see §8.

## 8. Sub-track — Qwen3.5-4B GDN prefill-BCG investigation (pivot, active)

Active branch: `debug/qwen35-4b-gdn-prefill-bcg` (cut from
`debug/qwen35-4b-bcg-deepstack` at close-out commit).
Investigation anchor: `experiments/qwen35_4b/gdn/README.md`.

**Target and constraints (from operator brief 2026-08-03):**

- Model: `Qwen/Qwen3.5-4B`, BF16, single GPU, TP=1.
- No MTP, no quantization, no MoE, no custom model patches.
- 4-arm matrix: `{eager prefill + eager decode}`,
  `{prefill BCG only}`, `{full decode CUDA Graph only}`,
  `{both enabled}`.
- Sweep: prompt length ∈ `{128, 512, 2048, 8192}` ×
  batch size ∈ `{1, 4, 16, 32}`.
- Tool: Nsight Systems (`nsys profile`); measure kernel counts,
  CPU launch gaps, graph breaks, TTFT, prefill throughput per arm.
- GDN focus areas: input projections, fused split/reshape, linear-
  attention call, gated norm, output projection. **Do not assume**
  recurrent-state handling is faulty; first identify a repeated
  graph break or measurable launch-overhead bottleneck.
- **Correctness gates (blocking before any perf claim):**
  eager-vs-BCG token/logprob equivalence; request-order isolation;
  chunked-prefill equivalence; graph-bucket equivalence.
- **Rule:** do not modify upstream SGLang source until the baseline
  profile identifies one specific BCG limitation.
- GPU allowlist widened to `{0..7}` per
  [`gdn/validation_plan.md`](experiments/qwen35_4b/gdn/validation_plan.md)
  Amendment 1 (2026-08-03, operator authorisation); idle-verification
  rules from `experiments/qwen35_4b/validation_plan.md` Amendments 1
  and 2 continue to apply on every attempt.
- Preservation invariants unchanged: read-only `/data/sglang-fork`
  at `986c89e69`; frozen SGLang checkout unchanged; §4 evidence
  read-only.

### 8.1 Phase log

- **2026-08-03** — GDN charter landed (`1b6c1b1`); CPU-only foundation
  scaffolding + fixture + preflight + tests (`de0569d`); 4-arm runner
  + sweep client + baseline instrumentation (`ff66db6`); correctness
  verifier + verdict runner + Nsight wrapper (`d9e185f`); allowlist
  widened to `{0..7}` (`47e6a37`); three live-fire fixes surfaced by
  smoke — preflight `text_config` (`66d91cd`), canonical prefill/
  decode flag names (`271a666`), setsid PID capture (`5736f96`);
  smoke test on GPU 2 recorded `SCAFFOLDING_PASS` (`2490057`).
- **2026-08-03** — Phase 1 consolidated audit landed
  ([`gdn/audit.md`](experiments/qwen35_4b/gdn/audit.md)). Three
  parallel agent reports (repo/harness, source/BCG, validation/
  methodology) merged into one document with three `SIGNAL_GOOD`
  records. 13 blocking harness gaps + no major blocker. Leading perf
  hypothesis for the smallest-cell test: GDN alt-stream branch is
  active under BCG for every prefill bucket with `padded_num_tokens <
  1024` (`_gdn_use_alt_stream = True` unconditionally on CUDA at
  `models/qwen3_5.py:128`; `get_is_capture_mode()` True during both
  BCG capture and replay; no BCG-specific short-circuit exists —
  only TC piecewise zeroes the threshold). One correctness risk
  added to Gate-1 attention list (R13.4 alt-stream capture join
  integrity, `runner_backend_utils/breakable_cuda_graph/breakable_cuda_graph.py:112-136`).
- **2026-08-03** — Stage 1 (`T8` arm self-repeat, `T9` first-token
  cross-arm, `T10` diagnostic signal) — all 4 arms internally
  deterministic; A0/A1/A2/A3 first-token 8/8 agreement across arms
  on the smallest cell (`p128 b1`). Correctness gate PASSED. Commits
  `d29adc5`, `7ab57bf`, `ba1cb5c`, `5efe0d3`, `84e5fdb`.
- **2026-08-03** — Stage 2 (A0/A1 reproducibility on steady-state
  kernel counts, windowed extractor). H_A supported: `+13.6 % ± 0.4 %`
  A1-vs-A0 kernel-count inflation, reproducible across reps. Commit
  `1aefa29`.
- **2026-08-03** — Stage 3 (threshold ladder to test the `<1024`
  alt-stream hypothesis, 4 prompt lengths × 2 arms × 2 reps = 16
  captures). **H12.1 REJECTED** — inflation is essentially constant
  (13.4–13.7 %) across all prompt lengths, independent of whether
  the padded bucket sits above or below the 1024 alt-stream
  threshold. `cudaGraphLaunch/request` is constant at 36.3 across
  every cell (evidence file
  [`gdn_stage3_gpu6_20260803T234412Z/stage3_summary.md`](experiments/qwen35_4b/gdn/results/gdn_stage3_gpu6_20260803T234412Z/stage3_summary.md)).
  Commits `75c55fc`, `1ffa8b0`.
- **2026-08-03** — Stage 4 (mechanism attribution from existing Nsight
  kernel names, no NVTX needed). Attribution:
  [`gdn/stage4_mechanism.md`](experiments/qwen35_4b/gdn/stage4_mechanism.md).
  Under BCG, SGLang switches Qwen3.5-4B's GDN prefill from the eager
  "recurrent packed" kernel family to the FLA "chunk" kernel family.
  Nine chunk-family kernels each fire ~4,176 additional times per
  Stage-2 trace (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`,
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel`, `chunk_fwd_kernel_o`,
  `chunk_local_cumsum_scalar_kernel`, `recompute_w_u_fwd_kernel`,
  `fused_qkv_split_gdn_prefill_kernel`, `fused_gdn_gating_kernel`,
  `_causal_conv1d_fwd_kernel`, and `l2norm_fwd_kernel` at 2×).
  Wall-clock impact: `+22 – 61 ms/req (0.6–1.6 %)` — real but modest
  because decode dominates e2e. **The FLA chunk kernel family is the
  structurally-required path for BCG on hybrid GDN**; removing it
  would disable BCG. Commit `408e99f`.
- **2026-08-03** — Final report + verdict `PASS_BCG_GDN_NOTABLE_GAP`:
  [`gdn/final_report.md`](experiments/qwen35_4b/gdn/final_report.md).
  No source patch is justified at the frozen SHA. The kernel-count
  inflation is the intrinsic cost of graph-compatible prefill on this
  architecture. Investigation pivots to an *incremental optimization*
  question: which of the chunk-family launches can be fused to
  reduce the number of launches without breaking BCG or correctness?
  This becomes §9. Commit `3531fd1`.

## 8.2 §8 out of scope

- Modifying the FLA chunk-kernel family at the frozen SGLang SHA. The
  Stage-4 verdict is `NOTABLE_GAP`, not `BUG`; the correct next step
  is incremental optimization (§9), not intrusive rewiring.
- Switching the GDN prefill back to the recurrent-packed kernel family
  under BCG. That would disable graph capture and defeat the very
  behaviour BCG exists to provide. Discussed and rejected in
  [`gdn/optimization_design.md`](experiments/qwen35_4b/gdn/optimization_design.md)
  Deliverable 3.
- Filing an upstream issue based on Stage 1–4 alone. The observed
  behaviour is a known structural trade-off, not an upstream defect.

## 9. Sub-track — Qwen3.5-4B GDN L2Norm-fusion optimisation (deferred)

> **Status: deferred, not failed.** Design review clean; production
> code path confirmed; hot-path re-confirmed on live traces; prototype
> written and Option-b signature adapted to the running fork.
> Bit-exact parity test not yet run because the host nvidia driver
> was upgraded on 2026-08-04 12:53 UTC to `595.71.05`, breaking the
> torch 2.11.0+cu130 build's `cuInit(0)` (returns
> `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`). Continuation gate is external
> to this project — driver rollback, torch upgrade, or alternate
> host. All work is reversible; nothing has been merged into
> `/data/sglang-fork` or the frozen checkout.

Active branch: `debug/qwen35-4b-gdn-prefill-bcg`. Anchor documents:
[`gdn/optimization_design.md`](experiments/qwen35_4b/gdn/optimization_design.md)
(design + feasibility), [`gdn/optimization_review.md`](experiments/qwen35_4b/gdn/optimization_review.md)
(Stage-1 independent 3-reviewer audit), and
[`gdn/optimization_review_addendum.md`](experiments/qwen35_4b/gdn/optimization_review_addendum.md)
(Stage-2 progress + blocker).

### 9.1 Why the direction was investigated

Stage 4 (§8.1) attributed the `+13.6 % ± 0.4 %` A1-vs-A0 kernel-count
inflation to 9 chunk-family kernels. The `PASS_BCG_GDN_NOTABLE_GAP`
verdict closed the *bug* question and opened the *optimization*
question: since the chunk family is structurally required, the only
value-generating direction is to *fuse* launches inside that family
so BCG pays fewer per-launch dispatches per prefill without
disturbing the graph shape. The candidate had to be (a) numerically
safe, (b) upstream-acceptable as a single small PR, (c) implementable
in ≤ 1 week, and (d) demonstrably reducing kernel-launch count on
the live traces. Six fusion candidates (F1–F6) were surveyed; F1
(fuse the two consecutive `l2norm_fwd_kernel` launches for Q and K
inside `chunk_gated_delta_rule_fwd`) was selected as the standalone
PR pick.

### 9.2 Confirmed execution path and production call sites

Verified by direct reading of `/data/sglang-fork` at HEAD
`986c89e69c…`. The paired-l2norm pattern (two independent
`l2norm_fwd` calls, one on Q one on K, back-to-back, guarded by
`use_qk_l2norm_in_kernel=True` — always True on the SGLang GDN
dispatcher path) appears at **5 production call sites**:

1. `python/sglang/srt/layers/attention/fla/chunk.py:108-110` — GDN
   chunk-prefill under the Triton dispatcher.
2. `python/sglang/srt/layers/attention/fla/kda.py:1155-1156` — KDA
   (Kimi delta attention) chunk-prefill.
3. `python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py:295-296`
   — GDN flashinfer backend, unconditional pre-norm.
4. `python/sglang/srt/layers/attention/linear/kernels/gdn_cutedsl.py:133-134`
   — GDN CuteDSL backend, via `self._l2norm_fn`.
5. `python/sglang/srt/layers/attention/linear/kernels/kda_cutedsl.py:121-122`
   — KDA CuteDSL backend, via `self._l2norm_fn`.

Plus 4 benchmark sites (`bench_gdn_prefill.py:212/213, 396/397`;
`bench_gdn_prefill_cutedsl.py:146/147, 272/273`). No single-tensor
`l2norm_fwd` caller exists in the SGLang tree — every existing call
is one of a paired q+k invocation, so a paired helper amortises
across all five backends.

Q and K have identical shape/dtype/stride on the Qwen3-Next / Qwen3.5
GDN path: `num_q_heads == num_k_heads` and `head_q_dim == head_k_dim
== 128` (verified from
`python/sglang/srt/models/qwen3_next.py:238-242` and
`python/sglang/srt/configs/qwen3_next.py:205-208`, gated by
`gdn_backend.py:87-88`). Both tensors are fresh contiguous
allocations produced by `fused_qkv_split_gdn_prefill_kernel`
(`triton_gdn_fused_proj.py:373-387`). The D=128 route triggers the
D≤512 branch of the l2norm launcher.

### 9.3 Hot-path and Nsight findings

Existing Stage-3 A1_p128_rep1 nsys capture (evidence at
[`gdn_stage3_gpu6_20260803T234412Z/A1_p128_rep1/raw/`](experiments/qwen35_4b/gdn/results/gdn_stage3_gpu6_20260803T234412Z/A1_p128_rep1/raw/)),
re-extracted 2026-08-04 via
`nsys stats --report cuda_gpu_kern_sum`:

| arm | kernel | launches | total time | avg per launch |
|---|---|---|---|---|
| A0_p128_rep1 | `l2norm_fwd_kernel` | 576 | 635 μs | 1.10 μs |
| A1_p128_rep1 | `l2norm_fwd_kernel` | 8,928 | 35.4 ms | 3.97 μs |
| Δ | — | +8,352 | +34.8 ms/trace | — |

The Δ 8,352 equals exactly 2 × 4,176 (the per-trace Δ of every
other chunk-family kernel), consistent with l2norm firing twice per
prefill layer as expected (once for Q, once for K). Fusing halves
the launch count on the chunk-family l2norm bucket to 4,176; savings
estimate 4,176 × 3.97 μs = **16.6 ms per trace = 1.66 ms per prefill**
(10 prefills / trace).

### 9.4 Proposed fused-kernel design (Option-b signature)

Introduce a new `l2norm_fwd_pair(x_q, x_k, eps, output_dtype)` helper
in `python/sglang/srt/layers/attention/fla/l2norm.py` that dispatches
one Triton launch fusing both q and k reductions:

* Grid shape `(cdiv(max(T_q, T_k), BT), 2)`.
* `pid1 ∈ {0, 1}` — compile-time (`tl.constexpr`) branch picks the
  (x_q, y_q, T_q) or (x_k, y_k, T_k) triple. Only one path is emitted
  per program — no runtime `if` on pid1, no register-pressure hit at
  `num_warps=8`.
* Kernel body byte-identical to `l2norm_fwd_kernel`: `tl.make_block_ptr`
  over `(T, D)` with block `(BT=16, BD)`, fp32 upcast, per-row
  reduction, `1 / sqrt(var + eps)`, stored via `tl.make_block_ptr`.
* Same `num_warps=8`, `num_stages=3`, same `BT=16`, same
  `MAX_FUSED_SIZE = 65536 / element_size`.
* D > 512 fallback dispatches two independent `l2norm_fwd_kernel1`
  launches (matches the existing per-row branch exactly).
* Launcher asserts `D_q == D_k` and `stride(-1) == 1` for both
  outputs.

Rationale for Option-b over Option-a (`(cdiv(T*H, BT), 2)` with
shared T): Option-b tolerates future `T_q ≠ T_k` without a rewrite
at negligible cost (4 extra kernel args). Prototype implemented in
scratchpad:
`/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/f1_prototype/l2norm_fwd_pair.py`.

Call-site change is 2 lines at each of the 5 production sites:

```python
if use_qk_l2norm_in_kernel:
    q, k = l2norm_fwd_pair(q, k)
```

### 9.5 Feasibility and reviewer conclusions

Three independent parallel reviewers were run on the design (Stage 1
of the L2Norm sub-track, commit `bfe1a84`) covering kernel feasibility,
performance skepticism, and integration/correctness:

* **Kernel feasibility → VIABLE.** Q and K identical shape/stride;
  bit-exact numerics; fusion implementable in `≈ 40` LOC; no autograd
  backward exists in SGLang's `L2NormFunction`; risk items limited to
  register pressure at `num_warps=8` (verify with a smoke bench
  before landing).
* **Performance skepticism → NEGLIGIBLE at wall-clock,
  MEASURABLE at kernel-count.** Expected saving `≈ 0.5 – 2 ms/req`
  prefill (below Stage-3's `0.36 %` within-cell noise floor);
  `≈ 0.05 %` of e2e (decode dominates 3200 ms of 3800 ms). Nsys
  kernel-count halving is the primary detectable evidence.
* **Integration/correctness → CLEAN.** Five production call sites
  benefit from a single helper; no single-tensor caller to preserve;
  no backward parity needed; `use_qk_l2norm_in_kernel` invariantly
  `True` on the production path; upstream route is FLA-first
  (comment `Adapt from https://github.com/fla-org/flash-linear-attention/...`
  in `l2norm.py:1`) with a local wrapper as a fallback.

Verdict: **`PLAN_ACCEPT`** (commit `bfe1a84`) with three amendments to
the original design: extend fusion to all 5 call sites (not just
`chunk.py:108-110`); prefer Option-b signature; frame any PR as
"launch-count reduction under CUDA-graph replay", not as wall-clock
speedup — honesty preserves upstream trust.

### 9.6 Expected kernel-launch reduction

Per prefill per GDN layer: **−1** on the chunk-family l2norm bucket
(2 launches → 1). On Qwen3.5-4B: 24 GDN layers × ≈ 17.4 BCG replays
per layer per prefill = ≈ 418 launches removed per prefill. In the
Stage-3 rig (10 prefills / trace) that is **≈ 4,176 launches / trace
eliminated**, directly measurable via
`nsys stats --report cuda_gpu_kern_sum` and cross-checked against the
Stage-4 per-kernel Δ table.

### 9.7 Realistic kernel-level and end-to-end benefit

* **Per-launch amortised.** ≈ 3.97 μs × 4,176 = **16.6 ms / trace**
  = **1.66 ms / prefill** (Stage-3 arithmetic; top end of the
  Stage-1 estimated `0.5 – 2 ms` band).
* **Prefill wall-clock.** ≈ 5 % reduction of the +30 ms A1-vs-A0
  gap; ≈ 3 – 7 % of the ~30 ms prefill delta if measured
  prefill-only.
* **End-to-end wall-clock.** ≈ 0.03 – 0.07 % — **below the Stage-3
  within-cell rep variability of 0.36 %**. Not detectable on total
  request latency, expected null result.
* **Better alternatives ranked below F1.** F2 (fold gating into
  cumsum) and F3 (fold l2norm into `fused_qkv_split_gdn_prefill`)
  save more launches but require touching SGLang-owned kernels
  with heavier rewrites; both are candidates for follow-on PRs once
  the reusable `l2norm_fwd_pair` helper is battle-tested.
* **Autotune enablement of `wy_fast.py`** (a parallel-track pick
  entirely orthogonal to F1) has unknown magnitude but a real per-
  launch improvement ceiling; should be pursued in parallel, not as
  a substitute for F1.

### 9.8 Correctness, integration, backward, and production risks

* **Correctness (numerical).** Very low. Bit-exact by construction —
  same tile order, same eps, same fp32 upcast, per-row reductions
  are independent between q and k. Verification is one unit test
  (`torch.equal` on `l2norm_fwd_pair` vs
  `(l2norm_fwd(q), l2norm_fwd(k))`) across the production shapes.
* **Correctness (shape/stride).** Low. Q and K are guaranteed same
  shape by the Qwen3-Next model config. The Option-b signature makes
  a future GQA-style split silent rather than a shape bug.
* **Backward parity.** Not needed. SGLang's `L2NormFunction`
  (`l2norm.py:125-127`) has only `forward`; no `l2norm_bwd` exists
  in the SGLang tree. Upstream FLA retains a backward; an upstream
  PR to FLA would need to add a paired backward variant for training
  users, but SGLang inference is unaffected.
* **Alternate GDN paths.** Unaffected. `decode`, `packed_decode`,
  `target_verify` (in `gdn_triton.py:46-241`) all fuse l2norm
  *inside* their respective kernels, so no separate `l2norm_fwd`
  launches exist to fuse there.
* **Register pressure at `num_warps=8`.** Low. Compile-time branch
  on `pid1` (constexpr) means only one path is emitted per program;
  smoke bench (planned M4) will catch any regression before landing.
* **Upstream maintenance risk.** Low. The file explicitly annotates
  `Adapt from https://github.com/fla-org/flash-linear-attention/…`,
  so the clean route is FLA first, SGLang inherits at next sync;
  fallback is a local SGLang wrapper.
* **Overselling risk in the PR description.** Real. The wall-clock
  story is a null result at this measurement resolution. The
  correct PR framing is "N launches removed under CUDA-graph replay,
  matching the existing 3→2 kkt+solve fusion at `chunk_fwd.py:349-357`",
  not "prefill speedup".

### 9.9 Work already completed (milestones M1 – M2)

* **M1 hot-path confirmation → EXEC_ACCEPT** (2026-08-04). Fresh
  `nsys stats` on existing Stage-3 nsys-rep files confirmed
  `l2norm_fwd_kernel` counts and per-launch avg latency (see §9.3).
* **M2 prototype fused kernel → EXEC_UNCLEAR → EXEC_ACCEPT** (same
  day). While adapting the prototype, discovered that the design
  report and Stage-1 review had cited a *different* SGLang checkout
  (scratchpad reference at HEAD `58974ca16c…`, paths under
  `python/sglang/kernels/ops/attention/fla/`) than the SGLang the
  profiler actually loads (`/data/sglang-fork` at HEAD `986c89e69c…`,
  paths under `python/sglang/srt/layers/attention/fla/`). Verified
  the F1 fusion target at `srt/layers/attention/fla/chunk.py:108-110`
  is byte-identical to the scratchpad citation; the live-fork
  kernel signature adds a `NB: tl.constexpr` argument and drops
  `do_not_specialize=["T"]`. Prototype adapted accordingly and
  preserved under
  `<scratchpad>/f1_prototype/l2norm_fwd_pair.py`. Both the drift
  and the adaptation are documented in
  [`optimization_review_addendum.md`](experiments/qwen35_4b/gdn/optimization_review_addendum.md).

### 9.10 Current blocker — CUDA / driver mismatch

The host nvidia driver was upgraded to `595.71.05` on
2026-08-04 12:53 UTC (evidence:
`/proc/driver/nvidia/version` modification time; verified via
`stat`). Torch 2.11.0+cu130 requires driver in one of the ranges
`[535,536), [550,551), [565,566), [570,571), [575,576)`;
`595` is outside all of them. `cuInit(0)` returns error
`803 = CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`, verified independently by
`ctypes` on `libcuda.so.1`. `nvidia-smi` still works because it uses
a separate compatibility layer, but no Python process on this host
can initialise CUDA under the current torch build. The pre-upgrade
driver at the time of the Stage-3 Aug-3 captures was compatible;
those captures remain valid evidence for §9.3.

M3 (bit-exact parity test), M4 (kernel-latency microbench), M5–M8
(integration + validation) all require a working CUDA runtime and
are therefore paused.

### 9.11 Relevant commits and artefacts

Commits on `debug/qwen35-4b-gdn-prefill-bcg`:

| Commit | Message | Kind |
|---|---|---|
| `cc21e0e` | `docs(qwen35): optimization design and feasibility plan for GDN BCG chunk pipeline` | Design report (all 5 deliverables) |
| `bfe1a84` | `docs(qwen35): Stage-1 review for F1 (l2norm q+k fusion) — PLAN_ACCEPT` | 3-reviewer independent audit |
| `f3224d5` | `docs(qwen35): Stage-2 progress addendum — M1/M2 done, M3 blocked on driver upgrade` | Stage-2 execution status + blocker record |

Artefacts on-branch:

* [`gdn/optimization_design.md`](experiments/qwen35_4b/gdn/optimization_design.md)
  — 442-line design + feasibility (F1–F6, ranked opportunities,
  recommendation).
* [`gdn/optimization_review.md`](experiments/qwen35_4b/gdn/optimization_review.md)
  — 260-line Stage-1 review, verdict `PLAN_ACCEPT` with 3 scope
  amendments (5 call sites, Option-b signature, launch-count framing).
* [`gdn/optimization_review_addendum.md`](experiments/qwen35_4b/gdn/optimization_review_addendum.md)
  — 131-line Stage-2 addendum: M1/M2 completion, live-fork
  adaptation, driver-mismatch blocker.

Scratchpad (uncommitted, disposable):

* `<scratchpad>/f1_prototype/l2norm_fwd_pair.py` — fused Triton
  kernel + launcher, Option-b signature, live-fork-compatible.
* `<scratchpad>/f1_prototype/test_parity.py` — bit-exact parity
  test harness, ready to run when CUDA returns.

### 9.12 Why the work is being deferred

The blocker is external to this project (system-level driver
upgrade), not internal to the F1 plan. The review and prototype
work are complete, correct, and ready. Attempting a substitute
validation path (e.g. running on a different host without a matched
frozen SGLang install) would introduce provenance drift and
provide no faster route to the same conclusion. Continuing in the
current environment is not possible.

The user has explicitly authorised a switch to the Qwen3-VL
DeepStack track (§10 below) in parallel. This sub-track is
**deferred**, not abandoned: no code has been discarded, no artefact
overwritten, and every conclusion so far is a `PLAN_ACCEPT`.

### 9.13 Exact conditions and next steps for resuming

Resume when **any** of the following is true:

* nvidia driver is rolled back to a CUDA-13.0-compatible version
  (system-level, out of this project's scope);
* torch is upgraded on the host to a build linking a CUDA runtime
  compatible with driver `595.x` (e.g., cu131 or a pytorch nightly);
* work migrates to an alternate host that already has a compatible
  driver + torch + `/data/sglang-fork` at the same pinned SHA.

Once CUDA works, resume from **M3** in
[`optimization_review.md` §Recommended validation sequence](experiments/qwen35_4b/gdn/optimization_review.md):

1. Run `python3 <scratchpad>/f1_prototype/test_parity.py` — expect
   element-wise equality on all shapes; abort if any mismatch.
2. Write and run the kernel-latency microbench (M4).
3. On pass, create a git worktree of `/data/sglang-fork` off a fresh
   `f1-l2norm-fusion` branch; apply F1 to `fla/l2norm.py` and
   `fla/chunk.py:108-110`. **Do not amend the frozen pin** — the
   fork-branch modification is what changes; the pinned SHA moves
   in a follow-on `chore(qwen35): bump frozen SGLang pin` commit.
4. Re-run Stage-3 A0/A1 on `p128 b1` and confirm `l2norm_fwd_kernel`
   count halves on A1 (M5).
5. Extend to the other 4 production call sites (M6).
6. E2E correctness + perf validation (M7); keep/revert decision (M8).
7. Draft an FLA upstream PR in parallel with the SGLang change; do
   not gate the SGLang change on FLA acceptance.

Do not upstream from `/data/sglang-fork` unchanged — apply the
change to a clean branch off current upstream `main` for the PR.

## 10. Sub-track — Qwen3-VL BCG DeepStack fix (planning, active)

> **Scope pivot from §7.** §7 closed the Qwen3.5 target as
> `NOT_APPLICABLE_QWEN35` because every shipped `Qwen/Qwen3.5-*`
> release carries `vision_config.deepstack_visual_indexes = []`.
> Attempt 03 (2026-08-01, `attempt_gpu1_20260801T115524Z/`) converted
> the source-level suspicion into a live-fire `FAIL_BCG_DEEPSTACK`
> on `Qwen/Qwen3-VL-8B-Instruct` under a profiler-owned test-only
> BCG allowlist monkey-patch. This sub-track picks up from that
> evidence and moves toward a general, production-safe upstream fix.

Active branch: `debug/qwen3vl-bcg-deepstack-fix` (cut from
`debug/qwen35-4b-gdn-prefill-bcg` HEAD `b8c0f45`, which is itself a
strict superset of `debug/qwen35-4b-bcg-deepstack` HEAD `d29b4a6`
and contains all Attempt 01-03 evidence, the DeepStack harness,
the L2Norm sub-track record, and the GDN Stages 1-4 record).

Planning anchor:
[`experiments/qwen3vl_bcg_deepstack_fix/plan.md`](experiments/qwen3vl_bcg_deepstack_fix/plan.md).

### 10.0 R1 upstream audit — DONE (2026-08-04, CPU-only)

Full write-up at
[`experiments/qwen3vl_bcg_deepstack_fix/r1_upstream_audit.md`](experiments/qwen3vl_bcg_deepstack_fix/r1_upstream_audit.md).
Fetched upstream `sgl-project/sglang` HEAD
`e76d0acdc923d992bbda20d4b2bc51db9ac314a7` (2026-08-04 17:17Z) via a
shallow clone under `<scratchpad>/upstream_main/` — no source
touched, no writes to `/data/sglang-fork` or the pinned scratchpad
checkout. Answers to the five questions the operator asked:

1. **Qwen3-VL eligible for BCG on current main?** NO. Allowlist
   `model_config.py:1839-1842` unchanged from the pinned SHA; still
   only `Qwen3_5ForConditionalGeneration` and its MoE variant. The
   `input_deepstack_embeds`-under-BCG failure remains **latent**
   unless monkey-patched (identical to the pinned SHA state).
2. **Non-empty DeepStack reaches LM entry?** YES when routed.
   `general_mm_embed_routine` synthesis path
   (`mm_utils.py:1122-1245`) unchanged from the pinned SHA;
   Attempt 03's `nonzero_frac ≈ 0.98` at the LM entry remains
   representative.
3. **Stable BCG replay slot?** NO. `cuda_graph_buffer_registry.py`
   registers `input_embeds` (plus `mrope_positions` and
   `num_token_non_padded`) for multimodal but no
   `input_deepstack_embeds` slot anywhere.
4. **Where omitted?** Three co-omissions on current main:
   - slot registration (`buffer_registry.py:867-877`);
   - capture-pass `_run_forward`
     (`prefill_cuda_graph_runner.py:660-668` — passes 4 positional
     args, no `input_deepstack_embeds` kwarg → captured graph has
     the DeepStack `add_` branch cold);
   - replay-pass `replay_layer_forward`
     (`prefill_cuda_graph_runner.py:1610-1628` — reads
     `layer_kwargs["input_embeds"]` and copies into the slot, but
     ignores `layer_kwargs["input_deepstack_embeds"]`).
5. **Can `input_embeds` design generalize safely?** YES, cleanly.
   The three-site register-slot-and-copy pattern already landed for
   `input_embeds` maps directly onto `input_deepstack_embeds`
   with a `num_deepstack_embeddings > 0` gate so Qwen3.5-style
   empty configurations see zero allocation and zero copy overhead.

**Substantive change to the working hypothesis since the pinned-SHA
plan.** The prior text described `replay_layer_forward` as
"drop `**layer_kwargs`, forward outer `**kwargs`", which was true
of the fork snapshot at `986c89e69c…`. On current main this
`input_embeds`-half is fixed (slot registered + kwarg read + copy
performed at replay). The **DeepStack half is unfixed** and now
reads as a symmetric absence — the fix delta is a three-site
mirror of a landed pattern, which is smaller and better-framed
for upstream review than the pinned-SHA plan assumed.

**Harness compatibility on current main.** All class-name and
import-path dependencies verified stable: `Qwen3LLMModel` still at
`qwen3_vl.py:1106` (was `:1104` on the fork); the monkey-patch
symbol path `sglang.srt.configs.model_config.multimodal_breakable_cuda_graph_supported_model_archs`
still importable; `general_mm_embed_routine` still writes
`other_info["input_deepstack_embeds"]`. CPU harness self-test
(`test_instrumentation.py`) passed 2026-08-04. No harness script
edits required to point at a current-main clone.

**Consequence for the R2-R8 ladder.** R3 (upstream-current
reproduction) is downgraded from "conditional on R1" to "optional
confirmation" because R1 answered the upstream-current source-level
question definitively. R4-R8 stay as-is. R2 still gates on the
shared driver blocker.

### 10.1 Prior evidence status

**Valid, preserved verbatim.**

* Source-level `replay_layer_forward` diagnosis (§7.3(4)) — verified
  again against `/data/sglang-fork` at
  `python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py:923-929`.
  Body drops `**layer_kwargs` (which contains
  `input_deepstack_embeds`) and forwards enclosing `**kwargs`.
* Live-fire `FAIL_BCG_DEEPSTACK`
  (`attempt_gpu1_20260801T115524Z/verdict.md`). Signature: `bcg_normal`
  bit-identical to `bcg_zero_deepstack` (l1_max_abs 0.0); both track
  `eager_zero_deepstack` within bf16 noise (l1_max_abs 0.066);
  `eager_zero_deepstack` diverges from `eager_normal` at first
  non-boilerplate token (7/15 common prefix, l1_max_abs 1.15). Textbook
  zero-DeepStack-under-BCG signature.
* Cross-arch DeepStack + BCG audit
  ([`latent_bug_analysis.md`](experiments/qwen35_4b/latent_bug_analysis.md)
  §2) — intersection of "on BCG allowlist" and "populates DeepStack"
  is empty on upstream `main @ 58974ca1`, hence latent-regression
  framing.
* Repaired harness under `experiments/qwen35_4b/scripts/`
  (`server_launcher.py`, `bcg_allowlist_patch.py`, `client.py`,
  `instrumentation.py`, `verdict.py`, `bootstrap/sitecustomize.py`)
  with CPU-only tests passing.
* Byte-pinned image fixture SHA-256
  `8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`.

**Must be revalidated.**

* Current upstream `main` HEAD — has anything landed since
  `58974ca16c…` (2026-07-31) that touches `replay_layer_forward`,
  the BCG allowlist, `cuda_graph_buffer_registry`, or DeepStack
  routing?
* Whether `Qwen3VLForConditionalGeneration` has since been added to
  `multimodal_breakable_cuda_graph_supported_model_archs` (if so,
  the bug is now shipped-live and the monkey-patch is unnecessary).
* Instrumentation robustness against upstream LM class-name churn.
* Fixture placeholder alignment against current Qwen3-VL processor.

**Blocked shared with §9.** All GPU reproductions are gated on the
`595.71.05` driver upgrade (2026-08-04 12:53 UTC) being resolved —
same shared constraint as the L2Norm sub-track.

### 10.2 Proposed reproduction ladder (short form)

Full detail in
[`experiments/qwen3vl_bcg_deepstack_fix/plan.md`](experiments/qwen3vl_bcg_deepstack_fix/plan.md)
§3.

* **R0** — preflight, CPU-only, executable now.
* **R1** — upstream state audit vs pinned SHA, CPU-only, executable now.
* **R2** — reproduce Attempt 03's FAIL on pinned SHA (GPU, blocked).
* **R3** — reproduce on current upstream (GPU, conditional on R1).
* **R4** — re-verify DeepStack is non-empty at LM entry.
* **R5** — direct evidence the captured graph lacks DeepStack `add_`
  kernels (nsys kernel-name diff).
* **R6** — zero-DeepStack ablation signature re-verification.
* **R7** — sensitivity to graph-bucket size (256 – 2048).
* **R8** — request-order isolation with mixed batches.

### 10.3 Proposed fix direction (short form)

Full detail in
[`experiments/qwen3vl_bcg_deepstack_fix/plan.md`](experiments/qwen3vl_bcg_deepstack_fix/plan.md)
§4.

**Recommended: 4.A register-slot-and-copy + 4.B numel-guard as
defence-in-depth.**

* Extend the existing `input_embeds` register-slot pattern (PR
  #30872) to `input_deepstack_embeds`. Slot allocation is
  data-driven — Qwen3.5-style empty DeepStack sees no allocation,
  Qwen3-VL populates the slot at replay.
* `replay_layer_forward` forwards `layer_kwargs` and copies live
  `input_deepstack_embeds` into the slot before `.replay(...)`.
* BCG capture-pass routes the LM with the slot buffer so the
  DeepStack `add_` branch is traced into the captured graph.
* Retain a `numel() > 0` guard around the copy as defence-in-depth.

**Alternatives.**
* **4.B alone** (numel guard + eager fallback) — correct but
  concedes BCG on image requests; kept as a fallback if 4.A's
  buffer-registry lifecycle is too complex for the current window.
* **4.C** (dummy-trace at capture only, no slot) — insufficient
  standalone; the slot is required.

### 10.4 Major risks and uncertainties

* Shared CUDA / driver mismatch blocker with §9.
* Upstream may have partially fixed this since the pin — R1
  characterises.
* Buffer-registry lifecycle for shape-variable slots needs care;
  matches the `input_embeds` slot pattern.
* Fixture / placeholder churn if the Qwen3-VL processor moves —
  caught by R0 before GPU work begins.

### 10.5 Immediate next steps (§10 track)

Executed only after this plan is reviewed and explicit approval is
given for each step. Detail in
[`experiments/qwen3vl_bcg_deepstack_fix/plan.md`](experiments/qwen3vl_bcg_deepstack_fix/plan.md)
§9.

1. **N1** — file this plan; update plan.md §10 (this section).
   Commit `docs(qwen3vl): plan BCG DeepStack fix — reproduction
   ladder and fix design`.
2. **N2** — R0 + R1 (CPU-only): preflight + upstream audit; commit
   the outcome.
3. **N3** — wait for §9's CUDA / driver blocker to lift.
4. **N4** — R2 baseline reproduction on the pinned SHA.
5. **N5** — climb R3 – R8.
6. **N6** — prototype fix 4.A + 4.B in a fresh upstream-`main`
   scratchpad worktree.
7. **N7** — write and stage the upstream PR (regression tests +
   correctness gates); do **not** open the PR until user review.

### 10.6 §10 out of scope

* Editing `/data/sglang-fork` — preserved read-only.
* Editing the pinned scratchpad checkout at `58974ca16c…` —
  preserved read-only. Fix work happens against a fresh clone.
* Filing an upstream PR before R2 and R3 both pass.
* Producing a BCG-vs-eager performance headline for Qwen3-VL under
  the fix — deferred to a follow-up track.
* Rewriting or repurposing the §7 Qwen3.5 close-out or the §4
  Qwen3-VL PCG capture-stream evidence.

## 11. Issue #4 execution plan v3 (drafted 2026-09-03; execution started 2026-09-04)

> **Execution status.** Phases 0 and 1 are complete and the phase-1 gate passed;
> phase 2 is running. The live log, with a status per step, is
> [`experiments/qwen3vl8b/v3_issue4/progress.md`](experiments/qwen3vl8b/v3_issue4/progress.md);
> the frozen stack is
> [`experiments/qwen3vl8b/v3_issue4/manifest.md`](experiments/qwen3vl8b/v3_issue4/manifest.md).
>
> Three places where execution departed from what is written below, each recorded
> with its reasoning at the link above:
>
> 1. **Step 0.2** prescribed pinning a *pre-merge* SHA while #33726 is open. Doing
>    so would have made `A3_bcg` run the very DeepStack replay bug the PR fixes,
>    producing a plausible latency number attached to numerically wrong output.
>    One merged-preview stack is pinned instead, and both worlds are reached by
>    explicit flags: `A0_default` resolves to `breakable` (post-merge default),
>    `A1_disabled` is today's actual behaviour for this arch. 0.2's intent — never
>    straddle the merge inside one bracket — is preserved.
> 2. **The pinned stack carries measurement-only instrumentation** (manifest §7).
>    Without it the `tc_piecewise` eager fallback is unbounded from the log, since
>    `print_warning_once` is `@lru_cache`d; with it the degradation is quantified
>    (measured at 600/7038 graph-eligible calls, 94.2% of calls after onset).
> 3. **Step 1.2 was extended** with cross-*backend* image parity, not just the
>    cross-framework text parity written below. Engagement verification is
>    structurally blind to an arm that uses the right backend and computes the
>    wrong thing; on Qwen3-VL-8B that is exactly the DeepStack replay path.
>
> Risk 4 below (GPU contention) has cleared — GPU 7 freed on 2026-09-04.

> Supersedes the arm design in
> [`image_text_benchmarks/protocol.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/protocol.md)
> §4–§6. The protocol's **goal, dataset, workload shapes, and artifact rules
> still stand**; only the variant matrix, the flag surface, and the gating are
> replaced. Written after the §3.5 upstream audit plus a second audit of the
> multimodal-transport and benchmark-harness surfaces on `upstream/main`
> @ `2da5802bfa`. The design below stands as written; the execution status above
> records where reality departed from it and why.

### 11.1 Why v2 cannot simply be resumed

Four upstream changes each independently break an assumption the v2 runner
encodes. Together they mean the existing bracket would still *run* — and would
silently measure the wrong thing.

| # | Change | Consequence |
|---|---|---|
| A | `SGLANG_USE_CUDA_IPC_TRANSPORT` is **deprecated** → `--mm-feature-transport={cpu,cuda_ipc,cuda_vmm}` (`serving_hook.py:774`). Unset resolves to **`cpu`** for single-node multimodal CUDA. | The protocol's rule "the SGLang image headline **must** set IPC on" now describes a *non-default* configuration. The production default is CPU transport. |
| B | GPU transports reserve `SGLANG_MM_FEATURE_CACHE_MB` (default **1024 MiB**) on the base GPU and, per the flag's own help text, **fall back to CPU transport when the pool is full**. | The IPC arm can degrade to CPU per-tensor mid-run. An "IPC-on" number is not evidence IPC was used. |
| C | The `AssertionError: PCG capture stream is not set` that killed `S2_ipc_pcg` is **gone**. `cuda_piecewise_backend.py:165` now emits `print_warning_once` and **executes that subgraph eagerly**. | The PCG arm no longer crashes — it silently partially-degrades. This is *worse* for measurement than the crash was: a clean-looking result can be mostly eager. |
| D | `sglang.bench_serving` is a deprecation shim; the implementation moved to **`sglang.benchmark.serving`**. Image flags survive unchanged (`--image-count/-resolution/-format/-content`); the generator now builds prompts via `processor.apply_chat_template`. | The runner's invocation path needs updating. The `<\|video_pad\|>` class of bug is structurally addressed by the chat-template path. |

Plus §3.5: **BCG is the default prefill backend on CUDA**, Qwen3-VL is
auto-disabled today, and **PR #33726 flips that on merge**.

**The through-line: every lever in this experiment now has a silent-degradation
path.** v2's design assumed levers either work or crash. That is no longer
true for any of them, so v3's central obligation is *verifying engagement*
rather than trusting flags.

### 11.2 Redesigned variant matrix

Two orthogonal levers, measured against the true production default rather
than against a chosen non-default:

**Transport** (`--mm-feature-transport`): `cpu` (default) · `cuda_ipc`
**Prefill graph** (`--cuda-graph-backend-prefill`): resolved default · `disabled` · `tc_piecewise` · `breakable`

Full cross is 8 cells; that is not affordable, and most cells answer nothing.
The minimum matrix that answers all four questions is **6 SGLang arms + 1
vLLM anchor**:

| id | transport | prefill backend | answers |
|---|---|---|---|
| `A0_default` | unset (resolves `cpu`) | unset (**record what it resolves to**) | the real production baseline |
| `A1_disabled` | unset | `disabled` | true no-prefill-graph floor |
| `A2_tcp` | unset | `tc_piecewise` | does #2's PCG win transfer to images? |
| `A3_bcg` | unset | `breakable` | what the default becomes once #33726 lands |
| `A4_ipc` | `cuda_ipc` | unset | IPC transport benefit, isolated |
| `A5_ipc_best` | `cuda_ipc` | winner of {A2, A3} | do the two levers compose or interfere? |
| `V0_vllm` | — | — | cross-framework anchor |

`A0` is re-run as `A0_repeat` at the end of the bracket for drift (≤5%).
`A5` is chosen *after* A2/A3 report — it is the one adaptive cell, and the
choice must be written down before A5 runs.

**Every arm records its resolved configuration**, not its requested one:
resolved prefill backend, resolved transport, and — for IPC arms — evidence
the pool was actually used and never exhausted. An arm that cannot prove
engagement is reported as `UNVERIFIED` and excluded from comparison, never
quietly folded into the average.

### 11.3 Step-by-step

**Phase 0 — desk work, no GPU.**

0.1 Re-pin the environment manifest: SGLang SHA, vLLM version, model revision,
harness revision, CUDA/driver/torch/`sgl_kernel`, attention backend, and the
exact launch flags per arm. The current container is ~771 commits behind and
needs `LD_PRELOAD` + `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`; **decide
explicitly** whether to rebuild or to run stale-but-controlled, and record the
choice. Every arm shares one stack or the comparison is void.

0.2 Decide the **SGLang SHA policy** against #33726's merge state, and write
the decision into the manifest before any run:
  - *Not merged* → pin a pre-merge SHA. `A0_default` will resolve to
    `disabled`; `A3_bcg` is the preview of the incoming default.
  - *Merged* → pin post-merge. `A0_default` resolves to `breakable`, and
    `A1_disabled` becomes the historical baseline.
  Never straddle the merge inside one bracket.

0.3 Port the runner (`scripts/run_image_text_imgA_fixed.py`, 649 lines) to the
v3 surface: `sglang.benchmark.serving` invocation, `--mm-feature-transport`,
`--cuda-graph-backend-prefill`, the 7-arm list, and the resolved-config
capture in 0.4. This is an edit, not a rewrite — bracket ordering, drift
checks, forbidden-token guards, and artifact layout are all reusable.

0.4 Build the **engagement verifier** — the piece v2 did not have and the one
thing that makes v3 trustworthy. For each arm, parse the server log and fail
the arm loudly on:
  - resolved prefill backend ≠ requested;
  - **any** `PCG capture stream is not set` warning (`tc_piecewise` arms) —
    that arm is partially eager, so its number is not a PCG number;
  - any multimodal-transport CPU-fallback or pool-exhaustion signal
    (`cuda_ipc` arms);
  - any deprecation warning naming a flag we set — proof we are still on an
    old surface.
  Emit a one-line `engagement: VERIFIED|UNVERIFIED (<reason>)` per arm into
  the summary. **No number is quotable without `VERIFIED`.**

**Phase 1 — cheap gates, ~2 GPU-hours.** Serialized, one arm at a time.

1.1 GPU idle check per the standing rules (all 8 GPUs were in use at drafting
time — this plan is *blocked* until one frees).
1.2 Phase-0 correctness parity: SGLang vs vLLM greedy agreement on a fixed
text fixture.
1.3 **vLLM image-anchor smoke** — `--backend sglang-oai-chat` against vLLM's
chat endpoint with data-URI images. Still `UNVERIFIED` from v2 and it gates
every cross-framework claim. If it fails, #4 degrades to an
SGLang-internal study and that must be stated, not glossed.
1.4 Tiny per-arm engagement smoke: ~20 requests on each of the 6 SGLang
configurations purely to run the 0.4 verifier. **Cheapest possible discovery
of a dead arm.** Any arm failing here is fixed or excluded *before* the
expensive bracket.

**Gate**: 1.2 + 1.3 pass, and ≥ `A0/A1/A2 or A3/A4` verify. Otherwise stop and
report — do not run a headline bracket on unverified arms.

**Phase 2 — IMG-A headline, ~1 GPU-day.** Workload unchanged from the
protocol: 1×720p PNG + ~128 text tokens, 128 out, c=1, 400 prompts, 30 warmup,
5 reps. Bracket order:

`A0_default → A1_disabled → A2_tcp → A3_bcg → A4_ipc → A5_ipc_best → V0_vllm → A0_repeat`

Drift gate `|A0_repeat − A0_default| ≤ 5%`; if it fails, the whole bracket is
discarded — no partial rescue.

**Phase 3 — analysis.** Report the four questions **separately**, each against
its own baseline, each with its engagement verdict:
Q1 gap = `A0` vs `V0`; Q2 PCG transfer = `A2` vs `A1`; Q3 IPC benefit =
`A4` vs `A0`; Q4 BCG value = `A3` vs `A1`; composition = `A5` vs
`max(A2, A3)`. A ≥5% delta with 5 reps and CV in band counts; anything less is
reported as "no material difference", never as a trend.

**Phase 4 — only then**, IMG-C (c=16, the Case-C shape analog) to test whether
#2's batched boundary holds for images. **IMG-B and IMG-D stay deferred.**

### 11.4 Disposition of existing artifacts

- `IMG_A_S0_ipc` (5/5 reps, TTFT p50 64.8 ms) — **retired, not reused.** Its
  stack predates changes A–D and its flag surface no longer exists. Keep as
  historical provenance; it must not appear in a v3 comparison.
- `IMG_A_S2_ipc_pcg` crash — **closed as obsolete.** The assertion it hit no
  longer exists (change C). Cite it only as the reason the arm was absent.
- The capture-stream sub-track (§4) — unchanged as history; its fix is
  upstream and its failure mode is now a warning, not a crash.
- Protocol §4–§6 — superseded by §11.2; the rest stands.

### 11.5 Acceptance criteria for closing #4

1. IMG-A reported with a vLLM anchor **and** an IPC ablation **and** a
   prefill-backend sweep, every arm carrying `engagement: VERIFIED`.
2. One frozen manifest covering every arm, with the #33726 merge-state
   decision recorded.
3. The four questions answered separately, PCG and BCG never conflated.
4. Any excluded arm documented with its exact current-upstream failure.
5. A residual gap after transport + graph coverage opens a **new** issue —
   #4 does not expand to chase it.

### 11.6 Risks

1. **Silent degradation on every lever** (A–D) — the dominant risk, and the
   reason Phase 1.4 exists. Mitigation is the verifier, not care.
2. **The default moves under us** when #33726 merges — mitigated by 0.2.
3. **vLLM anchor may not work** — gated at 1.3, before spend.
4. **GPU contention** — all 8 GPUs busy at drafting time; the plan is
   schedule-blocked, not technically blocked.
5. **Stale container** — every arm shares the confound, so internal contrasts
   hold but absolute numbers are not production claims. Say so in the report.
