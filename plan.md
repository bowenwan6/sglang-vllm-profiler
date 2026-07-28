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
| **4** | **Qwen3-VL image+text + CUDA IPC** | **P1 — PCG crash root-caused; clean fix on fork; R6 fix-value validation pending** | Image+text behavior + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`; separate from text-only conclusions | Generator `<\|video_pad\|>` bug merged upstream as `07f326c184` (#26864). PCG capture-stream crash root-caused (§5a) → Dynamo recompile of `Qwen3LLMModel.forward` on `input_deepstack_embeds is None` guard failure at first image request; recompiled `CUDAPiecewiseBackend` instance has no capture stream. Clean (Y) fix implemented on fork `bowenwan6/sglang` branch `fix/pcg-vlm-deepstack-warmup` HEAD `986c89e69`: thread-local warmup gate synthesizes zero deepstack embeds so Dynamo traces both branches at warmup + model-attached static deepstack buffer for capture/replay address stability. **R5 outcome:** crash / capture-stream assertion / inference-time recompile all eliminated on fork; original R5 image-only TTFT gate (p50 clearly < 64.8 ms) **FAILED as stated** — fork-PCG image+text ≈ 103 ms vs default ≈ 65 ms; R5.C correctness audit shows outputs diverge but no matched control has yet proven this is normal bf16 PCG-vs-eager noise rather than residual corruption. Next: **R6 fix-value validation** (see §5b) reframes acceptance around mixed-modality operational safety + retained text-only PCG benefit + workload characterization to locate any cell where fork-PCG > default; upstream PR gated on R6 PASS. Non-PCG IMG-A resume (`S0_ipc_repeat → V0_vllm → S0_noipc`) remains queued. |
| 3 | Qwen3.5 VL-model profiling | P1 | Same clean methodology on Qwen3.5; does the PCG finding transfer? | next candidate (parallel/after #2; transfer check) |
| 5 | Selective/default-on PCG PR plan | P2 | Minimum safe exception in VLM auto-disable + guards + fallback | planned (needs #4) |

## 5. Immediate Next Step

**Issue #2 is COMPLETE** (clean run, GPU 1, 0 failures; results under
`experiments/qwen3vl8b/v2/caseAC_rebaseline/results/`).

**Issue #4** — active on branch `debug/v2-imgA-pcg-capture-stream-fix`. The PCG
capture-stream crash has been root-caused (§5a) and fixed on the user fork
(`bowenwan6/sglang` branch `fix/pcg-vlm-deepstack-warmup` HEAD `986c89e69`).
The original R5 image-only performance gate FAILED as stated; the hypothesis
has been revised and formal fix-value validation is planned under **§5b R6**
(reframes acceptance around mixed-modality operational safety + retained
text-only PCG benefit on VLM servers + workload characterization sweep for
any cell where fork-PCG > default). **Upstream PR is gated on R6 PASS** —
not filed until then.

The following context (historical narrative on the pre-fix `62c505a196` HEAD)
is preserved for provenance:

**Issue #4 was PARTIAL — generator unblocked, PCG path blocked by upstream
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
at `/sgl-workspace/sglang` HEAD `da802dd`, profiling conda env at
`/opt/miniconda3/envs/profiling`, Qwen3-VL-8B-Instruct snapshot `0c351dd`
re-downloaded). Source-of-truth for sglang edits: user fork
`git@github.com:bowenwan6/sglang.git` cloned to `/data/sglang-fork` on branch
`fix/pcg-vlm-deepstack-warmup`, started from upstream `da802ddca` so patched
python files stay binary-compatible with the installed `sgl_kernel`. Runs source
the fork via `PYTHONPATH=/data/sglang-fork/python`.

Sub-track artifacts and full per-phase writeups under
[`v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/README.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/README.md).
All sglang source modifications are kept as revertable `.patch` files under
`root_cause/patches/`. Raw per-run server / bench logs stay under
`results/<R-id>/raw/` and are **not committed** unless explicitly approved.

### What we did (R1 → R4)

| Phase | What was run | Outcome |
|---|---|---|
| **R1** | Env-var-only Dynamo verbose tracing (`TORCH_LOGS=recompiles_verbose,dynamic,guards,graph_breaks`, `TORCHDYNAMO_VERBOSE=1`) on the E2a recipe (image 720p, c=1, n=32, warmup=30, PCG on, IPC on, GPU 0). | **Recompile trigger identified.** Four recompiles of `Qwen3LLMModel.forward` before the assertion. The decisive one (`[0/3]`) fail-reason: `input_deepstack_embeds is None` guard failure at `qwen3_vl.py:1129`. Failing token counts (80, 1024) are inside the captured 1..8192 range → **not** a shape recompile, it's a multimodal control-flow recompile. |
| **R2** | Source-level patch to `cuda_piecewise_backend.__call__` (`SGLANG_DEBUG_PCG_CALL_TRACE=1`) emitting per-call instance id / layer idx / warmup state / runtime_shape / capture-stream state. Patch saved as `R2_piecewise_call_logging.patch`. | **Mechanism confirmed.** The asserting `CUDAPiecewiseBackend` instance (`id=0x702f42eba060`, `sym_shape_indices=[1,4,9,10]`) is a **distinct Python object** from the warmup-frame layer-0 instance (which had `sym_shape_indices=[1,8]`). The recompiled instance never gets a capture stream because `set_pcg_capture_stream()` is only set inside `PiecewiseCudaGraphRunner.capture_session()`, which never re-runs at inference time. |
| **R3.A** | Source read of the warmup driver, dummy-batch builder, multimodal embed routine, and Qwen3-VL `forward`. | **Architecture mapped.** `_run_compile_pass` → `_run_dummy_forward(num_tokens)` → `capture_prepare` which **never sets `mm_inputs`** → `general_mm_embed_routine` gates `input_deepstack_embeds` on `contains_mm_inputs()` → therefore Dynamo only ever sees `input_deepstack_embeds = None` during warmup. |
| **R3.B** | Patch (X) on fork: extend the existing HIP eager fallback in `cuda_piecewise_backend.py:163` to CUDA (drop the `_is_hip and` guard so missing-stream falls back to `entry.runnable` instead of asserting). ~9-line change. Re-fired E2a. | **(X) PASS for safety.** 32 / 32 requests, no `AssertionError`, single `print_warning_once` fallback warning. TTFT median **103.05 ms**. |
| **R4.A** | Re-ran E2a with `SGLANG_DEBUG_PCG_CALL_TRACE` unset (production shape). | **(X) PASS without diagnostic gate.** TTFT median 106.25 ms; consistent with R3.B. |
| **R4.B** | Stretched (X) to the original Stage 4.2 IMG_A_S2_ipc_pcg recipe (n=400, warmup=30, single rep). | **(X) PASS at scale.** 400 / 400 requests, TTFT median **104.62 ms**, no regression. |
| **R4.C** | Naive (Y) prototype on fork: added `Qwen3VLForConditionalGeneration.pcg_warmup_multimodal_branch()` synthesizing `torch.zeros([num_tokens, hidden_size × num_deepstack_embeddings])` and calling `self.model(...)` directly; called from `_run_compile_pass` as a second per-shape loop. | **(Y) FAILURE (documented, instructive).** Server crashes on the first MM warmup call inside the torch.compile-traced model forward: `forward_context.py:59 assert _current is not None` fires because we bypassed `set_attention_metadata_context()`. Dynamo refuses to graph-break (`fullgraph=True`) → compile aborts. |

### Findings carried forward

1. **The bug is the multimodal control-flow recompile.** Dynamo specialises `Qwen3LLMModel.forward` on `input_deepstack_embeds is None`; PCG warmup only ever feeds the `None` branch, so the first real image request forces a recompile of a brand-new fx graph whose piecewise submodules have no capture stream attached.
2. **The defensive assertion at `cuda_piecewise_backend.py:171` is structurally unreachable from a recompiled instance.** `set_pcg_capture_stream()` is set only inside `capture_session()`; that session ends with server startup and the stream is `None` forever after.
3. **Cross-comparison vs the existing non-PCG image baseline** (Stage 4.2 IMG_A_S0_ipc, 5×400 reps): TTFT p50 ≈ **64.8 ms** without PCG. With (X) PCG-on + eager fallback, TTFT p50 is **~104 ms** at n=32 / n=400 — i.e. PCG-on with the band-aid is **slower** than PCG-off, because the recompiled multimodal frame loses cudagraph replay benefit on every layer.

### Why (X) is rejected for upstream

The (X) eager-fallback patch defeats the point of `--enforce-piecewise-cuda-graph`
on the multimodal path: it turns a hard crash into a silent ~38 ms TTFT
regression vs the no-PCG baseline. Merging (X) upstream would mean shipping a
known performance-negative path while still claiming PCG support for VLMs.

(X) stays **in our local fork history** as a documented safety net and is the
patch we'd recommend operators apply manually if they need to keep
`--enforce-piecewise-cuda-graph` running before the real fix lands. **It does
not get an upstream PR.**

### Architectural lesson from R4.C

The naive (Y) prototype bypassed `set_attention_metadata_context()`, which the
attention backend asserts is in scope (`forward_context.py:59`). Any working
(Y) **must reuse the same forward-context-wrapping path that the regular
`_run_dummy_forward` uses** — we cannot call `self.model(...)` directly from a
new entry point.

The cleanest viable shape, given R3.A's source read of `general_mm_embed_routine`:

- A **thread-local "force-multimodal-warmup" flag** read inside
  `general_mm_embed_routine`. When set **and** `use_deepstack` is truthy for
  the active modality, the routine synthesizes
  `kwargs["input_deepstack_embeds"] = torch.zeros([num_tokens, hidden_size ×
  num_deepstack_embeddings], dtype, device)` instead of (or alongside) the
  real mm path, then routes through `language_model.forward(...)` as usual.
- The warmup driver in `_run_compile_pass` enters this flag's `with` block,
  then calls the existing `cuda_graph_runner._run_dummy_forward(num_tokens)`
  — picking up `set_attention_metadata_context()` for free — once per
  `capture_num_tokens` shape, alongside the existing text-only sweep.
- Result: Dynamo traces both the `input_deepstack_embeds is None` branch
  (text-only sweep) **and** the non-None branch (MM sweep) during warmup;
  both branches' `CUDAPiecewiseBackend` instances get capture streams during
  the subsequent `capture_session`; the first real image request hits an
  already-captured graph and no recompile fires.

### New R5 mandate

R5 stops being "draft upstream handoff" and becomes **implement clean (Y) and
verify** so we can submit a performance-positive PR. Specifically:

1. **Design + implement a clean (Y)** on `/data/sglang-fork` branch
   `fix/pcg-vlm-deepstack-warmup` (a new commit on top of the existing X +
   broken-Y history). Use the thread-local flag in `general_mm_embed_routine`
   approach above, or document why a different shape works better and adopt
   it.
2. **Local verification** on the same E2a recipe (image 720p, c=1, n=32,
   warmup=30, PCG on, IPC on, GPU 0):
   - Server starts; the MM warmup loop ("Compiling MM num tokens") runs to
     completion without crashing.
   - No `AssertionError`, no `Falling back to eager execution` warning, no
     Dynamo recompile of `qwen3_vl.forward` observed at inference time
     (re-enable `TORCH_LOGS=recompiles_verbose` for one confirmation run).
   - All bench requests succeed.
3. **Stretch verification** at the Stage 4.2 IMG_A_S2_ipc_pcg shape (n=400,
   warmup=30, single rep) to confirm no regression at scale and to capture
   headline-quality TTFT numbers.
4. **Performance acceptance gate.** Clean (Y) must hit TTFT clearly **below**
   the IMG_A_S0_ipc PCG-off baseline (p50 ≈ 64.8 ms) on Case-A-like image+text
   workloads — otherwise the multimodal frame's cudagraphs aren't actually
   being captured / replayed and we have to keep iterating. Order of magnitude
   target: TTFT p50 within shouting distance of, or better than, the text-only
   Case A `--enforce-piecewise-cuda-graph` result (14.04 ms from v2 #2),
   bearing in mind image+text adds vision-tower work that is not PCG-covered
   so a strict equality is not expected.
5. **Then and only then** prepare the upstream PR description. R5 still does
   not auto-file — filing remains a user-triggered step — but the PR is
   gated on (Y) PASS, not on (X).

Out of scope here:

- v1 Phase 0–5 artifacts (never touched).
- IMG-A non-PCG resume (`S0_ipc_repeat → V0_vllm → S0_noipc`) — remains queued
  under `fixed_generator_plan.md`; orthogonal to this sub-track.
- Changes to `--enforce-piecewise-cuda-graph` defaults or the
  `is_multimodal_piecewise_cuda_graph_supported` table — that is Issue #5's
  scope. The clean (Y) lands inside the existing override semantics; it does
  not flip defaults.
- Submitting (X) upstream — explicitly rejected; kept as local fork history
  only.

### R5 actual outcome (recorded 2026-07-28)

- **Clean (Y) landed on fork** at branch `fix/pcg-vlm-deepstack-warmup` HEAD
  `986c89e69` (`fix(pcg): use stable model-attached deepstack buffer for
  capture+replay`), built on `1f19ecd1a` (warmup context manager) +
  `a4ff0b181` (capture-pass hook). At inference under `--enforce-piecewise-
  cuda-graph`: no `AssertionError`, no `Falling back to eager execution`
  warning, no Dynamo recompile of `qwen3_vl.forward` (R5.A n=32, R5.B n=400).
- **Original R5 performance gate FAILED as stated.** Gate 4 required TTFT p50
  "clearly below" the IMG_A_S0_ipc PCG-off baseline of 64.8 ms. Measured
  fork-PCG image+text TTFT p50 ≈ 102–104 ms (R5.A / R5.B) — well *above* the
  baseline. Not retroactively re-labelled as PASS. The gate itself was
  mis-framed: image+text prefill on Qwen3-VL is vision-tower-dominated
  (~40 ms eager either way), leaving too small a PCG-covered LM fraction
  for graph-launch savings to overcome capture/launch overhead — this is a
  workload property, not a fix bug, but it must be handled by re-framing
  R6's acceptance, not by silently rewriting R5's.
- **Correctness NOT formally closed.** R5.C audit reports OUTPUTS_DIFFER
  between fork-PCG-on and fork-default on 2 prompts (first differing offset
  4 / 126). The static-buffer fix reduced Prompt 1's first-diff offset from
  41 → 126 characters, which is evidence that some address-stability
  corruption existed and was mitigated, but does **not** prove the residual
  divergence is normal bf16 PCG-vs-eager noise (H2). No matched control
  (e.g. fork-default vs stock-default on the same image, or fork-eager vs
  fork-PCG on a non-multimodal workload) has yet been run to attribute the
  residual delta. Prior "H2 residual noise" language in this file and in
  the audit report was a hypothesis, not a measurement.
- **R5.B provenance caveat.** R5.B (n=400) was recorded 2026-06-30 15:37 UTC
  on fork SHA `a4ff0b181` (capture-pass hook), **21 min before** the
  static-buffer fix at `986c89e69` (15:58 UTC). R5.B is a historical
  datapoint only; it is **not** a valid comparator for the final fork SHA
  and must not be reused as R6's fork-PCG headline.
- **Hypothesis revised** for R6: the fix's value proposition is **not**
  "faster image+text prefill on the IMG-A recipe" but **"safe use of
  `--enforce-piecewise-cuda-graph` on a Qwen3-VL server that serves mixed
  text-only and image traffic, preserving the text-only PCG benefit
  measured on non-VLM Case A."** R6 (§5b) formally validates this reframe.

## 5b. R6 — Fix-value validation for mixed-modality PCG (active)

> Reframe of R5's acceptance around what the fix actually provides. Does
> **not** presume the fix passes — R6 must be able to conclude **PASS**,
> **FAIL**, or **R7_REQUIRED**. Filing the upstream PR is gated on R6 PASS.

### R6 goal

Three independent claims, each with its own gate:

1. **Correctness / safety.** Fork clean-Y is correctness-preserving on
   mixed-modality workloads, or the residual output divergence is
   *demonstrably* attributable to normal PCG-vs-eager bf16 noise (with a
   matched control) rather than silent corruption.
2. **Retained PCG benefit.** On text-only Case A running on a Qwen3-VL
   server, fork-PCG delivers the same mean TTFT as stock-PCG (fix does not
   regress the text path), and both are clearly below stock-default. The
   fix does **not** *create* the text-only PCG speedup — that already
   exists on stock for pure text-only traffic — it *preserves* it on a
   server that must also accept image traffic without crashing.
3. **Mixed-modality operational safety.** Interleaved text → image → text
   traffic on the same fork-PCG server produces 0 request failures, 0
   capture-stream assertions, 0 eager-fallback warnings, 0 inference-time
   Dynamo recompiles of `qwen3_vl.forward`. This is the *only* claim on
   which stock has no equivalent (stock crashes on the first image under
   `--enforce-piecewise-cuda-graph`).

### R6 entry gate: provenance freeze

| Item | Value | Source |
|---|---|---|
| Stock SGLang SHA | `da802ddcafe55e25b3e1db86b1e0444afc3e05bc` | `/sgl-workspace/sglang` HEAD (rebuilt 2026-06-28) |
| Final fork SHA | `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` (branch base = stock HEAD → binary-compatible with installed `sgl_kernel`) | `/data/sglang-fork` branch `fix/pcg-vlm-deepstack-warmup` |
| Model snapshot | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | HF `Qwen/Qwen3-VL-8B-Instruct` |
| System python (server) | `python 3.12.3` + torch 2.11.0+cu130 + flashinfer 0.6.12 + sgl_kernel 0.4.4 | `/usr/bin/python3`, installed sglang path |
| Profiling env (bench client, vLLM anchor) | `/opt/miniconda3/envs/profiling` — torch 2.11.0+cu130, vLLM 0.21.0 | rebuild 2026-06-28 |
| Text dataset | `datasets/qwen3vl8b/caseA_short.jsonl` (600 prompts, SHA-256 `fab4917772e087447d7c33d53ada63340b126088c1f195f118b9488d5f5b619e`) | v2 #2 provenance |
| Correctness image | Fixed real PNG chosen and recorded in R6.1 protocol (no `--image-content random`) | new for R6 |
| GPU | R6.1b attempt 02: **GPU 2 only** (user-authorized after attempt 01 INFRA_FAILURE). Runner enforces via `R6_GPU_ID=2`; may not silently substitute. R6.0 amendment A1 dynamic-selection rule (via `monitor_idle_gpu.py`) still applies to any future attempt where the caller does not pass a fixed GPU. Foreign compute PIDs abort the run with exit 71 (foreign process never signalled). | must not silently relocate; runner still requires explicit ID |
| Runtime libcuda (Amendment A3, 2026-07-28) | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` (91,501,576 bytes; driver 595.71.05). Fixes loader precedence where `cuda-compat-13-0` was putting `libcuda.so.580.82.07` first. Torch 2.11.0+cu130 initializes CUDA cleanly against the host lib. Never `LD_LIBRARY_PATH` under `/usr/local/cuda-*/compat`. | preflight `scripts/R6_preflight_libcuda.py` refuses to proceed unless the loaded libcuda is exactly the pinned host path |

Full frozen provenance table, verification commands, and historical /
reference numbers live in
[`.../results/R6_fix_value_validation/R6.0_provenance.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.0_provenance.md).

Numbers from any earlier HEAD — v2 #2 Case A `21.94 / 14.04 ms` on
`0c8049d9b`; IMG_A_S0_ipc `64.8 ms` on `62c505a196`; R5.A/B/C on fork SHAs
`1f19ecd1a` / `a4ff0b181` (pre-static-buffer) — are **historical reference
only** and do **not** carry forward as R6 baselines. R6 measures everything
fresh on the frozen (stock, fork) SHA pair.

### R6.1 Protocol Amendment A (2026-07-28, authoritative for attempts 03+)

R6.1 attempts 01 / 02 exposed two protocol-level gaps that R6.1
Protocol Amendment A closes for attempts 03+:

1. **Phase-scoped recompile markers** — startup/warmup recompiles
   are reported and never fail the safety gate; only recompiles
   between `SERVER_READY` and the last `LEG_END` inside a
   `[LEG_START, LEG_END]` interval may fail the safety gate.
2. **Cache-matched correctness controls** — matched cold-cache
   repeats on fresh servers replace the same-server sequential
   pattern that produced the attempt-02 cache-state confound.
   Radix caching stays enabled for the primary path;
   `--disable-radix-cache` is a diagnostic ablation only.
3. **Direct stock-PCG image negative control** — a fresh
   stock-PCG server serves the exact fixture / prompts of leg
   b; classified as `EXPECTED_STOCK_FAILURE`,
   `STOCK_NOW_SURVIVES`, or `UNRELATED_FAILURE`. An expected
   stock crash is isolated to its PGID.
4. **Three-tier verdict**: `SAFETY_SUPERIORITY_PASS` (stock-PCG
   reproduces the historical failure AND fork-PCG completes the
   same sequence cleanly), `CORRECTNESS_PASS` (all cross-config
   divergences fit inside matched-repeat determinism envelopes),
   overall R6.1 PASS = both, `SAFETY_PASS_CORRECTNESS_AMBIGUOUS`
   if only safety passes, `FAIL` if safety fails. Performance
   claims (R6.3) require overall PASS.
5. **Token-level metrics** (token IDs, common-prefix tokens,
   normalized token Levenshtein, envelope-based inside/outside
   test) supplement exact-equality; semantic evaluation is
   supplementary only, not verdict-authoritative.

Full text: [`results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md).
Attempts 01 / 02 stand as recorded under the original protocol;
attempt 03 executes under Amendment A.

### R6 phases

| Phase | Purpose | Exit / verdict |
|---|---|---|
| **R6.0** | Provenance freeze + protocol writeup; dataset SHA recorded; commit + push. | ✅ COMPLETE 2026-07-28 — `results/R6_fix_value_validation/{README.md, R6.0_provenance.md}` committed. |
| **R6.1a** | Correctness protocol + fixture + runner (CPU-only preparation). Predefine verdict rules; deterministic 1280×720 PNG fixture (three vertical color bands, muted RGB) with SHA-256 pinned; 3 image prompts + 3 text-only prompts + interleaved sequence; refuses-without-GPU-ID runner + Python client + verdict computation. | ✅ COMPLETE 2026-07-28 — `results/R6_fix_value_validation/R6.1_correctness/{protocol.md, fixtures/*}`, `scripts/{run_R6_1_correctness.sh, R6_1_client.py, R6_1_verdict.py}`. CPU-only validation: bash syntax, `python3 -m py_compile` for all `.py`, fixture regeneration bit-identical, runner refuses `--help`-style invocations without approved GPU (exit 64). |
| **R6.1b attempt 01** (historical) | Attempted 2026-07-28T10:46 UTC on GPU 1 (monitor-selected after 629 s idle). Runner exited 2 on the first server; no leg ran. **Corrected root cause per R6.0 Amendment A3**: `cuda-compat-13-0`'s `libcuda.so.580.82.07` took loader precedence over the host lib `libcuda.so.595.71.05`; torch 2.11.0+cu130 fails `cudaGetDeviceCount()` against the compat lib. Infrastructure blocker; not a clean-Y correctness failure. Committed as `test(v2): record R6 correctness gate` (`703ff69`). Preserved verbatim under `R6.1_correctness/{verdict.md, verdict.json, raw/}` — not rewritten. | historical only; attempt 02 supersedes for the correctness verdict |
| **R6.1b attempt 02** | Executed 2026-07-28T12:39–12:43 UTC on **GPU 0**. Artifacts under `R6.1_correctness/attempt_02_host_libcuda_595_gpu0/`. Full runner executed all 4 servers × 9 legs; every request saw HTTP 200. | ❌ **FAIL** by pre-declared rules; forensic analysis (see [`analysis.md`](experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/analysis.md)) shows the FAIL is on the pre-declared bit-identical axes but the divergences are cache-state artefacts, not fix-induced corruption. Confirmed facts: (i) `a1_vs_c` (fork-default cold vs stock-default cold) is bit-identical on all 3 image prompts — fix is a genuine no-op when PCG is off; (ii) same-config warm-vs-cold within one server (`a1_vs_a2`, tok_lev=4) has *larger* variance than cross-config cold-vs-cold (`a1_vs_b`, tok_lev=2); (iii) 0 crashes / 0 assertions / 0 fallbacks / 0 request failures / 0 post-server-ready recompiles. Open hypothesis: `d_vs_dp` (stock-PCG cold vs fork-PCG warm-after-b) diverges — attribution requires matched cache-state repeats. Attempt 02 also **never sent an image to stock-PCG**, so the historical first-image failure is neither reproduced nor ruled out on `da802dd`. R6.1 protocol amendment follows in the next commit; R6.1 attempt 03 will run under the refined protocol. |
| **R6.2** | Text-only Case A on Qwen3-VL server. Same recipe as v2 #2 Case A: `caseA_short.jsonl`, 128→128, c=1, n=400, warmup=30, seed=1, 5 reps. Variants: **(2a)** stock-default, **(2b)** stock-PCG (`--enforce-piecewise-cuda-graph`), **(2c)** fork-PCG, **(2d)** stock-default_repeat (drift bracket). Pre-declared thresholds: fork-PCG mean TTFT ≤ stock-PCG mean TTFT × 1.05 AND CV ≤ 6% AND drift bracket 2a↔2d ≤ 3%. | Datapoint = *retained* PCG benefit (2a → 2c). Do not present as fix-created speedup. |
| **R6.3** | Fresh image cost + workload characterization on final fork SHA. **R6.3a** — rebaseline IMG-A `S0_ipc` and fork-PCG at the R5.B recipe (720p, 128 text, c=1, n=400) on `da802ddca` / `986c89e69`. Do **not** reuse or symlink R5.B (wrong SHA). 3 reps each; report mean TTFT + CV. **R6.3b** — workload sweep to locate any cell where fork-PCG mean TTFT ≤ stock-default mean TTFT: matrix over text tokens ∈ {128, 512, 2048}, image resolution ∈ {224p, 720p}, concurrency ∈ {1, 4}, single rep per cell (n=100). Every cell reported, positive and negative. **R6.3c (mandatory)** — mixed-modality safety subtest: interleaved text → image → text → image on one fork-PCG server, ≥ 50 requests each modality, log recompiles + assertions + fallbacks. This is *not* optional and does not require perf conclusions. | R6.3a = cost datapoint on final SHA. R6.3b = winning-cell identification (if any). R6.3c = binary operational-safety verdict. |
| **R6.4** | Analytical crossover on **means** (not p50 — p50 is not a linear operator). Given `G = mean_text_off − mean_text_on > 0` and `C = mean_image_on − mean_image_off > 0`, `p* = C / (G + C)`. Bootstrap CI on `p*` from rep-level data (R6.2 gives 5 reps × 4 variants; R6.3a gives 3 reps × 2 variants). Table at p ∈ {0.5, 0.7, 0.8, p*, 0.9, 0.95, 1.0}. | Reported alongside R6.3b sweep. The analytical `p*` is *not* an empirical crossover — it must not be described as a measured mixed-workload TTFT crossover. |
| **R6.5** | Optional empirical mixed-workload perf validation. Only if R6.1 = PASS and R6.2/6.3/6.4 are all clean. Sweep ≥ 3 fixed mix ratios (below `p*`, at `p*`, above `p*`) with the identical fixed request order for stock-default and fork-PCG. Single 80/20 run is not accepted. | Gates strength of the empirical mixed claim; not required for R6 PASS. |

### R6 verdict framework

- **PASS** ← R6.1 = PASS AND R6.2 within thresholds AND R6.3c = 0 failures /
  0 assertions / 0 recompiles / 0 fallbacks AND (R6.3b found ≥ 1 winning
  cell OR R6.4 `p*` is in operator-realistic range ≤ 0.95).
- **FAIL** ← R6.1 = FAIL, or R6.2 fork-PCG regresses stock-PCG beyond
  threshold, or R6.3c surfaces any failure / assertion / recompile /
  fallback.
- **R7_REQUIRED** ← R6.1 = AMBIGUOUS, or R6.3b sweep + R6.4 shows no
  operator-realistic winning workload and `p*` > 0.95 → upstream PR
  framing must be redesigned (correctness-only, no perf headline) before
  submission.

### R6 out of scope

- Changing `is_multimodal_piecewise_cuda_graph_supported` defaults — Issue
  #5 owns that decision; R6 evaluates *within* the existing override
  semantics.
- Filing the upstream PR itself — R6 gates it; user triggers the actual
  filing.
- Retroactively rewriting the R5.C `audit_report.md`. R5.C stands as
  written; R6.1 supersedes it as the correctness authority. The current
  uncommitted local edit to `R5C_correctness_audit/audit_report.md` is
  preserved as-is under user control until user directs otherwise.

### R6 commit cadence (applies to all R6 work)

Per `CLAUDE.md` + Conventional Commits:

- `docs(v2): ...` — plan and status revisions; final R6 conclusion.
- `feat(v2): ...` — new runners, generators, analysis tooling.
- `test(v2): ...` — recorded experiment results, **including recorded
  failures** (a failed R6.1 or R6.2 result is a test commit, not a fix).
- `perf(v2): ...` — only when the commit itself is a perf implementation
  change; never for merely reporting perf numbers.
- `fix(v2): ...` — profiler repo bugfixes.

Every runner spec, every recorded experiment, and the final R6 conclusion
are each an independent focused commit, pushed immediately. Any SGLang
fork changes commit + push in `/data/sglang-fork` only; never mix fork
edits into profiler commits.

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
