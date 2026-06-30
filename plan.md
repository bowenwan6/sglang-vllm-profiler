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
