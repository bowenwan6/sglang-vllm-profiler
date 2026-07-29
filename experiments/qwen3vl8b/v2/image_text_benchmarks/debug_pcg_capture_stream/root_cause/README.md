# Root-cause sub-track for the PCG capture-stream assertion

> Closes out the open question left by
> [`../conclusion.md`](../conclusion.md): *why* does the multimodal forward
> path under `--enforce-piecewise-cuda-graph` trigger a Dynamo recompile
> that lands a piecewise submodule without a capture stream, and what is
> the minimal fix?

Branches:

- **Profiler repo:** `debug/v2-imgA-pcg-capture-stream-fix` (off `main`
  after PR #6 merged the prior debug). Carries runners, results, and
  the `patches/*.patch` copies of each sglang change for
  reproducibility.
- **SGLang source:** the user's fork at
  `git@github.com:bowenwan6/sglang.git`, cloned to `/data/sglang-fork`,
  branch `fix/pcg-vlm-deepstack-warmup` started from upstream commit
  `da802ddca` (the same HEAD `/sgl-workspace/sglang` runs at, so
  patched python files stay binary-compatible with the installed
  `sgl_kernel` extension). All actual code edits live here.

Runs source the fork via `PYTHONPATH=/data/sglang-fork/python` so the
container's installed sglang at `/sgl-workspace/sglang` stays
untouched. Verified: with the prefix prepended,
`import sglang` resolves to
`/data/sglang-fork/python/sglang/__init__.py`.

## 1. Where we start

- The prior debug ([`../conclusion.md`](../conclusion.md)) classified the
  assertion as **VLM image + PCG specifically unsupported on the
  `62c505a196` HEAD**, and recommended (a) file informational upstream
  issue, (b) continue #4 without PCG, (c) no PR. That recommendation
  still stands at the *#4 timeline* level.
- This sub-track narrows the scope: instead of "unsupported case" we ask
  **which Dynamo guard fires the recompile**, and whether the failing
  shape signature can be either captured during warmup or routed
  through a defensive eager fallback. The answer determines which of
  three fix shapes (X / Y / Z below) is appropriate.
- Server was rebuilt 2026-06-28 and the env re-set up from scratch:
  system sglang at `/sgl-workspace/sglang` (HEAD `da802dd`), profiling
  conda env at `/opt/miniconda3/envs/profiling` (vLLM 0.21.0 + torch
  2.11.0+cu130 + flashinfer 0.6.8.post1), Qwen3-VL-8B-Instruct snapshot
  `0c351dd` re-downloaded. Smoke parity confirmed; the PCG assertion
  reproduces deterministically on the rebuilt env on GPU 0.

## 2. Repro shape (rebuilt env)

- Recipe: image 720p, 1 image, c=1, n=32, warmup=30, output_len=128,
  `SGLANG_USE_CUDA_IPC_TRANSPORT=1`, `--enforce-piecewise-cuda-graph`,
  GPU 0, snapshot `0c351dd`.
- Wall-clock: ~60 s server warmup + ~17 s bench warmup + assertion.
- Failure point in server log:

  ```
  AssertionError: PCG capture stream is not set, please check if runtime recompilation happened
    File "/sgl-workspace/sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py", line 171
    File "/sgl-workspace/sglang/python/sglang/srt/models/qwen3_vl.py",   line 1136, in forward
  ```

- Successful warmup requests all log `cuda graph: True`; batch sizes
  vary across warmup (1 → 9 → 21 → 1) before the assertion fires on the
  first prefix-cache-hit prefill (`#new-seq: 1, #new-token: 1,
  #cached-token: 1020`).
- **Upstream surface shift since prior debug:** the VLM PCG auto-disable
  now lives behind a per-model knob
  `ModelConfig.is_multimodal_piecewise_cuda_graph_supported`
  (`server_args.py:3145-3146`). The defensive HIP fallback at
  `cuda_piecewise_backend.py:163-169` still exists; CUDA still asserts.

## 3. Phases R0–R6

Phases R0–R5 are complete on the profiler side. R6 is active and is the
formal fix-value validation gate for the upstream PR.

| Phase | Goal | Actual outcome + evidence |
|---|---|---|
| R0 | Plan + record findings to date | This README + `plan.md` §5a. Done. |
| R1 | Capture exact Dynamo recompile reason via `TORCH_LOGS=recompiles_verbose,dynamic,guards` (env-vars only, no source patch) | Recompile trigger identified: `input_deepstack_embeds is None` guard failure at `qwen3_vl.py:1129`. Multimodal control-flow recompile (not a shape recompile). Excerpts under `results/R1_dynamo_recompile_log/`. |
| R2 | Source-level instrumentation in `cuda_piecewise_backend.py.__call__` to log per-call shapes/dtypes/capture-stream-state | Recompiled `CUDAPiecewiseBackend` instance identified as a distinct Python object from the warmup-frame layer-0 instance; `set_pcg_capture_stream()` fires only inside `capture_session()` and never re-runs at inference. Trace under `results/R2_pcg_call_trace/`. Patch `patches/R2_piecewise_call_logging.patch`. |
| R3 | 2–3 ranked hypotheses + minimal differential experiments | (X) defensive CUDA fallback validated at `cuda_piecewise_backend.py:163-169`; (X) PASSES safety but degrades perf ≈ −38 ms vs PCG-off. Results under `results/R3_fix_feasibility/`; patch `patches/R3_fix_X_cuda_eager_fallback.patch`. |
| R4 | Fix (X) validation at scale + first (Y) attempt | (X) PASS n=32 (R4.A) and n=400 (R4.B). (X) rejected for upstream per §5a of `plan.md` (documented perf regression). Naive (Y) prototype (R4.C) crashed on `forward_context.py:59` assert — bypassed `set_attention_metadata_context()`. Patch `patches/R4_fix_Y_prototype_deepstack_warmup.patch`. |
| R5 | Implement clean (Y) + verify | **Clean Y landed on fork** at branch `fix/pcg-vlm-deepstack-warmup` HEAD `986c89e69` (three-commit stack: `1f19ecd1a` warmup-gate CM → `a4ff0b181` capture-pass hook → `986c89e69` static deepstack buffer). Original R5 image-only TTFT gate **FAILED as stated** (fork-PCG ≈ 102–104 ms vs 64.8 ms). R5.A (n=32) and R5.B (n=400) recorded; **R5.B is pre-static-buffer (fork SHA `a4ff0b181`), historical only.** R5.C correctness audit reports OUTPUTS_DIFFER; static buffer improved but did not eliminate divergence — matched controls not yet run, so residual delta is **not** yet proven to be normal PCG-vs-eager bf16 noise. Results under `results/R5_clean_Y/`; patches `patches/R5_fix_Y_clean_deepstack_warmup_cm.patch`, `R5_fix_Y_clean_capture_pass_hook.patch`, `R5_fix_Y_static_deepstack_buffer.patch`. |
| **R6** | **Fix-value validation for mixed-modality PCG.** Reframe R5's gate around what the fix actually provides: correctness / safety, retained text-only PCG benefit on VLM servers, mixed-modality operational safety, and workload characterization to find any winning cell. Detailed protocol in `plan.md` §5b. See §3.2 below for R6's directory layout and per-phase entry / exit conditions. | Verdict: **PASS / FAIL / R7_REQUIRED**. PR filing gated on PASS. **R6.0** ✅ provenance frozen 2026-07-28 (amendments A1 / A2 dated same, retiring "GPU 6 only" and tightening cleanup to PGID-scoped only) — [`results/R6_fix_value_validation/R6.0_provenance.md`](results/R6_fix_value_validation/R6.0_provenance.md). **R6.1a** ✅ correctness protocol + fixture + runner landed CPU-only 2026-07-28 — [`results/R6_fix_value_validation/R6.1_correctness/protocol.md`](results/R6_fix_value_validation/R6.1_correctness/protocol.md), `scripts/{run_R6_1_correctness.sh, R6_1_client.py, R6_1_verdict.py, R6_setsid_exec.py, monitor_idle_gpu.py}`. **R6.1b attempt 01** ⚠️ INFRA_FAILURE 2026-07-28T10:46 UTC — historical only. **R6.1b attempt 02** ❌ FAIL 2026-07-28. **R6.1b attempt 03** ⇒ CORRECTNESS_PASS (Amendment A). **R6.1b attempt 04** ✅ SAFETY_SUPERIORITY_PASS (Amendment B) 2026-07-28T14:40 UTC on GPU 0 ⇒ **R6.1 = PASS**. **R6.2 executed** 2026-07-29T00:49–02:27 UTC on GPU 0: machine verdict ❌ **FAIL** on the drift metric only (3.050% vs 3.0% cap, by 0.05 pp). Every substantive gate PASSED: fork/stock_pcg ratio 0.9617 (fork 4% faster than stock on text-only!), stock_default 26.86 → stock_pcg 18.35 → fork_pcg 17.65 ms mean TTFT (fork retains ~-34% PCG benefit), all CV ≤ 5.91%, 5/5 × 400/400 every variant, 0 safety anomalies. Drift trace = thermal/queueing noise from intermittent foreign PIDs on GPU 0 during stock_default rep-1 and rep-5. Per user directive: pre-declared thresholds not relaxed post-hoc ⇒ R6.2 FAIL ⇒ R6.3–R6.5 blocked. See [`R6.2_text_only_caseA/attempt_gpu0/verdict.md`](results/R6_fix_value_validation/R6.2_text_only_caseA/attempt_gpu0/verdict.md). |

### 3.0.2 R6.1 Protocol Amendment B (2026-07-28)

Amendment B **replaces Amendment A §2.3** (the 3-prompt image
negative control that produced 3 distinct prefill shapes and
therefore never exercised the historical trigger) with a
**repeated-shape negative control** that uses the exact historical
R1/E2a recipe: 720p image, `--random-input-len 128
--random-range-ratio 1.0`, `--num-prompts 32 --warmup-requests
30`, `--max-concurrency 1`. Identical bench recipe for stock-PCG
and fork-PCG; both under `--enforce-piecewise-cuda-graph`. All
other Amendment A rules (phase markers, cache-matched correctness,
PGID-scoped cleanup) remain in force. See
[`results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md`](results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md).

### 3.0.1 R6.1 Protocol Amendment A (2026-07-28)

The historical `R6.1_correctness/protocol.md` remains the authority
for attempts 01 and 02. **Attempts 03+ execute under
[`R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md`](results/R6_fix_value_validation/R6.1_correctness/protocol_amendment_A_direct_fix_comparison.md)**,
which adds:

- Phase-scoped recompile markers (`SERVER_READY`, `<LEG>_START`,
  `<LEG>_END`); only recompiles inside a leg-interval after
  server-ready may fail the safety gate.
- Cache-matched cold-cache repeats on fresh servers; radix cache
  remains enabled on the primary path (matches production).
- Direct `neg_stock_pcg_image` negative-control leg classified
  as `EXPECTED_STOCK_FAILURE` / `STOCK_NOW_SURVIVES` /
  `UNRELATED_FAILURE`. Expected stock crash isolated to its PGID.
- Three-tier verdict: `SAFETY_SUPERIORITY_PASS` (stock-PCG
  historical failure reproduced ∧ fork-PCG completes cleanly),
  `CORRECTNESS_PASS` (cross-config divergences fit inside
  matched-repeat determinism envelopes), overall PASS = both.
  Fallback: `SAFETY_PASS_CORRECTNESS_AMBIGUOUS` if only
  safety passes. Performance claims (R6.3) require overall PASS.
- Token-level metrics (token IDs, common-prefix tokens,
  normalized Levenshtein) supplement exact-equality.

### 3.1 Fix shape outcome (R4 / R5)

- **(X) defensive CUDA fallback** at `cuda_piecewise_backend.py:163-169` —
  mirror of the existing HIP path. Validated in R3.B / R4.A / R4.B.
  **Rejected for upstream** (silently degrades PCG-on VLM path by ~38 ms
  vs PCG-off; defeats the point of `--enforce-piecewise-cuda-graph`).
  Kept as local fork history + patch `patches/R3_fix_X_cuda_eager_fallback.patch`
  for operators who need a manual band-aid before the real fix lands.
- **(Y) broaden warmup capture** — implemented as clean-Y on fork HEAD
  `986c89e69`. Approach: thread-local `force_warmup_deepstack_embeds`
  gate read by `general_mm_embed_routine` (synthesises zero
  `input_deepstack_embeds` for `use_deepstack` models during PCG
  warmup); mirrored compile-pass hook in `TcPiecewiseCudaGraphBackend`
  and capture-pass hook in `PrefillCudaGraphRunner`; model-attached
  static deepstack buffer so captured cuda-graph replay reads from a
  stable address at inference. Bug-fix layer (crash / assertion /
  eager fallback / inference-time recompile) is eliminated; correctness
  and mixed-modality perf are R6's job.
- **(Z) per-model PCG opt-in** via
  `is_multimodal_piecewise_cuda_graph_supported` — remains Issue #5's
  scope; not implemented in R4 / R5 / R6. R6 evaluates within the
  existing override semantics only.

### 3.2 R6 protocol summary

Full protocol in `plan.md` §5b (do not duplicate here). This section
records the directory layout and phase entry conditions specific to
this sub-track.

#### Directory layout (`results/R6_fix_value_validation/`)

```
R6_fix_value_validation/
├── README.md                          — R6 headline + current verdict pointer
├── R6.0_provenance.md                 — frozen (stock, fork, snapshot, dataset SHA) tuple
├── R6.1_correctness/
│   ├── protocol.md                    — matched controls + fixed image + prompt list
│   ├── raw/                           — .gitignore'd
│   ├── summary.md
│   └── verdict.md                     — PASS / FAIL / AMBIGUOUS
├── R6.2_text_only_caseA/
│   ├── protocol.md                    — 4-way variant matrix + drift bracket
│   ├── R6.2a_stock_default/           — bench summaries only, raw gitignored
│   ├── R6.2b_stock_pcg/
│   ├── R6.2c_fork_pcg/
│   ├── R6.2d_stock_default_repeat/
│   └── summary.md
├── R6.3_image_cost_and_sweep/
│   ├── protocol.md
│   ├── R6.3a_fresh_baseline/          — IMG-A rebaseline on final SHAs
│   ├── R6.3b_workload_sweep/          — matrix cells, one dir per cell
│   ├── R6.3c_mixed_safety/            — interleaved text→image→text log
│   └── summary.md
├── R6.4_analytical_crossover/
│   ├── mix_analysis.py                — means-based p* + bootstrap CI
│   └── mix_table.md
└── R6.5_empirical_mixed/              — optional; created only if R6.1 = PASS
```

Runner scripts land under `scripts/run_R6_*.sh` and are `feat(v2): ...`
commits. Result recording is `test(v2): ...` per experiment. The final
R6 conclusion is a `docs(v2): ...` commit.

#### Phase entry conditions

- **R6.0** — requires §5b provenance freeze table filled in
  `R6.0_provenance.md` with actual dataset SHA-256 (compute at write
  time; no assumed values).
- **R6.1** — requires a fixed real PNG committed to the profiler repo
  (or referenced by absolute path with SHA-256 recorded) with clearly
  interpretable content (subject, background, expected caption). No
  `--image-content random` in the correctness gate.
- **R6.2** — GPU is selected by `scripts/monitor_idle_gpu.py` after
  600 s continuous idle on ONE GPU (0 compute PIDs AND mem ≤ 500 MiB
  AND util ≤ 5 %, polled every 30 s; foreign residual memory treated
  as busy; no GPU reset ever issued). Monitor holds an
  `fcntl.flock`-based lock on `raw/monitor.lock` to prevent concurrent
  copies. Runner remains the single point of GPU use and refuses to
  launch without an explicit ID. Foreign compute PIDs on the selected
  GPU during execution abort R6.1b with exit 71 without signalling
  the foreign process. See R6.0 Amendment A1 for authorization.
- **R6.3a** — must be a fresh run on `da802ddca` / `986c89e69`;
  symlinking to `results/R5_clean_Y/R5B_n400_stretch/` is explicitly
  disallowed (wrong fork SHA).
- **R6.3c** — mixed-modality safety subtest is **mandatory**, not
  optional; must be recorded even if perf legs are deferred.
- **R6.4** — must operate on means, not p50; bootstrap CI from
  rep-level data; historical estimates are illustrative only.
- **R6.5** — gated on R6.1 = PASS; sweep ≥ 3 mix ratios; identical
  fixed request order for stock-default vs fork-PCG.

## 4. Out of scope here

- v1 Phase 0–5 artifacts. Never touched.
- IMG-A non-PCG resume (`S0_ipc_repeat → V0_vllm → S0_noipc`). That is
  the parent `fixed_generator_plan.md` work and stays queued.
- Changes to `--enforce-piecewise-cuda-graph` defaults or the
  `is_multimodal_piecewise_cuda_graph_supported` table without explicit
  user approval. That is Issue #5's scope; R6 evaluates within the
  existing override semantics.
- Filing the upstream issue / PR. R6 gates it; user triggers filing.
- Retroactively rewriting `results/R5_clean_Y/R5C_correctness_audit/audit_report.md`.
  R5.C stands as recorded; R6.1 supersedes it as the correctness
  authority. A local uncommitted edit to that file is preserved under
  user control until user directs otherwise.

## 5. Artifact rules

- All sglang source modifications kept as revertable `.patch` files
  under `patches/`. The actual `/sgl-workspace/sglang` working tree
  must be clean between commits in this repo. All actual fork changes
  land in `/data/sglang-fork` branch `fix/pcg-vlm-deepstack-warmup`
  and are committed / pushed there independently — never mixed into a
  profiler commit.
- Raw per-run server logs go under `results/<R-id>/raw/` and are
  **NOT committed** unless explicitly approved. Aggregate summaries
  + trimmed excerpts are committed.
- Bench JSONLs are not committed; trimmed summaries only.
- `.claude/settings.local.json` is never staged.
- No empty `results/<R-id>/` directories are pre-created; each
  sub-phase directory is created only when its first artifact is
  written.

## 6. Commit cadence

Per `CLAUDE.md`: Conventional Commits `type(scope): action target/context`,
no `Co-Authored-By` trailers, no mention of Claude / Anthropic / AI in
any subject / body / scope / trailer. Prefix conventions for this
sub-track:

- `docs(v2): ...` — plan and status revisions, final R6 conclusion.
- `feat(v2): ...` — new runners / generators / analysis tooling.
- `test(v2): ...` — recorded experiment results, including recorded
  failures (a failed R6.1 or R6.2 result is a `test` commit, not a
  `fix`).
- `perf(v2): ...` — only when the commit itself is a perf
  implementation change; never for merely reporting perf numbers.
- `fix(v2): ...` — profiler repo bugfixes.

Every runner spec, every recorded experiment, and the final R6
conclusion are each an independent focused commit, pushed
immediately. Any SGLang fork changes commit + push in the fork
repository only.
