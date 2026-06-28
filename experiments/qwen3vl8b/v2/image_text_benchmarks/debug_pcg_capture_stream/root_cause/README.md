# Root-cause sub-track for the PCG capture-stream assertion

> Closes out the open question left by
> [`../conclusion.md`](../conclusion.md): *why* does the multimodal forward
> path under `--enforce-piecewise-cuda-graph` trigger a Dynamo recompile
> that lands a piecewise submodule without a capture stream, and what is
> the minimal fix?

Branch: `debug/v2-imgA-pcg-capture-stream-fix` (off `main` after PR #6
merged the prior debug). All sglang source patches are committed here as
`patches/*.patch` and applied / reverted around runs — no fork of sglang,
no edits left in `/sgl-workspace/sglang` between commits.

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

## 3. Phases R0–R5

Phase R0 (this commit): record plan + rebuilt-env repro context. No
sglang source edits. No bench runs beyond Step E of the env rebuild
(already reported in conversation; raw log lives in scratchpad only,
not committed).

| Phase | Goal | Exit condition |
|---|---|---|
| R0 | Plan + record findings to date | This README + `plan.md` §5a updated, committed. |
| R1 | Capture exact Dynamo recompile reason via `TORCH_LOGS=recompiles_verbose,dynamic,guards` (env-vars only, no source patch) | Raw recompile-reason excerpt + analysis under `results/R1_dynamo_recompile_log/`. |
| R2 | Source-level instrumentation in `cuda_piecewise_backend.py.__call__` to log per-call shapes/dtypes/capture-stream-state | Patch saved at `patches/R2_piecewise_call_logging.patch`, trace at `results/R2_pcg_call_trace/`, source reverted. |
| R3 | 2–3 ranked hypotheses + minimal differential experiments (one axis flipped per experiment) | Per-experiment result under `results/R3_<id>/`. |
| R4 | Fix proposal X / Y / Z + validation via E2a PASS, stretch IMG-A `S2_ipc_pcg` | Fix patch under `patches/R4_fix_<choice>.patch`, validation results under `results/R4_fix_<choice>/`. |
| R5 | Upstream issue / PR draft only — filing is user-triggered | Draft under `upstream_handoff.md`. |

### 3.1 Fix shapes considered for R4

- **(X) defensive CUDA fallback** at `cuda_piecewise_backend.py:163-169`
  — mirror the existing HIP path so a missing capture stream degrades
  to eager execution + warning instead of asserting. Smallest change;
  matches existing precedent.
- **(Y) broaden warmup capture** — ensure the failing guard signature
  (likely a small-batch prefix-cache-hit prefill) is captured during
  warmup so the recompile is unnecessary.
- **(Z) per-model PCG opt-in** — wire Qwen3-VL into
  `is_multimodal_piecewise_cuda_graph_supported` once R3 shows the
  offending path is bounded. Lower-risk than (X) for production, but
  partly overlaps Issue #5 scope; would be coordinated with #5.

R4 picks one of X / Y / Z based on R3 evidence, not in advance.

## 4. Out of scope here

- v1 Phase 0–5 artifacts. Never touched.
- IMG-A non-PCG resume (`S0_ipc_repeat → V0_vllm → S0_noipc`). That is
  the parent `fixed_generator_plan.md` work and stays queued.
- Changes to `--enforce-piecewise-cuda-graph` defaults or the
  `is_multimodal_piecewise_cuda_graph_supported` table without explicit
  user approval. That is Issue #5's scope.
- Filing the upstream issue / PR. R5 produces the draft only.

## 5. Artifact rules

- All sglang source modifications kept as revertable `.patch` files
  under `patches/`. The actual `/sgl-workspace/sglang` working tree
  must be clean between commits in this repo.
- Raw per-run server logs go under `results/<R-id>/raw/` and are
  **NOT committed** unless explicitly approved. Aggregate summaries
  + trimmed excerpts are committed.
- Bench JSONLs are not committed; trimmed summaries only.
- `.claude/settings.local.json` is never staged.

## 6. Commit cadence

Per `CLAUDE.md`: Conventional Commits `type(scope): action target/context`,
no `Co-Authored-By` trailers, no mention of Claude / Anthropic / AI in
any subject / body / scope / trailer. Commit before each phase milestone
and after each experiment result is recorded.
