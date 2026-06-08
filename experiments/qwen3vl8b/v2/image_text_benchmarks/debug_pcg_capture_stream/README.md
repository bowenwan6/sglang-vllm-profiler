# Debug — PCG capture-stream assertion on image+text + PCG + IPC

> **Active workstream.** Goal: classify the IMG-A `IMG_A_S2_ipc_pcg` server
> crash so we can decide whether to (a) file an upstream SGLang issue, (b)
> propose an SGLang PR, or (c) continue #4 without PCG with documented
> rationale. **Do not modify `/data/sglang-pr` source** during this debug.

## Blocker symptom

Stage 4.2 IMG-A formal (2026-06-08, GPU 7, fixed-generator path,
`/data/sglang-pr` HEAD `62c505a196`, merged fix `07f326c184` in history):

- `IMG_A_S0_ipc` (IPC on, PCG off): ✅ 5/5 reps, 2000 requests, 0 failures,
  TTFT p50 64.8 ms.
- `IMG_A_S2_ipc_pcg` (IPC on, **PCG on**): ❌ rep 1 server crash on first
  prefill.

Server traceback head
(`logs/qwen3vl8b/v2/image_text_benchmarks/results_fixed/IMG_A_S2_ipc_pcg_server.log`):

```text
File "/data/sglang-pr/python/sglang/srt/compilation/cuda_piecewise_backend.py",
     line 171, in __call__
    stream is not None
AssertionError: PCG capture stream is not set, please check if runtime
                recompilation happened
```

The assertion fires inside `cuda_piecewise_backend.__call__` for a compiled
graph-module call of the Qwen3-VL forward path (`qwen3_vl.py:1277 →
mm_utils.general_mm_embed_routine → language_model.forward → torch.compile
→ <eval_with_key>.669`).

Server args at crash:

- `attention-backend flashinfer`
- `disable_piecewise_cuda_graph = False`
- `enforce_piecewise_cuda_graph = True`
- `SGLANG_USE_CUDA_IPC_TRANSPORT = 1`

Per protocol §9, this is the documented stop condition `S2_ipc_pcg correctness
check fails`. Remaining variants (`IMG_A_S0_ipc_repeat`, `IMG_A_V0_vllm`,
`IMG_A_S0_noipc`) were skipped.

## What this debug is and is not

- **Is NOT a generator bug.** The fix gate stayed green throughout: bench
  client and server both imported the patched
  `sglang.benchmark.datasets.common.gen_mm_prompt` with the
  `get_available_multimodal_text_tokens` exclusion. Zero forbidden-token errors
  across the 5 clean reps of `IMG_A_S0_ipc`.
- **Is NOT a flashinfer / sglang-kernel issue.** Those were already upgraded
  (`flashinfer 0.6.11→0.6.12`, `sglang-kernel 0.4.2→0.4.3`) for the upstream
  `main` HEAD, and `IMG_A_S0_ipc` runs over them cleanly.
- **Is NOT a v1 / #2 regression.** Old `/sgl-workspace/sglang` (#2 SHA
  `0c8049d9b`) ran Case A text-only with PCG cleanly. The new upstream `main`
  PCG implementation under VLM + image + IPC is what changed.
- **Is the question of which factor matters:** PCG, IPC, image / VLM path, or
  broader upstream PCG regression.

## Layout (created as phases proceed)

- `README.md` — this file (blocker summary + status pointer).
- `static_audit.md` — read-only audit of the SGLang code path that fires the
  assertion (Step 3 of the debug plan).
- `experiment_plan.md` — D1–D6 staged matrix to isolate the factor (Step 4).
- `results/<stage>_summary.md` + `results/<stage>_results.json` — per-stage
  outputs (Step 6).
- `conclusion.md` — final classification + recommendation (Step 7).
- Raw bench JSONL: `results/raw/<stage>_*.jsonl` (NOT committed unless
  explicitly approved).
- Server logs:
  `logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/` (NOT
  committed unless explicitly approved).

## Safety rules

- GPU 7 only. Never auto-switch.
- No KAPI (`SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST` must be
  unset).
- No profiler.
- Do NOT modify `/data/sglang-pr` source during debug. If a code fix
  experiment is later approved, it goes on a fresh branch in a fresh
  worktree, not in-place.
- Do NOT overwrite `results_fixed/` (Stage 4.2 partial record) or `smoke_fixed/`
  or any historical artifact.
- Do NOT commit raw per-rep JSONL or server logs unless explicitly approved.

## Status pointer

- Plan stage: **Step 1 done** (this README + blocker recorded in `plan.md`,
  `README.md`, `fixed_generator_plan.md`).
- Next: Step 2 (create debug branch) → Step 3 (static audit, no GPU) → Step 4
  (debug experiment plan) → Step 5 (debug runner) → Step 6 (run D1–D4 on GPU
  7) → Step 7 (conclusion).
