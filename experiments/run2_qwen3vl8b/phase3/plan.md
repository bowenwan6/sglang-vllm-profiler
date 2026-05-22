# run2 Phase 3 — Profiling & Trace Collection Plan

> **Status: PLAN ONLY — not executed.** Phase 3 collects traces; it produces **no** hypotheses
> (that is Phase 4). Active run `run2_qwen3vl8b`. Gate input: `../phase2/selected_cases.md`.

## 0. Scope

Collect, per promoted case, a clean **SGLang two-trace pair** (mapping graph-off + formal graph-on,
stage-separated) and a **vLLM two-window pair** (prefill_like + decode_like), with per-collection
metadata. No interpretation. Mechanisms below were verified available 2026-05-22:

- SGLang: `python3 -m sglang.profiler --url ... --num-steps N --profile-by-stage --output-dir ...` ✅
- vLLM: server launched with `VLLM_TORCH_PROFILER_DIR=...`; profile window via `/start_profile` +
  `/stop_profile` HTTP endpoints (vLLM V1 engine supports both) ✅
- Skill collection drivers present: `~/.claude/skills/llm-torch-profiler-analysis/scripts/`
  `run_sglang_torch_profile_host.sh`, `run_vllm_torch_profile_host.sh` ✅
- `debug-cuda-crash` skill present (L1 passive on every SGLang launch) ✅

## 1. GPU & isolation

- **GPU: confirm with user at execution time** (Phase 0/1 used GPU 0; Phase 2 used GPU 7; Case C
  W500 probe used GPU 1). Default proposal: **GPU 7** unless told otherwise. One GPU, all
  servers + profiler clients on the same `CUDA_VISIBLE_DEVICES`.
- Servers strictly serial; GPU freed (<2000 MiB) and verified before the next server.
- SGLang: system `python3`. vLLM: `/opt/miniconda3/envs/profiling/bin/python`.
- `HF_HUB_OFFLINE=1`. Greedy via `--extra-request-body '{"temperature":0,"top_p":1}'` where a
  bench client drives load.

## 2. Preflight (read-only + dir creation)

1. `git status` — if uncommitted relevant changes, checkpoint first (per git workflow).
2. GPU idle (<2000 MiB) on the chosen index; no residual `sglang.launch_server` / `vllm.entrypoints`.
3. Dataset SHA check (no regen): `datasets/run2_qwen3vl8b/case{A,B,C,D}.jsonl`.
4. Model snapshot exists: `…/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.
5. Import check: `python3 -c "import sglang"`; `/opt/miniconda3/envs/profiling/bin/python -c "import vllm"`.
6. Tool/skill check: `sglang.profiler --help`; skill scripts listed above; `debug-cuda-crash/SKILL.md`.
7. Create artifact dirs (§3).

## 3. Artifact layout

```
traces/run2_qwen3vl8b/{case}/
  sglang_mapping/            # graph-off, --profile-by-stage (EXTEND + DECODE)
  sglang_formal/             # graph-on,  --profile-by-stage (EXTEND + DECODE)
  vllm/prefill_like/         # concurrency=1 window, prefill-dominated
  vllm/decode_like/          # steady-state concurrency window
  collection_notes.md        # per-case: commands run, trace sizes, sanity verdicts
logs/run2_qwen3vl8b/phase3/   # server logs + sglang_%i.log L1 kernel trails
experiments/run2_qwen3vl8b/phase3/
  plan.md                    # this file
  scripts/                   # collection orchestration (to be written at execution time)
  metadata/{case}_meta.json  # versions, flags, warmup, dataset SHA, GPU, timestamps, trace sizes
```

`{case}` ∈ {`caseA_short`, `caseC_batched`, `caseB_longprefill`, `caseD_decode`} (profiled in that order).

## 4. SGLang trace protocol (per case)

Two traces per case, both with `--profile-by-stage` to separate EXTEND (prefill) from DECODE:

- **Mapping (graph-off)** — launch with `--disable-cuda-graph --disable-piecewise-cuda-graph`
  (+ per-case flags from §6). Carries the `kernel → cpu_op → python_scope` mapping that Phase-4
  triage needs. Drive with `sglang.profiler --num-steps 8 --profile-by-stage`.
- **Formal (graph-on)** — relaunch with CUDA graph enabled (per-case flags otherwise identical);
  warm to stable batch shape; profile with the same `sglang.profiler` call.

All SGLang launches keep `SGLANG_KERNEL_API_LOGLEVEL=1`,
`SGLANG_KERNEL_API_LOGDEST=logs/run2_qwen3vl8b/phase3/sglang_%i.log`.

## 5. vLLM trace protocol (per case)

vLLM has no `--profile-by-stage`; stage separation comes from two driven windows:

- **prefill_like/** — open window (`POST /start_profile`), send N=8 requests at the case prompt
  length at concurrency=1, close (`POST /stop_profile`) right after the last first-token. Window is
  prefill-dominated (one decode token per request).
- **decode_like/** — warm to a stable steady-state batch at the case concurrency, open window after
  ≥30 s steady decode, capture ~5 s, close.

Server launched with `VLLM_TORCH_PROFILER_DIR=traces/run2_qwen3vl8b/{case}/vllm/<window>`. **Case B
vLLM traces carry confidence ceiling M** (bimodal baseline) — record in `collection_notes.md`.

## 6. Per-case command templates (drafts — not executed)

`SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
`GPU=<confirm>` · SGLang port 30000 · vLLM port 30001.

### Case A — 128→128, c=1 — flags `--disable-overlap-schedule` (warmup 30, reps 3)

SGLang mapping (graph-off):
```
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 SGLANG_KERNEL_API_LOGLEVEL=1 \
SGLANG_KERNEL_API_LOGDEST=logs/run2_qwen3vl8b/phase3/sglang_%i.log \
python3 -m sglang.launch_server --model-path $SNAP --dtype bfloat16 --port 30000 --tp 1 \
  --attention-backend flashinfer --disable-overlap-schedule \
  --disable-cuda-graph --disable-piecewise-cuda-graph
# then:
python3 -m sglang.profiler --url http://127.0.0.1:30000 --num-steps 8 --profile-by-stage \
  --output-dir traces/run2_qwen3vl8b/caseA_short/sglang_mapping
```
SGLang formal (graph-on): same launch **without** the two graph-disable flags →
`--output-dir traces/run2_qwen3vl8b/caseA_short/sglang_formal`.

vLLM:
```
CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 \
VLLM_TORCH_PROFILER_DIR=traces/run2_qwen3vl8b/caseA_short/vllm/prefill_like \
/opt/miniconda3/envs/profiling/bin/python -m vllm.entrypoints.openai.api_server \
  --model $SNAP --dtype bfloat16 --port 30001 --tensor-parallel-size 1
# prefill_like: POST /start_profile → send 8×(128-tok prompt) at c=1 → POST /stop_profile
# decode_like: relaunch with VLLM_TORCH_PROFILER_DIR=.../vllm/decode_like, warm to steady c=1, capture ~5s
```

### Case C — 512→128, c=16 — default flags (warmup 500, reps 5)
Same templates; **no** `--disable-overlap-schedule`. Load driven at c=16 from `caseC_batched.jsonl`.
decode_like window warms to steady c=16 batch.

### Case B — 2048→128, c=1 — default flags (warmup 300, reps 5) — **ceiling M**
Same templates; prompt length 2048 from `caseB_longprefill.jsonl`. Mark all vLLM traces ceiling M.

### Case D — 512→512, c=16 — default flags (warmup 30, reps 3)
Same templates; output length 512 from `caseD_decode.jsonl`.

## 7. Execution order (pilot-gated — do not run all blindly)

1. **Case A (pilot).** Collect all 4 traces. **Verify** non-empty, in 20–500 MB range, EXTEND+DECODE
   present, before proceeding.
2. **Case C.** Only after Case A traces verified useful.
3. **Decision point:** then decide whether to run **B** and **D** immediately or defer (B is noisy +
   ceiling M; D is low payoff). Bring this decision to the user.

## 8. Stop conditions

- Profiler unavailable / errors on invocation.
- Any produced trace empty or unreadable.
- Server OOM / CUDA error / traceback.
- GPU not freed (<2000 MiB) after server shutdown.
- Profiling perturbs behavior so much the trace is unrepresentative (note + re-collect with fewer steps).
- Need to modify SGLang source.
- Trace size explodes (>1 GB per trace → re-collect with fewer `--num-steps`).

## 9. Phase 3 exit criteria (per case actually profiled)

- `sglang_mapping/` non-empty (graph-off, EXTEND+DECODE).
- `sglang_formal/` non-empty (graph-on, EXTEND+DECODE).
- `vllm/prefill_like/` non-empty.
- `vllm/decode_like/` non-empty.
- `metadata/{case}_meta.json` records: framework versions, server flags, warmup, reps, dataset SHA,
  GPU index, timestamps, per-trace file sizes; Case B flagged ceiling M.
- No final hypotheses produced (Phase 4 does that).

## 10. Skill usage (Phase 3)

- `llm-torch-profiler-analysis` → **collection drivers only** (`run_sglang_torch_profile_host.sh`,
  `run_vllm_torch_profile_host.sh`); **no triage** in Phase 3.
- `debug-cuda-crash` → L1 passive on every SGLang launch; escalate to L3/L5/L10 only on an actual crash.
- `llm-serving-auto-benchmark` → not used in Phase 3.

## 11. Open items requiring user confirmation before execution

1. **GPU index** for Phase 3 (proposal: GPU 7).
2. **Scope of first run**: pilot Case A only, then pause for verification before Case C? (recommended).
3. Whether to **defer B and D** until A+C traces are reviewed.
