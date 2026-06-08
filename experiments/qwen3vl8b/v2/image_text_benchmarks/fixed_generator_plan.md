# Issue #4 recovery plan — image+text benchmarks on the fixed generator

> **Status: PLAN ONLY — nothing here has been executed.** Do **not** start any
> server, do **not** run any benchmark, do **not** modify SGLang source, and do
> **not** alter raw JSON/traces/logs from prior runs (including the invalidated
> partial IMG-A and the original smoke).
>
> Companion docs: [`protocol.md`](protocol.md) (original #4 protocol — still the
> source of truth for workload identity, dataset recipe, and acceptance rules),
> [`debug_video_pad/upstream_fix_plan.md`](debug_video_pad/upstream_fix_plan.md)
> (the generator fix in `/data/sglang-pr`), [`debug_video_pad/validation_plan.md`](debug_video_pad/validation_plan.md)
> (V1+V2 validation evidence — both PASS).

---

## 1. Goal

After the benchmark-generator special-token bug is fixed in SGLang, **resume #4
image+text benchmarking** to measure Qwen3-VL image+text behavior and separately
quantify three levers, each against its own baseline:

- **CUDA IPC benefit** = `S0_ipc` vs `S0_noipc` (same workload).
- **PCG benefit** = `S2_ipc_pcg` vs bracketed `S0_ipc` (same workload).
- **vLLM anchor** = clean cross-framework context, **not** causal proof.

Carry-over facts that constrain this plan:

1. **#2 (text-only, default-overlap) is COMPLETE/PASS.** Case A has a real
   text-only TTFT gap (SGLang 21.94 ms vs vLLM 13.12 ms; PCG brings SGLang to
   14.04 ms). Case C shows no material gap and no Case-A-like PCG benefit.
2. **#4 image+text was previously blocked** by a *benchmark-generator* bug
   (`gen_mm_prompt` emitted `<|video_pad|>` etc.), **not** by IPC, PCG, cache, or
   serving. V1 audit and V2 serving repro confirmed this.
3. **The prior partial IMG-A run is INVALID for performance conclusions.** Only
   3/5 reps of one of five variants ran, with 2 failures in rep 3. Do not cite
   any number from `results/imgA_summary.md` as evidence; keep that file as
   historical record of the failure path only.
4. **The sanitized monkeypatch wrapper (`bench_serving_sanitized.py`) is a
   fallback, not the main plan.** Use the fixed-upstream path as the primary
   route. The sanitized wrapper stays available only if the fixed clone is
   unavailable or fails Stage 4.1.
5. **No SGLang source modification under `/sgl-workspace/sglang`.** The fixed
   code lives in the separate clone at `/data/sglang-pr` and is selected via
   `PYTHONPATH`, never by editing `/sgl-workspace/sglang`.

---

## 2. Fixed SGLang sync plan (commands as templates — DO NOT EXECUTE YET)

The fix is committed on the local fork branch:

```text
repo:    /data/sglang-pr
branch:  fix/mm-benchmark-special-tokens
commit:  78e6c03e2  fix(benchmark): exclude special tokens from multimodal prompts
remote:  fork  git@github.com:bowenwan6/sglang.git
upstream remote: origin  https://github.com/sgl-project/sglang.git
```

**Note on upstream merge state:** as of writing, the fix is on the fork branch and
in the active local clone, but `origin/main` (upstream) does **not** yet contain
this commit. Treat the local fix branch as the source of truth for the
fixed-generator path. If/when the PR is merged into `origin/main`, this section's
templates apply equally — just point at the merged SHA instead.

Pre-execution sync (templates):

```bash
cd /data/sglang-pr
git fetch origin                                # upstream main
git fetch fork                                  # our fork
# A) if upstream-merged: rebase onto upstream main
git checkout main && git pull origin main
# B) otherwise stay on the fix branch
git checkout fix/mm-benchmark-special-tokens
git rev-parse HEAD                              # record SHA in results
```

Verify the fix is in the working tree (templates):

```bash
# 1. Fix is present in the source file
grep -n "get_available_multimodal_text_tokens\|all_special_ids" \
  /data/sglang-pr/python/sglang/benchmark/datasets/common.py

# 2. Python actually imports the fixed copy (not /sgl-workspace/sglang)
PYTHONPATH=/data/sglang-pr/python python3 -c \
  "import sglang, sglang.benchmark.datasets.common as c; \
   print('sglang_file=', sglang.__file__); \
   print('common_file=', c.__file__); \
   import inspect; src=inspect.getsource(c.gen_mm_prompt); \
   assert 'get_available_multimodal_text_tokens' in src, 'fix NOT loaded'"

# 3. Patched generator yields zero forbidden tokens (re-uses V1 audit)
PYTHONPATH=/data/sglang-pr/python:$PYTHONPATH \
  python3 experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/debug_payload_audit.py \
  --seeds 10 --prompts-per-seed 430
# expect: buggy_generator.prompts_with_any_forbidden == 0 (the "buggy" name is
# historical — the test runs the currently-imported gen_mm_prompt, which here is
# the FIXED one)
```

Provenance to record in every fixed-generator results file:

- SGLang clone path: `/data/sglang-pr`
- SGLang branch: `fix/mm-benchmark-special-tokens` (or `main` post-merge)
- SGLang commit SHA: from `git rev-parse HEAD`
- `sglang.__file__` and `sglang.benchmark.datasets.common.__file__` from the
  runtime `python -c` check above
- vLLM version: from the vLLM env (`/opt/miniconda3/envs/profiling/bin/python -c
  "import vllm; print(vllm.__version__)"`)

---

## 3. Workloads

Per [`protocol.md`](protocol.md) §3, unchanged shapes:

| id | images | resolution | text in (target) | output | concurrency | num-prompts | warmup | reps |
|---|---|---|---|---|---|---|---|---|
| **IMG-A** single image + short text | 1 | 720p | ~128 tok | 128 | 1 | 400 | 30 | 5 |
| **IMG-B** single image + medium text | 1 | 720p | ~512 tok | 128 | 1 | 400 | 30 | 5 |
| **IMG-C** single image + short/medium text, batched | 1 | 720p | ~128 tok | 128 | 16 | 2000 | 500 | 3 per interleaved block |

Optional later:

| id | images | resolution | text in | output | concurrency | num-prompts | warmup | reps |
|---|---|---|---|---|---|---|---|---|
| IMG-D multi-image | 2–3 (`--image-count`) | 720p | ~128 tok | 128 | 1 | 400 | 30 | 5 |

**Why IMG-A first:**

- Closest image-path analogue of #2 Case A (c=1, short text, 720p single image).
- Low-concurrency prefill is the path where PCG is most plausibly beneficial
  (text-only Case A: 21.94 → 14.04 ms).
- Smallest run footprint among the three primary workloads, so it is the
  fastest way to validate the fixed-generator path end-to-end and surface any
  IPC/PCG signal before committing to IMG-B/C runtime.

IMG-D only after IMG-A/B/C are stable **and** explicit approval.

---

## 4. Variants (per workload, per `protocol.md` §5)

| id | framework | env | distinguishing flags | role |
|---|---|---|---|---|
| `S0_ipc` | SGLang | `SGLANG_USE_CUDA_IPC_TRANSPORT=1` | *(none — overlap ON)* | **headline image baseline (IPC on)** |
| `S2_ipc_pcg` | SGLang | `SGLANG_USE_CUDA_IPC_TRANSPORT=1` | `--enforce-piecewise-cuda-graph` | PCG testing lever |
| `V0_vllm` | vLLM | *(IPC n/a)* | *(none)* | clean cross-framework anchor |
| `S0_noipc` | SGLang | `SGLANG_USE_CUDA_IPC_TRANSPORT` **unset** | *(none)* | **IPC ablation** |

Comparison map:

- **IPC benefit (Q3):** `S0_noipc` vs `S0_ipc` on the same workload.
- **PCG benefit (Q2):** `S2_ipc_pcg` vs **bracketed** `S0_ipc` on the same workload.
- **vLLM anchor (Q1):** `V0_vllm` vs `S0_ipc`. Anchor only — not causal proof of
  any SGLang internal mechanism. SGLang-vs-SGLang remains the causal comparison.

`S0_noipc` is required on **IMG-A only**. On IMG-B/C it is optional, run only on
explicit approval (runtime control).

---

## 5. Run design (gated stages)

### Stage 4.1 — fixed-generator smoke

Run a tiny smoke (analogous to the original 2026-05-30 smoke, but on the fixed
generator) covering the three paths:

- SGLang **IPC on** (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`)
- SGLang **no-IPC**
- vLLM anchor (via `sglang-oai-chat`)

Per-case: `num-prompts=2`, `warmup=1`, `--max-concurrency=1`, `--image-count=1`,
`--image-resolution=720p`, `--image-content=random`, `--image-format=png`,
`--random-input-len=128`, `--random-output-len=32`, `--random-range-ratio=1.0`,
`--seed=1`, greedy. No KAPI, no profiler. GPU selected at preflight (default
GPU 7; never auto-switched).

**Outputs (committed):**

- `image_text_benchmarks/smoke_fixed/smoke_results.json`
- `image_text_benchmarks/smoke_fixed/smoke_summary.md`

Server logs under `logs/qwen3vl8b/v2/image_text_benchmarks/smoke_fixed/` — **not
committed** unless approved.

**Acceptance:** see §7 generator-fix gate plus §7 clean-run gate.

### Stage 4.2 — IMG-A formal

Only if Stage 4.1 passes. Bracket order (unchanged from `protocol.md`):

```text
S0_ipc → S2_ipc_pcg → S0_ipc_repeat → V0_vllm → S0_noipc
```

`S0_ipc_repeat` brackets `S2_ipc_pcg` to measure baseline drift; `S0_noipc` last
gives the IPC delta. Each variant: fresh server, smoke check (non-empty greedy
output on a real image+text request) before reps, then reps; kill server and
confirm GPU returns below 2000 MiB before next variant. **Never co-resident.**

Reps: **5 if runtime permits** (per protocol §3). Reduced plan if needed: **reps
= 3** (still bracket S2). Never reduce `num-prompts`, warmup, the 0-failure gate,
the clean (no-KAPI/profiler) condition, the IPC-on requirement for SGLang
headline, or the bracket drift control.

**Outputs (committed):**

- `image_text_benchmarks/results_fixed/imgA_results.json` (aggregate)
- `image_text_benchmarks/results_fixed/imgA_summary.md`

Raw per-rep dumps under `results_fixed/raw/` and server logs under
`logs/qwen3vl8b/v2/image_text_benchmarks/results_fixed/` — **not committed**
unless approved.

### Stage 4.3 — IMG-B / IMG-C decision

After IMG-A. Only proceed if all of:

- IMG-A clean (0 failures across all variants/reps),
- bracket drift `S0_ipc` vs `S0_ipc_repeat` ≤ 5% (else downgrade absolute
  numbers to "indicative" and consider re-running with fresh servers per rep),
- runtime acceptable,
- explicit approval.

Then:

- **IMG-B** (≈ 512 text tokens, c=1) — same variant set, bracket order, reps=5
  (or 3 reduced). `S0_noipc` optional on IMG-B (runtime).
- **IMG-C** (c=16 batched, n=2000, warmup=500) — **interleaved** order:
  `S0_a → S2_a → S0_b → S2_b → S0_c → V0`. Reps = 3 per S0/S2 interleaved
  block. `S0_noipc` only on explicit approval.

**Do not run IMG-B/C automatically** if IMG-A has any failure, unstable bracket
drift > 5%, or a stop condition fires.

---

## 6. Metrics (per rep + per-variant median / CV)

Recorded per rep:

- **Timing:** TTFT p50/p95/p99, TPOT p50, E2E latency, output throughput
  (tok/s), request throughput (req/s).
- **Aggregation:** median across reps, CV across reps (population stdev / mean).
- **Quality:** completed count, failures (errors list non-empty), error rate.

Recorded per run (provenance):

- GPU id, exact server flags, KAPI/profiler-disabled confirmation
  (`kapi_logging=false, profiler=false`).
- `SGLANG_USE_CUDA_IPC_TRANSPORT` on/off.
- `--enforce-piecewise-cuda-graph` on/off.
- SGLang version + **commit SHA** + `sglang.__file__` (proves fixed path used).
- vLLM version (for `V0_vllm`).
- Model snapshot SHA.
- Benchmark backend (`sglang-oai-chat`).
- Dataset recipe: `--seed`, `--image-count`, `--image-resolution`,
  `--image-format`, `--image-content`, `--random-input-len`,
  `--random-output-len`, `--random-range-ratio`, `--num-prompts`,
  `--warmup-requests`.
- Smoke check output (non-empty greedy text from a real image+text request).

Recorded per request (from harness JSONL):

- `prompt_len` (text+vision), `text_prompt_len`, `vision_prompt_len`.
- Image params used for the run.

TTFT composition boundary (unchanged from `protocol.md` §7):

> The benchmark client measures **end-to-end serving TTFT** — it **includes**
> image preprocessing + vision-encoder + prefill. It does **not** split them.
> The headline metric is end-to-end serving TTFT. A preprocessing /
> vision-encoder / prefill / decode breakdown is an optional future profiler
> track (separate trace collection), **not** required for #4 headline.

---

## 7. Acceptance criteria

**Clean-run gate** (every counted rep):

- 0 failures (errors list empty).
- No KAPI env var set; no profiler active.
- Servers serialized — never co-resident.
- Smoke check (real image+text request → non-empty greedy output) passes before
  the variant's reps.
- After each variant: server killed, GPU returns to < 2000 MiB.

**Generator-fix gate** (specific to this recovery):

- No `<|video_pad|>` / `<|vision_*|>` / multimodal special-token failures.
- Pre-run check: `python -c "import sglang, sglang.benchmark.datasets.common as
  c; ..."` confirms fixed file is loaded (not `/sgl-workspace/sglang`).
- If a special-token-shaped failure reappears in any run, **stop and debug the
  import path / generator before continuing**. Do not treat as a transient.

**IPC-on gate** (headline):

- Every SGLang **headline** image run records
  `SGLANG_USE_CUDA_IPC_TRANSPORT=1`. `S0_noipc` is the only SGLang variant
  with it unset.

**No external download gate:**

- Synthetic image dataset only. No network fetch during benchmark.

**IPC benefit (Q3) declared only if:**

- `S0_ipc` improves TTFT (or throughput) vs `S0_noipc` by **≥ 5%** on the
  same workload, state direction explicitly.
- No TPOT regression (≤ baseline TPOT × 1.05).
- 0 failures on both variants.

**PCG benefit (Q2) declared only if:**

- `S2_ipc_pcg` median TTFT improves vs bracketed `S0_ipc` by **≥ 5%**.
- TPOT not materially worse (≤ baseline TPOT × 1.05).
- 0 failures.

**Bracket-drift gate:**

- `S0_ipc` vs `S0_ipc_repeat` (or IMG-C three S0 blocks) drift ideally **≤ 5%**.
- If > 5%, downgrade absolute numbers to "indicative" and rely on within-design
  comparisons.

**vLLM anchor rule:**

- Use as cross-framework context only. Do not infer SGLang internal mechanism
  from vLLM alone. Causal comparisons are SGLang-vs-SGLang (S0 vs S2,
  S0_ipc vs S0_noipc).

**Separation requirement:**

- Image+text conclusions reported **separately** from text-only (#2).
- IPC and PCG reported as **two distinct** benefit results, never merged.

**IMG-C verdict wording:**

- If S0, S2, vLLM all sit inside the noise band, conclude *"no material gap /
  no Case-A-like benefit on the image batched path."* Not parity, not a proven
  gap.

---

## 8. Stop conditions (abort, do not work around)

Stop immediately and record (without continuing) if any of:

- Target GPU not idle (≥ 2000 MiB / other heavy process) at run time.
- OOM (vision tower / KV cache) — record resolution/image-count and stop.
- Any counted variant has failures > 0.
- A **forbidden special-token error** reappears (`No data iterator found for
  token: <|video_*|>`). Means the import path is wrong or the fix did not
  cover this case → fix before continuing.
- `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST` set anywhere in
  the env at run time, or a profiler active.
- A server fails to release GPU memory after kill (still occupied before next
  variant).
- Server log / raw output grows abnormally large (instrumentation leak).
- `sglang.__file__` resolves to `/sgl-workspace/sglang/...` instead of
  `/data/sglang-pr/python/...` for any SGLang variant.
- Plan would require editing SGLang source.
- `S2_ipc_pcg` smoke / correctness check fails.

If `--backend sglang-oai-chat` against vLLM regresses (was confirmed working
2026-05-30), stop and report; do **not** improvise an unverified vLLM image
path.

---

## 9. Artifact plan

Write only under v2 paths. New directories for the fixed-generator path:

```text
experiments/qwen3vl8b/v2/image_text_benchmarks/
  fixed_generator_plan.md                       (this file)
  smoke_fixed/
    smoke_results.json                          (Stage 4.1 aggregate)
    smoke_summary.md                            (Stage 4.1 narrative)
  results_fixed/
    imgA_results.json                           (Stage 4.2 aggregate)
    imgA_summary.md                             (Stage 4.2 narrative)
    raw/<variant>_rep<N>.jsonl                  (NOT committed unless approved)
    imgB_results.json / imgB_summary.md         (Stage 4.3, if reached)
    imgC_results.json / imgC_summary.md         (Stage 4.3, if reached)

logs/qwen3vl8b/v2/image_text_benchmarks/
  smoke_fixed/<case>_server.log                 (NOT committed unless approved)
  results_fixed/<variant>_server.log            (NOT committed unless approved)
```

**Do not overwrite** any of:

- `results/imgA_summary.md` (historical, invalidated)
- `results/imgA_results.json` (historical, invalidated)
- `results/raw/IMG_A_S0_ipc_rep*.jsonl` (historical, invalidated)
- `smoke/smoke_summary.md`, `smoke/smoke_results.json` (original smoke)
- `debug_video_pad/` artifacts (V1/V2 evidence, audit, plans)
- `caseAC_rebaseline/` (#2 results)
- any v1 Phase 0–5 artifacts

Sanitized monkeypatch wrapper (`bench_serving_sanitized.py` and
`run_image_text_smoke_sanitized.py` / `run_image_text_imgA_sanitized.py`) stays
on disk as a **documented fallback only** — not used in the main fixed-generator
path. If invoked, it writes under `smoke_sanitized/` and `results/imgA_sanitized_*`
(separate namespaces).

---

## 10. Command templates — **DO NOT EXECUTE YET**

Common clean env for any GPU stage:

```bash
export CUDA_VISIBLE_DEVICES=<GPU_ID>            # 7 unless explicitly approved otherwise
export HF_HUB_OFFLINE=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST
export PYTHONPATH=/data/sglang-pr/python:${PYTHONPATH-}

# Provenance: record exactly which sglang gets imported
python3 -c "import sglang, sglang.benchmark.datasets.common as c; \
  print('sglang_file=', sglang.__file__); \
  print('common_file=', c.__file__)"

# vLLM uses its own env (no sglang import), unchanged from prior runs.
```

Stage 4.1 — fixed-generator smoke (template):

```bash
# preflight
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i <GPU_ID>
python3 -c "import os; assert 'SGLANG_KERNEL_API_LOGLEVEL' not in os.environ"

# server (SGLang IPC on)
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
  PYTHONPATH=/data/sglang-pr/python python3 -m sglang.launch_server \
  --model-path "$SNAP" --dtype bfloat16 --port 30000 --tp 1 \
  --attention-backend flashinfer
# (or unset IPC for noipc case; vLLM uses VLLM_PYTHON env)

# bench (image dataset; backend sglang-oai-chat for BOTH frameworks)
PYTHONPATH=/data/sglang-pr/python python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:<port> \
  --model "$SNAP" \
  --dataset-name image \
  --image-count 1 --image-resolution 720p \
  --image-format png --image-content random \
  --random-input-len 128 --random-output-len 32 \
  --random-range-ratio 1.0 \
  --max-concurrency 1 --num-prompts 2 \
  --warmup-requests 1 --seed 1 \
  --extra-request-body '{"temperature": 0, "top_p": 1}' \
  --output-details --output-file <smoke_fixed/<case>_bench.jsonl>
```

Stage 4.2 — IMG-A formal (template, single variant shown):

```bash
# preflight identical to Stage 4.1
# SGLang server (S0_ipc):
SGLANG_USE_CUDA_IPC_TRANSPORT=1 \
  PYTHONPATH=/data/sglang-pr/python python3 -m sglang.launch_server \
  --model-path "$SNAP" --dtype bfloat16 --port 30000 --tp 1 \
  --attention-backend flashinfer
# (for S2_ipc_pcg: add --enforce-piecewise-cuda-graph)
# (for S0_noipc: DROP SGLANG_USE_CUDA_IPC_TRANSPORT export)

# bench client (n=400 warmup=30 c=1; reps controlled by runner loop)
PYTHONPATH=/data/sglang-pr/python python3 -m sglang.bench_serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:30000 \
  --model "$SNAP" \
  --dataset-name image \
  --image-count 1 --image-resolution 720p \
  --image-format png --image-content random \
  --random-input-len 128 --random-output-len 128 \
  --random-range-ratio 1.0 \
  --max-concurrency 1 --num-prompts 400 \
  --warmup-requests 30 --seed 1 \
  --extra-request-body '{"temperature": 0, "top_p": 1}' \
  --output-details --output-file <results_fixed/raw/<variant>_rep<k>.jsonl>
```

Vetting note:

- The fixed-generator path is selected via `PYTHONPATH`. Never edit
  `/sgl-workspace/sglang` to make the test pass.
- The runner skeleton from the original
  `experiments/qwen3vl8b/v2/image_text_benchmarks/run_image_text_smoke.py` and
  `run_image_text_imgA.py` should be adapted (or new fixed-generator copies
  created) so that:
  - they propagate `PYTHONPATH=/data/sglang-pr/python` into both the server
    `subprocess.Popen` and the `bench_serving` `subprocess.run` calls;
  - they write under `smoke_fixed/` and `results_fixed/`;
  - they record `sglang.__file__` and SGLang commit SHA in the result JSON.

That adaptation is a follow-up implementation task, **not** part of this plan.

---

## 11. Documentation updates

When this plan is committed, also update:

- `plan.md`: change #4 row to **"P1 — UNBLOCKED / fixed-generator recovery
  plan drafted"** and update §5 Immediate Next Step to point at this file.
- `experiments/qwen3vl8b/v2/image_text_benchmarks/README.md`: replace the
  active-blocker section with a pointer to `fixed_generator_plan.md` and the
  fixed-clone path; keep the link to `validation_plan.md` for V1+V2 evidence.

**Do NOT** update the final report yet. Reports update only after clean IMG-A
fixed-generator data exists.

---

## 12. Execution checklist (forward-looking)

Pre-flight (no GPU):

- [ ] Sync `/data/sglang-pr` (or accept current `HEAD = 78e6c03e2`).
- [ ] Verify fix present in `python/sglang/benchmark/datasets/common.py`.
- [ ] Run the `python -c "import sglang, ..."` check; record `sglang.__file__`
      under `/data/sglang-pr/python/...`, **not** `/sgl-workspace/sglang/...`.
- [ ] Optional: re-run V1 audit via patched `PYTHONPATH`; assert 0
      forbidden-token hits.

GPU stages:

- [ ] Choose idle GPU (default GPU 7; never auto-switch).
- [ ] Stage 4.1 fixed-generator smoke; summarize → `smoke_fixed/smoke_summary.md`.
- [ ] Decide whether to proceed to IMG-A (must satisfy §7 gates).
- [ ] Stage 4.2 IMG-A formal (bracket order); summarize →
      `results_fixed/imgA_summary.md`.
- [ ] Decide whether to expand to IMG-B / IMG-C (§5 Stage 4.3).
- [ ] Stage 4.3 IMG-B (only if approved).
- [ ] Stage 4.3 IMG-C (only if approved).

Discipline at every commit:

- Commit only docs / processed result Markdown + aggregate JSON.
- **Do not** commit raw per-rep JSONL or server logs unless explicitly
  approved.
- **Do not** stage `.claude/settings.local.json`, scheduled-task locks, or
  unrelated working-tree drift.
- Commit messages follow `type(scope): summary`. No `Co-Authored-By` for AI;
  no Claude / Anthropic / AI mentions in subjects, bodies, scopes, or
  trailers.
- After each commit, `git log --max-count=5 --format='%h %s%n%B' | rg -i
  'claude|anthropic|co-authored-by'` must return nothing.
