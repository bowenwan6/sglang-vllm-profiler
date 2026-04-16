# SGLang vs vLLM Profiling — Task & Execution Plan

## Objective

Profile and compare SGLang against vLLM under fair, controlled conditions to extract actionable optimization insights for SGLang. The goal is not a generic benchmark summary — it is to explain *why* gaps exist and *what SGLang can do about it*. vLLM is used as a strong reference point so that SGLang can learn from its stronger behaviors.

## Core Questions

1. Where is the performance gap? (TTFT, TPOT, throughput)
2. Which stage: prefill/extend or decode?
3. What is responsible: kernels, communication, scheduling, memory?
4. Which vLLM behaviors are real design wins vs noise?
5. Which of those are actionable for SGLang?

## Executive Summary

We compare SGLang and vLLM on `Qwen3-VL-8B-Instruct` (text-only path first) to find concrete ways to improve SGLang. The plan is staged and signal-maximizing: establish one clean baseline at TP=1 on a single H200, identify the 1–2 cases with the most interesting gap, then profile those cases deeply. We do *not* run broad sweeps before we know what we are looking for.

The workflow follows a strict pipeline:

```
benchmark  →  gap identification  →  config shaping  →  profiling  →  interpretation  →  hypothesis  →  validation
 (Phase 1)        (Phase 1/2)        (Phase 2)        (Phase 3)      (Phase 4)         (Phase 4)      (Phase 5)
```

Each skill occupies a specific layer and only that layer:

- **`sglang-auto-benchmark`** is the *controlled-experimentation engine*. It shapes inputs (flags, QPS, concurrency, datasets) and produces metrics. Used for dataset canonicalization (Phase 1 prep), config-space shaping (Phase 2), and hypothesis-validation sweeps (Phase 5). Never used for the cross-framework leg — `sglang.bench_serving` does that because it supports both `--backend sglang-oai` and `--backend vllm` against a shared dataset.
- **`sglang-torch-profiler-analysis`** is the *trace interpreter*. It converts raw traces into kernel / overlap-opportunity / fuse-pattern tables, catalog-checked against known SGLang optimizations so we do not claim novelty on something that already exists. Used only after a case is locked (Phase 4), because it has nothing to do until there is a trace pair worth reading.
- **`debug-cuda-crash`** is the *evidence-preservation safety net*. Passively on at `LOGLEVEL=1` through Phase 1 and Phase 3; escalated to `3 / 5 / 10` only when a specific crash or numerical symptom is observed. Never on at high levels during benchmark or profiling runs, because it perturbs timing.

High-level decision: if the gap is small (<5%) on a case, we *change the workload* before changing the framework. If the gap is large and stable, we shape (Phase 2) before profiling. We only profile cases that survived shaping, because profiling a case that is actually just a missing flag is wasted effort.

## Models

- **Phase 1–4**: `Qwen/Qwen3-VL-8B-Instruct` (17GB, locally cached at `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`)
- **Later**: larger Qwen3-VL variant (TBD)

## Environment

- Host: `radixark02`, container: `sglang-bowenw`
- GPUs: 4× H200 144GB, CUDA 12.9
- SGLang: dev editable @ `/sgl-workspace/sglang`, Python 3.12, torch 2.9.1+cu129
- vLLM: 0.19.0 in conda env `vllm` @ `/opt/miniconda3/envs/vllm`, torch 2.10.0+cu128
- HF cache: `/root/.cache/huggingface`

---

## Skill Usage Model

This section defines the role, placement, and boundary of each skill. Anything outside these boundaries is misuse.

### `sglang-auto-benchmark` — controlled experimentation on the SGLang side

**What it fundamentally solves.** Manual flag-sweep tuning is combinatorial, error-prone, and non-resumable. This skill takes a YAML config describing candidate flags, QPS plan, concurrency, and SLA, then generates candidates, runs each with a fresh server, tracks SLA pass/fail, and writes resumable results (`live_results.jsonl` + `results.{jsonl,csv}` + `summary.md`). It also owns the canonical dataset format (`convert` / `validate`).

**Where it is used in this project.**

| Phase | Subcommand | Purpose |
|---|---|---|
| Phase 1 prep | `convert` + `validate` | Produce the shared `autobench` JSONL that both SGLang and vLLM will consume. Dataset *byte-identity* across frameworks is non-negotiable for fairness. |
| Phase 2 | `run` (tier 1, ≤4 candidates) | Rule out "SGLang loses because we picked the wrong flags." A small, targeted sweep over 1–2 axes (e.g. `chunked_prefill_size`, `schedule_conservativeness`) on *only the case that showed a gap*. Output feeds the Phase 2 decision rule: if shaping closes the gap below 5%, the case is configurational and gets dropped from profiling. |
| Phase 5 | `run` (tier 2, up to ~10 candidates, resumable) | Validate a specific hypothesis coming out of Phase 4 (e.g. "enable_custom_all_reduce × enable_torch_compile"). Resumability matters here because sweeps can run for hours and we do not want to restart on a single server death. |

**What it is NOT used for.**
- ❌ Cross-framework comparison. It cannot drive vLLM. The cross-framework leg uses `sglang.bench_serving --backend {sglang-oai,vllm}` against the JSONL produced by `convert`.
- ❌ Broad tier-3 sweeps before we know which axis matters. Tier-3 is a Phase-5 escalation, not a discovery tool.
- ❌ Primary data for "SGLang vs vLLM" tables. Its outputs describe SGLang at various configs; they do not describe vLLM at all.

**Downstream consumers.**
- Phase 1's `bench_serving` runs consume the JSONL it produced.
- Phase 2's `summary.md` directly determines which cases enter Phase 3 (the `selected_cases.md` artifact).
- Phase 5's `summary.md` updates the confidence column in `hypotheses.md`.

### `sglang-torch-profiler-analysis` — trace interpretation

**What it fundamentally solves.** A raw `trace.json.gz` is kilometers of kernel events with no notion of "which of these matter" or "is this a known pattern". This skill does three things a human should not do by hand: (1) rank kernels by cumulative GPU-time share with Python-source backing, (2) call out overlap headroom with dependency-risk context, (3) match observed kernel clusters against the SGLang fuse/overlap catalogs (including PR-backed / in-flight entries) so we never propose a "new optimization" that already exists.

**Single-trace vs two-trace.** Single-trace triage is enough for a first read on kernel share and fuse candidates. Two-trace (mapping graph-off + formal graph-on) is required before making *overlap claims*, because overlap attribution needs the graph-off trace to recover `kernel → cpu_op → python_scope` mapping that the graph-on trace has already collapsed.

**Where it is used in this project.**

| Phase | Flow | Purpose |
|---|---|---|
| Phase 3 | (trace collection only, no triage yet) | This skill's script can also *drive* `sglang.profiler` against a live server with `--profile-by-stage`. We use that entrypoint for live collection when convenient; otherwise we collect traces with the profiler directly and hand the files to the skill in Phase 4. |
| Phase 4 step 1 | `triage` (two-trace) per selected case, per stage | Produces the three tables per (case × {extend, decode}). This is the primary interpretive artifact of the whole project. |
| Phase 4 step 2 | `triage` (single-trace) on vLLM traces | vLLM traces cannot meaningfully be fed into "two-trace" mode because vLLM does not produce a mapping-style graph-off trace in the same shape. Single-trace triage still gives useful kernel-share numbers for cross-check. |
| Phase 4 step 3 | catalog lookup inside each triage | Before elevating any finding to a hypothesis, compare against `fuse-overlap-catalog.md` and `overlap-catalog.md`. If the pattern is already catalogued, the finding is either "existing path that should apply but is disabled/regressed" (high-confidence, cheap recommendation) or "existing in-flight PR" (not a new idea — flag accordingly). Only when nothing matches do we allow a "truly new" label, with a similarity label (`high/medium/low`) attached. |

**What it is NOT used for.**
- ❌ Phase 1 or Phase 2. There are no useful traces before a case is locked; running triage on a pre-shaping trace wastes the mapping/formal pairing budget on a moving target.
- ❌ Merged-rank traces. Skill explicitly prefers rank-local TP-0 traces; merged traces muddle attribution.
- ❌ Claiming a new optimization without the catalog step. Doing so is an anti-pattern; it is also how plans end up recommending things SGLang already ships.

**Downstream consumers.** The triage tables are the *evidence column* of every Phase 4 hypothesis. A hypothesis without a pointer to a specific row in a triage table is inadmissible.

### `debug-cuda-crash` — evidence-preservation safety net

**What it fundamentally solves.** When SGLang crashes on a CUDA error or silently produces NaN/Inf, the crash itself usually destroys the evidence needed to diagnose it — kernel name, input shapes, input values. The `@debug_kernel_api` decorator logs (and optionally dumps) at every covered kernel boundary *before* the call, so the evidence survives the crash.

**Log levels and cost model.** Level 1 costs essentially nothing (just names at each boundary). Level 3 adds shapes/dtypes (cheap but adds a per-call I/O cost). Level 5 adds tensor stats (requires host sync — measurable perturbation). Level 10 dumps full inputs (large disk + real perturbation). *Benchmark and profile runs require low perturbation*, so only level 1 is allowed during Phase 1 and Phase 3 unless we are deliberately investigating a crash.

**Where it is used in this project.**

| Situation | Setting | Purpose |
|---|---|---|
| Phase 1 baseline runs (all 4 cases × SGLang side) | `LOGLEVEL=1`, `LOGDEST=/data/profiling_lab/logs/phase1_%i.log` | Passive boundary trail. If any Phase 1 SGLang run crashes, we already know the last API boundary without re-running. |
| Phase 3 trace collection (mapping + formal) | `LOGLEVEL=1`, same `LOGDEST` pattern | Same rationale. A crash during a 30-minute profile run that we have to restart from zero is much worse than a negligible `O(ns)` per-call log overhead. |
| A crash occurs (any phase) | Re-run crashing case with `LOGLEVEL=3` | Inspect shapes/dtypes/device at boundary before the crash. |
| Output divergence from Phase 0 or suspected NaN/Inf in a trace | `LOGLEVEL=5` on the specific reproducer | Tensor stats at the boundary, including NaN/Inf counts. |
| Need offline reproduction (e.g. crash only on one prompt) | `LOGLEVEL=10` + `DUMP_DIR` + `DUMP_INCLUDE='sglang.custom_op.*'` | Crash-safe input dump. Temporarily pass `--disable-cuda-graph` because dumps are auto-skipped during graph capture. |

**What it is NOT used for.**
- ❌ Performance debugging. The tool is about correctness and crash forensics, not about figuring out why a kernel is slow — that is `sglang-torch-profiler-analysis`'s job.
- ❌ Running at `LOGLEVEL ≥ 3` during benchmark or profile collection. It changes timing and invalidates the numbers we are trying to measure.
- ❌ Diagnosing vLLM. The decorator only instruments SGLang's covered boundaries.

**Downstream consumers.** Crash logs feed the `Decision Gates` table ("If Phase 3 SGLang crashes on chosen case"). A successful level-10 dump becomes an offline reproducer attached to the hypothesis that produced the crash.

### How the three skills complement (not overlap)

| Layer | Role | Skill | Artifact type |
|---|---|---|---|
| Input control | Shape what SGLang runs, with what flags, on what data | `sglang-auto-benchmark` | JSONL datasets, `results.{jsonl,csv,md}` |
| Output interpretation | Turn raw traces into ranked, catalog-checked findings | `sglang-torch-profiler-analysis` | Kernel / overlap / fuse tables |
| Evidence preservation | Capture crash boundary when either pipeline fails | `debug-cuda-crash` | Boundary logs, level-10 input dumps |

They do not overlap because they operate on different *kinds of artifact*: configs and metrics vs traces vs crash boundaries. If we ever find ourselves reaching for auto-benchmark to explain a kernel, or for profiler-analysis to choose a flag, or for debug-cuda-crash to explain a slowdown, we are holding the wrong tool.

---

## Fairness Matrix

| Dimension | Value | Notes |
|---|---|---|
| Hardware | 1× H200 (GPU 0), same host, same run session | Pin with `CUDA_VISIBLE_DEVICES=0` |
| Model | `Qwen/Qwen3-VL-8B-Instruct` | Same HF revision for both |
| Precision | BF16 | No quantization in Phase 1 |
| TP / DP / PP | TP=1, DP=1, PP=1 | Avoid NCCL variance as a confounder |
| Max seq len | 8192 (both sides) | Matches Qwen3-VL default |
| Prompt set | Generated via `autobench` JSONL, fixed seed | Identical prompts to both servers |
| Sampling | temperature=0, top_p=1 (greedy) | Removes sampling variance for profiling runs |
| Output length | Fixed via `max_tokens` + `ignore_eos=true` | Prevents early termination drift |
| Concurrency | Fixed list: {1, 16} initially | 1 = latency-bound, 16 = batched throughput |
| Warmup | ≥30 requests at target concurrency before measurement window | Both sides |
| Chunked prefill | ON both sides, default chunk size | Document the default on each |
| CUDA graph | ON by default (formal) + OFF variant (mapping) | Graph-off is only for profiling, not benchmarking |
| Prefix cache | OFF in benchmark (randomized prefixes); ON in later experiments | Avoid hidden hit-rate asymmetry |

**Unavoidable mismatches to document explicitly**
- Python/torch versions differ: SGLang uses py3.12 + torch 2.9.1+cu129, vLLM uses py3.12 + torch 2.10.0+cu128 in a separate conda env. This affects kernel backends (Triton, FlashInfer, FlashAttention) subtly. Document versions in every run.
- Default attention backend may differ (SGLang tends to FlashInfer, vLLM may default to FlashAttention v3 on H200). Pin explicitly where possible, document the chosen backend per run.
- Scheduler policies differ by design (radix attention vs vLLM's cache manager). These are *design* differences; we do not try to align them, we measure them.

Mismatches that cannot be eliminated go into every benchmark summary under "Fairness notes" and are considered part of the observation, not a bug.

---

## Phase 0 — Environment & Functional Validation (0.5–1 day)

**Goal**: Prove both servers can serve `Qwen3-VL-8B` correctly and produce the same tokens on the same prompt.

**Actions**
1. Launch SGLang server (main env):
   ```
   python3 -m sglang.launch_server \
     --model-path Qwen/Qwen3-VL-8B-Instruct \
     --port 30000 --tp 1 --attention-backend flashinfer
   ```
2. Launch vLLM server (conda env `vllm`) on a different port:
   ```
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model Qwen/Qwen3-VL-8B-Instruct \
     --port 30001 --tensor-parallel-size 1
   ```
3. Send 3 identical text-only prompts (short / medium / long) with `temperature=0` to both; diff the token outputs.
4. Record a short env snapshot (both sides) into `experiments/env_snapshot.md`.

**Outputs**
- `experiments/env_snapshot.md` — versions, flags, attention backend, chunked-prefill settings, tokenizer hash
- Pass/fail notes on output equivalence

**Success criteria**: Same first ~64 greedy tokens on a matching prompt. Small tail divergence after ~64 tokens is acceptable (numeric drift); flag if divergence appears in the first 16 tokens.

**Risks / failure modes**
- Qwen3-VL may not be supported in SGLang or vLLM at the pinned versions. If so: fall back to `Qwen3-8B` (pure text) for the bulk of Phase 1–4 and add Qwen3-VL as a later variant.
- Vision tower may be loaded even for text-only requests (memory cost). Measure idle memory after load on both sides.

---

## Phase 1 — Minimal Fair Baseline (1 day)

**Goal**: Produce *one* head-to-head table for a minimum set of cases that is clean enough to believe.

**Case matrix (small on purpose)**

| Case | Prompt len | Output len | Concurrency |
|---|---|---|---|
| A. Latency-bound short | 128 | 128 | 1 |
| B. Latency-bound long-prefill | 2048 | 128 | 1 |
| C. Batched throughput | 512 | 128 | 16 |
| D. Decode-heavy | 512 | 512 | 16 |

4 cases × 2 frameworks = 8 runs. Each runs ≥120 s of steady-state after warmup. We do not expand this matrix until Phase 2 tells us which dimension matters.

**Actions**
1. Prepare a fixed canonical dataset for each case using `sglang-auto-benchmark`'s `convert` subcommand so both frameworks share bytes:
   ```
   python -m sglang.auto_benchmark convert \
     --input-format random \
     --prompt-len 512 --output-len 128 --num-prompts 400 \
     --output datasets/caseC.jsonl
   python -m sglang.auto_benchmark validate --input datasets/caseC.jsonl
   ```
2. Run each case via `sglang.bench_serving` against SGLang and then against vLLM, using the same dataset:
   ```
   python -m sglang.bench_serving \
     --backend sglang-oai --base-url http://127.0.0.1:30000 \
     --dataset-name autobench --dataset-path datasets/caseC.jsonl \
     --max-concurrency 16 --ignore-eos --random-seed 1
   # then
   python -m sglang.bench_serving \
     --backend vllm --base-url http://127.0.0.1:30001 \
     --dataset-name autobench --dataset-path datasets/caseC.jsonl \
     --max-concurrency 16 --ignore-eos --random-seed 1
   ```
3. Enable `SGLANG_KERNEL_API_LOGLEVEL=1` + `SGLANG_KERNEL_API_LOGDEST=/data/profiling_lab/logs/sglang_phase1_%i.log` on the SGLang server side. Minimal overhead, gives us a crash trail for free.
4. Repeat each case 3× with independent warmups; take median. Reject any run whose stdev across repeats exceeds 5% of median — rerun or expand warmup.

**Skill usage in this phase**
- `sglang-auto-benchmark` → `convert` + `validate` only. Produces `datasets/case{A,B,C,D}.jsonl`. *Not* `run` — Phase 1 is a cross-framework comparison, and `auto_benchmark run` cannot drive vLLM. The JSONL is the shared input both `bench_serving` invocations consume, guaranteeing byte-identical prompts.
- `debug-cuda-crash` → `LOGLEVEL=1` on the SGLang server side, `LOGDEST=/data/profiling_lab/logs/phase1_%i.log`. Passive safety net; zero impact on measured latency at level 1.
- `sglang-torch-profiler-analysis` → **not used.** Phase 1 produces no traces; there is nothing to interpret. Invoking it here would waste the mapping/formal budget on a case we have not yet decided is worth profiling.

**Outputs**
- `datasets/case{A,B,C,D}.jsonl` — canonical autobench datasets (feeds Phase 1 runs and carries into Phase 2/3 unchanged)
- `experiments/phase1/case{A,B,C,D}_{sglang,vllm}.json` — raw bench_serving output
- `experiments/phase1/summary.md` — 4×2 table of (TTFT p50, TTFT p95, TPOT, output tok/s, request throughput), with per-cell fairness notes
- `logs/phase1_*.log` — level-1 kernel API trails (only consulted if a run crashed)

**Downstream flow.** `summary.md` is the sole input to the Phase 2 decision rule below. The JSONL datasets are reused verbatim in Phase 2 shaping (no re-generation — that would break byte-identity).

**Success criteria**
- 8 runs complete, all within 5% run-to-run stability.
- Table shows a clear per-case picture of "who wins, by how much, on what dimension".

**Risks**
- Graph capture time dominates first run → fix with longer warmup.
- One framework OOMs earlier than the other → reduce max seq or batch size *equally* for both and document.
- Ratio is dominated by tokenizer / request framing overhead → Case A will catch this (TTFT is tiny, overhead is visible).

---

## Phase 2 — Find the Informative Cases (0.5–1 day)

**Goal**: From the 4 Phase-1 cases, choose the 1–2 cases that best expose a *structural* difference. Shape one SGLang-side exploration to confirm the gap isn't trivially a flag-tuning issue.

**Decision rule**
- **Case gap < 5%** → Not informative. Drop or reshape (e.g., push concurrency higher, push prompt len longer).
- **Case gap 5–15%** → Promising but could be config. Run the shaping sweep below before profiling.
- **Case gap > 15% and stable** → Interesting. Go straight to Phase 3 profiling.

**Shaping sweep (SGLang only, with `sglang-auto-benchmark`)**
This is where the skill earns its place — not to compare frameworks, but to rule out "SGLang loses because we picked the wrong flags".

Use **tier 1** (sanity, ~3–5 candidates, single QPS) so we do not burn time. Target 2 axes only:

```yaml
# configs/phase2_shaping.yaml
benchmark:
  model_path: Qwen/Qwen3-VL-8B-Instruct
  dataset_path: datasets/{picked_case}.jsonl
  scenario: shaping
search:
  tier: 1
  max_candidates: 4
server_flags:
  base:
    chunked_prefill_size: [2048, 4096]
    schedule_conservativeness: [0.8, 1.0]
load:
  qps_plan: fixed
  qps_values: [4.0]         # or whatever the Phase 1 saturating QPS was
  max_concurrency: [16]
sla:
  max_ttft_ms: 2000.0
  max_tpot_ms: 100.0
output:
  results_dir: experiments/phase2_shaping
```

Run:
```
python -m sglang.auto_benchmark run --config configs/phase2_shaping.yaml --backend sglang-oai
```

If the best candidate closes the gap to vLLM within 5%, the gap was *configurational*, not structural — document and move on to a harder case. If the gap persists, we have a real target for profiling.

**Skill usage in this phase**
- `sglang-auto-benchmark` → `run` at **tier 1**, bounded to 1 case and ≤2 flag axes. This is the first time the skill earns its place: we need a mechanized, resumable way to ask "is SGLang already at its best on this case?" Tier 1 is non-negotiable here — tier 2/3 would burn time searching a space we have not justified yet.
- `debug-cuda-crash` → still `LOGLEVEL=1` during the sweep (the skill spins up a fresh server per candidate; any of them may crash on flag combinations that interact badly with `Qwen3-VL-8B`).
- `sglang-torch-profiler-analysis` → **still not used.** We are eliminating configurational explanations; a trace cannot tell us whether the *other* flag combination would have been faster.

**Outputs**
- `experiments/phase2_shaping/{live_results.jsonl, results.jsonl, results.csv, summary.md}` — produced by the skill
- `experiments/phase2/selected_cases.md` — the 1–2 cases chosen for profiling, with explicit reason and the specific phenomenon to explain

**Success criteria**: We have a ≤2-case shortlist, each labeled with the specific phenomenon we want to explain (e.g., "TTFT 1.8× vLLM on long prefill at concurrency 16").

**Downstream flow.** `selected_cases.md` is the sole gate for Phase 3 — no case enters profiling without surviving this step. If shaping closed every gap, Phase 3 does not happen; we go back to Phase 1 and reshape the workload.

---

## Phase 3 — Profiling & Trace Collection (1–1.5 days)

**Goal**: For each selected case, collect a *pair* of traces (mapping + formal) on SGLang and enough evidence on vLLM to cross-check. Do not analyze yet; just produce clean artifacts.

**Why mapping + formal**: The `sglang-torch-profiler-analysis` skill's strongest output (Triage — Overlap Opportunity Table + Fuse Opportunity Table) requires both. Mapping (graph-off) recovers kernel → CPU op → Python scope mapping. Formal (graph-on) reflects real serving behavior.

**Collection sequence per case**

1. **Mapping trace (graph-off, SGLang)**
   ```
   python3 -m sglang.launch_server --model-path Qwen/Qwen3-VL-8B-Instruct \
     --tp 1 --disable-cuda-graph --attention-backend flashinfer
   # in another shell:
   python3 -m sglang.profiler \
     --url http://127.0.0.1:30000 --num-steps 8 \
     --profile-by-stage \
     --output-dir /data/profiling_lab/traces/{case}/sglang_mapping
   ```
   Separate outputs for `EXTEND` and `DECODE` are produced automatically by `--profile-by-stage`.

2. **Formal trace (graph-on, SGLang)**
   Re-launch with graph capture enabled, warm up, then profile again → `/data/profiling_lab/traces/{case}/sglang_formal/`.

3. **vLLM traces (best effort)**
   vLLM supports torch profiler via `VLLM_TORCH_PROFILER_DIR=... `. Collect one trace per stage equivalent if possible. If vLLM stage separation is not clean, collect a single full trace and note it.
   ```
   VLLM_TORCH_PROFILER_DIR=/data/profiling_lab/traces/{case}/vllm \
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server ...
   # trigger profile via API
   ```

4. **Crash safety**: keep `SGLANG_KERNEL_API_LOGLEVEL=1` + `SGLANG_KERNEL_API_LOGDEST=/data/profiling_lab/logs/sglang_phase3_%i.log` on during all SGLang runs. Cost is tiny; if anything crashes we already have the boundary log.

**Skill usage in this phase**
- `sglang-torch-profiler-analysis` → used only as a *collection driver*. Its script can drive `sglang.profiler` against a live server (`--url ... --profile-by-stage`) which is the recommended path here because it ensures stage separation is done the same way the triage step will later expect. **No `triage` is run yet** — Phase 3 only produces clean artifacts; interpretation is Phase 4.
- `debug-cuda-crash` → `LOGLEVEL=1` throughout. CUDA graph capture will skip level-5/10 features automatically, but level 1 still works and is exactly what we want: a boundary trail without perturbing the trace. If a specific case crashes mid-collection, we escalate *only that case* to level 3 and re-run (see Crash/Debug Workflow).
- `sglang-auto-benchmark` → **not used.** The case is already locked; flag search here would mean profiling a moving target.

**Outputs** (per case)
- `traces/{case}/sglang_mapping/*.trace.json.gz` × 2 (extend, decode)
- `traces/{case}/sglang_formal/*.trace.json.gz` × 2
- `traces/{case}/vllm/*.trace.json.gz` (best effort)
- `traces/{case}/collection_notes.md` — warmup protocol, number of iterations captured, any anomalies
- `logs/phase3_*.log` — level-1 boundary trails from both mapping and formal collection

**Success criteria**
- Each SGLang trace covers ≥5 steady-state iterations of its stage.
- Trace files are 20–500 MB; if >1 GB, re-collect with fewer iterations.
- No crash during collection (or, if there was one, we have the level-1 log pointing at the last API boundary).

**Risks**
- CUDA graph capture interferes with profile start → fix by warming up fully before toggling profile start.
- Profile size explodes → reduce `num-steps` or disable CPU-side annotations we don't need.
- vLLM profiler emits a different format → rely on vLLM's own tool for initial view, then extract comparable numbers.

**Downstream flow.** The trace pair per (case × stage) is the only input Phase 4 needs. Do not leave Phase 3 without at least one complete mapping+formal pair per selected case — Phase 4 two-trace triage has no meaningful fallback.

---

## Phase 4 — Trace Interpretation & Synthesis (1–2 days)

**Goal**: Convert raw traces into ranked, evidence-backed hypotheses for improving SGLang.

**Skill usage in this phase**
- `sglang-torch-profiler-analysis` → the workhorse here. Two-trace `triage` per (case × stage) on SGLang, single-trace `triage` on the best-effort vLLM traces for cross-check. The skill's catalog lookup (`fuse-overlap-catalog.md`, `overlap-catalog.md`) is a **mandatory step before any hypothesis is written** — a finding that already appears in the catalog gets classified as "existing path / disabled or regressed" (high-confidence, low-cost), not as a novel recommendation.
- `debug-cuda-crash` → consulted *only if* a trace shows unexplained tensor values or the Phase 3 run crashed. In that case, re-run the failing step at `LOGLEVEL=5` to inspect tensor stats at the kernel boundary; otherwise untouched.
- `sglang-auto-benchmark` → not used in Phase 4. Interpretation is not a sweep.

**Step 1: Triage each case**
```
python analyze_sglang_torch_profile.py triage \
  --mapping-trace traces/{case}/sglang_mapping/*DECODE*.trace.json.gz \
  --formal-trace traces/{case}/sglang_formal/*DECODE*.trace.json.gz \
  --model Qwen/Qwen3-VL-8B-Instruct \
  > analysis/{case}/decode_triage.md
```
Repeat for EXTEND. Each triage produces:
- Kernel Table — where time goes, with Python source mapping
- Overlap Opportunity Table — high-priority kernels with overlap headroom vs already-covered ones
- Fuse Opportunity Table — patterns that match known SGLang fuse/overlap paths that may be absent

**Step 2: Category-level breakdown per case**
```
python analyze_sglang_torch_profile.py breakdown \
  --trace traces/{case}/sglang_formal/*DECODE*.trace.json.gz \
  --categories attention,gemm,communication,norm,quantization,memory,scheduler
```
This tells us whether we are compute-, memory-, comm-, or scheduler-bound — the first-order diagnosis.

**Step 3: Cross-check against vLLM**
For each top SGLang hotspot, answer:
- Does vLLM have a kernel doing the same work? At roughly what time fraction?
- Does vLLM overlap this work with something else that SGLang does serially?
- Does vLLM fuse this with a neighboring op that SGLang keeps separate?

**Step 4: Hypothesis construction**
One hypothesis per finding, in standard form:

```
**Hypothesis**: [short title]
- Observation: [kernel X in SGLang takes Y% of decode time, and is serialized after Z; vLLM appears to overlap it or omit it]
- Impact: [estimated latency/throughput improvement if closed]
- Evidence: [pointer to specific lines in triage tables + breakdown]
- Confidence: [H/M/L — H if also supported by vLLM cross-check; M if only SGLang trace; L if speculative]
- Next step: [what experiment or code change would validate this]
```

**Outputs**
- `analysis/{case}/decode_triage.md`, `analysis/{case}/extend_triage.md`
- `analysis/{case}/breakdown.md`
- `analysis/{case}/vllm_crosscheck.md`
- `analysis/hypotheses.md` — all hypotheses across cases, structured, de-duplicated
- `analysis/ranked_recommendations.md` — top 5–10 sorted by `confidence × impact × feasibility`

**Success criteria**
- Each hypothesis references a specific kernel name AND a Python source location.
- Each hypothesis cites either a catalog entry ("existing path X, disabled/regressed here") or an explicit "no catalog match, similarity: {high|medium|low}" note.
- At least one H-confidence hypothesis per selected case.
- Ranking is explicit (not just a list).

**Downstream flow.** `ranked_recommendations.md` is the input to Phase 5: only the top 2–3 entries, and only if they need validation beyond the catalog cross-check, enter Phase 5.

---

## Phase 5 — Hypothesis Validation Sweeps (optional, 1 day per hypothesis)

**Goal**: For the top 2–3 hypotheses, run a controlled SGLang sweep that either confirms or refutes the mechanism, *before* anyone writes a PR.

**Skill usage in this phase**
- `sglang-auto-benchmark` → `run` at **tier 2** (~10 candidates), resumable. This is the skill's strongest form: a hypothesis-driven sweep on exactly the flags the Phase 4 hypothesis names, using the Phase 1 dataset so results are directly comparable to the baseline. Resumability earns its keep here because a tier-2 sweep can run for hours.
- `sglang-torch-profiler-analysis` → optionally re-invoked on the winning candidate if the sweep confirms improvement — a second triage on the better config confirms that the mechanism we hypothesized is the one that moved. This closes the loop: hypothesis → metric delta → trace-level confirmation.
- `debug-cuda-crash` → `LOGLEVEL=1` during the sweep, escalated only on candidate failure.

For a hypothesis like "SGLang's all-reduce is not overlapping with the following norm → use tier-2 sweep over the relevant overlap flag":

```yaml
search:
  tier: 2
  max_candidates: 10
server_flags:
  base:
    enable_custom_all_reduce: [true, false]
    enable_torch_compile: [true, false]
  ...
```

Resume-on-failure is valuable here because sweeps can run for hours.

**Outputs per hypothesis**
- `experiments/phase5/{hypothesis}/results.jsonl|csv|summary.md`
- `analysis/hypotheses_validated.md` — updated confidence ratings

---

## Crash / Debug Workflow (transverse, not a phase)

`debug-cuda-crash` is *always on at level 1* during Phase 1 and Phase 3 SGLang runs, because the overhead is trivial and the payoff on a crash is large. Escalation:

| Situation | Setting | Purpose |
|---|---|---|
| Normal runs (baseline / profiling) | `LOGLEVEL=1`, `LOGDEST=logs/%i.log` | Passive boundary trail |
| A crash happens | Re-run with `LOGLEVEL=3` | Inspect inputs at crash boundary |
| Numerical suspicion (NaN/Inf in trace, diverging outputs) | `LOGLEVEL=5` | Tensor min/max/mean + NaN/Inf counts |
| Need offline reproducer | `LOGLEVEL=10` + `DUMP_DIR` + `DUMP_INCLUDE='sglang.custom_op.*'` | Crash-safe input dump |

**Interaction with CUDA graphs**: dumps are auto-skipped during graph capture. Temporarily pass `--disable-cuda-graph` when we want level-10 dumps from a normal serving path.

**Rule**: a crash does not cause us to abandon the current case. We (a) capture enough to reproduce offline, (b) isolate the trigger (usually a specific batch shape), (c) return to the main plan and either mark the case as blocked or route around it.

---

## Skill Usage Quick-Reference

Authoritative definitions live in the **Skill Usage Model** section above. This is a one-glance summary for use while executing — if it disagrees with Skill Usage Model, the model wins.

| Phase | auto-benchmark | profiler-analysis | debug-cuda-crash |
|---|---|---|---|
| Phase 0 | — | — | L1 during server smoke |
| Phase 1 | `convert` + `validate` (dataset only) | — | L1 passive |
| Phase 2 | `run` tier 1, ≤4 candidates, 1 case | — | L1 passive |
| Phase 3 | — | collection driver (`sglang.profiler --profile-by-stage`), no triage | L1 passive; escalate on crash only |
| Phase 4 | — | `triage` two-trace (SGLang), single-trace (vLLM) + **catalog lookup** | L5 only if NaN/Inf suspected in a trace |
| Phase 5 | `run` tier 2, resumable, hypothesis-scoped | optional re-triage on winning config | L1 passive |

Never invert the row: e.g. do not run `auto-benchmark` in Phase 3, do not run `triage` in Phase 2, do not run `debug-cuda-crash` at L≥3 inside a measured run.

---

## Decision Gates

| Condition | Action |
|---|---|
| Phase 1 all 4 gaps < 5% | Abandon current workload shape. Push prompt length and concurrency before concluding "no gap". |
| Phase 1 gap exists but run-to-run variance > 5% | Increase warmup, pin GPU governor if possible, pin NUMA, rerun. Do not profile on noisy data. |
| Phase 2 shaping closes the gap | The gap was configurational. Document, pick a harder case, re-enter Phase 2. |
| Phase 3 SGLang crashes on chosen case | Switch to debug-cuda-crash flow. If crash is data-specific (e.g., one prompt), drop that prompt and keep the case. If it is structural, treat as a separate finding. |
| Phase 4 top hotspot is communication-bound on TP=1 | This is suspicious (single-GPU should have no NCCL traffic). Check attention backend and tensor dispatch paths; may indicate a fake "comm" kernel or CPU-side dispatch overhead. |
| Phase 4 top hotspot is scheduler-bound | GPU-only profiling is insufficient. Add `py-spy` or torch CPU-side profiler on the scheduler process, and instrument scheduler phases. |
| Phase 4 top hotspot maps to a kernel SGLang *already* has a fuse path for | The implementation may be gated off. Check feature flags; this is a low-cost, high-confidence recommendation. |
| Phase 4 yields no H-confidence hypothesis | Do *not* produce speculative recommendations. Expand Phase 3 (more iterations, more cases) before synthesizing. |

---

## Deliverables (final)

1. `experiments/env_snapshot.md` — fully reproducible environment state
2. `experiments/phase1/summary.md` — 4×2 baseline table + fairness notes
3. `experiments/phase2/selected_cases.md` — the shortlist, with justification
4. `traces/{case}/...` — mapping + formal traces per selected case
5. `analysis/{case}/{decode,extend}_triage.md`, `breakdown.md`, `vllm_crosscheck.md`
6. `analysis/hypotheses.md` — structured hypotheses
7. `analysis/ranked_recommendations.md` — top 5–10 for SGLang, ordered by confidence × impact × feasibility
8. (Optional) `experiments/phase5/{hypothesis}/...` — validation sweeps

---

## Anti-Patterns

- ❌ "vLLM is faster overall" with no explanation of why
- ❌ Attributing a slow kernel to a design flaw before checking the implementation
- ❌ Recommending something already in SGLang
- ❌ Confusing benchmark noise with real differences
- ❌ Proposing large refactors without evidence they help
- ❌ Producing speculative recommendations when Phase 4 yields no H-confidence hypothesis

---

## Prioritized Next-Step Checklist

1. Verify both SGLang and vLLM can serve `Qwen3-VL-8B-Instruct` (Phase 0, ≤2 h).
2. Write the 4 `autobench` dataset JSONLs for cases A–D (Phase 1 prep, ≤1 h).
3. Run Phase 1 (8 runs × 3 repeats, with passive level-1 crash logging), produce `summary.md` (half–1 day).
4. Apply Phase 2 decision rule; run tier-1 shaping sweep only on cases with 5–15% gap; pick the shortlist (half day).
5. For each shortlisted case: collect mapping + formal traces on SGLang, best-effort trace on vLLM (1 day).
6. Triage + breakdown per case, then cross-check against vLLM (1 day).
7. Write `hypotheses.md` and `ranked_recommendations.md` (half day).
8. Only if time permits: Phase 5 validation sweeps for the top 2 hypotheses.

**End condition**: the ranked recommendations file exists and each of its top 3 entries is concrete enough that an SGLang engineer could start an implementation PR without further investigation.
