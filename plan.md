# SGLang vs vLLM Profiling — Execution Plan

## 1. Objective

Profile and compare SGLang against vLLM under fair, controlled conditions to extract actionable optimization insights **for SGLang**. The goal is not a generic benchmark summary — it is to explain *why* gaps exist and *what SGLang can do about them*. vLLM is used as a strong reference system whose behavior can falsify or corroborate SGLang-side hypotheses.

## 2. Core Questions

1. Where is the performance gap? (TTFT, TPOT, throughput)
2. Which stage is responsible: prefill/extend or decode?
3. Which subsystem is responsible: kernels, communication, scheduling, memory?
4. Which vLLM behaviors are real design wins vs noise or configuration artifacts?
5. Which of those wins are actionable for SGLang (not already shipped, not already in-flight)?

## 3. Executive Summary

We compare SGLang and vLLM on `Qwen/Qwen3-VL-8B-Instruct` (text-only path first) at TP=1 on a single H200. The work is staged to maximize signal per GPU-hour: establish one small, clean baseline table; find the 1–2 cases whose gap is structural rather than configurational; profile those cases deeply; cross-check findings against vLLM traces; publish ranked recommendations with evidence pointers.

The workflow follows a strict pipeline:

```
 benchmark  →  gap identification  →  config shaping  →  profiling  →  interpretation  →  hypothesis  →  validation
  (Phase 1)       (Phase 1/2)         (Phase 2)       (Phase 3)      (Phase 4)        (Phase 4)     (Phase 5)
```

Three skills each own exactly one layer of this pipeline:

- **`sglang-auto-benchmark`** — controlled-experimentation engine (inputs: flags, datasets, QPS; outputs: metrics).
- **`sglang-torch-profiler-analysis`** — trace interpreter (inputs: traces; outputs: kernel/overlap/fuse tables with catalog cross-reference).
- **`debug-cuda-crash`** — evidence-preservation safety net (passive at `LOGLEVEL=1`; escalated only on specific failure).

Decision rule at every phase boundary: if the gap on a case is <5% we *reshape the workload* before profiling; if it is large and stable we shape SGLang-side configs before concluding it is structural; we only profile cases that survived shaping.

## 4. Models

- **Phase 0–5 (primary)**: `Qwen/Qwen3-VL-8B-Instruct` (≈17 GB, cached at `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`).
- **Text-only fallback** if Qwen3-VL is not supported on either framework at pinned versions: `Qwen/Qwen3-8B`.
- **Later (Phase 6+, TBD)**: a larger Qwen3-VL variant.

## 5. Environment

- Host `radixark02`, container `sglang-bowenw`.
- GPUs: 4× H200 144 GB, CUDA 12.9.
- **SGLang**: dev editable at `/sgl-workspace/sglang`, Python 3.12, torch 2.9.1+cu129.
- **vLLM**: 0.19.0 in conda env `vllm` at `/opt/miniconda3/envs/vllm`, torch 2.10.0+cu128.
- HF cache: `/root/.cache/huggingface`.

---

## 6. Fairness Control Model

Every dimension falls into exactly one tier. Conclusions are only admissible when they are consistent with the tier of the variable they depend on.

### 6.1 Controlled variables — pinned before Phase 1 runs, or the run is invalid

| Dimension | Pinned value | Enforcement |
|---|---|---|
| GPU | Single H200, GPU 0 | `CUDA_VISIBLE_DEVICES=0` on every server + every client |
| Model weights | `Qwen/Qwen3-VL-8B-Instruct` at a fixed HF revision | Record commit-sha of HF snapshot in `env_snapshot.md`; re-verify hash before Phase 3 |
| Tokenizer | Same HF snapshot as weights | Byte-equality check in Phase 0 |
| Dtype | BF16 | Pass `--dtype bfloat16` explicitly on both servers |
| TP / DP / PP | 1 / 1 / 1 | No NCCL on single-GPU baseline |
| Max seq len | 8192 | Pin on both sides |
| Prompt set | One `autobench` JSONL per case, byte-identical | SHA-256 of dataset logged in every run |
| Sampler | temperature=0, top_p=1, greedy | Pass explicitly in client; do not rely on server defaults |
| Output length | Fixed `max_tokens` + `ignore_eos=true` | Prevents early-termination drift between tokenizers |
| Concurrency | {1, 16} per case | Same `--max-concurrency` on both |
| Warmup | ≥30 requests at target concurrency, discarded | Identical warmup script on both |
| Prefix cache | OFF in benchmark runs | Randomized prefixes in the autobench generator |
| Seed | Fixed per case | Both client and dataset generator |

If any of these cannot be pinned on one side, the run is aborted and the mismatch is fixed before Phase 1 begins.

### 6.2 Measured-and-reported variables — cannot be pinned identically, must be logged on every run

| Dimension | Action | Where it goes |
|---|---|---|
| torch / CUDA / FlashInfer / FA version | Record at server startup | `env_snapshot.md`, each run's `meta.json` |
| Attention backend chosen at runtime | Pin where possible (`--attention-backend flashinfer` on SGLang); log actual choice on vLLM | per-run `meta.json` |
| Kernel dispatch (e.g. Triton vs CUDA path) | Observe from trace | Phase 4 triage footer |
| Chunked-prefill chunk size | Log server-reported default | per-run `meta.json` |
| Idle GPU memory after model load | Record once per server startup | `env_snapshot.md` |

Conclusions that depend on these variables must be re-confirmed if the variable changes (e.g. on an SGLang upgrade).

### 6.3 Framework-intrinsic variables — *not* aligned; differences are the observation

| Dimension | Why we don't align it |
|---|---|
| Scheduler policy (SGLang radix vs vLLM cache manager) | This *is* one of the things we are comparing |
| CUDA graph shape selection heuristics | Design difference |
| Chunked-prefill scheduling policy | Design difference |
| Sampler kernel implementation | Implementation difference — relevant if it shows in the trace |

Findings that rest on framework-intrinsic variables are valid *design observations* and go directly into hypotheses. They do not need re-confirmation across runs.

### 6.4 Interpretation rule

> A hypothesis is **admissible** only if every variable it depends on is either *controlled* (pinned) or *framework-intrinsic* (labeled as such in the hypothesis). A hypothesis that depends on a *measured-and-reported* variable carries a confidence ceiling of M until a version-matched re-run confirms it.

---

## 7. Skill Usage Model

Authoritative role definitions. The Quick-Reference in §13 is a summary; this section wins on any conflict.

### 7.1 `sglang-auto-benchmark` — controlled experimentation on SGLang

**Solves.** Manual flag sweeps are combinatorial, error-prone, non-resumable. This skill takes a YAML spec of candidate flags × QPS × concurrency × SLA, runs each candidate with a fresh server, tracks SLA pass/fail, and writes resumable results. It also owns the canonical autobench JSONL format (`convert` / `validate`).

**Where used.**

| Phase | Subcommand | Purpose |
|---|---|---|
| Phase 1 prep | `convert` + `validate` | Produce the shared autobench JSONL consumed by both `bench_serving --backend sglang-oai` and `bench_serving --backend vllm`. Byte-identity is non-negotiable. |
| Phase 2 | `run` tier 1, ≤4 candidates, scoped to 1 case × ≤2 axes | Rule out "SGLang loses because a flag was wrong". Gate into Phase 3. |
| Phase 5 | `run` tier 2, ≤10 candidates, resumable | Validate specific Phase-4 hypotheses on the exact flag the hypothesis names. |

**Not used for.** Cross-framework comparison (cannot drive vLLM). Broad tier-3 discovery (we do not sweep a space we have not justified). Any interpretation of kernels.

### 7.2 `sglang-torch-profiler-analysis` — trace interpretation

**Solves.** Raw traces are illegible. The skill produces three tables (kernel / overlap / fuse), catalog-checked against `fuse-overlap-catalog.md` and `overlap-catalog.md` so findings are correctly classified as *existing path (disabled/regressed)*, *in-flight PR*, or *truly new* with similarity label.

**Single-trace vs two-trace.** Single-trace is enough for kernel-share and fuse candidates. Two-trace (mapping graph-off + formal graph-on) is required before any overlap claim — graph-off carries the `kernel → cpu_op → python_scope` mapping that graph-on has collapsed.

**Where used.**

| Phase | Flow | Purpose |
|---|---|---|
| Phase 3 | Collection driver only — `sglang.profiler --profile-by-stage` via the skill's script. No triage yet. | Ensures stage separation is shaped the way Phase-4 triage expects. |
| Phase 4 | `triage` two-trace on SGLang per (case × stage); `triage` single-trace on vLLM per (case × window) | Primary interpretive artifact of the project. |
| Phase 4 | Catalog lookup inside each triage (mandatory gate before any hypothesis) | Prevents recommending things SGLang already ships. |
| Phase 5 (optional) | `triage` on winning Phase-5 candidate | Confirms the hypothesized mechanism is the one that moved. |

**Not used for.** Phase 1–2 (no locked case yet). Merged-rank traces (skill prefers rank-local TP-0). Any hypothesis without going through the catalog step.

### 7.3 `debug-cuda-crash` — evidence preservation

**Solves.** CUDA crashes destroy evidence. The `@debug_kernel_api` decorator logs boundary-level input metadata *before* each call so the evidence survives the crash.

**Cost ladder.** L1: names only, near-zero cost. L3: shapes/dtypes, small I/O cost. L5: tensor stats, requires host sync (perturbs timing). L10: full input dumps, disk + real perturbation. Benchmark and profile runs tolerate only L1.

**Where used.**

| Situation | Setting |
|---|---|
| All Phase 1, Phase 2, Phase 3, Phase 5 SGLang runs | `LOGLEVEL=1`, `LOGDEST=logs/{phase}/sglang_%i.log` |
| Crash occurs | Re-run failing case at `LOGLEVEL=3` |
| NaN/Inf suspected in a trace or output divergence appeared in Phase 0 | Targeted `LOGLEVEL=5` reproducer |
| Need offline reproducer | `LOGLEVEL=10` + `DUMP_DIR` + `DUMP_INCLUDE='sglang.custom_op.*'` + `--disable-cuda-graph` |

**Not used for.** Performance analysis. vLLM diagnosis (decorator only instruments SGLang). Running at L≥3 inside any measured run.

### 7.4 Complementarity (one line)

Auto-benchmark controls *inputs*; profiler-analysis interprets *outputs*; debug-cuda-crash preserves *evidence when either fails*. Reaching for the wrong one is the anti-pattern.

---

## 8. Artifact Framework

### 8.1 Filesystem layout

**Directory purpose rule:** `logs/` = infrastructure side-effects (server stderr, kernel-API boundary trails) — consult on failure, never cited in analysis. `experiments/` = research artifacts deliberately produced by the experiment protocol — cited in analysis and reports.

```
/data/profiling_lab/
├── plan.md                          this document
├── README.md                        layout + reviewer reading order
│
├── configs/                         YAML configs for auto_benchmark sweeps
│   ├── phase2_shaping/
│   └── phase5_validation/
│
├── datasets/                        canonical autobench JSONL (shared across frameworks)
│   ├── caseA_short.jsonl
│   ├── caseB_long_prefill.jsonl
│   ├── caseC_batched.jsonl
│   ├── caseD_decode_heavy.jsonl
│   └── README.md                    schema + generation command + SHA-256
│
├── logs/                            infrastructure side-effects (consult on failure)
│   ├── phase0/
│   │   ├── sglang_server.log        SGLang startup stderr (Phase 0)
│   │   ├── vllm_server.log          vLLM startup stderr (Phase 0)
│   │   ├── kernel_api_7067.log      L1 kernel-API boundary log PID 7067 (empty)
│   │   ├── kernel_api_7208.log      L1 kernel-API boundary log PID 7208 (16 MB, live run)
│   │   └── kernel_api_7209.log      L1 kernel-API boundary log PID 7209 (empty)
│   ├── phase1/
│   ├── phase2/
│   ├── phase3/
│   └── phase5/
│
├── experiments/                     research artifacts (cited in analysis)
│   ├── env_snapshot.md              global: versions, backends, memory — all phases
│   ├── phase0/
│   │   ├── equivalence.md           Tier A/B/C results — Phase-0 deliverable
│   │   ├── sglang_outputs.json      Tier-B greedy outputs from SGLang
│   │   ├── vllm_outputs.json        Tier-B greedy outputs from vLLM
│   │   └── scripts/                 reproducible scripts for Phase 0
│   │       ├── tier_a_tokenizer.py  tokenizer + vocab checks
│   │       ├── tier_b_sglang.py     query SGLang, save sglang_outputs.json
│   │       └── tier_b_vllm_compare.py  query vLLM, compare, exit 1 on fail
│   ├── phase1/
│   │   ├── raw/                     raw bench_serving JSON + per-run meta.json
│   │   └── summary.md               4×2 baseline table
│   ├── phase2_shaping/              auto_benchmark output dir
│   ├── phase2/
│   │   └── selected_cases.md        Phase-3 entry gate
│   └── phase5/
│       └── {hypothesis}/
│
├── traces/                          raw torch profiler artifacts
│   └── {case}/
│       ├── sglang_mapping/          graph-off, --profile-by-stage (EXTEND / DECODE)
│       ├── sglang_formal/           graph-on,  --profile-by-stage (EXTEND / DECODE)
│       ├── vllm/
│       │   ├── prefill_like/        concurrency=1 long-prompt window
│       │   └── decode_like/         concurrency=16 steady-state window
│       └── collection_notes.md
│
├── analysis/                        interpretation layer (processed from traces)
│   ├── {case}/
│   │   ├── extend_triage.md
│   │   ├── decode_triage.md
│   │   ├── breakdown.md             category split: attn/gemm/comm/norm/quant/mem/sched
│   │   └── vllm_crosscheck.md       falsification / corroboration record
│   ├── category_regex.md            shared regex applied symmetrically to both frameworks
│   ├── vllm_source_map.md           curated kernel-name → vllm/ module path
│   ├── hypotheses.md                structured hypotheses, de-duplicated
│   └── ranked_recommendations.md    top 5–10, sorted by confidence × impact × feasibility
│
└── reports/                         final deliverables (human-facing)
    ├── 01_experiment_summary.md
    ├── 02_benchmark_table.md
    ├── 03_profiling_analysis.md
    ├── 04_hypotheses.md
    └── 05_recommendations.md
```

### 8.2 Artifact layers

| Layer | Contents | Mutability | Purged on rerun? |
|---|---|---|---|
| **Raw** | `datasets/`, `logs/`, `traces/`, `experiments/*/raw/`, `experiments/phase2_shaping/live_results.jsonl` | Append-only, never edited by hand | Never — raw evidence is the ground truth |
| **Processed** | `experiments/*/summary.md`, `analysis/**`, `experiments/phase2/selected_cases.md` | Regenerated from raw | Yes, on rerun of the source phase |
| **Deliverable** | `reports/**`, `plan.md`, `experiments/env_snapshot.md`, `experiments/phase0_equivalence.md` | Hand-edited, reviewed | No — edited in place |

### 8.3 Reviewer reading order

A human reviewer validating the project should inspect artifacts in this sequence. Stopping at any layer where confidence is lost is the expected behavior.

1. `reports/05_recommendations.md` — the claims
2. `reports/02_benchmark_table.md` — are the numbers plausible?
3. `analysis/ranked_recommendations.md` — do rankings track evidence?
4. `analysis/{case}/decode_triage.md`, `extend_triage.md` — do the top rows justify the hypothesis?
5. `analysis/{case}/vllm_crosscheck.md` — does vLLM evidence agree or falsify?
6. `traces/{case}/…` — only when challenging a specific row of a triage table

### 8.4 Inter-phase flow

| Produced in | Consumed by | As what |
|---|---|---|
| `datasets/case*.jsonl` | Phase 1, Phase 2, Phase 5 | Byte-identical workload |
| `experiments/phase1/summary.md` | Phase 2 | Input to the Decision Rule |
| `experiments/phase2/selected_cases.md` | Phase 3 | Sole gate into profiling |
| `traces/{case}/sglang_{mapping,formal}` | Phase 4 | Two-trace triage input |
| `traces/{case}/vllm/{prefill_like,decode_like}` | Phase 4 | Single-trace falsification |
| `analysis/hypotheses.md` | Phase 5 | Source of hypotheses to validate |
| `experiments/phase5/{h}/summary.md` | Phase 5 close-out | Updates confidence in `hypotheses.md` |

---

## 9. Phases

### Phase 0 — Environment & Functional Equivalence (≤1 day)

**Goal.** Establish that both servers are comparable — weights, tokenizer, vocab identical; decoding behavior equivalent under a realistic equivalence standard.

**Operational constants (this run).**
- GPU: `CUDA_VISIBLE_DEVICES=6` (H200, 139 GB free at run start)
- Model snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- `HF_HUB_OFFLINE=1` — model is fully cached; no network call or token needed
- Servers run **sequentially** (SGLang first, then vLLM after shutdown) so each gets full GPU memory and there is no cross-process interference during the equivalence test. Phase 1 benchmarks follow the same pattern.

**Actions.**

1. Launch SGLang (background, log to `logs/phase1/sglang_server_phase0.log`):
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   SGLANG_KERNEL_API_LOGLEVEL=1 \
   SGLANG_KERNEL_API_LOGDEST=logs/phase1/sglang_%i.log \
   python3 -m sglang.launch_server \
     --model-path <snapshot_path> \
     --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer
   ```
   Wait for `server is fired up` in log. Record FlashInfer version, chunked-prefill default, idle GPU memory.
2. Run equivalence tiers (Tier A tokenizer check + Tier B greedy outputs). Save outputs to `experiments/phase0_sglang_outputs.json`.
3. Kill SGLang. Launch vLLM:
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model <snapshot_path> --dtype bfloat16 \
     --port 30001 --tensor-parallel-size 1
   ```
   Wait for `Application startup complete`. Record attention backend, idle GPU memory.
4. Run same Tier B prompts against vLLM. Save to `experiments/phase0_vllm_outputs.json`. Compare.
5. Record all findings in `experiments/env_snapshot.md` and `experiments/phase0_equivalence.md`.

**Equivalence framework.** Byte-identical greedy decoding across frameworks is not a realistic target — attention kernel, matmul tiling, and reduction order all legitimately differ, and bf16 accumulation order compounds the divergence. The correct standard is tiered:

| Tier | Check | Threshold | Disposition if fail |
|---|---|---|---|
| **A — Blocker** | Tokenizer byte-equality on 5 probe strings (ASCII, CJK, emoji, code, long) | Exact match | **Stop.** The runs are incomparable. Fix before continuing. |
| **A — Blocker** | Model weight hash (SHA-256 of each `*.safetensors` file loaded by each server) | Identical | **Stop.** Wrong snapshot. |
| **A — Blocker** | Vocab size, EOS/BOS/PAD ids, `max_position_embeddings` | Identical | **Stop.** Config drift. |
| **A — Blocker** | Chat-template rendered bytes on a fixed system+user pair | Identical | **Stop.** Template mismatch makes all latency numbers misleading. |
| **B — Correctness** | Top-1 next-token on ≥3 greedy prompts (short / medium / long) | Match on all 3 | **Stop.** Different top-1 at token 0 ⇒ weights loaded differently, wrong dtype, or RoPE misconfig. |
| **B — Correctness** | Top-5 logprob overlap on first token, averaged over the 3 prompts | Jaccard ≥ 0.8 | **Investigate** before Phase 1 — likely sampler or normalization divergence. |
| **B — Correctness** | Coherent continuation at 256 output tokens under greedy sampling | Human-readable, on-topic, no degenerate loops | **Investigate.** Coherent but byte-divergent is acceptable. |
| **C — Informational** | Token-level edit distance of full 256-token continuations | Logged, not gated | Report in `phase0_equivalence.md` |
| **C — Informational** | Output length under `ignore_eos=false` | Logged | Report |

**Rule.** A Tier-A failure halts the plan. A Tier-B failure at the *first token* halts the plan. A Tier-B failure only at token ≥ 2 is expected bf16 drift and proceeds with a note in every downstream conclusion ("greedy outputs diverge after first token — cross-framework output comparisons below are semantic, not token-level").

**Downstream effect on profiling.** Because token-level output equivalence is not required, we do not gate Phase 3 on it. What *does* matter for profiling validity: Tier-A identity (so both frameworks execute the same underlying model) and workload byte-identity (§6.1). Profiling under these conditions is methodologically sound even if produced tokens differ.

**Outputs.** `experiments/env_snapshot.md`, `experiments/phase0_equivalence.md`.

**Risks.** Qwen3-VL may not be fully supported at pinned versions; fall back to `Qwen3-8B` and record the substitution. Vision tower may load even for text-only; record idle memory.

---

### Phase 1 — Minimal Fair Baseline (1 day)

**Goal.** Produce one head-to-head table on a small deliberate matrix — clean enough to believe.

**Case matrix.**

| Case | Prompt len | Output len | Concurrency |
|---|---|---|---|
| A. Latency-bound short | 128 | 128 | 1 |
| B. Latency-bound long-prefill | 2048 | 128 | 1 |
| C. Batched throughput | 512 | 128 | 16 |
| D. Decode-heavy | 512 | 512 | 16 |

4 cases × 2 frameworks = 8 runs. Each ≥120 s steady-state after warmup, repeated 3× with independent warmups; take median, reject if stdev/median > 5 %.

**Actions.**

1. Generate byte-identical dataset per case:
   ```
   python -m sglang.auto_benchmark convert \
     --input-format random --prompt-len 512 --output-len 128 \
     --num-prompts 400 --output datasets/caseC_batched.jsonl
   python -m sglang.auto_benchmark validate --input datasets/caseC_batched.jsonl
   ```
   Log SHA-256 of each JSONL.
2. Run the two frameworks against the same JSONL:
   ```
   python -m sglang.bench_serving --backend sglang-oai \
     --base-url http://127.0.0.1:30000 \
     --dataset-name autobench --dataset-path datasets/caseC_batched.jsonl \
     --max-concurrency 16 --ignore-eos --random-seed 1
   python -m sglang.bench_serving --backend vllm \
     --base-url http://127.0.0.1:30001 \
     --dataset-name autobench --dataset-path datasets/caseC_batched.jsonl \
     --max-concurrency 16 --ignore-eos --random-seed 1
   ```
3. SGLang side: `SGLANG_KERNEL_API_LOGLEVEL=1`, `SGLANG_KERNEL_API_LOGDEST=logs/phase1/sglang_%i.log`.
4. Write `experiments/phase1/raw/{case}_{framework}_{rep}.json` per run with meta.json (versions, attn backend, dataset sha, seed).

**Skill usage.**

- `sglang-auto-benchmark` → `convert`/`validate` only (not `run`; it cannot drive vLLM).
- `debug-cuda-crash` → L1 passive, SGLang side only.
- `sglang-torch-profiler-analysis` → **not used.**

**Outputs.** `datasets/case{A,B,C,D}.jsonl`, `experiments/phase1/raw/*.json`, `experiments/phase1/summary.md` (4×2 table: TTFT p50/p95, TPOT, output tok/s, request throughput, fairness notes), `logs/phase1/*.log`.

**Success criteria.** 8 runs complete; stdev/median ≤ 5 % across repeats; per-cell fairness notes written.

**Risks.** Graph-capture overhead in first run (fix with longer warmup); OOM on one side (reduce equally and document); overhead-dominated ratios in Case A (flag as overhead, not as framework gap).

---

### Phase 2 — Identify Informative Cases (0.5–1 day)

**Goal.** From the 4 baseline cases, select 1–2 cases whose gap is *structural*, not configurational.

**Decision rule (applied per case).**

| Gap size (median across repeats) | Action |
|---|---|
| < 5 % | Non-informative. Drop, or reshape workload (push prompt length / concurrency). |
| 5 – 15 % | Run shaping sweep below. Promote to Phase 3 only if sweep cannot close the gap. |
| > 15 %, stable | Promote directly to Phase 3. |

**Shaping sweep.** `sglang-auto-benchmark run --tier 1`, bounded to 1 case × ≤2 axes. Axes chosen from the case's likely bottleneck (e.g. for long-prefill cases: `chunked_prefill_size`, `schedule_conservativeness`). Config template in `configs/phase2_shaping/` uses the existing `datasets/{case}.jsonl` — no regeneration.

**Skill usage.**

- `sglang-auto-benchmark` → `run` tier 1, ≤4 candidates.
- `debug-cuda-crash` → L1 passive (candidate servers are transient, may crash on odd flag combos).
- `sglang-torch-profiler-analysis` → **not used.**

**Outputs.** `experiments/phase2_shaping/{live_results.jsonl, results.jsonl, results.csv, summary.md}`, `experiments/phase2/selected_cases.md`.

**Success criterion.** ≤2-case shortlist, each labeled with the specific phenomenon to explain (e.g. "TTFT 1.8× vLLM on long prefill at concurrency 16"). If every gap collapsed under shaping, Phase 3 does not execute — return to Phase 1 with a reshaped matrix.

---

### Phase 3 — Profiling & Trace Collection (1–1.5 days)

**Goal.** For each selected case, produce a clean SGLang mapping+formal trace pair *and* a vLLM trace pair shaped to permit stage-level comparison. No interpretation here.

#### 3.1 SGLang traces

Two-trace protocol:

- **Mapping (graph-off)**: launch with `--disable-cuda-graph --disable-piecewise-cuda-graph --attention-backend flashinfer`; drive via `sglang.profiler --url ... --num-steps 8 --profile-by-stage --output-dir traces/{case}/sglang_mapping`.
- **Formal (graph-on)**: re-launch with graph capture enabled; warm up to stable batch shape; profile `traces/{case}/sglang_formal`.

`--profile-by-stage` is non-optional — it is what lets Phase 4 triage separate EXTEND from DECODE.

#### 3.2 vLLM traces — strengthened protocol

vLLM's profiling does not emit a clean mapping/formal pair, but it does not need to be left as "best effort". The protocol below produces a falsifiable vLLM artifact per case.

1. **Enable the profiler at server start.** Launch with `VLLM_TORCH_PROFILER_DIR=traces/{case}/vllm`. vLLM exposes `/start_profile` and `/stop_profile` HTTP endpoints that open/close one profile window each.
2. **Stage separation by workload shaping.** vLLM has no `--profile-by-stage`; we get the same separation naturally by driving two distinct profile windows per case:
   - **`prefill_like/`** — window opened immediately before sending `N=8` requests of the case's prompt length at concurrency 1, closed immediately after the first token of the last request. The window is then dominated by prefill kernels because decoding contribution is one token per request.
   - **`decode_like/`** — warm the server to a stable steady-state batch at the case's target concurrency, open the window after ≥30 s of steady decoding, capture ~5 s, close.
3. **Category alignment.** Both frameworks' traces are classified by the same regex rules defined in `analysis/category_regex.md` — attention, gemm, communication, norm, quantization, memory, scheduler. Rules are authored once, applied symmetrically; any kernel name not covered is accumulated in an `uncategorized` bucket which must shrink to < 2 % of GPU time before a breakdown is published.
4. **Hotspot-to-source mapping.** vLLM does not yield Python source backing for kernels the way SGLang's mapping trace does. We compensate with a curated static map `analysis/vllm_source_map.md` populated incrementally: every time a vLLM kernel crosses the 1 % GPU-time share threshold in any triage, its name is added with a manually-verified path into `/opt/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/…`. This trades completeness for correctness — the map covers exactly the kernels we cite, nothing more.
5. **Role in reasoning — falsification, not symmetry.** vLLM traces are used to *test* SGLang-side claims, not to mirror them:
   - Claim of the form *"vLLM overlaps X with Y"* requires X and Y to appear on distinct CUDA streams in the vLLM trace. If they do not, the claim is downgraded from H to at most M.
   - Claim of the form *"vLLM omits kernel Z"* requires Z to be absent from the vLLM kernel table at ≥ 0.5 % share. Otherwise the claim is rejected.
   - Claim of the form *"SGLang kernel K is slower per call than vLLM's equivalent K'"* requires matching K and K' by category (via `category_regex.md`) and comparable invocation count; divergence in invocation count is itself a finding.

#### 3.3 Crash safety

All SGLang runs in Phase 3: `SGLANG_KERNEL_API_LOGLEVEL=1`, `LOGDEST=logs/phase3/sglang_%i.log`. On any crash, re-run only the affected step at L3 (or L10 with `--disable-cuda-graph` if offline repro needed). Do not abandon the case — isolate the trigger.

**Skill usage.**

- `sglang-torch-profiler-analysis` → collection driver only (script calls `sglang.profiler`). No triage.
- `debug-cuda-crash` → L1 passive, escalated only on actual crash.
- `sglang-auto-benchmark` → **not used.**

**Outputs.** `traces/{case}/sglang_mapping/`, `traces/{case}/sglang_formal/` (EXTEND + DECODE each), `traces/{case}/vllm/{prefill_like,decode_like}/`, `traces/{case}/collection_notes.md`, `logs/phase3/*.log`.

**Success criteria.** Each SGLang trace covers ≥5 steady-state iterations per stage. Each vLLM window captures ≥5 complete iterations of its target mode. Files between 20 MB and 500 MB; >1 GB → re-collect with fewer steps. No crash, or crash with L1 boundary log preserved.

---

### Phase 4 — Trace Interpretation & Synthesis (1–2 days)

**Goal.** Convert traces into ranked evidence-backed hypotheses.

**Skill usage.**

- `sglang-torch-profiler-analysis` → two-trace `triage` on SGLang per (case × {EXTEND, DECODE}); single-trace `triage` on vLLM per (case × {prefill_like, decode_like}); mandatory catalog lookup.
- `debug-cuda-crash` → consulted only if a trace reveals NaN/Inf — then L5 on the targeted reproducer.
- `sglang-auto-benchmark` → not used.

**Step 1 — SGLang triage.**
```
python analyze_sglang_torch_profile.py triage \
  --mapping-input traces/{case}/sglang_mapping \
  --formal-input traces/{case}/sglang_formal \
  > analysis/{case}/decode_triage.md
```
(repeat per stage).

**Step 2 — Category breakdown.** Apply `analysis/category_regex.md` to the formal trace; emit `analysis/{case}/breakdown.md` with the compute / memory / comm / scheduler split. The same regex is then applied to the vLLM traces.

**Step 3 — vLLM single-trace triage and falsification.** Single-trace triage on each vLLM window produces a kernel table with catalog-backed pattern matches where applicable (most patterns will not match — that is fine; we are using vLLM triage to probe SGLang claims, not to extract vLLM recommendations). Results written to `analysis/{case}/vllm_crosscheck.md`, organized by the SGLang hypothesis each row tests.

**Step 4 — Hypothesis construction.** Every hypothesis uses this schema:
```
**Hypothesis**: <short title>
- Observation: <kernel/stage, time share, Python source pointer>
- vLLM evidence: <corroborates | falsifies | silent>, pointer to vllm_crosscheck row
- Catalog status: <existing disabled path | in-flight PR | truly new, similarity H/M/L>
- Impact: <estimated latency or throughput delta if closed>
- Evidence: <triage row refs, breakdown refs>
- Confidence: <H | M | L>   (H requires vLLM corroboration AND catalog-backed classification OR a disabled-path finding)
- Fairness dependence: <Controlled | Measured | Framework-intrinsic>
- Next step: <validation sweep, code pointer, or PR draft>
```

A hypothesis missing any field is inadmissible. The `Fairness dependence` field determines the confidence ceiling per §6.4.

**Outputs.** Per case: `extend_triage.md`, `decode_triage.md`, `breakdown.md`, `vllm_crosscheck.md`. Global: `analysis/hypotheses.md`, `analysis/ranked_recommendations.md` (top 5–10, sorted by `confidence × impact × feasibility`).

**Success criteria.** Every hypothesis has specific kernel name + source pointer + vLLM evidence + catalog classification. ≥1 H-confidence hypothesis per selected case. Ranking logic explicit.

---

### Phase 5 — Hypothesis Validation Sweeps (optional, 1 day per hypothesis)

**Goal.** For the top 2–3 hypotheses that can be tested with flag-level changes, confirm or refute the mechanism before any PR.

**Skill usage.**

- `sglang-auto-benchmark` → `run` tier 2, ≤10 candidates, resumable. Dataset is the Phase-1 `datasets/{case}.jsonl` so results are directly comparable to the baseline.
- `sglang-torch-profiler-analysis` → optional re-triage on the winning candidate to confirm the mechanism, not just the metric, moved.
- `debug-cuda-crash` → L1 passive.

**Outputs.** Per hypothesis: `experiments/phase5/{hypothesis}/{live_results.jsonl, results.jsonl, summary.md}` and optional `traces/{case}/sglang_phase5_{hypothesis}/`. Updates `analysis/hypotheses.md` confidence column.

---

## 10. Crash / Debug Workflow (transverse)

| Situation | Setting | Rationale |
|---|---|---|
| Normal runs (Phase 1 / 2 / 3 / 5) | `LOGLEVEL=1`, `LOGDEST=logs/{phase}/sglang_%i.log` | Negligible cost; free crash trail |
| Crash observed | Re-run crashing case with `LOGLEVEL=3` | Shapes/dtypes/device at crash boundary |
| NaN/Inf in trace or Phase-0 divergence | `LOGLEVEL=5` on targeted reproducer | Tensor stats at boundary |
| Offline reproducer needed | `LOGLEVEL=10` + `DUMP_DIR` + `DUMP_INCLUDE='sglang.custom_op.*'` + `--disable-cuda-graph` | Crash-safe input dump |

**Rule.** A crash is a finding, not an abort. Capture, isolate the trigger batch shape, route around, continue.

---

## 11. Decision Gates

| Condition | Action |
|---|---|
| Phase 0 Tier-A fail | Halt. Fix weights / tokenizer / template before continuing. |
| Phase 0 Tier-B fail at first token | Halt. Likely weight-load or RoPE config issue. |
| Phase 0 Tier-B fail only at token ≥ 2 | Proceed; annotate downstream comparisons as semantic-level. |
| Phase 1 all 4 gaps < 5 % | Reshape workload; longer prompts / higher concurrency before concluding "no gap". |
| Phase 1 gap real but stdev/median > 5 % | Increase warmup, pin GPU governor, re-run. Never profile on noisy data. |
| Phase 2 shaping closes the gap | Gap was configurational. Document and pick a harder case. |
| Phase 3 SGLang crash on selected case | Debug-crash flow. Data-specific → drop the prompt; structural → treat as a separate finding. |
| Phase 4 top hotspot is comm-bound at TP=1 | Suspicious (no NCCL on 1 GPU). Check attention backend and dispatch paths — may be mislabeled CPU overhead. |
| Phase 4 top hotspot is scheduler-bound | GPU-only profiling is insufficient; add `py-spy` + CPU-side torch profiler on scheduler process. |
| Phase 4 hotspot maps to a fuse path SGLang already has | Likely gated off. Low-cost, high-confidence recommendation to flip the gate. |
| Phase 4 yields no H-confidence hypothesis | Do not synthesize speculative recommendations. Expand Phase 3 (more iterations, more cases). |
| vLLM evidence contradicts an SGLang claim | Downgrade hypothesis confidence; keep the raw observation as a Phase-5 candidate only if the mechanism can be isolated without the vLLM comparison. |

---

## 12. Deliverables

Ordered by reviewer reading priority.

1. `reports/05_recommendations.md` — top 5–10 actionable directions for SGLang, ordered by `confidence × impact × feasibility`.
2. `reports/04_hypotheses.md` — structured hypotheses with vLLM evidence + catalog status.
3. `reports/03_profiling_analysis.md` — per-case triage + breakdown synthesis.
4. `reports/02_benchmark_table.md` — Phase-1 4×2 table with fairness notes.
5. `reports/01_experiment_summary.md` — environment, versions, fairness tier assignments, equivalence result.
6. All backing `analysis/**` and `experiments/**` artifacts.
7. `traces/**` preserved for independent re-analysis.

**End condition.** `reports/05_recommendations.md` exists and each of its top 3 entries is concrete enough that an SGLang engineer can open a PR without further investigation.

---

## 13. Skill Usage Quick-Reference

Authoritative definitions in §7. If this table disagrees, §7 wins.

| Phase | auto-benchmark | profiler-analysis | debug-cuda-crash |
|---|---|---|---|
| 0 | — | — | L1 during server smoke |
| 1 | `convert` + `validate` | — | L1 passive |
| 2 | `run` tier 1, ≤4 candidates, 1 case | — | L1 passive |
| 3 | — | collection driver (`--profile-by-stage`); no triage | L1 passive |
| 4 | — | `triage` 2-trace (SGLang) + 1-trace (vLLM) + **catalog lookup** | L5 only if NaN/Inf suspected |
| 5 | `run` tier 2, resumable, hypothesis-scoped | optional re-triage on winner | L1 passive |

Never invert a row. Auto-benchmark does not read kernels; profiler-analysis does not choose flags; debug-cuda-crash does not explain slowdowns.

---

## 14. Anti-Patterns

- ❌ "vLLM is faster overall" with no mechanistic explanation.
- ❌ Attributing a slow kernel to a design flaw before checking the implementation (and the catalog).
- ❌ Recommending something SGLang already ships or has an in-flight PR for.
- ❌ Confusing benchmark noise with a real difference (stdev/median > 5 %).
- ❌ Proposing a refactor without a validation path.
- ❌ Publishing an H-confidence hypothesis whose fairness dependence is `Measured` and unvalidated.
- ❌ Using vLLM traces only to mirror SGLang findings rather than to falsify them.
- ❌ Token-level equivalence as a gate for cross-framework profiling.

---

## 15. Results

### Phase 0 — Environment & Functional Equivalence (completed 2026-04-17)

#### Run conditions
- GPU: H200 index 6, `CUDA_VISIBLE_DEVICES=6`
- `HF_HUB_OFFLINE=1`, direct snapshot path (no network)
- Servers run sequentially (SGLang first, then vLLM after full shutdown)

#### Produced files

| File | Layer | Contents |
|---|---|---|
| `experiments/env_snapshot.md` | Deliverable | Full version table (SGLang, vLLM, torch, FlashInfer, CUDA), attention backends, chunked-prefill defaults, idle GPU memory per framework, fairness tier assignments |
| `experiments/phase0/equivalence.md` | Deliverable | Tier A/B/C equivalence matrix with pass/fail per check, tokenizer probe table, greedy output comparison table, conclusion |
| `experiments/phase0/sglang_outputs.json` | Raw | 3 prompts + SGLang greedy responses (temperature=0, top_p=1, max_tokens=128) |
| `experiments/phase0/vllm_outputs.json` | Raw | 3 prompts + vLLM greedy responses (same settings) |
| `experiments/phase0/scripts/tier_a_tokenizer.py` | Script | Reproducible tokenizer + vocab check; run against any snapshot path |
| `experiments/phase0/scripts/tier_b_sglang.py` | Script | Queries SGLang port 30000, saves `sglang_outputs.json` |
| `experiments/phase0/scripts/tier_b_vllm_compare.py` | Script | Queries vLLM port 30001, saves `vllm_outputs.json`, compares against SGLang reference; exit 1 on first-token divergence |
| `logs/phase0/sglang_server.log` | Log | SGLang server startup stderr (42 KB); contains full `ServerArgs`, CUDA graph capture progress, ready message |
| `logs/phase0/vllm_server.log` | Log | vLLM server startup stderr (19 KB); contains engine config, attention backend selection, KV cache size, CUDA graph capture |
| `logs/phase0/kernel_api_7208.log` | Log | L1 kernel-API boundary trail from live SGLang run (16 MB); passively records last API boundary before any potential crash |
| `logs/phase0/kernel_api_7067.log` | Log | Empty — server process that was killed before serving |
| `logs/phase0/kernel_api_7209.log` | Log | Empty — server process that was killed before serving |

#### Equivalence results

| Tier | Check | Result |
|---|---|---|
| A | Tokenizer byte-equality (5 probes) | ✅ PASS |
| A | Model weights (same snapshot path) | ✅ PASS |
| A | Vocab size (151,643), EOS/BOS/PAD ids | ✅ PASS |
| A | Chat template (ChatML) | ✅ PASS |
| B | Top-1 first token on 3 greedy prompts | ✅ PASS |
| B | Full 128-token output | ✅ **EXACT MATCH** on all 3 prompts |
| B | Coherent continuation | ✅ PASS |

#### Key environment findings (carry into Phase 1+)

| Finding | Variable tier | Impact on conclusions |
|---|---|---|
| SGLang attention backend: **FlashInfer 0.6.7.post3** (text) + FA3 (multimodal) | Measured | Any Phase-4 attention-kernel difference has confidence ceiling M until backends are aligned |
| vLLM attention backend: **FlashAttention v3** (text + multimodal) | Measured | Same as above |
| torch version differs: SGLang 2.9.1+cu129 vs vLLM 2.10.0+cu128 | Measured | Log in every run meta.json; re-confirm if version changes |
| KV cache: SGLang ~102 GB vs vLLM ~105.9 GB | Measured | Not a practical constraint at Phase-1 concurrency (≤16); no fairness action needed |
| SGLang `chunked_prefill_size=8192`, `piecewise_cuda_graph=disabled` | Controlled (logged) | Pinned; record in Phase-1 meta.json |
| Both frameworks: **EXACT greedy output match** at temperature=0 | — | No downstream "semantic-level only" annotation needed |

---

## 16. Prioritized Next-Step Checklist

1. ✅ Create the filesystem layout from §8.1 (placeholder READMEs in each directory).
2. ✅ Phase 0 — servers up, equivalence matrix run. All Tier-A/B pass; outputs EXACT match.
3. Generate `datasets/case{A..D}.jsonl` via `auto_benchmark convert`, log SHA-256 (≤1 h).
4. Phase 1 — 8 runs × 3 reps with passive L1 crash logging; `experiments/phase1/summary.md` (½–1 day).
5. Apply Phase-2 decision rule; run tier-1 shaping on 5–15 % cases; produce `selected_cases.md` (½ day).
6. Phase 3 — SGLang mapping+formal + vLLM prefill_like+decode_like per selected case (1 day).
7. Phase 4 — triage + breakdown + vLLM cross-check per case; author `hypotheses.md` and `ranked_recommendations.md` (1–1½ days).
8. Phase 5 (if warranted) — tier-2 validation sweeps for the top 2 hypotheses.
9. Promote `analysis/**` into `reports/**` deliverables.
