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
| GPU | Single H200, GPU 6 | `CUDA_VISIBLE_DEVICES=6` on every server + every client |
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
| **Deliverable** | `reports/**`, `plan.md`, `experiments/env_snapshot.md`, `experiments/phase0/equivalence.md` | Hand-edited, reviewed | No — edited in place |

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

1. Launch SGLang (background, log to `logs/phase0/sglang_server.log`):
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   SGLANG_KERNEL_API_LOGLEVEL=1 \
   SGLANG_KERNEL_API_LOGDEST=logs/phase0/sglang_%i.log \
   python3 -m sglang.launch_server \
     --model-path <snapshot_path> \
     --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer
   ```
   Wait for `server is fired up` in log. Record FlashInfer version, chunked-prefill default, idle GPU memory.
2. Run equivalence tiers (Tier A tokenizer check + Tier B greedy outputs). Save outputs to `experiments/phase0/sglang_outputs.json`.
3. Kill SGLang. Launch vLLM:
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model <snapshot_path> --dtype bfloat16 \
     --port 30001 --tensor-parallel-size 1
   ```
   Wait for `Application startup complete`. Record attention backend, idle GPU memory.
4. Run same Tier B prompts against vLLM. Save to `experiments/phase0/vllm_outputs.json`. Compare.
5. Record all findings in `experiments/env_snapshot.md` and `experiments/phase0/equivalence.md`.

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

1. Generate byte-identical dataset per case (run once; repeat for each case changing `--prompt-len`, `--output-len`, `--num-prompts` per the matrix):
   ```
   python -m sglang.auto_benchmark convert \
     --input-format random --prompt-len 512 --output-len 128 \
     --num-prompts 400 --output datasets/caseC_batched.jsonl
   python -m sglang.auto_benchmark validate --input datasets/caseC_batched.jsonl
   sha256sum datasets/case*.jsonl >> experiments/phase1/raw/dataset_sha256.txt
   ```
   Log SHA-256 of each JSONL in `experiments/phase1/raw/dataset_sha256.txt`.

2. Launch servers sequentially (one at a time; do not co-run).

   SGLang:
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   SGLANG_KERNEL_API_LOGLEVEL=1 \
   SGLANG_KERNEL_API_LOGDEST=logs/phase1/sglang_%i.log \
   python3 -m sglang.launch_server \
     --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
     --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer \
     2>&1 | tee logs/phase1/sglang_server.log
   ```

   vLLM (after shutting down SGLang):
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
     --dtype bfloat16 --port 30001 --tensor-parallel-size 1 \
     2>&1 | tee logs/phase1/vllm_server.log
   ```

3. Run the two frameworks against the same JSONL (`ignore_eos` is the **default** in bench_serving — do not pass `--disable-ignore-eos` unless intentionally testing variable-length output):
   ```
   python -m sglang.bench_serving --backend sglang-oai \
     --base-url http://127.0.0.1:30000 \
     --dataset-name autobench --dataset-path datasets/caseC_batched.jsonl \
     --max-concurrency 16 --seed 1 --warmup-requests 30
   python -m sglang.bench_serving --backend vllm \
     --base-url http://127.0.0.1:30001 \
     --dataset-name autobench --dataset-path datasets/caseC_batched.jsonl \
     --max-concurrency 16 --seed 1 --warmup-requests 30
   ```
4. Write `experiments/phase1/raw/{case}_{framework}_{rep}.json` per run with `meta.json` (versions, attn backend, dataset sha, seed).

**Skill usage.**

- `sglang-auto-benchmark` → `convert`/`validate` only (not `run`; it cannot drive vLLM).
- `debug-cuda-crash` → L1 passive, SGLang side only.
- `sglang-torch-profiler-analysis` → **not used.**

**Outputs.** `datasets/case{A,B,C,D}.jsonl`, `experiments/phase1/raw/*.json`, `experiments/phase1/summary.md` (4×2 table: TTFT p50/p95, TPOT, output tok/s, request throughput, fairness notes), `logs/phase1/*.log`.

**Success criteria.** 8 runs complete; stdev/median ≤ 5 % across repeats; per-cell fairness notes written.

**Risks.** Graph-capture overhead in first run (fix with longer warmup); OOM on one side (reduce equally and document); overhead-dominated ratios in Case A (flag as overhead, not as framework gap).

---

### Phase 2 — Identify Informative Cases (0.5–1 day)

**Goal.** Given Phase-1 evidence (TTFT gap is universal, TPOT at parity), decide which cases enter Phase-3 profiling and with what shaping. Concretely:

1. Determine whether SGLang's ~50 ms Case-A TTFT is *compressible by flags* or is a **structural scheduler/dispatch floor** (the latter is the Phase-3 target).
2. Determine whether Case-B's long-prefill gap shrinks under chunked-prefill shaping (same scheduler floor + prefill compute — we want to disentangle them).
3. Reduce TTFT variance on Cases C/D (CV 20–42%) to a profilable state (CV ≤10%), or explicitly drop them.
4. Decide whether vLLM's Case-B TTFT noise (CV=99.3%) invalidates that baseline, or is simply a first-request cold effect that a longer steady-state window absorbs.

#### Evidence inherited from Phase 1 (do not re-measure)

| Signal | Value | Phase-2 implication |
|---|---|---|
| SGLang TTFT delta 128→2048 tok | +7.7 ms | Prefill compute is cheap; a ~50 ms fixed overhead exists at c=1 |
| vLLM TTFT delta 128→2048 tok | +10.0 ms | Comparable scaling; vLLM floor is ~14 ms |
| TPOT ratio (all cases) | 0.98–1.02× | Decode path is not a Phase-2 or Phase-3 concern — do not shape it |
| TTFT CV SGLang Case A/B | 1.2% / 3.7% | Low-noise, profilable as-is |
| TTFT CV SGLang Case C/D | 20–42% | Not profilable; variance reduction precondition |
| vLLM TTFT CV Case B | 99.3% | Median is a lower bound; needs longer window or more reps on vLLM side |

#### Decision rule (applied per case)

| Gap size (median) | TTFT CV (both frameworks) | Action |
|---|---|---|
| < 5% | any | Drop, or reshape workload (more prompt / more concurrency). |
| 5–15% | ≤10% | Run tier-1 shaping sweep. Promote to Phase 3 only if sweep cannot close the gap. |
| > 15%, both CVs ≤10% | ≤10% | Run shaping sweep *before* promoting. If gap survives any flag combo, promote as **structural**. |
| > 15%, either CV > 10% | > 10% | Run **variance-reduction** sweep first. Re-evaluate gap after. Do not profile on noisy data (§14 anti-pattern). |

#### Sweep plan (per case, ≤4 candidates each, 1 rep initially, 3 reps on finalists)

All sweeps use `sglang-auto-benchmark run --tier 1` against the existing `datasets/{case}.jsonl` (no regeneration). vLLM is re-run in parallel only on finalists using a matching bench_serving invocation — auto_benchmark cannot drive vLLM. Config files live in `configs/phase2_shaping/{case}.yaml`.

**Case A — scheduling-overhead isolation (highest priority).** The ~50 ms floor at c=1 is where the Phase-3 hypothesis starts. Candidate axes:

| Axis | Values | Rationale |
|---|---|---|
| `--disable-overlap-schedule` | {off, on} | Overlap scheduler adds a step of pipelining; at c=1 it may be pure overhead |
| `--schedule-policy` | {lpm (default), fcfs} | Longest-prefix-match adds per-request bookkeeping; fcfs is the minimal path |
| `--chunked-prefill-size` | {8192 (default), -1 (disabled)} | At 128-token prompts, chunking should be a no-op; confirming this rules it out |
| `--stream-interval` | {1 (default), 8} | Eliminates per-token streaming cost from TTFT measurement path |

Pick ≤4 combinations, not the full grid. Start with each flag flipped individually from default; add one 2-way combo if a single flag moves TTFT by ≥10%.

**Case B — long-prefill disentanglement.** Step 2.1 produced no scheduler winner (A0 baseline = default was the finalist). Therefore Case B uses **default flags as base** — no scheduler flags inherited. The sweep focuses exclusively on the chunked-prefill axis:

| Axis | Values | Rationale |
|---|---|---|
| `--chunked-prefill-size` | {2048, 4096, 8192 (default), -1} | 2048-token prompt vs 8192 default chunk — chunking is currently a no-op; forcing smaller chunks probes whether chunked-prefill scheduling adds overhead on top of the 56 ms structural floor |

`--schedule-conservativeness` is **dropped** from the sweep: since Case A confirmed the floor is framework-intrinsic (not policy-driven), conservativeness (which only affects scheduling policy) is unlikely to move Case B differently. Adding it would expand the grid without evidence justification.

**Cases C & D — variance reduction (gate, not shaping).** Before any flag sweep, establish whether the TTFT CV is driven by insufficient warmup or by steady-state scheduler jitter. Sweep axes on the **client**, not the server:

| Axis | Values | Rationale |
|---|---|---|
| `--warmup-requests` | {30 (current), 100, 300} | 30 warmups may be insufficient for c=16 to reach steady batch |
| Bench duration | {current, 2×, 4×} | Longer sampling window reduces p50 variance |
| Repetitions | 3 → 5 on finalists | Median stabilizes; also reveals if the "noise" is actually a bimodal distribution (graph recapture events) |

If CV drops below 10% with extended warmup alone, promote C/D to Phase 3 with the new warmup setting baked into the protocol. If CV stays >10%, record the case as **not profilable at c=16** and drop from Phase-3 scope.

**vLLM Case-B noise check (not a sweep, a re-run).** Re-run Case B on vLLM with `warmup_requests=300` and 5 reps. If CV drops below 20%, the Phase-1 number was a cold-start artifact and the 2.59× ratio stands. If CV stays high, the vLLM side is genuinely bimodal and the Case-B gap carries a confidence ceiling of M for the same reason described in §6.2.

#### Artifacts

| File | Role |
|---|---|
| `configs/phase2_shaping/caseA.yaml`, `caseB.yaml` | auto_benchmark specs (server axes) |
| `configs/phase2_shaping/caseCD_variance.yaml` | client-side variance-reduction spec |
| `experiments/phase2_shaping/{caseA,caseB,caseCD}/live_results.jsonl` | Append-only raw |
| `experiments/phase2_shaping/{case}/results.jsonl`, `results.csv`, `summary.md` | Processed per-case sweep output |
| `experiments/phase2_shaping/vllm_caseB_recheck.json` | vLLM re-run result |
| `experiments/phase2/selected_cases.md` | **Phase-3 entry gate** — one row per promoted case with: phenomenon, shaping applied, residual gap, residual CV, fairness-dependence tier |

#### Skill usage

- `sglang-auto-benchmark` → `run` tier 1, ≤4 candidates per case, resumable.
- `debug-cuda-crash` → L1 passive (candidate servers are transient, odd flag combos may crash — free crash trail).
- `sglang-torch-profiler-analysis` → **not used.** No interpretation in Phase 2.

#### Success criteria (all must hold to exit Phase 2)

1. Case A and Case B either (a) have a shaped SGLang config that holds the TTFT gap ≥15%, in which case they promote to Phase 3 as *structural*, or (b) the gap collapses under a flag flip, in which case the finding is recorded and the case is **not** profiled.
2. Cases C/D either reach TTFT CV ≤10% (promote) or are formally dropped with a one-line justification in `selected_cases.md`.
3. vLLM Case-B baseline is either re-confirmed with CV <20% or explicitly labeled noisy in all downstream Case-B conclusions.
4. `experiments/phase2/selected_cases.md` lists 1–2 cases for Phase 3. Zero cases means returning to Phase 1 with a reshaped matrix (longer prompts, higher concurrency) — Phase 3 does not run on speculation.

#### Expected outcome (prediction, for Phase-3 pre-planning — not a gate)

Based on the Phase-1 evidence, the most likely Phase-2 result is: Case A promotes as structural (the 50 ms floor is unlikely to yield to flags), Case B promotes as structural with a secondary chunked-prefill note, Cases C/D either stabilize under longer warmup (promote as secondary) or drop. This is a hypothesis, not a plan commitment — the sweep outcomes govern.

#### Execution order

Steps are **serial** unless explicitly marked parallel. Case A results gate Case B's scheduler axis; C/D variance conclusion gates their Phase-3 eligibility. Step 2.4 (vLLM recheck) is fully independent and can run concurrently with any server-free window.

```
Step 2.0  Pre-flight (~15 min)
  ├─ mkdir configs/phase2_shaping/
  ├─ mkdir experiments/phase2_shaping/{caseA,caseB,caseCD}/
  ├─ mkdir experiments/phase2/  logs/phase2/
  ├─ verify datasets SHA-256 (no regen)
  └─ verify GPU 6 idle, no residual server processes

Step 2.1  Case A scheduler sweep (~2 h)   ← SERIAL, must finish first
  ├─ candidates A0 (baseline) / A1 (disable-overlap) / A2 (fcfs) / A3 (stream-interval 8)
  ├─ each: 1 rep, 30 warmup, LOGLEVEL=1 → logs/phase2/
  ├─ if single flag moves TTFT ≥10 ms → add A4 = best 2-way combo
  └─ decision: structural (→ 3-rep reconfirm) or configurational (→ record + drop)

Step 2.2  Case B chunked-prefill sweep (~1.5 h)   ← SERIAL, after 2.1
  ├─ base = default flags (2.1 produced no scheduler winner; A0 baseline = default)
  ├─ candidates B0/B1/B2/B3 on chunked-prefill-size axis only
  └─ decision: same-floor (→ Phase 3 as "same floor") or chunking-sensitive (→ note + promote)

Step 2.3  Case C/D variance gate (~2 h)   ← can interleave with 2.1/2.2 between server restarts
  ├─ client-only: V0 (warmup=30) / V1 (warmup=100) / V2 (warmup=300, 4× bench_n, 5 reps)
  └─ decision per case: CV ≤10% → promote; CV >10% → check bimodal → drop

Step 2.4  vLLM Case B noise recheck (~20 min)   ← PARALLEL with any step above
  └─ warmup=300, 5 reps, vLLM only; → experiments/phase2_shaping/vllm_caseB_recheck.json

Step 2.5  Synthesize selected_cases.md (~30 min)   ← SERIAL, after all above
  └─ one row per case; promote/drop decision with evidence pointers
```

**Script location:** `experiments/phase2/scripts/run_phase2_caseA.py`, `run_phase2_caseB.py`, `run_phase2_caseCD.py` — same structure as `experiments/phase1/scripts/run_phase1.py`. No auto_benchmark YAML runner; direct bench_serving orchestration for full control.

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

### Phase 1 — Minimal Fair Baseline (completed 2026-04-17)

#### Run conditions
- GPU: H200 index 6, `CUDA_VISIBLE_DEVICES=6`, `HF_HUB_OFFLINE=1`
- Servers run sequentially; 3 reps per (case × framework); median taken
- Dataset: text-only random prompts (token IDs 0–151642, special tokens excluded)

#### Key engineering issue resolved
The `sglang.auto_benchmark convert --kind random` sampler draws from the full tokenizer vocabulary including multimodal special tokens (`<|image_pad|>` ID 151655, `<|vision_start|>` 151652, etc.). These trigger `general_mm_embed_routine` in Qwen3-VL, exhausting GPU activation memory and causing OOM crashes. Fixed with a custom generator (`experiments/phase1/scripts/gen_datasets.py`) that restricts sampling to IDs 0–151642.

#### Produced files

| File | Contents |
|---|---|
| `datasets/case{A,B,C,D}.jsonl` | Byte-identical fixed-length datasets (text-only) |
| `experiments/phase1/raw/dataset_sha256.txt` | SHA-256 of each JSONL |
| `experiments/phase1/raw/{case}_{framework}_rep{1,2,3}.json` | Raw bench_serving output |
| `experiments/phase1/raw/{case}_{framework}_rep{1,2,3}_meta.json` | Versions, backend, seed, dataset SHA |
| `experiments/phase1/summary.md` | 4×2 table with ratios and CV flags |
| `logs/phase1/sglang_server.log` | SGLang Phase-1 startup log |
| `logs/phase1/vllm_server.log` | vLLM Phase-1 startup log |

#### Results summary (SGLang / vLLM ratios)

| Case | TTFT p50 ratio | TPOT p50 ratio | Req/s ratio | Verdict |
|---|---|---|---|---|
| A — Short (128→128, c=1) | **3.89×** ↑ SGLang slower | 1.00× parity | 0.95× | Gap >15% → Phase 3 candidate |
| B — Long prefill (2048→128, c=1) | **2.59×** ↑ SGLang slower | 0.99× parity | 0.96× | Gap >15% → Phase 3 candidate |
| C — Batched (512→128, c=16) | **1.49×** ↑ SGLang slower | 0.98× parity | 0.93× | CV ⚠ (20%) — stabilize first |
| D — Decode-heavy (512→512, c=16) | **1.34×** ↑ SGLang slower | 1.02× parity | 0.97× | Marginal gap; p99 CV ⚠ (42%) |

All CV values for TPOT and throughput are ≤2% — decode metrics are stable. TTFT is where all variance lives.

#### Key findings

1. **TTFT gap is universal; TPOT/throughput gap is negligible.** SGLang decode (TPOT) is on par with vLLM (within 2%) across all 4 cases. Every significant gap is in first-token latency.

2. **Case A TTFT overhead is scheduling/dispatch, not compute.** SGLang TTFT increases only 7.7 ms from Case A (128 tok) to Case B (2048 tok), while vLLM increases 10 ms. The actual prefill compute for 16× more tokens would be far larger — SGLang's TTFT is dominated by pre-prefill overhead (~50 ms fixed cost at concurrency=1).

3. **vLLM Case B TTFT is noisy (cv=99.3%).** Likely chunked-prefill scheduling jitter or CUDA graph warmup. The median (24.1 ms) is a lower bound; the gap with SGLang is real but the vLLM baseline needs re-examination before drawing strong conclusions for Case B.

4. **Cases C and D TTFT CV is elevated (20–42%)** — scheduler queuing jitter at concurrency=16. The gap is real (1.34–1.49×) but confidence is M until variance is reduced.

#### Phase 2 action
- Apply Phase-2 decision rule: all 4 cases have TTFT gap >15%; Cases A and B are primary candidates.
- Case A is highest priority: the scheduling overhead hypothesis is clean, low-noise, and directly actionable.
- Cases C and D: run a short reshaping sweep to reduce TTFT variance before committing to profiling.

### Phase 2 — Identify Informative Cases (in progress, 2026-04-24)

#### Step 2.1 — Case A scheduler-overhead sweep (completed 2026-04-24)

**Run conditions:** GPU 6, clock-locked 1980 MHz, dataset SHA verified, `SGLANG_KERNEL_API_LOGLEVEL=1`.

**Results:**

| Candidate | Flag | TTFT p50 (ms) | Δ vs baseline |
|---|---|---|---|
| A0 baseline | (default) | 57.1 | — |
| A1 | `--disable-overlap-schedule` | 55.4 | −1.7 ms |
| A2 | `--schedule-policy fcfs` | 57.5 | +0.4 ms |
| A3 | `--stream-interval 8` | 57.0 | −0.0 ms |

**Finalist 3-rep reconfirm (A0 baseline):** median = **56.0 ms**, CV = **0.1%**.

**Verdict: STRUCTURAL.** No scheduler flag moved TTFT by ≥10 ms (maximum Δ = 1.7 ms). The ~56 ms TTFT floor is intrinsic to SGLang's c=1 request-dispatch path and cannot be closed by any combination of overlap scheduling, scheduling policy, or stream interval settings. 2-way combo step was bypassed — threshold not triggered.

**Phase-3 entry:** Case A promotes with phenomenon label: *"SGLang ~56 ms structural scheduler/dispatch floor at c=1, unresponsive to overlap/policy/stream flags."* Base config = default (no shaping). Fairness dependence: Framework-intrinsic.

**Produced files:** `experiments/phase2_shaping/caseA/summary.md`, `experiments/phase2_shaping/caseA/A{0..3}_baseline_rep*.json`, `logs/phase2/sglang_caseA_*.log`.

---

## 16. Prioritized Next-Step Checklist

1. ✅ Create the filesystem layout from §8.1 (placeholder READMEs in each directory).
2. ✅ Phase 0 — servers up, equivalence matrix run. All Tier-A/B pass; outputs EXACT match.
3. ✅ Generate `datasets/case{A..D}.jsonl` — text-only random prompts (special tokens excluded), SHA-256 logged.
4. ✅ Phase 1 — 24 runs (4 cases × 2 frameworks × 3 reps); `experiments/phase1/summary.md` complete.
5. Phase 2 (in progress):
   - ✅ Step 2.1 — Case A sweep: 4 candidates, floor confirmed STRUCTURAL at 56 ms CV=0.1%. Base config = default.
   - Step 2.2 — Case B chunked-prefill sweep (base = default, no extra flags from 2.1).
   - Step 2.3 — Cases C/D variance reduction (client-side: warmup 30→300, 5 reps).
   - Step 2.4 — vLLM Case B noise re-check (warmup=300, 5 reps) — can run in parallel with 2.2/2.3.
   - Step 2.5 — Synthesize `experiments/phase2/selected_cases.md`.
6. Phase 3 — SGLang mapping+formal + vLLM prefill_like+decode_like per selected case (1 day).
7. Phase 4 — triage + breakdown + vLLM cross-check per case; author `hypotheses.md` and `ranked_recommendations.md` (1–1½ days).
8. Phase 5 (if warranted) — tier-2 validation sweeps for the top 2 hypotheses.
9. Promote `analysis/**` into `reports/**` deliverables.
