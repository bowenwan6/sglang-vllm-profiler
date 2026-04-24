<div align="center">

# SGLang vs vLLM — Latency Profiling

**Structured profiling of SGLang against vLLM on `Qwen3-VL-8B-Instruct` to find actionable optimization targets for SGLang.**

![Phase 0](https://img.shields.io/badge/Phase_0_Equivalence-✅_Complete-2ea44f)
![Phase 1](https://img.shields.io/badge/Phase_1_Baseline-✅_Complete-2ea44f)
![Phase 2](https://img.shields.io/badge/Phase_2_Shaping-✅_Complete-2ea44f)
![Phase 3](https://img.shields.io/badge/Phase_3_Profiling-⬜_Pending-lightgrey)
![Phase 4](https://img.shields.io/badge/Phase_4_Triage-⬜_Pending-lightgrey)
![Phase 5](https://img.shields.io/badge/Phase_5_Validation-⬜_Pending-lightgrey)

*Single H200 · TP=1 · Qwen3-VL-8B-Instruct · bfloat16 · text-only path*

</div>

---

## TL;DR

- **TTFT is the only gap.** TPOT and throughput are at parity (±2%) across all workloads.
- **SGLang has a ~56 ms fixed scheduler/dispatch overhead** at c=1 — independent of prompt length (128 → 2048 tokens adds only +7.7 ms). vLLM's floor is ~14 ms.
- **No scheduler flag closes it.** `--disable-overlap-schedule`, `--schedule-policy fcfs`, and `--stream-interval` all move TTFT by ≤2 ms — the gap is structural.
- **Phase 3 shortlist:** Cases A (128→128, c=1), B (2048→128, c=1), and C (512→128, c=16) enter profiling. Case D dropped — exhibits a persistent bimodal TTFT distribution under sustained decode load.
- **Secondary finding:** When chunked prefill is actually triggered (chunk_size < prompt_len), TTFT scales linearly with chunk count — each additional chunk costs ~65–85 ms, suggesting the dispatch floor is incurred *per chunk*, not per request.

---

## Baseline Results

> Phase 1 — 24 runs (4 cases × 2 frameworks × 3 reps), GPU 6 locked at 1980 MHz.

| Case | Prompt→Output | Concurrency | SGLang TTFT p50 | vLLM TTFT p50 | Ratio | TPOT |
|:-----|:-------------|:-----------:|----------------:|---------------:|:-----:|:----:|
| A — Short | 128 → 128 | 1 | 54.6 ms | 14.1 ms | **3.89×** | 1.00× ≈ |
| B — Long prefill | 2048 → 128 | 1 | 62.3 ms | 24.1 ms | **2.59×** | 0.99× ≈ |
| C — Batched | 512 → 128 | 16 | 243.9 ms | 164.1 ms | **1.49×** | 0.98× ≈ |
| D — Decode-heavy | 512 → 512 | 16 | 247.0 ms | 184.8 ms | **1.34×** | 1.02× ≈ |

The 7.7 ms TTFT delta from Case A→B (16× more tokens) implies prefill compute is cheap — the ~50 ms gap is overhead *before* the forward pass starts.

---

## Pipeline

```
Phase 1          Phase 2           Phase 3         Phase 4            Phase 5
Baseline   →   Shaping &    →   Trace       →   Triage &      →   Hypothesis
benchmark      case selection    collection      hypothesis          validation
                                                 synthesis
  ✅ done         ✅ done           ⬜ pending       ⬜ pending          ⬜ pending
```

Each phase is a hard gate — Phase 3 only runs on cases that survived Phase 2 shaping. See [`plan.md`](plan.md) for the full execution protocol and decision rules.

---

## Phase 2 — Shaping Results

### Case A: scheduler-overhead sweep

Tested whether any flag closes the ~56 ms floor at c=1.

| Candidate | Flag | TTFT p50 |
|:----------|:-----|--------:|
| A0 baseline | *(default)* | 57.1 ms |
| A1 | `--disable-overlap-schedule` | 55.4 ms |
| A2 | `--schedule-policy fcfs` | 57.5 ms |
| A3 | `--stream-interval 8` | 57.0 ms |

**Verdict:** No flag moved TTFT by ≥10 ms. Floor confirmed **structural** (3-rep reconfirm: 56.0 ms, CV=0.1%).

### Case B: chunked-prefill sweep

| Candidate | chunk-size | Chunks | TTFT p50 |
|:----------|:----------:|:------:|--------:|
| B0 | 8192 (default) | 1 | 68.5 ms |
| B3 | −1 (disabled) | — | 66.7 ms |
| B2 | 1024 | 2 | 169.2 ms |
| B1 | 512 | 4 | 261.5 ms |

**Verdict:** Default config is not chunking (prompt fits in one chunk) → gap is same structural floor. Secondary finding: TTFT ∝ chunk count when chunking is active, confirming the floor is per-chunk dispatch overhead.

### Cases C & D: variance reduction

| Case | V0 warmup=30 CV | V1 warmup=100 CV | V2 warmup=300 CV | Decision |
|:-----|:---------------:|:----------------:|:----------------:|:--------:|
| C (512→128, c=16) | 9.5% | **4.2%** | 2.1% | **PROMOTE** |
| D (512→512, c=16) | 19.8% | 0.1%* | 14.8% | **DROP** |

*\*V1 was a 3-rep lucky window — V2 (5 reps) re-exposed the bimodal pattern (~160 ms outlier vs ~243 ms steady).*

### Phase 3 entry list

| Case | Phenomenon to profile | Config |
|:-----|:---------------------|:-------|
| **A** *(primary)* | ~56 ms structural dispatch floor at c=1 | default, warmup=30 |
| **B** *(primary)* | Same floor; secondary: per-chunk dispatch cost | default, warmup=30 |
| **C** *(secondary)* | 1.47× TTFT gap at c=16, CV now stable | default, **warmup=100** |

---

## Environment

| | SGLang | vLLM |
|:--|:-------|:-----|
| Version | 0.0.0.dev1+ga4cf2ea12 | 0.19.0 |
| PyTorch | 2.9.1+cu129 | 2.10.0+cu128 |
| Attention (text) | FlashInfer 0.6.7.post3 | FlashAttention v3 |
| KV cache | ~102 GB | ~105.9 GB |

**Controlled:** GPU (H200 index 6), model snapshot (`0c351dd`), dtype (bfloat16), TP=1, tokenizer, sampler (greedy), prompt sets (byte-identical JSONL).

**Measured (not pinned):** torch version, attention backend — any Phase-4 attention-kernel finding carries confidence ceiling M until backends are aligned.

**Functional equivalence:** SGLang and vLLM produce byte-identical greedy outputs on all 3 test prompts (128 tokens, temperature=0). No "semantic-level only" caveat needed downstream.

---

## Reproducing

### Requirements

- Single H200 (or comparable 80 GB+ GPU)
- SGLang dev install: `/sgl-workspace/sglang`
- vLLM 0.19.0 in conda env `vllm`
- `Qwen/Qwen3-VL-8B-Instruct` cached in HF hub

### ⚠️ Critical: dataset generation

**Do not use `sglang.auto_benchmark convert --kind random`.** It samples from the full vocabulary including multimodal special tokens (`<|image_pad|>` ID 151655, etc.), which trigger Qwen3-VL's vision embedding path and cause OOM. Use the custom generator instead:

```bash
HF_HUB_OFFLINE=1 python3 experiments/phase1/scripts/gen_datasets.py
# Produces datasets/case{A,B,C,D}.jsonl with token IDs 0–151642 only
```

### Phase 1 — baseline

```bash
cd /data/profiling_lab
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase1/scripts/run_phase1.py
# Runs 24 bench_serving jobs; results → experiments/phase1/raw/
```

### Phase 2 — shaping sweeps

```bash
# Case A scheduler sweep (~2 h)
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase2/scripts/run_phase2_caseA.py

# Case B chunked-prefill sweep (~1.5 h)
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase2/scripts/run_phase2_caseB.py

# Cases C/D variance gate (~2 h)
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase2/scripts/run_phase2_caseCD.py
```

---

## Repository Layout

```
profiling_lab/
├── plan.md                        ← Full execution plan, decision rules, all results
│
├── datasets/                      ← Byte-identical autobench JSONL (never regenerate mid-project)
├── experiments/
│   ├── env_snapshot.md            ← Versions, backends, GPU memory
│   ├── phase0/equivalence.md      ← Tier A/B/C equivalence matrix
│   ├── phase1/                    ← Raw bench_serving JSON + summary
│   ├── phase2/selected_cases.md   ← Phase-3 entry gate
│   ├── phase2_shaping/            ← Per-case sweep raw results + summaries
│   └── phase2/scripts/            ← Orchestration scripts
│
├── traces/                        ← Torch profiler artifacts (Phase 3, pending)
├── analysis/                      ← Triage tables, hypotheses, recommendations (Phase 4, pending)
├── logs/                          ← Server stderr, kernel-API boundary trails
└── reports/                       ← Final human-facing deliverables (Phase 5, pending)
```

---

## Fairness Model

All conclusions are tagged with the tier of variables they depend on:

| Tier | Meaning | Example |
|:-----|:--------|:--------|
| **Controlled** | Pinned identically on both sides | GPU, model weights, dtype, seed |
| **Measured** | Cannot be pinned; logged on every run | torch version, attention backend |
| **Framework-intrinsic** | Not aligned — this *is* the observation | Scheduler policy, CUDA graph shapes |

A hypothesis depending on a **Measured** variable carries confidence ceiling **M** until a version-matched re-run confirms it.

---

<div align="center">
<sub>Profiling by Bowen Wang · radixark02 · sglang-bowenw container</sub>
</div>
