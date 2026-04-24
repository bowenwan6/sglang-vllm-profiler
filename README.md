<div align="center">

# SGLang vs vLLM — Latency Profiling

<p>
  <a href="https://github.com/sgl-project/sglang">
    <img src="https://img.shields.io/badge/SGLang-sgl--project-blue?logo=github&logoColor=white" />
  </a>
  &nbsp;
  <a href="https://github.com/vllm-project/vllm">
    <img src="https://img.shields.io/badge/vLLM-vllm--project-blueviolet?logo=github&logoColor=white" />
  </a>
</p>

<p>
  <img src="https://img.shields.io/badge/Phase_0-✅ Complete-2ea44f" />
  <img src="https://img.shields.io/badge/Phase_1-✅ Complete-2ea44f" />
  <img src="https://img.shields.io/badge/Phase_2-✅ Complete-2ea44f" />
  <img src="https://img.shields.io/badge/Phase_3-⬜ Pending-lightgrey" />
  <img src="https://img.shields.io/badge/Phase_4-⬜ Pending-lightgrey" />
  <img src="https://img.shields.io/badge/Phase_5-⬜ Pending-lightgrey" />
</p>

Structured, phase-gated profiling of SGLang against vLLM on **Qwen3-VL-8B-Instruct** to extract actionable optimization targets — not a benchmark report.

*Single H200 · TP=1 · bfloat16 · text-only path*

</div>

---

## Key Findings

- **TTFT is the only gap.** TPOT and throughput are at parity (±2%) across all tested workloads.
- **SGLang carries a ~56 ms fixed scheduler/dispatch overhead at c=1.** Prompt length barely moves it — 128→2048 tokens adds just +7.7 ms. vLLM's equivalent floor is ~14 ms.
- **The floor is structural, not configurational.** Four scheduler flags (`--disable-overlap-schedule`, `--schedule-policy fcfs`, `--stream-interval`, `--chunked-prefill-size`) each moved TTFT by ≤2 ms.
- **Secondary finding.** When chunked prefill is actually triggered (chunk_size < prompt_len), TTFT scales linearly with chunk count — each additional chunk adds ~65–85 ms — suggesting the dispatch floor is incurred *per chunk dispatched*, not per request.

---

## Baseline (Phase 1)

> 24 runs · 4 cases × 2 frameworks × 3 reps · H200 clocked at 1980 MHz

| Case | Prompt → Output | Concurrency | SGLang TTFT p50 | vLLM TTFT p50 | Ratio | TPOT |
|:-----|:---------------|:-----------:|----------------:|---------------:|------:|-----:|
| A — Short | 128 → 128 | 1 | 54.6 ms | 14.1 ms | **3.89×** | 1.00× |
| B — Long prefill | 2048 → 128 | 1 | 62.3 ms | 24.1 ms | **2.59×** | 0.99× |
| C — Batched | 512 → 128 | 16 | 243.9 ms | 164.1 ms | **1.49×** | 0.98× |
| D — Decode-heavy | 512 → 512 | 16 | 247.0 ms | 184.8 ms | **1.34×** | 1.02× |

> Case A→B spans 16× more tokens yet TTFT grows only 7.7 ms — prefill compute is cheap. The gap lives in pre-forward-pass overhead.

---

## Shaping & Case Selection (Phase 2)

### Case A — scheduler-overhead sweep

| Flag | TTFT p50 | Δ |
|:-----|--------:|--:|
| *(default — baseline)* | 57.1 ms | — |
| `--disable-overlap-schedule` | 55.4 ms | −1.7 ms |
| `--schedule-policy fcfs` | 57.5 ms | +0.4 ms |
| `--stream-interval 8` | 57.0 ms | −0.0 ms |

**→ Structural.** 3-rep reconfirm: **56.0 ms, CV = 0.1%.**

### Case B — chunked-prefill sweep

| chunk-size | Actual chunks | TTFT p50 | Δ vs default |
|:----------:|:-------------:|--------:|-------------:|
| 8192 *(default)* | 1 | 68.5 ms | — |
| −1 *(disabled)* | — | 66.7 ms | −1.8 ms |
| 1024 | 2 | 169.2 ms | +100.7 ms |
| 512 | 4 | 261.5 ms | +193.0 ms |

**→ Structural (same floor).** Default chunk=8192 never splits a 2048-tok prompt. TTFT ∝ chunk count when chunking is active — per-chunk dispatch overhead.

### Cases C & D — variance gate

| Case | warmup=30 CV | warmup=100 CV | warmup=300 CV | Result |
|:-----|:-----------:|:-------------:|:-------------:|:------:|
| C (512→128, c=16) | 9.5% | **4.2%** | 2.1% | ✅ **Promote** |
| D (512→512, c=16) | 19.8% | 0.1% * | 14.8% | ❌ **Drop** |

<sup>\* 3-rep window; V2 (5 reps) re-exposed a bimodal pattern — periodic ~160 ms outliers vs steady ~243 ms.</sup>

### Phase 3 entry list

| Case | Priority | Phenomenon | Protocol |
|:-----|:--------:|:-----------|:---------|
| A | Primary | ~56 ms structural dispatch floor at c=1 | default, warmup=30 |
| B | Primary | Same floor + per-chunk dispatch overhead | default, warmup=30 |
| C | Secondary | 1.47× TTFT gap at c=16 | default, **warmup=100** |
| D | — | Dropped — bimodal variance, Phase-4 candidate | — |

---

## Environment

| | SGLang | vLLM |
|:--|:-------|:-----|
| Version | `0.0.0.dev1+ga4cf2ea12` | `0.19.0` |
| PyTorch | 2.9.1+cu129 | 2.10.0+cu128 |
| Attention (text) | FlashInfer 0.6.7.post3 | FlashAttention v3 |
| KV cache | ~102 GB | ~105.9 GB |
| GPU | H200 index 6, 144 GB | ← same |
| Model | Qwen3-VL-8B-Instruct @ `0c351dd` | ← same |

**Functional equivalence verified:** byte-identical greedy outputs on 3 test prompts (128 tokens, temperature=0). Attention backend differs — any Phase-4 attention-kernel finding carries confidence ceiling **M** until backends are aligned.

---

## Reproducing

### Setup

```
GPU:    Single H200 (80 GB+ required for Qwen3-VL KV cache)
SGLang: dev install at /sgl-workspace/sglang
vLLM:   0.19.0 in conda env `vllm`
Model:  Qwen/Qwen3-VL-8B-Instruct (HF cache, offline)
```

### ⚠️ Dataset generation — read this first

`sglang.auto_benchmark convert --kind random` samples from the full vocabulary, including multimodal special tokens (`<|image_pad|>` ID 151655, `<|vision_start|>` 151652, etc.). On Qwen3-VL these trigger the vision embedding path and cause OOM. **Always use the custom generator:**

```bash
HF_HUB_OFFLINE=1 python3 experiments/phase1/scripts/gen_datasets.py
# Samples token IDs 0–151642 only → datasets/case{A,B,C,D}.jsonl
```

### Running the phases

```bash
# Phase 1 — baseline (24 runs, ~6 h)
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase1/scripts/run_phase1.py

# Phase 2 — shaping sweeps
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase2/scripts/run_phase2_caseA.py   # ~2 h
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase2/scripts/run_phase2_caseB.py   # ~1.5 h
CUDA_VISIBLE_DEVICES=6 python3 experiments/phase2/scripts/run_phase2_caseCD.py  # ~2 h
```

---

## Repository Layout

```
profiling_lab/
├── plan.md                          ← Execution plan, decision rules, all results
├── datasets/                        ← Canonical autobench JSONL (never regenerate mid-project)
├── experiments/
│   ├── env_snapshot.md              ← Versions, attention backends, GPU memory
│   ├── phase0/equivalence.md        ← Tier A/B/C equivalence results
│   ├── phase1/                      ← Raw bench_serving JSON per run
│   ├── phase2/selected_cases.md     ← Phase-3 entry gate
│   └── phase2_shaping/              ← Per-case sweep results + summaries
├── logs/                            ← Server stderr + kernel-API boundary trails (L1)
├── traces/                          ← Torch profiler artifacts (Phase 3, pending)
├── analysis/                        ← Triage tables, hypotheses, recommendations (Phase 4, pending)
└── reports/                         ← Final deliverables (Phase 5, pending)
```

---

<div align="center">
<sub>Bowen Wang · radixark02 · sglang-bowenw</sub>
</div>
