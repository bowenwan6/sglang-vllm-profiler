# Phase 1 — Baseline Benchmark Summary

Generated: 2026-04-17 04:15 UTC

Ratio column: SGLang / vLLM. >1 means SGLang is **slower** (for latency metrics) or **higher** (for throughput).

## A — Short latency (128→128, c=1)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 54.6 (cv=1.2%) | 14.1 (cv=3.3%) | 3.89× ↑ |
| TTFT p99 (ms) | 61.3 (cv=2.1%) | 21.1 (cv=23.1% ⚠) | 2.90× ↑ |
| TPOT p50 (ms) | 5.3 (cv=0.0%) | 5.3 (cv=0.0%) | 1.00× ≈ |
| TPOT p99 (ms) | 5.3 (cv=0.0%) | 5.3 (cv=0.1%) | 1.00× ≈ |
| Out tok/s | 177.4 (cv=0.1%) | 186.8 (cv=0.1%) | 0.95× ↓ |
| Req/s | 1.4 (cv=0.1%) | 1.5 (cv=0.1%) | 0.95× ↓ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+ga4cf2ea12", "torch": "2.9.1+cu129", "flashinfer": "0.6.7.post3"}
**vllm versions**: {"vllm": "0.19.0", "torch": "2.10.0+cu128"}

---

## B — Long prefill  (2048→128, c=1)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 62.3 (cv=3.7%) | 24.1 (cv=99.3% ⚠) | 2.59× ↑ |
| TTFT p99 (ms) | 75.0 (cv=4.2%) | 48.3 (cv=62.1% ⚠) | 1.55× ↑ |
| TPOT p50 (ms) | 5.3 (cv=0.2%) | 5.4 (cv=0.1%) | 0.99× ≈ |
| TPOT p99 (ms) | 5.3 (cv=0.2%) | 5.4 (cv=0.4%) | 0.99× ≈ |
| Out tok/s | 173.5 (cv=0.5%) | 180.7 (cv=3.3%) | 0.96× ≈ |
| Req/s | 1.4 (cv=0.5%) | 1.4 (cv=3.3%) | 0.96× ≈ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+ga4cf2ea12", "torch": "2.9.1+cu129", "flashinfer": "0.6.7.post3"}
**vllm versions**: {"vllm": "0.19.0", "torch": "2.10.0+cu128"}

---

## C — Batched       (512→128, c=16)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 243.9 (cv=20.6% ⚠) | 164.1 (cv=9.9% ⚠) | 1.49× ↑ |
| TTFT p99 (ms) | 384.0 (cv=36.5% ⚠) | 233.7 (cv=2.5%) | 1.64× ↑ |
| TPOT p50 (ms) | 5.9 (cv=8.9% ⚠) | 6.1 (cv=2.4%) | 0.98× ≈ |
| TPOT p99 (ms) | 7.5 (cv=4.0%) | 7.1 (cv=0.5%) | 1.06× ↑ |
| Out tok/s | 2017.7 (cv=1.4%) | 2163.3 (cv=0.5%) | 0.93× ↓ |
| Req/s | 15.8 (cv=1.4%) | 16.9 (cv=0.5%) | 0.93× ↓ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+ga4cf2ea12", "torch": "2.9.1+cu129", "flashinfer": "0.6.7.post3"}
**vllm versions**: {"vllm": "0.19.0", "torch": "2.10.0+cu128"}

---

## D — Decode-heavy  (512→512, c=16)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 247.0 (cv=2.6%) | 184.8 (cv=6.7% ⚠) | 1.34× ↑ |
| TTFT p99 (ms) | 415.6 (cv=42.1% ⚠) | 233.4 (cv=9.5% ⚠) | 1.78× ↑ |
| TPOT p50 (ms) | 6.0 (cv=0.1%) | 5.9 (cv=0.1%) | 1.02× ≈ |
| TPOT p99 (ms) | 6.3 (cv=1.4%) | 6.2 (cv=1.0%) | 1.01× ≈ |
| Out tok/s | 2438.0 (cv=0.3%) | 2518.4 (cv=0.2%) | 0.97× ≈ |
| Req/s | 4.8 (cv=0.3%) | 4.9 (cv=0.2%) | 0.97× ≈ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+ga4cf2ea12", "torch": "2.9.1+cu129", "flashinfer": "0.6.7.post3"}
**vllm versions**: {"vllm": "0.19.0", "torch": "2.10.0+cu128"}

---

## Fairness Notes

| Variable | SGLang | vLLM | Tier |
|----------|--------|------|------|
| GPU | H200 index 6 | H200 index 6 | Controlled |
| Model | Qwen3-VL-8B-Instruct 0c351dd | same | Controlled |
| dtype | bfloat16 | bfloat16 | Controlled |
| TP | 1 | 1 | Controlled |
| Attention backend | FlashInfer 0.6.7.post3 (text) | FlashAttention v3 | Measured |
| torch version | 2.9.1+cu129 | 2.10.0+cu128 | Measured |
| KV cache | ~102 GB | ~105.9 GB | Measured |
| ignore_eos | default (True) | default (True) | Controlled |
| Scheduler | radix cache | vLLM cache manager | Framework-intrinsic |
| CUDA graph shapes | 36 graphs (SGLang) | 51 graphs (vLLM) | Framework-intrinsic |
