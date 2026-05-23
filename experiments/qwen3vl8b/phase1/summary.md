# Phase 1 — Baseline Benchmark Summary

Generated: 2026-05-22 00:00 UTC

Active run `qwen3vl8b` · model Qwen3-VL-8B-Instruct @ 0c351dd · GPU 0 · TP=1 · bf16.
Ratio = SGLang / vLLM. >1 means SGLang slower (latency) or higher (throughput).
Sampling: explicit greedy via `--extra-request-body {"temperature":0,"top_p":1}`, `ignore_eos` default True, fixed output_len.

> ⚠️ run1 Phase 1 (`experiments/phase1/`) is historical reference under a different environment. Compare directionally only — do NOT mix with the earlier round (removed).

## A — Short latency (128→128, c=1)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 61.8 (cv=1.6%) | 12.6 (cv=4.4%) | 4.89× ↑ |
| TTFT p95 (ms) | 66.1 (cv=1.5%) | 17.9 (cv=3.1%) | 3.69× ↑ |
| TTFT p99 (ms) | 66.4 (cv=1.7%) | 18.0 (cv=3.3%) | 3.69× ↑ |
| TPOT p50 (ms) | 5.2 (cv=0.0%) | 5.3 (cv=0.0%) | 0.97× ≈ |
| TPOT p99 (ms) | 5.2 (cv=0.0%) | 5.3 (cv=0.0%) | 0.97× ≈ |
| Out tok/s | 178.6 (cv=0.2%) | 186.0 (cv=0.1%) | 0.96× ≈ |
| Req/s | 1.4 (cv=0.2%) | 1.5 (cv=0.1%) | 0.96× ≈ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+g0c8049d9b", "torch": "2.11.0+cu130", "flashinfer": "0.6.11.post1"}
**vllm versions**: {"vllm": "0.21.0", "torch": "2.11.0+cu130", "flashinfer": "0.6.8.post1"}

---

## B — Long prefill  (2048→128, c=1)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 66.7 (cv=2.9%) | 20.8 (cv=114.6% ⚠) | 3.20× ↑ |
| TTFT p95 (ms) | 70.6 (cv=3.8%) | 25.4 (cv=93.1% ⚠) | 2.78× ↑ |
| TTFT p99 (ms) | 71.6 (cv=3.8%) | 26.0 (cv=95.2% ⚠) | 2.75× ↑ |
| TPOT p50 (ms) | 5.2 (cv=0.1%) | 5.4 (cv=0.1%) | 0.97× ≈ |
| TPOT p99 (ms) | 5.2 (cv=0.1%) | 5.4 (cv=0.1%) | 0.97× ≈ |
| Out tok/s | 175.0 (cv=0.4%) | 180.9 (cv=3.3%) | 0.97× ≈ |
| Req/s | 1.4 (cv=0.4%) | 1.4 (cv=3.3%) | 0.97× ≈ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+g0c8049d9b", "torch": "2.11.0+cu130", "flashinfer": "0.6.11.post1"}
**vllm versions**: {"vllm": "0.21.0", "torch": "2.11.0+cu130", "flashinfer": "0.6.8.post1"}

---

## C — Batched       (512→128, c=16)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 247.5 (cv=9.4% ⚠) | 187.9 (cv=5.8% ⚠) | 1.32× ↑ |
| TTFT p95 (ms) | 255.4 (cv=0.3%) | 196.7 (cv=0.8%) | 1.30× ↑ |
| TTFT p99 (ms) | 257.4 (cv=0.6%) | 209.1 (cv=1.8%) | 1.23× ↑ |
| TPOT p50 (ms) | 5.8 (cv=8.8% ⚠) | 5.8 (cv=1.7%) | 0.99× ≈ |
| TPOT p99 (ms) | 7.3 (cv=3.9%) | 7.1 (cv=0.4%) | 1.04× ≈ |
| Out tok/s | 1984.9 (cv=2.1%) | 2192.5 (cv=0.3%) | 0.91× ↓ |
| Req/s | 15.5 (cv=2.1%) | 17.1 (cv=0.3%) | 0.91× ↓ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+g0c8049d9b", "torch": "2.11.0+cu130", "flashinfer": "0.6.11.post1"}
**vllm versions**: {"vllm": "0.21.0", "torch": "2.11.0+cu130", "flashinfer": "0.6.8.post1"}

---

## D — Decode-heavy  (512→512, c=16)

| Metric | SGLang | vLLM | SGLang/vLLM |
|--------|--------|------|-------------|
| TTFT p50 (ms) | 253.0 (cv=9.8% ⚠) | 189.7 (cv=6.2% ⚠) | 1.33× ↑ |
| TTFT p95 (ms) | 257.3 (cv=0.2%) | 196.2 (cv=1.4%) | 1.31× ↑ |
| TTFT p99 (ms) | 390.6 (cv=47.1% ⚠) | 222.4 (cv=0.4%) | 1.76× ↑ |
| TPOT p50 (ms) | 5.9 (cv=1.9%) | 5.9 (cv=0.9%) | 1.00× ≈ |
| TPOT p99 (ms) | 6.3 (cv=1.6%) | 6.2 (cv=0.3%) | 1.01× ≈ |
| Out tok/s | 2469.1 (cv=0.5%) | 2531.3 (cv=0.4%) | 0.98× ≈ |
| Req/s | 4.8 (cv=0.5%) | 4.9 (cv=0.4%) | 0.98× ≈ |

**sglang-oai versions**: {"sglang": "0.0.0.dev1+g0c8049d9b", "torch": "2.11.0+cu130", "flashinfer": "0.6.11.post1"}
**vllm versions**: {"vllm": "0.21.0", "torch": "2.11.0+cu130", "flashinfer": "0.6.8.post1"}

---

## Error rate

| Case | Framework | Failed | Total | Rate |
|------|-----------|-------:|------:|-----:|
| caseA_short | sglang-oai | 0 | 1200 | 0.00% |
| caseA_short | vllm | 0 | 1200 | 0.00% |
| caseB_longprefill | sglang-oai | 0 | 600 | 0.00% |
| caseB_longprefill | vllm | 0 | 600 | 0.00% |
| caseC_batched | sglang-oai | 0 | 6000 | 0.00% |
| caseC_batched | vllm | 0 | 6000 | 0.00% |
| caseD_decode | sglang-oai | 0 | 3000 | 0.00% |
| caseD_decode | vllm | 0 | 3000 | 0.00% |

## Fairness Notes

| Variable | SGLang | vLLM | Tier |
|----------|--------|------|------|
| GPU | H200 index 0 | H200 index 0 | Controlled |
| Model | Qwen3-VL-8B-Instruct 0c351dd | same | Controlled |
| dtype | bfloat16 | bfloat16 | Controlled |
| TP | 1 | 1 | Controlled |
| Sampling | temperature=0, top_p=1 (explicit), ignore_eos default | same client | Controlled |
| torch / CUDA | 2.11.0+cu130 / 13.0 | 2.11.0+cu130 / 13.0 | Measured (aligned) |
| Attention backend | FlashInfer 0.6.11.post1 (text) | FlashAttention v3 | Measured |
| FlashInfer | 0.6.11.post1 | 0.6.8.post1 (sampling) | Measured |
| Scheduler | fcfs / radix cache | vLLM cache mgr (prefix-cache ON) | Framework-intrinsic |
