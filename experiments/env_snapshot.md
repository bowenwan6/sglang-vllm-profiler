# Environment Snapshot — ACTIVE: run2_qwen3vl8b

Generated: 2026-05-21 (run2)

> **This file describes the ACTIVE run2 environment.** Run-local detail is mirrored at
> `experiments/run2_qwen3vl8b/env_snapshot.md`.
>
> ⚠️ **Historical note — do NOT mix run1 and run2 numbers.** The old run1 baselines in
> `experiments/phase1/`, `experiments/phase2/`, `experiments/phase2_shaping/` were measured on a
> *different* environment (see "run1 historical environment" at the bottom). Per plan.md §6.4,
> conclusions that depend on a changed Controlled/Measured variable must be re-confirmed. run1
> Phase 1/2 numbers are reference only and cannot be compared directly against run2 results.

## Active run
- Run id: `run2_qwen3vl8b`
- Phase 0: ✅ complete + PASS (canonical artifacts in `experiments/phase0/`)
- Phase 1–5 (run2): not started

## Host / GPU
- 8× NVIDIA H200, 143,771 MiB each
- **GPU: index 0, `CUDA_VISIBLE_DEVICES=0`** (run1 used index 6)
- NVIDIA driver 580.159.03 | CUDA (driver) 13.0 | nvcc release 13.0
- One server at a time (SGLang and vLLM never co-resident)

## Model (identical weights to run1)
- HF ID: `Qwen/Qwen3-VL-8B-Instruct`
- Snapshot SHA: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (re-downloaded; sha256-verified identical to run1)
- Path: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- dtype bfloat16 | arch `Qwen3VLForConditionalGeneration` (dense VL), model_type `qwen3_vl`
- vocab (tokenizer) 151643 | config vocab_size 151936 (embedding) | eos 151645 | bos None | pad 151643 | ChatML
- weights: 4 safetensors shards, ~17 GB, each shard's sha256 == HF content-addressed blob (see `phase0/model_files_sha256.txt`)

## SGLang (system python3 — option 1, NOT in profiling env)
- python `/usr/bin/python3` 3.12.3
- SGLang `0.0.0.dev1+g0c8049d9b` (editable `/sgl-workspace/sglang`); git HEAD `0c8049d9b…` (run1: `ga4cf2ea12`)
- torch 2.11.0+cu130 | flashinfer 0.6.11.post1
- attention: text=flashinfer (pinned `--attention-backend flashinfer`); multimodal=fa3
- chunked_prefill_size 8192 | mem_fraction_static 0.8388 | piecewise_cuda_graph disabled | schedule_policy default **fcfs**
- Phase-0 startup (GPU 0): weight load 16.52 GB / 3.35 s; KV cache 729,076 tokens (K 50.06 + V 50.06 ≈ 100 GB); idle avail ~18.88 GB | port 30000

## vLLM (conda env `/opt/miniconda3/envs/profiling`)
- python 3.12.13 (conda-forge)
- **vLLM 0.21.0** (V1 engine) — run1: 0.19.0
- torch 2.11.0+cu130 | flashinfer 0.6.8.post1 | flash_attn (standalone) not installed (vLLM uses bundled FA)
- transformers 5.9.0, huggingface_hub 1.16.1, numpy 2.3.5, pandas 3.0.3, datasets 4.8.5, aiohttp 3.13.5, requests 2.34.2, tqdm 4.67.3
- attention: text-decoder=**FLASH_ATTN (FlashAttention v3)** (chosen from FLASH_ATTN/FLASHINFER/TRITON_ATTN/FLEX_ATTENTION); vit/mm=FLASH_ATTN; sampling=FlashInfer
- enable_prefix_caching=True, enable_chunked_prefill=True (V1 defaults)
- Phase-0 startup (GPU 0): KV cache 108.15 GiB / 787,488 tokens; gpu_mem_util eff. ~0.92 | port 30001

## Fairness tier assignments (run2, §6 of plan.md)
- **Controlled**: GPU (index 0), model snapshot `0c351dd`, dtype bf16, TP=1, sampler, HF_HUB_OFFLINE
- **Measured-and-reported**:
  - torch/CUDA now **aligned** across both frameworks (both 2.11.0+cu130) — improvement over run1's split (2.9.1+cu129 vs 2.10.0+cu128)
  - attention backend differs: SGLang FlashInfer (text) + fa3 (mm) vs vLLM FlashAttention v3 → Phase-4 attention-kernel findings carry ceiling **M** until aligned
  - FlashInfer version differs between envs: SGLang 0.6.11.post1 vs vLLM-env 0.6.8.post1
  - KV cache size: SGLang ~100 GB vs vLLM 108.15 GiB
- **Framework-intrinsic**: scheduler policy (SGLang fcfs/radix vs vLLM cache manager), CUDA graph shape selection, chunked-prefill scheduling, vLLM prefix caching (ON by default)

## Functional equivalence (run2 Phase 0)
- Tier A PASS (tokenizer/config/template identical; safetensors sha256 verified)
- Tier B **EXACT** byte-identical greedy outputs on all 3 prompts
- Verdict: **PASS** — cleared for run2 Phase 1. Full detail: `experiments/phase0/equivalence.md`.

---

## run1 historical environment (reference only — NOT comparable to run2)
The numbers in `experiments/phase1/`, `experiments/phase2/`, `experiments/phase2_shaping/` were produced under:
- GPU index 6 | CUDA 12.9
- SGLang `0.0.0.dev1+ga4cf2ea12`, torch 2.9.1+cu129, FlashInfer 0.6.7.post3
- vLLM 0.19.0, torch 2.10.0+cu128, FlashAttention v3
- Same model snapshot `0c351dd` (weights identical), but every framework/lib version differs from run2.

Because Controlled (SGLang/vLLM versions, GPU index) and Measured (torch/CUDA/FlashInfer) variables
changed, run1 Phase 1/2 results are historical reference. run2 re-measures Phase 1 from scratch.
