# Environment Snapshot — run2_qwen3vl8b

Generated: 2026-05-21

This is a **new profiling run** on a rebuilt machine. It reuses the *methodology* of the
original run (plan.md) but **all numbers must be re-measured** — see §"Why run2 ≠ run1".
The original `experiments/phase0/1/2`, `datasets/case*.jsonl`, `plan.md`, `README.md` are
untouched historical reference.

## Host / GPU
- GPU: 8× NVIDIA H200, 143,771 MiB each (all idle at run start)
- **This run pins `CUDA_VISIBLE_DEVICES=0`** (run1 used index 6; changed by user 2026-05-21)
- NVIDIA driver: 580.159.03 | CUDA (driver): 13.0 | nvcc: release 13.0, V13.0.88
- One server at a time (SGLang and vLLM never co-resident), same as run1

## Model (identical weights to run1)
- HF ID: `Qwen/Qwen3-VL-8B-Instruct`
- Snapshot SHA: `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` — **same snapshot as run1**, re-downloaded
- Cache path: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- dtype: bfloat16 | architecture: `Qwen3VLForConditionalGeneration` (dense VL), model_type `qwen3_vl`
- Weights: 4 safetensors shards, 750 tensors, total_size 17,534,247,392 B (~17 GB)
- config.json `vocab_size`: 151936 (embedding dim) | tokenizer vocab: 151643 (run1 recorded the tokenizer value)
- eos_token: `<|im_end|>` (151645) | bos: 151643 | pad_token: `<|endoftext|>` | chat_template: ChatML
- Integrity: snapshot hash identical to run1 ⇒ model weights are byte-for-byte the same.

## conda env `profiling` (vLLM side)
- Path: `/opt/miniconda3/envs/profiling` (Miniconda freshly installed to /opt/miniconda3; env built from conda-forge, python 3.12.13)
- **vLLM: 0.21.0** (V1 engine) — run1 was 0.19.0
- torch: 2.11.0+cu130 | CUDA: 13.0
- flashinfer: 0.6.8.post1 | flash_attn (standalone): **not installed** (vLLM uses its bundled FA / FlashInfer)
- transformers 5.9.0, huggingface_hub 1.16.1, numpy 2.3.5, pandas 3.0.3, datasets 4.8.5, aiohttp 3.13.5, requests 2.34.2, tqdm 4.67.3
- vLLM smoke (GPU 0): KV cache 108.15 GiB / 787,488 tokens; gpu_mem_util eff. ~0.92; prefix caching + chunked prefill ON
- vLLM attention: vit/MM-encoder = FLASH_ATTN; sampling = FlashInfer; **text-decoder backend not explicitly logged** (V1 default; pin/log in Phase 1)

## SGLang (system python — option 1, NOT in profiling env)
- python: `/usr/bin/python3` 3.12.3
- SGLang: `0.0.0.dev1+g0c8049d9b` (editable at `/sgl-workspace/sglang/python`); git HEAD `0c8049d9b…` "Update CI permissions and CODEOWNERS (#25826)" — run1 was `ga4cf2ea12`
- torch: 2.11.0+cu130 | flashinfer: 0.6.11.post1
- attention_backend (text): flashinfer (pinned via `--attention-backend flashinfer`); multimodal: fa3 (auto)
- chunked_prefill_size: 8192 | mem_fraction_static: 0.8388 | piecewise_cuda_graph: disabled
- **schedule_policy default is now `fcfs`** (run1 plan referenced `lpm` as default — verify before Phase 2 sweep)
- CUDA graph batch sizes: 1–256 (36 graphs)
- SGLang smoke (GPU 0): weight load 16.52 GB / 3.03 s; KV cache 729,076 tokens (K 50.06 GB + V 50.06 GB ≈ 100 GB); idle avail mem ~20.44 GB

## Smoke test (2026-05-21, GPU 0, max_tokens=8, temperature=0)
Prompt: "What is 2+2? Answer in one word."
- SGLang  → `"  \n**Two**\n\nWait, that's"` (e2e 0.097 s)
- vLLM    → `"  \n**Two**\n\nWait, that's"` (identical text)
- Both servers shut down after; all 8 GPUs verified at 0 MiB. **No benchmark/profiler run.**
- This is an informal equivalence signal only; the formal Phase-0 equivalence matrix is still required before Phase 1.

## Why run2 ≠ run1 — old Phase 1/2 numbers CANNOT be reused
Model weights are identical (same snapshot), but every other dimension changed:
| Dimension | run1 | run2 | Tier (§6 of plan.md) |
|---|---|---|---|
| SGLang git | ga4cf2ea12 | 0c8049d9b | Controlled (changed) |
| vLLM | 0.19.0 | 0.21.0 | Controlled (changed) |
| torch / CUDA | 2.9.1+cu129 / 2.10.0+cu128; CUDA 12.9 | 2.11.0+cu130 (both); CUDA 13.0 | Measured |
| FlashInfer (SGLang) | 0.6.7.post3 | 0.6.11.post1 | Measured |
| FlashInfer (vLLM env) | n/a | 0.6.8.post1 | Measured |
| GPU index | 6 | 0 | Controlled |
Per plan §6.4, any conclusion depending on a changed Controlled/Measured variable must be
re-confirmed. ⇒ **run2 must redo Phase 0 (equivalence) and Phase 1 (baseline) from scratch.**

## Fairness tier assignments (run2)
- Controlled: GPU (index 0), model snapshot (0c351dd), dtype bf16, TP=1, sampler, HF_HUB_OFFLINE
- Measured-and-reported: torch/CUDA (now aligned: both 2.11.0+cu130 — improvement over run1's split), FlashInfer (SGLang 0.6.11 vs vLLM 0.6.8), attention backend (SGLang flashinfer text/fa3 mm vs vLLM FLASH_ATTN), KV cache size (SGLang ~100 GB vs vLLM 108 GiB)
- Framework-intrinsic: scheduler policy (SGLang fcfs/radix vs vLLM cache manager), CUDA graph shape selection, chunked-prefill scheduling, prefix caching (vLLM ON by default)

## Notable run2 vs run1 differences to carry into analysis
1. **torch/CUDA now aligned** across both frameworks (2.11.0+cu130) — removes run1's torch-version mismatch as a Measured confound. Net methodology improvement.
2. **FlashInfer versions differ between the two envs** (0.6.11 vs 0.6.8) — new Measured variable.
3. vLLM **0.21.0 V1 engine** with prefix caching + chunked prefill ON by default — different scheduler surface than run1's 0.19.0.
4. SGLang **schedule_policy default = fcfs** in this build (was lpm-referenced in run1 plan).
