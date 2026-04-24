# Environment Snapshot

Generated: 2026-04-17

## Host
- Host: radixark02, container: sglang-bowenw
- GPU: NVIDIA H200, index 6, 144 GB (CUDA_VISIBLE_DEVICES=6)
- CUDA: 12.9
- HF_HUB_OFFLINE=1 (model fully cached, no network call)

## SGLang
- Version: 0.0.0.dev1+ga4cf2ea12
- torch: 2.9.1+cu129
- FlashInfer: 0.6.7.post3
- attention_backend (text): flashinfer (pinned via --attention-backend flashinfer)
- attention_backend (multimodal): fa3 (auto-selected by SGLang for VL model)
- chunked_prefill_size: 8192
- mem_fraction_static: 0.8388
- max_total_num_tokens: 729090
- CUDA graph batch sizes: 1–256 (36 graphs captured)
- piecewise_cuda_graph: disabled (disable_piecewise_cuda_graph=True)
- Weight load time: 4.07 s, weight size: 16.52 GB
- KV cache allocated: ~102 GB (avail after pool: 20.30 GB → 18.58 GB after graphs)
- GPU memory after load + graphs (idle): 124,914 MiB used / 18,244 MiB free
- Port: 30000

## vLLM
- Version: 0.19.0
- torch: 2.10.0+cu128 (conda env vllm)
- attention_backend (text): FLASH_ATTN (FlashAttention v3, auto-selected)
- attention_backend (multimodal/VIT): FLASH_ATTN (FlashAttention v3)
- gpu_memory_utilization: 0.90 (default)
- CUDA graph: PIECEWISE (51 graphs) + FULL (51 graphs)
- Weight load time: 5.33 s, weight size: 16.78 GB
- KV cache available: 105.89 GiB
- GPU memory after load + graphs (idle): 129,933 MiB used / 13,224 MiB free
- Port: 30001

## Model
- HF ID: Qwen/Qwen3-VL-8B-Instruct
- Snapshot SHA: 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
- dtype: bfloat16
- vocab_size: 151643
- eos_token_id: 151645
- pad_token_id: 151643
- model_max_length: 262144
- chat_template: ChatML (<|im_start|>/<|im_end|>)

## Fairness notes (§6 tier assignments)
- **Controlled**: GPU, model snapshot, dtype, TP=1, sampler, HF_HUB_OFFLINE
- **Measured-and-reported**:
  - torch version differs: SGLang 2.9.1+cu129 vs vLLM 2.10.0+cu128
  - attention_backend differs: SGLang uses FlashInfer (text) + FA3 (mm); vLLM uses FA3 (both)
  - KV cache size differs: SGLang ~102 GB vs vLLM ~105.89 GB (both >100 GB on H200; not a practical constraint at the concurrency levels used in Phase 1)
  - chunked_prefill_size: SGLang 8192 (logged); vLLM default (to be confirmed from log)
- **Framework-intrinsic**: scheduler policy, CUDA graph shape selection, chunked-prefill scheduling
