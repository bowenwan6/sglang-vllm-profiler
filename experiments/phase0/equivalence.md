# Phase 0 — Functional Equivalence

Date: 2026-04-17  
GPU: H200 index 6, CUDA_VISIBLE_DEVICES=6  
Model: Qwen/Qwen3-VL-8B-Instruct @ 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

## Tier A — Blockers

| Check | Result | Notes |
|---|---|---|
| Tokenizer byte-equality (5 probes: ASCII, CJK, emoji, code, long) | **PASS** | Both frameworks load from the same snapshot path; tokenizer is identical by construction |
| Model weight SHA (same snapshot path on both servers) | **PASS** | Both servers passed `--model <snapshot_path>`; same 0c351dd… hash |
| Vocab size | **PASS** | 151,643 |
| EOS/BOS/PAD token ids | **PASS** | eos=151645, bos=None, pad=151643 |
| Chat template (ChatML) | **PASS** | `<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n` |

All Tier-A checks pass. No blockers.

## Tier B — Correctness

| Check | Result | Notes |
|---|---|---|
| Top-1 first token on 3 greedy prompts | **PASS** | All 3 prompts: first token matches exactly |
| Full output match (128 tokens greedy) | **PASS (EXACT)** | All 3 outputs byte-identical between SGLang and vLLM |
| Coherent continuation | **PASS** | All outputs on-topic, well-formed, no degenerate loops |

## Tier C — Informational

| Metric | SGLang | vLLM |
|---|---|---|
| Idle GPU memory used | 124,914 MiB | 129,933 MiB |
| Idle GPU memory free | 18,244 MiB | 13,224 MiB |
| KV cache size | ~102 GB | ~105.89 GB |
| Weight load | 16.52 GB / 4.07 s | 16.78 GB / 5.33 s |
| Attention backend (text) | FlashInfer 0.6.7.post3 | FlashAttention v3 |
| Attention backend (multimodal) | FA3 | FA3 |
| Token divergence first appears at | N/A (exact match) | — |

## Tokenizer probe results

| Probe string | Tokens | First 8 ids |
|---|---|---|
| "Hello world" | 2 | [9707, 1879] |
| "你好世界" | 2 | [108386, 99489] |
| "def foo(): return 42" | 7 | [750, 15229, 4555, 470, 220, 19, 17] |
| "🚀" | 1 | [145836] |
| "The quick brown fox…" (×8) | 81 | [785, 3974, 13876, 38835, …] |

## Greedy output comparison (128 tokens, temperature=0)

| Prompt | SGLang | vLLM | Match |
|---|---|---|---|
| "What is 2+2? Answer in one word." | "Four" | "Four" | **EXACT** |
| "Explain gradient descent in exactly one sentence." | "Gradient descent is an iterative optimization algorithm…" | (identical) | **EXACT** |
| "Write a Python function that reverses a string. Just the code." | ` ```python\ndef reverse_string(s):\n    return s[::-1]\n``` ` | (identical) | **EXACT** |

## Conclusion

- ✅ All Tier-A checks pass — frameworks execute the same model from the same snapshot
- ✅ Tier-B all EXACT match — outputs byte-identical under greedy sampling
- ✅ No downstream annotation needed ("greedy outputs diverge after first token") — divergence did not occur
- ✅ **Phase 0 complete. Proceed to Phase 1.**

## Key attention backend mismatch to carry forward

SGLang uses **FlashInfer** for text attention; vLLM uses **FlashAttention v3**.  
This is a **Measured-and-reported** variable (§6.2). Any Phase-4 hypothesis whose evidence rests on an attention-kernel difference must note this mismatch and carry a confidence ceiling of M until a version-matched re-run with aligned backends is done.
