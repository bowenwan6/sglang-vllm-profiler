# Phase 0 — Functional Equivalence

Date: 2026-05-21
GPU: H200 index 0, `CUDA_VISIBLE_DEVICES=0`
Model: Qwen/Qwen3-VL-8B-Instruct @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
SGLang: system python3, editable `/sgl-workspace/sglang` | vLLM: `/opt/miniconda3/envs/profiling`
Servers run strictly serially (SGLang first, then vLLM after full shutdown).

## Tier A — Blockers

| Check | Result | Notes |
|---|---|---|
| Tokenizer byte-equality (5 probes: ASCII, CJK, emoji, code, long) | **PASS** | Same snapshot path on both servers; identical by construction. Probe IDs match run1 exactly |
| Model weight integrity (sha256 of each safetensors shard) | **PASS** | All 4 shards: computed sha256 == HF content-addressed blob filename. See `model_files_sha256.txt` |
| Vocab size | **PASS** | 151,643 (tokenizer) |
| EOS/BOS/PAD ids | **PASS** | eos=151645, bos=None, pad=151643 |
| Chat template (ChatML) | **PASS** | `<\|im_start\|>user\n…<\|im_end\|>\n<\|im_start\|>assistant\n` |

All Tier-A checks pass. No blockers. Identical to run1 Tier-A.

### safetensors sha256 (full hash)
| Shard | Size (B) | sha256 (== blob name) |
|---|---|---|
| model-00001-of-00004 | 4,902,275,944 | d5d0aef0eb170fc7453a296c43c0849a56f510555d3588e4fd662bb35490aefa |
| model-00002-of-00004 | 4,915,962,496 | 8be88fb5501e4d5719a6d4cc212e6a13480330e74f3e8c77daa1a68f199106b5 |
| model-00003-of-00004 | 4,999,831,048 | 83de00eafe6e0d57ccd009dbcf71c9974d74df2f016c27afb7e95aafd16b2192 |
| model-00004-of-00004 | 2,716,270,024 | 0a88b98e9f96270973f567e6a2c103ede6ccdf915ca3075e21c755604d0377a5 |

Total ~17 GB across 4 shards. HF blob filenames are content-addressed sha256; equality with the
recomputed hash proves the weights are intact and identical to what HF serves for this snapshot.

## Tier B — Correctness (greedy: temperature=0, top_p=1, max_tokens=128)

| Check | Result | Notes |
|---|---|---|
| Top-1 first token on 3 greedy prompts | **PASS** | All 3: first token matches |
| Full output match (≤128 tokens greedy) | **PASS (EXACT)** | All 3 outputs byte-identical between SGLang and vLLM |
| Coherent continuation | **PASS** | On-topic, well-formed, no degenerate loops |

Compare script exit code: **0** (Tier-B PASS).

### Greedy output comparison
| Prompt | SGLang | vLLM | Match |
|---|---|---|---|
| "What is 2+2? Answer in one word." | `Four` | `Four` | **EXACT** |
| "Explain gradient descent in exactly one sentence." | "Gradient descent is an iterative optimization algorithm that updates parameters in the direction opposite to the gradient of a loss function to minimize it." | identical | **EXACT** |
| "Write a Python function that reverses a string. Just the code." | ` ```python\ndef reverse_string(s):\n    return s[::-1]\n``` ` | identical | **EXACT** |

No "semantic-level only" annotation needed — divergence did not occur.

## Tier C — Informational (backend / versions / memory)

| Metric | SGLang | vLLM |
|---|---|---|
| Framework version | 0.0.0.dev1+g0c8049d9b | 0.21.0 (V1 engine) |
| torch | 2.11.0+cu130 | 2.11.0+cu130 |
| CUDA | 13.0 | 13.0 |
| Attention backend (text) | FlashInfer 0.6.11.post1 | FLASH_ATTN (FlashAttention v3) |
| Attention backend (vit/mm) | fa3 | FLASH_ATTN |
| Sampling backend | flashinfer | FlashInfer (top-p/top-k) |
| Weight load | 16.52 GB / 3.35 s | (load OK) |
| KV cache | 729,076 tokens (K 50.06 + V 50.06 ≈ 100 GB) | 108.15 GiB / 787,488 tokens |
| Idle avail mem after load+graphs | ~18.88 GB | (gpu_mem_util eff. ~0.92) |

## Conclusion

- ✅ All Tier-A checks pass — both frameworks load the **same snapshot** `0c351dd`, weights sha256-verified intact.
- ✅ tokenizer / config / chat template identical.
- ✅ Tier-B all **EXACT byte-identical** greedy outputs on all 3 prompts.
- ✅ **Phase 0 PASS. Cleared to proceed to Phase 1.**

## Comparison to run1 Phase 0
- **Same verdict as run1**: Tier-A PASS, Tier-B EXACT match. Identical greedy outputs.
- Model weights are byte-identical to run1 (same snapshot, sha256-verified).
- **Methodology improvement vs run1:** torch/CUDA now aligned across both frameworks (both 2.11.0+cu130) — run1 had a torch-version split (SGLang 2.9.1+cu129 vs vLLM 2.10.0+cu128).
- **Attention backend mismatch persists (Measured variable):** SGLang FlashInfer (text) vs vLLM FlashAttention v3. Any Phase-4 attention-kernel finding carries confidence ceiling **M** until backends are aligned — same caveat as run1.
- New Measured variable: FlashInfer version differs *between* the two envs (SGLang 0.6.11.post1 vs vLLM-env 0.6.8.post1).
