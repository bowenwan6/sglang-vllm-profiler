# BCG DeepStack — cross-arch audit and retarget plan

> Follow-up analysis after `harness_gpu1_20260801T062833Z` recorded
> `HARNESS_NOT_DIAGNOSTIC`. Written 2026-08-01 against frozen SGLang
> `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`.

## 1. Why Attempt 01 and the harness-validation attempt could not decide

`harness_gpu1_20260801T062833Z` proved the repaired harness works on GPU
(pre-hook fires, correct visual-placeholder tokens, image consumed).
It also proved that the target model — any `Qwen/Qwen3.5-*` checkpoint —
cannot exercise DeepStack, because every shipped Qwen3.5 config carries
`vision_config.deepstack_visual_indexes = []`.

That surfaced a broader question: **is there any current SGLang-supported
model that both enters BCG replay and populates DeepStack?**

## 2. Cross-arch audit

### 2.1 DeepStack in shipped checkpoints (HF API, 2026-08-01)

| Model | `vision_config.deepstack_visual_indexes` | Populates DeepStack? |
|---|---|---|
| `Qwen/Qwen3.5-4B` | `[]` | No |
| `Qwen/Qwen3.5-2B` | `[]` | No |
| `Qwen/Qwen3.5-0.8B` | `[]` | No |
| `Qwen/Qwen3.5-9B` | `[]` | No |
| `Qwen/Qwen3.5-27B` | `[]` | No |
| `Qwen/Qwen3.5-35B-A3B` (MoE) | `[]` | No |
| `Qwen/Qwen3-VL-8B-Instruct` | `[8, 16, 24]` | **Yes (3 layers)** |
| `Qwen/Qwen3-VL-4B-Instruct` | `[5, 11, 17]` | **Yes (3 layers)** |
| `Qwen/Qwen3-VL-2B-Instruct` | `[5, 11, 17]` | **Yes (3 layers)** |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` (MoE) | `[8, 16, 24]` | **Yes (3 layers)** |

Command: `curl -sSf https://huggingface.co/$m/raw/main/config.json |
jq .vision_config.deepstack_visual_indexes`.

### 2.2 BCG allowlist × per-file DeepStack references

| Arch (declared in shipped `config.json`) | On BCG allowlist? | `grep -c deepstack` in model source | DeepStack in any shipped release? |
|---|---|---|---|
| `Cohere2VisionForConditionalGeneration` | ✅ | 0 (`cohere2_vision.py`) | n/a |
| `KimiK25ForConditionalGeneration` | ✅ | 0 (`kimi_k25.py`) | n/a |
| `MiniMaxM3SparseForCausalLM` | ✅ | 0 (`minimax_m3.py`) | n/a |
| `MiniMaxM3SparseForConditionalGeneration` | ✅ | 0 (`minimax_m3_vl.py`) | n/a |
| `Qwen3_5ForConditionalGeneration` | ✅ | 7 (`qwen3_5.py`, scaffolded) | **No** — every release ships `[]` |
| `Qwen3_5MoeForConditionalGeneration` | ✅ | inherits from `qwen3_5.py` | **No** — release ships `[]` |
| `Qwen3VLForConditionalGeneration` | ❌ | **43** (`qwen3_vl.py`, primary user) | **Yes** — every release populates |
| `Qwen3VLMoeForConditionalGeneration` | ❌ | 18 (`qwen3_vl_moe.py`) | **Yes** — release populates |

BCG allowlist location: `python/sglang/srt/configs/model_config.py:1845-1848` at
frozen `58974ca1`.

### 2.3 The gating rule

`server_args.py:4460-4470`:

```python
# Multimodal prefill replay faults under BCG; allowlisted archs opt back in.
(
    "multimodal model",
    lambda: self.get_model_config().is_multimodal
    and not self.get_model_config().is_multimodal_breakable_cuda_graph_supported,
),
```

Any multimodal arch not on the BCG allowlist gets BCG prefill auto-disabled
with a warning. Qwen3-VL therefore cannot enter BCG replay on current upstream
without a source patch or runtime monkey-patch.

### 2.4 Conclusion

**The intersection of "on BCG allowlist" and "actually populates DeepStack" is
empty on current upstream.** The source-level suspicion is unchanged:
`replay_layer_forward` copies `input_embeds` into a stable slot but does not
touch `input_deepstack_embeds`, and the buffer registry has no DeepStack slot.
That code path is real. It just isn't reachable in production today.

The bug is a **latent regression** that would activate if either:

- **(a)** a future `Qwen/Qwen3.5-*` release ships with
  `vision_config.deepstack_visual_indexes != []`, or
- **(b)** `Qwen3VLForConditionalGeneration` (or its MoE variant) is added to
  `multimodal_breakable_cuda_graph_supported_model_archs`.

## 3. Retarget plan (attempt 03 onward)

To convert the latent-bug hypothesis into runtime evidence, install a
**profiler-owned runtime monkey-patch** in `server_launcher.py` that overrides
`is_multimodal_breakable_cuda_graph_supported` to return `True` for
`Qwen3VLForConditionalGeneration` (path b above), then rerun the 2×2 with
`Qwen/Qwen3-VL-8B-Instruct @ 0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

Design notes:

- **The frozen SGLang source is not modified.** The monkey-patch lives in our
  profiler-owned `scripts/server_launcher.py` (or a sibling module), is
  installed after `sglang` is imported but before `launch_server` runs, and is
  clearly labelled `# TEST-ONLY: reproduces latent bug in
  replay_layer_forward's missing DeepStack copy`.
- **Evidence hierarchy is unchanged.** All 5 verdict labels remain valid; a
  live-fire `FAIL_BCG_DEEPSTACK` obtained under the monkey-patch is direct
  evidence of the source-level bug, marked with the "obtained under monkey-
  patched allowlist" caveat.
- **Model swap is symmetric.** The repaired instrumentation targets the LM
  module by class name (`Qwen3_5ForCausalLM`); we need a small generalization
  to also match `Qwen3VLForCausalLM` (or whatever the language-model class
  name is in `qwen3_vl.py`).
- **Fixture stays deterministic.** Reuse the byte-pinned image fixture from
  Attempt 01 (SHA-256 `8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`);
  swap the tokenizer/processor to Qwen3-VL-8B's pinned revision so the visual
  placeholder tokens match the new target.
- **New attempt directory.** Never reuse `attempt_gpu7_20260801T013522Z/` or
  `harness_gpu1_20260801T062833Z/`. Attempts 01 and 02 (`AMBIGUOUS` and
  `HARNESS_NOT_DIAGNOSTIC`) are preserved verbatim.
- **Verdict schema unchanged.** The 5-label schema from `hypothesis.md` §5 and
  the 2×2 verdict rules from the previous brief still apply.

## 4. Upstream implications regardless of runtime outcome

Even without runtime evidence, the source review already justifies a
defensive upstream note: `replay_layer_forward` should either

- **register an `input_deepstack_embeds` slot** and copy the live tensor into
  it (parallel with `input_embeds`), *or*
- **explicitly assert** that models on the BCG allowlist do not use DeepStack
  (numel guard on `input_deepstack_embeds` before capture), which would fail
  fast the moment a Qwen3.5 release populates DeepStack or Qwen3-VL is added.

Which framing to file with upstream depends on the retarget runtime outcome.
Do not file anything until attempt 03 verdict is in.

## 5. What this document is not

- Not a fix. `plan.md` §7 still says "no fix in this pass."
- Not a claim that upstream has an active production bug — no shipped config
  triggers it.
- Not a verdict on Qwen3-VL's own BCG suitability outside this specific
  DeepStack question.
