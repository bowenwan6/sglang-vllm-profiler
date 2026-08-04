# Verdict — HARNESS_NOT_DIAGNOSTIC

Harness-validation attempt (Step 2 of the harness-repair pass) on
GPU 1 under the frozen SGLang checkout `58974ca16…` and the
repaired instrumentation / client committed as
`fix(qwen35): repair DeepStack instrumentation and image input`.
Companion data: `metadata.json`, `verdict.json`, `raw/` per-arm
evidence.

## Machine verdict

`HARNESS_NOT_DIAGNOSTIC`.

This is a Step-2-only label defined in the brief and
`validation_plan.md` Amendment 2: it fires when the repaired harness
is verified to instrument correctly but the model / fixture
combination cannot exercise a measurable DeepStack difference under
the zero-DeepStack ablation, so Step 3's scored 2×2 rerun is
skipped by design. It is **not** one of the five predeclared
verdicts (`PASS_BCG_CORRECT` / `FEATURE_GAP_EAGER_FALLBACK` /
`FAIL_BCG_DEEPSTACK` / `AMBIGUOUS` / `INFRA_FAILURE`) and does
not score against any hypothesis in `hypothesis.md` §3.

## Harness repair confirmed on GPU 1

- Repaired `nn.Module.register_forward_pre_hook(..., with_kwargs=True)`
  interceptor **fires 111 times** per arm on real requests (up from
  0 in Attempt 01). The hook installs at
  `general_mm_embed_routine_enter` and is removed in `finally` at
  `_exit`. `lm_forward_input_deepstack` events are attributed to
  `module_class=Qwen3_5ForCausalLM` with the expected kwarg keys
  (`forward_batch`, `input_embeds`, `input_ids`, `positions`,
  `pp_proxy_tensors`) and `arg_types=[]`.
- Corrected multimodal request: prompt renders as
  `<|vision_start|><|image_pad|><|vision_end|>Describe the colours
  in this image in one short sentence.`. Placeholder count = 1,
  image count = 1, zero server-side "More image data items provided
  than corresponding tokens found in the prompt" warnings in either
  arm (down from 2 per arm in Attempt 01).
- Image is really being consumed: greedy output describes the
  fixture's colour palette (`" The image features three vertical
  stripes in red, green, and blue."`) — the byte-pinned
  `image_bands.png` renders three ~426 px bands in muted RGB.

## Why the ablation is non-diagnostic on this model

`vision_config.deepstack_visual_indexes = []` on **every publicly
released Qwen/Qwen3.5-\* checkpoint** at HuggingFace (0.8B, 2B, 4B,
9B, 27B, 35B-A3B, all with architecture
`Qwen3_5ForConditionalGeneration` /
`Qwen3_5MoeForConditionalGeneration`). Consequence chain:

1. `Qwen3VLForConditionalGeneration.__init__` reads
   `self.deepstack_visual_indexes = config.vision_config.deepstack_visual_indexes`
   (`models/qwen3_vl.py:1302` in the pinned SGLang) → `[]`.
2. `self.num_deepstack_embeddings = len([]) = 0`.
3. In `general_mm_embed_routine`
   (`managers/mm_utils.py:1116-1129`),
   `deepstack_embedding_shape = input_embeds.shape[:-1] + (input_embeds.shape[-1] * 0,)`
   → the DeepStack tensor is allocated with shape `(N, 0)` /
   `numel = 0`.
4. Instrumentation confirms this at runtime: every observed
   `lm_forward_input_deepstack` event carries shape `[893, 0]` or
   `[80, 0]` and `numel = 0`.
5. `Qwen3_5ForCausalLM.forward`
   (`models/qwen3_5.py:1449-1457`) gates DeepStack on
   `input_deepstack_embeds is not None and input_deepstack_embeds.numel() > 0`
   — trivially False, so the `add_` branch never runs.
6. Consequently the zero-DeepStack pre-hook has nothing to
   substitute — the guard `if zero_mode and ds is not None and
   torch.is_tensor(ds) and ds.numel() > 0` correctly skips the
   `torch.zeros_like` replacement, and the arm behaves identically
   to `eager_normal`.
7. Greedy text is bit-identical across `eager_normal` and
   `eager_zero_deepstack` for both scored image requests.

## What was actually verified vs left untested

| Verified this attempt | Not verified (needs a DeepStack-active checkpoint) |
|---|---|
| Repaired pre-hook installs / fires / cleans up correctly | Whether BCG replay retains a **non-empty** DeepStack tensor |
| Corrected placeholder produces zero SGLang alignment warnings | Whether `bcg_normal` and `bcg_zero_deepstack` diverge for image requests |
| Image data reaches the LM (output describes the fixture correctly) | Whether the source-level BCG DeepStack suspicion (F5–F8) produces runtime divergence |
| Instrumentation clearly reports `numel = 0` for empty DeepStack | Any test of Qwen3-VL-8B DeepStack under BCG (not this investigation's target) |

## Verdict paths considered

| Verdict | Why not chosen |
|---|---|
| `PASS_BCG_CORRECT` | Would require BCG arms; not run. Even if run, DeepStack is trivially zero on this checkpoint, so `bcg_normal == bcg_zero` and the required `bcg_normal != bcg_zero` cannot hold. |
| `FEATURE_GAP_EAGER_FALLBACK` | Would require BCG arms; not run. |
| `FAIL_BCG_DEEPSTACK` | Would require BCG arms; not run. Even if run, the DeepStack `add_` branch cannot fire on this checkpoint (numel is 0), so a BCG-vs-eager divergence attributable to DeepStack is unreachable. |
| `AMBIGUOUS` | Step 3 was deliberately skipped per the brief's Step 2 fail path; no scored 2×2 result exists to score. |
| `INFRA_FAILURE` | Contraindicated. Preflight PASS, imported SGLang `INSIDE_FROZEN`, `/data/sglang-fork` HEAD unchanged, GPU 1 acquired cleanly (0 MiB / 0 % / 0 compute apps pre and post), no foreign PID contact. |
| **`HARNESS_NOT_DIAGNOSTIC`** | Chosen per the brief's Step 2 fail-path rule: the repaired harness works, but the model+fixture is not diagnostic, so do not run Step 3. |

## Implications for the investigation

- The branch-owned harness repair in commit
  `fix(qwen35): repair DeepStack instrumentation and image input`
  is functionally correct on GPU: the DeepStack pre-hook fires with
  the correct kwargs on real Qwen3.5 requests, and the corrected
  multimodal request produces no alignment warnings.
- The BCG DeepStack correctness hypothesis (H_A) **cannot be
  validated against any publicly released Qwen/Qwen3.5-\* checkpoint**
  at the pinned SGLang SHA, because `deepstack_visual_indexes` is
  empty on every one of those checkpoints.
- The source-level suspicion (F5, F6, F7, F8: BCG has no
  `input_deepstack_embeds` slot and its replay closure only forwards
  `input_embeds`) remains **neither confirmed nor refuted** by
  runtime evidence — untestable against this model family.
- A follow-up validation would need to target a Qwen3-VL or
  Qwen3-Omni checkpoint whose config ships a non-empty
  `deepstack_visual_indexes` list (e.g. `Qwen/Qwen3-VL-8B`, which is
  already the target of a distinct PCG investigation on
  `debug/v2-imgA-pcg-capture-stream-fix`). That is outside the §7
  scope of this branch.

## GPU 1 acquisition and safety evidence

- Pre-run: GPU 1 = 0 MiB / 0 % / 0 compute apps (Amendment 1
  idle-waiver applies: target GPU is already qualifying).
- Foreign-PID snapshot at start: 11 compute apps distributed over
  6 other GPUs (0, 2, 3, 4, 5, 6, 7). GPU 1 has none.
- Post `eager_normal`: GPU 1 back to 0 MiB / 0 % / 0 compute apps;
  runner signalled only its own PGID (917012) on cleanup.
- Post `eager_zero_deepstack`: GPU 1 back to 0 MiB / 0 % / 0 compute
  apps; runner signalled only its own PGID (918714).
- No foreign PID was signalled at any point. No
  `nvidia-smi --gpu-reset`. No `pkill` / `killall`.
- `/data/sglang-fork` HEAD unchanged at `986c89e69…` on branch
  `fix/pcg-vlm-deepstack-warmup`.

## Reporting rule

`HARNESS_NOT_DIAGNOSTIC` is Step-2-only. It records that the
harness is now trustworthy and that the target checkpoint cannot
produce a diagnostic ablation. The scored 2×2 rerun (Step 3) is
skipped by design; the branch closes out at this commit until a
DeepStack-active model target is authorised.
