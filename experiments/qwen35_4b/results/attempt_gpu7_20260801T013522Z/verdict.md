# Verdict -- AMBIGUOUS

Correctness/path validation attempt `attempt_gpu7_20260801T013522Z` on
the authorised alternate GPU 7 under the frozen SGLang checkout
`58974ca16...`, running the three predeclared controls of
`validation_plan.md` sec 5.2: `eager_normal`, `eager_zero_deepstack`,
and `bcg_normal`. Companion data: `metadata.json`, `verdict.json`,
`raw/` per-config evidence.

## Machine verdict

`AMBIGUOUS`.

Per `validation_plan.md` sec 6, the `AMBIGUOUS` label fires when
"instrumentation self-inconsistent, DeepStack was trivially zero,
ablation arm was corrupted, cache state was not matched." Two of
those clauses apply here.

## Actual execution path per scored leg (from instrumentation events + server stderr)

| Config | Scored leg | Path (BCG vs eager) | Evidence |
|---|---|---|---|
| `eager_normal` | P_IMG scored 1 | eager | `bcg_execute_body_enter=0`; server logs `cuda graph: False` for the 14-token image prefill; `--disable-cuda-graph` in effect. |
| `eager_normal` | P_IMG scored 2 (envelope repeat) | eager | `cuda graph: False`. |
| `eager_zero_deepstack` | P_IMG scored 1 | eager | `cuda graph: False`; `--disable-cuda-graph`; `QWEN35_ZERO_DEEPSTACK=1` set in env but never applied (see caveat 2 below). |
| `eager_zero_deepstack` | P_IMG scored 2 | eager | `cuda graph: False`. |
| `bcg_normal` | P_IMG scored 1 | **BCG replay** | `bcg_execute_body_enter` event, `contains_mm_inputs=true`, `shape_key=ShapeKey(size=16, ...)`, `cuda graph: True` in server stderr, matching `bcg_execute_body_exit`, no error. |
| `bcg_normal` | P_IMG scored 2 | **BCG replay** | second `bcg_execute_body_enter` event, `contains_mm_inputs=true`, `shape_key=ShapeKey(size=16, ...)`, `cuda graph: True`, no error. |

Total BCG execute events in `bcg_normal` across the config: 5 (2 image
scored + 2 text prefills + 1 warmup). `can_run_graph` distribution
in `bcg_normal`: 143 true / 2 false (the two false are 1-token
prefills that fall outside the BCG shape buckets).

## DeepStack presence, nonzero fraction, checksum -- NOT MEASURED

None of the three configs emitted an `lm_forward_input_deepstack`
event. The reason is a limitation in the instrumentation:
`_patch_general_mm_embed_routine` attempts to intercept
`language_model.__call__` via
`language_model.__dict__["__call__"] = _lm_call_intercept`. For
`nn.Module` subclasses, Python resolves `__call__` on the class
(`nn.Module.__call__`), not on the instance's `__dict__`, so this
assignment is silently overridden and `_lm_call_intercept` never
runs.

Consequences:

- `input_deepstack_embeds` shape / dtype / numel / nonzero fraction /
  checksum / `.data_ptr()` diagnostic all unavailable for this
  attempt.
- `QWEN35_ZERO_DEEPSTACK=1` in the `eager_zero_deepstack` config had
  no effect. That config is effectively an independent `eager_normal`
  repeat.

## Token / logprob / hidden-state comparisons across legs

Greedy sampling (`temperature=0`, `top_k=1`, fixed seed): the
generated text for the same prompt is expected to be bit-identical
across independent server instances when the code path is equivalent.

- **`P_IMG` scored 1 greedy text (all three configs, identical):**
  `\n\n<think>\n\n</think>\n\nThe image features a vibrant palette of warm oranges, deep reds, and rich browns, accented by touches of green and yellow`
- **`P_IMG` scored 2 greedy text (all three configs, identical):**
  same as scored 1.
- **`P_TXT` scored greedy text (all three configs, identical):**
  `\n\n<think>\nThinking Process:\n\n1.  **Analyze the Request:**\n    *   Input: "Say the words 'hello world' and nothing`

Per-token top-5 logprobs were requested (`return_logprob=True`) and
are in each `raw/client_records_<config>.json`. Given that greedy
texts are bit-identical, the top-5 logprob distributions are also
expected to be equivalent within bf16 noise; a formal per-token
logprob comparison is not needed to reach the verdict. Hidden-state
RMS across layers was not requested by the client
(`--return-hidden-states` defaulted false) since the greedy match
already ruled out `FAIL_BCG_DEEPSTACK`.

## Which bcg_normal track: `eager_normal` or `eager_zero_deepstack`?

`bcg_normal` matches `eager_normal` **and** `eager_zero_deepstack`
exactly. Because the ablation did not fire, this three-way tie is
degenerate:

- Under a working ablation, `bcg_normal == eager_normal` and
  `bcg_normal != eager_zero_deepstack` would score `PASS_BCG_CORRECT`.
- Under the broken ablation, we cannot distinguish "BCG correctly
  preserves DeepStack" from "DeepStack was trivially zero and BCG had
  nothing to preserve either way".

## Verdict paths considered and why exactly one was chosen

| Verdict | Why not chosen |
|---|---|
| `PASS_BCG_CORRECT` | Requires `input_deepstack_embeds.nonzero_frac > 0` evidence. The broken interceptor prevented that measurement. |
| `FEATURE_GAP_EAGER_FALLBACK` | Contraindicated: `bcg_execute_body_enter` fired with `contains_mm_inputs=true` for both scored image prefills; server logged `cuda graph: True` for both. BCG was **not** bypassed. |
| `FAIL_BCG_DEEPSTACK` | Contraindicated: no `bcg_execute_body_error`, no illegal-memory-access, no greedy-text divergence between `bcg_normal` and `eager_normal`. |
| `INFRA_FAILURE` | Contraindicated: server came up on GPU 7 in 108 s, every provenance hard pin matched, no foreign PID appeared on GPU 7, teardown clean, no crash. |
| **`AMBIGUOUS`** | Chosen: the ablation arm was corrupted; DeepStack presence was not verified; the image-placeholder mismatch left the vision-token / DeepStack alignment uncertain. |

## Fixture caveat

Both image prefills in every config produced the SGLang warning
`Warning: More image data items provided than corresponding tokens
found in the prompt.` The client's request payload uses `<image>` as
the image placeholder in the text, but SGLang's Qwen VL processor
(`python/sglang/srt/multimodal/processors/qwen_vl.py`) expects
`<|vision_start|><|image_pad|><|vision_end|>`. The image was still
processed (`contains_mm_inputs=true` in every image-prefill
`bcg_execute_body_enter`) but the placeholder alignment is
imperfect, and this reinforces the AMBIGUOUS classification --
even a working ablation might not have exercised a meaningful
DeepStack contribution against this fixture.

## Is a real upstream correctness bug now demonstrated?

**Not yet.** No divergence, no crash. The source-level suspicion
(BCG has no `input_deepstack_embeds` slot and its replay closure
only forwards `input_embeds`) is not confirmed by runtime evidence
in this attempt; it is also not refuted. The BCG code path served
image requests without producing any output different from the
eager reference, at least for this specific fixture-and-prompt
combination.

## GPU 7 idle interval

Under Amendment 1 of the validation plan (2026-08-01), the
10-continuous-minute idle requirement is waived when the target
GPU is currently in the qualifying state. GPU 7 was at 4 MiB /
0 % / 0 compute apps at every pre-launch check across all three
configs and returned to the same state after each teardown. The
11 foreign compute processes on other GPUs were unchanged
pre-vs-post. No foreign PID was signalled at any point.

## Recommended next step (out of scope for this attempt; no fix here)

- Fix the `language_model.__call__` interceptor to hook the class or
  install an `nn.Module` forward pre-hook instead of writing to the
  instance `__dict__`.
- Use the correct Qwen VL image placeholder in the client's prompt.
- Rerun the three-way comparison on the same frozen SGLang SHA.
- Optionally add hidden-state RMS at layers 0..3 and the final layer
  for a tighter attribution envelope.

None of these fixes are implemented in this attempt (per the plan's
"no fix in this pass" rule). This attempt records the AMBIGUOUS
verdict and closes out Step 5.
