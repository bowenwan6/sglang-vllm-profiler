Qwen3.5-4B BCG DeepStack correctness/path validation attempt
`attempt_gpu7_20260801T013522Z` on the authorised alternate GPU 7,
frozen SGLang `58974ca16...`: verdict `AMBIGUOUS`. `bcg_normal`
exercised BCG replay for both scored image prefills
(`bcg_execute_body_enter` events with `contains_mm_inputs=true`,
`shape_key size=16`, `cuda graph: True` in server stderr, no
`bcg_execute_body_error`), and the greedy text output was
bit-identical to `eager_normal` for every scored request, so
`FEATURE_GAP_EAGER_FALLBACK` and `FAIL_BCG_DEEPSTACK` are both
contraindicated. However, the pre-declared `PASS_BCG_CORRECT`
criterion requires positive evidence that `input_deepstack_embeds`
was non-trivially nonzero at LM forward time, and this evidence is
unavailable: the branch instrumentation's `language_model.__call__`
interceptor is ineffective (it writes to the instance `__dict__`
but `nn.Module` resolves `__call__` on the class), so
`lm_forward_input_deepstack` never fires in any config and
`QWEN35_ZERO_DEEPSTACK=1` had no effect -- making `eager_zero_deepstack`
degenerate to `eager_normal`. This triggers the AMBIGUOUS clause
"ablation arm was corrupted" from `validation_plan.md` sec 6. A
fixture caveat reinforces this: both image prefills produced the
"More image data items provided than corresponding tokens found in
the prompt" warning because the client used `<image>` rather than
Qwen VL's `<|vision_start|><|image_pad|><|vision_end|>` placeholder,
so even a working ablation might not have driven a meaningful
DeepStack contribution against this fixture. No upstream bug is
demonstrated by this attempt; the source-level suspicion is neither
confirmed nor refuted. GPU 7 was clean pre-and-post (4 MiB / 0 %,
zero compute apps), the 11 foreign PIDs on other GPUs are unchanged,
`/data/sglang-fork` still `986c89e69...`, no fix or upstream issue
opened per plan.
