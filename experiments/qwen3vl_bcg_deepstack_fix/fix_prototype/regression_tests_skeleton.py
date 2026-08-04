"""Regression-test skeleton for the current-main DeepStack fix.

**Not runnable as-is.** This is a design draft. Each test is fully
specified — inputs, expected outputs, and hard-fail conditions —
so it can be dropped into `test/srt/` once the fix is applied on a
fresh upstream clone.

Tests are ordered by dependency: unit tests first, then integration,
then end-to-end. Each test must FAIL against upstream `main` pre-fix
and PASS post-fix. If a test would pass pre-fix, it is not exercising
the fix and should be dropped from the PR.
"""

import unittest


class TestBufferRegistryDeepStackSlot(unittest.TestCase):
    """Unit tests for the slot registration change.

    Target file:
      python/sglang/srt/model_executor/cuda_graph_buffer_registry.py
    Fix hunk: adds a `num_deepstack_embeddings` parameter and a
    conditional `GraphSlot("input_deepstack_embeds", ...)`.
    """

    def test_slot_registered_when_num_ds_positive(self):
        """Slot is registered iff num_deepstack_embeddings > 0.

        Setup: build_prefill_registry(is_multimodal=True,
        register_input_embeds=True, num_deepstack_embeddings=3,
        hidden_size=4096, embed_dtype=torch.bfloat16, ...).

        Expected: reg.has_slot("input_deepstack_embeds") == True.
        Slot shape at (bs=1, num_tokens=896) resolves to (896, 12288).
        """

    def test_slot_not_registered_when_num_ds_zero(self):
        """Zero-DeepStack models (Qwen3.5 today) see no allocation.

        Setup: build_prefill_registry(is_multimodal=True,
        register_input_embeds=True, num_deepstack_embeddings=0, ...).

        Expected: reg.has_slot("input_deepstack_embeds") == False.
        """

    def test_slot_not_registered_when_not_multimodal(self):
        """Text-only models pay nothing regardless of the parameter.

        Setup: build_prefill_registry(is_multimodal=False, ...,
        num_deepstack_embeddings=3).

        Expected: reg.has_slot("input_deepstack_embeds") == False.
        (The slot registration is nested inside the is_multimodal
        block.)
        """

    def test_slot_not_registered_when_register_input_embeds_false(self):
        """Eager extend path (register_input_embeds=False) also
        skips input_deepstack_embeds — the DeepStack slot is
        gated on the same block as input_embeds."""

    def test_slot_dtype_matches_embed_dtype(self):
        """The slot dtype must match embed_dtype (bfloat16 typical).

        Rationale: the LM's `add_` reads bf16 from the slot; a dtype
        mismatch would either upcast in-graph or fail at capture.
        """

    def test_slot_padding_policy_is_zero(self):
        """PaddingPolicy.ZERO ensures the padded tail is zeros so the
        captured `add_` is a no-op on padded tokens (matches
        input_embeds behaviour)."""


class TestReplayLayerForwardDeepStackCopy(unittest.TestCase):
    """Unit tests for the replay-side copy change.

    Target file:
      python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
    Fix hunk: mirror of the input_embeds copy for
    input_deepstack_embeds.
    """

    def test_copy_happens_when_slot_present_and_kwarg_populated(self):
        """live_deepstack.data_ptr() != slot_buf.data_ptr() before copy;
        after replay_layer_forward runs, slot_buf[:live.shape[0]]
        must byte-equal live_deepstack.

        Setup: mock buffer_registry with a real GraphSlot; call
        replay_layer_forward(input_deepstack_embeds=live_ds,
        input_embeds=live_ie).
        """

    def test_copy_skipped_when_slot_absent(self):
        """When num_ds=0 (Qwen3.5), buffer_registry.has_slot returns
        False; the copy branch is skipped without raising."""

    def test_copy_skipped_when_kwarg_none(self):
        """Text-only requests pass input_deepstack_embeds=None; the
        `de is not None` guard skips the copy."""

    def test_copy_skipped_when_kwarg_numel_zero(self):
        """Empty tensor (0-token or 0-dim); the `numel() > 0` guard
        skips the copy — no crash on empty batches."""

    def test_replay_still_receives_shape_key(self):
        """The backend.replay call must still fire with (shape_key,
        static_forward_batch, **kwargs). This regression test guards
        against a future edit accidentally re-routing kwargs into the
        replay call — mistake symmetric to the pinned-SHA bug."""


class TestCapturePassDeepStackKwarg(unittest.TestCase):
    """Unit tests for the capture-pass kwarg change.

    Target file:
      python/sglang/srt/model_executor/runner/prefill_cuda_graph_runner.py
      _run_forward BCG branch.
    Fix hunk: pass input_deepstack_embeds=<slot> when the slot exists.
    """

    def test_capture_passes_deepstack_kwarg_when_slot_registered(self):
        """The layer_model.forward call must receive
        input_deepstack_embeds=<slot buffer> so Dynamo/capture traces
        the add_ branch.

        Setup: register the slot; monkey-patch layer_model.forward
        to a callable that records its kwargs; drive one capture pass.

        Expected: recorded kwargs contains
        input_deepstack_embeds is slot_buffer.
        """

    def test_capture_omits_deepstack_kwarg_when_slot_absent(self):
        """For num_ds=0 models, capture must not pass the kwarg —
        those LMs' forward signature may not accept it. Backwards-
        compat check."""

    def test_capture_slot_buffer_is_zero_at_capture_time(self):
        """At capture time the slot has not yet been filled by replay
        — the DeepStack `add_` traces as a zero-add. Verify the slot
        contents are all zero when capture invokes the LM."""


class TestEndToEndBcgVsEagerCorrectness(unittest.TestCase):
    """Integration tests. Requires a small VLM (Qwen3-VL-2B or 4B)
    and a byte-pinned image fixture. GPU required — will not run in
    CPU-only CI."""

    def test_bcg_normal_matches_eager_normal_on_qwen3vl_image(self):
        """The regression test that catches the fix.

        Setup: launch two SGLang servers on the same Qwen3-VL-2B
        checkpoint, one eager-only (no BCG), one BCG-only. Under a
        `bcg_allowlist_patch` if the shipped allowlist does not yet
        include Qwen3-VL. Send the same pinned image (SHA-256 hash-
        verified) at the same padded bucket to both.

        Expected: identical output_ids and per-token logprob diff
        within bf16 noise floor (l1_max_abs ≤ 0.1 — the eager-repeat
        noise floor established in the Qwen3.5 sub-track).

        Pre-fix: bcg_normal has 7-token common prefix with
        eager_normal then diverges (Attempt 03 signature); test FAILS.
        Post-fix: full common prefix; test PASSES.
        """

    def test_bcg_matches_eager_on_qwen35_no_regression(self):
        """Ensure the fix does not regress the currently-shipping
        empty-DeepStack case. Qwen3.5-4B image request under BCG
        should still produce identical results pre- and post-fix.

        Expected: bcg vs eager token-identical on Qwen3.5 both pre-
        fix (already true) and post-fix.
        """

    def test_bcg_correct_on_bucket_sweep(self):
        """Repeat the Qwen3-VL image test at padded buckets ∈ {256,
        512, 896, 1024, 2048}. Each bucket must produce a
        token-identical (within bf16 noise) match to eager.

        Guards against slot-lifetime / bucket-shape bugs.
        """

    def test_text_only_requests_unaffected(self):
        """Text-only requests to the same fork-BCG Qwen3-VL server
        must produce identical outputs to eager. DeepStack tensor is
        None on text-only, so the copy is skipped."""

    def test_mixed_batch_correctness(self):
        """Batch containing both image and text-only requests: each
        request produces the correct per-request output; DeepStack
        contribution of the image request does not leak into the
        text request's row."""


class TestPerformanceGates(unittest.TestCase):
    """Performance regression gates. Requires GPU; may be optional in
    upstream CI depending on the reviewer's tolerance."""

    def test_bcg_replay_latency_within_5pct_of_prefix_bcg(self):
        """After the fix, BCG replay latency at each bucket must be
        within 5% of pre-fix BCG replay latency. The extra copy adds
        one memcpy per replay comparable in size to the input_embeds
        copy already present."""

    def test_peak_gpu_memory_within_5pct_of_prefix(self):
        """Registered slot adds max_num_tokens × hidden × num_ds ×
        dtype_size bytes. On Qwen3-VL-8B this is
        4096 × 4096 × 3 × 2 = 96 MiB — measurable but small.
        Peak GPU memory delta must be within 5% or 100 MiB
        (whichever larger)."""


if __name__ == "__main__":
    unittest.main()
