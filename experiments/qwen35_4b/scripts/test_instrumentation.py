#!/usr/bin/env python3
"""CPU-only tests for the DeepStack instrumentation forward-pre-hook
and the profiler-owned test-only BCG allowlist monkey-patch.

Proves that the branch-owned instrumentation:

1. In normal mode, observes a nonzero ``input_deepstack_embeds``
   tensor without changing the module's output.
2. In zero mode, replaces the tensor with ``torch.zeros_like`` before
   the LM forward runs, and the module's observed output differs from
   the normal-mode output for the same inputs.
3. Records a post-zero summary whose ``nonzero_frac == 0`` and
   ``abs_sum == 0`` and whose shape / dtype match the pre-zero
   summary.
4. Removes the pre-hook after the ``general_mm_embed_routine`` call
   returns, so no residual state persists on the module.
5. Does not accumulate hooks across repeated calls (the hook count
   before and after N calls is unchanged).

Proves that the BCG allowlist monkey-patch:

6. When the opt-in flag is set, adds ``Qwen3VLForConditionalGeneration``
   and ``Qwen3VLMoeForConditionalGeneration`` to a stand-in
   ``multimodal_breakable_cuda_graph_supported_model_archs`` list in
   memory, and reports the pre/post state.
7. Is a no-op when neither the env var nor ``force=True`` is set —
   the list is unchanged.
8. Is idempotent — repeated application on an already-patched list
   does not duplicate entries.

Proves the pre-hook generalises to Qwen3-VL:

9. Fires correctly on a toy ``nn.Module`` whose class inherits a name
   from the ``KNOWN_LM_CLASS_NAMES`` set (e.g. a subclass named
   ``Qwen3LLMModel``), tagging
   ``module_class_recognised = true`` in the event log while still
   performing the same 5 behaviours from tests 1-5.

Runs entirely on CPU. No GPU import. No SGLang import. Uses a toy
``nn.Module`` stand-in for the language model plus a stand-in for
``sglang.srt.managers.mm_utils.general_mm_embed_routine``.

Usage
-----

    python3 test_instrumentation.py

Exit 0 on success. Exit 1 on any failure.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import tempfile
from pathlib import Path


def _load_instrumentation(log_path: Path, zero_mode: bool):
    """Import instrumentation.py fresh with the given env, so
    module-level env-var reads see the current settings.
    """
    for k in ("QWEN35_INSTRUMENTATION_LOG", "QWEN35_ZERO_DEEPSTACK",
              "QWEN35_LAUNCH_ID", "QWEN35_CONFIG_LABEL"):
        os.environ.pop(k, None)
    os.environ["QWEN35_INSTRUMENTATION_LOG"] = str(log_path)
    os.environ["QWEN35_ZERO_DEEPSTACK"] = "1" if zero_mode else "0"
    os.environ["QWEN35_LAUNCH_ID"] = "cpu_test"
    os.environ["QWEN35_CONFIG_LABEL"] = "cpu_test_zero" if zero_mode else "cpu_test_normal"

    inst_path = Path(__file__).resolve().parent / "instrumentation.py"
    spec = importlib.util.spec_from_file_location(
        f"qwen35_instrumentation_test_{'zero' if zero_mode else 'normal'}",
        str(inst_path),
    )
    mod = importlib.util.module_from_spec(spec)
    # DO NOT put into sys.modules under a name that would collide with
    # subsequent imports. Fresh module every call.
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _read_log(log_path: Path) -> list[dict]:
    events = []
    if not log_path.is_file():
        return events
    for line in log_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    return events


class _ToyLanguageModel:
    """Stand-in for ``Qwen3_5ForCausalLM``. It's a ``nn.Module`` whose
    forward reads ``input_deepstack_embeds`` from kwargs and adds it
    (broadcasted) to a base tensor.

    The wrapped module's class name is chosen at construction time so
    the ``module_class_recognised`` field on the hook event can be
    exercised for both known (``Qwen3LLMModel``, ``Qwen3_5ForCausalLM``)
    and unknown (fallback ``_LM``) class names.
    """

    def __init__(self, cls_name: str = "_LM"):
        import torch
        import torch.nn as nn

        class _LMBase(nn.Module):
            def __init__(self):
                super().__init__()
                # a trivial parameter so this really is an nn.Module
                self.bias = nn.Parameter(torch.zeros(4))

            def forward(self, input_ids=None, forward_batch=None,
                        input_embeds=None, input_deepstack_embeds=None,
                        **kwargs):
                if input_embeds is None:
                    input_embeds = torch.zeros(4, 4)
                out = input_embeds + self.bias
                if input_deepstack_embeds is not None and input_deepstack_embeds.numel() > 0:
                    # add the FIRST 4 columns just as the real Qwen3.5
                    # LM would do on layer 0.
                    out = out + input_deepstack_embeds[:, :4]
                return out

        # Rebind the class name so ``type(module).__name__`` reflects
        # the requested name — that's what the instrumentation records.
        _LMBase.__name__ = cls_name
        _LMBase.__qualname__ = cls_name
        self.module = _LMBase()

    def __call__(self, *args, **kwargs):
        return self.module(*args, **kwargs)


def _make_toy_routine(instrumentation_mod):
    """Return a stand-in for ``general_mm_embed_routine`` that uses the
    branch-owned ``_patch_general_mm_embed_routine`` wrapping semantics.

    We call the patched routine directly by re-implementing the
    manager module boundary in memory so the test does not import
    SGLang.
    """
    import types

    # Fake ``sglang.srt.managers.mm_utils`` module carrying only what
    # instrumentation.py needs (a callable ``general_mm_embed_routine``
    # and no ``_qwen35_instrumented`` flag).
    fake_mm = types.ModuleType("sglang.srt.managers.mm_utils")

    def real_routine(*args, **kwargs):
        # Emulates the real routine — calls language_model with the
        # kwargs (including input_deepstack_embeds) and returns its
        # output.
        lm = kwargs.get("language_model") or (args[2] if len(args) > 2 else None)
        # Peel off the routine-only kwargs; forward the rest.
        forward_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in {"language_model", "multimodal_model", "input_ids",
                         "forward_batch", "positions", "use_deepstack",
                         "placeholder_tokens", "data_embedding_funcs",
                         "pp_proxy_tensors", "input_embeds"}
        }
        # Match the real routine: it forwards input_deepstack_embeds
        # (and any other extras) as kwargs.
        return lm(
            input_ids=None,
            forward_batch=kwargs.get("forward_batch"),
            input_embeds=kwargs.get("input_embeds"),
            **forward_kwargs,
        )

    fake_mm.general_mm_embed_routine = real_routine
    sys.modules["sglang.srt.managers.mm_utils"] = fake_mm

    # Install the wrapper.
    ok = instrumentation_mod._patch_general_mm_embed_routine()
    if not ok:
        raise AssertionError("wrapper install failed")
    return fake_mm


def _run_one_scenario(zero_mode: bool, toy_class_name: str = "_LM"):
    import torch

    with tempfile.TemporaryDirectory() as td:
        log_path = Path(td) / "inst.jsonl"

        # Fresh SGLang-manager stub for EACH mode: the instrumentation
        # module ships a module-level `_INSTALLED` guard and the mm
        # module ships a `_qwen35_instrumented` marker — both need
        # resetting for a clean scenario.
        for k in list(sys.modules.keys()):
            if k.startswith("sglang.srt.managers.mm_utils"):
                del sys.modules[k]

        inst = _load_instrumentation(log_path, zero_mode=zero_mode)
        mm = _make_toy_routine(inst)

        toy = _ToyLanguageModel(cls_name=toy_class_name)

        # Sanity: initial hook count on the module is zero.
        pre_hooks_before = list(toy.module._forward_pre_hooks.values())
        assert not pre_hooks_before, (
            f"toy module started with pre-hooks: {pre_hooks_before}"
        )

        input_embeds = torch.ones(4, 4)
        ds = torch.arange(1, 4 * 8 + 1, dtype=torch.float32).reshape(4, 8)

        # Call the wrapped routine.
        out = mm.general_mm_embed_routine(
            input_ids=None,
            forward_batch=None,
            language_model=toy.module,
            input_embeds=input_embeds,
            input_deepstack_embeds=ds,
            use_deepstack={},
        )

        # A second call — proves no hook accumulation.
        out2 = mm.general_mm_embed_routine(
            input_ids=None,
            forward_batch=None,
            language_model=toy.module,
            input_embeds=input_embeds,
            input_deepstack_embeds=ds,
            use_deepstack={},
        )

        # Hook count after both calls must be zero again.
        pre_hooks_after = list(toy.module._forward_pre_hooks.values())
        assert not pre_hooks_after, (
            f"toy module retained pre-hooks after routine: {pre_hooks_after}"
        )

        events = _read_log(log_path)
        return out, out2, events, ds, input_embeds


def _assert_events_have_hook_fired(events, expected_count):
    fired = [e for e in events if e.get("event") == "lm_forward_input_deepstack"]
    assert len(fired) == expected_count, (
        f"expected {expected_count} lm_forward_input_deepstack events, "
        f"got {len(fired)}: {[e.get('event') for e in events]}"
    )
    return fired


def _assert_events_have_zeroed(events, expected_count):
    zeroed = [
        e for e in events
        if e.get("event") == "lm_forward_input_deepstack_zeroed"
    ]
    assert len(zeroed) == expected_count, (
        f"expected {expected_count} lm_forward_input_deepstack_zeroed events, "
        f"got {len(zeroed)}: {[e.get('event') for e in events]}"
    )
    return zeroed


def main() -> int:
    print("== toy CPU test — normal mode ==")
    out_n, out_n2, evts_n, ds_n, ie_n = _run_one_scenario(zero_mode=False)
    fired_n = _assert_events_have_hook_fired(evts_n, expected_count=2)
    for e in fired_n:
        ds_sum = (e.get("input_deepstack_embeds") or {})
        assert ds_sum.get("nonzero_frac", 0.0) > 0.0, (
            f"normal mode observed a zero DeepStack tensor: {ds_sum}"
        )
        assert ds_sum.get("abs_sum", 0.0) > 0.0, (
            f"normal mode abs_sum was zero: {ds_sum}"
        )
    zeroed_n = _assert_events_have_zeroed(evts_n, expected_count=0)
    # Sanity: the toy LM output must reflect the added DeepStack.
    import torch
    expected_normal = ie_n + torch.zeros(4) + ds_n[:, :4]
    assert torch.allclose(out_n, expected_normal), (
        "normal-mode toy output did not include the DeepStack contribution"
    )
    print("  normal mode: hook fired 2× with nonzero DeepStack "
          "and no zero-substitution — OK")

    print("== toy CPU test — zero mode ==")
    out_z, out_z2, evts_z, ds_z, ie_z = _run_one_scenario(zero_mode=True)
    fired_z = _assert_events_have_hook_fired(evts_z, expected_count=2)
    zeroed_z = _assert_events_have_zeroed(evts_z, expected_count=2)
    for e in zeroed_z:
        before = e["before"]
        after = e["after"]
        assert before["shape"] == after["shape"], (
            f"zero-substitution changed shape: {before} -> {after}"
        )
        assert before["dtype"] == after["dtype"], (
            f"zero-substitution changed dtype: {before} -> {after}"
        )
        assert after["nonzero_frac"] == 0.0, (
            f"zero-substitution did not zero the tensor: {after}"
        )
        assert after["abs_sum"] == 0.0, (
            f"zero-substitution abs_sum non-zero: {after}"
        )
    expected_zero = ie_z + torch.zeros(4)  # DeepStack contribution absent
    assert torch.allclose(out_z, expected_zero), (
        "zero-mode toy output STILL contains the DeepStack contribution — "
        "the pre-hook did not actually mutate the LM kwargs"
    )
    assert not torch.allclose(out_z, out_n), (
        "zero-mode output equals normal-mode output — ablation had no effect"
    )
    print("  zero mode: hook fired 2× with observed zero replacement; "
          "toy LM output differs from normal-mode by exactly the "
          "DeepStack contribution — OK")

    print("== toy CPU test — hook lifecycle ==")
    # Repeated calls do not accumulate hooks — proved by scenario-level
    # asserts above (pre_hooks_after == 0 after N calls).
    print("  hooks removed after each call, no accumulation — OK")

    print("== toy CPU test — Qwen3-VL generalisation ==")
    # Run the same normal-mode scenario against a toy whose class is
    # named "Qwen3LLMModel" (Qwen3-VL's LM module class). Assert the
    # hook fires, the event marks the module as recognised, and the
    # zero-mode ablation still measurably diverges.
    out_ng, out_ng2, evts_ng, ds_ng, ie_ng = _run_one_scenario(
        zero_mode=False, toy_class_name="Qwen3LLMModel"
    )
    fired_ng = _assert_events_have_hook_fired(evts_ng, expected_count=2)
    for e in fired_ng:
        assert e.get("module_class") == "Qwen3LLMModel", (
            f"Qwen3-VL toy: expected module_class=Qwen3LLMModel, got {e}"
        )
        assert e.get("module_class_recognised") is True, (
            f"Qwen3-VL toy: module_class_recognised was False on event {e}"
        )
        ds_sum = (e.get("input_deepstack_embeds") or {})
        assert ds_sum.get("nonzero_frac", 0.0) > 0.0, (
            f"Qwen3-VL normal-mode observed a zero DeepStack tensor: {ds_sum}"
        )
    _assert_events_have_zeroed(evts_ng, expected_count=0)

    out_zg, out_zg2, evts_zg, ds_zg, ie_zg = _run_one_scenario(
        zero_mode=True, toy_class_name="Qwen3LLMModel"
    )
    fired_zg = _assert_events_have_hook_fired(evts_zg, expected_count=2)
    for e in fired_zg:
        assert e.get("module_class_recognised") is True, (
            f"Qwen3-VL zero-mode: module_class_recognised was False on event {e}"
        )
    zeroed_zg = _assert_events_have_zeroed(evts_zg, expected_count=2)
    for e in zeroed_zg:
        assert e["after"]["nonzero_frac"] == 0.0
        assert e["after"]["abs_sum"] == 0.0
    assert not torch.allclose(out_zg, out_ng), (
        "Qwen3-VL toy: zero-mode output equals normal-mode output — "
        "ablation had no effect on the recognised LM class"
    )
    print("  Qwen3-VL toy (Qwen3LLMModel-named class): pre-hook fires, "
          "module_class_recognised=true, zero ablation diverges — OK")

    # Sanity: an unknown class name is still hooked, but marked
    # module_class_recognised=false. Protects the recognised flag from
    # silently drifting to "always true".
    out_uk, out_uk2, evts_uk, ds_uk, ie_uk = _run_one_scenario(
        zero_mode=False, toy_class_name="TotallyNotAKnownLM"
    )
    fired_uk = _assert_events_have_hook_fired(evts_uk, expected_count=2)
    for e in fired_uk:
        assert e.get("module_class_recognised") is False, (
            f"Unknown class name incorrectly marked recognised: {e}"
        )
    print("  unknown-class toy: hook fires with module_class_recognised=false — OK")

    print("== CPU test — BCG allowlist monkey-patch ==")
    _run_bcg_allowlist_patch_tests()
    print("  BCG allowlist patch: opt-in adds Qwen3-VL classes, opt-out is no-op, "
          "repeated application is idempotent — OK")

    print("all CPU tests passed")
    return 0


def _load_bcg_allowlist_patch():
    """Load ``bcg_allowlist_patch.py`` as a fresh module (independent of
    any prior import in this test process)."""
    import importlib.util

    patch_path = Path(__file__).resolve().parent / "bcg_allowlist_patch.py"
    spec = importlib.util.spec_from_file_location(
        "qwen35_bcg_allowlist_patch_test", str(patch_path)
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run_bcg_allowlist_patch_tests():
    """Test the BCG allowlist patch against a stand-in
    ``sglang.srt.configs.model_config`` module.

    We install a fake module in ``sys.modules`` before loading the
    patch, so the patch's ``install()`` mutates the fake list. This
    keeps the CPU test hermetic — no real SGLang import required.
    """
    import types

    # Clean any prior stubs.
    for k in list(sys.modules.keys()):
        if k.startswith("sglang.srt.configs.model_config"):
            del sys.modules[k]

    # Build stand-in module chain: sglang / sglang.srt / sglang.srt.configs /
    # sglang.srt.configs.model_config.
    def _install_stub(initial_list):
        for pkg in ("sglang", "sglang.srt", "sglang.srt.configs"):
            if pkg not in sys.modules:
                m = types.ModuleType(pkg)
                m.__path__ = []  # mark as package
                sys.modules[pkg] = m
        fake_mc = types.ModuleType("sglang.srt.configs.model_config")
        fake_mc.__file__ = "<fake sglang.srt.configs.model_config>"
        fake_mc.multimodal_breakable_cuda_graph_supported_model_archs = list(initial_list)
        sys.modules["sglang.srt.configs.model_config"] = fake_mc
        return fake_mc

    # --- Test 6a: env var + force=False — no mutation ------------------
    os.environ.pop("QWEN35_PATCH_BCG_ALLOWLIST", None)
    fake = _install_stub(["Qwen3_5ForConditionalGeneration",
                          "Qwen3_5MoeForConditionalGeneration"])
    patch = _load_bcg_allowlist_patch()
    result = patch.install(force=False)
    assert result["enabled"] is False, (
        f"opt-in should be off when env var unset and force=False: {result}"
    )
    assert result["mutated"] is False
    assert "Qwen3VLForConditionalGeneration" not in fake.multimodal_breakable_cuda_graph_supported_model_archs

    # --- Test 6b: env var set — mutation ------------------------------
    os.environ["QWEN35_PATCH_BCG_ALLOWLIST"] = "1"
    fake = _install_stub(["Qwen3_5ForConditionalGeneration",
                          "Qwen3_5MoeForConditionalGeneration"])
    # Reload so ``is_env_enabled()`` at call time sees the updated env.
    patch = _load_bcg_allowlist_patch()
    result = patch.install()
    assert result["enabled"] is True, (
        f"env var should enable the patch: {result}"
    )
    assert result["mutated"] is True
    assert set(result["added"]) == {
        "Qwen3VLForConditionalGeneration",
        "Qwen3VLMoeForConditionalGeneration",
    }
    assert "Qwen3VLForConditionalGeneration" in fake.multimodal_breakable_cuda_graph_supported_model_archs
    assert "Qwen3VLMoeForConditionalGeneration" in fake.multimodal_breakable_cuda_graph_supported_model_archs
    # Pre/post state present and comparable.
    assert result["pre_state"]["contains"]["Qwen3VLForConditionalGeneration"] is False
    assert result["post_state"]["contains"]["Qwen3VLForConditionalGeneration"] is True

    # --- Test 6c: force=True regardless of env ------------------------
    os.environ.pop("QWEN35_PATCH_BCG_ALLOWLIST", None)
    fake = _install_stub(["Qwen3_5ForConditionalGeneration"])
    patch = _load_bcg_allowlist_patch()
    result = patch.install(force=True)
    assert result["enabled"] is True and result["mutated"] is True
    assert "Qwen3VLForConditionalGeneration" in fake.multimodal_breakable_cuda_graph_supported_model_archs

    # --- Test 8: idempotence ------------------------------------------
    # Second application should not duplicate.
    result2 = patch.install(force=True)
    assert result2["enabled"] is True
    assert result2["mutated"] is False, (
        f"second install should be a no-op, got: {result2}"
    )
    assert result2["added"] == []
    # No duplicates.
    l = fake.multimodal_breakable_cuda_graph_supported_model_archs
    for arch in ("Qwen3VLForConditionalGeneration",
                 "Qwen3VLMoeForConditionalGeneration"):
        assert l.count(arch) == 1, (
            f"idempotence broken: arch {arch} appears {l.count(arch)} times in {l}"
        )

    # Cleanup: leave sys.modules without our stubs so subsequent tests
    # or callers do not see the fake sglang.
    for k in list(sys.modules.keys()):
        if k.startswith("sglang"):
            del sys.modules[k]
    os.environ.pop("QWEN35_PATCH_BCG_ALLOWLIST", None)


if __name__ == "__main__":
    sys.exit(main())
