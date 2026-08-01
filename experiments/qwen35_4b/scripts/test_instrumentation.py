#!/usr/bin/env python3
"""CPU-only tests for the DeepStack instrumentation forward-pre-hook.

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
    """

    def __init__(self):
        import torch
        import torch.nn as nn

        class _LM(nn.Module):
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

        self.module = _LM()

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


def _run_one_scenario(zero_mode: bool):
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

        toy = _ToyLanguageModel()

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

    print("all CPU tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
