#!/usr/bin/env python3
"""Branch-owned instrumentation for the Qwen3.5-4B / Qwen3-VL BCG DeepStack validation.

This module is imported by ``server_launcher.py`` before SGLang's server
starts. It applies **narrow, reversible** monkey-patches so the runner
can objectively record, per prefill request:

- whether the model executor dispatched to BCG (`PrefillCudaGraphRunner.
  _execute_body_capture`) or the eager runner;
- for the DeepStack tensor, at the moment the language-model forward is
  entered: shape / dtype / numel / finite-ness / nonzero fraction /
  a compact checksum (`abs().sum()`, `(x*x).sum()`, 16-byte SHA-256
  prefix of the .cpu().numpy() bytes);
- its `.data_ptr()` value as **diagnostic only** (see source_audit.md
  §4.1 — pointer equality alone is not correctness evidence);
- when ``QWEN35_ZERO_DEEPSTACK=1`` is set, the second-pass summary
  proving the replacement really is zero (nonzero_frac == 0,
  abs_sum == 0), with matched shape/dtype.

The hook is installed via ``nn.Module.register_forward_pre_hook(
hook, with_kwargs=True)`` on the ``language_model`` instance for the
duration of a single ``general_mm_embed_routine`` call, then removed
in ``finally``. Repeated calls do not accumulate hooks. This replaces
the earlier ``instance-dict __call__`` assignment which nn.Module
silently ignored (nn.Module resolves ``__call__`` at the class level,
not from the instance ``__dict__``).

The pre-hook is installed unconditionally on whatever ``nn.Module`` is
passed as ``language_model=`` by ``general_mm_embed_routine``. On
current upstream that resolves to:

- ``Qwen3_5ForCausalLM`` / ``Qwen3_5MoeForCausalLM`` for Qwen3.5
  (targeted by Attempts 01-02).
- ``Qwen3LLMModel`` / ``Qwen3MoeLLMModel`` for Qwen3-VL (targeted by
  Attempt 03 onward under ``experiments/qwen35_4b/latent_bug_analysis.md``
  § 3, with the profiler-owned BCG allowlist monkey-patch enabled).

The class name of the module the hook lands on is recorded on every
``lm_forward_input_deepstack`` event as ``module_class`` so post-hoc
inspection can confirm targeting.

All events are logged as JSON lines to the file named by
``QWEN35_INSTRUMENTATION_LOG``. If that env var is unset, this module
is a no-op — safe to import unconditionally.

Never dumps full DeepStack or hidden-state tensors. Never modifies
anything under ``/data/sglang-fork``. Never signals a foreign PID.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Optional


# --- Configuration ----------------------------------------------------

_LOG_PATH = os.environ.get("QWEN35_INSTRUMENTATION_LOG", "")
_ZERO_DEEPSTACK = os.environ.get("QWEN35_ZERO_DEEPSTACK", "0") == "1"
_LAUNCH_ID = os.environ.get("QWEN35_LAUNCH_ID", "")
_CONFIG_LABEL = os.environ.get("QWEN35_CONFIG_LABEL", "")
_INSTALLED = False
_LOCK = threading.Lock()
_LOG_FH = None
_REQUEST_COUNTER = 0

# Known SGLang language-model classes that ``general_mm_embed_routine``
# passes as ``language_model=`` for the two model families this
# investigation targets. The pre-hook is installed unconditionally on
# whatever module is passed (see ``_patch_general_mm_embed_routine``);
# this set is only used to tag the recognised-vs-unknown status on the
# ``lm_forward_input_deepstack`` event so post-run inspection can
# confirm the harness landed on the intended module.
KNOWN_LM_CLASS_NAMES = frozenset(
    {
        # Qwen3.5 (Attempts 01-02).
        "Qwen3_5ForCausalLM",
        "Qwen3_5MoeForCausalLM",
        # Qwen3-VL / Qwen3-VL-MoE (Attempt 03+; ``self.model`` in
        # ``Qwen3VLForConditionalGeneration.__init__`` is ``Qwen3LLMModel``
        # for non-MoE and ``Qwen3MoeLLMModel`` for MoE).
        "Qwen3LLMModel",
        "Qwen3MoeLLMModel",
    }
)


def _log(event: str, **fields: Any) -> None:
    """Append one JSON line to the instrumentation log.

    No-op when the log path is unset. Never raises.
    """
    global _LOG_FH
    if not _LOG_PATH:
        return
    payload = {
        "event": event,
        "launch_id": _LAUNCH_ID,
        "config": _CONFIG_LABEL,
        "ts": time.time(),
        "pid": os.getpid(),
        **fields,
    }
    line = json.dumps(payload, default=repr, sort_keys=True) + "\n"
    with _LOCK:
        try:
            if _LOG_FH is None:
                os.makedirs(os.path.dirname(_LOG_PATH), exist_ok=True)
                _LOG_FH = open(_LOG_PATH, "a", buffering=1)
            _LOG_FH.write(line)
        except Exception:  # noqa: BLE001
            # Never let instrumentation take down the server.
            traceback.print_exc(file=sys.stderr)


def _next_request_id() -> int:
    global _REQUEST_COUNTER
    with _LOCK:
        _REQUEST_COUNTER += 1
        return _REQUEST_COUNTER


def _tensor_summary(t: Any) -> dict:
    """Compact, non-destructive summary of a tensor. Never dumps values."""
    try:
        import torch  # local import — module may be loaded on CPU

        if t is None:
            return {"present": False}
        if not torch.is_tensor(t):
            return {"present": True, "type": type(t).__name__}
        summary = {
            "present": True,
            "shape": list(t.shape),
            "dtype": str(t.dtype),
            "numel": int(t.numel()),
            "device": str(t.device),
            "data_ptr_hex": hex(t.data_ptr()) if t.numel() > 0 else "0x0",
        }
        if t.numel() == 0:
            summary.update({"finite": True, "nonzero_frac": 0.0,
                            "abs_sum": 0.0, "sq_sum": 0.0, "sha256_16": ""})
            return summary
        # bf16 needs a promotion for .abs().sum()
        promoted = t.detach().to(torch.float32) if t.dtype in (
            torch.bfloat16, torch.float16
        ) else t.detach()
        summary["finite"] = bool(torch.isfinite(promoted).all().item())
        summary["nonzero_frac"] = float((promoted != 0).to(torch.float32).mean().item())
        summary["abs_sum"] = float(promoted.abs().sum().item())
        summary["sq_sum"] = float((promoted * promoted).sum().item())
        # 16-byte sha256 prefix of the *cpu bytes* — enough to detect
        # tensor value drift without persisting the whole tensor.
        try:
            cpu_bytes = promoted.contiguous().cpu().numpy().tobytes()
            summary["sha256_16"] = hashlib.sha256(cpu_bytes).hexdigest()[:32]
        except Exception:  # noqa: BLE001
            summary["sha256_16"] = ""
        return summary
    except Exception as exc:  # noqa: BLE001
        return {"present": True, "summary_error": str(exc)[:200]}


# --- DeepStack forward-pre-hook ---------------------------------------


def _make_lm_deepstack_pre_hook(rid: int, zero_mode: bool):
    """Return a forward-pre-hook that records / optionally zeros
    ``input_deepstack_embeds`` in the kwargs of one LM forward call.

    The hook signature must match what
    ``nn.Module.register_forward_pre_hook(with_kwargs=True)`` invokes:

        hook(module, args, kwargs) -> None | (args, kwargs)
    """
    fired = {"count": 0}

    def _hook(module, args, kwargs):
        fired["count"] += 1
        try:
            import torch  # noqa: WPS433

            ds = kwargs.get("input_deepstack_embeds")
            before = _tensor_summary(ds)
            module_cls = type(module).__name__
            # Walk the MRO so subclass instances of the known classes
            # still count as "recognised".
            module_mro = [c.__name__ for c in type(module).mro()]
            recognised = any(
                name in KNOWN_LM_CLASS_NAMES for name in module_mro
            )
            _log(
                "lm_forward_input_deepstack",
                request_id=rid,
                zero_deepstack_mode=zero_mode,
                fired_count=fired["count"],
                module_class=module_cls,
                module_class_recognised=recognised,
                module_mro=module_mro[:6],
                input_deepstack_embeds=before,
                other_kwarg_keys=sorted(
                    k for k in kwargs.keys() if k != "input_deepstack_embeds"
                ),
                arg_types=[type(a).__name__ for a in args],
            )
            if zero_mode and ds is not None and torch.is_tensor(ds) and ds.numel() > 0:
                zeroed = torch.zeros_like(ds)
                kwargs["input_deepstack_embeds"] = zeroed
                after = _tensor_summary(kwargs["input_deepstack_embeds"])
                _log(
                    "lm_forward_input_deepstack_zeroed",
                    request_id=rid,
                    fired_count=fired["count"],
                    before=before,
                    after=after,
                )
                # Return the mutated args/kwargs so nn.Module uses them.
                return args, kwargs
        except Exception as exc:  # noqa: BLE001
            _log(
                "lm_forward_input_deepstack_error",
                request_id=rid,
                error=str(exc)[:400],
            )
        # No mutation — signal by returning None (pytorch keeps the
        # unchanged args/kwargs).
        return None

    _hook._qwen35_fired = fired  # type: ignore[attr-defined]
    return _hook


# --- Monkey-patches ---------------------------------------------------


def _patch_prefill_cuda_graph_runner() -> bool:
    """Wrap `_execute_body_capture` to record entry / exit + arg summary.

    This wraps the outer capture-body call so we can attribute a BCG
    replay to a specific request. We do NOT try to intercept the
    inline ``replay_layer_forward`` closure — the previous attempt
    declared a wrapper that was never installed and produced a
    misleading ``bcg_replay_layer_forward_enter`` event that never
    fired. Removed.
    """
    try:
        from sglang.srt.model_executor.runner import (
            prefill_cuda_graph_runner as _pcg,
        )
    except Exception as exc:  # noqa: BLE001
        _log("patch_error", target="prefill_cuda_graph_runner",
             error=str(exc)[:400])
        return False

    Runner = _pcg.PrefillCudaGraphRunner
    if getattr(Runner, "_qwen35_instrumented", False):
        return True

    original_execute_body_capture = Runner._execute_body_capture

    def wrapped_execute_body_capture(
        self,
        forward_batch,
        static_forward_batch,
        static_num_tokens,
        raw_num_tokens,
        shape_key,
        **kwargs,
    ):
        rid = _next_request_id()
        try:
            fb_summary = {
                "batch_size": int(getattr(forward_batch, "batch_size", -1)),
                "num_tokens": int(raw_num_tokens),
                "static_num_tokens": int(static_num_tokens),
                "contains_mm_inputs": bool(
                    getattr(forward_batch, "mm_inputs", None) is not None
                    and any(
                        m is not None for m in (forward_batch.mm_inputs or [])
                    )
                ),
                "input_embeds_present": bool(
                    getattr(forward_batch, "input_embeds", None) is not None
                ),
                "return_logprob": bool(getattr(forward_batch, "return_logprob", False)),
                "shape_key": repr(shape_key),
            }
        except Exception as exc:  # noqa: BLE001
            fb_summary = {"summary_error": str(exc)[:200]}

        _log(
            "bcg_execute_body_enter",
            request_id=rid,
            forward_batch=fb_summary,
        )

        try:
            output = original_execute_body_capture(
                self,
                forward_batch,
                static_forward_batch,
                static_num_tokens,
                raw_num_tokens,
                shape_key,
                **kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            _log(
                "bcg_execute_body_error",
                request_id=rid,
                error=str(exc)[:400],
                exc_type=type(exc).__name__,
                traceback=traceback.format_exc()[-2000:],
            )
            raise
        finally:
            _log("bcg_execute_body_exit", request_id=rid)

        return output

    Runner._execute_body_capture = wrapped_execute_body_capture
    Runner._qwen35_instrumented = True
    _log("patch_ok", target="prefill_cuda_graph_runner")
    return True


def _patch_model_runner() -> bool:
    """Record whether the model_runner dispatched to BCG or eager."""
    try:
        from sglang.srt.model_executor import model_runner as _mr
    except Exception as exc:  # noqa: BLE001
        _log("patch_error", target="model_runner", error=str(exc)[:400])
        return False

    ModelRunner = _mr.ModelRunner
    if getattr(ModelRunner, "_qwen35_instrumented", False):
        return True

    original_forward = ModelRunner.forward

    def wrapped_forward(self, forward_batch, *args, **kwargs):
        rid = _next_request_id()
        try:
            has_mm = bool(
                getattr(forward_batch, "mm_inputs", None) is not None
                and any(m is not None for m in (forward_batch.mm_inputs or []))
            )
            fm = getattr(forward_batch, "forward_mode", None)
            fm_repr = repr(fm) if fm is not None else "<none>"
            in_embeds = getattr(forward_batch, "input_embeds", None) is not None
            _log(
                "model_runner_forward_enter",
                request_id=rid,
                forward_mode=fm_repr,
                contains_mm_inputs=has_mm,
                input_embeds_present=in_embeds,
                batch_size=int(getattr(forward_batch, "batch_size", -1)),
            )
        except Exception as exc:  # noqa: BLE001
            _log("model_runner_forward_enter_error", request_id=rid,
                 error=str(exc)[:400])

        output = original_forward(self, forward_batch, *args, **kwargs)
        try:
            _log(
                "model_runner_forward_exit",
                request_id=rid,
                can_run_graph=bool(getattr(output, "can_run_graph", False)),
            )
        except Exception:  # noqa: BLE001
            pass
        return output

    ModelRunner.forward = wrapped_forward
    ModelRunner._qwen35_instrumented = True
    _log("patch_ok", target="model_runner")
    return True


def _patch_general_mm_embed_routine() -> bool:
    """Install a per-request forward-pre-hook on the ``language_model``
    module for the duration of ``general_mm_embed_routine``.

    The hook records the incoming ``input_deepstack_embeds`` before
    the LM forward runs, and (when ``QWEN35_ZERO_DEEPSTACK=1``) replaces
    it with ``torch.zeros_like``. The handle is removed in ``finally``
    so no hook persists across requests.
    """
    try:
        from sglang.srt.managers import mm_utils as _mm
    except Exception as exc:  # noqa: BLE001
        _log("patch_error", target="mm_utils", error=str(exc)[:400])
        return False

    if getattr(_mm, "_qwen35_instrumented", False):
        return True

    original_routine = _mm.general_mm_embed_routine

    def wrapped_routine(*args, **kwargs):
        rid = _next_request_id()
        _log(
            "general_mm_embed_routine_enter",
            request_id=rid,
            zero_deepstack_mode=_ZERO_DEEPSTACK,
        )

        language_model = kwargs.get("language_model")
        if language_model is None and len(args) > 2:
            language_model = args[2]

        handle = None
        hook = None
        try:
            if language_model is not None and hasattr(
                language_model, "register_forward_pre_hook"
            ):
                hook = _make_lm_deepstack_pre_hook(rid, _ZERO_DEEPSTACK)
                handle = language_model.register_forward_pre_hook(
                    hook, with_kwargs=True
                )
            else:
                _log(
                    "lm_forward_pre_hook_install_error",
                    request_id=rid,
                    reason=(
                        "language_model is None"
                        if language_model is None
                        else f"missing register_forward_pre_hook on {type(language_model).__name__}"
                    ),
                )
            output = original_routine(*args, **kwargs)
        finally:
            if handle is not None:
                try:
                    handle.remove()
                except Exception:  # noqa: BLE001
                    pass
            fired = 0
            try:
                if hook is not None:
                    fired = hook._qwen35_fired["count"]  # type: ignore[attr-defined]
            except Exception:  # noqa: BLE001
                pass
            _log(
                "general_mm_embed_routine_exit",
                request_id=rid,
                lm_forward_hook_fired=fired,
            )
        return output

    _mm.general_mm_embed_routine = wrapped_routine
    _mm._qwen35_instrumented = True

    # Rebind on any multimodal wrapper that captured the symbol at
    # import time.
    for mod_name in (
        "sglang.srt.models.qwen3_vl",
        "sglang.srt.models.qwen3_5",
        "sglang.srt.models.qwen3_vl_moe",
    ):
        try:
            mod = sys.modules.get(mod_name)
            if mod is not None and hasattr(mod, "general_mm_embed_routine"):
                mod.general_mm_embed_routine = wrapped_routine
        except Exception:  # noqa: BLE001
            pass

    _log("patch_ok", target="mm_utils.general_mm_embed_routine")
    return True


def install() -> None:
    """Idempotent monkey-patch install.

    Called from ``server_launcher.py`` immediately before
    ``sglang.launch_server.main()``. Safe to call multiple times.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    _INSTALLED = True

    _log(
        "install_start",
        zero_deepstack=_ZERO_DEEPSTACK,
        log_path=_LOG_PATH,
        launch_id=_LAUNCH_ID,
        config=_CONFIG_LABEL,
    )

    # These imports may fail on CPU-only paths; log and continue.
    ok = True
    ok &= _patch_prefill_cuda_graph_runner()
    ok &= _patch_model_runner()
    ok &= _patch_general_mm_embed_routine()

    _log("install_done", success=bool(ok))


if __name__ == "__main__":
    # Direct invocation prints a small self-check.
    print(
        json.dumps(
            {
                "log_path": _LOG_PATH,
                "zero_deepstack": _ZERO_DEEPSTACK,
                "launch_id": _LAUNCH_ID,
                "config": _CONFIG_LABEL,
                "installed": _INSTALLED,
            },
            indent=2,
        )
    )
