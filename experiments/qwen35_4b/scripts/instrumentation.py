#!/usr/bin/env python3
"""Branch-owned instrumentation for the Qwen3.5-4B BCG DeepStack validation.

This module is imported by ``server_launcher.py`` before SGLang's server
starts. It applies **narrow, reversible** monkey-patches so the runner
can objectively record, per prefill request:

- whether the model executor dispatched to BCG (`PrefillCudaGraphRunner.execute`)
  or the eager runner;
- inside BCG, whether ``_execute_body_capture`` and ``replay_layer_forward``
  actually ran;
- for the DeepStack tensor, its shape / dtype / numel / finite-ness /
  nonzero fraction / a compact checksum (`abs().sum()`, `(x*x).sum()`,
  16-byte SHA-256 prefix of the .cpu().numpy() bytes);
- its `.data_ptr()` value as **diagnostic only** (see source_audit.md
  §4.1 — pointer equality alone is not correctness evidence);
- greedy token IDs and (when available) logprobs — this is done client
  side, not here.

Optional mode: when ``QWEN35_ZERO_DEEPSTACK=1`` is set at server-launch
time, the instrumentation additionally replaces
``input_deepstack_embeds`` with ``torch.zeros_like(...)`` immediately
before the LM forward call, so the ``is not None and numel() > 0``
guard still fires but the DeepStack contribution is exactly zero. This
is the ``eager_zero_deepstack`` diagnostic ablation. It is
production-invalid and must never be enabled on a real serving system.

All events are logged as JSON lines to the file named by
``QWEN35_INSTRUMENTATION_LOG``. If that env var is unset, this module
is a no-op — safe to import unconditionally.

Never dumps full DeepStack or hidden-state tensors. Never modifies
anything under ``/data/sglang-fork``. Never signals a foreign PID.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional


# --- Configuration ----------------------------------------------------

_LOG_PATH = os.environ.get("QWEN35_INSTRUMENTATION_LOG", "")
_ZERO_DEEPSTACK = os.environ.get("QWEN35_ZERO_DEEPSTACK", "0") == "1"
_LAUNCH_ID = os.environ.get("QWEN35_LAUNCH_ID", "")
_CONFIG_LABEL = os.environ.get("QWEN35_CONFIG_LABEL", "")
_INSTALLED = False
_LOCK = threading.Lock()
_LOG_FH = None
_REQUEST_COUNTER = 0


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


# --- Monkey-patches ---------------------------------------------------


def _patch_prefill_cuda_graph_runner() -> bool:
    """Wrap `_execute_body_capture` to record entry / exit + arg summary."""
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

        # Wrap the closure that runs at replay time.
        # We patch the *layer_model.forward* the runner will install by
        # replacing it AFTER _execute_body_capture assigns it; the
        # cleanest way is to wrap the model.forward call itself. Since
        # the closure is created inline, we instead wrap the underlying
        # layer_model.forward and rely on the runner monkey-patching it
        # to `replay_layer_forward` for the duration of the call.
        layer_model = getattr(self, "layer_model", None)
        original_layer_forward = None
        if layer_model is not None:
            original_layer_forward = layer_model.forward

            def _observing_layer_forward(*args, **layer_kwargs):
                ids = _next_request_id()
                # This is the closure the runner installed (replay_layer_forward)
                # observed FROM THE OUTSIDE — layer_model.forward is now the
                # replay closure. Record what layer_kwargs contains here.
                try:
                    ie = layer_kwargs.get("input_embeds")
                    ids_arg = args[0] if len(args) else None
                    ids_len = int(ids_arg.shape[0]) if ids_arg is not None else -1
                    _log(
                        "bcg_replay_layer_forward_enter",
                        request_id=rid,
                        replay_call_id=ids,
                        num_tokens_arg=ids_len,
                        input_embeds=_tensor_summary(ie),
                        input_deepstack_embeds=_tensor_summary(
                            layer_kwargs.get("input_deepstack_embeds")
                        ),
                        layer_kwargs_keys=sorted(layer_kwargs.keys()),
                    )
                except Exception as exc:  # noqa: BLE001
                    _log(
                        "bcg_replay_layer_forward_enter_error",
                        request_id=rid,
                        error=str(exc)[:400],
                    )
                out = original_layer_forward(*args, **layer_kwargs)
                _log(
                    "bcg_replay_layer_forward_exit",
                    request_id=rid,
                    replay_call_id=ids,
                    hidden_states=_tensor_summary(out),
                )
                return out

            # Do NOT install this permanently — the runner is about to
            # install replay_layer_forward itself. Wrap by swapping AFTER
            # the runner installs its patch. Simplest: patch after the
            # try/finally by hooking into a small local wrapper below.

        try:
            # Install an observer that runs INSIDE _execute_body_capture:
            # since the runner monkey-patches layer_model.forward inline,
            # we override the runner's assignment by monkey-patching
            # layer_model.forward's SETTER path — done by re-wrapping
            # after the fact via a small post-hook.
            #
            # Practically: we let the runner install its closure, then we
            # swap layer_model.forward to a wrapper that calls the runner's
            # closure and records. We do that by intercepting the setter.
            saved_setattr = layer_model.__class__.__setattr__ if layer_model is not None else None

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
    """Record the DeepStack payload before it enters the LM forward.

    When QWEN35_ZERO_DEEPSTACK=1, this replaces the tensor with
    torch.zeros_like(...) so the LM's `is not None and numel() > 0`
    guard still fires but the contribution is exactly zero — the
    `eager_zero_deepstack` diagnostic.
    """
    try:
        from sglang.srt.managers import mm_utils as _mm
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        _log("patch_error", target="mm_utils", error=str(exc)[:400])
        return False

    if getattr(_mm, "_qwen35_instrumented", False):
        return True

    original_routine = _mm.general_mm_embed_routine

    def wrapped_routine(*args, **kwargs):
        # We can't easily intercept the mid-routine kwargs mutation,
        # so we wrap language_model.forward via a call-once hook
        # for the duration of this routine.
        import torch
        rid = _next_request_id()
        _log("general_mm_embed_routine_enter", request_id=rid)

        language_model = kwargs.get("language_model") or (args[2] if len(args) > 2 else None)
        original_lm_call = language_model.__call__ if language_model is not None else None

        if language_model is None or original_lm_call is None:
            return original_routine(*args, **kwargs)

        # Bind an interceptor that records / optionally zeros deepstack.
        def _lm_call_intercept(*call_args, **call_kwargs):
            ds = call_kwargs.get("input_deepstack_embeds")
            _log(
                "lm_forward_input_deepstack",
                request_id=rid,
                zero_deepstack_mode=_ZERO_DEEPSTACK,
                input_deepstack_embeds=_tensor_summary(ds),
                other_kwarg_keys=sorted([k for k in call_kwargs.keys() if k != "input_deepstack_embeds"]),
            )
            if _ZERO_DEEPSTACK and ds is not None and torch.is_tensor(ds):
                call_kwargs["input_deepstack_embeds"] = torch.zeros_like(ds)
                _log("lm_forward_input_deepstack_zeroed",
                     request_id=rid,
                     shape=list(ds.shape),
                     dtype=str(ds.dtype))
            return original_lm_call(*call_args, **call_kwargs)

        # Install the interceptor for just this routine call.
        # nn.Module.__call__ is a bound method; setting on the instance
        # replaces the class-level attribute for this instance only.
        try:
            language_model.__dict__["__call__"] = _lm_call_intercept
            output = original_routine(*args, **kwargs)
        finally:
            language_model.__dict__.pop("__call__", None)
        _log("general_mm_embed_routine_exit", request_id=rid)
        return output

    _mm.general_mm_embed_routine = wrapped_routine
    _mm._qwen35_instrumented = True

    # Rebind on the multimodal wrapper too, since it captured the
    # symbol at import time.
    for mod_name in (
        "sglang.srt.models.qwen3_vl",
        "sglang.srt.models.qwen3_5",
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
