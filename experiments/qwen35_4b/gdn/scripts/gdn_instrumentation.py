"""Minimal GDN instrumentation module.

The DeepStack sub-track's ``server_launcher.py`` requires an
``--instrumentation`` module that exposes ``install()``. For the GDN
baseline sweep we do **not** want any behaviour-changing hooks: the
Nsight profile should reflect the frozen SGLang code path as-is,
without profiler-owned add/replace patches on the model. This module
therefore installs no hooks.

When the investigation escalates to per-op NVTX tagging (see
``validation_plan.md`` §5 "Nsight capture protocol"), replace this
module's ``install()`` with one that wraps
``Qwen3_5GatedDeltaNet.forward`` and each of its op call sites in
``torch.cuda.nvtx.range_push`` / ``range_pop``. That step is gated
on evidence from the baseline profile per the pivot brief.

The module records a small JSON blob to ``/tmp`` so provenance is
auditable: which mode was installed, at what PID, and against what
Python module resolution.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _log_path() -> Path:
    return Path(f"/tmp/qwen35_gdn_instrumentation_{os.getpid()}.json")


def install() -> None:
    """No-op install for the GDN baseline sweep.

    Records a provenance blob under /tmp/. Any future NVTX-tagging
    variant must overwrite this file (or write a sibling) so an
    attempt can be audited from the recorded metadata alone.
    """
    payload = {
        "instrumentation": "gdn_baseline",
        "installs_hooks": False,
        "pid": os.getpid(),
        "python_executable": sys.executable,
        "sglang_module_file": _sglang_module_file(),
        "note": (
            "No behaviour-changing hooks installed. See "
            "experiments/qwen35_4b/gdn/validation_plan.md §5 for the "
            "NVTX-tagging protocol to swap in on evidence-triggered "
            "escalation."
        ),
    }
    try:
        _log_path().write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    except OSError:
        # Provenance logging is best-effort; do not disturb the server.
        pass


def _sglang_module_file() -> str | None:
    try:
        import sglang  # noqa: WPS433
    except Exception:  # noqa: BLE001
        return None
    try:
        return sglang.__file__
    except Exception:  # noqa: BLE001
        return None
