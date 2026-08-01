#!/usr/bin/env python3
"""Profiler-owned test-only BCG allowlist monkey-patch.

TEST-ONLY: reproduces latent bug in ``replay_layer_forward``'s missing
DeepStack copy. Enabled via ``QWEN35_PATCH_BCG_ALLOWLIST=1`` or
``--patch-bcg-allowlist``. See
``experiments/qwen35_4b/latent_bug_analysis.md`` § 3.

Rationale
---------

At frozen SGLang ``58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8``, the
list ``sglang.srt.configs.model_config.multimodal_breakable_cuda_graph_supported_model_archs``
contains ``Qwen3_5ForConditionalGeneration`` and
``Qwen3_5MoeForConditionalGeneration`` — both of which ship
``vision_config.deepstack_visual_indexes = []`` on every publicly
released Qwen3.5 checkpoint. The intersection of "on BCG allowlist"
and "actually populates DeepStack" is therefore empty, and the
source-level suspicion in ``latent_bug_analysis.md`` § 2 cannot be
exercised live-fire against any shipped model without a source
patch or a runtime monkey-patch.

This module mutates the allowlist **in memory only**, after
``sglang`` is imported and before ``launch_server`` runs, to include
``Qwen3VLForConditionalGeneration`` (primary target) and
``Qwen3VLMoeForConditionalGeneration`` (symmetry / future-proofing).
The frozen SGLang checkout's source files are never modified — the
mutation is a runtime ``list.append`` on the module-level list
object.

The patch is **opt-in**. It only fires when

- ``QWEN35_PATCH_BCG_ALLOWLIST=1`` is in the process environment, or
- the caller invokes ``install(force=True)`` (e.g. because a runner
  flag was passed).

The patch is **idempotent** — repeated application does not duplicate
entries.

The pre-mutation and post-mutation allowlist snapshots are returned
by ``install()`` and are written to the launch-context JSON by
``server_launcher.py`` for provenance auditing.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any


# Attribute name on the model_config module we mutate.
_ALLOWLIST_ATTR = "multimodal_breakable_cuda_graph_supported_model_archs"

# Classes we add to the allowlist for the retarget study. Order is
# preserved to keep the runtime log stable and diffable.
_ARCHS_TO_ADD = (
    "Qwen3VLForConditionalGeneration",
    "Qwen3VLMoeForConditionalGeneration",
)


def is_env_enabled() -> bool:
    """True when the caller-visible env var opts the patch in."""
    return os.environ.get("QWEN35_PATCH_BCG_ALLOWLIST", "0") == "1"


def read_allowlist_snapshot() -> dict[str, Any]:
    """Import the model_config module and return a snapshot of the
    allowlist plus a per-target contains-check.

    Does not mutate anything. Safe to call at any time after ``sglang``
    is importable.
    """
    try:
        from sglang.srt.configs import model_config as _mc  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "importable": False,
            "error": str(exc)[:400],
            "module_path": None,
            "allowlist": None,
            "contains": {a: False for a in _ARCHS_TO_ADD},
        }

    allowlist = list(getattr(_mc, _ALLOWLIST_ATTR, []) or [])
    return {
        "importable": True,
        "module_path": getattr(_mc, "__file__", None),
        "allowlist": allowlist,
        "contains": {a: (a in allowlist) for a in _ARCHS_TO_ADD},
    }


def install(force: bool = False) -> dict[str, Any]:
    """Idempotently mutate the BCG allowlist to include the Qwen3-VL
    architectures. Returns a JSON-serialisable summary of pre-state
    and post-state suitable for the launch-context JSON.

    Parameters
    ----------
    force:
        If True, apply the patch regardless of ``QWEN35_PATCH_BCG_ALLOWLIST``.
        The runner uses this when the operator passed ``--patch-bcg-allowlist``.
    """
    enabled = force or is_env_enabled()
    if not enabled:
        pre = read_allowlist_snapshot()
        return {
            "target_archs": list(_ARCHS_TO_ADD),
            "enabled": False,
            "reason": "opt-in flag/env not set",
            "pre_state": pre,
            "post_state": pre,  # unchanged
            "mutated": False,
        }

    # We must mutate the module attribute in place so any reference
    # captured elsewhere (e.g. by ``is_multimodal_breakable_cuda_graph_supported``
    # closures) also sees the new entries.
    try:
        from sglang.srt.configs import model_config as _mc  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return {
            "target_archs": list(_ARCHS_TO_ADD),
            "enabled": True,
            "mutated": False,
            "error": f"cannot import sglang.srt.configs.model_config: {exc!r}",
        }

    pre_list = list(getattr(_mc, _ALLOWLIST_ATTR, []) or [])
    pre = {
        "importable": True,
        "module_path": getattr(_mc, "__file__", None),
        "allowlist": pre_list,
        "contains": {a: (a in pre_list) for a in _ARCHS_TO_ADD},
    }

    # Idempotent append.
    live_list = getattr(_mc, _ALLOWLIST_ATTR)
    if not isinstance(live_list, list):
        # Defensive: refuse to replace an unexpected type.
        return {
            "target_archs": list(_ARCHS_TO_ADD),
            "enabled": True,
            "mutated": False,
            "error": (
                f"{_ALLOWLIST_ATTR} is not a list (got "
                f"{type(live_list).__name__}); refusing to mutate."
            ),
            "pre_state": pre,
        }

    added = []
    for arch in _ARCHS_TO_ADD:
        if arch not in live_list:
            live_list.append(arch)
            added.append(arch)

    post_list = list(live_list)
    post = {
        "importable": True,
        "module_path": getattr(_mc, "__file__", None),
        "allowlist": post_list,
        "contains": {a: (a in post_list) for a in _ARCHS_TO_ADD},
    }

    return {
        "target_archs": list(_ARCHS_TO_ADD),
        "enabled": True,
        "mutated": bool(added),
        "added": added,
        "pre_state": pre,
        "post_state": post,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI self-check: prints the current allowlist snapshot as JSON.

    Does not mutate anything unless ``--apply`` is passed. Intended
    for manual dry-runs and cross-checks; the runner does not invoke
    this entry point.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    apply = "--apply" in argv
    result = install(force=apply) if apply else {
        "target_archs": list(_ARCHS_TO_ADD),
        "enabled": is_env_enabled(),
        "mutated": False,
        "pre_state": read_allowlist_snapshot(),
    }
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
