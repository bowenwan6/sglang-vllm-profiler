#!/usr/bin/env python3
"""Server launcher wrapper for the Qwen3.5-4B BCG DeepStack validation.

Applies ``instrumentation.install()`` before delegating to
``sglang.launch_server.main()`` so the branch-local monkey-patches are
in place when the model loads.

The wrapper does NOT parse SGLang args itself; it forwards every argv
after ``--`` to ``sglang.launch_server`` unchanged. It also enforces
that the imported ``sglang`` resolves inside the frozen checkout — a
runtime guard on top of the runner's ``PYTHONPATH`` override.

Usage
-----

    python3 server_launcher.py \\
        --instrumentation /path/to/instrumentation.py \\
        --frozen-sglang <checkout> \\
        --frozen-sglang-sha <sha> \\
        [--patch-bcg-allowlist] \\
        [--patch-log <path>] \\
        -- \\
        --model-path Qwen/Qwen3-VL-8B-Instruct \\
        --port 30000 \\
        ...

The optional ``--patch-bcg-allowlist`` flag (or ``QWEN35_PATCH_BCG_ALLOWLIST=1``
in the environment) installs the profiler-owned test-only BCG allowlist
monkey-patch defined in ``scripts/bcg_allowlist_patch.py``. See that
module's docstring and ``experiments/qwen35_4b/latent_bug_analysis.md``
§ 3 for the rationale. The launcher writes the pre-mutation /
post-mutation allowlist snapshots to ``--patch-log`` (default:
``/tmp/qwen35_bcg_allowlist_patch_<pid>.json``) so provenance is
auditable.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path


def _abort(msg: str, code: int = 2) -> None:
    print(f"server_launcher: FATAL: {msg}", file=sys.stderr)
    sys.exit(code)


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--instrumentation", required=True, type=Path)
    parser.add_argument("--frozen-sglang", required=True, type=Path)
    parser.add_argument("--frozen-sglang-sha", required=True)
    parser.add_argument(
        "--patch-bcg-allowlist",
        action="store_true",
        help=(
            "Install the test-only BCG allowlist monkey-patch that adds "
            "Qwen3-VL classes to "
            "multimodal_breakable_cuda_graph_supported_model_archs at runtime. "
            "See scripts/bcg_allowlist_patch.py and "
            "experiments/qwen35_4b/latent_bug_analysis.md § 3."
        ),
    )
    parser.add_argument(
        "--patch-log",
        type=Path,
        default=None,
        help=(
            "Where to write the pre/post allowlist snapshot as JSON. Default: "
            "/tmp/qwen35_bcg_allowlist_patch_<pid>.json"
        ),
    )
    parser.add_argument("--help", "-h", action="store_true")
    # Everything after -- is forwarded to sglang.launch_server.
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        my_argv = sys.argv[1:idx]
        forwarded = sys.argv[idx + 1 :]
    else:
        my_argv = sys.argv[1:]
        forwarded = []
    args = parser.parse_args(my_argv)

    if args.help:
        parser.print_help()
        return 0

    # Load instrumentation module by path so it is unambiguous.
    inst_path = args.instrumentation.resolve()
    if not inst_path.is_file():
        _abort(f"instrumentation module not found: {inst_path}")
    spec = importlib.util.spec_from_file_location(
        "qwen35_instrumentation", str(inst_path)
    )
    if spec is None or spec.loader is None:
        _abort(f"cannot load instrumentation module: {inst_path}")
    inst = importlib.util.module_from_spec(spec)
    sys.modules["qwen35_instrumentation"] = inst
    spec.loader.exec_module(inst)  # type: ignore[union-attr]

    # Verify frozen checkout is on PYTHONPATH BEFORE we import sglang.
    frozen = args.frozen_sglang.resolve()
    frozen_python = frozen / "python"
    if not (frozen_python / "sglang" / "__init__.py").is_file():
        _abort(f"frozen sglang checkout missing sglang/ under {frozen_python}")
    if str(frozen_python) not in sys.path:
        sys.path.insert(0, str(frozen_python))

    # Verify HEAD matches the pin.
    head_file = frozen / ".git" / "HEAD"
    if head_file.is_file():
        # Resolve ref.
        head_val = head_file.read_text().strip()
        if head_val.startswith("ref: "):
            ref_path = frozen / ".git" / head_val[len("ref: "):]
            if ref_path.is_file():
                head_val = ref_path.read_text().strip()
        if head_val != args.frozen_sglang_sha:
            _abort(
                f"frozen sglang HEAD ({head_val}) != pin "
                f"({args.frozen_sglang_sha})"
            )

    # Assert `sglang` resolves inside the frozen checkout.
    import sglang  # noqa: WPS433
    sglang_path = Path(sglang.__file__).resolve()
    if not str(sglang_path).startswith(str(frozen_python)):
        _abort(
            f"import sglang resolved to {sglang_path}, "
            f"outside frozen checkout {frozen_python}"
        )

    # Install the profiler-owned test-only BCG allowlist patch BEFORE
    # the server reads the allowlist (which happens during model_config
    # instantiation on server startup). Opt-in via CLI flag OR
    # QWEN35_PATCH_BCG_ALLOWLIST=1 env var. See
    # experiments/qwen35_4b/latent_bug_analysis.md § 3.
    patch_log_path = args.patch_log or Path(
        f"/tmp/qwen35_bcg_allowlist_patch_{os.getpid()}.json"
    )
    try:
        patch_spec = importlib.util.spec_from_file_location(
            "qwen35_bcg_allowlist_patch",
            str(Path(__file__).resolve().parent / "bcg_allowlist_patch.py"),
        )
        assert patch_spec is not None and patch_spec.loader is not None
        patch_mod = importlib.util.module_from_spec(patch_spec)
        sys.modules["qwen35_bcg_allowlist_patch"] = patch_mod
        patch_spec.loader.exec_module(patch_mod)  # type: ignore[union-attr]
        patch_result = patch_mod.install(force=bool(args.patch_bcg_allowlist))
    except Exception as exc:  # noqa: BLE001
        patch_result = {
            "target_archs": [
                "Qwen3VLForConditionalGeneration",
                "Qwen3VLMoeForConditionalGeneration",
            ],
            "enabled": False,
            "mutated": False,
            "error": f"bcg_allowlist_patch failed to load: {exc!r}",
        }
    try:
        patch_log_path.parent.mkdir(parents=True, exist_ok=True)
        patch_log_path.write_text(
            json.dumps(patch_result, indent=2, sort_keys=True, default=str) + "\n"
        )
        print(
            f"server_launcher: bcg_allowlist_patch mutated="
            f"{patch_result.get('mutated')} enabled={patch_result.get('enabled')} "
            f"log={patch_log_path}"
        )
    except Exception as exc:  # noqa: BLE001
        print(
            f"server_launcher: WARN: failed to write patch log to {patch_log_path}: {exc!r}",
            file=sys.stderr,
        )

    # Install instrumentation BEFORE the server starts loading modules.
    inst.install()

    # Delegate to sglang.launch_server (which has no main(); reproduce
    # its __main__ block here so instrumentation is already installed).
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree
    from sglang.launch_server import run_server

    load_plugins()
    server_args = prepare_server_args(forwarded)
    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
