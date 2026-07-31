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
        -- \\
        --model-path Qwen/Qwen3.5-4B \\
        --port 30000 \\
        ...
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
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
