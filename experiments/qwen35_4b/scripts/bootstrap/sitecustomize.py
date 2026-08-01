"""Branch-owned sitecustomize for the Qwen3.5-4B BCG DeepStack investigation.

Purpose: propagate ``instrumentation.install()`` into every Python
interpreter that inherits the runner's environment, including SGLang's
scheduler / model-worker subprocesses that are started via
``multiprocessing.set_start_method('spawn', force=True)`` (SGLang's
``engine.py:1621``). Spawn children re-import all modules fresh, so
the monkey-patches installed by ``server_launcher.py`` in the launcher
parent do not carry over on their own; without this shim,
request-level events like ``bcg_execute_body_enter`` and
``lm_forward_input_deepstack`` never fire, and the
``eager_zero_deepstack`` ablation degenerates to ``eager_normal``.

Design:

- Reads the absolute path to ``instrumentation.py`` from the env var
  ``QWEN35_INSTRUMENTATION_PATH``. When that env var is unset the shim
  is a no-op (safe to leave on PYTHONPATH in unrelated invocations).
- Loads the instrumentation module without going through Python's
  regular import system (``importlib.util.spec_from_file_location``),
  so the module identity is unambiguous regardless of ``sys.path``.
- Registers a light ``builtins.__import__`` wrapper. After every
  import, if all three instrumentation target modules
  (``sglang.srt.model_executor.runner.prefill_cuda_graph_runner``,
  ``sglang.srt.model_executor.model_runner``,
  ``sglang.srt.managers.mm_utils``) are present in ``sys.modules``,
  it calls ``instrumentation.install()`` exactly once. The install
  itself is idempotent, so double-calls (parent + child) are safe.
- All logic is wrapped in defensive ``try/except``. This shim MUST
  never fail interpreter startup.
- Preserves the Debian ``apport_python_hook`` that the system
  sitecustomize would have installed (since our shim shadows the
  system one when our bootstrap dir is first on ``PYTHONPATH``).

Not committed to the frozen SGLang checkout; lives on the profiler
branch only.
"""

# Preserve the system sitecustomize's only side effect (Debian apport
# hook). Silent if apport is not installed.
try:  # noqa: SIM105
    import apport_python_hook  # type: ignore  # noqa: WPS433
    apport_python_hook.install()
except Exception:  # noqa: BLE001
    pass

import os as _os  # noqa: E402

_QWEN35_PATH = _os.environ.get("QWEN35_INSTRUMENTATION_PATH", "")

if _QWEN35_PATH and _os.path.isfile(_QWEN35_PATH):
    try:  # noqa: WPS229
        import builtins as _b
        import importlib.util as _iu
        import sys as _sys

        _spec = _iu.spec_from_file_location(
            "qwen35_instrumentation", _QWEN35_PATH
        )
        _mod = _iu.module_from_spec(_spec)
        _sys.modules["qwen35_instrumentation"] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]

        # (target module name, attribute that indicates the module body
        # has finished executing — presence in sys.modules alone is not
        # enough because Python registers a partially initialised module
        # in sys.modules before executing its body).
        _TARGETS = (
            (
                "sglang.srt.model_executor.runner.prefill_cuda_graph_runner",
                "PrefillCudaGraphRunner",
            ),
            ("sglang.srt.model_executor.model_runner", "ModelRunner"),
            ("sglang.srt.managers.mm_utils", "general_mm_embed_routine"),
        )
        _installed = [False]
        _orig_import = _b.__import__

        def _hooked_import(  # noqa: WPS430
            name, globals=None, locals=None, fromlist=(), level=0,
        ):
            _r = _orig_import(name, globals, locals, fromlist, level)
            if not _installed[0]:
                ready = True
                for mod_name, attr in _TARGETS:
                    m = _sys.modules.get(mod_name)
                    if m is None or not hasattr(m, attr):
                        ready = False
                        break
                if ready:
                    # Set FIRST so re-entrant imports fired from inside
                    # install() do not recurse.
                    _installed[0] = True
                    try:
                        _mod.install()
                    except Exception:  # noqa: BLE001
                        # Leave _installed as True — a broken install
                        # is not going to be fixed by retrying.
                        pass
            return _r

        _b.__import__ = _hooked_import
    except Exception:  # noqa: BLE001
        # Never break the interpreter. The parent-process patches will
        # still work; only spawn-child propagation is degraded.
        pass
