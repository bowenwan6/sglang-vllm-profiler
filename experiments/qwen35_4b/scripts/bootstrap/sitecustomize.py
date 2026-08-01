"""Branch-owned sitecustomize for the Qwen3.5-4B BCG DeepStack investigation.

Purpose: propagate ``instrumentation.install()`` — and the profiler-
owned test-only BCG allowlist monkey-patch — into every Python
interpreter that inherits the runner's environment, including SGLang's
scheduler / model-worker subprocesses that are started via
``multiprocessing.set_start_method('spawn', force=True)`` (SGLang's
``engine.py:1621``). Spawn children re-import all modules fresh, so
the monkey-patches installed by ``server_launcher.py`` in the launcher
parent do not carry over on their own; without this shim,
request-level events like ``bcg_execute_body_enter`` and
``lm_forward_input_deepstack`` never fire, the
``eager_zero_deepstack`` ablation degenerates to ``eager_normal``, and
the child's re-imported ``sglang.srt.configs.model_config`` allowlist
is the shipped one (which excludes Qwen3-VL).

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
_QWEN35_BCG_PATCH_PATH = _os.environ.get("QWEN35_BCG_ALLOWLIST_PATCH_PATH", "")
_QWEN35_BCG_PATCH_ENABLED = (
    _os.environ.get("QWEN35_PATCH_BCG_ALLOWLIST", "0") == "1"
)

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

        # Load the BCG allowlist patch module too (opt-in via env var).
        # Loaded even when disabled so the module object is available
        # for the diagnostic snapshot; ``_bcg_patch_installed`` gates
        # the actual ``install(force=True)`` call.
        _bcg_mod = None
        if _QWEN35_BCG_PATCH_PATH and _os.path.isfile(_QWEN35_BCG_PATCH_PATH):
            try:
                _bcg_spec = _iu.spec_from_file_location(
                    "qwen35_bcg_allowlist_patch", _QWEN35_BCG_PATCH_PATH
                )
                _bcg_mod = _iu.module_from_spec(_bcg_spec)
                _sys.modules["qwen35_bcg_allowlist_patch"] = _bcg_mod
                _bcg_spec.loader.exec_module(_bcg_mod)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                _bcg_mod = None

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
        _bcg_patch_installed = [False]
        _orig_import = _b.__import__

        def _try_install_bcg_patch():  # noqa: WPS430
            """Apply the BCG allowlist mutation as soon as
            ``sglang.srt.configs.model_config`` is fully imported.

            Fires early — well before ``ModelConfig`` is constructed —
            so the child's ``is_multimodal_breakable_cuda_graph_supported``
            check sees the mutated list.
            """
            if _bcg_patch_installed[0]:
                return
            if not _QWEN35_BCG_PATCH_ENABLED or _bcg_mod is None:
                return
            mc = _sys.modules.get("sglang.srt.configs.model_config")
            if mc is None or not hasattr(
                mc, "multimodal_breakable_cuda_graph_supported_model_archs"
            ):
                return
            _bcg_patch_installed[0] = True
            try:
                _bcg_mod.install(force=True)
            except Exception:  # noqa: BLE001
                # Leave the flag True; retrying will not help.
                pass

        def _hooked_import(  # noqa: WPS430
            name, globals=None, locals=None, fromlist=(), level=0,
        ):
            _r = _orig_import(name, globals, locals, fromlist, level)
            # BCG allowlist patch fires the moment model_config lands.
            _try_install_bcg_patch()
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
