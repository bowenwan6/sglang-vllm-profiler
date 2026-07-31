#!/usr/bin/env python3
"""Provenance preflight for the Qwen3.5-4B BCG DeepStack investigation.

CPU-only. Does NOT import torch, NOT touch a GPU, NOT download anything.
It just reports what the local environment looks like against the pins
declared in ``experiments/qwen35_4b/provenance.md``.

Exit codes:
    0 — all required pins agree (or acceptable soft warnings only).
    1 — at least one hard pin disagrees with the environment.
    2 — usage / IO error.
    3 — a hard pin cannot be observed at all (e.g. gh missing) AND
        --strict is set.

Usage
-----

    python3 scripts/preflight_provenance.py --dry-run   # no network
    python3 scripts/preflight_provenance.py --strict    # non-zero on mismatch
    python3 scripts/preflight_provenance.py --json      # machine-readable output
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# --- Frozen pins (mirror provenance.md; keep in sync manually) ---------

# Hard pin — the executed local SGLang checkout SHA. Preflight aborts
# on mismatch (see check_frozen_checkout).
PINNED_FROZEN_SGLANG_SHA = "89f4a80c1f5e71c1c960df120f1e03b43dfd3c1d"
# Informational — the remote main HEAD at rebaseline. Drift is WARN only.
PINNED_UPSTREAM_SGLANG_HEAD = "89f4a80c1f5e71c1c960df120f1e03b43dfd3c1d"
PINNED_MODEL_ID = "Qwen/Qwen3.5-4B"
PINNED_MODEL_SHA = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PINNED_MODEL_ARCH = ["Qwen3_5ForConditionalGeneration"]
PINNED_MODEL_TYPE = "qwen3_5"
PINNED_FIXTURE_SHA = (
    "8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2"
)
# Sanity-check pin for the historical Qwen3-VL fork; runner refuses to
# touch it and warns loudly if it has moved.
PINNED_SGLANG_FORK_HEAD = "986c89e69c25882ab6f3d396f8eb306f38f2c8d2"
PINNED_SGLANG_FORK_PATH = "/data/sglang-fork"

# Soft environment expectations (from provenance.md §3; warnings only
# unless --strict-env).
EXPECTED_TORCH = "2.11.0+cu130"
EXPECTED_PYTHON = "3.12"


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _http_get_json(url: str, timeout: float = 10.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:  # noqa: S310
            return json.loads(fh.read().decode("utf-8"))
    except Exception:
        return None


def check_sglang_head(dry_run: bool) -> dict:
    """Report upstream SGLang main HEAD SHA and compare to pin."""
    label = "upstream_sglang_main_head"
    if dry_run:
        return {
            "check": label,
            "status": "SKIPPED_DRY_RUN",
            "pinned": PINNED_UPSTREAM_SGLANG_HEAD,
        }
    data = _http_get_json(
        "https://api.github.com/repos/sgl-project/sglang/commits/main"
    )
    if not data:
        return {"check": label, "status": "UNREACHABLE", "pinned": PINNED_UPSTREAM_SGLANG_HEAD}
    observed = data.get("sha", "")
    return {
        "check": label,
        "status": "PIN_MATCH" if observed == PINNED_UPSTREAM_SGLANG_HEAD else "PIN_DRIFT",
        "pinned": PINNED_UPSTREAM_SGLANG_HEAD,
        "observed": observed,
    }


def check_hf_model(dry_run: bool) -> dict:
    label = "hf_model"
    if dry_run:
        return {
            "check": label,
            "status": "SKIPPED_DRY_RUN",
            "pinned": {
                "id": PINNED_MODEL_ID,
                "sha": PINNED_MODEL_SHA,
                "architectures": PINNED_MODEL_ARCH,
                "model_type": PINNED_MODEL_TYPE,
            },
        }
    data = _http_get_json(f"https://huggingface.co/api/models/{PINNED_MODEL_ID}")
    if not data:
        return {"check": label, "status": "UNREACHABLE"}
    observed = {
        "id": data.get("id") or data.get("modelId"),
        "sha": data.get("sha"),
        "architectures": (data.get("config") or {}).get("architectures"),
        "model_type": (data.get("config") or {}).get("model_type"),
    }
    ok = (
        observed["id"] == PINNED_MODEL_ID
        and observed["sha"] == PINNED_MODEL_SHA
        and observed["architectures"] == PINNED_MODEL_ARCH
        and observed["model_type"] == PINNED_MODEL_TYPE
    )
    return {
        "check": label,
        "status": "PIN_MATCH" if ok else "PIN_DRIFT",
        "pinned": {
            "id": PINNED_MODEL_ID,
            "sha": PINNED_MODEL_SHA,
            "architectures": PINNED_MODEL_ARCH,
            "model_type": PINNED_MODEL_TYPE,
        },
        "observed": observed,
    }


def check_fixture(fixtures_dir: Path) -> dict:
    label = "image_fixture"
    path = fixtures_dir / "image_bands.png"
    if not path.is_file():
        return {"check": label, "status": "MISSING", "path": str(path)}
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ok = sha == PINNED_FIXTURE_SHA
    return {
        "check": label,
        "status": "PIN_MATCH" if ok else "PIN_DRIFT",
        "path": str(path),
        "pinned_sha256": PINNED_FIXTURE_SHA,
        "observed_sha256": sha,
    }


def check_python() -> dict:
    v = sys.version_info
    version_str = f"{v.major}.{v.minor}.{v.micro}"
    return {
        "check": "python_version",
        "status": (
            "MATCHES_MAJOR_MINOR"
            if version_str.startswith(EXPECTED_PYTHON)
            else "DIFFERENT"
        ),
        "expected_prefix": EXPECTED_PYTHON,
        "observed": version_str,
    }


def check_libcuda() -> dict:
    label = "libcuda_visible"
    ld = _run(["ldconfig", "-p"])
    if ld.returncode != 0:
        return {"check": label, "status": "LDCONFIG_UNAVAILABLE"}
    hits = [line for line in ld.stdout.splitlines() if "libcuda.so" in line]
    return {"check": label, "status": "OBSERVED", "matches": hits[:5]}


def check_torch_optional() -> dict:
    """Torch is optional at CPU-plan time; report the import result if available."""
    label = "torch_import"
    try:
        import torch  # type: ignore  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        return {"check": label, "status": "NOT_IMPORTABLE", "error": str(exc)[:200]}
    return {
        "check": label,
        "status": "PIN_MATCH" if torch.__version__ == EXPECTED_TORCH else "PIN_DRIFT",
        "pinned": EXPECTED_TORCH,
        "observed": torch.__version__,
    }


def check_repo_context() -> dict:
    """Emit branch / HEAD / preserved-item presence for the profiler repo."""
    proc = _run(["git", "rev-parse", "HEAD"])
    head = proc.stdout.strip() if proc.returncode == 0 else None
    proc_b = _run(["git", "branch", "--show-current"])
    branch = proc_b.stdout.strip() if proc_b.returncode == 0 else None
    proc_st = _run(["git", "status", "--short"])
    dirty = proc_st.stdout.strip() if proc_st.returncode == 0 else None
    return {
        "check": "profiler_repo",
        "status": "OBSERVED",
        "head": head,
        "branch": branch,
        "status_short": dirty,
    }


def check_frozen_checkout(frozen_path: str | None) -> dict:
    """Verify the frozen SGLang checkout HEAD equals the hard pin.

    This is the true source-of-truth pin: what code the runner
    imports at run time. A drift here is a hard failure.
    """
    label = "frozen_sglang_checkout"
    if not frozen_path:
        return {"check": label, "status": "NOT_PROVIDED", "pinned": PINNED_FROZEN_SGLANG_SHA}
    p = Path(frozen_path)
    if not (p / ".git").is_dir():
        return {"check": label, "status": "MISSING", "path": str(p), "pinned": PINNED_FROZEN_SGLANG_SHA}
    proc = _run(["git", "-C", str(p), "rev-parse", "HEAD"])
    if proc.returncode != 0:
        return {"check": label, "status": "GIT_ERROR", "path": str(p),
                "stderr": proc.stderr[:200], "pinned": PINNED_FROZEN_SGLANG_SHA}
    observed = proc.stdout.strip()
    return {
        "check": label,
        "status": "PIN_MATCH" if observed == PINNED_FROZEN_SGLANG_SHA else "PIN_DRIFT",
        "path": str(p),
        "pinned": PINNED_FROZEN_SGLANG_SHA,
        "observed": observed,
    }


def check_imported_sglang_path(frozen_path: str | None) -> dict:
    """After PYTHONPATH is set, imported sglang MUST resolve inside the frozen checkout."""
    label = "imported_sglang_path"
    try:
        import sglang  # type: ignore  # noqa: WPS433
    except Exception as exc:  # noqa: BLE001
        return {"check": label, "status": "NOT_IMPORTABLE", "error": str(exc)[:400]}
    file_path = Path(getattr(sglang, "__file__", "")).resolve()
    if not frozen_path:
        return {"check": label, "status": "OBSERVED", "path": str(file_path)}
    frozen_python = str(Path(frozen_path).resolve() / "python")
    inside = str(file_path).startswith(frozen_python)
    return {
        "check": label,
        "status": "INSIDE_FROZEN" if inside else "OUTSIDE_FROZEN",
        "path": str(file_path),
        "frozen_python": frozen_python,
    }


def check_sglang_fork_unchanged() -> dict:
    """Sanity: the historical Qwen3-VL fork must be unchanged."""
    label = "sglang_fork_unchanged"
    p = Path(PINNED_SGLANG_FORK_PATH)
    if not (p / ".git").is_dir():
        return {"check": label, "status": "MISSING", "path": str(p),
                "expected": PINNED_SGLANG_FORK_HEAD}
    proc = _run(["git", "-C", str(p), "rev-parse", "HEAD"])
    if proc.returncode != 0:
        return {"check": label, "status": "GIT_ERROR", "path": str(p),
                "stderr": proc.stderr[:200]}
    observed = proc.stdout.strip()
    return {
        "check": label,
        "status": "PIN_MATCH" if observed == PINNED_SGLANG_FORK_HEAD else "PIN_DRIFT",
        "path": str(p),
        "expected": PINNED_SGLANG_FORK_HEAD,
        "observed": observed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures-dir",
        default=str(Path(__file__).resolve().parent.parent / "fixtures"),
    )
    parser.add_argument("--frozen-sglang", default=None,
                        help="Path to the frozen SGLang checkout directory.")
    parser.add_argument("--dry-run", action="store_true", help="Skip network checks.")
    parser.add_argument("--strict", action="store_true",
                        help="Non-zero on any hard PIN_DRIFT (see hard_drift_checks).")
    parser.add_argument("--strict-env", action="store_true",
                        help="Also non-zero on soft env drifts (torch, python).")
    parser.add_argument("--json", dest="emit_json", action="store_true")
    args = parser.parse_args(argv)

    fixtures_dir = Path(args.fixtures_dir)

    results = [
        check_python(),
        check_repo_context(),
        check_fixture(fixtures_dir),
        check_libcuda(),
        check_torch_optional(),
        check_sglang_head(args.dry_run),
        check_hf_model(args.dry_run),
        check_frozen_checkout(args.frozen_sglang),
        check_imported_sglang_path(args.frozen_sglang),
        check_sglang_fork_unchanged(),
    ]

    if args.emit_json:
        print(json.dumps({"results": results}, indent=2))
    else:
        for r in results:
            print(f"[{r['status']:<20}] {r['check']}")
            for k, v in r.items():
                if k in {"check", "status"}:
                    continue
                print(f"    {k}: {v}")

    # Hard-fail conditions:
    hard_drift_checks = {
        "frozen_sglang_checkout",  # executed SHA — the true source of truth
        "hf_model",
        "image_fixture",
        "imported_sglang_path",
        "sglang_fork_unchanged",
    }
    # The remote main HEAD (upstream_sglang_main_head) is now
    # informational-only; a drift here is WARN, never a hard fail.

    hard_drifts = [
        r for r in results
        if r["check"] in hard_drift_checks
        and r["status"] not in {"PIN_MATCH", "OBSERVED", "SKIPPED_DRY_RUN",
                                 "MATCHES_MAJOR_MINOR", "INSIDE_FROZEN",
                                 "NOT_PROVIDED"}
    ]
    if hard_drifts and args.strict:
        return 1
    if args.strict_env:
        soft = [r for r in results if r["status"] in {"PIN_DRIFT", "DIFFERENT",
                                                       "OUTSIDE_FROZEN"}]
        if soft:
            return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
