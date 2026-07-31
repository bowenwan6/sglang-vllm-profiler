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

PINNED_UPSTREAM_SGLANG_HEAD = "5f9b0db18c787cf56ed9bbaf255f083f26c6ebc2"
PINNED_MODEL_ID = "Qwen/Qwen3.5-4B"
PINNED_MODEL_SHA = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PINNED_MODEL_ARCH = ["Qwen3_5ForConditionalGeneration"]
PINNED_MODEL_TYPE = "qwen3_5"
PINNED_FIXTURE_SHA = (
    "8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2"
)

# Soft environment expectations (from provenance.md §3; warnings only).
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixtures-dir",
        default=str(Path(__file__).resolve().parent.parent / "fixtures"),
    )
    parser.add_argument("--dry-run", action="store_true", help="Skip network checks.")
    parser.add_argument("--strict", action="store_true", help="Non-zero on any PIN_DRIFT.")
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

    hard_drifts = [
        r
        for r in results
        if r["status"] == "PIN_DRIFT"
        and r["check"] in {"upstream_sglang_main_head", "hf_model", "image_fixture"}
    ]
    if hard_drifts and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
