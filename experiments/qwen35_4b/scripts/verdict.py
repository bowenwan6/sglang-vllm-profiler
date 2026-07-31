#!/usr/bin/env python3
"""Verdict computation skeleton for the Qwen3.5-4B BCG DeepStack validation.

CPU-only. Given an attempt directory containing ``metadata.json``,
``raw/`` (bench + server logs), and — once live — per-request tensor
dumps, this script computes the predeclared verdict from
``validation_plan.md`` §1 and writes ``verdict.json`` + ``verdict.md``.

Right now this is a **skeleton**: it validates directory layout,
loads / echoes the launch context, and computes the verdict *only from
sentinel inputs* if provided (``--force-verdict`` for the dry-run test
harness). It does NOT infer a verdict from real data yet — that logic
is deferred to the follow-up commit that lands real evidence loaders.

Historical guard: this script MUST refuse to score across mismatched
launches. In Qwen3-VL R6.5, an earlier verdict script combined stale
ratio artifacts predating the actual launch. Here, every raw evidence
file the scorer consumes must carry a ``launch_id`` (recorded at
launch by ``runner_skeleton.sh`` — the ``prelaunch_utc`` field of the
launch-context JSON), and ``verdict.py`` rejects any file whose
``launch_id`` does not equal the launch-context's ``prelaunch_utc``.

Usage
-----

    python3 verdict.py --attempt-dir <path>
    python3 verdict.py --attempt-dir <path> --force-verdict INFRA_FAILURE --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

VALID_VERDICTS = {
    "PASS_H_B",
    "PASS_H_D",
    "FAIL_H_A",
    "FAIL_H_C",
    "AMBIGUOUS",
    "INFRA_FAILURE",
}


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def load_metadata(attempt_dir: Path) -> dict:
    return load_json(attempt_dir / "metadata.json")


def enforce_launch_context(attempt_dir: Path, launch_id: str) -> list[str]:
    """Reject raw files whose recorded launch_id disagrees with ours.

    Returns the list of accepted raw filenames. Empty list is fine at
    skeleton stage (before any raw data exists).
    """
    raw_dir = attempt_dir / "raw"
    if not raw_dir.is_dir():
        return []
    accepted: list[str] = []
    for path in sorted(raw_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"REJECT: {path.name} unreadable: {exc}", file=sys.stderr)
            continue
        recorded = data.get("launch_id") or data.get("prelaunch_utc")
        if recorded != launch_id:
            print(f"REJECT: {path.name} launch_id={recorded!r} != {launch_id!r}", file=sys.stderr)
            continue
        accepted.append(path.name)
    return accepted


def write_verdict(attempt_dir: Path, verdict: str, evidence: dict) -> None:
    if verdict not in VALID_VERDICTS:
        raise SystemExit(f"invalid verdict: {verdict}")
    payload = {"verdict": verdict, "evidence": evidence}
    (attempt_dir / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    md = [
        f"# Verdict — {verdict}",
        "",
        "This machine verdict was produced by `scripts/verdict.py`.",
        "See `validation_plan.md` §1 for the verdict shape.",
        "",
        "## Evidence",
        "",
        "```json",
        json.dumps(evidence, indent=2, sort_keys=True),
        "```",
        "",
    ]
    (attempt_dir / "verdict.md").write_text("\n".join(md))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument(
        "--force-verdict",
        default=None,
        choices=sorted(VALID_VERDICTS),
        help="Skeleton stage only: bypass evidence inference.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    attempt_dir = args.attempt_dir
    if not attempt_dir.is_dir():
        print(f"FATAL: attempt-dir does not exist: {attempt_dir}", file=sys.stderr)
        return 1

    if args.dry_run and not (attempt_dir / "metadata.json").is_file():
        # In dry-run, allow a synthetic metadata.
        metadata = {
            "launch_context": {"prelaunch_utc": "1970-01-01T00:00:00Z"},
            "dry_run": True,
        }
    else:
        metadata = load_metadata(attempt_dir)

    launch_ctx = metadata.get("launch_context") or {}
    launch_id = launch_ctx.get("prelaunch_utc") or metadata.get("prelaunch_utc")
    if not launch_id:
        print("FATAL: metadata.json missing launch_context.prelaunch_utc", file=sys.stderr)
        return 1

    accepted = enforce_launch_context(attempt_dir, launch_id)

    evidence = {
        "launch_id": launch_id,
        "accepted_raw_files": accepted,
        "dry_run": bool(args.dry_run),
        "notes": (
            "skeleton stage — real evidence inference is a follow-up. "
            "Score from raw files only when they carry launch_id == prelaunch_utc."
        ),
    }

    if args.force_verdict:
        write_verdict(attempt_dir, args.force_verdict, evidence)
        print(f"wrote {attempt_dir / 'verdict.json'} (forced={args.force_verdict})")
        return 0

    if args.dry_run:
        # Skeleton dry run: refuse to guess a verdict from empty raw.
        print("dry-run: no --force-verdict; leaving verdict.* untouched.")
        return 0

    print(
        "FATAL: verdict inference from real evidence is not yet enabled.",
        file=sys.stderr,
    )
    return 66


if __name__ == "__main__":
    sys.exit(main())
