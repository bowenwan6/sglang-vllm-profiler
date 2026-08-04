#!/usr/bin/env python3
"""Client skeleton for the Qwen3.5-4B BCG DeepStack validation.

CPU-only, dry-run-safe. This skeleton loads and validates the
launch-context JSON, the fixture manifest, and the request schedule,
but does NOT open a TCP connection, NOT import torch, NOT talk to the
running SGLang server.

The live client that goes here later must:

- Read the launch-context JSON to know its configuration label.
- Load the byte-pinned image fixture and text prompts.
- Send deterministic greedy requests via ``sglang.launch_server``'s
  OpenAI-compatible endpoint (or the ``/generate`` endpoint) with
  ``temperature=0``, ``top_p=1``, and a fixed sampler seed.
- Record per-request: request id, prompt hash, image fixture hash,
  server response, response tokens, response logprobs (if requested),
  server-log ``cuda graph: <bool>`` line for that request, and the
  ``input_deepstack_embeds.abs().sum()`` echo from the branch-local
  instrumentation.

Usage
-----

    python3 client_skeleton.py --dry-run --launch-ctx <path>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE_NAME = "image_bands.png"


def load_launch_ctx(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"launch-ctx not found: {path}")
    return json.loads(path.read_text())


def load_manifest(fixtures_dir: Path) -> dict:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    return json.loads(manifest_path.read_text())


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_request_schedule(launch_ctx: dict) -> list[dict]:
    """Deterministic request schedule for the CPU dry run.

    30 P_TXT warmup then 30 P_IMG measured, plus 5 P_LONG multi-shape
    stress requests. The live client will pace these serially with the
    same fixed inter-request delay.
    """
    schedule: list[dict] = []
    for i in range(30):
        schedule.append({"step": i, "kind": "P_TXT", "warmup": True})
    for i in range(30):
        schedule.append({"step": 30 + i, "kind": "P_IMG", "warmup": False})
    for i in range(5):
        schedule.append({"step": 60 + i, "kind": "P_LONG", "warmup": False})
    return schedule


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-ctx", required=True, type=Path)
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "fixtures",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    ctx = load_launch_ctx(args.launch_ctx)
    manifest = load_manifest(args.fixtures_dir)

    fixture_entry = manifest.get(FIXTURE_NAME) or {}
    pinned_sha = fixture_entry.get("sha256")
    observed_sha = sha256_of(args.fixtures_dir / FIXTURE_NAME)
    if pinned_sha != observed_sha:
        print(f"FATAL: fixture SHA mismatch: pinned={pinned_sha}, observed={observed_sha}", file=sys.stderr)
        return 1

    schedule = build_request_schedule(ctx)

    summary = {
        "config": ctx.get("config"),
        "attempt_id": ctx.get("attempt_id"),
        "gpu_id": ctx.get("gpu_id"),
        "fixture_sha256": observed_sha,
        "request_count": len(schedule),
        "kinds": sorted({r["kind"] for r in schedule}),
        "dry_run": bool(args.dry_run),
    }

    if args.dry_run:
        summary["network"] = "SKIPPED_DRY_RUN"
        summary["server_ping"] = "SKIPPED_DRY_RUN"
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    print("FATAL: live client wiring is not yet enabled in this skeleton.", file=sys.stderr)
    print("       Live GPU work requires a follow-up change.", file=sys.stderr)
    return 66


if __name__ == "__main__":
    sys.exit(main())
