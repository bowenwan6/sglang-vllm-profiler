#!/usr/bin/env python3
"""Live client for the Qwen3.5-4B BCG DeepStack validation.

Reads the launch-context JSON, loads the byte-pinned image fixture,
issues a small deterministic request set against the local SGLang
server, and records per-request evidence. Never dumps whole hidden-
state tensors — the runner-side instrumentation is authoritative for
tensor-level attribution.

Usage
-----

    python3 client.py --launch-ctx <path> --server-url http://127.0.0.1:30000 \\
        --results-dir <path> [--dry-run]
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

FIXTURE_NAME = "image_bands.png"

PROMPT_IMAGE = "Describe the colours in this image in one short sentence."
PROMPT_TEXT = "Say the words 'hello world' and nothing else."


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


def http_json_post(url: str, payload: dict, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def http_get(url: str, timeout: float) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
            return resp.getcode(), resp.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001
        return -1, repr(exc)


def build_request_schedule(config_label: str, image_b64: str) -> list[dict]:
    """Small predeclared request schedule for the correctness/path validation.

    Each request has a unique client-side id so the runner can match it
    against the instrumentation log emitted by the server.
    """
    now_us = int(time.time() * 1_000_000)
    reqs: list[dict] = []

    # 1 text warmup so the server has settled.
    reqs.append({
        "client_request_id": f"{config_label}-warmup-txt-1-{now_us}",
        "kind": "P_TXT",
        "role": "warmup",
        "text": PROMPT_TEXT,
        "image_b64": None,
    })
    # Text-only scored control.
    reqs.append({
        "client_request_id": f"{config_label}-scored-txt-1-{now_us + 1}",
        "kind": "P_TXT",
        "role": "scored",
        "text": PROMPT_TEXT,
        "image_b64": None,
    })
    # Primary image scored.
    reqs.append({
        "client_request_id": f"{config_label}-scored-img-1-{now_us + 2}",
        "kind": "P_IMG",
        "role": "scored",
        "text": PROMPT_IMAGE,
        "image_b64": image_b64,
    })
    # Image repeat for eager noise envelope / BCG confirmation.
    reqs.append({
        "client_request_id": f"{config_label}-scored-img-2-{now_us + 3}",
        "kind": "P_IMG",
        "role": "scored",
        "text": PROMPT_IMAGE,
        "image_b64": image_b64,
    })
    return reqs


def send_generate(
    server_url: str,
    req: dict,
    max_new_tokens: int,
    timeout: float,
    return_logprob: bool,
    return_hidden_states: bool,
) -> dict:
    """Send one deterministic greedy request via SGLang /generate."""
    payload = {
        "text": None,
        "input_ids": None,
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "max_new_tokens": max_new_tokens,
            "n": 1,
        },
        "stream": False,
        "return_logprob": return_logprob,
        "return_text_in_logprobs": False,
        "logprob_start_len": 0,
        "top_logprobs_num": 5 if return_logprob else 0,
        "rid": req["client_request_id"],
    }
    if return_hidden_states:
        payload["return_hidden_states"] = True

    if req["image_b64"] is not None:
        # Qwen3.5-VL uses `<image>` placeholder in the prompt.
        # Prompt template: "<image>Describe the colours..." — this
        # matches SGLang's default multimodal handling.
        payload["text"] = f"<image>{req['text']}"
        payload["image_data"] = [f"data:image/png;base64,{req['image_b64']}"]
    else:
        payload["text"] = req["text"]

    t0 = time.time()
    err = None
    resp = None
    try:
        resp = http_json_post(f"{server_url}/generate", payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        err = repr(exc)
    dt = time.time() - t0

    record = {
        "client_request_id": req["client_request_id"],
        "kind": req["kind"],
        "role": req["role"],
        "duration_s": dt,
        "error": err,
        "http_response": resp,
    }
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-ctx", required=True, type=Path)
    parser.add_argument("--server-url", default="http://127.0.0.1:30000")
    parser.add_argument("--results-dir", required=True, type=Path)
    parser.add_argument("--config-label", default="")
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "fixtures",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--return-logprob", action="store_true", default=True)
    parser.add_argument("--return-hidden-states", action="store_true", default=False)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    ctx = load_launch_ctx(args.launch_ctx)
    manifest = load_manifest(args.fixtures_dir)

    fixture_entry = manifest.get(FIXTURE_NAME) or {}
    pinned_sha = fixture_entry.get("sha256")
    fixture_path = args.fixtures_dir / FIXTURE_NAME
    observed_sha = sha256_of(fixture_path)
    if pinned_sha != observed_sha:
        print(
            f"FATAL: fixture SHA mismatch: pinned={pinned_sha}, "
            f"observed={observed_sha}",
            file=sys.stderr,
        )
        return 1

    image_b64 = base64.b64encode(fixture_path.read_bytes()).decode("ascii")
    label = args.config_label or ctx.get("config") or "unknown"
    schedule = build_request_schedule(label, image_b64)

    summary = {
        "config": label,
        "attempt_id": ctx.get("attempt_id"),
        "gpu_id": ctx.get("gpu_id"),
        "server_url": args.server_url,
        "fixture_sha256": observed_sha,
        "request_count": len(schedule),
        "kinds": sorted({r["kind"] for r in schedule}),
        "dry_run": bool(args.dry_run),
        "launch_id": ctx.get("prelaunch_utc"),
    }

    if args.dry_run:
        summary["network"] = "SKIPPED_DRY_RUN"
        summary["server_ping"] = "SKIPPED_DRY_RUN"
        summary["scheduled_ids"] = [r["client_request_id"] for r in schedule]
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    # Real run: ping /health first (informational only).
    code, body = http_get(f"{args.server_url}/health", timeout=5.0)
    summary["server_health_code"] = code
    summary["server_health_body_prefix"] = (body or "")[:120]

    args.results_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for req in schedule:
        rec = send_generate(
            args.server_url,
            req,
            max_new_tokens=args.max_new_tokens,
            timeout=args.request_timeout,
            return_logprob=args.return_logprob,
            return_hidden_states=args.return_hidden_states,
        )
        rec["launch_id"] = ctx.get("prelaunch_utc")
        rec["config"] = label
        records.append(rec)

    out_path = args.results_dir / f"client_records_{label}.json"
    out_path.write_text(json.dumps(
        {"summary": summary, "records": records}, indent=2, default=str, sort_keys=True,
    ) + "\n")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
