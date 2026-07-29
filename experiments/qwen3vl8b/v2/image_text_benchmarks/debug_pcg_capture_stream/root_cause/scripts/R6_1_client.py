#!/usr/bin/env python3
"""R6.1 correctness-gate client. Sends fixed prompts (image + text-only)
to a running OpenAI-chat compatible server and records outputs to JSON.

CPU-only client. Does not launch or manage servers; the orchestrator
shell script handles server lifecycle and passes the base URL.

Usage:
    R6_1_client.py \\
        --base-url http://127.0.0.1:30003 \\
        --model-path /root/.cache/huggingface/hub/.../snapshots/0c351dd... \\
        --fixture /path/to/R6.1_fixture.png \\
        --prompts /path/to/prompts.json \\
        --mode image|text|interleaved \\
        --out /path/to/output.json

Modes:
    image        — send image_prompts (each with the fixture image)
    text         — send text_only_prompts (no image)
    interleaved  — text_only[0] → image[0] → text_only[1] → image[1] → text_only[2],
                   all on the same server, sequentially

Output JSON schema (deterministic key order):
    {
      "meta": {
        "base_url": ..., "model_path": ..., "fixture_sha256": ...,
        "mode": ..., "sampling": {...}, "wall_clock_utc_start": ...,
        "wall_clock_utc_end": ...
      },
      "requests": [
        {
          "idx": 0, "kind": "image"|"text",
          "prompt": ..., "response_text": ..., "finish_reason": ...,
          "usage": {...}, "latency_s": ..., "http_status": ...,
          "error": null|str
        },
        ...
      ]
    }

Determinism: single-request-at-a-time (no concurrency), fixed sampling
seed, greedy decoding.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def load_fixture_b64(fixture_path: Path) -> tuple[str, str]:
    raw = fixture_path.read_bytes()
    sha256 = hashlib.sha256(raw).hexdigest()
    b64 = base64.b64encode(raw).decode("ascii")
    return sha256, b64


def build_body(model_path: str, prompt: str, image_b64: str | None,
               sampling: dict[str, Any]) -> dict[str, Any]:
    if image_b64 is not None:
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        ]
    else:
        content = prompt
    return {
        "model": model_path,
        "messages": [{"role": "user", "content": content}],
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "max_tokens": sampling["max_tokens"],
        "seed": sampling["seed"],
        "stream": False,
    }


def do_one(session: requests.Session, base_url: str, body: dict[str, Any],
           idx: int, kind: str, prompt: str) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    t0 = time.perf_counter()
    try:
        r = session.post(url, json=body, timeout=300)
        latency = time.perf_counter() - t0
        status = r.status_code
        try:
            payload = r.json()
        except Exception as e:
            return {"idx": idx, "kind": kind, "prompt": prompt,
                    "response_text": None, "finish_reason": None,
                    "usage": None, "latency_s": latency, "http_status": status,
                    "error": f"json_decode_failed: {e!r}; body_head={r.text[:400]!r}"}
        if status != 200:
            return {"idx": idx, "kind": kind, "prompt": prompt,
                    "response_text": None, "finish_reason": None,
                    "usage": None, "latency_s": latency, "http_status": status,
                    "error": f"non_200: {json.dumps(payload)[:400]}"}
        choice = payload.get("choices", [{}])[0]
        return {"idx": idx, "kind": kind, "prompt": prompt,
                "response_text": choice.get("message", {}).get("content"),
                "finish_reason": choice.get("finish_reason"),
                "usage": payload.get("usage"),
                "latency_s": latency, "http_status": status, "error": None}
    except requests.RequestException as e:
        return {"idx": idx, "kind": kind, "prompt": prompt,
                "response_text": None, "finish_reason": None,
                "usage": None, "latency_s": time.perf_counter() - t0,
                "http_status": None, "error": f"request_exception: {e!r}"}


def run(base_url: str, model_path: str, fixture_path: Path,
        prompts_path: Path, mode: str, out_path: Path) -> int:
    prompts_cfg = json.loads(prompts_path.read_text())
    sampling = prompts_cfg["sampling"]
    image_prompts = prompts_cfg["image_prompts"]
    text_prompts = prompts_cfg["text_only_prompts"]
    fixture_sha, fixture_b64 = load_fixture_b64(fixture_path)

    plan: list[tuple[int, str, str, str | None]] = []
    if mode == "image":
        for i, p in enumerate(image_prompts):
            plan.append((i, "image", p, fixture_b64))
    elif mode == "text":
        for i, p in enumerate(text_prompts):
            plan.append((i, "text", p, None))
    elif mode == "interleaved":
        n = max(len(text_prompts), len(image_prompts))
        idx = 0
        for i in range(n):
            if i < len(text_prompts):
                plan.append((idx, "text", text_prompts[i], None))
                idx += 1
            if i < len(image_prompts):
                plan.append((idx, "image", image_prompts[i], fixture_b64))
                idx += 1
    else:
        print(f"unknown mode: {mode!r}", file=sys.stderr)
        return 2

    meta = {"base_url": base_url, "model_path": model_path,
            "fixture_path": str(fixture_path.resolve()),
            "fixture_sha256": fixture_sha,
            "prompts_path": str(prompts_path.resolve()), "mode": mode,
            "sampling": sampling, "wall_clock_utc_start": utc_now(),
            "wall_clock_utc_end": None}
    results: list[dict[str, Any]] = []
    session = requests.Session()
    any_error = False
    for idx, kind, prompt, img_b64 in plan:
        body = build_body(model_path, prompt, img_b64, sampling)
        rec = do_one(session, base_url, body, idx, kind, prompt)
        results.append(rec)
        if rec["error"] is not None:
            any_error = True
        print(f"[client] idx={rec['idx']} kind={rec['kind']} "
              f"http={rec['http_status']} latency={rec['latency_s']:.3f}s "
              f"err={rec['error'] is not None}", flush=True)

    meta["wall_clock_utc_end"] = utc_now()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"meta": meta, "requests": results},
                                    indent=2, sort_keys=True))
    return 0 if not any_error else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--prompts", required=True, type=Path)
    ap.add_argument("--mode", required=True,
                    choices=["image", "text", "interleaved"])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    return run(args.base_url, args.model_path, args.fixture, args.prompts,
               args.mode, args.out)


if __name__ == "__main__":
    sys.exit(main())
