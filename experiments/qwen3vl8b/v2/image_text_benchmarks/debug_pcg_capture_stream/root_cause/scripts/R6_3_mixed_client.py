#!/usr/bin/env python3
"""R6.3c mixed-safety client: interleaved text + image requests on one
fork-PCG server. Writes per-request JSONL + summary JSON. CPU-only
client; server runs on the GPU launched by the runner.

Layout:
  * 50 image requests + 50 text requests, deterministic interleaved
    order [text, image, text, image, ...].
  * Image = R6.1 fixture (deterministic PNG).
  * Text = generic prompts sampled deterministically from a small
    pool (seed 42).
  * temperature=0, top_p=1, max_tokens=64, seed=42.
"""
from __future__ import annotations
import argparse, base64, hashlib, json, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

TEXT_POOL = [
    "In one word, what does a green traffic light mean?",
    "What is 5 plus 7?",
    "List two colors of the rainbow.",
    "What is the capital of France?",
    "What is 2 times 3?",
    "Name one primary color.",
    "What sound does a dog make?",
    "What number comes after 9?",
    "What is the past tense of 'run'?",
    "Complete: the sky is",
]

IMAGE_PROMPTS = [
    "Describe this image in one word.",
    "What is the dominant shape in the image?",
    "How many distinct colors are visible?",
]


def load_fixture(fp):
    raw = fp.read_bytes()
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii")


def build(model, prompt, img_b64):
    if img_b64:
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    else:
        content = prompt
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "top_p": 1, "max_tokens": 64, "seed": 42, "stream": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--n-text", type=int, default=50)
    ap.add_argument("--n-image", type=int, default=50)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    args = ap.parse_args()

    fixture_sha, fixture_b64 = load_fixture(args.fixture)

    # Interleaved sequence
    plan = []
    for i in range(max(args.n_text, args.n_image)):
        if i < args.n_text:
            plan.append(("text", TEXT_POOL[i % len(TEXT_POOL)], None))
        if i < args.n_image:
            plan.append(("image", IMAGE_PROMPTS[i % len(IMAGE_PROMPTS)], fixture_b64))

    session = requests.Session()
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    reqs = []
    req_failures = 0
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for idx, (kind, prompt, img_b64) in enumerate(plan):
        body = build(args.model, prompt, img_b64)
        t0 = time.perf_counter()
        try:
            r = session.post(url, json=body, timeout=300)
            latency = time.perf_counter() - t0
            status = r.status_code
            try: payload = r.json()
            except: payload = None
            content = None; finish = None; usage = None
            if payload:
                ch = payload.get("choices", [{}])[0]
                content = ch.get("message", {}).get("content")
                finish = ch.get("finish_reason")
                usage = payload.get("usage")
            rec = {"idx": idx, "kind": kind, "http_status": status,
                   "latency_s": latency, "response_text": content,
                   "finish_reason": finish, "usage": usage,
                   "error": None if status == 200 else f"non_200: {json.dumps(payload)[:200] if payload else '?'}"}
            if status != 200 or not content: req_failures += 1
        except Exception as e:
            rec = {"idx": idx, "kind": kind, "http_status": None,
                   "latency_s": time.perf_counter()-t0, "response_text": None,
                   "finish_reason": None, "usage": None,
                   "error": f"request_exception: {e!r}"}
            req_failures += 1
        reqs.append(rec)
        print(f"[mixed] idx={idx} kind={kind} status={rec['http_status']} lat={rec['latency_s']:.3f}s err={rec['error'] is not None}", flush=True)

    ended = datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for rec in reqs:
            f.write(json.dumps(rec) + "\n")

    n_text = sum(1 for r in reqs if r["kind"] == "text")
    n_image = sum(1 for r in reqs if r["kind"] == "image")
    args.out_summary.write_text(json.dumps({
        "started_utc": started, "ended_utc": ended,
        "fixture_sha256": fixture_sha,
        "n_text": n_text, "n_image": n_image, "n_total": len(reqs),
        "completed": sum(1 for r in reqs if r["http_status"] == 200 and r["error"] is None),
        "request_failures": req_failures,
    }, indent=2, sort_keys=True))
    print(f"[mixed] done: total={len(reqs)} failures={req_failures}")
    return 0 if req_failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
