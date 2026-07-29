#!/usr/bin/env python3
"""R6.5 empirical mixed-workload client. Serves a deterministic
mixture of text and image requests to a running server.

Deterministic sequence: given a text_ratio p and total N requests,
generate an interleaved sequence with exactly round(p*N) text
requests and (N - round(p*N)) image requests. Order is deterministic
per (p, seed) via the same permutation algorithm on both stock and
fork runs.

For each request:
  text: 128-token target prompt from caseA_short.jsonl
  image: 720p fixture image, short prompt from a fixed pool

sampling: temperature=0 top_p=1 seed=42 max_tokens=128
"""
from __future__ import annotations
import argparse, base64, hashlib, json, random, sys, time
from datetime import datetime, timezone
from pathlib import Path
import requests

IMAGE_PROMPTS = ["Describe this image.", "What colors are present?",
                 "How many distinct regions do you see?"]


def load_fixture(fp):
    raw = fp.read_bytes()
    return hashlib.sha256(raw).hexdigest(), base64.b64encode(raw).decode("ascii")


def load_text_prompts(caseA_path, n):
    """Load first n prompts from caseA_short.jsonl."""
    ps = []
    with open(caseA_path) as f:
        for i, ln in enumerate(f):
            if i >= n: break
            try:
                obj = json.loads(ln)
                ps.append(obj.get("prompt", ""))
            except: pass
    return ps


def build(model, prompt, img_b64, max_tokens):
    if img_b64:
        content = [{"type": "text", "text": prompt},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}]
    else:
        content = prompt
    return {"model": model, "messages": [{"role": "user", "content": content}],
            "temperature": 0, "top_p": 1, "max_tokens": max_tokens, "seed": 42, "stream": False}


def make_plan(text_ratio, n, text_prompts, img_b64, seed):
    n_text = int(round(text_ratio * n))
    n_image = n - n_text
    # Build a deterministic interleaved order per (text_ratio, seed).
    rng = random.Random(seed)
    kinds = ["text"] * n_text + ["image"] * n_image
    # Fisher-Yates shuffle with seeded RNG for identical order across runs
    for i in range(len(kinds) - 1, 0, -1):
        j = rng.randint(0, i)
        kinds[i], kinds[j] = kinds[j], kinds[i]
    plan = []
    t_idx = i_idx = 0
    for k in kinds:
        if k == "text":
            plan.append(("text", text_prompts[t_idx % len(text_prompts)], None))
            t_idx += 1
        else:
            plan.append(("image", IMAGE_PROMPTS[i_idx % len(IMAGE_PROMPTS)], img_b64))
            i_idx += 1
    return plan, n_text, n_image


def stats(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    if not xs: return {"n": 0}
    xs_s = sorted(xs); n = len(xs)
    mean = sum(xs)/n
    return {"n": n, "mean": mean,
            "median": xs_s[n//2] if n % 2 else (xs_s[n//2-1]+xs_s[n//2])/2,
            "p90": xs_s[int(0.9*(n-1))], "p99": xs_s[int(0.99*(n-1))],
            "stdev": (sum((x-mean)**2 for x in xs)/(n-1))**0.5 if n>1 else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--fixture", required=True, type=Path)
    ap.add_argument("--caseA", required=True, type=Path)
    ap.add_argument("--text-ratio", type=float, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument("--out-summary", required=True, type=Path)
    args = ap.parse_args()

    fixture_sha, img_b64 = load_fixture(args.fixture)
    text_prompts = load_text_prompts(args.caseA, max(args.n, 100))
    plan, n_text, n_image = make_plan(args.text_ratio, args.n, text_prompts,
                                       img_b64, args.seed)

    session = requests.Session()
    url = f"{args.base_url.rstrip('/')}/v1/chat/completions"
    reqs = []
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for idx, (kind, prompt, ib64) in enumerate(plan):
        body = build(args.model, prompt, ib64, args.max_tokens)
        t0 = time.perf_counter()
        first_byte_t = None
        try:
            # Non-streaming: TTFT is total request time until final response.
            r = session.post(url, json=body, timeout=300)
            end = time.perf_counter()
            latency = end - t0
            status = r.status_code
            try: payload = r.json()
            except: payload = None
            content = None; finish = None; usage = None
            if payload:
                ch = payload.get("choices", [{}])[0]
                content = ch.get("message", {}).get("content")
                finish = ch.get("finish_reason")
                usage = payload.get("usage")
            reqs.append({"idx": idx, "kind": kind, "http_status": status,
                         "latency_s": latency, "usage": usage,
                         "response_text_len": len(content or ""),
                         "finish_reason": finish,
                         "error": None if status == 200 and content else f"non_200_or_empty: status={status}"})
        except Exception as e:
            reqs.append({"idx": idx, "kind": kind, "http_status": None,
                         "latency_s": time.perf_counter()-t0, "usage": None,
                         "response_text_len": 0, "finish_reason": None,
                         "error": f"request_exception: {e!r}"})
        if (idx + 1) % 20 == 0:
            print(f"[R6.5] request {idx+1}/{len(plan)}", flush=True)

    ended = datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.out_jsonl.open("w") as f:
        for r in reqs: f.write(json.dumps(r) + "\n")

    # Per-kind and overall stats
    all_lat = [r["latency_s"] for r in reqs if r["error"] is None]
    text_lat = [r["latency_s"] for r in reqs if r["kind"] == "text" and r["error"] is None]
    img_lat  = [r["latency_s"] for r in reqs if r["kind"] == "image" and r["error"] is None]
    req_fail = sum(1 for r in reqs if r["error"] is not None)
    args.out_summary.write_text(json.dumps({
        "started_utc": started, "ended_utc": ended,
        "text_ratio": args.text_ratio, "seed": args.seed,
        "n_planned": len(plan), "n_text": n_text, "n_image": n_image,
        "n_completed": len(reqs) - req_fail, "request_failures": req_fail,
        "fixture_sha256": fixture_sha,
        "all_latency_s_stats": stats(all_lat),
        "text_latency_s_stats": stats(text_lat),
        "image_latency_s_stats": stats(img_lat),
    }, indent=2, sort_keys=True))
    print(f"[R6.5] done: n={len(reqs)} fail={req_fail}")
    return 0 if req_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
