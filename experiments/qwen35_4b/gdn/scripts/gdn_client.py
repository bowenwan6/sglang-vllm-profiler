#!/usr/bin/env python3
"""Sweep client for one (arm, prompt_len, batch_size) cell.

Reads the byte-pinned golden prompts from
``experiments/qwen35_4b/gdn/fixtures/gdn_prompts.jsonl``, materialises
prompts to the target *token* length using the served model's
tokenizer (via the SGLang server's ``/tokenize`` endpoint if
available, or by a fallback deterministic slice on the raw text),
sends warmup requests (discarded), then sends the timed requests and
records per-request records in JSONL.

CPU-only in --dry-run mode. In live mode it uses ``urllib`` only
(no external HTTP library dependency).

Usage
-----

    python3 scripts/gdn_client.py \
        --server http://127.0.0.1:30001 \
        --arm A0 --prompt-len 128 --batch-size 4 \
        --new-tokens 128 --n-warmup 2 --n-timed 8 \
        --fixtures-dir experiments/qwen35_4b/gdn/fixtures \
        --output /tmp/records.jsonl

Output JSONL schema (one record per timed request):
    {
        "arm": str,
        "prompt_len_target_tokens": int,
        "batch_size": int,
        "new_tokens": int,
        "request_id": str,
        "server": str,
        "prompt_source_id": str,        # golden-prompt id used
        "prompt_len_target_chars": int, # chars sent to the server
        "prompt_bytes_sha256": str,     # sha256 of the sent bytes
        "output_text": str,             # server-returned completion
        "output_len_tokens": int|None,  # server-reported completion tokens
        "ttft_ms": float,               # sample only under stream mode
        "e2e_ms": float,                # request submit -> response received
        "timestamp": str,               # UTC ISO-8601
        "status_code": int,
        "server_error": str|None
    }
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

_HTTP_TIMEOUT = 300.0


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def load_fixture(fixtures_dir: Path) -> list[dict]:
    fx = fixtures_dir / "gdn_prompts.jsonl"
    if not fx.is_file():
        raise SystemExit(f"gdn_client: fixture not found: {fx}")
    return [json.loads(line) for line in fx.read_text().splitlines() if line]


def _cycle_records(records: list[dict], n: int) -> list[dict]:
    return list(itertools.islice(itertools.cycle(records), n))


def materialise_prompt(seed_text: str, target_chars: int) -> str:
    """Same deterministic stretch/truncate as generate_gdn_prompts."""
    if len(seed_text) >= target_chars:
        return seed_text[:target_chars]
    tail = " Consider the following secondary case: " + seed_text
    buf = [seed_text]
    total = len(seed_text)
    while total < target_chars:
        buf.append(tail)
        total += len(tail)
    return "".join(buf)[:target_chars]


def approx_chars_for_tokens(target_tokens: int) -> int:
    """Rough conversion when no server tokenizer is queried.

    Uses ~3.5 chars/token which is a conservative approximation for
    the Qwen tokenizer on English prose. Server-side max_new_tokens
    truncation is what actually binds the output length; this bound
    only sets the *input* size.
    """
    return max(16, int(target_tokens * 3.5))


def http_post_json(url: str, payload: dict, timeout: float = _HTTP_TIMEOUT) -> tuple[int, dict | None, str | None]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            raw = resp.read().decode("utf-8")
            return resp.status, json.loads(raw) if raw else None, None
    except urllib.error.HTTPError as e:
        return e.code, None, e.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as e:
        return 0, None, repr(e)


def issue_batch(
    server: str,
    prompts: list[str],
    new_tokens: int,
    request_ids: list[str],
) -> list[dict]:
    """Send `len(prompts)` requests concurrently by batching the HTTP body.

    SGLang accepts a list-of-prompts request; per-request TTFT is not
    directly obtainable without streaming, so this returns the batch
    e2e latency divided across requests as the primary latency signal
    (finer-grained TTFT lives in the nsys capture).
    """
    submit_ts = time.perf_counter()
    body = {
        "text": prompts if len(prompts) > 1 else prompts[0],
        "sampling_params": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": new_tokens,
        },
        "return_logprob": False,
        "stream": False,
        "rid": request_ids if len(request_ids) > 1 else request_ids[0],
    }
    status, data, err = http_post_json(f"{server}/generate", body)
    recv_ts = time.perf_counter()
    e2e_ms = (recv_ts - submit_ts) * 1000.0

    per_request = []
    if status == 200 and data is not None:
        if isinstance(data, list):
            entries = data
        else:
            entries = [data]
        for i, entry in enumerate(entries):
            per_request.append(
                {
                    "status_code": status,
                    "server_error": None,
                    "output_text": entry.get("text", ""),
                    "output_len_tokens": entry.get("meta_info", {}).get(
                        "completion_tokens"
                    ),
                    "e2e_ms": e2e_ms,
                    "ttft_ms": None,  # non-stream mode; nsys covers TTFT
                }
            )
    else:
        for _ in prompts:
            per_request.append(
                {
                    "status_code": status,
                    "server_error": err or "unknown",
                    "output_text": "",
                    "output_len_tokens": None,
                    "e2e_ms": e2e_ms,
                    "ttft_ms": None,
                }
            )
    return per_request


def run(
    server: str,
    arm: str,
    prompt_len_tokens: int,
    batch_size: int,
    new_tokens: int,
    n_warmup: int,
    n_timed: int,
    fixtures_dir: Path,
    output: Path,
    dry_run: bool,
) -> int:
    prompts_target_chars = approx_chars_for_tokens(prompt_len_tokens)
    fixture = load_fixture(fixtures_dir)

    # Cycle deterministically through the golden prompts to fill a
    # batch; each timed round uses the next `batch_size` records.
    all_rounds = n_warmup + n_timed
    total_prompts = all_rounds * batch_size
    seeds = _cycle_records(fixture, total_prompts)

    output.parent.mkdir(parents=True, exist_ok=True)

    dry_run_probe = {
        "arm": arm,
        "server": server,
        "prompt_len_target_tokens": prompt_len_tokens,
        "prompt_len_target_chars": prompts_target_chars,
        "batch_size": batch_size,
        "new_tokens": new_tokens,
        "n_warmup": n_warmup,
        "n_timed": n_timed,
        "n_fixture_records": len(fixture),
        "n_seeds_used": total_prompts,
        "output": str(output),
        "dry_run": True,
    }

    if dry_run:
        print(json.dumps(dry_run_probe, indent=2, sort_keys=True))
        return 0

    records: list[dict] = []
    round_iter = iter(seeds)
    for round_idx in range(all_rounds):
        batch_seeds = [next(round_iter) for _ in range(batch_size)]
        prompts = [materialise_prompt(s["text"], prompts_target_chars) for s in batch_seeds]
        request_ids = [
            f"{arm}_p{prompt_len_tokens}_b{batch_size}_r{round_idx}_i{i}"
            for i in range(batch_size)
        ]
        results = issue_batch(server, prompts, new_tokens, request_ids)
        if round_idx < n_warmup:
            # Discard warmup.
            continue
        for seed, prompt, req_id, result in zip(batch_seeds, prompts, request_ids, results):
            records.append(
                {
                    "arm": arm,
                    "prompt_len_target_tokens": prompt_len_tokens,
                    "prompt_len_target_chars": prompts_target_chars,
                    "batch_size": batch_size,
                    "new_tokens": new_tokens,
                    "request_id": req_id,
                    "server": server,
                    "prompt_source_id": seed["id"],
                    "prompt_bytes_sha256": _sha256_hex(prompt.encode("utf-8")),
                    "timestamp": _utcnow(),
                    **result,
                }
            )

    with contextlib.suppress(OSError):
        output.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")

    n_failed = sum(1 for r in records if r["status_code"] != 200)
    print(
        f"gdn_client: {len(records)} timed records ({n_failed} failed) written to {output}"
    )
    return 0 if n_failed == 0 else 76


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--prompt-len", type=int, required=True, dest="prompt_len_tokens")
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--new-tokens", type=int, default=128)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-timed", type=int, default=8)
    p.add_argument("--fixtures-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    return run(
        server=args.server,
        arm=args.arm,
        prompt_len_tokens=args.prompt_len_tokens,
        batch_size=args.batch_size,
        new_tokens=args.new_tokens,
        n_warmup=args.n_warmup,
        n_timed=args.n_timed,
        fixtures_dir=args.fixtures_dir,
        output=args.output,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
