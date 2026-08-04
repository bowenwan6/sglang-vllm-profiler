#!/usr/bin/env python3
"""Generate the byte-pinned golden prompt set for the GDN prefill-BCG sweep.

CPU-only, no network, no torch, no HF hub. Reproducibility is by literal
prompt strings plus a deterministic length-materialisation function. The
output is a JSONL file whose SHA-256 is pinned in ``../provenance.md`` /
``../fixtures/manifest.json``.

Length materialisation is character-level, not tokenizer-level: the sweep
records per-cell the tokenizer-side length after the target model's
processor is applied, but the fixture file itself is tokenizer-agnostic
so it can be regenerated on any machine without a torch install.

Usage
-----

    python3 scripts/generate_gdn_prompts.py            # write fixture
    python3 scripts/generate_gdn_prompts.py --check    # verify SHA-256
    python3 scripts/generate_gdn_prompts.py --check --strict  # non-zero on mismatch
    python3 scripts/generate_gdn_prompts.py --print    # emit to stdout, no file
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

FIXTURE_REL = Path("experiments/qwen35_4b/gdn/fixtures/gdn_prompts.jsonl")
MANIFEST_REL = Path("experiments/qwen35_4b/gdn/fixtures/manifest.json")

# Eight prompt seeds spanning short/long, prose/code/QA/multi-turn shapes.
# Text is deliberately generic and self-contained so no external asset is
# required. Any change here changes the fixture SHA and requires an
# explicit provenance amendment.
SEEDS: list[dict] = [
    {
        "id": "g0_short_prose",
        "kind": "prose",
        "seed": (
            "Explain in one paragraph why a laminar flame front becomes "
            "unstable as the Reynolds number grows past the critical "
            "value."
        ),
    },
    {
        "id": "g1_short_qa",
        "kind": "qa",
        "seed": (
            "Question: What is the primary difference between snapshot "
            "isolation and serializable isolation in a relational "
            "database? Answer:"
        ),
    },
    {
        "id": "g2_short_code",
        "kind": "code",
        "seed": (
            "Write a Python function `median_of_medians(arr)` that "
            "returns the exact median in O(n) using the median-of-"
            "medians pivot rule. Include a short docstring."
        ),
    },
    {
        "id": "g3_short_multiturn",
        "kind": "multiturn",
        "seed": (
            "User: I want to design a merge sort that is stable and "
            "in-place. Assistant: True in-place stable merge sort is "
            "notoriously hard. User: Why? Assistant:"
        ),
    },
    {
        "id": "g4_long_prose",
        "kind": "prose",
        "seed": (
            "Compare and contrast the operational-transformation and "
            "CRDT approaches to real-time collaborative editing. "
            "Consider latency, causality, garbage collection, and "
            "the failure modes of each under network partition. Cite "
            "at least two textbook examples for each family."
        ),
    },
    {
        "id": "g5_long_code",
        "kind": "code",
        "seed": (
            "Implement a small typed interpreter in Rust for an "
            "arithmetic expression language with integers, floats, "
            "let-bindings, and function application. Include the AST "
            "definition, a recursive-descent parser, an evaluator, "
            "and three end-to-end tests. Use `thiserror` for the "
            "error type."
        ),
    },
    {
        "id": "g6_long_qa",
        "kind": "qa",
        "seed": (
            "Question: Describe how a modern CUDA-graph runtime handles "
            "kernel launches whose input tensor addresses vary across "
            "replays. Focus on the buffer-registry pattern, the "
            "stable-slot copy-in, and the failure modes when a slot is "
            "not registered for a particular argument. Answer:"
        ),
    },
    {
        "id": "g7_long_multiturn",
        "kind": "multiturn",
        "seed": (
            "User: I want to profile a hybrid attention model with "
            "Nsight Systems. Which trace channels should I enable? "
            "Assistant: For CUDA-graph-covered kernels you need cuda "
            "and nvtx; for host-side stalls add osrt; for libraries "
            "add cublas and cudnn. User: How do I tell whether a "
            "specific op is inside the captured graph or launched "
            "host-side? Assistant:"
        ),
    },
]


def materialise(seed_text: str, target_chars: int) -> str:
    """Deterministically stretch or truncate `seed_text` to `target_chars`.

    Stretching is by repeating a short delimiter-separated tail:
    " Consider the following secondary case: <seed> " until the length
    target is met, then truncating to exactly `target_chars`.
    """
    if len(seed_text) >= target_chars:
        return seed_text[:target_chars]
    tail_template = " Consider the following secondary case: " + seed_text
    buf = [seed_text]
    total = len(seed_text)
    while total < target_chars:
        buf.append(tail_template)
        total += len(tail_template)
    return "".join(buf)[:target_chars]


def build_records() -> list[dict]:
    # Length targets chosen so that the golden prompt set includes both
    # short and long shapes at the char level; the sweep prompt-length
    # bins (128/512/2048/8192 tokens) are materialised per-cell by the
    # client, not here.
    records = []
    for seed in SEEDS:
        for target_chars in (256, 4096):
            text = materialise(seed["seed"], target_chars)
            records.append(
                {
                    "id": f"{seed['id']}_c{target_chars}",
                    "seed_id": seed["id"],
                    "kind": seed["kind"],
                    "target_chars": target_chars,
                    "text": text,
                }
            )
    return records


def dump_jsonl(records: list[dict]) -> bytes:
    parts = [json.dumps(rec, sort_keys=True, ensure_ascii=True) for rec in records]
    return ("\n".join(parts) + "\n").encode("ascii")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def write_fixture(dry_run: bool = False) -> tuple[Path, str]:
    records = build_records()
    payload = dump_jsonl(records)
    sha = hashlib.sha256(payload).hexdigest()
    out = repo_root() / FIXTURE_REL
    if not dry_run:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(payload)
        manifest = {
            "path": str(FIXTURE_REL).replace("\\", "/"),
            "sha256": sha,
            "n_records": len(records),
            "n_seeds": len(SEEDS),
        }
        (repo_root() / MANIFEST_REL).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    return out, sha


def check_fixture(strict: bool) -> int:
    manifest_path = repo_root() / MANIFEST_REL
    fixture_path = repo_root() / FIXTURE_REL
    if not manifest_path.exists() or not fixture_path.exists():
        print("MISS: fixture or manifest missing", file=sys.stderr)
        return 1 if strict else 0
    expected = json.loads(manifest_path.read_text())["sha256"]
    got = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    if expected != got:
        print(f"MISMATCH: fixture sha {got} != manifest sha {expected}", file=sys.stderr)
        return 1
    print(f"OK: fixture sha {got}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check", action="store_true", help="verify SHA-256 only")
    p.add_argument("--strict", action="store_true", help="non-zero on any drift")
    p.add_argument("--print", action="store_true", help="emit to stdout, no file")
    args = p.parse_args(argv)
    if args.check:
        return check_fixture(strict=args.strict)
    if args.print:
        records = build_records()
        sys.stdout.buffer.write(dump_jsonl(records))
        return 0
    path, sha = write_fixture(dry_run=False)
    print(f"wrote {path.relative_to(repo_root())}  sha256={sha}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
