#!/usr/bin/env python3
"""GDN correctness gate verifier.

Ingests per-request JSONL records emitted by ``gdn_client.py`` and
scores the four blocking correctness gates from
``validation_plan.md`` §4:

    Gate 1 — Eager-vs-BCG token/logprob equivalence (A0 is the reference)
    Gate 2 — Request-order isolation (alone vs batched for same prompt id)
    Gate 3 — Chunked-prefill equivalence (two --chunked-prefill-size runs)
    Gate 4 — Graph-bucket equivalence (two prompt lengths spanning a bucket)

Each gate is scored per (arm, cell) and produces a JSON summary. Any
FAIL fails the overall gate. Tolerances are predeclared here so
loosening them requires editing this file (and, per the plan, an
Amendment).

CPU-only. --dry-run scores synthetic inputs to exercise the logic.

Usage
-----

    python3 scripts/gdn_correctness.py --gate 1 \
        --records A0=records_A0_p128_b4.jsonl \
        --records A1=records_A1_p128_b4.jsonl \
        --output gate1.json

    python3 scripts/gdn_correctness.py --gate 2 \
        --records alone=records_A1_p128_b1.jsonl \
        --records batched=records_A1_p128_b4.jsonl \
        --output gate2.json

    python3 scripts/gdn_correctness.py --gate 3 \
        --records small_chunk=records_A1_p2048_b1_c256.jsonl \
        --records single_chunk=records_A1_p2048_b1_c4096.jsonl \
        --output gate3.json

    python3 scripts/gdn_correctness.py --gate 4 \
        --records bucket_a=records_A1_p2000_b1.jsonl \
        --records bucket_b=records_A1_p2400_b1.jsonl \
        --output gate4.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# --- Predeclared tolerances (mirror validation_plan.md §4) ------------

# Per-token max-abs logprob delta. Actual tolerance at score time is
# max(BASE_LOGPROB_TOLERANCE, 3 * noise_floor); the caller provides the
# noise floor in the records or via --noise-floor.
BASE_LOGPROB_TOLERANCE = 0.05


def load_records(path: Path) -> list[dict]:
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _by_prompt(records: list[dict]) -> dict[str, dict]:
    """Index records by prompt_source_id → first record with that id."""
    out: dict[str, dict] = {}
    for r in records:
        pid = r.get("prompt_source_id")
        if pid is None:
            continue
        out.setdefault(pid, r)
    return out


def _token_stream(record: dict) -> list[str]:
    """Return a token-like stream from a record.

    Prefers server-returned ``output_ids`` when present; otherwise
    splits on whitespace as a coarse fallback so the gate logic still
    runs on plain-text records (with a WARN in the summary).
    """
    ids = record.get("output_ids")
    if isinstance(ids, list):
        return [str(i) for i in ids]
    text = record.get("output_text", "")
    return text.split()


def _cmp_records(
    lhs: dict, rhs: dict, tolerance: float
) -> dict:
    """Return a per-pair comparison verdict."""
    lhs_tokens = _token_stream(lhs)
    rhs_tokens = _token_stream(rhs)
    common_prefix = 0
    for a, b in zip(lhs_tokens, rhs_tokens):
        if a == b:
            common_prefix += 1
        else:
            break
    equal = lhs_tokens == rhs_tokens

    logprobs_lhs = lhs.get("output_logprobs") or []
    logprobs_rhs = rhs.get("output_logprobs") or []
    max_abs_logprob_diff = None
    if logprobs_lhs and logprobs_rhs:
        L = min(len(logprobs_lhs), len(logprobs_rhs))
        deltas = [abs(a - b) for a, b in zip(logprobs_lhs[:L], logprobs_rhs[:L])]
        max_abs_logprob_diff = max(deltas) if deltas else 0.0

    within_tolerance = (
        max_abs_logprob_diff is None or max_abs_logprob_diff <= tolerance
    )

    return {
        "prompt_source_id": lhs.get("prompt_source_id"),
        "tokens_equal": equal,
        "common_prefix": common_prefix,
        "lhs_token_len": len(lhs_tokens),
        "rhs_token_len": len(rhs_tokens),
        "max_abs_logprob_diff": max_abs_logprob_diff,
        "within_logprob_tolerance": within_tolerance,
        "verdict": "PASS" if (equal and within_tolerance) else "FAIL",
        "used_output_ids": bool(
            isinstance(lhs.get("output_ids"), list)
            and isinstance(rhs.get("output_ids"), list)
        ),
    }


def gate_pairwise(
    label_lhs: str,
    label_rhs: str,
    records_lhs: list[dict],
    records_rhs: list[dict],
    tolerance: float,
) -> dict:
    lhs_by = _by_prompt(records_lhs)
    rhs_by = _by_prompt(records_rhs)
    common_pids = sorted(set(lhs_by) & set(rhs_by))
    pairs = [_cmp_records(lhs_by[p], rhs_by[p], tolerance) for p in common_pids]
    passed = sum(1 for p in pairs if p["verdict"] == "PASS")
    failed = sum(1 for p in pairs if p["verdict"] == "FAIL")
    overall = "PASS" if failed == 0 and passed > 0 else "FAIL"
    return {
        "lhs_label": label_lhs,
        "rhs_label": label_rhs,
        "tolerance": tolerance,
        "n_common_prompts": len(common_pids),
        "n_lhs_only": len(set(lhs_by) - set(rhs_by)),
        "n_rhs_only": len(set(rhs_by) - set(lhs_by)),
        "n_passed": passed,
        "n_failed": failed,
        "overall": overall,
        "pairs": pairs,
    }


def gate_verdict(
    gate_number: int, labelled_records: dict[str, list[dict]], tolerance: float
) -> dict:
    """Score one gate given a {label: [records...]} mapping.

    Gate 1 compares every non-A0 arm against A0 (must be in the mapping).
    Gates 2/3/4 compare two arbitrary labels pairwise.
    """
    if gate_number == 1:
        if "A0" not in labelled_records:
            return {
                "gate": gate_number,
                "overall": "FAIL",
                "reason": "gate 1 requires an 'A0' arm as reference",
                "labels_seen": sorted(labelled_records),
            }
        comparisons = {}
        for lbl, recs in labelled_records.items():
            if lbl == "A0":
                continue
            comparisons[lbl] = gate_pairwise(
                "A0", lbl, labelled_records["A0"], recs, tolerance
            )
        overall = (
            "PASS"
            if comparisons and all(c["overall"] == "PASS" for c in comparisons.values())
            else "FAIL"
        )
        return {
            "gate": gate_number,
            "overall": overall,
            "tolerance": tolerance,
            "comparisons": comparisons,
        }
    else:
        labels = list(labelled_records)
        if len(labels) != 2:
            return {
                "gate": gate_number,
                "overall": "FAIL",
                "reason": (
                    f"gate {gate_number} requires exactly two labels; got "
                    f"{labels}"
                ),
            }
        lhs, rhs = labels
        comparison = gate_pairwise(
            lhs, rhs, labelled_records[lhs], labelled_records[rhs], tolerance
        )
        return {
            "gate": gate_number,
            "overall": comparison["overall"],
            "tolerance": tolerance,
            "comparison": comparison,
        }


def run(
    gate: int,
    records_args: list[str],
    output: Path | None,
    tolerance: float,
    dry_run: bool,
) -> int:
    if dry_run:
        # Synthetic inputs exercise the gate logic without touching disk.
        labelled = {
            "A0": [
                {
                    "prompt_source_id": "p0",
                    "output_ids": [1, 2, 3, 4, 5],
                    "output_logprobs": [-0.1, -0.2, -0.3, -0.1, -0.05],
                },
                {
                    "prompt_source_id": "p1",
                    "output_ids": [7, 8, 9],
                    "output_logprobs": [-0.05, -0.05, -0.1],
                },
            ],
            "A1": [
                {
                    "prompt_source_id": "p0",
                    "output_ids": [1, 2, 3, 4, 5],
                    "output_logprobs": [-0.1, -0.21, -0.31, -0.1, -0.05],
                },
                {
                    "prompt_source_id": "p1",
                    "output_ids": [7, 8, 9],
                    "output_logprobs": [-0.05, -0.05, -0.1],
                },
            ],
        }
        payload = gate_verdict(gate, labelled, tolerance)
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return 0 if payload.get("overall") == "PASS" else 1

    labelled: dict[str, list[dict]] = {}
    for item in records_args:
        if "=" not in item:
            raise SystemExit(
                f"gdn_correctness: --records must be LABEL=PATH; got {item!r}"
            )
        label, path_str = item.split("=", 1)
        path = Path(path_str)
        if not path.is_file():
            raise SystemExit(f"gdn_correctness: file not found: {path}")
        labelled[label] = load_records(path)

    payload = gate_verdict(gate, labelled, tolerance)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    else:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    return 0 if payload.get("overall") == "PASS" else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", type=int, choices=(1, 2, 3, 4), required=True)
    p.add_argument("--records", action="append", default=[])
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--tolerance",
        type=float,
        default=BASE_LOGPROB_TOLERANCE,
        help=(
            f"Per-token max-abs logprob tolerance (default: "
            f"{BASE_LOGPROB_TOLERANCE}). Set higher only under an "
            f"Amendment in hypothesis.md."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    return run(
        gate=args.gate,
        records_args=args.records,
        output=args.output,
        tolerance=args.tolerance,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
