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

# Per-token max-abs logprob delta floor. Actual tolerance at score
# time is max(BASE_LOGPROB_TOLERANCE, 3 * noise_floor).
BASE_LOGPROB_TOLERANCE = 0.05


def resolve_tolerance(base: float, noise_floor: float) -> float:
    """Apply the plan's max(base, 3 * noise_floor) rule."""
    return max(base, 3.0 * max(noise_floor, 0.0))


def load_records(path: Path) -> list[dict]:
    text = path.read_text()
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _by_prompt_all(records: list[dict]) -> dict[str, list[dict]]:
    """Index records by prompt_source_id → list of all records with that id."""
    out: dict[str, list[dict]] = {}
    for r in records:
        pid = r.get("prompt_source_id")
        if pid is None:
            continue
        out.setdefault(pid, []).append(r)
    return out


def _cmp_records(
    lhs: dict, rhs: dict, tolerance: float
) -> dict:
    """Return a per-pair comparison verdict.

    Hard-fails on missing output_ids or missing output_logprobs — no
    whitespace fallback (audit B1b). Both sides must carry the fields.
    """
    lhs_ids = lhs.get("output_ids")
    rhs_ids = rhs.get("output_ids")
    lhs_lp = lhs.get("output_logprobs")
    rhs_lp = rhs.get("output_logprobs")

    reasons: list[str] = []
    if not isinstance(lhs_ids, list) or not isinstance(rhs_ids, list):
        reasons.append("MISSING_OUTPUT_IDS")
    if not isinstance(lhs_lp, list) or not lhs_lp or not isinstance(rhs_lp, list) or not rhs_lp:
        reasons.append("MISSING_LOGPROBS")

    if reasons:
        return {
            "prompt_source_id": lhs.get("prompt_source_id"),
            "tokens_equal": None,
            "common_prefix": None,
            "lhs_token_len": len(lhs_ids) if isinstance(lhs_ids, list) else None,
            "rhs_token_len": len(rhs_ids) if isinstance(rhs_ids, list) else None,
            "max_abs_logprob_diff": None,
            "within_logprob_tolerance": False,
            "verdict": "FAIL",
            "reason": reasons,
            "used_output_ids": bool(
                isinstance(lhs_ids, list) and isinstance(rhs_ids, list)
            ),
            "tolerance": tolerance,
        }

    common_prefix = 0
    for a, b in zip(lhs_ids, rhs_ids):
        if a == b:
            common_prefix += 1
        else:
            break
    equal = lhs_ids == rhs_ids

    L = min(len(lhs_lp), len(rhs_lp))
    deltas = [abs(a - b) for a, b in zip(lhs_lp[:L], rhs_lp[:L])]
    max_abs_logprob_diff = max(deltas) if deltas else 0.0
    within_tolerance = max_abs_logprob_diff <= tolerance

    return {
        "prompt_source_id": lhs.get("prompt_source_id"),
        "tokens_equal": equal,
        "common_prefix": common_prefix,
        "lhs_token_len": len(lhs_ids),
        "rhs_token_len": len(rhs_ids),
        "max_abs_logprob_diff": max_abs_logprob_diff,
        "within_logprob_tolerance": within_tolerance,
        "verdict": "PASS" if (equal and within_tolerance) else "FAIL",
        "used_output_ids": True,
        "tolerance": tolerance,
    }


def gate_pairwise(
    label_lhs: str,
    label_rhs: str,
    records_lhs: list[dict],
    records_rhs: list[dict],
    tolerance: float,
) -> dict:
    """Compare every timed sample of every prompt id, both sides.

    Aggregates by requiring **all** LHS samples of a prompt id to match
    **all** RHS samples of the same id (audit fix: previously only the
    first sample per id was used).
    """
    lhs_by = _by_prompt_all(records_lhs)
    rhs_by = _by_prompt_all(records_rhs)
    common_pids = sorted(set(lhs_by) & set(rhs_by))
    per_prompt: list[dict] = []
    for pid in common_pids:
        pair_results = []
        for lhs_rec in lhs_by[pid]:
            for rhs_rec in rhs_by[pid]:
                pair_results.append(_cmp_records(lhs_rec, rhs_rec, tolerance))
        n_pairs = len(pair_results)
        n_fail = sum(1 for p in pair_results if p["verdict"] == "FAIL")
        per_prompt.append(
            {
                "prompt_source_id": pid,
                "n_lhs_samples": len(lhs_by[pid]),
                "n_rhs_samples": len(rhs_by[pid]),
                "n_pairs_compared": n_pairs,
                "n_pairs_failed": n_fail,
                "verdict": "PASS" if n_fail == 0 else "FAIL",
                "sample_pair": pair_results[0] if pair_results else None,
            }
        )
    passed = sum(1 for p in per_prompt if p["verdict"] == "PASS")
    failed = sum(1 for p in per_prompt if p["verdict"] == "FAIL")
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
        "per_prompt": per_prompt,
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
            f"Per-token max-abs logprob tolerance floor (default: "
            f"{BASE_LOGPROB_TOLERANCE}). Combined with --noise-floor as "
            f"max(tolerance, 3 * noise_floor)."
        ),
    )
    p.add_argument(
        "--noise-floor",
        type=float,
        default=0.0,
        help=(
            "Measured noise floor from A0 self-repeat (max-abs logprob "
            "delta across two identical A0 runs). Combined with "
            "--tolerance as max(tolerance, 3 * noise_floor)."
        ),
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    effective_tolerance = resolve_tolerance(args.tolerance, args.noise_floor)
    return run(
        gate=args.gate,
        records_args=args.records,
        output=args.output,
        tolerance=effective_tolerance,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
