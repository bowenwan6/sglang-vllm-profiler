#!/usr/bin/env python3
"""GDN verdict runner.

Ingests the four correctness-gate JSON summaries plus (optionally)
per-cell Nsight CSVs and emits exactly one of the labels declared in
``hypothesis.md`` §5:

    PASS_BCG_GDN_NOTABLE_GAP
    PASS_BCG_GDN_NO_GAP
    FAIL_BCG_GDN_CORRECTNESS
    AMBIGUOUS
    INFRA_FAILURE

The label strings are the source of truth. Adding a label requires an
Amendment in ``hypothesis.md``.

Arm-comparison thresholds (from ``validation_plan.md`` §6):

    H_A support (kernel-count inflation): kernel_count_gdn strictly
        greater than A0 mean by >= 10% AND > 2 * A0_stddev, across
        >= 2 captures.
    H_B support (graph break):            graph_breaks > 0 under a
        BCG-enabled arm on a request whose prefill fits one bucket,
        reproducible across >= 2 captures.
    H_C support (launch-overhead):        p95_launch_gap under a
        BCG-enabled arm >= 2 * A0 p95_launch_gap at the same cell.

The verdict runner is CPU-only. --dry-run scores synthetic gate + nsys
inputs to exercise the logic.

Usage
-----

    python3 scripts/gdn_verdict.py \
        --gate 1=gate1.json --gate 2=gate2.json --gate 3=gate3.json --gate 4=gate4.json \
        --nsys-csv-dir results/attempt_.../nsys/ \
        --output verdict.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

VERDICT_LABELS = (
    "PASS_BCG_GDN_NOTABLE_GAP",
    "PASS_BCG_GDN_NO_GAP",
    "FAIL_BCG_GDN_CORRECTNESS",
    "PARTIAL_SWEEP",
    "AMBIGUOUS",
    "INFRA_FAILURE",
)

# BCG-enabled arms only (audit fix: H_A previously iterated A2 too,
# risking a false positive PASS_BCG_GDN_NOTABLE_GAP from a decode-CG
# artefact — A2's prefill is eager, so any GDN-side kernel divergence
# there cannot be attributed to BCG).
_BCG_ARMS = ("A1", "A3")

# From validation_plan.md §6.
KERNEL_INFLATION_FACTOR = 0.10  # >= 10% over A0 mean
KERNEL_STDDEV_MULTIPLIER = 2.0  # AND > 2 * A0 stddev
LAUNCH_GAP_MULTIPLIER = 2.0  # p95 launch gap >= 2 * A0 p95
MIN_CAPTURES_FOR_REPRO = 2


def _load_gate(path_or_arg: str) -> tuple[int, dict]:
    if "=" not in path_or_arg:
        raise SystemExit(
            f"gdn_verdict: --gate must be N=PATH; got {path_or_arg!r}"
        )
    n, p = path_or_arg.split("=", 1)
    return int(n), json.loads(Path(p).read_text())


def _score_gates(gates: dict[int, dict]) -> dict:
    """Score the correctness gates.

    Distinguishes MISSING from FAIL so decide() can emit PARTIAL_SWEEP
    when a scaffolding gap left a gate unrun, versus a real correctness
    failure. Previously any MISSING silently collapsed to
    FAIL_BCG_GDN_CORRECTNESS.
    """
    per_gate = {}
    all_pass = True
    any_missing = False
    any_failed = False
    for n in (1, 2, 3, 4):
        g = gates.get(n)
        if g is None:
            per_gate[n] = {"overall": "MISSING"}
            any_missing = True
            all_pass = False
            continue
        overall = g.get("overall", "MISSING")
        per_gate[n] = {"overall": overall}
        if overall == "MISSING":
            any_missing = True
            all_pass = False
        elif overall != "PASS":
            any_failed = True
            all_pass = False
    return {
        "per_gate": per_gate,
        "all_pass": all_pass,
        "any_missing": any_missing,
        "any_failed": any_failed,
        "partial": any_missing and not any_failed,
    }


def _read_nsys_csv(csv_dir: Path) -> list[dict]:
    if not csv_dir.is_dir():
        return []
    out = []
    for f in sorted(csv_dir.glob("*.csv")):
        with f.open() as fh:
            for row in csv.DictReader(fh):
                out.append({**row, "_source_file": f.name})
    return out


def _to_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _score_perf(rows: list[dict]) -> dict:
    """Score the arm comparisons per (prompt_len, batch) cell."""
    findings: list[dict] = []
    # Group by cell.
    cells: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        key = (r.get("prompt_len", "?"), r.get("batch", "?"))
        cells.setdefault(key, []).append(r)
    for (prompt_len, batch), cell_rows in cells.items():
        by_arm: dict[str, list[dict]] = {}
        for r in cell_rows:
            by_arm.setdefault(r.get("arm", "?"), []).append(r)
        a0 = by_arm.get("A0", [])
        if len(a0) < MIN_CAPTURES_FOR_REPRO:
            continue
        a0_kern = [_to_float(x.get("kernel_count_gdn")) for x in a0]
        a0_kern = [v for v in a0_kern if v is not None]
        a0_gap = [_to_float(x.get("p95_launch_gap_us")) for x in a0]
        a0_gap = [v for v in a0_gap if v is not None]
        if not a0_kern or not a0_gap:
            continue
        a0_kern_mean = statistics.fmean(a0_kern)
        a0_kern_std = statistics.pstdev(a0_kern) if len(a0_kern) > 1 else 0.0
        a0_gap_p95 = statistics.fmean(a0_gap)
        for arm in ("A1", "A2", "A3"):
            arm_rows = by_arm.get(arm, [])
            if len(arm_rows) < MIN_CAPTURES_FOR_REPRO:
                continue
            arm_kern = [_to_float(x.get("kernel_count_gdn")) for x in arm_rows]
            arm_kern = [v for v in arm_kern if v is not None]
            arm_gap = [_to_float(x.get("p95_launch_gap_us")) for x in arm_rows]
            arm_gap = [v for v in arm_gap if v is not None]
            arm_breaks = [
                _to_float(x.get("graph_breaks")) or 0.0 for x in arm_rows
            ]
            kern_mean = statistics.fmean(arm_kern) if arm_kern else None
            gap_p95 = statistics.fmean(arm_gap) if arm_gap else None
            max_breaks = max(arm_breaks) if arm_breaks else 0.0

            # H_A: kernel inflation. BCG-enabled arms only — A2's
            # prefill is eager, so any GDN-side kernel divergence there
            # cannot be attributed to BCG (audit B8 fix).
            h_a = (
                arm in _BCG_ARMS
                and kern_mean is not None
                and a0_kern_mean > 0
                and kern_mean >= a0_kern_mean * (1 + KERNEL_INFLATION_FACTOR)
                and (kern_mean - a0_kern_mean) > KERNEL_STDDEV_MULTIPLIER * a0_kern_std
            )
            # H_B: graph break present (BCG-enabled arms only).
            h_b = arm in _BCG_ARMS and max_breaks > 0
            # H_C: launch-gap inflation (BCG-enabled arms only).
            h_c = (
                arm in _BCG_ARMS
                and gap_p95 is not None
                and a0_gap_p95 > 0
                and gap_p95 >= LAUNCH_GAP_MULTIPLIER * a0_gap_p95
            )
            findings.append(
                {
                    "cell": {"prompt_len": prompt_len, "batch": batch},
                    "arm": arm,
                    "a0_kern_mean": a0_kern_mean,
                    "a0_kern_stddev": a0_kern_std,
                    "arm_kern_mean": kern_mean,
                    "a0_gap_p95_mean": a0_gap_p95,
                    "arm_gap_p95_mean": gap_p95,
                    "arm_max_graph_breaks": max_breaks,
                    "H_A": bool(h_a),
                    "H_B": bool(h_b),
                    "H_C": bool(h_c),
                }
            )
    any_h_a = any(f["H_A"] for f in findings)
    any_h_b = any(f["H_B"] for f in findings)
    any_h_c = any(f["H_C"] for f in findings)
    return {
        "findings": findings,
        "any_H_A": any_h_a,
        "any_H_B": any_h_b,
        "any_H_C": any_h_c,
        "any_bcg_gap": any_h_a or any_h_b or any_h_c,
    }


def decide(
    gate_summary: dict,
    perf_summary: dict,
    infra_failure: bool,
) -> str:
    if infra_failure:
        return "INFRA_FAILURE"
    # A true correctness failure (gate reached, FAIL result) wins over
    # PARTIAL_SWEEP: even if other gates were skipped, one gate FAILing
    # is decisive evidence.
    if gate_summary.get("any_failed"):
        return "FAIL_BCG_GDN_CORRECTNESS"
    # Missing gates without any failure means the sweep did not
    # complete enough to conclude — emit PARTIAL_SWEEP (audit B9 fix).
    if gate_summary.get("partial"):
        return "PARTIAL_SWEEP"
    if perf_summary.get("any_bcg_gap"):
        return "PASS_BCG_GDN_NOTABLE_GAP"
    if perf_summary.get("findings"):
        return "PASS_BCG_GDN_NO_GAP"
    return "AMBIGUOUS"


def run(
    gate_args: list[str],
    nsys_csv_dir: Path | None,
    output: Path | None,
    infra_failure: bool,
    dry_run: bool,
) -> int:
    if dry_run:
        # Synthetic scenario: all gates PASS, no perf gap.
        gate_summary = {
            "per_gate": {i: {"overall": "PASS"} for i in (1, 2, 3, 4)},
            "all_pass": True,
            "any_missing": False,
            "any_failed": False,
            "partial": False,
        }
        perf_summary = {
            "findings": [],
            "any_H_A": False,
            "any_H_B": False,
            "any_H_C": False,
            "any_bcg_gap": False,
        }
        verdict = decide(gate_summary, perf_summary, infra_failure)
        payload = {
            "verdict": verdict,
            "gate_summary": gate_summary,
            "perf_summary": perf_summary,
            "dry_run": True,
        }
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
        return 0

    gates = dict(_load_gate(g) for g in gate_args)
    gate_summary = _score_gates(gates)
    perf_rows = _read_nsys_csv(nsys_csv_dir) if nsys_csv_dir else []
    perf_summary = _score_perf(perf_rows)
    verdict = decide(gate_summary, perf_summary, infra_failure)
    payload = {
        "verdict": verdict,
        "gate_summary": gate_summary,
        "perf_summary": perf_summary,
        "n_perf_rows": len(perf_rows),
    }
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    else:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True, default=str)
        sys.stdout.write("\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gate", action="append", default=[])
    p.add_argument("--nsys-csv-dir", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument(
        "--infra-failure",
        action="store_true",
        help="Override: emit INFRA_FAILURE regardless of gate content.",
    )
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    return run(
        gate_args=args.gate,
        nsys_csv_dir=args.nsys_csv_dir,
        output=args.output,
        infra_failure=args.infra_failure,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
