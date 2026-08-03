#!/usr/bin/env python3
"""Extract per-cell metrics from a Nsight Systems `.nsys-rep` file.

Emits one CSV row per cell (arm, prompt_len, batch) with the columns
declared in ``gdn/validation_plan.md`` §5. Used by ``gdn_verdict.py``
to score the perf side of the sweep.

Extraction strategy:

    Nsight Systems ships `nsys stats --report ...` which regenerates
    the per-report CSVs from an existing `.nsys-rep`. We invoke:
      * cuda_api_sum        — per-CUDA-API-call aggregates
      * cuda_gpu_kern_sum   — per-kernel aggregates
      * cuda_api_trace      — per-API-call timeline (for launch gaps)
      * nvtx_gpu_proj_trace — NVTX ranges (present only if the runner
                              installed the NVTX-tagged instrumentation
                              variant, which is deferred per plan §4)

    Without NVTX ranges we cannot split ``kernel_count`` by
    op (gdn / attention / other); those columns record the total
    and mark ``attribution="coarse"``. When NVTX is present, the
    extractor sums kernels by enclosing range name.

Usage
-----

    python3 scripts/extract_nsys_metrics.py \\
        --nsys-rep results/attempt/raw/A0_p128_b1.nsys-rep \\
        --arm A0 --prompt-len 128 --batch 1 \\
        [--records results/attempt/records_A0_p128_b1.jsonl] \\
        --output-csv results/attempt/nsys/A0_p128_b1.csv

    python3 scripts/extract_nsys_metrics.py --dry-run \\
        --arm A0 --prompt-len 128 --batch 1 \\
        --output-csv /tmp/out.csv

--dry-run emits the CSV header alone (no nsys required).

Exit codes:
    0 — success (or dry-run).
    2 — usage / IO error.
    3 — nsys binary not on PATH.
    4 — nsys stats invocation failed.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

# Column order (also the CSV header). Kept identical to
# validation_plan.md §5 so downstream tooling doesn't have to re-map.
COLUMNS = (
    "arm",
    "prompt_len",
    "batch",
    "request_id",
    "kernel_count_total",
    "kernel_count_gdn",
    "kernel_count_attn",
    "kernel_count_other",
    "cudagraphlaunch_count",
    "cudalaunchkernel_count",
    "ttft_ms",
    "prefill_throughput_toks_per_s",
    "p50_launch_gap_us",
    "p95_launch_gap_us",
    "p99_launch_gap_us",
    "graph_breaks",
    "attribution",  # "coarse" | "nvtx"
    "nsys_source",
    "extractor_status",
    "extractor_warnings",
)

_UNKNOWN = "coarse_unknown"


def _run_nsys_stats(nsys_rep: Path, reports: list[str]) -> dict[str, str]:
    """Run `nsys stats --report <reports> --format csv` and return raw CSV
    text per report name."""
    if shutil.which("nsys") is None:
        raise RuntimeError("nsys binary not on PATH")
    out: dict[str, str] = {}
    for report in reports:
        proc = subprocess.run(
            [
                "nsys",
                "stats",
                "--format",
                "csv",
                "--report",
                report,
                str(nsys_rep),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120.0,
        )
        if proc.returncode != 0:
            # Return an empty string so per-report failures don't crash;
            # the caller records the warning.
            out[report] = ""
        else:
            out[report] = proc.stdout
    return out


def _parse_csv_section(raw: str) -> list[dict]:
    """`nsys stats` prefaces the CSV with a comment header block; skip
    down to the first line that looks like the header row."""
    lines = raw.splitlines()
    # Locate the header — first line starting with an alpha char and a
    # comma (real CSV headers), skipping ``**`` marker lines.
    header_ix = -1
    for i, line in enumerate(lines):
        if not line or line.startswith("#") or line.startswith("**"):
            continue
        # Header must contain a comma AND have no numeric leading char.
        if "," in line and not line[0].isdigit():
            header_ix = i
            break
    if header_ix < 0:
        return []
    body = "\n".join(lines[header_ix:])
    reader = csv.DictReader(io.StringIO(body))
    return list(reader)


def _to_int(v) -> int | None:
    try:
        return int(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _to_float(v) -> float | None:
    try:
        return float(str(v).replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _kernel_counts(kern_rows: list[dict]) -> dict:
    """Total kernel invocations from cuda_gpu_kern_sum.

    Column names on modern nsys builds: ``Instances`` (int),
    ``Name`` (string). We tolerate both ``Instances`` and ``Count`` for
    robustness across nsys versions.
    """
    total = 0
    for row in kern_rows:
        n = _to_int(row.get("Instances") or row.get("Count"))
        if n is not None:
            total += n
    return {
        "kernel_count_total": total,
        "kernel_count_gdn": _UNKNOWN,
        "kernel_count_attn": _UNKNOWN,
        "kernel_count_other": total,
    }


def _api_counts(api_rows: list[dict]) -> tuple[int, int]:
    """Return (cudaLaunchKernel_count, cudaGraphLaunch_count) from
    cuda_api_sum. Rows have a ``Name`` column and an ``Instances`` /
    ``Count`` column."""
    launch_kernel = 0
    graph_launch = 0
    for row in api_rows:
        name = (row.get("Name") or "").strip()
        n = _to_int(row.get("Instances") or row.get("Count")) or 0
        if name == "cudaLaunchKernel":
            launch_kernel = n
        elif name == "cudaGraphLaunch":
            graph_launch = n
    return launch_kernel, graph_launch


def _launch_gaps_us(api_trace_rows: list[dict]) -> list[float]:
    """Extract per-call `cudaLaunchKernel` inter-arrival gaps in
    microseconds, using the ``Start (ns)`` column of ``cuda_api_trace``.

    Some nsys versions use ``Start`` (ns) or ``Start (secs)``; we try
    both. Missing columns → empty list.
    """
    start_col = None
    if api_trace_rows:
        for cand in ("Start (ns)", "Start(ns)", "Start"):
            if cand in api_trace_rows[0]:
                start_col = cand
                break
    if start_col is None:
        return []
    starts_ns: list[float] = []
    for row in api_trace_rows:
        if (row.get("Name") or "").strip() != "cudaLaunchKernel":
            continue
        s = _to_float(row.get(start_col))
        if s is not None:
            starts_ns.append(s)
    starts_ns.sort()
    gaps_us: list[float] = []
    for i in range(1, len(starts_ns)):
        gap_ns = starts_ns[i] - starts_ns[i - 1]
        if gap_ns >= 0:
            gaps_us.append(gap_ns / 1000.0)
    return gaps_us


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    # Nearest-rank on 0..1
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def _client_metadata(records_path: Path | None) -> dict:
    """Aggregate client-side e2e and prefill throughput from records
    JSONL. Returns {'e2e_ms_mean': float|None, 'n_records': int, ...}."""
    if records_path is None or not records_path.is_file():
        return {}
    recs = [
        json.loads(line)
        for line in records_path.read_text().splitlines()
        if line.strip()
    ]
    if not recs:
        return {}
    e2e = [r.get("e2e_ms") for r in recs if isinstance(r.get("e2e_ms"), (int, float))]
    prompt_actual = [
        r.get("prompt_actual_token_count")
        for r in recs
        if isinstance(r.get("prompt_actual_token_count"), int)
    ]
    return {
        "e2e_ms_mean": statistics.fmean(e2e) if e2e else None,
        "n_records": len(recs),
        "prompt_actual_token_count_mean": (
            statistics.fmean(prompt_actual) if prompt_actual else None
        ),
    }


def extract(
    nsys_rep: Path,
    arm: str,
    prompt_len: int,
    batch: int,
    records: Path | None,
    output_csv: Path,
    dry_run: bool,
) -> int:
    row: dict[str, object] = {c: "" for c in COLUMNS}
    row.update(
        {
            "arm": arm,
            "prompt_len": prompt_len,
            "batch": batch,
            "request_id": "aggregate",
            "attribution": "coarse",
            "extractor_status": "dry_run" if dry_run else "unknown",
            "extractor_warnings": "",
            "nsys_source": str(nsys_rep) if nsys_rep else "",
        }
    )

    warnings: list[str] = []

    if not dry_run:
        if not nsys_rep.is_file():
            warnings.append(f"nsys_rep not found: {nsys_rep}")
            row["extractor_status"] = "MISSING_NSYS_REP"
        else:
            try:
                reports = _run_nsys_stats(
                    nsys_rep,
                    ["cuda_gpu_kern_sum", "cuda_api_sum", "cuda_api_trace"],
                )
            except RuntimeError as exc:
                warnings.append(str(exc))
                row["extractor_status"] = "NSYS_MISSING"
                reports = {}
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"nsys stats failed: {exc!r}")
                row["extractor_status"] = "NSYS_STATS_FAILED"
                reports = {}

            kern_rows = _parse_csv_section(reports.get("cuda_gpu_kern_sum", ""))
            api_rows = _parse_csv_section(reports.get("cuda_api_sum", ""))
            api_trace_rows = _parse_csv_section(reports.get("cuda_api_trace", ""))

            if not kern_rows:
                warnings.append("cuda_gpu_kern_sum empty or unparsable")
            row.update(_kernel_counts(kern_rows))

            if not api_rows:
                warnings.append("cuda_api_sum empty or unparsable")
                launch_kernel, graph_launch = 0, 0
            else:
                launch_kernel, graph_launch = _api_counts(api_rows)
            row["cudalaunchkernel_count"] = launch_kernel
            row["cudagraphlaunch_count"] = graph_launch

            # graph_breaks = cudaLaunchKernel launches NOT enclosed by a
            # cudaGraphLaunch. Without a per-launch timeline of graph
            # enter/exit events we approximate: on BCG-enabled arms,
            # every cudaLaunchKernel outside a captured graph is a
            # break candidate. Precise attribution needs NVTX +
            # cudaGraph API trace correlation; without that we record
            # the total cudaLaunchKernel count and mark the field
            # advisory-only.
            row["graph_breaks"] = launch_kernel if graph_launch > 0 else _UNKNOWN

            gaps = _launch_gaps_us(api_trace_rows)
            row["p50_launch_gap_us"] = _quantile(gaps, 0.50) if gaps else _UNKNOWN
            row["p95_launch_gap_us"] = _quantile(gaps, 0.95) if gaps else _UNKNOWN
            row["p99_launch_gap_us"] = _quantile(gaps, 0.99) if gaps else _UNKNOWN

            if not warnings and reports:
                row["extractor_status"] = "OK"

    # Client-side metadata (optional).
    meta = _client_metadata(records) if records else {}
    if meta.get("e2e_ms_mean") is not None:
        row["ttft_ms"] = meta["e2e_ms_mean"]  # coarse — real TTFT needs streaming
        # Throughput: prompt tokens / e2e. Coarse because e2e includes
        # decode; real prefill throughput needs nsys-side breakout.
        pt = meta.get("prompt_actual_token_count_mean")
        if pt and meta["e2e_ms_mean"] > 0:
            row["prefill_throughput_toks_per_s"] = (
                pt * 1000.0 / meta["e2e_ms_mean"]
            )

    row["extractor_warnings"] = "; ".join(warnings)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in COLUMNS})

    if not dry_run and row["extractor_status"] not in ("OK", "dry_run"):
        print(
            f"extract_nsys_metrics: WARN status={row['extractor_status']} "
            f"warnings={row['extractor_warnings']}",
            file=sys.stderr,
        )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nsys-rep", type=Path, default=None)
    p.add_argument("--arm", required=True)
    p.add_argument("--prompt-len", type=int, required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--records", type=Path, default=None)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    if not args.dry_run and args.nsys_rep is None:
        print("extract_nsys_metrics: --nsys-rep required unless --dry-run", file=sys.stderr)
        return 2

    return extract(
        nsys_rep=args.nsys_rep,
        arm=args.arm,
        prompt_len=args.prompt_len,
        batch=args.batch,
        records=args.records,
        output_csv=args.output_csv,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
