#!/usr/bin/env python3
"""Split a torch-profiler trace (Chrome JSON, optionally .gz) into the Q3 segments.

The instrumentation patch names three windows with `record_function`:
  Q3_VIT_FORWARD   the vision encoder call (patch-embed .. merger, one per prefill)
  Q3_STEP_EXTEND   one prefill forward step (contains Q3_VIT_FORWARD)
  Q3_STEP_DECODE   one decode forward step

For every window this reports
  cpu_wall_ms          CPU span of the annotation
  gpu_busy_ms          union of GPU activity (kernels, memcpy, memset) launched inside
                       the window, matched through CUPTI correlation ids
  gpu_busy_window_ms   the same matched by time window (fallback / cross-check)
  gpu_span_ms          first GPU start -> last GPU end of the matched activity
  n_kernels, n_launches (cudaLaunchKernel*), n_graph_launches (cudaGraphLaunch)

The eager arm's (cpu_wall - gpu_busy) is the un-overlapped launch time the graph
can recover: that is the H1 predictor.

Usage: parse_trace.py <trace.json[.gz]> [--json out.json]
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

RUNTIME_CATS = {"cuda_runtime", "cuda_driver"}
GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
ANNOT_CATS = {"user_annotation", "cpu_op"}
NAMES = ("Q3_VIT_FORWARD", "Q3_STEP_EXTEND", "Q3_STEP_DECODE")
GPU_LAG_US = 2000.0  # slack for the time-window fallback


def load_events(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data["traceEvents"] if isinstance(data, dict) else data


def union_ms(intervals: List[tuple]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    total, cs, ce = 0.0, intervals[0][0], intervals[0][1]
    for s, e in intervals[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    total += ce - cs
    return total / 1000.0


def analyze(events: List[Dict[str, Any]], names=NAMES) -> Dict[str, List[Dict[str, Any]]]:
    xs = [e for e in events if e.get("ph") == "X" and "ts" in e and "dur" in e]
    runtime = sorted((e for e in xs if e.get("cat") in RUNTIME_CATS), key=lambda e: e["ts"])
    gpu = sorted((e for e in xs if e.get("cat") in GPU_CATS), key=lambda e: e["ts"])
    rt_ts = [e["ts"] for e in runtime]
    gpu_ts = [e["ts"] for e in gpu]
    by_corr: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for g in gpu:
        c = (g.get("args") or {}).get("correlation")
        if c is not None:
            by_corr[c].append(g)

    out: Dict[str, List[Dict[str, Any]]] = {}
    for name in names:
        rows = []
        for w in (e for e in xs if e.get("cat") in ANNOT_CATS and e.get("name") == name):
            t0, t1 = float(w["ts"]), float(w["ts"]) + float(w["dur"])
            lo, hi = bisect.bisect_left(rt_ts, t0), bisect.bisect_right(rt_ts, t1)
            rt = runtime[lo:hi]
            matched: List[Dict[str, Any]] = []
            for r in rt:
                c = (r.get("args") or {}).get("correlation")
                if c is not None:
                    matched.extend(by_corr.get(c, []))
            glo, ghi = bisect.bisect_left(gpu_ts, t0), bisect.bisect_right(gpu_ts, t1 + GPU_LAG_US)
            win = gpu[glo:ghi]
            iv = [(float(g["ts"]), float(g["ts"]) + float(g["dur"])) for g in matched]
            rows.append({
                "ts_us": t0,
                "cpu_wall_ms": round((t1 - t0) / 1000.0, 3),
                "gpu_busy_ms": round(union_ms(iv), 3),
                "gpu_busy_window_ms": round(union_ms(
                    [(float(g["ts"]), float(g["ts"]) + float(g["dur"])) for g in win]), 3),
                "gpu_span_ms": round((max(e for _, e in iv) - min(s for s, _ in iv)) / 1000.0, 3) if iv else 0.0,
                "n_kernels": sum(1 for g in matched if g.get("cat") == "kernel"),
                "n_launches": sum(1 for r in rt if str(r.get("name", "")).startswith("cudaLaunchKernel")),
                "n_graph_launches": sum(1 for r in rt if str(r.get("name", "")).startswith("cudaGraphLaunch")),
                "n_runtime_calls": len(rt),
            })
        rows.sort(key=lambda r: r["ts_us"])
        out[name] = rows
    return out


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"n": 0}
    keys = ("cpu_wall_ms", "gpu_busy_ms", "gpu_busy_window_ms", "gpu_span_ms",
            "n_kernels", "n_launches", "n_graph_launches")
    s: Dict[str, Any] = {"n": len(rows)}
    for k in keys:
        s[k + "_p50"] = round(statistics.median(r[k] for r in rows), 3)
    s["unoverlapped_ms_p50"] = round(statistics.median(r["cpu_wall_ms"] - r["gpu_busy_ms"] for r in rows), 3)
    return s


def analyze_file(path: Path) -> Dict[str, Any]:
    events = load_events(path)
    per = analyze(events)
    return {"trace": str(path), "n_events": len(events),
            "summary": {k: summarize(v) for k, v in per.items()},
            "windows": per}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    res = analyze_file(Path(a.trace))
    print(json.dumps(res["summary"], indent=2))
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(res, indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
