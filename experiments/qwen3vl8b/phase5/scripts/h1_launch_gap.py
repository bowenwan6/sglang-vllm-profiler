#!/usr/bin/env python3
"""
Phase 5.1 (H1) — offline launch/graph-coverage metric extractor (READ-ONLY).

Reads EXISTING graph-on formal SGLang traces + vLLM prefill/decode traces and computes
launch / graph-coverage / idle-gap metrics to test H1 *observationally*. It does NOT launch
servers, run benchmarks, re-collect, or modify any trace/JSON. It reuses the canonical
profiler parsing helpers from the llm-torch-profiler-analysis skill (loader, event extractor,
kernel/launch classification predicates) — it does not reimplement a trace parser.

Methodology guards:
  - SGLang performance/coverage metrics come ONLY from the graph-on FORMAL traces
    (sglang_formal / sglang_extend_formal), never from graph-off mapping traces.
  - A graph-off mapping trace is a kernel->source tool and cannot prove serving-path eagerness.
  - Any metric the trace cannot support cleanly is reported as "unavailable"/"ambiguous",
    never estimated or fabricated.

Output: markdown + JSON under analysis/qwen3vl8b/phase5/h1_launch_gap/.
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path

LAB = Path("/data/sglang-vllm-profiler")
SKILL = Path("/root/.claude/skills/llm-torch-profiler-analysis/scripts")
sys.path.insert(0, str(SKILL))

from profile_common import (  # canonical helpers (reused, not reimplemented)
    load_trace_json, extract_trace_events, is_complete_duration_event,
)
from triage_kernel_helpers import is_cuda_launch_event, is_gpu_kernel_event

TRACES = LAB / "traces/qwen3vl8b"
OUT = LAB / "analysis/qwen3vl8b/phase5/h1_launch_gap"

GEMM_RE = re.compile(r"nvjet|cutlass.*(gemm|mm)|cublas|splitkreduce|\bgemm\b|matmul", re.I)


def pick_trace(d: Path):
    """Pick the representative *.gz in a dir; for vLLM prefer the per-rank rank0 file."""
    if not d.exists():
        return None
    gz = sorted(d.rglob("*.gz"))
    if not gz:
        return None
    rank0 = [p for p in gz if "rank0" in p.name]
    if rank0:
        return rank0[0]
    # SGLang: the TP-0 trace
    tp0 = [p for p in gz if "TP-0" in p.name]
    return tp0[0] if tp0 else gz[0]


def coerce_int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def analyze(path: Path) -> dict:
    trace = load_trace_json(path)
    events = extract_trace_events(trace)

    # correlation -> launch-op name (cudaGraphLaunch / cudaLaunchKernel(ExC) / ...)
    launch_name_by_corr: dict[int, str] = {}
    launch_cpu_us = 0.0
    launch_count = 0
    for e in events:
        name = str(e.get("name", ""))
        cat = str(e.get("cat", ""))
        if is_cuda_launch_event(name, cat):
            args = e.get("args") or {}
            corr = coerce_int(args.get("correlation"))
            if corr is not None:
                launch_name_by_corr.setdefault(corr, name)
            launch_cpu_us += float(e.get("dur", 0) or 0)
            launch_count += 1

    # GPU kernels
    kernels = []  # (ts, dur, name, launch_name, pidtid)
    for e in events:
        if not is_gpu_kernel_event(e):
            continue
        args = e.get("args") or {}
        corr = coerce_int(args.get("correlation"))
        lname = launch_name_by_corr.get(corr) if corr is not None else None
        kernels.append((
            float(e.get("ts", 0) or 0), float(e.get("dur", 0) or 0),
            str(e.get("name", "")), lname, (e.get("pid"), e.get("tid")),
        ))

    n = len(kernels)
    total_dur = sum(k[1] for k in kernels)

    def cls(lname):
        if lname is None:
            return "unclassified"
        l = lname.lower()
        if "graphlaunch" in l:
            return "graph"
        if "launchkernel" in l or "launch" in l:
            return "eager"
        return "unclassified"

    share = {"graph": 0.0, "eager": 0.0, "unclassified": 0.0}
    cnt = {"graph": 0, "eager": 0, "unclassified": 0}
    gemm_dur = 0.0
    for ts, dur, kn, lname, _ in kernels:
        c = cls(lname)
        share[c] += dur
        cnt[c] += 1
        if GEMM_RE.search(kn):
            gemm_dur += dur

    # dominant GPU stream: the (pid,tid) with the most kernel GPU time
    by_stream: dict = {}
    for ts, dur, kn, lname, pt in kernels:
        by_stream.setdefault(pt, []).append((ts, dur))
    dom = max(by_stream, key=lambda k: sum(d for _, d in by_stream[k])) if by_stream else None
    idle_us = busy_us = span_us = None
    if dom:
        iv = sorted(by_stream[dom])
        span_us = (iv[-1][0] + iv[-1][1]) - iv[0][0]
        # union of intervals = busy; idle = span - busy
        busy = 0.0
        cur_s, cur_e = iv[0][0], iv[0][0] + iv[0][1]
        for ts, dur in iv[1:]:
            if ts > cur_e:
                busy += cur_e - cur_s
                cur_s, cur_e = ts, ts + dur
            else:
                cur_e = max(cur_e, ts + dur)
        busy += cur_e - cur_s
        busy_us = busy
        idle_us = max(0.0, span_us - busy)

    def pct(x):
        return round(100.0 * x / total_dur, 1) if total_dur > 0 else None

    return {
        "trace": str(path.relative_to(LAB)),
        "kernel_count": n,
        "kernel_launch_count": launch_count,
        "total_kernel_gpu_us": round(total_dur, 1),
        "graph_covered_share_pct": pct(share["graph"]),
        "eager_dispatch_share_pct": pct(share["eager"]),
        "unclassified_share_pct": pct(share["unclassified"]),
        "graph_kernel_count": cnt["graph"],
        "eager_kernel_count": cnt["eager"],
        "unclassified_kernel_count": cnt["unclassified"],
        "gemm_share_pct": pct(gemm_dur),
        "dominant_stream_span_us": round(span_us, 1) if span_us is not None else None,
        "dominant_stream_busy_us": round(busy_us, 1) if busy_us is not None else None,
        "dominant_stream_idle_gap_us": round(idle_us, 1) if idle_us is not None else None,
        "dominant_stream_idle_pct": (round(100.0 * idle_us / span_us, 1)
                                     if (span_us and span_us > 0) else None),
        "launch_op_total_cpu_us": round(launch_cpu_us, 1),
        # explicit limits:
        "_notes": [
            "graph/eager classified by the kernel's correlated CPU launch op "
            "(cudaGraphLaunch=graph, cudaLaunchKernel*=eager, no correlation=unclassified).",
            "idle gap is on the dominant GPU stream (union-of-intervals); cross-stream overlap not modeled.",
            "per-forward-step critical-path segmentation NOT computed (num_steps>1 windows are not "
            "cleanly separable from the trace alone) -> reported as window-level span/busy/idle, "
            "and per-step critical path is AMBIGUOUS/unavailable.",
            "launch_op_total_cpu_us is the summed CPU duration of launch runtime ops (a proxy); "
            "true critical-path CPU launch-gap across threads is AMBIGUOUS and not asserted.",
        ],
    }


INPUTS = {
    "caseA_short": {
        "sglang_formal_DECODE": "caseA_short/sglang_formal",
        "sglang_formal_EXTEND": "caseA_short/sglang_extend_formal",
        "vllm_prefill_like": "caseA_short/vllm/prefill_like",
        "vllm_decode_like": "caseA_short/vllm/decode_like",
    },
    "caseC_batched": {
        "sglang_formal_DECODE": "caseC_batched/sglang_formal",
        "sglang_formal_EXTEND": "caseC_batched/sglang_extend_formal",
        "vllm_prefill_like": "caseC_batched/vllm/prefill_like",
        "vllm_decode_like": "caseC_batched/vllm/decode_like",
    },
}

COLS = [
    ("kernel_count", "kernels"),
    ("kernel_launch_count", "launch ops"),
    ("total_kernel_gpu_us", "GPU µs"),
    ("graph_covered_share_pct", "graph %"),
    ("eager_dispatch_share_pct", "eager %"),
    ("unclassified_share_pct", "uncl %"),
    ("gemm_share_pct", "GEMM %"),
    ("dominant_stream_idle_pct", "GPU idle %"),
    ("dominant_stream_idle_gap_us", "idle µs"),
    ("launch_op_total_cpu_us", "launch CPU µs"),
]


def md_table(rows: dict) -> str:
    head = "| window | " + " | ".join(c[1] for c in COLS) + " |\n"
    head += "|" + "---|" * (len(COLS) + 1) + "\n"
    body = ""
    for label, m in rows.items():
        if m is None:
            body += f"| {label} | _input missing_ |" + " |" * (len(COLS) - 1) + "\n"
            continue
        cells = []
        for key, _ in COLS:
            v = m.get(key)
            cells.append("n/a" if v is None else str(v))
        body += f"| {label} | " + " | ".join(cells) + " |\n"
    return head + body


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for case, inputs in INPUTS.items():
        rows = {}
        for label, rel in inputs.items():
            tp = pick_trace(TRACES / rel)
            if tp is None:
                print(f"[WARN] missing trace: {rel}")
                rows[label] = None
                continue
            print(f"[{case}] analyzing {label}: {tp.name}")
            rows[label] = analyze(tp)
        all_results[case] = rows
        # per-case markdown
        md = [f"# Phase 5.1 — H1 launch/graph-coverage metrics · {case}", "",
              "Read-only offline analysis of existing graph-on formal (SGLang) + vLLM traces. "
              "SGLang rows are from FORMAL traces only (not graph-off mapping).", "",
              md_table(rows), "",
              "## Notes / limits", ""]
        notes = next((m["_notes"] for m in rows.values() if m), [])
        for nt in notes:
            md.append(f"- {nt}")
        (OUT / f"{case}.md").write_text("\n".join(md) + "\n")
    (OUT / "metrics.json").write_text(json.dumps(all_results, indent=2))
    print(f"wrote {OUT}/caseA_short.md, caseC_batched.md, metrics.json")


if __name__ == "__main__":
    main()
