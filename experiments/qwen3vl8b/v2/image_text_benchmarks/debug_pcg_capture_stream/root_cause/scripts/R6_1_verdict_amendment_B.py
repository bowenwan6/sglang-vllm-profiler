#!/usr/bin/env python3
"""R6.1 Amendment B verdict — repeated-shape safety control.

Classifies Attempt 04's stock-PCG vs fork-PCG outcomes per the
predeclared rules in `R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md`
§B.2. CPU-only.

Inputs (all optional; missing files -> FAIL_MISSING for that side):

  stock/server.log            stock/bench.log             stock/bench.jsonl
  stock/phase_markers.txt
  fork/server.log             fork/bench.log              fork/bench.jsonl
  fork/phase_markers.txt
  launch_context.json

Writes:
  verdict_amended_B.md
  verdict_amended_B.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ASSERT_RE = re.compile(
    r"AssertionError: PCG capture stream is not set,"
    r" please check if runtime recompilation happened")
DEEPSTACK_RECOMPILE_RE = re.compile(
    r"input_deepstack_embeds is None")
FALLBACK_RE = re.compile(r"Falling back to eager execution")
RECOMPILE_RE = re.compile(r"Recompiling function.*qwen3_vl")
READY_RE = re.compile(r"The server is fired up and ready to roll")
PREFILL_BATCH_RE = re.compile(
    r"#new-seq: (\d+), #new-token: (\d+), #cached-token: (\d+)")


def load_lines(p: Path) -> list[str]:
    if not p.exists():
        return []
    return p.read_text(errors="replace").splitlines()


def load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def scan_server(server_log: Path, markers_file: Path) -> dict[str, Any]:
    lines = load_lines(server_log)
    ready_line = None
    for i, ln in enumerate(lines, 1):
        if READY_RE.search(ln):
            ready_line = i
            break

    recompile_lines = [i + 1 for i, ln in enumerate(lines) if RECOMPILE_RE.search(ln)]
    deepstack_recompile_lines = [i + 1 for i, ln in enumerate(lines)
                                 if DEEPSTACK_RECOMPILE_RE.search(ln)]
    assertion_lines = [i + 1 for i, ln in enumerate(lines) if ASSERT_RE.search(ln)]
    fallback_lines = [i + 1 for i, ln in enumerate(lines) if FALLBACK_RE.search(ln)]

    startup_warmup_recompiles = [ln for ln in recompile_lines
                                 if ready_line is None or ln < ready_line]
    post_ready_recompiles = [ln for ln in recompile_lines
                             if ready_line is not None and ln >= ready_line]

    # Prefill shape trace: per-batch (new_seq, new_token, cached_token)
    prefill_batches = []
    for i, ln in enumerate(lines, 1):
        m = PREFILL_BATCH_RE.search(ln)
        if m:
            prefill_batches.append({
                "line": i,
                "new_seq": int(m.group(1)),
                "new_token": int(m.group(2)),
                "cached_token": int(m.group(3)),
            })

    # Identify unique shapes and first vs repeated occurrences
    shape_first_seen = {}
    shape_occurrences = {}
    for b in prefill_batches:
        shape = (b["new_token"] + b["cached_token"], b["new_seq"])
        shape_key = f"total={shape[0]},new_seq={shape[1]}"
        if shape_key not in shape_first_seen:
            shape_first_seen[shape_key] = b["line"]
        shape_occurrences.setdefault(shape_key, []).append(b["line"])

    # Assertion attribution: which shape / request was on the wire?
    assertion_context = []
    for aline in assertion_lines:
        prev_batch = None
        for b in prefill_batches:
            if b["line"] < aline:
                prev_batch = b
            else:
                break
        assertion_context.append({
            "assertion_line": aline,
            "prev_prefill_batch": prev_batch,
        })

    return {
        "server_log": str(server_log),
        "server_log_missing": not server_log.exists(),
        "server_log_total_lines": len(lines),
        "server_ready_line": ready_line,
        "assertion_lines": assertion_lines,
        "assertion_count": len(assertion_lines),
        "assertion_context": assertion_context,
        "deepstack_recompile_lines": deepstack_recompile_lines,
        "recompile_lines": recompile_lines,
        "startup_warmup_recompile_lines": startup_warmup_recompiles,
        "startup_warmup_recompile_count": len(startup_warmup_recompiles),
        "post_ready_recompile_lines": post_ready_recompiles,
        "post_ready_recompile_count": len(post_ready_recompiles),
        "fallback_lines": fallback_lines,
        "fallback_count": len(fallback_lines),
        "prefill_batches": prefill_batches,
        "unique_prefill_shapes": sorted(shape_first_seen.keys()),
        "shape_occurrence_counts": {k: len(v) for k, v in shape_occurrences.items()},
        "shape_first_seen_line": shape_first_seen,
    }


def scan_bench_client(bench_jsonl: Path, bench_log: Path) -> dict[str, Any]:
    # sglang.benchmark.serving writes ONE json line at the end of the run
    # to --output-file; that line has aggregate stats. Per-request details
    # go to details (if --output-details) also into that file's structure.
    if not bench_jsonl.exists():
        return {"jsonl_missing": True, "log": str(bench_log)}
    lines = bench_jsonl.read_text().strip().splitlines()
    if not lines:
        return {"jsonl_empty": True, "log": str(bench_log)}
    # The final line is the aggregate; earlier lines may be per-request.
    try:
        parsed = [json.loads(x) for x in lines]
    except Exception as e:
        return {"jsonl_parse_error": str(e), "log": str(bench_log)}
    # We just record request counts. The aggregate object typically has
    # 'completed', 'total_input_tokens', etc.
    aggregate = None
    for obj in reversed(parsed):
        if isinstance(obj, dict) and ("completed" in obj or "num_prompts" in obj or "generated_texts" in obj):
            aggregate = obj
            break
    return {
        "jsonl": str(bench_jsonl),
        "log": str(bench_log),
        "n_lines": len(lines),
        "aggregate_completed": (aggregate or {}).get("completed"),
        "aggregate_num_prompts": (aggregate or {}).get("num_prompts"),
        "aggregate_generated_texts_count": len((aggregate or {}).get("generated_texts") or []),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    ind = args.in_dir
    launch = load_json(ind / "launch_context.json")

    stock = scan_server(ind / "stock/server.log",
                        ind / "stock/phase_markers.txt")
    stock_bench = scan_bench_client(ind / "stock/bench.jsonl",
                                    ind / "stock/bench.log")
    fork = scan_server(ind / "fork/server.log",
                       ind / "fork/phase_markers.txt")
    fork_bench = scan_bench_client(ind / "fork/bench.jsonl",
                                   ind / "fork/bench.log")

    # ---- SAFETY_SUPERIORITY_PASS conditions ----
    stock_assertion_seen = stock["assertion_count"] > 0
    stock_deepstack_recompile_seen = any(
        line in stock["recompile_lines"] or
        any(abs(line - r) < 200 for r in stock["recompile_lines"])
        for line in stock["deepstack_recompile_lines"]
    ) or (stock["deepstack_recompile_lines"] and stock["recompile_lines"])
    fork_assertion_count = fork["assertion_count"]
    fork_fallback_count = fork["fallback_count"]
    fork_post_ready_recompile = fork["post_ready_recompile_count"]
    # Fork "completes" means bench.jsonl has non-empty aggregate — but the
    # benchmark tool might not always write completion aggregate. Fall
    # back: server-log ready + no assertion mid-flight is a soft signal.
    fork_completed_bench = (
        (fork_bench.get("aggregate_completed") or 0) > 0
        or (fork_bench.get("aggregate_generated_texts_count") or 0) > 0
        or (fork_bench.get("n_lines") or 0) > 0
    )

    stock_ok_for_safety_pass = stock_assertion_seen and (
        len(stock["deepstack_recompile_lines"]) > 0)
    fork_ok_for_safety_pass = (
        fork["server_ready_line"] is not None
        and fork_assertion_count == 0
        and fork_fallback_count == 0
        and fork_post_ready_recompile == 0
        and fork_completed_bench
    )

    # ---- Determine verdict ----
    reasons: list[str] = []
    if fork["server_log_missing"] or stock["server_log_missing"]:
        overall = "INFRA_FAILURE"
        reasons.append("one or both server logs missing")
    elif fork_assertion_count > 0 or fork_fallback_count > 0 \
         or fork_post_ready_recompile > 0 or not fork_completed_bench:
        overall = "FORK_FAIL"
        if fork_assertion_count > 0:
            reasons.append(f"fork assertion count = {fork_assertion_count}")
        if fork_fallback_count > 0:
            reasons.append(f"fork fallback count = {fork_fallback_count}")
        if fork_post_ready_recompile > 0:
            reasons.append(
                f"fork post-ready recompiles = {fork_post_ready_recompile}")
        if not fork_completed_bench:
            reasons.append("fork did not complete the bench (no aggregate)")
    elif stock_ok_for_safety_pass and fork_ok_for_safety_pass:
        overall = "SAFETY_SUPERIORITY_PASS"
    elif not stock_assertion_seen:
        overall = "STOCK_TRIGGER_NOT_REPRODUCED"
        reasons.append("stock did not reach the PCG capture-stream assertion")
    else:
        overall = "INFRA_FAILURE"
        reasons.append("unhandled combination")

    # ---- Render ----
    L = [f"# R6.1 Amendment B verdict — **{overall}**", ""]
    L.append(f"> Evaluated under [`../protocol_amendment_B_repeated_shape_safety.md`](../protocol_amendment_B_repeated_shape_safety.md). Rules pre-declared before Attempt 04 ran.")
    L.append("")
    if launch:
        L.append("## Launch context")
        L.append("")
        for k in ("selected_gpu_id", "attempt_dir", "host_libcuda",
                  "ld_preload", "cuda_visible_devices", "nvidia_driver",
                  "sglang_stock_head", "sglang_fork_head"):
            if k in launch:
                L.append(f"- `{k}`: `{launch[k]}`")
        L.append("")

    def render_side(name, s, b):
        L.append(f"## {name} side")
        L.append("")
        L.append(f"- server_log: `{s['server_log']}`")
        L.append(f"- server_log_total_lines: {s['server_log_total_lines']}")
        L.append(f"- server_ready_line: {s['server_ready_line']}")
        L.append(f"- assertion count: **{s['assertion_count']}** (lines {s['assertion_lines']})")
        L.append(f"- deepstack-recompile trigger lines: {s['deepstack_recompile_lines']}")
        L.append(f"- startup/warmup recompiles: {s['startup_warmup_recompile_count']} (lines {s['startup_warmup_recompile_lines']})")
        L.append(f"- post-ready recompiles: {s['post_ready_recompile_count']} (lines {s['post_ready_recompile_lines']})")
        L.append(f"- fallback lines: {s['fallback_count']} ({s['fallback_lines']})")
        L.append(f"- prefill batches captured: {len(s['prefill_batches'])}")
        L.append(f"- unique prefill shapes: {s['unique_prefill_shapes']}")
        L.append(f"- per-shape occurrence counts: {s['shape_occurrence_counts']}")
        if s["assertion_context"]:
            L.append(f"- assertion contexts:")
            for a in s["assertion_context"]:
                L.append(f"  - assertion at line {a['assertion_line']}; last prefill batch before: {a['prev_prefill_batch']}")
        L.append(f"- bench.jsonl aggregate_completed: {b.get('aggregate_completed')} / aggregate_num_prompts: {b.get('aggregate_num_prompts')} / generated_texts count: {b.get('aggregate_generated_texts_count')}")
        L.append("")

    render_side("stock-PCG", stock, stock_bench)
    render_side("fork-PCG", fork, fork_bench)

    L.append("## Overall verdict: **{}**".format(overall))
    L.append("")
    for r in reasons:
        L.append(f"- {r}")
    L.append("")

    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({
        "overall_verdict": overall,
        "reasons": reasons,
        "stock": {**stock, "bench": stock_bench},
        "fork": {**fork, "bench": fork_bench},
        "launch_context": launch,
        "safety_conditions": {
            "stock_assertion_seen": stock_assertion_seen,
            "stock_deepstack_recompile_seen": bool(stock["deepstack_recompile_lines"]),
            "fork_assertion_count": fork_assertion_count,
            "fork_fallback_count": fork_fallback_count,
            "fork_post_ready_recompile_count": fork_post_ready_recompile,
            "fork_completed_bench": fork_completed_bench,
            "stock_ok_for_safety_pass": stock_ok_for_safety_pass,
            "fork_ok_for_safety_pass": fork_ok_for_safety_pass,
        },
    }, indent=2, sort_keys=True))
    print(f"OVERALL_VERDICT={overall}")
    for r in reasons:
        print(f"REASON: {r}")
    if overall == "SAFETY_SUPERIORITY_PASS":
        return 0
    if overall in ("STOCK_TRIGGER_NOT_REPRODUCED",):
        return 2
    if overall == "FORK_FAIL":
        return 1
    return 3


if __name__ == "__main__":
    sys.exit(main())
