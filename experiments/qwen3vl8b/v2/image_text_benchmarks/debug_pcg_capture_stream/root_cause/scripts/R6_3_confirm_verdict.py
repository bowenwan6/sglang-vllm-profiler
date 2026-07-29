#!/usr/bin/env python3
"""R6.3 confirmation verdict.

Consumes per-cell per-variant per-rep bench.jsonl and reports rep-mean
statistics for each cell, alongside per-cell fork/stock mean-TTFT ratio
computed over accepted reps (means-of-means).

Verdict per cell:
  - CONFIRMED_WIN   if fork mean-of-means <= stock mean-of-means AND both
                    variants have >= 2 accepted reps AND fork safety zeros
                    hold AND stock safety zeros hold.
  - NOT_CONFIRMED   if the discovery win did not survive confirmation
                    (fork > stock in confirmation).
  - AMBIGUOUS       if a variant lost > half its reps to foreign PIDs
                    (mid-run invalidation).

Overall verdict:
  - PASS if >= 1 cell reaches CONFIRMED_WIN with all safety zeros.
  - AMBIGUOUS if every cell is AMBIGUOUS.
  - FAIL if every cell is NOT_CONFIRMED with clean data.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path

ASSERT_RE = re.compile(r"AssertionError: PCG capture stream is not set")
FALLBACK_RE = re.compile(r"Falling back to eager execution")
RECOMPILE_RE = re.compile(r"Recompiling function.*qwen3_vl")
READY_RE = re.compile(r"The server is fired up and ready to roll")


def load_lines(p):
    if not p.exists(): return []
    return p.read_text(errors="replace").splitlines()


def load_json(p):
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except Exception: return None


def extract_agg(p):
    if not p.exists(): return None
    lines = p.read_text().strip().splitlines()
    for ln in reversed(lines):
        try: o = json.loads(ln)
        except Exception: continue
        if isinstance(o, dict) and any(k in o for k in ("completed","median_ttft_ms","mean_ttft_ms","num_prompts")):
            return o
    return None


def stats(xs):
    xs = [x for x in xs if isinstance(x,(int,float))]
    n = len(xs)
    if n == 0: return {"n":0}
    mean = sum(xs)/n
    sd = math.sqrt(sum((x-mean)**2 for x in xs)/(n-1)) if n>1 else 0.0
    return {"n":n,"mean":mean,"stdev":sd,"cv_pct":(sd/mean*100.0) if mean else 0.0,"values":xs,
            "median":sorted(xs)[n//2] if n%2 else (sorted(xs)[n//2-1]+sorted(xs)[n//2])/2}


def scan_srv(log_path):
    lines = load_lines(log_path)
    ready = None
    for i, ln in enumerate(lines, 1):
        if READY_RE.search(ln): ready = i; break
    recomp = [i+1 for i, ln in enumerate(lines) if RECOMPILE_RE.search(ln)]
    return {
        "total_lines": len(lines), "ready_line": ready,
        "assertions": sum(1 for ln in lines if ASSERT_RE.search(ln)),
        "fallbacks":  sum(1 for ln in lines if FALLBACK_RE.search(ln)),
        "startup_recompiles": len([r for r in recomp if ready is None or r < ready]),
        "post_ready_recompiles": len([r for r in recomp if ready is not None and r >= ready]),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()
    ind = args.in_dir

    launch = load_json(ind / "launch_context.json")
    cells = (launch or {}).get("cells", [])
    reps_per_variant = (launch or {}).get("reps_per_cell_per_variant", 3)

    # Discover cells if not in launch context
    if not cells:
        cells = sorted([d.name for d in ind.iterdir() if d.is_dir() and d.name.startswith("cell_")])

    # Safety per variant server (shared across cells)
    servers = {}
    for v in ("stock_default", "fork_pcg"):
        srv_dir = ind / f"_server_{v}"
        servers[v] = {"safety": scan_srv(srv_dir / "server.log")}

    per_cell = {}
    for cell in cells:
        cell_dir = ind / cell
        cell_rec = {"variants": {}}
        for v in ("stock_default", "fork_pcg"):
            vdir = cell_dir / v
            reps = []
            for r in range(1, reps_per_variant + 1):
                inv = (vdir / f"rep{r}.INVALIDATED").exists()
                m = extract_agg(vdir / f"rep{r}.jsonl")
                if inv:
                    reps.append({"rep": r, "missing": True, "invalidated": True, "reason": "foreign_pid"})
                elif m is None:
                    reps.append({"rep": r, "missing": True, "invalidated": False})
                else:
                    reps.append({"rep": r, "missing": False, "invalidated": False,
                                 "completed": m.get("completed"),
                                 "mean_ttft_ms": m.get("mean_ttft_ms"),
                                 "median_ttft_ms": m.get("median_ttft_ms"),
                                 "mean_tpot_ms": m.get("mean_tpot_ms"),
                                 "output_throughput": m.get("output_throughput")})
            ttfts = [r.get("mean_ttft_ms") for r in reps if not r["missing"]]
            n_accepted = len(ttfts)
            n_invalidated = sum(1 for r in reps if r.get("invalidated"))
            cell_rec["variants"][v] = {
                "reps": reps, "n_accepted": n_accepted, "n_invalidated": n_invalidated,
                "mean_ttft_stats": stats(ttfts),
            }
        s = cell_rec["variants"]["stock_default"]["mean_ttft_stats"].get("mean")
        f = cell_rec["variants"]["fork_pcg"]["mean_ttft_stats"].get("mean")
        ns = cell_rec["variants"]["stock_default"]["n_accepted"]
        nf = cell_rec["variants"]["fork_pcg"]["n_accepted"]
        cell_rec["fork_over_stock_ratio"] = (f/s) if (isinstance(s,(int,float)) and isinstance(f,(int,float)) and s) else None
        # verdict per cell
        ambiguous = (ns < 2 or nf < 2)
        if ambiguous:
            cell_rec["verdict"] = "AMBIGUOUS"
        elif cell_rec["fork_over_stock_ratio"] is not None and cell_rec["fork_over_stock_ratio"] <= 1.0:
            cell_rec["verdict"] = "CONFIRMED_WIN"
        else:
            cell_rec["verdict"] = "NOT_CONFIRMED"
        per_cell[cell] = cell_rec

    confirmed = [c for c, d in per_cell.items() if d["verdict"] == "CONFIRMED_WIN"]
    not_confirmed = [c for c, d in per_cell.items() if d["verdict"] == "NOT_CONFIRMED"]
    ambiguous_cells = [c for c, d in per_cell.items() if d["verdict"] == "AMBIGUOUS"]

    safe_all = all(
        servers[v]["safety"]["assertions"] == 0
        and servers[v]["safety"]["fallbacks"] == 0
        and servers[v]["safety"]["post_ready_recompiles"] == 0
        for v in servers
    )

    reasons = []
    if not safe_all: reasons.append(f"server safety non-zero: {servers}")
    if not confirmed:
        if ambiguous_cells and not not_confirmed:
            reasons.append("all cells AMBIGUOUS (insufficient clean reps)")
        else:
            reasons.append("no cell confirmed (fork ratio > 1 or unclean)")

    verdict = "PASS" if (confirmed and safe_all) else (
        "AMBIGUOUS" if ambiguous_cells and not not_confirmed else "FAIL"
    )

    def _f(x, d=3):
        return f"{x:.{d}f}" if isinstance(x,(int,float)) else "—"

    L = [f"# R6.3 confirmation verdict — **{verdict}**", ""]
    if launch:
        L.append("## Launch context")
        for k in ("selected_gpu_id","attempt_dir","cells","reps_per_cell_per_variant","host_libcuda","ld_preload","nvidia_driver","sglang_stock_head","sglang_fork_head"):
            if k in launch: L.append(f"- `{k}`: `{launch[k]}`")
        L.append("")

    L.append("## Per-cell confirmation")
    L.append("")
    L.append("| cell | stock reps_ok/inv | fork reps_ok/inv | stock mean_ttft_ms | fork mean_ttft_ms | fork/stock | verdict |")
    L.append("|---|---|---|---|---|---|---|")
    for cell in cells:
        d = per_cell.get(cell, {})
        sv = d.get("variants", {}).get("stock_default", {})
        fv = d.get("variants", {}).get("fork_pcg", {})
        L.append(f"| `{cell}` | {sv.get('n_accepted',0)}/{sv.get('n_invalidated',0)} | {fv.get('n_accepted',0)}/{fv.get('n_invalidated',0)} | "
                 f"{_f(sv.get('mean_ttft_stats',{}).get('mean'))} | {_f(fv.get('mean_ttft_stats',{}).get('mean'))} | "
                 f"{_f(d.get('fork_over_stock_ratio'),4)} | {d.get('verdict','?')} |")
    L.append("")

    L.append("## Per-rep detail (mean TTFT ms)")
    L.append("")
    for cell in cells:
        d = per_cell.get(cell, {})
        L.append(f"### `{cell}` — verdict **{d.get('verdict','?')}**")
        for v in ("stock_default", "fork_pcg"):
            vv = d.get("variants", {}).get(v, {})
            reps_desc = ", ".join(
                f"rep{r['rep']}={_f(r.get('mean_ttft_ms'))}" if not r["missing"]
                else (f"rep{r['rep']}=INVALIDATED({r.get('reason','?')})" if r.get("invalidated") else f"rep{r['rep']}=missing")
                for r in vv.get("reps", [])
            )
            s = vv.get("mean_ttft_stats", {})
            L.append(f"- **{v}**: {reps_desc} | mean={_f(s.get('mean'))} CV%={_f(s.get('cv_pct'),2)}")
        L.append("")

    L.append("## Server safety (shared across cells per variant)")
    for v, srv in servers.items():
        sf = srv["safety"]
        L.append(f"- **{v}**: assertions={sf['assertions']}, fallbacks={sf['fallbacks']}, post_ready_recompiles={sf['post_ready_recompiles']}, ready_line={sf['ready_line']}, total_lines={sf['total_lines']}")
    L.append("")

    L.append(f"**CONFIRMED_WIN cells** ({len(confirmed)}): {confirmed}")
    L.append(f"**NOT_CONFIRMED cells** ({len(not_confirmed)}): {not_confirmed}")
    L.append(f"**AMBIGUOUS cells** ({len(ambiguous_cells)}): {ambiguous_cells}")
    L.append("")

    L.append(f"## Overall verdict: **{verdict}**")
    for r in reasons: L.append(f"- {r}")

    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({
        "verdict": verdict, "reasons": reasons,
        "per_cell": per_cell,
        "confirmed_cells": confirmed,
        "not_confirmed_cells": not_confirmed,
        "ambiguous_cells": ambiguous_cells,
        "server_safety": {k: v["safety"] for k, v in servers.items()},
        "launch_context": launch,
    }, indent=2, sort_keys=True, default=str))
    print(f"VERDICT={verdict}")
    for r in reasons: print(f"REASON: {r}")
    return 0 if verdict == "PASS" else (2 if verdict == "AMBIGUOUS" else 1)


if __name__ == "__main__":
    sys.exit(main())
