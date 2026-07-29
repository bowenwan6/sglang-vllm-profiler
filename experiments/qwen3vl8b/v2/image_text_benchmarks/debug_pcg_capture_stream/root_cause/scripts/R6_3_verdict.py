#!/usr/bin/env python3
"""R6.3 verdict — image cost + workload sweep + mixed safety.

Consumes:
  raw/a_rebaseline/{stock_default,fork_pcg}/rep{1..3}.jsonl
  raw/b_sweep/cell_<txt>x<res>xc<conc>/{stock_default,fork_pcg}/bench.jsonl
  raw/c_mixed_safety/fork_pcg_interleaved.jsonl
  raw/c_mixed_safety/server.log

Emits:
  * R6.3a rebaseline stats + delta stock-vs-fork
  * R6.3b sweep matrix; every cell reported; winning cells flagged
    (fork_pcg mean TTFT <= stock_default mean TTFT)
  * R6.3c mixed-safety pass/fail (0 assertions/fallbacks/req_fail/post-ready recompile)
  * Overall gate: PASS iff R6.3c pass AND at least one winning cell in R6.3b
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
    except: return None


def extract_agg(p):
    if not p.exists(): return None
    lines = p.read_text().strip().splitlines()
    for ln in reversed(lines):
        try: o = json.loads(ln)
        except: continue
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


def scan_srv(log):
    lines = load_lines(log)
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

    # ---- R6.3a rebaseline (3 reps each variant) ----
    rebaseline = {}
    for v in ("stock_default", "fork_pcg"):
        vdir = ind / "a_rebaseline" / v
        reps = []
        for i in range(1, 4):
            m = extract_agg(vdir / f"rep{i}.jsonl")
            if m is None:
                reps.append({"rep": i, "missing": True})
            else:
                reps.append({"rep": i, "missing": False,
                             "completed": m.get("completed"),
                             "mean_ttft_ms": m.get("mean_ttft_ms"),
                             "median_ttft_ms": m.get("median_ttft_ms"),
                             "mean_tpot_ms": m.get("mean_tpot_ms"),
                             "output_throughput": m.get("output_throughput")})
        ttfts = [r.get("mean_ttft_ms") for r in reps if not r["missing"]]
        rebaseline[v] = {"reps": reps, "mean_ttft_stats": stats(ttfts),
                         "safety": scan_srv(vdir / "server.log")}
    # cost delta = fork - stock
    if rebaseline["stock_default"]["mean_ttft_stats"].get("mean") and rebaseline["fork_pcg"]["mean_ttft_stats"].get("mean"):
        cost_delta_ms = rebaseline["fork_pcg"]["mean_ttft_stats"]["mean"] - rebaseline["stock_default"]["mean_ttft_stats"]["mean"]
        cost_ratio = rebaseline["fork_pcg"]["mean_ttft_stats"]["mean"] / rebaseline["stock_default"]["mean_ttft_stats"]["mean"]
    else:
        cost_delta_ms = None; cost_ratio = None

    # ---- R6.3b sweep matrix ----
    sweep = {}
    sweep_dir = ind / "b_sweep"
    if sweep_dir.exists():
        for cell in sorted(sweep_dir.iterdir()):
            if not cell.is_dir(): continue
            cell_id = cell.name
            per_variant = {}
            for v in ("stock_default", "fork_pcg"):
                agg = extract_agg(cell / v / "bench.jsonl")
                srv = scan_srv(cell / v / "server.log")
                per_variant[v] = {
                    "agg": agg, "safety": srv,
                    "mean_ttft_ms": (agg or {}).get("mean_ttft_ms"),
                    "median_ttft_ms": (agg or {}).get("median_ttft_ms"),
                    "mean_tpot_ms": (agg or {}).get("mean_tpot_ms"),
                    "completed": (agg or {}).get("completed"),
                    "output_throughput": (agg or {}).get("output_throughput"),
                }
            sd_ttft = per_variant["stock_default"]["mean_ttft_ms"]
            fp_ttft = per_variant["fork_pcg"]["mean_ttft_ms"]
            fp_safe = (per_variant["fork_pcg"]["safety"]["assertions"] == 0 and
                       per_variant["fork_pcg"]["safety"]["fallbacks"] == 0 and
                       per_variant["fork_pcg"]["safety"]["post_ready_recompiles"] == 0)
            sd_safe = (per_variant["stock_default"]["safety"]["assertions"] == 0 and
                       per_variant["stock_default"]["safety"]["fallbacks"] == 0 and
                       per_variant["stock_default"]["safety"]["post_ready_recompiles"] == 0)
            winning = (isinstance(sd_ttft, (int,float)) and isinstance(fp_ttft, (int,float))
                       and fp_ttft <= sd_ttft and fp_safe and sd_safe
                       and per_variant["fork_pcg"]["completed"] and per_variant["stock_default"]["completed"])
            sweep[cell_id] = {"per_variant": per_variant, "fork_over_stock_ratio":
                              (fp_ttft/sd_ttft) if (isinstance(sd_ttft,(int,float)) and isinstance(fp_ttft,(int,float)) and sd_ttft) else None,
                              "fp_pcg_safe": fp_safe, "stock_default_safe": sd_safe,
                              "winning": winning}

    winning_cells = [c for c, d in sweep.items() if d.get("winning")]

    # ---- R6.3c mixed safety ----
    mixed_dir = ind / "c_mixed_safety"
    mixed = {"missing": True}
    if mixed_dir.exists():
        srv = scan_srv(mixed_dir / "server.log")
        cli = load_json(mixed_dir / "client_summary.json")
        interleaved = load_json(mixed_dir / "fork_pcg_interleaved.jsonl")
        req_fail = 0
        completed = 0
        if cli:
            req_fail = cli.get("request_failures", 0)
            completed = cli.get("completed", 0)
        mixed = {"missing": False, "server": srv, "client": cli or {},
                 "request_failures": req_fail, "completed": completed}

    mixed_ok = (not mixed.get("missing")
                and mixed["server"]["assertions"] == 0
                and mixed["server"]["fallbacks"] == 0
                and mixed["server"]["post_ready_recompiles"] == 0
                and mixed["request_failures"] == 0)

    # Verdict
    reasons = []
    if not mixed_ok: reasons.append(f"R6.3c mixed safety failed: {mixed}")
    if not winning_cells:
        reasons.append("R6.3b workload sweep: no winning cells (fork-PCG >= stock-default on every cell)")
    verdict = "PASS" if not reasons else ("SAFETY_ONLY" if mixed_ok and not winning_cells else "FAIL")

    # Render
    L = [f"# R6.3 verdict — **{verdict}**", ""]
    if launch:
        L.append("## Launch context")
        for k in ("selected_gpu_id","attempt_dir","host_libcuda","ld_preload","nvidia_driver","sglang_stock_head","sglang_fork_head"):
            if k in launch: L.append(f"- `{k}`: `{launch[k]}`")
        L.append("")

    L.append("## R6.3a — Fresh IMG-A rebaseline (720p 1 image, 128->128, c=1, n=400 × 3 reps)")
    L.append("")
    L.append("| variant | reps | mean_ttft_ms | median_ttft_ms | CV% | assertions | fallbacks | post_ready_recompiles |")
    L.append("|---|---|---|---|---|---|---|---|")
    def _f(x, d=3):
        return f"{x:.{d}f}" if isinstance(x,(int,float)) else "—"
    for v in ("stock_default", "fork_pcg"):
        s = rebaseline[v]["mean_ttft_stats"]; sf = rebaseline[v]["safety"]
        L.append(f"| {v} | {s.get('n',0)}/3 | {_f(s.get('mean'))} | {_f(s.get('median'))} | {_f(s.get('cv_pct'),2)} | "
                 f"{sf['assertions']} | {sf['fallbacks']} | {sf['post_ready_recompiles']} |")
    L.append("")
    if cost_delta_ms is not None:
        L.append(f"- fork_pcg - stock_default mean TTFT delta: {_f(cost_delta_ms)} ms (ratio {_f(cost_ratio,4)})")
        L.append("")

    L.append("## R6.3b — Workload sweep (text_tokens × image_res × concurrency)")
    L.append("")
    L.append("| cell | stock_default mean_ttft_ms | fork_pcg mean_ttft_ms | fork/stock ratio | winning? | stock_safe | fork_safe |")
    L.append("|---|---|---|---|---|---|---|")
    for cid in sorted(sweep.keys()):
        d = sweep[cid]; pv = d["per_variant"]
        L.append(f"| `{cid}` | {_f(pv['stock_default']['mean_ttft_ms'])} | {_f(pv['fork_pcg']['mean_ttft_ms'])} | "
                 f"{_f(d['fork_over_stock_ratio'],4)} | {'✅' if d['winning'] else '❌'} | {d['stock_default_safe']} | {d['fp_pcg_safe']} |")
    L.append("")
    L.append(f"**Winning cells** ({len(winning_cells)}): {winning_cells}")
    L.append("")

    L.append("## R6.3c — Mixed-modality safety (fork-PCG interleaved text+image)")
    if mixed.get("missing"):
        L.append("- MISSING")
    else:
        L.append(f"- request_failures: {mixed['request_failures']}")
        L.append(f"- server assertions: {mixed['server']['assertions']}")
        L.append(f"- server fallbacks: {mixed['server']['fallbacks']}")
        L.append(f"- server post_ready_recompiles: {mixed['server']['post_ready_recompiles']}")
    L.append("")

    L.append(f"## Overall verdict: **{verdict}**")
    for r in reasons: L.append(f"- {r}")

    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({
        "verdict": verdict, "reasons": reasons,
        "rebaseline": rebaseline,
        "cost_delta_ms": cost_delta_ms, "cost_ratio": cost_ratio,
        "sweep": sweep, "winning_cells": winning_cells,
        "mixed_safety": mixed, "launch_context": launch,
    }, indent=2, sort_keys=True, default=str))
    print(f"VERDICT={verdict}")
    for r in reasons: print(f"REASON: {r}")
    return 0 if verdict == "PASS" else (2 if verdict == "SAFETY_ONLY" else 1)


if __name__ == "__main__":
    sys.exit(main())
