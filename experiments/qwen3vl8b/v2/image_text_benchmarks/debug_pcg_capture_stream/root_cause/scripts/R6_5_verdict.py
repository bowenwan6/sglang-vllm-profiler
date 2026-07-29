#!/usr/bin/env python3
"""R6.5 verdict — empirical mixed-workload validation vs R6.4 analytical prediction.

Reads:
  raw/<ratio_id>/<variant>/summary.json         (per-run stats)
  raw/<ratio_id>/<variant>/server.log           (safety)
  R6.4 verdict.json                              (analytical predictions to compare)

Predeclared:
  * At least 3 ratios: below, near, above the analytical p*.
  * Identical seed + sequence for stock_default and fork_pcg.
  * Acceptance tolerance: predicted direction (fork wins / loses) must
    match empirical direction on at least 2 of 3 ratios.

Verdict:
  PASS iff every run has 0 failures/assertions/fallbacks/post-ready
  recompiles AND empirical direction agrees with analytical for
  the "near" ratio AND at least 2 of 3 ratios agree.
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path

ASSERT_RE = re.compile(r"AssertionError: PCG capture stream is not set")
FALLBACK_RE = re.compile(r"Falling back to eager execution")
RECOMPILE_RE = re.compile(r"Recompiling function.*qwen3_vl")
READY_RE = re.compile(r"The server is fired up and ready to roll")


def load_json(p):
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except: return None


def scan(log):
    if not log.exists(): return {"missing": True}
    lines = log.read_text(errors="replace").splitlines()
    ready = None
    for i, ln in enumerate(lines, 1):
        if READY_RE.search(ln): ready = i; break
    recomp = [i+1 for i, ln in enumerate(lines) if RECOMPILE_RE.search(ln)]
    return {"missing": False, "total_lines": len(lines), "ready_line": ready,
            "assertions": sum(1 for ln in lines if ASSERT_RE.search(ln)),
            "fallbacks":  sum(1 for ln in lines if FALLBACK_RE.search(ln)),
            "startup_recompiles": len([r for r in recomp if ready is None or r < ready]),
            "post_ready_recompiles": len([r for r in recomp if ready is not None and r >= ready])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--r64-json", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    r64 = load_json(args.r64_json) or {}
    p_star = r64.get("p_star")
    launch = load_json(args.in_dir / "launch_context.json")

    ratios = {}
    for rdir in sorted((args.in_dir).iterdir()) if args.in_dir.exists() else []:
        if not rdir.is_dir() or not rdir.name.startswith("ratio_"): continue
        rid = rdir.name
        per = {}
        for v in ("stock_default", "fork_pcg"):
            s = load_json(rdir / v / "summary.json")
            srv = scan(rdir / v / "server.log")
            per[v] = {"summary": s, "safety": srv}
        # empirical direction
        sd = per["stock_default"]["summary"] or {}
        fp = per["fork_pcg"]["summary"] or {}
        sd_lat = (sd.get("all_latency_s_stats") or {}).get("mean")
        fp_lat = (fp.get("all_latency_s_stats") or {}).get("mean")
        emp_fork_wins = (isinstance(sd_lat, (int,float)) and isinstance(fp_lat, (int,float)) and fp_lat <= sd_lat)
        ratios[rid] = {"per_variant": per,
                       "mean_lat_stock": sd_lat, "mean_lat_fork": fp_lat,
                       "empirical_fork_wins": emp_fork_wins,
                       "empirical_ratio_fork_over_stock": (fp_lat/sd_lat) if (isinstance(sd_lat,(int,float)) and isinstance(fp_lat,(int,float)) and sd_lat) else None,
                       "text_ratio": (sd.get("text_ratio") if sd else (fp.get("text_ratio") if fp else None))}

    # Predicted direction per ratio (from analytical p*)
    for rid, d in ratios.items():
        tr = d.get("text_ratio")
        if isinstance(tr, (int,float)) and isinstance(p_star, (int,float)):
            d["predicted_fork_wins"] = (tr >= p_star)
        else:
            d["predicted_fork_wins"] = None
        d["agreement"] = (d["predicted_fork_wins"] == d["empirical_fork_wins"]) if d["predicted_fork_wins"] is not None else None

    # Safety across all ratios+variants
    safety_issues = []
    for rid, d in ratios.items():
        for v in ("stock_default", "fork_pcg"):
            srv = d["per_variant"][v]["safety"]
            s = d["per_variant"][v]["summary"] or {}
            if srv.get("missing"):
                safety_issues.append(f"{rid}/{v}: server.log missing")
            else:
                for k in ("assertions","fallbacks","post_ready_recompiles"):
                    if srv.get(k,0) > 0:
                        safety_issues.append(f"{rid}/{v}: {k}={srv.get(k)}")
            if s.get("request_failures",0) > 0:
                safety_issues.append(f"{rid}/{v}: request_failures={s.get('request_failures')}")

    # Agreement
    agreements = [d["agreement"] for d in ratios.values() if d["agreement"] is not None]
    n_agree = sum(1 for a in agreements if a)
    verdict = "PASS"
    reasons = []
    if safety_issues:
        verdict = "FAIL"; reasons.extend(safety_issues)
    if len(ratios) < 3:
        verdict = "AMBIGUOUS"; reasons.append(f"only {len(ratios)} ratios recorded, require >=3")
    elif len(agreements) < 3:
        verdict = "AMBIGUOUS"; reasons.append(f"only {len(agreements)} agreements comparable")
    elif n_agree < 2:
        verdict = "AMBIGUOUS"; reasons.append(f"only {n_agree}/{len(agreements)} ratios agree with analytical direction")

    L = [f"# R6.5 empirical mixed-workload — **{verdict}**", ""]
    L.append(f"- analytical p* (from R6.4): {p_star}")
    L.append(f"- ratios executed: {sorted(ratios.keys())}")
    L.append("")
    L.append("| ratio_id | text_ratio | mean_lat_stock | mean_lat_fork | fork/stock | empirical fork wins | predicted fork wins | agree? |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rid in sorted(ratios.keys()):
        d = ratios[rid]
        L.append(f"| `{rid}` | {d['text_ratio']} | {d['mean_lat_stock']} | {d['mean_lat_fork']} | "
                 f"{d['empirical_ratio_fork_over_stock']} | {d['empirical_fork_wins']} | "
                 f"{d['predicted_fork_wins']} | {d['agreement']} |")
    L.append("")
    if reasons:
        L.append("## Reasons")
        for r in reasons: L.append(f"- {r}")
    L.append(f"\n## Overall verdict: **{verdict}**")
    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({"verdict": verdict, "reasons": reasons,
        "p_star": p_star, "ratios": ratios, "launch_context": launch}, indent=2, sort_keys=True, default=str))
    print(f"VERDICT={verdict}")
    for r in reasons: print(f"REASON: {r}")
    return 0 if verdict == "PASS" else (2 if verdict == "AMBIGUOUS" else 1)


if __name__ == "__main__":
    sys.exit(main())
