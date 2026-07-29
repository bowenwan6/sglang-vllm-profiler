#!/usr/bin/env python3
"""R6.2 verdict — text-only Case A non-regression on Qwen3-VL.

Aggregates per-rep bench.jsonl aggregates from
  raw/<variant>/rep<N>.jsonl
for variants: stock_default, stock_pcg, fork_pcg, stock_default_repeat.

Predeclared gates:
  * fork_pcg mean TTFT <= stock_pcg mean TTFT * 1.05
  * per-variant CV <= 6%
  * drift bracket stock_default vs stock_default_repeat <= 3%
  * per-server safety: 0 assertions / 0 fallbacks / 0 post-ready recompiles
    / 0 request failures
"""
from __future__ import annotations
import argparse, json, math, re, sys
from pathlib import Path

VARIANTS = ["stock_default", "stock_pcg", "fork_pcg", "stock_default_repeat"]

ASSERT_RE = re.compile(r"AssertionError: PCG capture stream is not set")
FALLBACK_RE = re.compile(r"Falling back to eager execution")
RECOMPILE_RE = re.compile(r"Recompiling function.*qwen3_vl")
READY_RE = re.compile(r"The server is fired up and ready to roll")


def load_json(p: Path):
    if not p.exists(): return None
    try: return json.loads(p.read_text())
    except: return None


def scan_server(log_path: Path):
    if not log_path.exists(): return {"missing": True}
    lines = log_path.read_text(errors="replace").splitlines()
    ready = None
    for i, ln in enumerate(lines, 1):
        if READY_RE.search(ln):
            ready = i; break
    recomp = [i+1 for i, ln in enumerate(lines) if RECOMPILE_RE.search(ln)]
    return {
        "missing": False,
        "total_lines": len(lines),
        "ready_line": ready,
        "assertions": sum(1 for ln in lines if ASSERT_RE.search(ln)),
        "fallbacks": sum(1 for ln in lines if FALLBACK_RE.search(ln)),
        "startup_recompiles": len([r for r in recomp if ready is None or r < ready]),
        "post_ready_recompiles": len([r for r in recomp if ready is not None and r >= ready]),
    }


def extract_rep_metrics(jsonl_path: Path):
    if not jsonl_path.exists(): return None
    lines = jsonl_path.read_text().strip().splitlines()
    if not lines: return None
    # Aggregate is typically the LAST JSON line from sglang.benchmark.serving
    for ln in reversed(lines):
        try: obj = json.loads(ln)
        except: continue
        if isinstance(obj, dict) and any(k in obj for k in
            ("completed", "num_prompts", "median_ttft_ms", "mean_ttft_ms")):
            return obj
    return None


def stats(xs):
    xs = [x for x in xs if x is not None]
    n = len(xs)
    if n == 0: return {"n": 0}
    mean = sum(xs)/n
    if n > 1:
        var = sum((x-mean)**2 for x in xs)/(n-1)
        sd = math.sqrt(var)
    else:
        sd = 0.0
    med = sorted(xs)[n//2] if n % 2 else (sorted(xs)[n//2-1]+sorted(xs)[n//2])/2
    cv = (sd/mean*100.0) if mean else float("nan")
    return {"n": n, "mean": mean, "median": med, "stdev": sd, "cv_pct": cv,
            "min": min(xs), "max": max(xs), "values": xs}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()
    ind = args.in_dir

    launch = load_json(ind / "launch_context.json")
    per_variant = {}
    for v in VARIANTS:
        vdir = ind / v
        srv = scan_server(vdir / "server.log")
        reps = []
        for i in range(1, 6):
            m = extract_rep_metrics(vdir / f"rep{i}.jsonl")
            if m is None:
                reps.append({"rep": i, "missing": True})
            else:
                reps.append({
                    "rep": i, "missing": False,
                    "completed": m.get("completed"),
                    "num_prompts": m.get("num_prompts"),
                    "mean_ttft_ms": m.get("mean_ttft_ms"),
                    "median_ttft_ms": m.get("median_ttft_ms"),
                    "mean_tpot_ms": m.get("mean_tpot_ms"),
                    "median_tpot_ms": m.get("median_tpot_ms"),
                    "output_throughput": m.get("output_throughput"),
                    "duration": m.get("duration") or m.get("total_duration_s"),
                })
        ttfts = [r.get("mean_ttft_ms") for r in reps if not r["missing"]]
        p50s  = [r.get("median_ttft_ms") for r in reps if not r["missing"]]
        tpots = [r.get("mean_tpot_ms") for r in reps if not r["missing"]]
        thrs  = [r.get("output_throughput") for r in reps if not r["missing"]]
        per_variant[v] = {
            "reps": reps,
            "safety": srv,
            "mean_ttft_stats": stats(ttfts),
            "p50_ttft_stats": stats(p50s),
            "tpot_stats": stats(tpots),
            "throughput_stats": stats(thrs),
        }

    # Predeclared gates
    sp_ttft = per_variant["stock_pcg"]["mean_ttft_stats"].get("mean")
    fp_ttft = per_variant["fork_pcg"]["mean_ttft_stats"].get("mean")
    sd_ttft = per_variant["stock_default"]["mean_ttft_stats"].get("mean")
    sdr_ttft = per_variant["stock_default_repeat"]["mean_ttft_stats"].get("mean")

    reasons = []
    fork_ratio = (fp_ttft/sp_ttft) if (sp_ttft and fp_ttft) else None
    if fork_ratio is None:
        reasons.append("cannot compute fork_pcg / stock_pcg ratio (missing means)")
    elif fork_ratio > 1.05:
        reasons.append(f"fork_pcg mean_ttft {fp_ttft:.3f} > stock_pcg {sp_ttft:.3f} * 1.05 (ratio {fork_ratio:.3f})")

    for v in VARIANTS:
        s = per_variant[v]["mean_ttft_stats"]
        if s.get("n", 0) == 0:
            reasons.append(f"{v}: no rep data")
            continue
        cv = s.get("cv_pct")
        if cv is not None and cv > 6.0:
            reasons.append(f"{v}: CV {cv:.2f}% > 6%")

    drift = None
    if sd_ttft and sdr_ttft:
        drift = abs(sd_ttft - sdr_ttft) / sd_ttft * 100.0
        if drift > 3.0:
            reasons.append(f"drift stock_default vs repeat {drift:.2f}% > 3%")

    # Safety
    for v in VARIANTS:
        s = per_variant[v]["safety"]
        if s.get("missing"):
            reasons.append(f"{v}: server.log missing"); continue
        for k in ("assertions", "fallbacks", "post_ready_recompiles"):
            if s.get(k, 0) > 0:
                reasons.append(f"{v}: safety {k}={s.get(k)}")

    verdict = "PASS" if not reasons else "FAIL"

    # Render
    L = [f"# R6.2 verdict — **{verdict}**", ""]
    if launch:
        L.append("## Launch context")
        for k in ("selected_gpu_id","attempt_dir","host_libcuda","ld_preload",
                  "cuda_visible_devices","nvidia_driver","sglang_stock_head",
                  "sglang_fork_head"):
            if k in launch: L.append(f"- `{k}`: `{launch[k]}`")
        L.append("")

    L.append("## Predeclared gates")
    if fork_ratio is not None:
        L.append(f"- fork_pcg mean TTFT / stock_pcg mean TTFT: {fork_ratio:.4f} (require ≤ 1.05)")
    else:
        L.append("- fork_pcg / stock_pcg ratio: n/a")
    if drift is not None:
        L.append(f"- drift stock_default vs stock_default_repeat: {drift:.3f}% (require ≤ 3.0%)")
    else:
        L.append("- drift: n/a")
    L.append("- per-variant CV ≤ 6% and safety zeros")
    L.append("")

    L.append("## Per-variant summary")
    L.append("")
    L.append("| variant | reps completed | mean_ttft_ms | median_ttft_ms | CV% | mean_tpot_ms | output_throughput | assertions | fallbacks | post_ready_recompiles |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    def _fmt(v, digits=3):
        return f"{v:.{digits}f}" if isinstance(v, (int, float)) else "—"
    for v in VARIANTS:
        s = per_variant[v]["mean_ttft_stats"]; p = per_variant[v]["p50_ttft_stats"]
        t = per_variant[v]["tpot_stats"]; th = per_variant[v]["throughput_stats"]
        sf = per_variant[v]["safety"]
        L.append(f"| {v} | {s.get('n', 0)}/5 | "
                 f"{_fmt(s.get('mean'))} | {_fmt(s.get('median'))} | {_fmt(s.get('cv_pct'), 2)} | "
                 f"{_fmt(t.get('mean'))} | {_fmt(th.get('mean'))} | "
                 f"{sf.get('assertions', '—')} | {sf.get('fallbacks', '—')} | {sf.get('post_ready_recompiles', '—')} |")
    L.append("")

    L.append("## Per-rep detail")
    L.append("")
    for v in VARIANTS:
        L.append(f"### {v}")
        L.append("| rep | completed | mean_ttft_ms | median_ttft_ms | mean_tpot_ms | output_throughput |")
        L.append("|---|---|---|---|---|---|")
        for r in per_variant[v]["reps"]:
            if r["missing"]: L.append(f"| {r['rep']} | MISSING | — | — | — | — |"); continue
            L.append(f"| {r['rep']} | {r.get('completed')} | "
                     f"{_fmt(r.get('mean_ttft_ms'))} | {_fmt(r.get('median_ttft_ms'))} | "
                     f"{_fmt(r.get('mean_tpot_ms'))} | {_fmt(r.get('output_throughput'))} |")
        L.append("")

    if reasons:
        L.append("## Failure reasons")
        for r in reasons: L.append(f"- {r}")
        L.append("")

    L.append(f"## Overall verdict: **{verdict}**")
    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({
        "verdict": verdict, "reasons": reasons,
        "fork_pcg_over_stock_pcg_ratio": fork_ratio,
        "drift_pct": drift, "per_variant": per_variant,
        "launch_context": launch,
    }, indent=2, sort_keys=True, default=str))
    print(f"VERDICT={verdict}")
    for r in reasons: print(f"REASON: {r}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
