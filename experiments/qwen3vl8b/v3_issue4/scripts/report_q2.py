#!/usr/bin/env python3
"""
Issue #4 follow-on — Q2 report (plan.md §12.2).

The deployment question: the prefill-graph flag is server-wide, so at what image
arrival fraction does enabling it stop paying?

Reports three quantities separately and never merges them, because the aggregate
is exactly where the effect being looked for would hide:

  1. **text-class TTFT vs f.** Under `disabled` this is queueing alone. Under
     `breakable`, any *additional* degradation as f rises is co-batching
     interference — a text request that shares a prefill batch with an image
     request sits in a large-N batch and loses the benefit it would have had
     alone. No prior measurement in this study could see this: every earlier
     prefill batch carried exactly one request.
  2. **image-class TTFT vs f** — expected flat; a control that catches confounds.
  3. **aggregate**, derived, giving the break-even image fraction.

Effects are computed **within** each A/B block pair and then summarised, so
common-mode drift cancels rather than being folded back in.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from datetime import datetime, timezone
from pathlib import Path


def load(paths):
    return [r for p in paths for r in json.loads(Path(p).read_text())]


def raw_metrics(results_path):
    """(fraction, arm, block, class) -> {ttft, e2e, tpot} straight from the
    client output files.

    The cells run before the parser fix stored only ttft in `results.json`; the
    raw files always carried the rest. Reading them here means the end-to-end
    view needs no re-run — and it is not optional, because the TTFT break-even
    and the end-to-end break-even are different numbers."""
    raw = Path(results_path).parent / "raw"
    out = {}
    for fp in sorted(raw.glob("f*__*__b*_*.jsonl")):
        m = re.match(r"f([\d.]+)__(\w+)__b(\d+)_(text|image)\.jsonl", fp.name)
        if not m:
            continue
        d = None
        for line in fp.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    d = json.loads(line)
                except Exception:
                    pass
        if not d:
            continue
        out[(float(m.group(1)), m.group(2), int(m.group(3)), m.group(4))] = {
            "ttft": d.get("median_ttft_ms"),
            "e2e": d.get("median_e2e_latency_ms"),
            "tpot": d.get("median_tpot_ms"),
        }
    return out


def paired_metric(raw, f, kind, metric):
    got = {}
    for (ff, arm, blk, kk), m in raw.items():
        if ff == f and kk == kind and m.get(metric) is not None:
            got[(arm, blk)] = m[metric]
    return [(got[("disabled", b)], got[("breakable", b)])
            for b in sorted({b for (_, b) in got})
            if ("disabled", b) in got and ("breakable", b) in got]


def verified(r):
    return (r.get("status") == "OK"
            and (r.get("engagement") or {}).get("engagement") == "VERIFIED")


def paired(recs, f, kind):
    """Per-block (disabled, breakable) TTFT p50 for one class at one fraction."""
    got = {}
    for r in recs:
        if not verified(r) or r.get("image_fraction") != f:
            continue
        m = (r.get("classes") or {}).get(kind)
        if not m or m.get("ttft_p50") is None:
            continue
        got[(r["backend"], r["block"])] = m["ttft_p50"]
    out = []
    for blk in sorted({b for (_, b) in got}):
        if ("disabled", blk) in got and ("breakable", blk) in got:
            out.append((got[("disabled", blk)], got[("breakable", blk)]))
    return out


def summarise(pairs):
    if len(pairs) < 2:
        return None
    effs = [100 * (b - d) / d for d, b in pairs]
    return {"disabled": statistics.median(d for d, _ in pairs),
            "breakable": statistics.median(b for _, b in pairs),
            "effect": statistics.median(effs),
            "spread": max(effs) - min(effs),
            "blocks": len(pairs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    recs = load(a.results)
    fracs = sorted({r["image_fraction"] for r in recs if "image_fraction" in r})

    L = ["# Issue #4 follow-on — Q2: the request-stream mix\n",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
         f"Design: [`plan.md` §12.2](../../../plan.md).\n",
         "Two bench clients against one server at once — one text-only, one "
         "image — each with its own Poisson arrival rate, so the image share of "
         "arrivals is set by their ratio at a fixed total rate. Launching them "
         "together is the point: a sequential run would never co-batch.\n"]

    con = [r["observed_concurrency"]["mean"] for r in recs
           if r.get("observed_concurrency")]
    if con:
        L.append(f"Observed in-flight requests: mean **{statistics.mean(con):.2f}** "
                 f"across cells (target was a mean of ~8; Poisson arrivals make "
                 f"this a distribution, so the measured value is reported rather "
                 f"than the target).\n")

    for kind, title in (("text", "Text-class requests"),
                        ("image", "Image-class requests")):
        rows = [(f, summarise(paired(recs, f, kind))) for f in fracs]
        rows = [(f, s) for f, s in rows if s]
        if not rows:
            continue
        L.append(f"## {title}\n")
        L.append("| image share of arrivals | graph off | graph on | graph effect | "
                 "spread | blocks |")
        L.append("|---|---|---|---|---|---|")
        for f, s in rows:
            L.append(f"| {f:g} | {s['disabled']:.2f} ms | {s['breakable']:.2f} ms | "
                     f"**{s['effect']:+.2f}%** | {s['spread']:.2f} pp | {s['blocks']} |")
        L.append("")
        if kind == "text" and len(rows) > 1:
            base = rows[0][1]["effect"]
            L.append("Change in the graph's benefit to text requests as images "
                     "enter the stream, measured against the image-free case:\n")
            for f, s in rows[1:]:
                L.append(f"- at f={f:g}: **{s['effect'] - base:+.2f} pp** "
                         f"({base:+.2f}% → {s['effect']:+.2f}%)")
            L.append("")
            L.append("A value near zero means co-batching does not erode the "
                     "benefit and the aggregate is a simple weighted average of "
                     "the homogeneous results. A large negative shift is the "
                     "interference this experiment exists to detect.\n")

    # ---- both metrics, and the break-even under each ----------------------
    raw = raw_metrics(a.results[0])
    if raw:
        L.append("## The same result under two metrics\n")
        L.append("Everything above is time-to-first-token, where #2 and #4 "
                 "located the gap. End-to-end latency is what a user waits for, "
                 "and here it is decode-dominated — about 870 ms of ~910 — so a "
                 "few milliseconds of TTFT dilute to well under a percent.\n")
        L.append("| image share | class | TTFT | end-to-end | TPOT |")
        L.append("|---|---|---|---|---|")
        for f in fracs:
            for kind in ("text", "image"):
                cells = []
                for metric in ("ttft", "e2e", "tpot"):
                    pr = paired_metric(raw, f, kind, metric)
                    if len(pr) < 2:
                        cells.append("—")
                        continue
                    effs = [100 * (b - d) / d for d, b in pr]
                    cells.append(f"{statistics.median(effs):+.2f}%")
                if cells != ["—", "—", "—"]:
                    L.append(f"| {f:g} | {kind} | {cells[0]} | **{cells[1]}** | {cells[2]} |")
        L.append("")

        # ---- load confound, stated before any break-even ------------------
        loads = {}
        for r in recs:
            if r.get("observed_concurrency"):
                loads.setdefault(r["image_fraction"], []).append(
                    r["observed_concurrency"]["mean"])
        if len(loads) > 1:
            L.append("## A confound this design could not avoid\n")
            L.append("| image share | mean in-flight requests |")
            L.append("|---|---|")
            for f in sorted(loads):
                L.append(f"| {f:g} | {statistics.mean(loads[f]):.2f} |")
            L.append("")
            L.append("The arrival **rate** was held fixed, not the **load**. "
                     "Image requests take longer to serve, so a higher image "
                     "share at the same rate produces a busier server. "
                     "`f = 0` and `f = 0.2` land within 3% of each other and are "
                     "comparable; `f = 1` runs about 54% busier and differs from "
                     "the others in *two* ways at once.\n")
            L.append("**Consequence:** the image class's end-to-end effect moves "
                     "from −1.45% at f = 0.2 to +3.43% at f = 1, and that "
                     "difference **cannot be attributed** — composition and load "
                     "both changed. Fixing it means tuning the arrival rate per "
                     "fraction to equalise in-flight requests, which is a "
                     "follow-up, not a reinterpretation of this data.\n")

        # ---- break-even, from load-matched cells only ----------------------
        mixed = [f for f in fracs if 0 < f < 1]
        for metric, label in (("ttft", "TTFT"), ("e2e", "end-to-end latency")):
            for f in mixed:
                tp = paired_metric(raw, f, "text", metric)
                ip = paired_metric(raw, f, "image", metric)
                if len(tp) < 2 or len(ip) < 2:
                    continue
                t_sav = statistics.median([d - b for d, b in tp])
                i_cost = statistics.median([b - d for d, b in ip])
                lo = statistics.mean(loads.get(f, [0]))
                if t_sav <= 0:
                    continue
                if i_cost <= 0:
                    L.append(f"**{label}, at ~{lo:.1f} in flight: no break-even.** "
                             f"The graph is faster for *both* classes — text by "
                             f"{t_sav:.2f} ms and image by {-i_cost:.2f} ms per "
                             f"request — so it pays at every image fraction "
                             f"**at this load**.")
                else:
                    be = t_sav / (t_sav + i_cost)
                    L.append(f"**{label}, at ~{lo:.1f} in flight: break-even at "
                             f"f ≈ {be:.2f}** (text saves {t_sav:.2f} ms, image "
                             f"costs {i_cost:.2f} ms per request).")
                # the same arithmetic using the busier pure-image cell, flagged
                ip1 = paired_metric(raw, 1.0, "image", metric)
                if len(ip1) >= 2:
                    c1 = statistics.median([b - d for d, b in ip1])
                    l1 = statistics.mean(loads.get(1.0, [0]))
                    if c1 > 0:
                        be1 = t_sav / (t_sav + c1)
                        L.append(f"  Using instead the image cost measured on the "
                                 f"pure-image stream ({c1:.2f} ms at ~{l1:.1f} in "
                                 f"flight, 600 requests rather than 120) gives "
                                 f"f ≈ {be1:.2f} — but that mixes two operating "
                                 f"points and is quoted only to bound the range.")
                L.append("")
                break

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
