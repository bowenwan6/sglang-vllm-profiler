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
import statistics
from datetime import datetime, timezone
from pathlib import Path


def load(paths):
    return [r for p in paths for r in json.loads(Path(p).read_text())]


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

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
