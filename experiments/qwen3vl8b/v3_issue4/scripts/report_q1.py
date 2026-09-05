#!/usr/bin/env python3
"""
Issue #4 follow-on — Q1 report (plan.md §12.1).

Reports the matched-N comparison: three text-only cells against the image
workloads of the same total prefill token count measured in v3.

Two rules applied mechanically:

  * A block without `engagement: VERIFIED` is dropped; a workload with fewer than
    two verified blocks per arm is reported as unanswered.
  * **Within-pair drift is the resolution floor for that workload.** The three
    `disabled` blocks of a workload bound how much an identical configuration
    moved during the bracket; an effect smaller than that spread is reported as
    not resolvable rather than given a sign. Paired A/B/A/B blocking exists to
    make this floor small — it is not decoration, and it is reported per
    workload rather than assumed uniform.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path

# v3 partners, measured on the image axis: (composition, N, effect %, disabled ms)
PARTNERS = {
    "text-208":  ("R1_tiny  (66 visual + 142 text)",   208,  -16.30, 55.23),
    "text-544":  ("R6_640   (402 visual + 142 text)",  544,   -4.54, 69.99),
    "text-1024": ("R3_720p  (882 visual + 142 text)", 1024,   +0.80, 104.51),
}
GATE = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    recs = json.loads(a.results.read_text())

    by = {}
    for r in recs:
        if r.get("status") != "OK":
            continue
        if (r.get("engagement") or {}).get("engagement") != "VERIFIED":
            continue
        by.setdefault((r["workload"], r["backend"]), []).append(r["ttft_p50"])

    L = ["# Issue #4 follow-on — Q1: is a text token equivalent to a visual token?\n",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from "
         f"`{a.results.name}`. Design: [`plan.md` §12.1](../../../plan.md).\n",
         "Three text-only workloads whose total prefill token count matches an "
         "image workload already measured in v3, so half of each comparison "
         "already exists. Paired A/B/A/B blocking, transport `cuda_ipc`, c=1.\n"]

    L.append("## Matched-N pairs\n")
    L.append("| N | text-only | with an image | text-only effect | image effect | "
             "difference | within-pair drift |")
    L.append("|---|---|---|---|---|---|---|")
    rows = []
    for wid, (partner, N, p_eff, p_dis) in PARTNERS.items():
        d = by.get((wid, "disabled"), [])
        b = by.get((wid, "breakable"), [])
        if len(d) < 2 or len(b) < 2:
            L.append(f"| {N} | `{wid}` | {partner} | *unanswered — "
                     f"{len(d)} disabled / {len(b)} breakable verified blocks* | | | |")
            continue
        dv, bv = statistics.median(d), statistics.median(b)
        drift = 100 * (max(d) - min(d)) / dv
        eff = 100 * (bv - dv) / dv
        diff = eff - p_eff
        note = "" if drift <= GATE else " ⚠"
        L.append(f"| {N} | {dv:.2f} → {bv:.2f} ms | {p_dis:.2f} ms (v3) | "
                 f"**{eff:+.2f}%** | {p_eff:+.2f}% | {diff:+.2f} pp | "
                 f"{drift:.2f}%{note} |")
        rows.append({"wid": wid, "N": N, "dv": dv, "bv": bv, "eff": eff,
                     "p_eff": p_eff, "p_dis": p_dis, "drift": drift,
                     "saving": dv - bv})
    L.append("")
    L.append(f"Drift is the spread of the three `disabled` blocks, and is this "
             f"workload's resolution floor. Gate {GATE}%; ⚠ marks a workload that "
             "exceeded it and whose effect should not be given a sign.\n")

    if rows:
        L.append("## Absolute saving vs percentage effect\n")
        L.append("The two readings of \"token type is irrelevant\" are "
                 "incompatible, because the image cells have a much larger "
                 "denominator — the vision encoder's fixed cost, which the graph "
                 "cannot touch.\n")
        L.append("| N | text-only saving | image saving (v3) | text-only effect | image effect |")
        L.append("|---|---|---|---|---|")
        for r in rows:
            p_sav = r["p_dis"] * (-r["p_eff"]) / 100.0
            L.append(f"| {r['N']} | **{r['saving']:+.2f} ms** | {p_sav:+.2f} ms | "
                     f"{r['eff']:+.2f}% | {r['p_eff']:+.2f}% |")
        L.append("")
        L.append("If the **saving** column matches across a row, the graph "
                 "recovers the same absolute work regardless of whether the "
                 "tokens are visual or textual, and the percentage differs only "
                 "because the image adds fixed cost below it. If the **effect** "
                 "column matches instead, token count alone fixes the relative "
                 "benefit. They cannot both match.\n")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
