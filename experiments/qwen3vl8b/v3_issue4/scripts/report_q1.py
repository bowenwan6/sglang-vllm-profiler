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
    "text-208":  ("R1_tiny  (66 visual + 142 text)",   208,  -16.30, 55.231),
    "text-544":  ("R6_640   (402 visual + 142 text)",  544,   -4.54, 69.992),
    # `text-1016`, not `text-1024`: the client's --random-input-len excludes the
    # chat template's ~8 tokens, so 1024 arrives as 1032 and pads to the 1280
    # bucket (24% waste) while the partner lands on 1024 exactly. 1016 makes the
    # server see 1024 and the comparison genuinely matched.
    "text-1016": ("R3_720p  (882 visual + 142 text)", 1024,   +0.80, 104.507),
}
GATE = 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    recs = [r for f in a.results for r in json.loads(f.read_text())]

    by = {}
    for r in recs:
        if r.get("status") != "OK":
            continue
        if (r.get("engagement") or {}).get("engagement") != "VERIFIED":
            continue
        by[(r["workload"], r["backend"], r["block"])] = r["ttft_p50"]

    L = ["# Issue #4 follow-on — Q1: is a text token equivalent to a visual token?\n",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from "
         f"`{', '.join(f.name for f in a.results)}`. Design: [`plan.md` §12.1](../../../plan.md).\n",
         "Three text-only workloads whose total prefill token count matches an "
         "image workload already measured in v3, so half of each comparison "
         "already exists. Paired A/B/A/B blocking, transport `cuda_ipc`, c=1.\n"]

    L.append("## Matched-N pairs\n")
    L.append("| N | text-only | with an image | text-only effect | image effect | "
             "difference | within-pair drift |")
    L.append("|---|---|---|---|---|---|---|")
    rows = []
    for wid, (partner, N, p_eff, p_dis) in PARTNERS.items():
        blocks = [k[2] for k in by if k[0] == wid and k[1] == "disabled"]
        pairs = [(by[(wid, "disabled", i)], by[(wid, "breakable", i)])
                 for i in sorted(blocks)
                 if (wid, "breakable", i) in by]
        if len(pairs) < 2:
            L.append(f"| {N} | `{wid}` | {partner} | *unanswered — "
                     f"{len(pairs)} paired blocks* | | | |")
            continue
        # Effects are computed *within* each A/B pair, then summarised. The
        # absolute levels carry common-mode drift that the pairing exists to
        # cancel; summarising the levels first would put it straight back in.
        effs = [100 * (bv - dv) / dv for dv, bv in pairs]
        savs = [dv - bv for dv, bv in pairs]
        dv = statistics.median([p[0] for p in pairs])
        bv = statistics.median([p[1] for p in pairs])
        eff = statistics.median(effs)
        drift = max(effs) - min(effs)
        diff = eff - p_eff
        note = "" if drift <= max(GATE, abs(eff) / 2) else " ⚠"
        L.append(f"| {N} | {dv:.2f} → {bv:.2f} ms | {p_dis:.2f} ms (v3) | "
                 f"**{eff:+.2f}%** | {p_eff:+.2f}% | {diff:+.2f} pp | "
                 f"{drift:.2f}%{note} |")
        rows.append({"wid": wid, "N": N, "dv": dv, "bv": bv, "eff": eff,
                     "p_eff": p_eff, "p_dis": p_dis, "drift": drift,
                     "saving": statistics.median(savs)})
    L.append("")
    L.append("Drift here is the spread of the **paired** effects — the "
             "resolution floor after A/B/A/B blocking has removed common-mode "
             "drift. (Gating the absolute levels instead would reinstate exactly "
             "what the pairing removed: `text-208`'s levels move 19.4% across "
             "the bracket on a cold-start ramp while its paired effects agree to "
             "3 pp.) ⚠ marks a workload whose spread is comparable to its own "
             "effect, which should therefore not be given a sign.\n")

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
