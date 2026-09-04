#!/usr/bin/env python3
"""
Issue #4 v3 — IMG-R sweep report (plan.md §11.8).

Answers "is there an image+text mix where the prefill CUDA graph pays?" by
reporting the graph effect per workload against total prefill token count N.

Two rules are applied mechanically rather than by eye:

  * A cell without `engagement: VERIFIED` is excluded, and the workload it
    belongs to is reported as unanswered.
  * **An effect smaller than the bracket's own drift is not resolvable.** The
    drift figure (from the repeated reference cell) is compared against every
    workload's effect, and any workload whose |effect| falls below it is reported
    as "not resolvable at this bracket's precision" rather than given a sign.
    This is the difference between "the graph is neutral here" and "we cannot
    tell" — at large N those are the same number and different claims.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

MATERIAL_PCT = 5.0


def load(paths):
    cells = {}
    for p in paths:
        for r in json.loads(Path(p).read_text()):
            cells[r["cell"]] = r
    return cells


def verified(rec):
    return (rec is not None and rec.get("status") == "OK"
            and (rec.get("engagement") or {}).get("engagement") == "VERIFIED")


def reason(rec):
    if rec is None:
        return "cell not run"
    if rec.get("status") != "OK":
        return f"status={rec.get('status')}"
    return "; ".join((rec.get("engagement") or {}).get("reasons") or []) or "?"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="IMG-R ratio sweep")
    a = ap.parse_args()

    cells = load(a.results)
    L = [f"# Issue #4 v3 — {a.title}\n",
         f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC}. "
         f"Stack and model: [`manifest.md`](manifest.md). "
         f"Design: [`plan.md` §11.8](../../../plan.md).\n",
         "Transport is pinned to `cuda_ipc` on every cell — issue #4's stated "
         "standard condition for SGLang image runs — so the only variable is the "
         "prefill CUDA-graph backend.\n"]

    # ---- drift ------------------------------------------------------------
    drift_pct = None
    base, rep = cells.get("R3_720p__disabled"), cells.get("R3_720p__disabled__repeat")
    L.append("## Bracket validity\n")
    if base and rep and base.get("ttft_p50_median") and rep.get("ttft_p50_median"):
        drift_pct = abs(100.0 * (rep["ttft_p50_median"] - base["ttft_p50_median"])
                        / base["ttft_p50_median"])
        L.append(f"Reference cell `R3_720p__disabled` {base['ttft_p50_median']:.2f} ms "
                 f"→ repeat {rep['ttft_p50_median']:.2f} ms, drift **{drift_pct:.2f}%** "
                 f"(gate ≤ 5%): **{'PASS' if drift_pct <= 5 else 'FAIL'}**\n")
        L.append(f"**Resolution floor: {drift_pct:.2f}%.** Any effect smaller than "
                 "this cannot be given a sign by this bracket, however tight the "
                 "per-cell CV looks — the repeat of an *identical* configuration "
                 "moved by that much.\n")
    else:
        L.append("Reference repeat missing — drift not checked.\n")

    # ---- per-workload -----------------------------------------------------
    order = [c[:-len("__disabled")] for c in cells if c.endswith("__disabled")
             and not c.endswith("__repeat")]
    order.sort(key=lambda w: (cells[f"{w}__disabled"].get("measured_N") or 0))

    L.append("## Graph effect by workload\n")
    L.append("| workload | N | vision tok | text tok | disabled | breakable | "
             "saving | graph effect | verdict |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for w in order:
        d, b = cells.get(f"{w}__disabled"), cells.get(f"{w}__breakable")
        if not verified(d) or not verified(b):
            bad = w + ("__disabled" if not verified(d) else "__breakable")
            L.append(f"| `{w}` | — | — | — | — | — | — | — | excluded: {reason(cells.get(bad))} |")
            continue
        dv, bv = d["ttft_p50_median"], b["ttft_p50_median"]
        eff = 100.0 * (bv - dv) / dv
        saving = dv - bv
        if drift_pct is not None and abs(eff) < drift_pct:
            verdict = "**not resolvable** (below drift)"
        elif abs(eff) < MATERIAL_PCT:
            verdict = "no material difference"
        else:
            verdict = "**graph wins**" if eff < 0 else "**graph costs**"
        rows.append((w, d.get("measured_N"), saving, eff, verdict))
        L.append(f"| `{w}` | {d.get('measured_N')} | {d.get('vision_tok_per_req')} "
                 f"| {d.get('text_tok_per_req')} | {dv:.2f} ms | {bv:.2f} ms "
                 f"| {saving:+.2f} ms | **{eff:+.2f}%** | {verdict} |")
    L.append("")

    wins = [r for r in rows if r[3] <= -MATERIAL_PCT]
    if wins:
        L.append("## Answer\n")
        L.append("**Yes — there are image+text mixes where the prefill CUDA graph "
                 "pays, and the boundary is measurable.**\n")
        for w, n, s, e, _ in wins:
            L.append(f"- `{w}` (N={n}): **{e:+.2f}%**, {s:+.2f} ms")
        losers = [r for r in rows if abs(r[3]) < MATERIAL_PCT]
        if losers:
            lo = min(r[1] for r in losers)
            hi = max(r[1] for r in wins)
            L.append(f"\nThe transition sits between **N ≈ {hi}** (last workload with "
                     f"a material win) and **N ≈ {lo}** (first workload without one).\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
