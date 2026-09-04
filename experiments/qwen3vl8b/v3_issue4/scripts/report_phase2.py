#!/usr/bin/env python3
"""
Issue #4 v3 — Phase 3 analysis (plan.md §11.3).

Reads a bracket's results.json and writes the report. Four questions, each
answered **against its own baseline** and each carrying its arms' engagement
verdicts; PCG and BCG are never conflated.

    Q1  cross-framework gap   A0_default  vs V0_vllm
    Q2  PCG transfer          A2_tcp      vs A1_disabled
    Q3  IPC transport benefit A4_ipc      vs A0_default
    Q4  BCG value             A3_bcg      vs A1_disabled
    Qc  composition           A5_ipc_best vs max(A2_tcp, A3_bcg)

Reporting rules, applied mechanically rather than by judgement:

  * An arm without `engagement: VERIFIED` is never used in a comparison. The
    question it would have answered is reported as unanswered, with the arm's
    exact failure quoted.
  * A delta is called only when |Δ| ≥ 5% AND both arms' TTFT p50 CV ≤ 5%.
    Anything smaller is "no material difference" -- never "a trend".
  * The bracket is void unless |A0_repeat − A0_default| ≤ 5%. No partial rescue.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DELTA_THRESHOLD_PCT = 5.0
CV_BAND_PCT = 5.0
DRIFT_GATE_PCT = 5.0

QUESTIONS = [
    ("Q1", "Cross-framework gap (image path)", "A0_default", "V0_vllm",
     "How far is SGLang's production default from vLLM on image+text? "
     "This is #2's Case-A question moved to the image path."),
    ("Q2", "Does #2's PCG win transfer to images?", "A2_tcp", "A1_disabled",
     "#2 showed tc_piecewise took text-only Case A from 21.94 ms to 14.04 ms. "
     "Against the no-prefill-graph floor, does the same lever pay on images?"),
    ("Q3", "CUDA IPC feature transport, isolated", "A4_ipc", "A0_default",
     "Transport is orthogonal to graph coverage. Measured against the real "
     "production default (cpu transport), not against a chosen non-default."),
    ("Q4", "Breakable prefill CUDA graph value", "A3_bcg", "A1_disabled",
     "What PR #33726 buys once Qwen3-VL is on the breakable allowlist, against "
     "the floor that upstream gives the arch today."),
]


def load(path: Path):
    return {r["arm"]: r for r in json.loads(path.read_text())}


def verified(rec) -> bool:
    return (rec is not None
            and rec.get("status") == "OK"
            and (rec.get("engagement") or {}).get("engagement") == "VERIFIED")


def eng_reason(rec) -> str:
    if rec is None:
        return "arm not run"
    if rec.get("status") != "OK":
        return f"status={rec.get('status')}"
    e = rec.get("engagement") or {}
    return "; ".join(e.get("reasons") or []) or "no reason recorded"


def fmt_ms(v):
    return "—" if v is None else f"{v:.2f} ms"


def compare(a, b):
    """Return (delta_pct, verdict) for arm `a` measured against baseline `b`."""
    va, vb = a.get("ttft_p50_median"), b.get("ttft_p50_median")
    if va is None or vb is None or not vb:
        return None, "no number"
    d = 100.0 * (va - vb) / vb
    cva = a.get("ttft_p50_cv_pct") or 0.0
    cvb = b.get("ttft_p50_cv_pct") or 0.0
    if cva > CV_BAND_PCT or cvb > CV_BAND_PCT:
        return d, f"inconclusive — CV out of band ({cva}% / {cvb}%)"
    if abs(d) < DELTA_THRESHOLD_PCT:
        return d, "no material difference"
    return d, ("faster" if d < 0 else "slower")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title", default="IMG-A headline")
    a = ap.parse_args()

    arms = load(a.results)
    L = []
    L.append(f"# Issue #4 v3 — {a.title}\n")
    L.append(f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M UTC} from "
             f"`{a.results.name}`. Stack and model: [`../manifest.md`](../manifest.md).\n")

    # ---- Drift gate -------------------------------------------------------
    a0, a0r = arms.get("A0_default"), arms.get("A0_repeat")
    L.append("## Bracket validity\n")
    if a0 and a0r and a0.get("ttft_p50_median") and a0r.get("ttft_p50_median"):
        drift = abs(100.0 * (a0r["ttft_p50_median"] - a0["ttft_p50_median"])
                    / a0["ttft_p50_median"])
        ok = drift <= DRIFT_GATE_PCT
        L.append(f"`A0_default` {fmt_ms(a0['ttft_p50_median'])} → `A0_repeat` "
                 f"{fmt_ms(a0r['ttft_p50_median'])}, drift **{drift:.2f}%** "
                 f"(gate ≤ {DRIFT_GATE_PCT}%): "
                 f"**{'PASS' if ok else 'FAIL — the whole bracket is discarded'}**\n")
        if not ok:
            L.append("> No partial rescue: with the bracket drifting beyond the "
                     "gate, none of the comparisons below are reportable.\n")
    else:
        L.append("`A0_repeat` missing — bracket drift **cannot be checked**, so "
                 "no comparison below is reportable as a headline result.\n")

    # ---- Per-arm table ----------------------------------------------------
    L.append("## Arms\n")
    L.append("| arm | transport | prefill backend | TTFT p50 | CV | TPOT p50 | engagement |")
    L.append("|---|---|---|---|---|---|---|")
    for arm_id in ("A0_default", "A1_disabled", "A2_tcp", "A3_bcg", "A4_ipc",
                   "A5_ipc_best", "V0_vllm", "A0_repeat"):
        r = arms.get(arm_id)
        if r is None:
            L.append(f"| `{arm_id}` | — | — | — | — | — | *not run* |")
            continue
        e = r.get("engagement") or {}
        verdict = e.get("engagement", "n/a")
        L.append(
            f"| `{arm_id}` | {e.get('resolved_mm_transport') or r.get('requested_transport') or '—'} "
            f"| {e.get('resolved_prefill_backend') or r.get('requested_prefill_backend') or '—'} "
            f"| {fmt_ms(r.get('ttft_p50_median'))} "
            f"| {r.get('ttft_p50_cv_pct', '—')}% "
            f"| {fmt_ms(r.get('tpot_p50_median'))} "
            f"| {'**VERIFIED**' if verdict == 'VERIFIED' else '**UNVERIFIED**'} |")
    L.append("")
    L.append("Arms marked UNVERIFIED are excluded from every comparison below. "
             "Their numbers are shown only so the exclusion can be audited.\n")

    for arm_id in ("A0_default", "A1_disabled", "A2_tcp", "A3_bcg", "A4_ipc",
                   "A5_ipc_best", "V0_vllm", "A0_repeat"):
        r = arms.get(arm_id)
        if r is not None and not verified(r):
            L.append(f"- `{arm_id}` — {eng_reason(r)}")
    L.append("")

    # ---- Questions --------------------------------------------------------
    L.append("## The four questions, answered separately\n")
    rows = list(QUESTIONS)
    best_graph = None
    for cand in ("A2_tcp", "A3_bcg"):
        r = arms.get(cand)
        if verified(r) and r.get("ttft_p50_median") is not None:
            if best_graph is None or r["ttft_p50_median"] < arms[best_graph]["ttft_p50_median"]:
                best_graph = cand
    if best_graph:
        rows.append(("Qc", "Do the two levers compose?", "A5_ipc_best", best_graph,
                     "Transport and graph coverage are orthogonal knobs; this asks "
                     "whether stacking them adds up or interferes."))

    for qid, title, arm_id, base_id, why in rows:
        arm, base = arms.get(arm_id), arms.get(base_id)
        L.append(f"### {qid} — {title}\n")
        L.append(f"{why}\n")
        if not verified(arm) or not verified(base):
            bad = arm_id if not verified(arm) else base_id
            L.append(f"**Unanswered.** `{bad}` did not verify: {eng_reason(arms.get(bad))}.\n")
            continue
        d, verdict = compare(arm, base)
        L.append(f"`{arm_id}` {fmt_ms(arm['ttft_p50_median'])} vs `{base_id}` "
                 f"{fmt_ms(base['ttft_p50_median'])} → **{d:+.2f}%** — {verdict}.\n")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n".join(L) + "\n")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
