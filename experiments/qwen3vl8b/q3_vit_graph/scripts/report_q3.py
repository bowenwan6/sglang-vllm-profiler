#!/usr/bin/env python3
"""Q3 — turn results/*.json into results/q3_report.md (runs on the node or on the Mac).

Reads whatever exists: preflight.json, parity.json, pilot.json, sweep.json, mixed.json.
Implements the pre-registered rules of PLAN.md as amended on 2026-09-29:
  H1  per size: tol = max(1 ms, 0.25*|pred|, 2*SE); non-informative if 2*SE > |pred| + 1 ms;
      supported if >= 4 informative sizes with at most 1 miss, the text control stays inside
      the 3.6 % floor, the saving is non-increasing with patch count (1 ms slack) and 1080p is
      inside the floor. Only workloads whose cells are all VERIFIED count.
  H2  at 256x256: encoder >= 40 % of TTFT and un-overlapped >= 60 % of the encoder call;
      from 720p: G_v > U*.
  H3  mean TTFT of the graph arm worse than eager under mixed shapes; hit rate by key ~ 45 %.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results"
H1_ABS_MS, H1_REL = 1.0, 0.25
FLOOR_PCT = 3.6           # v3's resolution floor
SHAPE_SLACK_MS = 1.0
IMAGE_ORDER = ["R1_256", "R2_360p", "R3_512", "R4_640", "R5_720p", "R6_1080p"]


def load(p: Path) -> Optional[Dict[str, Any]]:
    return json.loads(p.read_text()) if p.exists() else None


def f(x: Any, nd: int = 2, suffix: str = "") -> str:
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}{suffix}"
    return str(x)


def g(d: Optional[Dict[str, Any]], *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def h1_rows(preds: Dict[str, Any], swp: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = []
    for wid in IMAGE_ORDER:
        w = (swp.get("workloads") or {}).get(wid)
        if not w:
            continue
        pred = g(preds, wid, "pred_point_ms")
        lo, hi = g(preds, wid, "pred_lo_ms"), g(preds, wid, "pred_hi_ms")
        m, se = w.get("saving_ms_median"), w.get("saving_ms_se")
        row = {"wid": wid, "patches": w.get("expected_patches"), "pred": pred, "lo": lo, "hi": hi,
               "measured": m, "se": se, "n": w.get("n_blocks"), "verified": w.get("all_verified")}
        if pred is None or m is None:
            row.update(tol=None, informative=False, ok=None, reason="no prediction or no measurement")
        else:
            se_v = se if se is not None else 0.0
            tol = max(H1_ABS_MS, H1_REL * abs(pred), 2 * se_v)
            informative = not (2 * se_v > abs(pred) + 1.0)
            row.update(tol=round(tol, 3), informative=informative, err=round(m - pred, 3),
                       ok=(abs(m - pred) <= tol), in_interval=(lo is not None and hi is not None and lo - tol <= m <= hi + tol))
        rows.append(row)
    return rows


def h1_verdict(rows: List[Dict[str, Any]], swp: Dict[str, Any]) -> Dict[str, Any]:
    usable = [r for r in rows if r.get("verified") and r.get("ok") is not None]
    inform = [r for r in usable if r["informative"]]
    misses = [r["wid"] for r in inform if not r["ok"]]
    ctrl = (swp.get("workloads") or {}).get("R0_text") or {}
    ctrl_ok = ctrl.get("effect_pct_median") is not None and abs(ctrl["effect_pct_median"]) <= FLOOR_PCT
    # shape: saving non-increasing with patch count (1 ms slack), across verified sizes
    by_p = sorted((r for r in usable if r["measured"] is not None), key=lambda r: r["patches"])
    shape_ok = all(by_p[i + 1]["measured"] <= by_p[i]["measured"] + SHAPE_SLACK_MS for i in range(len(by_p) - 1)) if by_p else False
    big = (swp.get("workloads") or {}).get("R6_1080p") or {}
    big_ok = big.get("effect_pct_median") is not None and abs(big["effect_pct_median"]) <= FLOOR_PCT
    conds = {"informative_sizes>=4": len(inform) >= 4, "misses<=1": len(misses) <= 1,
             "text_control_inside_floor": ctrl_ok, "saving_non_increasing": shape_ok, "1080p_inside_floor": big_ok}
    return {"verdict": "SUPPORTED" if all(conds.values()) else "NOT SUPPORTED", "conditions": conds,
            "informative": [r["wid"] for r in inform], "misses": misses,
            "non_informative": [r["wid"] for r in usable if not r["informative"]]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    R = a.results
    out = a.out or (R / "q3_report.md")
    pre, par, pil, swp, mix = (load(R / n) for n in
                               ("preflight.json", "parity.json", "pilot.json", "sweep.json", "mixed.json"))
    verdicts: Dict[str, Any] = {}
    L: List[str] = []
    L.append("# Q3 — ViT CUDA graph on Qwen3-VL-8B: results\n")
    L.append("Generated from `results/*.json`. Design and pre-registered predictions: [`PLAN.md`](../PLAN.md) "
             "(2026-09-29 amendment applies).\n")

    # ---- stack
    if pre:
        L.append("## Stack\n")
        L.append(f"- sglang `{pre.get('sglang_version')}` @ `{str(pre.get('sglang_commit', ''))[:12]}` "
                 f"(measurement patch applied: {pre.get('patch_applied')}), torch `{pre.get('torch')}`")
        L.append(f"- GPU: {pre.get('gpu')} ({f(pre.get('gpu_total_mib'), 0)} MiB); snapshot `{Path(str(pre.get('snapshot'))).name}`; "
                 f"profiler repo `{pre.get('profiler_branch')}` @ `{pre.get('profiler_commit')}`; output length {pre.get('output_len')}")
        L.append(f"- server: `{pre.get('server_cmd')}`\n")

    # ---- parity
    L.append("## Parity (both arms encode the same fixtures)\n")
    if par:
        rows = g(par, "dumps", "rows") or []
        pj = par.get("parity_judge")
        if pj:
            L.append(f"Pre-registered verdict **{par.get('verdict_preregistered')}**; post-hoc downstream rule "
                     f"(approved by {pj.get('approved_by')}): on = **{pj['arms']['on']['verdict']}**, "
                     + ", ".join(f"{k} = {v['verdict']}" for k, v in pj["arms"].items() if k != "on")
                     + f"; rule: {pj['arms']['on']['rule']}\n")
        L.append(f"**{par.get('verdict')}** — encoder-output relative error per image fixture: "
                 f"{[round(r.get('rel_fro') or -1, 4) for r in rows]} (tolerance {par.get('tolerance_rel_fro')}); "
                 f"reasons: {par.get('reasons')}")
        tx = par.get("text") or {}
        div = {k: v for k, v in tx.items() if not v.get("identical")}
        L.append(f"- greedy text: {len(tx) - len(div)}/{len(tx)} fixtures identical; divergences: "
                 + (", ".join(f"`{k}` at token {v.get('first_divergence')} (off-arm margin {f(v.get('off_margin_nat'))} nat, "
                              f"{'benign' if v.get('benign') else 'not benign'})" for k, v in div.items()) or "none"))
        for k, label in (("dumps_off_rot_vs_off", "eager with the unfused rotary vs eager (implementation noise floor)"),
                         ("dumps_on_vs_off_rot", "graph arm vs eager with the unfused rotary")):
            if par.get(k):
                L.append(f"- {label}: rel_fro {[round(r.get('rel_fro') or -1, 4) for r in par[k].get('rows') or []]}")
        if par.get("dumps_default_interp_vs_off"):
            r2 = par["dumps_default_interp_vs_off"].get("rows") or []
            L.append(f"- graph arm with the **default** interpolation flag vs eager: rel_fro "
                     f"{[round(r.get('rel_fro') or -1, 4) for r in r2]} (upstream-issue evidence)")
        L.append("")
    else:
        L.append("not run\n")

    # ---- pilot
    preds = (pil or {}).get("predictions") or {}
    if pil:
        P = pil["params"]
        L.append("## Pilot: the eager encoder's time budget and the amended H1 prediction\n")
        L.append(f"Servers: {P.get('servers')}. Per workload: {P['n_warm']} warmup, {P['n_profile']} profiled, "
                 f"{P['n_measure']} measured requests; c=1. W_v is the unprofiled `VIT_TIMING` CPU wall, G_v the "
                 f"trace's GPU busy time, U* = W_v − G_v, dG_rot the extra GPU time of the unfused rotary, "
                 f"overlap = min(G_v, max(0, W_l − G_l)) from the sync-free eager profile; r = {P['replay_residual_ms']} ms. "
                 f"Prediction: lo = U* − r − dG_rot, point = lo + overlap, hi = W_v − r − dG_rot.\n")
        L.append("| workload | patches | W_v | G_v | **U\\*** | infl. | dG_rot | overlap | launches off→on | graph launches on | "
                 "**pred lo / point / hi** | pilot ΔTTFT | TTFT off | TTFT on | verified |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for wid, p in preds.items():
            L.append(f"| `{wid}` | {f(p.get('patches'), 0)} | {f(p.get('W_v_ms'))} | {f(p.get('G_v_ms'))} | "
                     f"**{f(p.get('U_star_ms'))}** | {f(p.get('profiler_inflation_ms'))} | {f(p.get('dG_rot_ms'))} | "
                     f"{f(p.get('overlap_ms'))} | {f(p.get('vit_n_launches_off'), 0)}→{f(p.get('vit_n_launches_on'), 0)} | "
                     f"{f(p.get('vit_n_graph_launches_on'), 0)} | **{f(p.get('pred_lo_ms'))} / {f(p.get('pred_point_ms'))} / "
                     f"{f(p.get('pred_hi_ms'))}** | {f(p.get('pilot_gain_ms'))} | {f(p.get('ttft_off_p50'))} | "
                     f"{f(p.get('ttft_on_p50'))} | {f(p.get('verified'))} |")
        gt = pil.get("gate_g1") or {}
        L.append(f"\nGate G1: **{gt.get('verdict')}** {gt.get('reasons') or ''} warnings: {gt.get('warnings') or 'none'}; "
                 f"output-length check: {pil.get('output_len_check')}\n")

        # decomposition (critical path) — eager arm
        L.append("### TTFT decomposition, eager arm (critical path, medians)\n")
        L.append("prefill step = `Q3_STEP_EXTEND` critical path from the sync-free profile; ViT = `VIT_TIMING` GPU span "
                 "(encoder critical path); LM = step − ViT; outside forward = client TTFT − queue − step (HTTP, base64/PNG "
                 "decode, image processor, tokenisation, feature transport, first-token streaming; random-content PNGs do "
                 "not compress, so this term carries a benchmark artifact that grows with resolution).\n")
        L.append("| workload | client TTFT | queue | prefill step (crit) | ViT (crit) | of which un-overlapped U\\* | LM | outside forward |")
        L.append("|---|---|---|---|---|---|---|---|")
        h2: Dict[str, Any] = {}
        for wid, cell in (pil.get("cells") or {}).items():
            o = cell.get("off") or {}
            p = preds.get(wid) or {}
            ttft, q = g(o, "bench", "ttft_p50"), g(o, "req_times", "queue_ms_p50")
            step = p.get("prefill_step_crit_ms") or g(o, "trace", "Q3_STEP_EXTEND", "crit_ms_p50")
            vit = p.get("vit_gpu_span_off_ms") or g(o, "timings", "gpu_span_ms_p50")
            lm = (step - vit) if (step is not None and vit is not None) else None
            outside = (ttft - (q or 0) - step) if (ttft is not None and step is not None) else None
            L.append(f"| `{wid}` | {f(ttft)} | {f(q)} | {f(step)} | {f(vit)} | {f(p.get('U_star_ms'))} | {f(lm)} | {f(outside)} |")
            if wid in IMAGE_ORDER and ttft and vit is not None:
                h2[wid] = {"vit_share_of_ttft": round(vit / ttft, 3),
                           "unoverlapped_share_of_vit": (round(p["U_star_ms"] / p["W_v_ms"], 3)
                                                         if p.get("U_star_ms") is not None and p.get("W_v_ms") else None),
                           "G_v_gt_U": (p.get("G_v_ms") is not None and p.get("U_star_ms") is not None
                                        and p["G_v_ms"] > p["U_star_ms"])}
        L.append("")
        L.append("### H2 — what the fixed cost is made of\n")
        c256 = h2.get("R1_256") or {}
        conds = {
            "256²: encoder ≥ 40 % of TTFT": (c256.get("vit_share_of_ttft") or 0) >= 0.40,
            "256²: un-overlapped ≥ 60 % of the encoder call": (c256.get("unoverlapped_share_of_vit") or 0) >= 0.60,
            "720p and 1080p: G_v > U* (compute-dominated)": all((h2.get(w) or {}).get("G_v_gt_U") for w in ("R5_720p", "R6_1080p")),
        }
        for k, v in conds.items():
            L.append(f"- {'✅' if v else '❌'} {k}")
        L.append(f"\nshares: {json.dumps(h2)}\n")
        verdicts["H2"] = {"verdict": "SUPPORTED" if all(conds.values()) else "NOT SUPPORTED", "conditions": conds}
        L.append(f"**H2 {verdicts['H2']['verdict']}.**\n")

    # ---- sweep + H1
    if swp and swp.get("workloads"):
        L.append("## Sweep: measured TTFT effect (A/B/B/A blocks, one server per cell)\n")
        L.append("| workload | patches | blocks | TTFT off p50 | TTFT on p50 | saving (median) | SE | effect | paired spread | gate | verified |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for wid, w in swp["workloads"].items():
            r = w["blocks"]
            L.append(f"| `{wid}` | {w.get('expected_patches')} | {w.get('n_blocks')} | "
                     f"{f(statistics.median(x['off_p50'] for x in r))} ms | "
                     f"{f(statistics.median(x['on_p50'] for x in r))} ms | **{f(w['saving_ms_median'], 2, ' ms')}** | "
                     f"{f(w.get('saving_ms_se'))} | **{w['effect_pct_median']:+.2f}%** | {f(w.get('spread_pp'))} pp | "
                     f"{w['gate']} | {f(w.get('all_verified'))} |")
        L.append("")
        L.append("## H1: does the eager trace predict the graph's gain?\n")
        L.append(f"tol = max({H1_ABS_MS} ms, {int(H1_REL * 100)} % of |pred|, 2·SE); non-informative if 2·SE > |pred| + 1 ms. "
                 f"Supported if ≥ 4 informative sizes with ≤ 1 miss, text control inside the {FLOOR_PCT} % floor, "
                 f"saving non-increasing with patches (±{SHAPE_SLACK_MS} ms) and 1080p inside the floor.\n")
        rows = h1_rows(preds, swp)
        L.append("| workload | patches | pred lo / point / hi | measured saving | SE | tol | error | informative | in interval | pass |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            L.append(f"| `{r['wid']}` | {r['patches']} | {f(r['lo'])} / {f(r['pred'])} / {f(r['hi'])} | {f(r['measured'])} ms | "
                     f"{f(r['se'])} | {f(r.get('tol'))} | {f(r.get('err'))} | {f(r.get('informative'))} | "
                     f"{f(r.get('in_interval'))} | {'✅' if r.get('ok') else ('❌' if r.get('ok') is False else '—')}"
                     f"{'' if r.get('verified') else ' (UNVERIFIED, excluded)'} |")
        ctrl = swp["workloads"].get("R0_text")
        if ctrl:
            L.append(f"\nText control `R0_text`: {ctrl['effect_pct_median']:+.2f}% "
                     f"({'inside' if abs(ctrl['effect_pct_median']) <= FLOOR_PCT else 'OUTSIDE'} the {FLOOR_PCT} % floor).")
        v = h1_verdict(rows, swp)
        verdicts["H1"] = v
        L.append(f"\n**H1 {v['verdict']}** — conditions: {v['conditions']}; informative: {v['informative']}; "
                 f"misses: {v['misses']}; non-informative: {v['non_informative']}\n")

    # ---- mixed / H3
    if mix and mix.get("arms"):
        L.append("## Mixed resolutions: exact-shape keys under realistic variety (H3)\n")
        s = mix.get("summary") or {}
        n, o = mix["arms"].get("on") or {}, mix["arms"].get("off") or {}
        P = mix.get("params") or {}
        L.append(f"{P.get('prompts')} requests, heights 256–720 and widths 256–1280 px drawn uniformly with a fixed seed, "
                 f"so both arms see the same sequence. The graph key is the patch count, so distinct 32-px grids with the "
                 f"same product share a graph: {P.get('expected_distinct_keys')} captures and a hit rate of "
                 f"{P.get('expected_hit_rate')} were expected.\n")
        L.append("| | off | on |")
        L.append("|---|---|---|")
        L.append(f"| **mean TTFT** (H3 metric) | {f(s.get('ttft_mean_off'))} ms | {f(s.get('ttft_mean_on'))} ms "
                 f"(penalty {f(s.get('mean_penalty_ms'))} ms) |")
        L.append(f"| TTFT p50 / p99 | {f(s.get('ttft_p50_off'))} / {f(s.get('ttft_p99_off'))} ms | "
                 f"{f(s.get('ttft_p50_on'))} / {f(s.get('ttft_p99_on'))} ms |")
        L.append(f"| graph captures / requests | — | {n.get('n_captures')} / {n.get('n_requests')} (hit rate by key {f(s.get('hit_rate_by_key'), 3)}) |")
        L.append(f"| capture cost p50 / max / total | — | {f(n.get('capture_ms_p50'))} / {f(n.get('capture_ms_max'))} / {f(n.get('capture_ms_total'), 0)} ms |")
        L.append(f"| TTFT, first-seen shapes (p50) | — | {f(n.get('ttft_first_seen_p50'))} ms, penalty vs off on the same requests {f(s.get('first_seen_penalty_ms_p50'))} ms |")
        L.append(f"| TTFT, repeated shapes (p50) | — | {f(n.get('ttft_repeat_p50'))} ms, gain vs off on the same requests {f(s.get('repeat_gain_ms_p50'))} ms |")
        L.append(f"| GPU memory end / peak | {f(o.get('gpu_used_mib_end'), 0)} / {f(o.get('gpu_used_mib_peak'), 0)} MiB | "
                 f"{f(n.get('gpu_used_mib_end'), 0)} / {f(n.get('gpu_used_mib_peak'), 0)} MiB (guard tripped: {n.get('mem_guard_tripped')}) |")
        conds = {"mean TTFT worse on the graph arm": (s.get("mean_penalty_ms") or 0) > 0,
                 "hit rate within ±5 pp of 45 %": (s.get("hit_rate_by_key") is not None and abs(s["hit_rate_by_key"] - 0.45) <= 0.05),
                 "repeated shapes still gain": (s.get("repeat_gain_ms_p50") or 0) > 0}
        for k, v in conds.items():
            L.append(f"- {'✅' if v else '❌'} {k}")
        verdicts["H3"] = {"verdict": "SUPPORTED" if all(conds.values()) else "NOT SUPPORTED", "conditions": conds}
        L.append(f"\n**H3 {verdicts['H3']['verdict']}.**\n")

    if verdicts:
        L.append("## Verdicts\n")
        for k, v in verdicts.items():
            L.append(f"- **{k}: {v['verdict']}**")
        L.append("")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    (out.parent / "verdicts.json").write_text(json.dumps(verdicts, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
