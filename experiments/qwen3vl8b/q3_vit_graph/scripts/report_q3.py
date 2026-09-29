#!/usr/bin/env python3
"""Q3 — turn results/*.json into results/q3_report.md (runs on the node or on the Mac).

Reads whatever exists: preflight.json, parity.json, pilot.json, sweep.json, mixed.json.
The H1 test compares the sweep's measured saving per size with the pilot's trace-based
prediction (eager un-overlapped CPU time minus the pre-registered residual).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
DEFAULT_RESULTS = HERE.parent / "results"
H1_ABS_MS, H1_REL = 1.0, 0.25
FLOOR_PCT = 3.6   # v3's resolution floor; used only to read the text control


def load(p: Path) -> Optional[Dict[str, Any]]:
    return json.loads(p.read_text()) if p.exists() else None


def f(x: Any, nd: int = 2, suffix: str = "") -> str:
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}{suffix}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    R = a.results
    out = a.out or (R / "q3_report.md")
    pre, par, pil, swp, mix = (load(R / n) for n in
                               ("preflight.json", "parity.json", "pilot.json", "sweep.json", "mixed.json"))
    L: List[str] = []
    L.append("# Q3 — ViT CUDA graph on Qwen3-VL-8B: results\n")
    L.append("Generated from `results/*.json`. Design and pre-registered predictions: [`PLAN.md`](../PLAN.md).\n")

    # ---- stack
    if pre:
        L.append("## Stack\n")
        L.append(f"- sglang `{pre.get('sglang_version')}` @ `{str(pre.get('sglang_commit', ''))[:12]}` "
                 f"(+ measurement patch: {pre.get('patch_applied')}), torch `{pre.get('torch')}`")
        L.append(f"- GPU: {pre.get('gpu')}; snapshot `{Path(str(pre.get('snapshot'))).name}`; "
                 f"profiler repo `{pre.get('profiler_commit')}`")
        L.append(f"- server: `{pre.get('server_cmd')}`\n")

    # ---- parity
    L.append("## Parity (both arms, greedy, identical fixtures)\n")
    if par:
        L.append(f"**{par['verdict']}** — differing fixtures: {par.get('differing_fixtures')}, "
                 f"errored: {par.get('errored_fixtures')}; on-arm captures: "
                 f"{len((par['arms'].get('on') or {}).get('captures') or [])} (two image shapes sent).\n")
    else:
        L.append("not run\n")

    # ---- pilot / decomposition
    preds = (pil or {}).get("predictions") or {}
    if pil:
        L.append("## Pilot: where the encoder's time goes (eager arm) and what the graph should recover\n")
        L.append(f"Per workload: {pil['params']['n_warm']} warmup, {pil['params']['n_profile']} profiled, "
                 f"{pil['params']['n_measure']} measured requests; c=1; `Q3_VIT_TIMING=1` (adds one sync per "
                 f"request on both arms). Residual pre-registered at {pil['params']['replay_residual_ms']} ms.\n")
        L.append("| workload | patches | ViT wall (eager) | ViT GPU busy | un-overlapped | launches | "
                 "ViT wall (graph) | graph launches | predicted gain | pilot ΔTTFT | TTFT off | TTFT on | verified |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for wid, p in preds.items():
            L.append(f"| `{wid}` | {f(p.get('patches'), 0)} | {f(p.get('vit_cpu_wall_off_ms'))} ms | "
                     f"{f(p.get('vit_gpu_busy_off_ms'))} ms | **{f(p.get('vit_unoverlapped_off_ms'))} ms** | "
                     f"{f(p.get('vit_n_launches_off'), 0)} | {f(p.get('vit_cpu_wall_on_ms'))} ms | "
                     f"{f(p.get('vit_n_graph_launches_on'), 0)} | **{f(p.get('pred_gain_ms'))} ms** | "
                     f"{f(p.get('pilot_gain_ms'))} ms | {f(p.get('ttft_off_p50'))} | {f(p.get('ttft_on_p50'))} | "
                     f"{'yes' if p.get('verified') else 'NO'} |")
        g = pil.get("gate_g1") or {}
        L.append(f"\nGate G1: **{g.get('verdict')}** {g.get('reasons') or ''}\n")

        # decomposition of TTFT on the eager arm
        L.append("### TTFT decomposition, eager arm (medians)\n")
        L.append("`client-side` = client TTFT − queue − prefill step; it holds HTTP, base64/PNG decode, the image "
                 "processor, tokenisation, feature transport and first-token streaming. `LM prefill` = prefill "
                 "step − ViT. Random-content PNGs do not compress, so the client-side term carries a benchmark "
                 "artifact that grows with resolution.\n")
        L.append("| workload | client TTFT | queue | prefill step | ViT | of which un-overlapped | LM prefill | client-side |")
        L.append("|---|---|---|---|---|---|---|---|")
        for wid, cell in (pil.get("cells") or {}).items():
            o = cell.get("off") or {}
            b, rt, tr = o.get("bench") or {}, o.get("req_times") or {}, o.get("trace") or {}
            vit, step = (tr.get("Q3_VIT_FORWARD") or {}), (tr.get("Q3_STEP_EXTEND") or {})
            ttft, q = b.get("ttft_p50"), rt.get("queue_ms_p50")
            sw, vw, vu = step.get("cpu_wall_ms_p50"), vit.get("cpu_wall_ms_p50"), vit.get("unoverlapped_ms_p50")
            lm = (sw - vw) if (sw is not None and vw is not None) else (sw if vw is None else None)
            cs = (ttft - (q or 0) - sw) if (ttft is not None and sw is not None) else None
            L.append(f"| `{wid}` | {f(ttft)} | {f(q)} | {f(sw)} | {f(vw)} | {f(vu)} | {f(lm)} | {f(cs)} |")
        L.append("")

    # ---- sweep + H1
    if swp and swp.get("workloads"):
        L.append("## Sweep: measured TTFT effect (A/B/B/A blocks)\n")
        L.append("| workload | patches | TTFT off p50 | TTFT on p50 | saving | effect | paired spread | gate | verified |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for wid, w in swp["workloads"].items():
            r = w["blocks"]
            L.append(f"| `{wid}` | {w.get('expected_patches')} | "
                     f"{f(statistics.median(x['off_p50'] for x in r))} ms | "
                     f"{f(statistics.median(x['on_p50'] for x in r))} ms | **{f(w['saving_ms_median'], 2, ' ms')}** | "
                     f"**{w['effect_pct_median']:+.2f}%** | {f(w.get('spread_pp'))} pp | {w['gate']} | "
                     f"{'yes' if w.get('all_verified') else 'NO'} |")
        L.append("")
        L.append("## H1: does the eager trace predict the graph's gain?\n")
        L.append(f"Pass per size if |measured − predicted| ≤ max({H1_ABS_MS} ms, {int(H1_REL*100)}% of predicted).\n")
        L.append("| workload | predicted gain | measured saving | error | pass |")
        L.append("|---|---|---|---|---|")
        passes, tested = 0, 0
        for wid, w in swp["workloads"].items():
            if wid == "R0_text":
                continue
            p = (preds.get(wid) or {}).get("pred_gain_ms")
            m = w["saving_ms_median"]
            if p is None:
                L.append(f"| `{wid}` | — | {f(m)} ms | — | no prediction |")
                continue
            err = m - p
            ok = abs(err) <= max(H1_ABS_MS, H1_REL * p)
            tested += 1
            passes += int(ok)
            L.append(f"| `{wid}` | {f(p)} ms | {f(m)} ms | {err:+.2f} ms | {'✅' if ok else '❌'} |")
        ctrl = swp["workloads"].get("R0_text")
        if ctrl:
            L.append(f"\nText control `R0_text`: {ctrl['effect_pct_median']:+.2f}% "
                     f"({'inside' if abs(ctrl['effect_pct_median']) <= FLOOR_PCT else 'OUTSIDE'} the {FLOOR_PCT}% floor).")
        if tested:
            verdict = "SUPPORTED" if passes >= max(1, tested - 1) else "NOT SUPPORTED"
            L.append(f"\n**H1 {verdict}: {passes}/{tested} sizes within tolerance.**\n")

    # ---- mixed
    if mix and mix.get("arms"):
        L.append("## Mixed resolutions (exact-shape keys under realistic variety)\n")
        s = mix.get("summary") or {}
        n, o = mix["arms"].get("on") or {}, mix["arms"].get("off") or {}
        L.append(f"{mix['params']['prompts']} requests, resolutions drawn uniformly from 256×256 to 1280×720 with a "
                 f"fixed seed, so both arms see the same sequence.\n")
        L.append("| | off | on |")
        L.append("|---|---|---|")
        L.append(f"| TTFT p50 | {f(s.get('ttft_p50_off'))} ms | {f(s.get('ttft_p50_on'))} ms |")
        L.append(f"| TTFT p99 | {f(s.get('ttft_p99_off'))} ms | {f(s.get('ttft_p99_on'))} ms |")
        L.append(f"| graph captures / requests | — | {n.get('n_captures')} / {n.get('n_requests')} "
                 f"(hit rate {f(s.get('hit_rate'), 3)}) |")
        L.append(f"| capture cost p50 / total | — | {f(n.get('capture_ms_p50'))} ms / {f(n.get('capture_ms_total'), 0)} ms |")
        L.append(f"| TTFT, first-seen shapes p50 | {f(None)} | {f(n.get('ttft_first_seen_p50'))} ms "
                 f"(penalty vs off on the same requests: {f(s.get('first_seen_penalty_ms_p50'))} ms) |")
        L.append(f"| TTFT, repeated shapes p50 | — | {f(n.get('ttft_repeat_p50'))} ms "
                 f"(gain vs off on the same requests: {f(s.get('repeat_gain_ms_p50'))} ms) |")
        L.append(f"| GPU memory at end | {f(o.get('gpu_used_mib_end'), 0)} MiB | {f(n.get('gpu_used_mib_end'), 0)} MiB |")
        L.append("")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
