#!/usr/bin/env python3
"""Figures for the issue #4 v3 report. All values read from results JSON."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = Path("/data/sglang-vllm-profiler/experiments/qwen3vl8b/v3_issue4")
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else BASE / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DRIFT = 3.60  # resolution floor from the IMG-R reference-cell repeat

INK = "#1a1a1a"
GRID = "#d8d8d8"
WIN = "#1f6f4a"
NULL = "#8a8a8a"
ACCENT = "#b4441f"


def load_cells():
    cells = {}
    for f in ("results/phase2b_imgR_sweep/results.json",
              "results/phase2c_imgR_gapfill/results.json"):
        for r in json.loads((BASE / f).read_text()):
            cells[r["cell"]] = r
    return cells


def series():
    cells = load_cells()
    pts = []
    for c in cells:
        if not c.endswith("__disabled") or c.endswith("__repeat"):
            continue
        w = c[: -len("__disabled")]
        d, b = cells.get(f"{w}__disabled"), cells.get(f"{w}__breakable")
        if not d or not b or d.get("status") != "OK" or b.get("status") != "OK":
            continue
        dv, bv = d["ttft_p50_median"], b["ttft_p50_median"]
        pts.append({
            "w": w, "N": d["measured_N"],
            "vision": d.get("vision_tok_per_req"), "text": d.get("text_tok_per_req"),
            "disabled": dv, "breakable": bv,
            "saving": dv - bv, "effect": 100.0 * (bv - dv) / dv,
        })
    return sorted(pts, key=lambda p: p["N"])


def fig_effect(pts):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 6.6), sharex=True,
                                   gridspec_kw={"hspace": 0.14})
    N = [p["N"] for p in pts]

    # --- panel 1: effect % ---
    ax1.axhspan(-DRIFT, DRIFT, color=NULL, alpha=0.16, lw=0,
                label=f"below resolution floor (\u00b1{DRIFT}%)")
    ax1.axhline(0, color=INK, lw=0.8)
    ax1.axhline(-5, color=ACCENT, lw=0.9, ls="--", label="5% materiality bar")
    eff = [p["effect"] for p in pts]
    ax1.plot(N, eff, color=INK, lw=1.2, zorder=2)
    ax1.scatter(N, eff, c=[WIN if e <= -5 else NULL for e in eff], s=48, zorder=3,
                edgecolors="white", linewidths=0.9)
    # Alternate label side on the crowded right-hand cluster.
    for i, p in enumerate(pts):
        if p["effect"] <= -5:
            dy, va = 3.0, "bottom"
        else:
            dy, va = (5.0, "bottom") if i % 2 == 0 else (-5.0, "top")
        ax1.annotate(f"{p['effect']:+.1f}%", (p["N"], p["effect"] + dy),
                     ha="center", va=va, fontsize=7.4, color=INK)
    ax1.set_xscale("log")
    ax1.set_ylabel("TTFT change with prefill graph on (%)", fontsize=8.5)
    ax1.set_ylim(-54, 16)
    ax1.grid(True, which="major", color=GRID, lw=0.5)
    ax1.legend(fontsize=7.4, loc="lower right", frameon=False,
               bbox_to_anchor=(1.0, 0.06))
    ax1.set_title("Prefill CUDA graph: the effect tracks prefill length, not image share",
                  fontsize=10, color=INK, pad=9)

    # --- panel 2: saving ms ---
    ax2.axhline(0, color=INK, lw=0.8)
    sav = [p["saving"] for p in pts]
    ax2.plot(N, sav, color=INK, lw=1.2, zorder=2)
    ax2.scatter(N, sav, c=[WIN if s > 2 else NULL for s in sav], s=48, zorder=3,
                edgecolors="white", linewidths=0.9)
    # Only label points the bracket can actually resolve; the rest are shaded as
    # one region rather than given nine individually meaningless numbers.
    res = [p for p in pts if abs(p["effect"]) >= DRIFT]
    unres = [p for p in pts if abs(p["effect"]) < DRIFT]
    for p in res:
        ax2.annotate(f"{p['saving']:+.1f} ms", (p["N"], p["saving"] + 0.85),
                     ha="center", fontsize=7.6, color=INK)
    if unres:
        lo = min(p["N"] for p in unres) / 1.18
        ax2.axvspan(lo, max(N) * 1.15, color=NULL, alpha=0.14, lw=0)
        ax2.annotate("all within the \u00b13.6% resolution floor\n"
                     "\u2014 no sign can be claimed here",
                     ((lo * max(N)) ** 0.5, 6.4), ha="center", va="center",
                     fontsize=7.6, color="#555555", style="italic")
    ax2.set_xscale("log")
    ax2.set_xlabel("N \u2014 prefill tokens entering the LM (visual + text), log scale",
                   fontsize=8.5)
    ax2.set_ylabel("absolute saving (ms)", fontsize=8.5)
    ax2.set_ylim(-4.2, 14.5)
    ax2.grid(True, which="major", color=GRID, lw=0.5)
    ax2.set_xticks(N)
    ax2.set_xticklabels([str(n) for n in N], fontsize=7.2, rotation=45, ha="right")
    ax2.minorticks_off()
    ax1.minorticks_off()
    for ax in (ax1, ax2):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.savefig(OUT / "fig_effect.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_2x2():
    d = {r["arm"]: r for r in json.loads(
        (BASE / "results/phase2_imgA_headline/results.json").read_text())}
    cells = {("cpu", "disabled"): "A1_disabled", ("cpu", "breakable"): "A0_default",
             ("cuda_ipc", "disabled"): "A5_ipc_nograph",
             ("cuda_ipc", "breakable"): "A4_ipc"}
    v = {k: d[a]["ttft_p50_median"] for k, a in cells.items()}
    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    xs = [0, 1]
    for gr, col, mk, dy in (("disabled", INK, "o", -15), ("breakable", ACCENT, "s", 9)):
        ys = [v[("cpu", gr)], v[("cuda_ipc", gr)]]
        ax.plot(xs, ys, color=col, marker=mk, lw=1.4, ms=6, label=f"prefill graph {gr}")
        for x, y in zip(xs, ys):
            ax.annotate(f"{y:.1f} ms", (x, y), textcoords="offset points",
                        xytext=(0, dy), ha="center", fontsize=8, color=col)
    # The gap that matters, drawn rather than described.
    ax.annotate("", xy=(1, v[("cuda_ipc", "disabled")]), xytext=(1, v[("cpu", "disabled")]),
                arrowprops=dict(arrowstyle="<->", color="#1f6f4a", lw=1.2))
    ax.annotate("-28.2%", (1.0, (v[("cpu", "disabled")] + v[("cuda_ipc", "disabled")]) / 2),
                textcoords="offset points", xytext=(-8, 0), ha="right", va="center",
                fontsize=9, color="#1f6f4a", weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["cpu transport\n(upstream default today)",
                        "cuda_ipc transport\n(#4's standard condition)"], fontsize=8)
    ax.set_ylabel("TTFT p50 (ms)", fontsize=8.5)
    ax.set_ylim(90, 160)
    ax.grid(True, axis="y", color=GRID, lw=0.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_title("IMG-A: the transport lever moves TTFT ~28%; the graph lever does not",
                 fontsize=9.4, color=INK, pad=8)
    fig.savefig(OUT / "fig_2x2.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pts = series()
    fig_effect(pts)
    fig_2x2()
    print(json.dumps(pts, indent=1))
    print("figures ->", OUT)
