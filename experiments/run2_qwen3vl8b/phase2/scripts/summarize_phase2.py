#!/usr/bin/env python3
"""
run2 Phase 2 summarizer — reads all phase2/raw/ results and produces:
  - experiments/run2_qwen3vl8b/phase2/summary.md
  - experiments/run2_qwen3vl8b/phase2/selected_cases.md

Usage (from repo root):
    python3 experiments/run2_qwen3vl8b/phase2/scripts/summarize_phase2.py
"""

import json, statistics
from datetime import datetime, timezone
from pathlib import Path

LAB = Path("/data/sglang-vllm-profiler")
RAW = LAB / "experiments/run2_qwen3vl8b/phase2/raw"
PHASE2 = LAB / "experiments/run2_qwen3vl8b/phase2"
PHASE1_SUMMARY = {
    "caseA": {"sglang_p50": 61.8, "sglang_p95": 66.1, "sglang_p99": 66.4,
              "vllm_p50": 12.6, "vllm_p95": 17.9, "vllm_p99": 18.0},
    "caseB": {"sglang_p50": 66.7, "sglang_p95": 70.6, "sglang_p99": 71.6,
              "vllm_p50": 20.8, "vllm_p95": 25.4, "vllm_p99": 26.0, "vllm_cv": 114.6},
    "caseC": {"sglang_p50": 247.5, "vllm_p50": 187.9, "gap": 1.32},
    "caseD": {"sglang_p50": 253.0, "vllm_p50": 189.7, "gap": 1.33},
}
CASE_A_VARIANTS = ["default", "no_overlap", "stream8", "chunk_off", "chunk_64"]
CASE_B_VARIANTS = ["default", "chunk_off", "chunk_1024", "chunk_512"]
WARMUP_LEVELS = [30, 100, 300]


def percentile(data, q):
    d = sorted(v for v in data if v is not None)
    if not d:
        return None
    rank = (len(d) - 1) * q / 100.0
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(d):
        return d[lo]
    return d[lo] + frac * (d[lo + 1] - d[lo])


def cv_stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    cv = (statistics.stdev(vals) / med * 100) if (len(vals) >= 2 and med) else 0.0
    return med, cv


def load_bench_p50s(pattern_glob):
    """Load TTFT p50 values from all files matching glob pattern."""
    vals = []
    for f in sorted(RAW.glob(pattern_glob)):
        try:
            d = json.load(open(f))
            ttfts = [t * 1000 for t in d.get("ttfts", []) if t is not None]
            p50 = percentile(ttfts, 50)
            p95 = percentile(ttfts, 95)
            p99 = percentile(ttfts, 99)
            if p50:
                vals.append({"p50": p50, "p95": p95, "p99": p99, "file": f.name})
        except Exception:
            pass
    return vals


def load_json_safe(path):
    try:
        return json.load(open(path))
    except Exception:
        return {}


def ratio_str(a, b):
    if not a or not b or b == 0:
        return "N/A"
    r = a / b
    arrow = "↑" if r > 1.05 else ("↓" if r < 0.95 else "≈")
    return f"{r:.2f}× {arrow}"


def fmt(v, cv=None):
    if v is None:
        return "N/A"
    s = f"{v:.1f}"
    if cv is not None:
        s += f" (cv={cv:.1f}%{' ⚠' if cv > 5 else ''})"
    return s


def main():
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ===== Load raw results =====
    caseA_result = load_json_safe(RAW / "caseA_shaping_result.json")
    caseB_result = load_json_safe(RAW / "caseB_shaping_result.json")
    variance_results = load_json_safe(RAW / "variance_gate_results.json")
    vllm_recheck = load_json_safe(RAW / "vllm_recheck_results.json")

    # ===== Case A finalist bench data =====
    caseA_finalist_data = {}
    if caseA_result.get("finalist_summary"):
        for v, s in caseA_result["finalist_summary"].items():
            caseA_finalist_data[v] = s
    # Also load from raw bench files if available
    for v in CASE_A_VARIANTS:
        reps = load_bench_p50s(f"caseA_sglang_{v}_finalist_rep*.json")
        if reps:
            p50s = [r["p50"] for r in reps]
            med, cv = cv_stats(p50s)
            all_p95 = [r["p95"] for r in reps if r["p95"]]
            all_p99 = [r["p99"] for r in reps if r["p99"]]
            caseA_finalist_data[v] = {
                "median_p50_ms": med, "cv": cv,
                "median_p95_ms": statistics.median(all_p95) if all_p95 else None,
                "median_p99_ms": statistics.median(all_p99) if all_p99 else None,
            }

    # ===== Case B finalist bench data =====
    caseB_finalist_data = {}
    for v in CASE_B_VARIANTS:
        reps = load_bench_p50s(f"caseB_sglang_{v}_finalist_rep*.json")
        if reps:
            p50s = [r["p50"] for r in reps]
            med, cv = cv_stats(p50s)
            all_p95 = [r["p95"] for r in reps if r["p95"]]
            all_p99 = [r["p99"] for r in reps if r["p99"]]
            caseB_finalist_data[v] = {
                "median_p50_ms": med, "cv": cv,
                "median_p95_ms": statistics.median(all_p95) if all_p95 else None,
                "median_p99_ms": statistics.median(all_p99) if all_p99 else None,
            }
    if caseB_result.get("finalist_summary"):
        for v, s in caseB_result["finalist_summary"].items():
            if v not in caseB_finalist_data:
                caseB_finalist_data[v] = s

    # ===== vLLM recheck data =====
    vllm_B = vllm_recheck.get("caseB_longprefill", {})
    vllm_C = vllm_recheck.get("caseC_batched", {})

    # Load directly from bench files too
    vllm_B_reps = load_bench_p50s("caseB_vllm_w300_recheck_rep*.json")
    if vllm_B_reps:
        p50s = [r["p50"] for r in vllm_B_reps]
        med, cv = cv_stats(p50s)
        vllm_B["median_p50_ms"] = med
        vllm_B["cv_pct"] = cv
        vllm_B["bimodal_resolved"] = cv is not None and cv < 15.0
        vllm_B["ceiling_M"] = not vllm_B["bimodal_resolved"]

    vllm_C_reps = load_bench_p50s("caseC_vllm_w300_recheck_rep*.json")
    if vllm_C_reps:
        p50s = [r["p50"] for r in vllm_C_reps]
        med, cv = cv_stats(p50s)
        vllm_C["median_p50_ms"] = med
        vllm_C["cv_pct"] = cv

    # ===== Variance gate data =====
    caseC_var = variance_results.get("caseC_batched", {})
    caseD_var = variance_results.get("caseD_decode", {})

    # Fold in the W500 follow-up probe (SGLang only, GPU 1) if present.
    caseC_w500 = load_json_safe(RAW / "caseC_w500_result.json")
    if caseC_w500.get("ttft_p50_median_ms") is not None:
        caseC_var.setdefault("warmup_results", {})["500"] = {
            "reps": caseC_w500.get("ttft_p50_ms_per_rep", []),
            "median_p50_ms": caseC_w500.get("ttft_p50_median_ms"),
            "cv_pct": caseC_w500.get("ttft_p50_cv_pct"),
        }
        w500_cv = caseC_w500.get("ttft_p50_cv_pct")
        caseC_var["recommended_warmup"] = 500
        caseC_var["recommended_reps"] = 5
        caseC_var["gate_passed"] = (w500_cv is not None and w500_cv < 5.0)

    def var_for(case_var, warmup):
        wr = case_var.get("warmup_results", {})
        entry = wr.get(warmup) or wr.get(str(warmup))
        if entry:
            return entry.get("median_p50_ms"), entry.get("cv_pct")
        return None, None

    # ===== Winners =====
    caseA_winner = caseA_result.get("winner", "default")
    caseA_flags = caseA_result.get("winner_flags", [])
    caseB_winner = caseB_result.get("winner", "default")
    caseB_flags = caseB_result.get("winner_flags", [])

    caseC_rec_warmup = caseC_var.get("recommended_warmup", 300)
    caseD_rec_warmup = caseD_var.get("recommended_warmup", 300)
    caseC_rec_reps = caseC_var.get("recommended_reps", 3)
    caseD_rec_reps = caseD_var.get("recommended_reps", 3)
    caseC_gate = caseC_var.get("gate_passed", False)
    caseD_gate = caseD_var.get("gate_passed", False)

    # Phase 2 Case A winner metrics
    cA_w = caseA_finalist_data.get(caseA_winner, {})
    cA_p50 = cA_w.get("median_p50_ms")
    cA_cv = cA_w.get("cv")

    # Phase 2 Case B winner metrics
    cB_w = caseB_finalist_data.get(caseB_winner, {})
    cB_p50 = cB_w.get("median_p50_ms")
    cB_cv = cB_w.get("cv")

    # Phase 2 Case C variance result at recommended warmup
    cC_p50, cC_cv = var_for(caseC_var, caseC_rec_warmup)
    cD_p50, cD_cv = var_for(caseD_var, caseD_rec_warmup)

    # ===== Build summary.md =====
    S = []
    S.append("# run2 Phase 2 — Shaping / Variance Gate Summary")
    S.append("")
    S.append(f"Generated: {now}")
    S.append("")
    S.append("Active run `run2_qwen3vl8b` · GPU 7 · model Qwen3-VL-8B-Instruct @ 0c351dd · TP=1 · bf16.")
    S.append("")

    # Case A
    S.append("## Case A — Short latency (128→128, c=1) — Shaping")
    S.append("")
    S.append("### Screen results (1 rep each)")
    S.append("")
    S.append("| Variant | Flags | TTFT p50 (ms) |")
    S.append("|---------|-------|---------------|")
    for v in CASE_A_VARIANTS:
        p50_screen = caseA_result.get("screen_results", {}).get(v)
        flags_desc = {
            "default": "(none)", "no_overlap": "--disable-overlap-schedule",
            "stream8": "--stream-interval 8", "chunk_off": "--chunked-prefill-size -1",
            "chunk_64": "--chunked-prefill-size 64"
        }.get(v, "")
        p50_str = f"{p50_screen:.1f}" if p50_screen else "N/A"
        S.append(f"| {v} | `{flags_desc}` | {p50_str} |")
    S.append("")
    S.append("### Finalist results (3 reps each)")
    S.append("")
    S.append("| Variant | TTFT p50 median (ms) | CV | TTFT p95 | TTFT p99 |")
    S.append("|---------|---------------------|-----|----------|----------|")
    for v, d in caseA_finalist_data.items():
        med = d.get("median_p50_ms")
        cv = d.get("cv")
        p95 = d.get("median_p95_ms")
        p99 = d.get("median_p99_ms")
        S.append(f"| {v} | {fmt(med)} | {f'{cv:.1f}%' if cv is not None else 'N/A'} | {fmt(p95)} | {fmt(p99)} |")
    S.append("")
    S.append(f"**Winner**: `{caseA_winner}` (flags: `{'  '.join(caseA_flags) or 'none'}`)")
    S.append(f"**Phase 1 vLLM reference**: 12.6 ms")
    if cA_p50:
        S.append(f"**Phase 2 residual gap**: {ratio_str(cA_p50, 12.6)}")
    S.append("")
    S.append("---")
    S.append("")

    # Case B
    S.append("## Case B — Long prefill (2048→128, c=1) — Shaping")
    S.append("")
    S.append("### Screen results (1 rep each)")
    S.append("")
    S.append("| Variant | Flags | TTFT p50 (ms) |")
    S.append("|---------|-------|---------------|")
    for v in CASE_B_VARIANTS:
        p50_screen = caseB_result.get("screen_results", {}).get(v)
        flags_desc = {
            "default": "(none)", "chunk_off": "--chunked-prefill-size -1",
            "chunk_1024": "--chunked-prefill-size 1024", "chunk_512": "--chunked-prefill-size 512"
        }.get(v, "")
        p50_str = f"{p50_screen:.1f}" if p50_screen else "N/A"
        S.append(f"| {v} | `{flags_desc}` | {p50_str} |")
    S.append("")
    S.append("### Finalist results (3 reps each)")
    S.append("")
    S.append("| Variant | TTFT p50 median (ms) | CV | TTFT p95 | TTFT p99 |")
    S.append("|---------|---------------------|-----|----------|----------|")
    for v, d in caseB_finalist_data.items():
        med = d.get("median_p50_ms")
        cv = d.get("cv")
        p95 = d.get("median_p95_ms")
        p99 = d.get("median_p99_ms")
        S.append(f"| {v} | {fmt(med)} | {f'{cv:.1f}%' if cv is not None else 'N/A'} | {fmt(p95)} | {fmt(p99)} |")
    S.append("")
    S.append(f"**Winner**: `{caseB_winner}` (flags: `{'  '.join(caseB_flags) or 'none'}`)")
    S.append(f"**Phase 1 SGLang reference**: 66.7 ms (p50) · Phase 1 vLLM: 20.8 ms (cv=114.6% ⚠)")
    if cB_p50:
        vllm_B_p50 = vllm_B.get("median_p50_ms")
        ref = vllm_B_p50 or 20.8
        S.append(f"**Phase 2 residual gap (vs vLLM recheck)**: {ratio_str(cB_p50, ref)}")
    S.append("")
    S.append("---")
    S.append("")

    # Variance gate
    S.append("## Case C — Batched (512→128, c=16) — Variance Gate")
    S.append("")
    S.append("| Warmup | Reps | TTFT p50 median (ms) | CV | Gate |")
    S.append("|--------|------|---------------------|----|------|")
    caseC_levels = WARMUP_LEVELS + ([500] if "500" in (caseC_var.get("warmup_results") or {}) else [])
    for w in caseC_levels:
        med, cv = var_for(caseC_var, w)
        gate = "PASS ✅" if (cv is not None and cv < 5.0) else ("FAIL ⚠" if cv is not None else "N/A")
        n_reps = len((caseC_var.get("warmup_results") or {}).get(w, {}).get("reps") or
                     (caseC_var.get("warmup_results") or {}).get(str(w), {}).get("reps") or [])
        S.append(f"| {w} | {n_reps or 3} | {fmt(med)} | {f'{cv:.1f}%' if cv is not None else 'N/A'} | {gate} |")
    S.append("")
    S.append(f"**Recommended warmup for Phase 3**: {caseC_rec_warmup} · reps: {caseC_rec_reps}")
    S.append(f"**Gate passed**: {'YES ✅' if caseC_gate else 'NO ⚠ — W500 may be needed'}")
    if "500" in (caseC_var.get("warmup_results") or {}):
        S.append("")
        S.append("> **W500 follow-up probe (SGLang only, GPU 1, 5 reps):** TTFT p50 CV dropped to "
                 f"{caseC_w500.get('ttft_p50_cv_pct'):.1f}% at median {caseC_w500.get('ttft_p50_median_ms'):.1f} ms — "
                 "**clean, profilable.** Note the stable median (~249 ms) is *higher* than the noisy "
                 "W100/W300 medians: the earlier \"SGLang faster / 0.79× reversal\" was an under-warmup "
                 "artifact. At stable warmup SGLang is ~1.32× vs vLLM W300 (189.0 ms), matching the Phase-1 "
                 "W30 ratio. vLLM not re-run at W500 (W300 recheck is warmup-insensitive: 187.9→189.0 ms "
                 "across W30→W300), so this is a stable reference, not a strict same-warmup comparison.")
    S.append("")
    S.append("---")
    S.append("")

    S.append("## Case D — Decode-heavy (512→512, c=16) — Variance Gate")
    S.append("")
    S.append("| Warmup | Reps | TTFT p50 median (ms) | CV | Gate |")
    S.append("|--------|------|---------------------|----|------|")
    for w in WARMUP_LEVELS:
        med, cv = var_for(caseD_var, w)
        gate = "PASS ✅" if (cv is not None and cv < 5.0) else ("FAIL ⚠" if cv is not None else "N/A")
        n_reps = len((caseD_var.get("warmup_results") or {}).get(w, {}).get("reps") or
                     (caseD_var.get("warmup_results") or {}).get(str(w), {}).get("reps") or [])
        S.append(f"| {w} | {n_reps or 3} | {fmt(med)} | {f'{cv:.1f}%' if cv is not None else 'N/A'} | {gate} |")
    S.append("")
    S.append(f"**Recommended warmup for Phase 3**: {caseD_rec_warmup} · reps: {caseD_rec_reps}")
    S.append(f"**Gate passed**: {'YES ✅' if caseD_gate else 'NO ⚠ — W500 may be needed'}")
    S.append("")
    S.append("---")
    S.append("")

    # vLLM recheck
    S.append("## vLLM Recheck (Cases B and C, warmup=300, 5 reps)")
    S.append("")
    S.append("| Case | vLLM p50 (ms) | CV | Bimodal resolved | Ceiling M |")
    S.append("|------|---------------|----|-----------------|-----------|")
    vllm_B_p50r = vllm_B.get("median_p50_ms")
    vllm_B_cv = vllm_B.get("cv_pct")
    vllm_C_p50r = vllm_C.get("median_p50_ms")
    vllm_C_cv = vllm_C.get("cv_pct")
    S.append(f"| caseB | {fmt(vllm_B_p50r)} | {f'{vllm_B_cv:.1f}%' if vllm_B_cv is not None else 'N/A'} | "
             f"{'YES' if vllm_B.get('bimodal_resolved') else 'NO'} | "
             f"{'YES ⚠' if vllm_B.get('ceiling_M') else 'no'} |")
    S.append(f"| caseC | {fmt(vllm_C_p50r)} | {f'{vllm_C_cv:.1f}%' if vllm_C_cv is not None else 'N/A'} | "
             f"YES | no |")
    S.append("")
    S.append("Phase 1 vLLM Case B reference: 20.8 ms (cv=114.6% ⚠)")
    S.append("")

    (PHASE2 / "summary.md").write_text("\n".join(S))
    print(f"Written: {PHASE2 / 'summary.md'}")

    # ===== Build selected_cases.md =====
    P = []
    P.append("# run2 Phase 3 Protocol — Locked from Phase 2")
    P.append("")
    P.append(f"Generated: {now}")
    P.append("")
    P.append("> This file is the authoritative Phase 3 input. Do not modify without re-running Phase 2.")
    P.append("")

    # Case A
    cA_flags_str = " ".join(caseA_flags) if caseA_flags else "(none)"
    cA_gap = f"{(cA_p50 / 12.6):.2f}×" if cA_p50 else "N/A"
    P.append("## Case A — Short latency (128→128, c=1)")
    P.append("")
    P.append(f"- **Decision**: PROMOTE to Phase 3")
    P.append(f"- **SGLang server flags**: `{cA_flags_str}`")
    P.append(f"- **warmup**: 30 · **reps**: 3 · **bench_n**: 400 · **concurrency**: 1")
    P.append(f"- **SGLang TTFT p50 (Phase 2)**: {fmt(cA_p50, cA_cv)}")
    P.append(f"- **SGLang TTFT p95 (Phase 2)**: {fmt(cA_w.get('median_p95_ms'))}")
    P.append(f"- **SGLang TTFT p99 (Phase 2)**: {fmt(cA_w.get('median_p99_ms'))}")
    P.append(f"- **vLLM TTFT p50 (Phase 1 reference)**: 12.6 ms")
    P.append(f"- **Residual gap**: {cA_gap}")
    P.append(f"- **vLLM ceiling**: attention backend differs (FlashInfer vs FA3) — ceiling M on kernel-level findings")
    P.append(f"- **Phase 3 rationale**: Largest gap (4.89× Phase 1); clean variance; primary profiling target")
    P.append("")

    # Case B
    cB_flags_str = " ".join(caseB_flags) if caseB_flags else "(none)"
    vllm_B_ref = vllm_B.get("median_p50_ms") or 20.8
    cB_gap = f"{(cB_p50 / vllm_B_ref):.2f}×" if cB_p50 else "N/A"
    cB_warmup_rec = 300 if vllm_B.get("ceiling_M") else 30
    P.append("## Case B — Long prefill (2048→128, c=1)")
    P.append("")
    P.append(f"- **Decision**: PROMOTE to Phase 3")
    P.append(f"- **SGLang server flags**: `{cB_flags_str}`")
    P.append(f"- **warmup**: 300 · **reps**: 5 · **bench_n**: 200 · **concurrency**: 1")
    P.append(f"- **SGLang TTFT p50 (Phase 2)**: {fmt(cB_p50, cB_cv)}")
    P.append(f"- **SGLang TTFT p95 (Phase 2)**: {fmt(cB_w.get('median_p95_ms'))}")
    P.append(f"- **SGLang TTFT p99 (Phase 2)**: {fmt(cB_w.get('median_p99_ms'))}")
    P.append(f"- **vLLM TTFT p50 (Phase 2 recheck, w=300)**: {fmt(vllm_B_ref)}")
    P.append(f"- **Residual gap**: {cB_gap}")
    vllm_B_ceil = vllm_B.get("ceiling_M", True)
    P.append(f"- **vLLM ceiling**: {'ceiling M — bimodality NOT fully resolved at w=300; compare ratio directionally only' if vllm_B_ceil else 'no additional ceiling — bimodality resolved at w=300'}")
    P.append(f"- **Attention ceiling**: FlashInfer vs FA3 — ceiling M on kernel-level findings")
    P.append(f"- **Phase 3 rationale**: Second-largest gap; chunked-prefill path tests overhead at long input")
    P.append("")

    # Case C
    vllm_C_ref = vllm_C.get("median_p50_ms") or 187.9
    cC_gap = f"{(cC_p50 / vllm_C_ref):.2f}×" if cC_p50 else "N/A"
    P.append("## Case C — Batched (512→128, c=16)")
    P.append("")
    P.append(f"- **Decision**: PROMOTE to Phase 3")
    P.append(f"- **SGLang server flags**: `(none — default config)`")
    P.append(f"- **warmup**: {caseC_rec_warmup} · **reps**: {caseC_rec_reps} · **bench_n**: 2000 · **concurrency**: 16")
    P.append(f"- **SGLang TTFT p50 (Phase 2, w={caseC_rec_warmup})**: {fmt(cC_p50, cC_cv)}")
    P.append(f"- **vLLM TTFT p50 (Phase 2 recheck, w=300)**: {fmt(vllm_C_ref)}")
    P.append(f"- **Residual gap**: {cC_gap} (SGLang slower)")
    P.append(f"- **Gate**: {'PASS ✅ (W500 probe, CV<5%)' if caseC_gate else 'MARGINAL ⚠ — verify before Phase 3'}")
    if "500" in (caseC_var.get("warmup_results") or {}):
        P.append(f"- **W500 probe**: 5 reps, CV {caseC_w500.get('ttft_p50_cv_pct'):.1f}% — **cleanly profilable.** "
                 f"The W100/W300 \"SGLang faster (0.79×)\" reading was an under-warmup artifact; the stable "
                 f"W500 median ({caseC_w500.get('ttft_p50_median_ms'):.1f} ms) restores the ~1.32× SGLang-slower "
                 f"gap seen in Phase-1 W30.")
        P.append(f"- **Warmup-mismatch caveat**: SGLang stabilized at W500 vs vLLM W300. vLLM is warmup-insensitive "
                 f"here (187.9→189.0 ms across W30→W300), so the ~1.32× comparison is sound as a stable reference. "
                 f"A strict same-warmup vLLM W500 is **only needed if a 'SGLang faster' claim is made** (it is not).")
    P.append(f"- **vLLM ceiling**: attention backend differs — ceiling M on kernel-level findings")
    P.append(f"- **Phase 3 rationale**: Batched decode tests scheduler throughput path; ~1.32× gap warrants profiling")
    P.append("")

    # Case D
    vllm_D_ref = 189.7  # Phase 1 reference
    cD_gap = f"{(cD_p50 / vllm_D_ref):.2f}×" if cD_p50 else "N/A"
    P.append("## Case D — Decode-heavy (512→512, c=16)")
    P.append("")
    P.append(f"- **Decision**: PROMOTE to Phase 3")
    P.append(f"- **SGLang server flags**: `(none — default config)`")
    P.append(f"- **warmup**: {caseD_rec_warmup} · **reps**: {caseD_rec_reps} · **bench_n**: 1000 · **concurrency**: 16")
    P.append(f"- **SGLang TTFT p50 (Phase 2, w={caseD_rec_warmup})**: {fmt(cD_p50, cD_cv)}")
    P.append(f"- **vLLM TTFT p50 (Phase 1 reference)**: {vllm_D_ref} ms")
    P.append(f"- **Residual gap**: {cD_gap}")
    P.append(f"- **Gate**: {'PASS ✅' if caseD_gate else 'MARGINAL ⚠ — verify before Phase 3'}")
    P.append(f"- **vLLM ceiling**: attention backend differs — ceiling M on kernel-level findings")
    P.append(f"- **Phase 3 rationale**: Decode-heavy exposes continuous-batching overhead; TTFT gap 1.33× + p99 cv=47.1% in Phase 1")
    P.append("")

    P.append("## Global Confidence Ceilings (inherited from Phase 1, confirmed in Phase 2)")
    P.append("")
    P.append("| Variable | SGLang | vLLM | Ceiling |")
    P.append("|----------|--------|------|---------|")
    P.append("| Attention backend | FlashInfer 0.6.11.post1 | FlashAttention v3 | M: kernel-level attribution uncertain |")
    P.append("| FlashInfer version | 0.6.11.post1 | 0.6.8.post1 (sampling) | M: sampling kernel timing not directly comparable |")
    P.append("| vLLM prefix caching | ON (default, V1) | ON | Framework-intrinsic, not controlled |")
    P.append("| vLLM Case B bimodality | — | " +
             ("resolved at w=300" if not vllm_B.get("ceiling_M") else f"cv={vllm_B_cv:.1f}% at w=300, NOT resolved") +
             f" | {'no extra ceiling' if not vllm_B.get('ceiling_M') else 'M: compare directionally only'} |")
    P.append("")

    (PHASE2 / "selected_cases.md").write_text("\n".join(P))
    print(f"Written: {PHASE2 / 'selected_cases.md'}")
    print("Phase 2 summarization complete.")


if __name__ == "__main__":
    main()
