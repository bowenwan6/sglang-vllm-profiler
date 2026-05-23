#!/usr/bin/env python3
"""
Phase 1 summarizer — reads raw bench_serving JSON (+ --output-details) and meta,
computes median across reps + CV, TTFT p50/p95/p99, TPOT p50/p99, throughput, error rate.

Usage (from repo root):
    python3 experiments/qwen3vl8b/phase1/scripts/summarize_phase1.py
"""

import json, statistics
from datetime import datetime, timezone
from pathlib import Path

LAB = Path("/data/sglang-vllm-profiler")
RAW = LAB / "experiments/qwen3vl8b/phase1/raw"
OUT = LAB / "experiments/qwen3vl8b/phase1/summary.md"

CASES_ORDER = ["caseA_short", "caseB_longprefill", "caseC_batched", "caseD_decode"]
CASE_LABELS = {
    "caseA_short":       "A — Short latency (128→128, c=1)",
    "caseB_longprefill": "B — Long prefill  (2048→128, c=1)",
    "caseC_batched":     "C — Batched       (512→128, c=16)",
    "caseD_decode":      "D — Decode-heavy  (512→512, c=16)",
}
FRAMEWORKS = ["sglang-oai", "vllm"]
REPS = 3


def percentile(data, q):
    d = sorted(v for v in data if v is not None)
    if not d:
        return None
    if len(d) == 1:
        return d[0]
    rank = (len(d) - 1) * q / 100.0
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(d):
        return d[lo]
    return d[lo] + frac * (d[lo + 1] - d[lo])


def load(case, fw, rep):
    p = RAW / f"{case}_{fw}_rep{rep}.json"
    return json.load(open(p)) if p.exists() else None


def load_meta(case, fw, rep):
    p = RAW / f"{case}_{fw}_rep{rep}_meta.json"
    return json.load(open(p)) if p.exists() else {}


def rep_metrics(d):
    """Per-rep metric dict from one bench JSON."""
    if d is None:
        return None
    ttfts_ms = [t * 1000 for t in d.get("ttfts", []) if t is not None]
    failures = sum(1 for e in d.get("errors", []) if e)
    completed = d.get("completed", 0)
    total = completed + failures if (completed or failures) else len(d.get("ttfts", []))
    return {
        "ttft_p50": percentile(ttfts_ms, 50) if ttfts_ms else d.get("median_ttft_ms"),
        "ttft_p95": percentile(ttfts_ms, 95) if ttfts_ms else None,
        "ttft_p99": percentile(ttfts_ms, 99) if ttfts_ms else d.get("p99_ttft_ms"),
        "tpot_p50": d.get("median_tpot_ms"),
        "tpot_p99": d.get("p99_tpot_ms"),
        "out_tok_s": d.get("output_throughput"),
        "req_s": d.get("request_throughput"),
        "failures": failures,
        "total": total,
    }


METRIC_ROWS = [
    ("ttft_p50", "TTFT p50 (ms)"),
    ("ttft_p95", "TTFT p95 (ms)"),
    ("ttft_p99", "TTFT p99 (ms)"),
    ("tpot_p50", "TPOT p50 (ms)"),
    ("tpot_p99", "TPOT p99 (ms)"),
    ("out_tok_s", "Out tok/s"),
    ("req_s", "Req/s"),
]


def med_cv(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    cv = (statistics.stdev(vals) / med * 100) if (len(vals) >= 2 and med) else None
    return med, cv


def fmt(val, cv):
    if val is None:
        return "N/A"
    s = f"{val:.1f}"
    if cv is not None:
        s += f" (cv={cv:.1f}%{' ⚠' if cv > 5 else ''})"
    return s


def ratio(a, b):
    if a is None or b is None or b == 0:
        return "N/A"
    r = a / b
    arrow = "↑" if r > 1.05 else ("↓" if r < 0.95 else "≈")
    return f"{r:.2f}× {arrow}"


def main():
    L = []
    L.append("# Phase 1 — Baseline Benchmark Summary")
    L.append("")
    L.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    L.append("")
    L.append("Active run `qwen3vl8b` · model Qwen3-VL-8B-Instruct @ 0c351dd · GPU 0 · TP=1 · bf16.")
    L.append("Ratio = SGLang / vLLM. >1 means SGLang slower (latency) or higher (throughput).")
    L.append("Sampling: explicit greedy via `--extra-request-body {\"temperature\":0,\"top_p\":1}`, "
             "`ignore_eos` default True, fixed output_len.")
    L.append("")
    L.append("> ⚠️ run1 Phase 1 (`experiments/phase1/`) is historical reference under a different "
             "environment. Compare directionally only — do NOT mix with the earlier round (removed).")
    L.append("")

    err_summary = []
    for case in CASES_ORDER:
        L.append(f"## {CASE_LABELS[case]}")
        L.append("")
        fw_metrics = {}
        for fw in FRAMEWORKS:
            metas = [load_meta(case, fw, r) for r in range(1, REPS + 1)]
            fails = [m for m in metas if m.get("status") == "FAILED"]
            if fails:
                L.append(f"**{fw}: {len(fails)}/{REPS} runs FAILED — see meta.json**")
                L.append("")
            reps = [rep_metrics(load(case, fw, r)) for r in range(1, REPS + 1)]
            reps = [x for x in reps if x is not None]
            agg = {}
            for key, _ in METRIC_ROWS:
                agg[key] = med_cv([r[key] for r in reps])
            tot_fail = sum(r["failures"] for r in reps)
            tot_req = sum(r["total"] for r in reps)
            fw_metrics[fw] = agg
            err_summary.append((case, fw, tot_fail, tot_req))

        L.append("| Metric | SGLang | vLLM | SGLang/vLLM |")
        L.append("|--------|--------|------|-------------|")
        for key, label in METRIC_ROWS:
            sm, sc = fw_metrics.get("sglang-oai", {}).get(key, (None, None))
            vm, vc = fw_metrics.get("vllm", {}).get(key, (None, None))
            L.append(f"| {label} | {fmt(sm, sc)} | {fmt(vm, vc)} | {ratio(sm, vm)} |")
        L.append("")
        for fw in FRAMEWORKS:
            v = load_meta(case, fw, 1).get("versions", {})
            L.append(f"**{fw} versions**: {json.dumps(v)}")
        L.append("")
        L.append("---")
        L.append("")

    # Error-rate table
    L.append("## Error rate")
    L.append("")
    L.append("| Case | Framework | Failed | Total | Rate |")
    L.append("|------|-----------|-------:|------:|-----:|")
    for case, fw, nf, nt in err_summary:
        rate = f"{(nf/nt*100):.2f}%" if nt else "N/A"
        L.append(f"| {case} | {fw} | {nf} | {nt} | {rate} |")
    L.append("")

    # Fairness notes
    L.append("## Fairness Notes")
    L.append("")
    L.append("| Variable | SGLang | vLLM | Tier |")
    L.append("|----------|--------|------|------|")
    L.append("| GPU | H200 index 0 | H200 index 0 | Controlled |")
    L.append("| Model | Qwen3-VL-8B-Instruct 0c351dd | same | Controlled |")
    L.append("| dtype | bfloat16 | bfloat16 | Controlled |")
    L.append("| TP | 1 | 1 | Controlled |")
    L.append("| Sampling | temperature=0, top_p=1 (explicit), ignore_eos default | same client | Controlled |")
    L.append("| torch / CUDA | 2.11.0+cu130 / 13.0 | 2.11.0+cu130 / 13.0 | Measured (aligned) |")
    L.append("| Attention backend | FlashInfer 0.6.11.post1 (text) | FlashAttention v3 | Measured |")
    L.append("| FlashInfer | 0.6.11.post1 | 0.6.8.post1 (sampling) | Measured |")
    L.append("| Scheduler | fcfs / radix cache | vLLM cache mgr (prefix-cache ON) | Framework-intrinsic |")
    L.append("")
    OUT.write_text("\n".join(L))
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
