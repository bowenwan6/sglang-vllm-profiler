#!/usr/bin/env python3
"""
Reads raw bench_serving JSON + meta.json files from experiments/phase1/raw/,
computes median across 3 reps, checks stdev/median, writes summary.md.

Usage (from /data/profiling_lab):
    python3 experiments/phase1/scripts/summarize_phase1.py
"""

import json, os, statistics
from pathlib import Path

LAB = Path("/data/profiling_lab")
RAW = LAB / "experiments/phase1/raw"
OUT = LAB / "experiments/phase1/summary.md"

CASES_ORDER = ["caseA_short", "caseB_longprefill", "caseC_batched", "caseD_decode"]
CASE_LABELS = {
    "caseA_short":       "A — Short latency (128→128, c=1)",
    "caseB_longprefill": "B — Long prefill  (2048→128, c=1)",
    "caseC_batched":     "C — Batched       (512→128, c=16)",
    "caseD_decode":      "D — Decode-heavy  (512→512, c=16)",
}
FRAMEWORKS = ["sglang-oai", "vllm"]
REPS = 3

METRIC_KEYS = {
    "ttft_p50":   ("median_ttft_ms",          "TTFT p50 (ms)"),
    "ttft_p99":   ("p99_ttft_ms",             "TTFT p99 (ms)"),
    "tpot_p50":   ("median_tpot_ms",          "TPOT p50 (ms)"),
    "tpot_p99":   ("p99_tpot_ms",             "TPOT p99 (ms)"),
    "out_tok_s":  ("output_throughput",       "Out tok/s"),
    "req_s":      ("request_throughput",      "Req/s"),
}


def load_result(case: str, fw: str, rep: int):
    p = RAW / f"{case}_{fw}_rep{rep}.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def load_meta(case: str, fw: str, rep: int):
    p = RAW / f"{case}_{fw}_rep{rep}_meta.json"
    if not p.exists():
        return {}
    with open(p) as f:
        return json.load(f)


def extract_metric(result: dict, json_key: str):
    if result is None:
        return None
    return result.get(json_key)


def median_and_cv(vals):
    """Return (median, cv%) or (None, None) if fewer than 2 valid values."""
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    if len(vals) >= 2:
        sd = statistics.stdev(vals)
        cv = (sd / med * 100) if med else 0
    else:
        cv = None
    return med, cv


def fmt(val, cv):
    if val is None:
        return "N/A"
    s = f"{val:.1f}"
    if cv is not None:
        flag = " ⚠" if cv > 5 else ""
        s += f" (cv={cv:.1f}%{flag})"
    return s


def ratio_str(sgl_val, vllm_val):
    if sgl_val is None or vllm_val is None or vllm_val == 0:
        return "N/A"
    r = sgl_val / vllm_val
    arrow = "↑" if r > 1.05 else ("↓" if r < 0.95 else "≈")
    return f"{r:.2f}× {arrow}"


def main():
    lines = []
    lines.append("# Phase 1 — Baseline Benchmark Summary")
    lines.append("")
    lines.append(f"Generated: {__import__('datetime').datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    lines.append("")
    lines.append("Ratio column: SGLang / vLLM. >1 means SGLang is **slower** (for latency metrics) or **higher** (for throughput).")
    lines.append("")

    all_rows = {}

    for case in CASES_ORDER:
        label = CASE_LABELS[case]
        lines.append(f"## {label}")
        lines.append("")

        fw_data = {}
        for fw in FRAMEWORKS:
            results = [load_result(case, fw, r) for r in range(1, REPS + 1)]
            metas   = [load_meta(case, fw, r) for r in range(1, REPS + 1)]

            # check for failures
            fails = [m.get("status") for m in metas if m.get("status") == "FAILED"]
            if fails:
                lines.append(f"**{fw}**: {len(fails)}/{REPS} runs FAILED — check meta.json")
                lines.append("")

            fw_data[fw] = {}
            for key, (json_key, _) in METRIC_KEYS.items():
                vals = [extract_metric(r, json_key) for r in results]
                med, cv = median_and_cv(vals)
                fw_data[fw][key] = (med, cv)

        # Print table
        header = "| Metric | SGLang | vLLM | SGLang/vLLM |"
        sep    = "|--------|--------|------|-------------|"
        lines.append(header)
        lines.append(sep)
        for key, (_, label) in METRIC_KEYS.items():
            sgl_med, sgl_cv   = fw_data.get("sglang-oai", {}).get(key, (None, None))
            vllm_med, vllm_cv = fw_data.get("vllm", {}).get(key, (None, None))
            ratio = ratio_str(sgl_med, vllm_med)
            lines.append(f"| {label} | {fmt(sgl_med, sgl_cv)} | {fmt(vllm_med, vllm_cv)} | {ratio} |")
        lines.append("")

        # Versions from meta
        for fw in FRAMEWORKS:
            m = load_meta(case, fw, 1)
            v = m.get("versions", {})
            lines.append(f"**{fw} versions**: {json.dumps(v)}")
        lines.append("")
        lines.append("---")
        lines.append("")

        all_rows[case] = fw_data

    # Fairness notes
    lines.append("## Fairness Notes")
    lines.append("")
    lines.append("| Variable | SGLang | vLLM | Tier |")
    lines.append("|----------|--------|------|------|")
    lines.append("| GPU | H200 index 6 | H200 index 6 | Controlled |")
    lines.append("| Model | Qwen3-VL-8B-Instruct 0c351dd | same | Controlled |")
    lines.append("| dtype | bfloat16 | bfloat16 | Controlled |")
    lines.append("| TP | 1 | 1 | Controlled |")
    lines.append("| Attention backend | FlashInfer 0.6.7.post3 (text) | FlashAttention v3 | Measured |")
    lines.append("| torch version | 2.9.1+cu129 | 2.10.0+cu128 | Measured |")
    lines.append("| KV cache | ~102 GB | ~105.9 GB | Measured |")
    lines.append("| ignore_eos | default (True) | default (True) | Controlled |")
    lines.append("| Scheduler | radix cache | vLLM cache manager | Framework-intrinsic |")
    lines.append("| CUDA graph shapes | 36 graphs (SGLang) | 51 graphs (vLLM) | Framework-intrinsic |")
    lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"Written: {OUT}")


if __name__ == "__main__":
    main()
