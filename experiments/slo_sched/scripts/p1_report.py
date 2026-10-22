#!/usr/bin/env python3
"""Turn session P1's summary.jsonl into the tables and acceptance checks of PLAN.md stage 1.

  python p1_report.py <dir with summary.jsonl and cells/> [--alt ttft:2000,tpot:50,e2el:10000 ...]

Prints Markdown. --alt re-evaluates the T1 sweep and the T2 caps under another SLO set from the
per-request details in cells/*.jsonl (no rerun needed); it may be given several times.
"""

import argparse
import json
import statistics
from pathlib import Path


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [r for r in rows if not r.get("failed")]


def recompute(result, slo_ms):
    good = 0
    for ttft, itl, out_len, err in zip(
        result["ttfts"], result["itls"], result["output_lens"], result["errors"]
    ):
        if err:
            continue
        latency = ttft + sum(itl)
        tpot = (latency - ttft) / (out_len - 1) if out_len > 1 else 0.0
        obs = {"ttft": ttft * 1000, "tpot": tpot * 1000, "e2el": latency * 1000}
        good += all(obs[k] <= v for k, v in slo_ms.items())
    return good


def by_metric(row):
    return " ".join(f"{k} {v * 100:.0f}" for k, v in row["slo_attainment_by_metric"].items())


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--alt", action="append", default=[])
    a = ap.parse_args()
    d = Path(a.dir)
    rows = load(d / "summary.jsonl")
    t = lambda name: [r for r in rows if r["task"] == name]  # noqa: E731

    # ---- A1.5 ---------------------------------------------------------------
    mism = [(r["cell"], r["good_printed"], r["good_recomputed"], r["sent"]) for r in rows
            if r["good_printed"] != r["good_recomputed"]]  # fmt: skip
    print(f"## A1.5 — printed vs recomputed good-request count\n\n{len(rows)} cells, "
          f"{len(mism)} differ.")  # fmt: skip
    if mism:
        print(table(["cell", "printed", "recomputed", "sent"], mism))

    # ---- T1 -----------------------------------------------------------------
    t1 = t("T1")
    probe = [r for r in t1 if "probe" in r["cell"]]
    sweep = sorted((r for r in t1 if "mult" in r["params"]), key=lambda r: r["params"]["mult"])
    reps = t("T1-repeat")
    print("\n## T1 — load sweep\n")
    if probe:
        p = probe[0]
        print(f"Capacity probe (closed loop, concurrency 256): {p['request_throughput']:.2f} req/s, "
              f"{p['output_throughput']:.0f} output tok/s, mean TPOT {p['mean_tpot_ms']:.1f} ms.\n")  # fmt: skip
    print(table(
        ["× c0", "rate (req/s)", "sent", "req thr", "out tok/s", "goodput (req/s)",
         "attain %", "per SLO %", "mean TTFT ms", "p99 TTFT ms", "mean TPOT ms", "p99 E2E ms"],
        [(f"{r['params']['mult']:g}", f"{r['params']['rate']:.2f}", r["sent"],
          f"{r['request_throughput']:.2f}", f"{r['output_throughput']:.0f}",
          f"{r['request_goodput']:.2f}", f"{r['slo_attainment'] * 100:.1f}", by_metric(r),
          f"{r['mean_ttft_ms']:.0f}", f"{r['p99_ttft_ms']:.0f}", f"{r['mean_tpot_ms']:.1f}",
          f"{r['p99_e2e_latency_ms']:.0f}") for r in sweep]))  # fmt: skip
    sigma_rel = sigma_thr = None
    if sweep:
        best = max(sweep, key=lambda r: r["request_goodput"])
        same = [best] + reps
        if len(same) >= 2:
            g = [r["request_goodput"] for r in same]
            sigma_rel = statistics.stdev(g) / statistics.mean(g)
            att = [r["slo_attainment"] * 100 for r in same]
            thr = [r["output_throughput"] for r in same]
            sigma_thr = statistics.stdev(thr) / statistics.mean(thr)
            print(f"\nNoise at {best['params']['mult']:g} × c0 over {len(same)} runs: goodput "
                  f"{', '.join(f'{x:.2f}' for x in g)} req/s (σ = {sigma_rel * 100:.1f} % of the mean); "
                  f"attainment {', '.join(f'{x:.1f}' for x in att)} % "
                  f"(σ = {statistics.stdev(att) / statistics.mean(att) * 100:.1f} %); output throughput "
                  f"{', '.join(f'{x:.0f}' for x in thr)} tok/s (σ = {sigma_thr * 100:.1f} %).")  # fmt: skip
        top, last = sweep[-1], best
        peak_thr = max(r["output_throughput"] for r in sweep)
        drop = 1 - top["request_goodput"] / last["request_goodput"]
        thr_ratio = top["output_throughput"] / peak_thr
        ok = drop >= 0.30 and thr_ratio >= 0.90
        print(f"\n**A1.6**: goodput peaks at {last['params']['mult']:g} × c0 "
              f"({last['request_goodput']:.2f} req/s); at {top['params']['mult']:g} × c0 it is "
              f"{drop * 100:.0f} % below the peak while output throughput is "
              f"{thr_ratio * 100:.0f} % of its own peak → {'PASS' if ok else 'FAIL'}.")  # fmt: skip

    thresh = max(0.05, 3 * (sigma_rel or 0.0))

    def disagreement(name, cells, label, by_key, higher_is_better=True):
        if len(cells) < 2:
            return None
        pick = max if higher_is_better else min
        best_other = pick(cells, key=lambda r: r[by_key])
        best_good = max(cells, key=lambda r: r["request_goodput"])
        gap = 1 - best_other["request_goodput"] / best_good["request_goodput"]
        differs = label(best_other) != label(best_good) and gap >= thresh
        print(f"\n{name}: best by {by_key} = {label(best_other)}, best by goodput = "
              f"{label(best_good)}, goodput gap {gap * 100:.1f} % (threshold {thresh * 100:.1f} %) → "
              f"{'decision differs' if differs else 'same decision'}.")  # fmt: skip
        if label(best_other) != label(best_good):
            margin = abs(1 - best_good[by_key] / best_other[by_key])
            noise = (f"; σ of output throughput is {sigma_thr * 100:.1f} %, so that ranking is "
                     f"{'inside' if margin < 2 * sigma_thr else 'outside'} the noise"
                     if by_key == "output_throughput" and sigma_thr else "")  # fmt: skip
            print(f"The two differ by {margin * 100:.1f} % in {by_key}{noise}.")
        return differs

    # ---- T2 -----------------------------------------------------------------
    t2 = sorted(t("T2"), key=lambda r: r["params"]["max_running_requests"])
    print("\n## T2 — batch cap at 1.25 × c0\n")
    print(table(
        ["max running", "req thr", "out tok/s", "goodput (req/s)", "attain %", "per SLO %",
         "mean TTFT ms", "mean TPOT ms", "p99 E2E ms", "peak concurrency"],
        [(r["params"]["max_running_requests"], f"{r['request_throughput']:.2f}",
          f"{r['output_throughput']:.0f}", f"{r['request_goodput']:.2f}",
          f"{r['slo_attainment'] * 100:.1f}", by_metric(r), f"{r['mean_ttft_ms']:.0f}",
          f"{r['mean_tpot_ms']:.1f}", f"{r['p99_e2e_latency_ms']:.0f}",
          r["max_concurrent_requests"]) for r in t2]))  # fmt: skip
    d2 = disagreement("T2", t2, lambda r: f"cap {r['params']['max_running_requests']}",
                      "output_throughput")  # fmt: skip

    # ---- T3 -----------------------------------------------------------------
    t3 = t("T3")
    print("\n## T3 — queue policy, short and long clients at once\n")
    print(table(
        ["policy", "client", "rate", "sent", "goodput (req/s)", "attain %", "per SLO %",
         "mean TTFT ms", "p99 TTFT ms", "p99 E2E ms"],
        [(r["params"]["policy"], r["cell"].rsplit("_", 1)[-1], f"{r['params']['rate']:.2f}",
          r["sent"], f"{r['request_goodput']:.2f}", f"{r['slo_attainment'] * 100:.1f}",
          by_metric(r), f"{r['mean_ttft_ms']:.0f}", f"{r['p99_ttft_ms']:.0f}",
          f"{r['p99_e2e_latency_ms']:.0f}") for r in t3]))  # fmt: skip
    d3 = None
    pol = {}
    for r in t3:
        e = pol.setdefault(r["params"]["policy"], {"request_goodput": 0.0, "sent": 0, "ttft_sum": 0.0})
        e["request_goodput"] += r["request_goodput"]
        e["sent"] += r["sent"]
        e["ttft_sum"] += r["mean_ttft_ms"] * r["sent"]
    cells3 = [{"policy": k, "request_goodput": v["request_goodput"],
               "mean_ttft_ms": v["ttft_sum"] / max(v["sent"], 1)} for k, v in pol.items()]  # fmt: skip
    if cells3:
        d3 = disagreement("T3 (both clients pooled)", cells3, lambda r: r["policy"],
                          "mean_ttft_ms", higher_is_better=False)  # fmt: skip

    # ---- T4 -----------------------------------------------------------------
    t4 = t("T4")
    print("\n## T4 — global waiting timeout at 1.5 × c0, max running 128\n")
    print(table(
        ["waiting timeout (s)", "sent", "failed (503)", "req thr", "out tok/s",
         "goodput (req/s)", "attain %", "per SLO %", "mean TTFT ms", "p99 E2E ms"],
        [("off" if r["params"]["waiting_timeout_s"] is None else r["params"]["waiting_timeout_s"],
          r["sent"], r["failed_requests"], f"{r['request_throughput']:.2f}",
          f"{r['output_throughput']:.0f}", f"{r['request_goodput']:.2f}",
          f"{r['slo_attainment'] * 100:.1f}", by_metric(r), f"{r['mean_ttft_ms']:.0f}",
          f"{r['p99_e2e_latency_ms']:.0f}") for r in t4]))  # fmt: skip

    def label4(r):
        return f"timeout {r['params']['waiting_timeout_s'] or 'off'}"

    d4 = disagreement("T4", t4, label4, "output_throughput")

    # T4 as the benchmark reports it after the in-stream-error fix: a response with no token
    # and no text is a request the server aborted, i.e. a failed request.
    lines, t4_fixed = [], []
    for r in t4:
        path = d / "cells" / f"{r['cell']}.jsonl"
        if not path.exists():
            continue
        res = json.loads(path.read_text().strip().splitlines()[-1])
        n = len(res["ttfts"])
        aborted = {i for i in range(n) if not res["generated_texts"][i] and not res["itls"][i]}
        if not aborted:
            lines.append(("off" if r["params"]["waiting_timeout_s"] is None else r["params"]["waiting_timeout_s"],
                          0, f"{r['request_throughput']:.2f}", f"{r['output_throughput']:.0f}",
                          f"{r['request_goodput']:.2f}", f"{r['slo_attainment'] * 100:.1f}", "0"))  # fmt: skip
            t4_fixed.append(r)
            continue
        phantom = sum(res["output_lens"][i] for i in aborted)
        served = {**res, "errors": [e or ("aborted" if i in aborted else "") for i, e in enumerate(res["errors"])]}
        good = recompute(served, r["slo_ms"])
        dur = res["duration"]
        lines.append((r["params"]["waiting_timeout_s"], len(aborted), f"{(n - len(aborted)) / dur:.2f}",
                      f"{(res['total_output_tokens'] - phantom) / dur:.0f}", f"{good / dur:.2f}",
                      f"{good / n * 100:.1f}", f"{phantom / res['total_output_tokens'] * 100:.0f}"))  # fmt: skip
        t4_fixed.append({**r, "output_throughput": (res["total_output_tokens"] - phantom) / dur,
                         "request_goodput": good / dur})  # fmt: skip
    d4_fixed = None
    if lines:
        print("\nT4 with server-aborted responses counted as failed (what the fixed benchmark reports):\n")
        print(table(["waiting timeout (s)", "aborted by the server", "req thr", "out tok/s",
                     "goodput (req/s)", "attain %", "phantom output tokens in the stock report %"], lines))  # fmt: skip
        if len(t4_fixed) == len(t4):
            d4_fixed = disagreement("T4, server-aborted requests counted as failed", t4_fixed,
                                    label4, "output_throughput")  # fmt: skip

    # The stock T4 numbers are inflated by the aborted requests, so A1.7 is stated for both.
    for title, v4 in (("on the benchmark's stock output", d4),
                      ("with server-aborted requests counted as failed", d4_fixed)):  # fmt: skip
        if v4 is None:
            continue
        hits = [n for n, v in (("T2", d2), ("T3", d3), ("T4", v4)) if v]
        print(f"\n**A1.7**, {title}: decision differs in {', '.join(hits) or 'none'} → "
              f"{'PASS' if hits else 'FAIL'}.")  # fmt: skip

    # ---- alternative SLO set, offline ----------------------------------------
    for alt in a.alt:
        slo = {k: float(v) for k, v in (kv.split(":") for kv in alt.split(","))}
        for title, column, cells, label in (
            ("T1", "× c0", sweep, lambda r: f"{r['params']['mult']:g}"),
            ("T2", "max running", t2, lambda r: r["params"]["max_running_requests"]),
        ):
            lines = []
            for r in cells:
                path = d / "cells" / f"{r['cell']}.jsonl"
                if not path.exists():
                    continue
                res = json.loads(path.read_text().strip().splitlines()[-1])
                good = recompute(res, slo)
                lines.append((label(r), f"{good / res['duration']:.2f}",
                              f"{good / len(res['ttfts']) * 100:.1f}"))  # fmt: skip
            print(f"\n## {title} re-evaluated offline under {alt}\n")
            print(table([column, "goodput (req/s)", "attain %"], lines))


if __name__ == "__main__":
    main()
