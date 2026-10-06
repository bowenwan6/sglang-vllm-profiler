#!/usr/bin/env python3
"""Tables and acceptance checks for the PR-B sessions (PLAN.md, amendment of 2026-10-06).

  python prb_report.py <run dir> [<run dir> ...] [--p1 results/pra/summary.jsonl]

Each run dir holds the summary.jsonl written by prb_run.py. Prints Markdown.
"""

import argparse
import json
import statistics
from pathlib import Path


def load(dirs):
    rows = []
    for d in dirs:
        p = Path(d) / "summary.jsonl"
        if p.exists():
            rows += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return [r for r in rows if not r.get("failed")]


def table(header, lines):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in line) + " |" for line in lines]
    return "\n".join(out)


def ms(values, fmt="{:.1f}"):
    """mean (sd) over seeds; a single value is printed bare."""
    values = [v for v in values if v is not None]
    if not values:
        return "—"
    if len(values) == 1:
        return fmt.format(values[0])
    return f"{fmt.format(statistics.mean(values))} ({fmt.format(statistics.stdev(values))})"


def att(row, cls=None):
    c = row["all"] if cls is None else row["classes"].get(cls)
    return None if c is None else 100 * c["attainment"]


def arms_table(rows, task):
    rs = [r for r in rows if r["task"] == task]
    if not rs:
        return {}
    order, by_arm = [], {}
    for r in rs:
        if r["arm"] not in by_arm:
            order.append(r["arm"])
        by_arm.setdefault(r["arm"], []).append(r)
    lines = []
    for arm in order:
        g = by_arm[arm]
        sent = sum(r["all"]["sent"] for r in g)
        lines.append((
            arm, len(g), ms([att(r, "chat") for r in g]), ms([att(r, "batch") for r in g]),
            ms([att(r) for r in g]),
            f"{100 * sum(r['all']['refused'] for r in g) / sent:.1f}",
            f"{100 * sum(r['all']['wasted_tokens'] for r in g) / max(sum(r['all']['out_tokens'] for r in g), 1):.0f}",
            ms([r["classes"]["chat"]["mean_ttft_ms"] for r in g if "chat" in r["classes"]], "{:.0f}"),
            ms([r["classes"]["batch"]["p99_e2e_ms"] / 1e3 for r in g
                if "batch" in r["classes"] and r["classes"]["batch"].get("p99_e2e_ms")], "{:.1f}"),
            ms([r["all"]["out_tokens_per_s"] for r in g], "{:.0f}"),
        ))  # fmt: skip
    print(table(["arm", "runs", "chat %", "batch %", "all %", "refused %", "wasted tokens %",
                 "chat mean TTFT ms", "batch p99 E2E s", "out tok/s"], lines))  # fmt: skip
    return by_arm


def pooled_sd(*groups):
    """Pooled standard deviation of several small samples (each needs two values)."""
    num = den = 0.0
    for g in groups:
        if len(g) >= 2:
            num += (len(g) - 1) * statistics.variance(g)
            den += len(g) - 1
    return (num / den) ** 0.5 if den else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--p1", default="")
    a = ap.parse_args()
    rows = load(a.dirs)
    # The pilot ran U1's configuration with seed 1; with U1 present its rows are U1 rows.
    # A pilot run that was repeated under U1 (same arm and seed) stays a pilot row.
    if any(r["task"] == "u1" for r in rows):
        done = {(r["arm"], r["seed"]) for r in rows if r["task"] == "u1"}
        rows = [{**r, "task": "u1"} if r["task"] == "pilot" and (r["arm"], r["seed"]) not in done else r
                for r in rows]  # fmt: skip

    # ---- ladder ---------------------------------------------------------------
    lad = [r for r in rows if r["task"] == "ladder"]
    if lad:
        print("## Debug ladder (one slot, a long request holding it)\n")
        print(table(["run", "build", "checks passed", "failed"], [
            (r["cell"], r.get("build"), f"{sum(x['ok'] for x in r['rows'])} of {len(r['rows'])}",
             "; ".join(x["check"] for x in r["rows"] if not x["ok"]) or "—") for r in lad]))  # fmt: skip
        for r in lad:
            print(f"\n**{r['cell']}**\n")
            print(table(["check", "result", "observed"],
                        [(x["check"], "pass" if x["ok"] else "**FAIL**", x["detail"]) for x in r["rows"]]))  # fmt: skip

    # ---- capacity and client check ---------------------------------------------
    cal = {r["cell"]: r for r in rows if r["task"] == "calib"}
    if cal:
        print("\n## Capacity of each class alone (closed loop, concurrency 128, `--max-running-requests 128`)\n")
        print(table(["class", "requests", "req/s", "output tok/s", "mean TTFT ms", "mean TPOT ms"], [
            (n, r["all"]["sent"], f"{r['all']['req_per_s']:.2f}", f"{r['all']['out_tokens_per_s']:.0f}",
             f"{c['mean_ttft_ms']:.0f}", f"{c['mean_tpot_ms']:.1f}")
            for r in cal.values() for n, c in r["classes"].items()]))  # fmt: skip
    chk = {r["cell"]: r for r in rows if r["task"] == "check"}
    if "check_client" in chk and "check_bench" in chk:
        c, b = chk["check_client"], chk["check_bench"]
        cc = c["classes"]["chat"]
        print(f"\n## The client against `bench_serving --goodput` ({c['params']['rate']} req/s, 256-token prompts, 128 output tokens)\n")
        print(table(["tool", "requests", "attainment %", "mean TTFT ms", "mean TPOT ms", "output tok/s"], [
            ("slo_client.py", c["all"]["sent"], f"{att(c):.1f}", f"{cc['mean_ttft_ms']:.0f}", f"{cc['mean_tpot_ms']:.1f}",
             f"{c['all']['out_tokens_per_s']:.0f}"),
            ("bench_serving", b.get("completed"), f"{100 * (b.get('slo_attainment') or 0):.1f}", f"{b.get('mean_ttft_ms') or 0:.0f}",
             f"{b.get('mean_tpot_ms') or 0:.1f}", f"{b.get('output_throughput') or 0:.0f}")]))  # fmt: skip

    # ---- U1 / pilot ---------------------------------------------------------------
    for task, title in (("pilot", "U1 pilot (one seed)"), ("u1", "U1 — chat bursts above capacity, steady batch, priority to chat")):
        if any(r["task"] == task for r in rows):
            print(f"\n## {title}\n")
            g = arms_table(rows, task)
            pr, tight = g.get("per_request"), g.get("global_1.5")
            if task == "u1" and pr and tight:
                globals_ = {k: v for k, v in g.items() if k.startswith("global_")}
                best_name, best = max(globals_.items(), key=lambda kv: statistics.mean(att(r) for r in kv[1]))
                mean = lambda rs, cls=None: statistics.mean(att(r, cls) for r in rs)  # noqa: E731
                sd = {cls: pooled_sd([att(r, cls) for r in pr], [att(r, cls) for r in tight]) for cls in ("chat", "batch", None)}
                d_batch = mean(pr, "batch") - mean(tight, "batch")
                d_chat = mean(pr, "chat") - mean(tight, "chat")
                d_all = mean(pr) - mean(best)
                ok = (d_batch >= max(10, 3 * (sd["batch"] or 0)) and d_chat >= -max(3, 3 * (sd["chat"] or 0))
                      and d_all >= max(3, 3 * (sd[None] or 0)))  # fmt: skip
                print(f"\n**A3.1′**: batch {d_batch:+.1f} pp against `global_1.5` (needs ≥ max(10, 3σ = {3 * (sd['batch'] or 0):.1f})); "
                      f"chat {d_chat:+.1f} pp (needs ≥ −max(3, 3σ = {3 * (sd['chat'] or 0):.1f})); "
                      f"total {d_all:+.1f} pp against the best global arm, `{best_name}` "
                      f"(needs ≥ max(3, 3σ = {3 * (sd[None] or 0):.1f})) → {'PASS' if ok else 'FAIL'}.")  # fmt: skip

    if any(r["task"] == "u1b" for r in rows):
        print("\n## U1b — the same use case with heavier chat bursts (2.0 × c_chat)\n")
        arms_table(rows, "u1b")
    if any(r["task"] == "tp2" for r in rows):
        print("\n## Two GPUs, tensor parallel (`--tp-size 2`): chat alone at 1.3 × the one-GPU c_chat, 1.5 s bound\n")
        print(table(["form", "sent", "attainment %", "refused", "statuses", "mean TTFT ms", "out tok/s"], [
            (r["arm"], r["all"]["sent"], f"{att(r):.1f}", r["all"]["refused"], r["classes"]["chat"]["status"],
             f"{r['classes']['chat']['mean_ttft_ms'] or 0:.0f}", f"{r['all']['out_tokens_per_s']:.0f}")
            for r in rows if r["task"] == "tp2"]))  # fmt: skip

    # ---- U2 ---------------------------------------------------------------------
    if any(r["task"] == "u2" for r in rows):
        print("\n## U2 — steady chat and a batch burst, first come first served (reported, not a gate)\n")
        g = arms_table(rows, "u2")
        pr = g.get("per_request")
        globals_ = {k: v for k, v in g.items() if k.startswith("global_")}
        if pr and globals_:
            best_name, best = max(globals_.items(), key=lambda kv: statistics.mean(att(r) for r in kv[1]))
            d = statistics.mean(att(r) for r in pr) - statistics.mean(att(r) for r in best)
            print(f"\n**A3.2′**: `per_request` is {d:+.1f} pp against the best global arm (`{best_name}`) in total. "
                  f"The queue model predicted a tie.")  # fmt: skip

    # ---- U3 ---------------------------------------------------------------------
    u3 = [r for r in rows if r["task"] == "u3"]
    if u3:
        print("\n## U3 — no-op control at 0.8 of capacity, a fresh server process per arm\n")
        keys = [("chat", "mean_ttft_ms"), ("chat", "p99_ttft_ms"), ("chat", "mean_tpot_ms"),
                ("batch", "mean_ttft_ms"), ("batch", "p99_ttft_ms"), ("batch", "mean_tpot_ms")]  # fmt: skip
        print(table(["arm", "build", "attainment %", *[f"{c} {k[:-3].replace('_', ' ')} ms" for c, k in keys], "out tok/s"], [
            (r["arm"], r["build"], f"{att(r):.1f}", *[f"{r['classes'][c][k]:.1f}" for c, k in keys],
             f"{r['all']['out_tokens_per_s']:.0f}") for r in u3]))  # fmt: skip
        by = {r["arm"]: r for r in u3}
        if {"base_a", "base_b", "patched_absent", "patched_loose"} <= set(by):
            worst = []
            for c, k in keys + [("all", "out_tokens_per_s")]:
                get = lambda arm: by[arm]["all"][k] if c == "all" else by[arm]["classes"][c][k]  # noqa: E731
                base = (get("base_a") + get("base_b")) / 2
                spread = abs(get("base_a") - get("base_b")) / base
                for arm in ("patched_absent", "patched_loose"):
                    dev = abs(get(arm) - base) / base
                    worst.append((dev - max(spread, 0.03), arm, c, k, dev, spread))
            w = max(worst)
            print(f"\n**A3.3**: the largest excess over the allowance (the A/A spread of the unpatched build, or 3 %) is "
                  f"{w[1]} {w[2]} {w[3]}: {w[4] * 100:.1f} % from the unpatched mean, A/A spread {w[5] * 100:.1f} % "
                  f"→ {'PASS' if w[0] <= 0 else 'FAIL'}.")  # fmt: skip

    # ---- U4 ---------------------------------------------------------------------
    u4 = [r for r in rows if r["task"] == "u4"]
    if u4:
        print("\n## U4 — the same 1.5 s bound as a request field and as the global knob (chat alone, 1.3 × c_chat)\n")
        print(table(["form", "seed", "sent", "attainment %", "refused", "mean TTFT ms", "p99 TTFT ms"], [
            (r["arm"], r["seed"], r["all"]["sent"], f"{att(r):.1f}", r["all"]["refused"],
             f"{r['classes']['chat']['mean_ttft_ms']:.0f}", f"{r['classes']['chat']['p99_ttft_ms']:.0f}") for r in u4]))  # fmt: skip
        f = [att(r) for r in u4 if r["arm"] == "field"]
        g = [att(r) for r in u4 if r["arm"] == "global"]
        if f and g:
            sd = pooled_sd(f, g) or 0.0
            d = statistics.mean(f) - statistics.mean(g)
            print(f"\n**A3.4′**: field − global = {d:+.1f} pp (allowance max(2, 3σ = {3 * sd:.1f})) → "
                  f"{'PASS' if abs(d) <= max(2, 3 * sd) else 'FAIL'}.")  # fmt: skip

    # ---- PR-A repeats -------------------------------------------------------------
    t4 = [r for r in rows if r["task"] == "T4-repeat"]
    if t4:
        print("\n## T4 with the fixed benchmark, three seeds per setting (ShareGPT, 1.5 × c0, cap 128)\n")
        lines = []
        for timeout in (None, 2, 10):
            g = [r for r in t4 if r["params"]["waiting_timeout_s"] == timeout]
            if not g:
                continue
            lines.append(("off" if timeout is None else timeout, len(g),
                          ms([r["failed_requests"] for r in g], "{:.0f}"),
                          ms([r["output_throughput"] for r in g], "{:.0f}"),
                          ms([r["request_goodput"] for r in g], "{:.2f}"),
                          ms([100 * r["slo_attainment"] for r in g]),
                          "yes" if all(r["good_printed"] == r["good_recomputed"] for r in g) else "NO"))  # fmt: skip
        print(table(["waiting timeout (s)", "runs", "failed requests", "out tok/s", "goodput (req/s)",
                     "attainment %", "printed = recomputed"], lines))  # fmt: skip
    t1 = [r for r in rows if r["task"] == "T1-repeat"]
    if t1:
        p1 = []
        if a.p1 and Path(a.p1).exists():
            p1 = [json.loads(line) for line in Path(a.p1).read_text().splitlines() if line.strip()]
            p1 = [r for r in p1 if r.get("task") in ("T1", "T1-repeat") and "mult" in r.get("params", {})]
        allr = p1 + t1
        print("\n## T1 load sweep with repeats (session P1's runs plus two more seeds at three points)\n")
        lines = []
        for mult in sorted({r["params"]["mult"] for r in allr}):
            g = [r for r in allr if r["params"]["mult"] == mult]
            lines.append((f"{mult:g}", len(g), ms([r["output_throughput"] for r in g], "{:.0f}"),
                          ms([r["request_goodput"] for r in g], "{:.2f}"), ms([100 * r["slo_attainment"] for r in g])))  # fmt: skip
        print(table(["× c0", "runs", "out tok/s", "goodput (req/s)", "attainment %"], lines))


if __name__ == "__main__":
    main()
