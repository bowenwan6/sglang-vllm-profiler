#!/usr/bin/env python3
"""PR-A final-head check on a live server: what bench_serving reports under --goodput.

Runs on the node inside the sgl-profiler env, after pra_node.sh has installed the branch
head. Every cell is one bench_serving run with --output-details. For a cell with --goodput
three things are compared: the fields in the result file, the block printed to the console
and a recount from the per-request details. One cell runs without the flag and must show
neither the block nor the fields.

  python pra_verify.py --out ~/sgl/logs/pra --deadline-epoch <unix time to stop by>

Stop cleanly between cells with: touch <out>/STOP
"""

import argparse
import json
import re
import subprocess
import sys

from p1_bench import ARRIVAL_SPAN_S, SLO_MS, Runner, slo_args

GOODPUT_KEYS = ("goodput_slos_ms", "request_goodput", "slo_attainment", "slo_attainment_by_metric")  # fmt: skip
EXTRA_SWEEP = (0.5, 0.8, 1.25, 2.0)


def recount(result, slo_ms):
    """Good requests, and requests meeting each SLO alone, from --output-details."""
    good = 0
    by_metric = dict.fromkeys(slo_ms, 0)
    for ttft, itl, out_len, err in zip(
        result["ttfts"], result["itls"], result["output_lens"], result["errors"]
    ):
        if err:
            continue
        latency = ttft + sum(itl)
        tpot = (latency - ttft) / (out_len - 1) if out_len > 1 else 0.0
        observed = {"ttft": ttft * 1000, "tpot": tpot * 1000, "e2el": latency * 1000}
        met = {name: observed[name] <= slo for name, slo in slo_ms.items()}
        good += all(met.values())
        for name, ok in met.items():
            by_metric[name] += ok
    return good, by_metric


def printed_block(log_text):
    """The Goodput block of the console output as {label: value text}, or None."""
    lines = log_text.splitlines()
    starts = [i for i, line in enumerate(lines) if re.fullmatch(r"-+Goodput-+", line)]
    if not starts:
        return None
    block = {}
    for line in lines[starts[-1] + 1 :]:
        if re.fullmatch(r"[-=]+.*[-=]+", line):
            break
        label, _, value = line.partition(":")
        block[label.strip()] = value.strip()
    return block


class Verifier(Runner):
    def __init__(self, args):
        super().__init__(args)
        self.failures = []
        self.checked = 0

    def bench_cmd(self, cell, *, rate, num_prompts, slo_ms, dataset, seed, max_concurrency,
                  backend="sglang-oai-chat", goodput=True):  # fmt: skip
        cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", backend, "--host", "127.0.0.1", "--port", str(self.a.port),
            "--model", self.a.model, *dataset,
            "--num-prompts", str(num_prompts), "--request-rate", str(rate),
            "--seed", str(seed), "--disable-tqdm", "--tag", cell,
            "--output-file", str(self.out / "cells" / f"{cell}.jsonl"), "--output-details",
        ]  # fmt: skip
        if goodput:
            cmd += ["--goodput", *slo_args(slo_ms)]
        if max_concurrency:
            cmd += ["--max-concurrency", str(max_concurrency)]
        return cmd

    def check(self, cell, name, ok, detail=""):
        self.checked += 1
        if not ok:
            self.failures.append(f"{cell}: {name} {detail}")
        self.say(f"    {'ok  ' if ok else 'FAIL'} {cell}: {name} {detail}")

    def run_cell(self, cell, *, rate, num_prompts, backend="sglang-oai-chat", goodput=True,
                 max_concurrency=None, params=None):  # fmt: skip
        cmd = self.bench_cmd(cell, rate=rate, num_prompts=num_prompts, slo_ms=SLO_MS,
                             dataset=("--dataset-name", "sharegpt"), seed=1,
                             max_concurrency=max_concurrency, backend=backend, goodput=goodput)  # fmt: skip
        log_path = self.out / "cells" / f"{cell}.log"
        with open(log_path, "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        path = self.out / "cells" / f"{cell}.jsonl"
        if rc != 0 or not path.exists():
            self.check(cell, "bench_serving exits 0 and writes a result", False, f"rc={rc}")
            return None
        result = json.loads(path.read_text().strip().splitlines()[-1])
        block = printed_block(log_path.read_text())
        sent = len(result["ttfts"])
        failed = sum(bool(e) for e in result["errors"])
        row = {
            "cell": cell, "backend": backend, "goodput_flag": goodput, "params": params or {},
            "sent": sent, "failed_requests": failed,
            **{k: result.get(k) for k in (
                "duration", "completed", "request_throughput", "output_throughput",
                "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_e2e_latency_ms", *GOODPUT_KEYS)},
        }  # fmt: skip
        if not goodput:
            self.check(cell, "no goodput fields in the result", not any(k in result for k in GOODPUT_KEYS))  # fmt: skip
            self.check(cell, "no Goodput block printed", block is None)
            self.say(f"cell {cell}: {row['request_throughput']:.2f} req/s, "
                     f"{row['output_throughput']:.0f} tok/s, failed {failed} (no --goodput)")  # fmt: skip
            self.record(row)
            return row

        self.check(cell, "result has the four goodput fields", all(k in result for k in GOODPUT_KEYS))  # fmt: skip
        if not all(k in result for k in GOODPUT_KEYS):
            self.record(row)
            return row
        good, by_metric = recount(result, SLO_MS)
        good_reported = round(result["slo_attainment"] * sent)
        by_metric_reported = {k: round(v * sent) for k, v in result["slo_attainment_by_metric"].items()}  # fmt: skip
        self.check(cell, "SLOs in the result are the ones given", result["goodput_slos_ms"] == SLO_MS)  # fmt: skip
        self.check(cell, "good requests: reported == recount", good_reported == good, f"{good_reported} / {good} of {sent}")  # fmt: skip
        self.check(cell, "per-SLO counts: reported == recount", by_metric_reported == by_metric, f"{by_metric_reported} / {by_metric}")  # fmt: skip
        self.check(cell, "goodput == good / duration",
                   abs(result["request_goodput"] - good / result["duration"]) < 1e-9,
                   f"{result['request_goodput']:.4f} req/s")  # fmt: skip
        expected_block = {
            "SLOs (ms)": " ".join(slo_args(SLO_MS)),
            "Request goodput (req/s)": f"{result['request_goodput']:.2f}",
            "SLO attainment (%)": f"{result['slo_attainment'] * 100:.2f}",
            **{f"{k.upper()} SLO attainment (%)": f"{v * 100:.2f}"
               for k, v in result["slo_attainment_by_metric"].items()},
        }  # fmt: skip
        self.check(cell, "printed block == result fields", block == expected_block, "" if block == expected_block else f"printed {block}")  # fmt: skip
        row.update(good_recount=good, good_reported=good_reported)
        self.record(row)
        self.say(f"cell {cell}: {row['request_throughput']:.2f} req/s, {row['output_throughput']:.0f} tok/s | "
                 f"goodput {result['request_goodput']:.2f} req/s, attainment {result['slo_attainment'] * 100:.1f}% "
                 f"{ {k: round(v * 100, 1) for k, v in result['slo_attainment_by_metric'].items()} } | failed {failed}")  # fmt: skip
        return row

    def run(self):
        try:
            if not self.start_server("default"):
                self.failures.append("server did not start")
                return 1
            probe = self.run_cell("fin_probe_c256", rate=float("inf"), num_prompts=1500,
                                  max_concurrency=256, params={"max_concurrency": 256})  # fmt: skip
            if probe is None:
                return 1
            c0 = probe["request_throughput"]
            self.say(f"capacity probe: c0 = {c0:.2f} req/s at concurrency 256")

            def at(mult):
                rate = round(c0 * mult, 3)
                return {"rate": rate, "num_prompts": max(200, round(rate * ARRIVAL_SPAN_S)),
                        "params": {"mult": mult, "rate": rate}}  # fmt: skip

            self.run_cell("fin_x1_chat", **at(1.0))
            self.run_cell("fin_x1.6_chat", **at(1.6))
            self.run_cell("fin_x1.6_native", backend="sglang", **at(1.6))
            self.run_cell("fin_x1_noflag", goodput=False, **at(1.0))
            for mult in EXTRA_SWEEP:
                if self.out_of_time():
                    break
                self.run_cell(f"fin_x{mult:g}_chat", **at(mult))
        finally:
            self.stop_server()
            verdict = "PASS" if not self.failures else "FAIL"
            summary = {"verdict": verdict, "checks": self.checked, "failures": self.failures}
            (self.out / "verify_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            self.say(f"VERIFY {verdict}: {self.checked} checks, {len(self.failures)} failed")
            for failure in self.failures:
                self.say(f"  failed: {failure}")
            (self.out / ".verify_finished").touch()
        return 0 if not self.failures else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    p.add_argument("--dry-run", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    args.dry_run = False
    sys.exit(Verifier(args).run())


if __name__ == "__main__":
    main()
