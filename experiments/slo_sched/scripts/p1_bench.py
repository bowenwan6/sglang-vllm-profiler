#!/usr/bin/env python3
"""P1 runner: tasks T1-T4 of PLAN.md stage 1, with the goodput-patched bench_serving.

Runs on the node inside the sgl-profiler env. Every cell is one bench_serving run with
--goodput and --output-details; its JSONL goes to <out>/cells/, a one-line summary to
<out>/summary.jsonl, progress to <out>/progress.log. Each cell's goodput is recomputed
from the per-request details and compared with the value the benchmark printed (A1.5).

  python p1_bench.py --out ~/sgl/logs/p1 --deadline-epoch <unix time to stop by>
  python p1_bench.py --out /tmp/p1 --dry-run        # control flow only, fake numbers

Stop cleanly between cells with: touch <out>/STOP
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SLO_MS = {"ttft": 2000.0, "tpot": 100.0, "e2el": 20000.0}
SHORT_SLO_MS = {"ttft": 1000.0, "tpot": 100.0}
LONG_SLO_MS = {"e2el": 30000.0}
SWEEP = (0.5, 0.8, 1.0, 1.25, 1.6, 2.0)
ARRIVAL_SPAN_S = 50
CELL_BUDGET_S = 240  # skip a cell if less than this is left before the deadline


def slo_args(slo_ms):
    return [f"{name}:{value:g}" for name, value in slo_ms.items()]


def recompute_good(result, slo_ms):
    """Good-request count from --output-details, with latency = ttft + sum(itl)."""
    good = 0
    for ttft, itl, out_len, err in zip(
        result["ttfts"], result["itls"], result["output_lens"], result["errors"]
    ):
        if err:
            continue
        latency = ttft + sum(itl)
        tpot = (latency - ttft) / (out_len - 1) if out_len > 1 else 0.0
        observed = {"ttft": ttft * 1000, "tpot": tpot * 1000, "e2el": latency * 1000}
        good += all(observed[name] <= slo for name, slo in slo_ms.items())
    return good


class Runner:
    def __init__(self, args):
        self.a = args
        self.out = Path(args.out).expanduser()
        (self.out / "cells").mkdir(parents=True, exist_ok=True)
        self.server = None
        self.server_tag = None
        self.t0 = time.time()
        self.rng = random.Random(0)

    # ---- bookkeeping -------------------------------------------------------
    def say(self, msg):
        line = f"[{time.strftime('%H:%M:%S')} +{int(time.time() - self.t0) // 60:02d}m] {msg}"
        print(line, flush=True)
        with open(self.out / "progress.log", "a") as f:
            f.write(line + "\n")

    def out_of_time(self):
        if (self.out / "STOP").exists():
            self.say("STOP file found")
            return True
        if self.a.deadline_epoch and time.time() + CELL_BUDGET_S > self.a.deadline_epoch:
            self.say("deadline too close for another cell")
            return True
        return False

    def record(self, row):
        with open(self.out / "summary.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

    # ---- server ------------------------------------------------------------
    def start_server(self, tag, extra=(), env=None):
        self.stop_server()
        cmd = [
            sys.executable, "-m", "sglang.launch_server",
            "--model-path", self.a.model, "--host", "127.0.0.1", "--port", str(self.a.port),
            *extra,
        ]  # fmt: skip
        self.say(f"server[{tag}] start: {' '.join(extra) or '<defaults>'} env={env or {}}")
        self.server_tag = tag
        if self.a.dry_run:
            return True
        log = open(self.out / f"server_{tag}.log", "w")
        self.server = subprocess.Popen(
            cmd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})}
        )
        t_start = time.time()
        while time.time() - t_start < 900:
            if self.server.poll() is not None:
                self.say(f"server[{tag}] DIED rc={self.server.returncode}")
                return False
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{self.a.port}/health", timeout=3
                ) as r:
                    if r.status == 200:
                        self.say(f"server[{tag}] healthy after {int(time.time() - t_start)} s")
                        return True
            except Exception:
                pass
            time.sleep(5)
        self.say(f"server[{tag}] HEALTH TIMEOUT")
        return False

    def stop_server(self):
        if self.server is None:
            return
        self.server.terminate()
        try:
            self.server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self.server.kill()
            self.server.wait()
        subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], check=False)
        time.sleep(5)
        self.server = None

    # ---- one benchmark cell ------------------------------------------------
    def bench_cmd(self, cell, *, rate, num_prompts, slo_ms, dataset, seed, max_concurrency):
        cmd = [
            sys.executable, "-m", "sglang.bench_serving",
            "--backend", "sglang-oai-chat", "--host", "127.0.0.1", "--port", str(self.a.port),
            "--model", self.a.model, *dataset,
            "--num-prompts", str(num_prompts), "--request-rate", str(rate),
            "--seed", str(seed), "--disable-tqdm", "--tag", cell,
            "--output-file", str(self.out / "cells" / f"{cell}.jsonl"), "--output-details",
            "--goodput", *slo_args(slo_ms),
        ]  # fmt: skip
        if max_concurrency:
            cmd += ["--max-concurrency", str(max_concurrency)]
        return cmd

    def fake_result(self, rate, num_prompts, slo_ms):
        n = num_prompts
        ttfts = [self.rng.uniform(0.05, 3.0) for _ in range(n)]
        lens = [self.rng.randint(1, 300) for _ in range(n)]
        itls = [[0.02] * (k - 1) for k in lens]
        errors = ["" if self.rng.random() > 0.02 else "boom" for _ in range(n)]
        res = {"ttfts": ttfts, "itls": itls, "output_lens": lens, "errors": errors}
        dur = max(n / max(rate, 1e-9), 1.0) if rate != float("inf") else 30.0
        good = recompute_good(res, slo_ms)
        res.update(
            duration=dur, completed=sum(not e for e in errors),
            request_throughput=n / dur, output_throughput=sum(lens) / dur,
            request_goodput=good / dur, slo_attainment=good / n,
            slo_attainment_by_metric={k: 0.5 for k in slo_ms},
            mean_ttft_ms=1.0, p99_ttft_ms=2.0, mean_tpot_ms=20.0, p99_tpot_ms=30.0,
            mean_e2e_latency_ms=1.0, p99_e2e_latency_ms=2.0, max_concurrent_requests=1,
        )  # fmt: skip
        return res

    def finish_cell(self, cell, task, params, slo_ms, proc, rate, num_prompts):
        path = self.out / "cells" / f"{cell}.jsonl"
        if self.a.dry_run:
            result = self.fake_result(rate, num_prompts, slo_ms)
        else:
            rc = proc.wait()
            if rc != 0 or not path.exists():
                self.say(f"cell {cell}: bench FAILED rc={rc}")
                self.record({"cell": cell, "task": task, "params": params, "failed": True})
                return None
            result = json.loads(path.read_text().strip().splitlines()[-1])
        good_printed = round(result["request_goodput"] * result["duration"])
        good_recomputed = recompute_good(result, slo_ms)
        keys = (
            "duration", "completed", "request_throughput", "output_throughput",
            "request_goodput", "slo_attainment", "slo_attainment_by_metric",
            "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "p99_tpot_ms",
            "mean_e2e_latency_ms", "p99_e2e_latency_ms", "max_concurrent_requests",
        )  # fmt: skip
        row = {
            "cell": cell, "task": task, "server": self.server_tag, "params": params,
            "slo_ms": slo_ms, "sent": len(result["ttfts"]),
            "failed_requests": sum(bool(e) for e in result["errors"]),
            "good_printed": good_printed, "good_recomputed": good_recomputed,
            **{k: result.get(k) for k in keys},
        }  # fmt: skip
        self.record(row)
        self.say(
            f"cell {cell}: thr {row['request_throughput']:.2f} req/s, "
            f"{row['output_throughput']:.0f} tok/s | goodput {row['request_goodput']:.2f} req/s, "
            f"attain {row['slo_attainment'] * 100:.1f}% {row['slo_attainment_by_metric']} | "
            f"good printed/recomputed {good_printed}/{good_recomputed} | failed {row['failed_requests']}"
        )
        return row

    def cell(self, cell, task, *, rate, num_prompts, params, slo_ms=SLO_MS,
             dataset=("--dataset-name", "sharegpt"), seed=1, max_concurrency=None):  # fmt: skip
        cmd = self.bench_cmd(cell, rate=rate, num_prompts=num_prompts, slo_ms=slo_ms,
                             dataset=dataset, seed=seed, max_concurrency=max_concurrency)  # fmt: skip
        proc = None
        if not self.a.dry_run:
            log = open(self.out / "cells" / f"{cell}.log", "w")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        return self.finish_cell(cell, task, params, slo_ms, proc, rate, num_prompts)

    # ---- tasks -------------------------------------------------------------
    def t1(self):
        if not self.start_server("default"):
            return None
        probe = self.cell("t1_probe_c256", "T1", rate=float("inf"), num_prompts=1500,
                          max_concurrency=256, params={"max_concurrency": 256})  # fmt: skip
        if probe is None:
            return None
        c0 = probe["request_throughput"]
        self.say(f"capacity probe: c0 = {c0:.2f} req/s at concurrency 256")
        rows = []
        for mult in SWEEP:
            if self.out_of_time():
                break
            rate = round(c0 * mult, 3)
            row = self.cell(f"t1_rate_x{mult:g}", "T1", rate=rate,
                            num_prompts=max(200, round(rate * ARRIVAL_SPAN_S)),
                            params={"mult": mult, "rate": rate})  # fmt: skip
            if row:
                rows.append(row)
        if rows:
            best = max(rows, key=lambda r: r["request_goodput"])
            mult, rate = best["params"]["mult"], best["params"]["rate"]
            for seed in (2, 3):
                if self.out_of_time():
                    break
                self.cell(f"t1_rate_x{mult:g}_seed{seed}", "T1-repeat", rate=rate, seed=seed,
                          num_prompts=max(200, round(rate * ARRIVAL_SPAN_S)),
                          params={"mult": mult, "rate": rate, "seed": seed})  # fmt: skip
        return c0

    def t2(self, c0):
        rate = round(c0 * 1.25, 3)
        for cap in (32, 128, 512):
            if self.out_of_time():
                return
            if self.start_server(f"cap{cap}", extra=("--max-running-requests", str(cap))):
                self.cell(f"t2_cap{cap}", "T2", rate=rate,
                          num_prompts=max(200, round(rate * ARRIVAL_SPAN_S)),
                          params={"max_running_requests": cap, "rate": rate})  # fmt: skip

    def t3(self, c0):
        short = ("--dataset-name", "random", "--random-input-len", "256",
                 "--random-output-len", "128", "--random-range-ratio", "0.5")  # fmt: skip
        long_ = ("--dataset-name", "random", "--random-input-len", "12000",
                 "--random-output-len", "256", "--random-range-ratio", "0.9")  # fmt: skip
        short_rate, long_rate = round(c0 * 0.6, 3), 2.0
        for policy in ("fcfs", "hrrn"):
            if self.out_of_time():
                return
            if not self.start_server(f"policy_{policy}", extra=("--schedule-policy", policy)):
                continue
            n_short = max(200, round(short_rate * ARRIVAL_SPAN_S))
            n_long = round(long_rate * ARRIVAL_SPAN_S)
            cells = [
                (f"t3_{policy}_short", short_rate, n_short, SHORT_SLO_MS, short, 1),
                (f"t3_{policy}_long", long_rate, n_long, LONG_SLO_MS, long_, 2),
            ]
            procs = []
            for name, rate, n, slo, dataset, seed in cells:
                cmd = self.bench_cmd(name, rate=rate, num_prompts=n, slo_ms=slo,
                                     dataset=dataset, seed=seed, max_concurrency=None)  # fmt: skip
                proc = None
                if not self.a.dry_run:
                    log = open(self.out / "cells" / f"{name}.log", "w")
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
                procs.append(proc)
            for (name, rate, n, slo, _, _), proc in zip(cells, procs):
                self.finish_cell(name, "T3", {"policy": policy, "rate": rate}, slo, proc, rate, n)

    def t4(self, c0):
        rate = round(c0 * 1.5, 3)
        for timeout in (None, 2, 10):
            if self.out_of_time():
                return
            env = {} if timeout is None else {"SGLANG_REQ_WAITING_TIMEOUT": str(timeout)}
            tag = f"wt{'off' if timeout is None else timeout}"
            if self.start_server(tag, extra=("--max-running-requests", "128"), env=env):
                self.cell(f"t4_{tag}", "T4", rate=rate,
                          num_prompts=max(200, round(rate * ARRIVAL_SPAN_S)),
                          params={"waiting_timeout_s": timeout, "rate": rate,
                                  "max_running_requests": 128})  # fmt: skip

    def run(self):
        tasks = self.a.only.split(",") if self.a.only else ["T1", "T2", "T3", "T4"]
        c0 = self.a.c0
        try:
            if "T1" in tasks:
                c0 = self.t1() or c0
            if not c0:
                self.say("no capacity estimate; stopping")
                return 1
            for name, fn in (("T2", self.t2), ("T3", self.t3), ("T4", self.t4)):
                if name in tasks and not self.out_of_time():
                    fn(c0)
        finally:
            self.stop_server()
            (self.out / ".finished").touch()
            self.say("runner finished")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    p.add_argument("--only", default="", help="comma list of T1,T2,T3,T4")
    p.add_argument("--c0", type=float, default=0.0, help="capacity (req/s) when T1 is skipped")
    p.add_argument("--dry-run", action="store_true")
    sys.exit(Runner(p.parse_args()).run())


if __name__ == "__main__":
    main()
