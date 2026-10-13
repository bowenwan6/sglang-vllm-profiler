#!/usr/bin/env python3
"""PR-C check on a live server: sglang:num_good_requests_total against an exact recount.

Runs on the node inside the sgl-profiler env, after prc_node.sh has installed the branch
head. A server is started with --enable-metrics and --goodput-slo; a closed-loop client
sends native /generate requests (half of them streaming) in phases of rising concurrency.
Every finished response carries the server's own first_token_latency, e2e_latency and
decode_throughput in meta_info, the values the counter is computed from, so the number of
good requests can be recounted on the client. After each phase the growth of the counter
on /metrics must equal the recount, separately for is_streaming true and false.

Servers: three SLOs; one SLO; a queue limit that rejects part of a burst (a rejected
request is aborted and must not count); no --goodput-slo (the counter must stay empty);
three SLOs with two tokenizer workers.

  python prc_verify.py --out ~/sgl/logs/prc --deadline-epoch <unix time to stop by>

Stop cleanly between servers with: touch <out>/STOP
"""

import argparse
import asyncio
import json
import random
import re
import sys
import time
import urllib.request

import aiohttp
from p1_bench import Runner

SLO3 = {"ttft": 1000, "tpot": 40, "e2el": 8000}
SLO1 = {"tpot": 20}
LENIENT = {"e2el": 600000}
# (concurrency, requests)
PHASES = ((8, 300), (64, 900), (256, 1500), (1024, 2500))
SLO1_PHASES = ((8, 300), (64, 900), (512, 2000))
BURST = ((256, 600),)
QUEUE_LIMIT = ["--max-running-requests", "32", "--max-queued-requests", "16"]
TWO_WORKERS = ["--tokenizer-worker-num", "2"]
# A request this close to an SLO may fall on either side of it in the recount:
# TPOT is rebuilt as 1 / decode_throughput, which can differ in the last bit.
BOUNDARY_MS = 1e-6
WORDS = "the of and to in is that for it as was with be by on not he this are or his from at which but have an had they you were their one all we can her has there been if more when will would who so no".split()  # fmt: skip
SAMPLE = re.compile(r'^(sglang:num_(?:good_)?requests_total)\{(.*)\} ([0-9.e+]+)$')  # fmt: skip


def make_requests(n, seed):
    rng = random.Random(seed)
    reqs = []
    for i in range(n):
        r = rng.random()
        max_new = 1 if r < 0.10 else 2 if r < 0.15 else rng.randint(16, 200)
        reqs.append(
            {
                "text": " ".join(rng.choices(WORDS, k=rng.randint(40, 400))) + f" [{seed}-{i}]",
                "max_new_tokens": max_new,
                "stream": rng.random() < 0.5,
            }
        )  # fmt: skip
    return reqs


async def send_one(session, url, req, sem, results):
    payload = {
        "text": req["text"],
        "sampling_params": {"temperature": 0, "max_new_tokens": req["max_new_tokens"]},
        "stream": req["stream"],
    }
    rec = {"stream": req["stream"], "meta": None, "error": None}
    async with sem:
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    rec["error"] = f"http {resp.status}"
                elif not req["stream"]:
                    rec["meta"] = (await resp.json())["meta_info"]
                else:
                    async for raw in resp.content:
                        line = raw.decode().strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        meta = json.loads(line[5:]).get("meta_info")
                        if meta is None:
                            rec["error"] = "chunk without meta_info"
                        elif meta.get("finish_reason"):
                            rec["meta"] = meta
        except Exception as e:  # the record keeps what went wrong
            rec["error"] = f"{type(e).__name__}: {e}"[:200]
    results.append(rec)


async def run_phase(url, reqs, concurrency):
    sem = asyncio.Semaphore(concurrency)
    results = []
    conn = aiohttp.TCPConnector(limit=0)
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
        await asyncio.gather(*(send_one(session, url, r, sem, results) for r in reqs))
    return results


def observed_ms(meta):
    """The three latencies of a finished response in ms, as the server measured them."""
    tput = meta.get("decode_throughput")
    return {
        "ttft": meta["first_token_latency"] * 1000,
        "tpot": 1000 / tput if tput else 0.0,
        "e2el": meta["e2e_latency"] * 1000,
    }


def recount(results, slo_ms):
    """Good requests per is_streaming label, and how many sit on an SLO boundary."""
    good = {"true": 0, "false": 0}
    finished = {"true": 0, "false": 0}
    boundary = 0
    untimed = 0
    by_metric = dict.fromkeys(slo_ms, 0)
    for rec in results:
        meta = rec["meta"]
        if meta is None or (meta["finish_reason"] or {}).get("type") not in ("stop", "length"):
            continue  # fmt: skip
        label = "true" if rec["stream"] else "false"
        finished[label] += 1
        if "first_token_latency" not in meta or "e2e_latency" not in meta:
            untimed += 1
            continue
        obs = observed_ms(meta)
        met = {name: obs[name] <= slo for name, slo in slo_ms.items()}
        boundary += any(abs(obs[name] - slo) < BOUNDARY_MS for name, slo in slo_ms.items())
        good[label] += bool(slo_ms) and all(met.values())
        for name, ok in met.items():
            by_metric[name] += ok
    return good, finished, boundary + untimed, by_metric, untimed


def scrape(port):
    """{metric: {is_streaming: value}} summed over the other labels, plus the raw text."""
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        text = r.read().decode()
    out = {"sglang:num_good_requests_total": {}, "sglang:num_requests_total": {}}
    for line in text.splitlines():
        m = SAMPLE.match(line)
        if not m:
            continue
        stream = re.search(r'is_streaming="(\w+)"', m.group(2)).group(1)
        out[m.group(1)][stream] = out[m.group(1)].get(stream, 0.0) + float(m.group(3))
    return out, text


class Verify(Runner):
    def __init__(self, args):
        super().__init__(args)
        self.checks = []

    def check(self, name, ok, detail=""):
        self.checks.append({"check": name, "ok": bool(ok), "detail": detail})
        self.say(f"  {'ok  ' if ok else 'FAIL'} {name} {detail}")

    def run_server(self, tag, slo_ms, phases, seed, extra_args=(), expect_rejects=False):
        extra = ["--enable-metrics", *extra_args]
        if slo_ms is not None:
            extra += ["--goodput-slo", json.dumps(slo_ms)]
        if not self.start_server(tag, extra=extra):
            self.check(f"{tag}: server starts", False)
            return
        url = f"http://127.0.0.1:{self.a.port}/generate"
        # The first request on a fresh server is slow on this build; keep it out of the phases.
        asyncio.run(run_phase(url, make_requests(8, seed=seed - 1), 8))
        before, _ = scrape(self.a.port)
        mixed = False
        for i, (concurrency, n) in enumerate(phases):
            results = asyncio.run(run_phase(url, make_requests(n, seed + i), concurrency))
            # A response can reach the client just before its request is counted.
            time.sleep(2)
            after, text = scrape(self.a.port)
            good, finished, boundary, by_metric, untimed = recount(results, slo_ms or {})
            delta = {
                metric: {
                    s: after[metric].get(s, 0.0) - before[metric].get(s, 0.0)
                    for s in ("true", "false")
                }
                for metric in after
            }
            row = {
                "server": tag, "slo_ms": slo_ms, "concurrency": concurrency, "sent": n,
                "finished": finished, "recount_good": good, "by_metric": by_metric,
                "counter_delta": delta, "boundary": boundary, "untimed": untimed,
                "errors": sum(r["error"] is not None for r in results),
                "not_finished": n - sum(finished.values()),
            }  # fmt: skip
            self.record(row)
            (self.out / "cells" / f"{tag}_c{concurrency}.json").write_text(
                json.dumps({"row": row, "results": results})
            )
            name = f"{tag} c={concurrency}"
            for s in ("true", "false"):
                got = delta["sglang:num_good_requests_total"][s]
                self.check(
                    f"{name}: counter == recount (is_streaming={s})",
                    abs(got - good[s]) <= boundary,
                    f"counter +{got:g}, recount {good[s]} of {finished[s]} finished",
                )
                total = delta["sglang:num_requests_total"][s]
                self.check(
                    f"{name}: good <= counted requests (is_streaming={s})",
                    got <= total and total >= finished[s],
                    f"num_requests_total +{total:g}",
                )
            self.check(f"{name}: every finished response has its latencies", untimed == 0, f"{untimed}")  # fmt: skip
            if expect_rejects:
                self.check(
                    f"{name}: part of the burst is rejected",
                    0 < row["not_finished"] < n,
                    f"{row['not_finished']} of {n} did not finish normally",
                )
            else:
                self.check(
                    f"{name}: every request finished normally",
                    row["not_finished"] == 0 and row["errors"] == 0,
                    f"{row['not_finished']} not finished, {row['errors']} errors",
                )
            all_good = sum(good.values())
            mixed = mixed or 0 < all_good < sum(finished.values())
            before = after
        if slo_ms is None:
            _, text = scrape(self.a.port)
            samples = [line for line in text.splitlines() if line.startswith("sglang:num_good_requests_total")]  # fmt: skip
            self.check(f"{tag}: no samples without --goodput-slo", not samples, f"{len(samples)} samples")  # fmt: skip
        elif not expect_rejects:
            self.check(f"{tag}: some phase has both good and late requests", mixed)
        (self.out / f"metrics_{tag}.txt").write_text(text)
        self.stop_server()

    def run(self):
        plan = (
            ("slo3", SLO3, PHASES, 100, (), False),
            ("slo1", SLO1, SLO1_PHASES, 200, (), False),
            ("reject", LENIENT, BURST, 300, QUEUE_LIMIT, True),
            ("noslo", None, PHASES[1:2], 400, (), False),
            ("workers2", SLO3, PHASES[1:], 500, TWO_WORKERS, False),
        )
        only = set(filter(None, self.a.only.split(",")))
        for tag, slo_ms, phases, seed, extra_args, expect_rejects in plan:
            if only and tag not in only:
                continue
            if self.out_of_time():
                self.check(f"{tag}: ran", False, "skipped, out of time")
                continue
            self.run_server(tag, slo_ms, phases, seed, extra_args, expect_rejects)
        self.stop_server()
        failed = [c for c in self.checks if not c["ok"]]
        summary = {"checks": len(self.checks), "failed": failed}
        (self.out / "verify_summary.json").write_text(json.dumps(summary, indent=1))
        self.say(f"VERIFY: {len(self.checks) - len(failed)} of {len(self.checks)} checks passed")
        return 1 if failed else 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    p.add_argument("--only", default="", help="comma list of server tags")
    p.add_argument("--dry-run", action="store_true")
    sys.exit(Verify(p.parse_args()).run())


if __name__ == "__main__":
    main()
