#!/usr/bin/env python3
"""Open-loop, seeded, multi-class client for the PR-B benchmark (PLAN.md stage 3).

  python slo_client.py --base-url http://127.0.0.1:30000 --config cfg.json --seed 1 --tag arm --out run.jsonl
  python slo_client.py --base-url ... --config cfg.json --closed-loop 128 --cls chat --num 1024 --out probe.jsonl

Config:
  {"horizon_s": 180,
   "classes": {
     "chat": {"prompt_tokens": 256, "max_new_tokens": 128,
              "period_s": 60, "segments": [[0, 40, 19.2], [40, 60, 96.0]],   # start, end, req/s
              "slo_ms": {"ttft": 2000, "tpot": 100},                          # same keys as --goodput
              "body": {"priority": 1, "waiting_timeout": 1.5},                # merged into the request
              "hangup_s": null}}}                                             # client gives up before a token

Every request is a streaming native /generate call with random input_ids, temperature 0 and
ignore_eos: a fixed amount of work, and no two prompts share a prefix. Arrival times and prompt
lengths depend only on the seed and on each class's segments, never on `body`, so arms are paired.
The token values also depend on --tag, so a later arm on the same server finds nothing in the cache.

A request is classified by what the stream says, not by the HTTP status: a server that drops a
streaming request answers 200 and puts the abort in the stream.

Writes one JSON line per request to --out and a summary to <out>.summary.json.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp

try:
    import orjson

    loads = orjson.loads
except ImportError:  # the Mac
    loads = json.loads

TOKEN_LO, TOKEN_HI = 1000, 100000  # ordinary vocabulary ids for Qwen3


def build_requests(cfg, seed):
    reqs = []
    horizon = cfg["horizon_s"]
    for name in sorted(cfg["classes"]):
        c = cfg["classes"][name]
        rng = random.Random(f"{seed}/{name}/arrivals")
        period = c.get("period_s", horizon)
        t0, n = 0.0, 0
        while t0 < horizon:
            for start, end, rate in c["segments"]:
                t = t0 + start
                while rate > 0:
                    t += rng.expovariate(rate)
                    if t >= t0 + end or t >= horizon:
                        break
                    reqs.append({"cls": name, "t": t, "i": n})
                    n += 1
            t0 += period
    reqs.sort(key=lambda r: r["t"])
    return reqs


def prompt_ids(seed, tag, cls, i, n):
    rng = random.Random(f"{seed}/{tag}/{cls}/{i}")
    return [rng.randrange(TOKEN_LO, TOKEN_HI) for _ in range(n)]


async def one(session, url, rec, body, t_base, hangup_s):
    """Send one request, fill `rec`. Times are seconds from t_base."""
    state = {"first": None}
    task = asyncio.current_task()
    loop = asyncio.get_running_loop()
    rec["t_send"] = time.perf_counter() - t_base
    n_tok, t_last, finish = 0, None, None
    watchdog = None
    if hangup_s is not None:
        watchdog = loop.call_later(
            hangup_s, lambda: state["first"] is None and task.cancel()
        )
    try:
        async with session.post(url, json=body) as resp:
            rec["http"] = resp.status
            if resp.status != 200:
                rec["status"] = "refused" if resp.status == 503 else "http_error"
                rec["message"] = (await resp.text())[:200]
                rec["t_end"] = time.perf_counter() - t_base
                return
            async for raw in resp.content:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                data = loads(payload)
                now = time.perf_counter() - t_base
                meta = data.get("meta_info") or {}
                n = meta.get("completion_tokens") or 0
                if n > 0 and state["first"] is None:
                    state["first"] = now
                if n > n_tok:
                    n_tok, t_last = n, now
                if meta.get("finish_reason"):
                    finish = meta["finish_reason"]
                    rec["prompt_tokens"] = meta.get("prompt_tokens")
                if "error" in data:  # the OpenAI-style error event, should one appear
                    finish = {"type": "abort", "message": str(data["error"])[:200]}
        rec["t_end"] = time.perf_counter() - t_base
        if isinstance(finish, dict) and finish.get("type") == "abort":
            rec["status"] = "aborted"
            rec["abort_code"] = finish.get("status_code")
            rec["message"] = finish.get("message")
        elif finish is None:
            rec["status"] = "incomplete"
        else:
            rec["status"] = "ok"
    except asyncio.CancelledError:
        rec["t_end"] = time.perf_counter() - t_base
        rec["status"] = "hung_up" if hangup_s is not None else "client_timeout"
    except Exception as e:  # a failed request is a data point, not a crash
        rec["t_end"] = time.perf_counter() - t_base
        rec["status"] = "error"
        rec["message"] = f"{type(e).__name__}: {e}"[:200]
    finally:
        if watchdog is not None:
            watchdog.cancel()
        rec["out_tokens"] = n_tok
        if state["first"] is not None:
            rec["ttft"] = state["first"] - rec["t_send"]
            rec["latency"] = t_last - rec["t_send"]


def make_body(cfg, seed, tag, r, rid):
    c = cfg["classes"][r["cls"]]
    return {
        "input_ids": prompt_ids(seed, tag, r["cls"], r["i"], c["prompt_tokens"]),
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": c["max_new_tokens"],
            "ignore_eos": True,
        },
        "stream": True,
        "rid": rid,
        **c.get("body", {}),
    }


async def open_loop(a, cfg):
    reqs = build_requests(cfg, a.seed)
    recs, tasks = [], []
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.read_timeout)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), timeout=timeout
    ) as session:
        url = a.base_url + "/generate"
        t_base = time.perf_counter()
        for k, r in enumerate(reqs):
            delay = r["t"] - (time.perf_counter() - t_base)
            if delay > 0:
                await asyncio.sleep(delay)
            # Fixed width: the server aborts by rid prefix.
            rid = f"{a.tag}-s{a.seed}-{r['cls'][0]}{k:07d}"
            rec = {"rid": rid, "cls": r["cls"], "t_sched": r["t"]}
            recs.append(rec)
            body = make_body(cfg, a.seed, a.tag, r, rid)
            hangup = cfg["classes"][r["cls"]].get("hangup_s")
            tasks.append(
                asyncio.create_task(one(session, url, rec, body, t_base, hangup))
            )
        done, pending = await asyncio.wait(tasks, timeout=a.drain_timeout)
        for t in pending:
            t.cancel()
        if pending:
            await asyncio.wait(pending)
        duration = time.perf_counter() - t_base
    return recs, duration


async def closed_loop(a, cfg):
    c = cfg["classes"][a.cls]
    recs = []
    sem = asyncio.Semaphore(a.closed_loop)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=a.read_timeout)
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=0), timeout=timeout
    ) as session:
        url = a.base_url + "/generate"
        t_base = time.perf_counter()

        async def worker(k):
            async with sem:
                rid = f"{a.tag}-s{a.seed}-{a.cls[0]}{k:07d}"
                rec = {"rid": rid, "cls": a.cls, "t_sched": 0.0}
                recs.append(rec)
                r = {"cls": a.cls, "i": k}
                body = make_body(cfg, a.seed, a.tag, r, rid)
                body.pop("waiting_timeout", None)
                await one(session, url, rec, body, t_base, None)

        await asyncio.gather(*(worker(k) for k in range(a.num)))
        duration = time.perf_counter() - t_base
    return recs, duration


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round(p / 100 * (len(xs) - 1)))] if xs else None


def is_good(rec, slo_ms):
    if rec.get("status") != "ok" or "ttft" not in rec:
        return False, {k: False for k in slo_ms}
    n = rec["out_tokens"]
    tpot = (rec["latency"] - rec["ttft"]) / (n - 1) if n > 1 else 0.0
    obs = {"ttft": rec["ttft"] * 1e3, "tpot": tpot * 1e3, "e2el": rec["latency"] * 1e3}
    per = {k: obs[k] <= v for k, v in slo_ms.items()}
    return all(per.values()), per


def summarize(recs, cfg, duration):
    out = {"duration_s": duration, "classes": {}}
    total = {"sent": 0, "good": 0, "refused": 0, "out_tokens": 0, "wasted_tokens": 0}
    for name, c in cfg["classes"].items():
        rs = [r for r in recs if r["cls"] == name]
        if not rs:
            continue
        slo = c.get("slo_ms", {})
        good, per_slo, wasted = 0, {k: 0 for k in slo}, 0
        for r in rs:
            ok, per = is_good(r, slo)
            good += ok
            for k, v in per.items():
                per_slo[k] += v
            if not ok:
                wasted += r.get("out_tokens", 0)
        status = {}
        for r in rs:
            status[r.get("status", "unknown")] = status.get(r.get("status", "unknown"), 0) + 1
        served = [r for r in rs if r.get("status") == "ok" and "ttft" in r]
        ttfts = [r["ttft"] * 1e3 for r in served]
        tpots = [
            (r["latency"] - r["ttft"]) / (r["out_tokens"] - 1) * 1e3
            for r in served
            if r["out_tokens"] > 1
        ]
        refused = sum(
            1
            for r in rs
            if r.get("status") in ("aborted", "refused") and not r.get("out_tokens")
        )
        tokens = sum(r.get("out_tokens", 0) for r in rs)
        lags = [r["t_send"] - r["t_sched"] for r in rs if "t_send" in r]
        out["classes"][name] = {
            "sent": len(rs),
            "good": good,
            "attainment": good / len(rs),
            "attainment_by_slo": {k: v / len(rs) for k, v in per_slo.items()},
            "status": status,
            "refused": refused,
            "out_tokens": tokens,
            "wasted_tokens": wasted,
            "mean_ttft_ms": statistics.mean(ttfts) if ttfts else None,
            "p50_ttft_ms": pct(ttfts, 50),
            "p99_ttft_ms": pct(ttfts, 99),
            "mean_tpot_ms": statistics.mean(tpots) if tpots else None,
            "p99_tpot_ms": pct(tpots, 99),
            "mean_e2e_ms": statistics.mean(r["latency"] * 1e3 for r in served) if served else None,
            "p99_e2e_ms": pct([r["latency"] * 1e3 for r in served], 99),
            "max_send_lag_ms": max(lags) * 1e3 if lags else None,
            "prompt_tokens_seen": next((r["prompt_tokens"] for r in rs if r.get("prompt_tokens")), None),
        }  # fmt: skip
        total["sent"] += len(rs)
        total["good"] += good
        total["refused"] += refused
        total["out_tokens"] += tokens
        total["wasted_tokens"] += wasted
    total["attainment"] = total["good"] / max(total["sent"], 1)
    total["good_per_s"] = total["good"] / duration
    total["req_per_s"] = sum(1 for r in recs if r.get("status") == "ok") / duration
    total["out_tokens_per_s"] = total["out_tokens"] / duration
    out["all"] = total
    return out


async def warmup(a, cfg):
    """A few small greedy requests so the first timed one meets a warm server."""
    async with aiohttp.ClientSession() as session:
        for k in range(a.warmup):
            body = {
                "input_ids": prompt_ids(a.seed, a.tag + "/warm", "w", k, 64),
                "sampling_params": {"temperature": 0, "max_new_tokens": 8, "ignore_eos": True},
            }  # fmt: skip
            async with session.post(a.base_url + "/generate", json=body) as resp:
                await resp.read()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="run")
    p.add_argument("--out", required=True)
    p.add_argument("--closed-loop", type=int, default=0, help="concurrency of a closed-loop probe")
    p.add_argument("--cls", default="", help="class for --closed-loop")
    p.add_argument("--num", type=int, default=0, help="requests for --closed-loop")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--read-timeout", type=float, default=600.0)
    p.add_argument("--drain-timeout", type=float, default=900.0)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())

    async def go():
        if a.warmup:
            await warmup(a, cfg)
        return await (closed_loop(a, cfg) if a.closed_loop else open_loop(a, cfg))

    recs, duration = asyncio.run(go())
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    summary = summarize(recs, cfg, duration)
    summary.update(tag=a.tag, seed=a.seed, closed_loop=a.closed_loop)
    Path(str(out) + ".summary.json").write_text(json.dumps(summary, indent=1))
    al = summary["all"]
    print(
        f"{a.tag} seed {a.seed}: sent {al['sent']} good {al['good']} "
        f"({al['attainment'] * 100:.1f} %) refused {al['refused']} "
        f"{al['req_per_s']:.2f} req/s {al['out_tokens_per_s']:.0f} tok/s in {duration:.1f} s"
    )
    for name, c in summary["classes"].items():
        print(f"  {name}: {c['good']}/{c['sent']} = {c['attainment'] * 100:.1f} %  {c['status']}  "
              f"ttft mean {c['mean_ttft_ms'] and round(c['mean_ttft_ms'])} ms")  # fmt: skip


if __name__ == "__main__":
    main()
