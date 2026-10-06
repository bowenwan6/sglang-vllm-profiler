#!/usr/bin/env python3
"""Debug ladder for the per-request waiting timeout (PRB_PLAN.md §7.3, D2 and D3).

The server must run with --max-running-requests 1. One long greedy request holds the only slot
while bounded requests queue behind it.

  python prb_ladder.py --model Qwen/Qwen3-8B --mode main    --out ladder_main.json
  python prb_ladder.py --model Qwen/Qwen3-8B --mode global2 --out ladder_global2.json   # server has SGLANG_REQ_WAITING_TIMEOUT=2
  python prb_ladder.py --model Qwen/Qwen3-8B --mode control --out ladder_control.json   # unpatched server

Prints one line per check and exits 1 if any check fails. Every request is greedy: the first sampled
request on a fresh server stalls the scheduler (results/pra_usage.md §3.2).
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

MSG = "Request waiting timeout reached."
SLACK_S = 0.6  # a bound is noticed at the next scheduler step; allow this much on top


class Ladder:
    def __init__(self, a):
        self.a = a
        self.rows = []
        self.session = None

    def check(self, name, ok, detail):
        self.rows.append({"check": name, "ok": bool(ok), "detail": detail})
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)

    def native(self, **kw):
        body = {
            "text": "hi",
            "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True},
        }  # fmt: skip
        body["sampling_params"].update(kw.pop("sampling", {}))
        body.update(kw)
        return "/generate", body

    def chat(self, **kw):
        body = {
            "model": self.a.model, "temperature": 0, "max_tokens": 16,
            "messages": [{"role": "user", "content": "hi"}],
        }  # fmt: skip
        body.update(kw)
        return "/v1/chat/completions", body

    def completions(self, **kw):
        body = {"model": self.a.model, "prompt": "hi", "temperature": 0, "max_tokens": 16}
        body.update(kw)
        return "/v1/completions", body

    async def call(self, path, body):
        """Returns (http status, elapsed s, parsed body or list of stream events)."""
        t = time.perf_counter()
        async with self.session.post(self.a.base_url + path, json=body) as resp:
            if body.get("stream"):
                events = []
                async for raw in resp.content:
                    line = raw.strip()
                    if line.startswith(b"data:") and line[5:].strip() != b"[DONE]":
                        events.append(json.loads(line[5:]))
                return resp.status, time.perf_counter() - t, events
            text = await resp.text()
            try:
                data = json.loads(text)
            except ValueError:
                data = {"raw": text[:300]}
            return resp.status, time.perf_counter() - t, data

    async def hold(self, tokens):
        path, body = self.native(rid="hold-000001", text="hold", sampling={"max_new_tokens": tokens})
        return await self.call(path, body)

    def aborted_in_stream(self, events):
        """(carried the 503 abort, produced a token) for either stream dialect."""
        abort, token = False, False
        for e in events:
            err = e.get("error")
            if isinstance(err, dict) and err.get("code") == 503 and err.get("message") == MSG:
                abort = True
            meta = e.get("meta_info") or {}
            fr = meta.get("finish_reason")
            if isinstance(fr, dict) and fr.get("type") == "abort" and fr.get("status_code") == 503:
                abort = abort or fr.get("message") == MSG
            if (meta.get("completion_tokens") or 0) > 0:
                token = True
            for ch in e.get("choices") or []:
                delta = ch.get("delta") or {}
                if delta.get("content") or ch.get("text"):
                    token = True
        return abort, token

    async def bounded(self, name, req, bound, holder, lo=None):
        """A queued request with a bound must be refused after about `bound` seconds."""
        path, body = req
        status, dt, data = await self.call(path, body)
        lo = bound if lo is None else lo
        timely = lo - 0.05 <= dt <= lo + SLACK_S
        held = not holder.done()
        if body.get("stream"):
            abort, token = self.aborted_in_stream(data)
            ok = status == 200 and abort and not token and timely and held
            detail = f"http {status}, abort event {abort}, token {token}, after {dt:.2f} s"
        else:
            ok = status == 503 and data.get("message") == MSG and timely and held
            detail = f"http {status}, message {data.get('message')!r}, after {dt:.2f} s"
        if not held:
            detail += " — INCONCLUSIVE: the slot was already free"
        self.check(name, ok, detail)

    async def main_mode(self):
        a = self.a
        status, dt, data = await self.call(*self.native(sampling={"max_new_tokens": 64}))
        n = (data.get("meta_info") or {}).get("completion_tokens")
        self.check("R0 idle server, no field", status == 200 and n == 64, f"http {status}, {n} tokens in {dt:.2f} s")
        rate = 64 / max(dt, 1e-3)
        hold_tokens = int(min(a.max_hold_tokens, max(2000, rate * a.hold_s)))
        print(f"      {rate:.0f} tok/s alone; the holder generates {hold_tokens} tokens", flush=True)

        holder = asyncio.create_task(self.hold(hold_tokens))
        await asyncio.sleep(1.5)
        b = a.bound
        await self.bounded("R2 /generate, non-streaming", self.native(rid="bound-000002", waiting_timeout=b), b, holder)
        await self.bounded("R3 /generate, streaming", self.native(rid="bound-000003", waiting_timeout=b, stream=True), b, holder)
        await self.bounded("R4 chat, non-streaming", self.chat(waiting_timeout=b), b, holder)
        await self.bounded("R5 chat, streaming", self.chat(waiting_timeout=b, stream=True), b, holder)
        await self.bounded("R6 completions, non-streaming", self.completions(waiting_timeout=b), b, holder)
        await self.bounded("R6b completions, streaming", self.completions(waiting_timeout=b, stream=True), b, holder)

        for bad in (0, -1, "abc", 10**400):
            path, body = self.native(rid="valid-000009", waiting_timeout=bad)
            status, dt, data = await self.call(path, body)
            self.check(f"R9 /generate rejects waiting_timeout={str(bad)[:8]}", 400 <= status < 500 and dt < 1.0,
                       f"http {status} after {dt:.2f} s: {str(data.get('message') or data)[:90]}")  # fmt: skip
        status, dt, data = await self.call(*self.chat(waiting_timeout=0))
        self.check("R9 chat rejects waiting_timeout=0", 400 <= status < 500 and dt < 1.0, f"http {status} after {dt:.2f} s")

        loose = asyncio.create_task(self.call(*self.native(rid="loose-000007", waiting_timeout=3600)))
        free = asyncio.create_task(self.call(*self.native(rid="free-0000008")))
        await asyncio.sleep(2.0)
        self.check("R7/R8 still queued behind the holder after 2 s", not loose.done() and not free.done() and not holder.done(),
                   f"loose done {loose.done()}, unbounded done {free.done()}, holder done {holder.done()}")  # fmt: skip
        status, dt, data = await holder
        meta = data.get("meta_info") or {}
        fr = meta.get("finish_reason") or {}
        self.check("R1 the holder is undisturbed", status == 200 and meta.get("completion_tokens") == hold_tokens and fr.get("type") == "length",
                   f"http {status}, {meta.get('completion_tokens')} of {hold_tokens} tokens, finish {fr.get('type')}, {dt:.1f} s")  # fmt: skip
        for name, task in (("R7 loose bound (3600 s)", loose), ("R8 no bound", free)):
            status, dt, data = await task
            n = (data.get("meta_info") or {}).get("completion_tokens")
            self.check(f"{name} is served once the slot frees", status == 200 and n == 16, f"http {status}, {n} tokens after {dt:.1f} s")

    async def global2_mode(self):
        """Server started with SGLANG_REQ_WAITING_TIMEOUT=2: the smaller bound wins."""
        status, dt, data = await self.call(*self.native(sampling={"max_new_tokens": 64}))
        rate = 64 / max(dt, 1e-3)
        hold_tokens = int(min(self.a.max_hold_tokens, max(2000, rate * self.a.hold_s)))
        holder = asyncio.create_task(self.hold(hold_tokens))
        await asyncio.sleep(1.5)
        await self.bounded("G1 request 30 s, global 2 s", self.native(rid="glob-0000001", waiting_timeout=30), 2.0, holder)
        await self.bounded("G2 request 0.5 s, global 2 s", self.native(rid="glob-0000002", waiting_timeout=0.5), 0.5, holder)
        await self.bounded("G3 no field, global 2 s", self.native(rid="glob-0000003"), 2.0, holder)
        await self.bounded("G4 request 0.5 s, streaming chat", self.chat(waiting_timeout=0.5, stream=True), 0.5, holder)
        # Do not wait for the holder: the server is restarted after this mode.
        holder.cancel()

    async def control_mode(self):
        """Unpatched server: the field is ignored, the request waits for the slot."""
        status, dt, data = await self.call(*self.native(sampling={"max_new_tokens": 64}))
        rate = 64 / max(dt, 1e-3)
        hold_tokens = int(min(self.a.max_hold_tokens, max(2000, rate * 8)))
        holder = asyncio.create_task(self.hold(hold_tokens))
        await asyncio.sleep(1.5)
        path, body = self.native(rid="ctrl-0000002", waiting_timeout=self.a.bound)
        status, dt, data = await self.call(path, body)
        n = (data.get("meta_info") or {}).get("completion_tokens")
        self.check("C1 unpatched build ignores the field", status == 200 and n == 16 and dt > self.a.bound + SLACK_S,
                   f"http {status}, {n} tokens after {dt:.1f} s (bound {self.a.bound} s)")  # fmt: skip
        await holder

    async def run(self):
        timeout = aiohttp.ClientTimeout(total=None, sock_read=600)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            self.session = session
            # Unmeasured: the first request on a fresh server is slower than the rest.
            await self.call(*self.native())
            await {"main": self.main_mode, "global2": self.global2_mode, "control": self.control_mode}[self.a.mode]()
            async with session.get(self.a.base_url + "/health") as resp:
                self.check("server healthy at the end", resp.status == 200, f"http {resp.status}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", required=True)
    p.add_argument("--mode", choices=("main", "global2", "control"), default="main")
    p.add_argument("--bound", type=float, default=1.0)
    p.add_argument("--hold-s", type=float, default=30.0, help="how long the holder should keep the slot")
    p.add_argument("--max-hold-tokens", type=int, default=20000)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    ladder = Ladder(a)
    try:
        asyncio.run(ladder.run())
    except Exception as e:
        ladder.check("ladder ran to the end", False, f"{type(e).__name__}: {e}")
    Path(a.out).write_text(json.dumps({"mode": a.mode, "model": a.model, "rows": ladder.rows}, indent=1))
    failed = [r["check"] for r in ladder.rows if not r["ok"]]
    print(f"{a.mode}: {len(ladder.rows) - len(failed)} of {len(ladder.rows)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
