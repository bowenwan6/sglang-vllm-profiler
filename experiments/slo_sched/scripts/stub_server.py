#!/usr/bin/env python3
"""A stand-in for the SGLang server, for testing slo_client.py and the runners without a GPU.

  python stub_server.py --port 30011 --slots 4 --tpot-ms 2 [--global-timeout 1.5] [--priority]

It speaks just enough of /generate (streaming and not), /health and /flush_cache: a fixed number of
slots, a waiting queue in arrival order (or by `priority`, higher first), and the waiting bound of
PRB_PLAN.md §2.6 with the abort shapes captured in results/pra/abort_responses/.
"""

import argparse
import asyncio
import json
import sys

from aiohttp import web

MSG = "Request waiting timeout reached."


class Stub:
    def __init__(self, a):
        self.a = a
        self.free = a.slots
        self.waiting = []
        self.seq = 0

    def kick(self):
        now = asyncio.get_running_loop().time()
        for e in list(self.waiting):
            if e["bound"] is not None and now - e["t"] > e["bound"]:
                self.waiting.remove(e)
                e["fut"].set_result(False)
        while self.free > 0 and self.waiting:
            e = min(self.waiting, key=lambda e: (-e["prio"] if self.a.priority else 0, e["seq"]))  # fmt: skip
            self.waiting.remove(e)
            self.free -= 1
            e["fut"].set_result(True)

    async def scan(self, app):
        async def loop():
            while True:
                await asyncio.sleep(0.005)
                self.kick()

        task = asyncio.create_task(loop())
        yield
        task.cancel()

    async def generate(self, request):
        body = await request.json()
        openai = request.path != "/generate"
        sp = body.get("sampling_params") or {}
        n_out = int(body.get("max_tokens", 16) if openai else sp.get("max_new_tokens", 16))
        n_in = len(body.get("input_ids") or [])
        bound = body.get("waiting_timeout")
        if bound is not None and not (
            isinstance(bound, (int, float)) and 0 < bound <= sys.float_info.max
        ):
            return web.json_response(
                {"object": "error", "message": "waiting_timeout should be a positive, finite number of seconds.", "code": 400}, status=400
            )  # fmt: skip
        g = self.a.global_timeout
        if bound is None or (g is not None and g < bound):
            bound = g
        self.seq += 1
        e = {
            "seq": self.seq,
            "prio": body.get("priority") or 0,
            "t": asyncio.get_running_loop().time(),
            "bound": bound,
            "fut": asyncio.get_running_loop().create_future(),
        }
        self.waiting.append(e)
        self.kick()
        try:
            started = await e["fut"]
        except asyncio.CancelledError:  # the client hung up while queued
            if e in self.waiting:
                self.waiting.remove(e)
            elif e["fut"].done() and e["fut"].result():
                self.free += 1
            raise
        rid = body.get("rid", "stub")
        abort = {"type": "abort", "status_code": 503, "message": MSG}
        if not body.get("stream"):
            if not started:
                return web.json_response(
                    {"object": "error", "message": MSG, "type": "503", "code": 503}, status=503
                )  # fmt: skip
            try:
                await asyncio.sleep(self.a.prefill_ms / 1e3 + n_out * self.a.tpot_ms / 1e3)
            finally:
                self.free += 1
                self.kick()
            return web.json_response(
                {"text": "x" * n_out, "meta_info": {"id": rid, "completion_tokens": n_out,
                 "prompt_tokens": n_in, "finish_reason": {"type": "length", "length": n_out}}}
            )  # fmt: skip
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await resp.prepare(request)

        async def send(obj):
            await resp.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if not started and openai:
            await send({"error": {"object": "error", "message": MSG,
                                  "type": "SERVICE_UNAVAILABLE", "param": None, "code": 503}})  # fmt: skip
        elif not started:
            await send({"text": "", "output_ids": [], "meta_info": {
                "id": rid, "finish_reason": abort, "completion_tokens": 0}})  # fmt: skip
        elif openai:
            try:
                await asyncio.sleep(self.a.prefill_ms / 1e3 + n_out * self.a.tpot_ms / 1e3)
                await send({"choices": [{"index": 0, "delta": {"content": "x"}, "text": "x"}]})
            finally:
                self.free += 1
                self.kick()
        else:
            try:
                await asyncio.sleep(self.a.prefill_ms / 1e3)
                for n in range(1, n_out + 1):
                    await asyncio.sleep(self.a.tpot_ms / 1e3)
                    done = n == n_out
                    await send({"text": "x" * n, "output_ids": [], "meta_info": {
                        "id": rid, "completion_tokens": n, "prompt_tokens": n_in,
                        "finish_reason": {"type": "length", "length": n_out} if done else None}})  # fmt: skip
            finally:
                self.free += 1
                self.kick()
        await resp.write(b"data: [DONE]\n\n")
        return resp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30011)
    p.add_argument("--slots", type=int, default=4)
    p.add_argument("--tpot-ms", type=float, default=2.0)
    p.add_argument("--prefill-ms", type=float, default=5.0)
    p.add_argument("--global-timeout", type=float, default=None)
    p.add_argument("--priority", action="store_true")
    a = p.parse_args()
    stub = Stub(a)
    app = web.Application()
    app.cleanup_ctx.append(stub.scan)
    app.router.add_post("/generate", stub.generate)
    app.router.add_post("/v1/chat/completions", stub.generate)
    app.router.add_post("/v1/completions", stub.generate)
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    app.router.add_post("/flush_cache", lambda r: web.Response(text="ok"))
    web.run_app(app, host="127.0.0.1", port=a.port, print=None, handler_cancellation=True)


if __name__ == "__main__":
    main()
