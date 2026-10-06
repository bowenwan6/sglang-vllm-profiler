#!/usr/bin/env python3
"""Runner for the PR-B sessions (PLAN.md stage 3 and its amendment of 2026-10-06).

Runs on the node inside the sgl-profiler env, next to p1_bench.py (whose server handling it reuses),
slo_client.py and prb_ladder.py.

  python prb_run.py --out ~/sgl/logs/p2 --deadline-epoch <unix time> --phases ladder,calib,check,pilot,u3,u4,t1rep,t4rep
  python prb_run.py --out ~/sgl/logs/p3 --deadline-epoch <unix time> --phases u1,u2 --c-chat 60 --c-batch 30
  python prb_run.py --out /tmp/x --dry-run --phases ...           # control flow only, against stub_server.py

Two source trees: the installed one (patched) and --base-tree (the same commit without the patch),
selected per server with PYTHONPATH. Stop cleanly between cells with: touch <out>/STOP
"""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import p1_bench

HERE = Path(__file__).resolve().parent
CHAT = {"prompt_tokens": 256, "max_new_tokens": 128, "slo_ms": {"ttft": 2000, "tpot": 100}}
BATCH = {"prompt_tokens": 1024, "max_new_tokens": 256, "slo_ms": {"e2el": 40000}}
CHAT_BOUND, BATCH_BOUND = 1.5, 30.0
CAP = ("--max-running-requests", "128")
PRIORITY = ("--enable-priority-scheduling", "--disable-priority-preemption")
P1_C0 = 40.222  # req/s; session P1's capacity probe, kept so the PR-A repeats use P1's rates


def config(horizon, chat=None, batch=None, period=60):
    """chat / batch: (segments, body, hangup_s) or None."""
    classes = {}
    for name, base, spec in (("chat", CHAT, chat), ("batch", BATCH, batch)):
        if spec is None:
            continue
        segments, body, hangup = spec
        classes[name] = {**base, "segments": segments, "period_s": period, "body": body}
        if hangup is not None:
            classes[name]["hangup_s"] = hangup
    return {"horizon_s": horizon, "classes": classes}


class Runner(p1_bench.Runner):
    def __init__(self, args):
        super().__init__(args)
        self.c_chat, self.c_batch = args.c_chat, args.c_batch
        self.build = None
        self.skip = {c for c in args.skip.split(",") if c}

    # ---- servers -------------------------------------------------------------
    def tree_env(self, build):
        tree = self.a.base_tree if build == "base" else self.a.patched_tree
        return {"PYTHONPATH": str(Path(tree).expanduser() / "python")} if tree else {}

    def start(self, tag, extra=(), env=None, build="patched", model=None):
        """Start a server from the given build; checks which source tree it imports."""
        env = {**self.tree_env(build), **(env or {})}
        self.build = build
        if not self.a.dry_run:
            probe = subprocess.run(
                [sys.executable, "-c",
                 "import sglang, dataclasses;"
                 "from sglang.srt.managers.io_struct import GenerateReqInput as G;"
                 "print(sglang.__file__, 'waiting_timeout' in {f.name for f in dataclasses.fields(G)})"],
                capture_output=True, text=True, env={**os.environ, **env},
            )  # fmt: skip
            line = (probe.stdout.strip().splitlines() or ["?"])[-1]
            self.say(f"build[{build}] imports {line}")
            if not line.endswith("True" if build == "patched" else "False"):
                self.say(f"build[{build}] is NOT the expected tree: {probe.stderr[-300:]}")
                return False
        old_model = self.a.model
        if model:
            self.a.model = model
        try:
            if self.a.dry_run:
                return self.start_stub(tag, extra, env)
            return self.start_server(tag, extra=extra, env=env)
        finally:
            self.a.model = old_model

    def start_stub(self, tag, extra, env):
        self.stop_server()
        one_slot = "--max-running-requests" in extra and extra[extra.index("--max-running-requests") + 1] == "1"
        cmd = [sys.executable, str(HERE / "stub_server.py"), "--port", str(self.a.port),
               "--tpot-ms", "2" if one_slot else "4", "--slots", "1" if one_slot else "8"]  # fmt: skip
        if "--enable-priority-scheduling" in extra:
            cmd.append("--priority")
        if (env or {}).get("SGLANG_REQ_WAITING_TIMEOUT"):
            cmd += ["--global-timeout", env["SGLANG_REQ_WAITING_TIMEOUT"]]
        self.say(f"stub[{tag}] {' '.join(cmd[2:])}")
        self.server_tag = tag
        self.server = subprocess.Popen(cmd)
        time.sleep(1.0)
        return True

    def stop_server(self):
        if self.a.dry_run:
            if self.server is not None:
                self.server.terminate()
                self.server.wait()
                self.server = None
            return
        super().stop_server()

    def flush(self):
        try:
            req = urllib.request.Request(f"http://127.0.0.1:{self.a.port}/flush_cache", method="POST")
            urllib.request.urlopen(req, timeout=30).read()
        except Exception as e:
            self.say(f"flush_cache: {type(e).__name__} (ignored)")
        time.sleep(2)

    def left(self):
        return self.a.deadline_epoch - time.time() if self.a.deadline_epoch else 1e9

    def room(self, need_s, what):
        if (self.out / "STOP").exists():
            self.say("STOP file found")
            return False
        if self.left() < need_s:
            self.say(f"skip {what}: needs {need_s:.0f} s, {self.left():.0f} s left")
            return False
        return True

    # ---- one client run --------------------------------------------------------
    def client(self, cell, task, cfg, *, seed=1, arm="", params=None, closed=None):
        cfg_path = self.out / "cells" / f"{cell}.config.json"
        cfg_path.write_text(json.dumps(cfg))
        out = self.out / "cells" / f"{cell}.jsonl"
        cmd = [sys.executable, str(HERE / "slo_client.py"), "--base-url", f"http://127.0.0.1:{self.a.port}",
               "--config", str(cfg_path), "--seed", str(seed), "--tag", cell, "--out", str(out),
               "--drain-timeout", str(cfg["horizon_s"] + 300)]  # fmt: skip
        if closed:
            cmd += ["--closed-loop", str(closed[0]), "--cls", closed[1], "--num", str(closed[2])]
        with open(self.out / "cells" / f"{cell}.log", "w") as log:
            try:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=cfg["horizon_s"] + 600).returncode  # fmt: skip
            except subprocess.TimeoutExpired:
                rc = -9
        summary_path = Path(str(out) + ".summary.json")
        if rc != 0 or not summary_path.exists():
            self.say(f"cell {cell}: client FAILED rc={rc}")
            self.record({"cell": cell, "task": task, "failed": True})
            return None
        s = json.loads(summary_path.read_text())
        row = {"cell": cell, "task": task, "arm": arm, "seed": seed, "server": self.server_tag,
               "build": self.build, "params": params or {}, **s}  # fmt: skip
        self.record(row)
        al = s["all"]
        per = "  ".join(
            f"{n} {c['attainment'] * 100:.1f}% (refused {c['refused']}, ttft {c['mean_ttft_ms'] and round(c['mean_ttft_ms'])} ms)"
            for n, c in s["classes"].items()
        )
        self.say(f"cell {cell}: all {al['attainment'] * 100:.1f}% of {al['sent']} | {per} | "
                 f"{al['out_tokens_per_s']:.0f} tok/s, {s['duration_s']:.0f} s")  # fmt: skip
        return row

    # ---- phases ----------------------------------------------------------------
    def ladder(self, only=None, prefix="", flags=()):
        """D2 (dummy weights), D3 (real model), the global combination, the unpatched control."""
        runs = [
            ("ladder_dummy", "main", "patched", self.a.small_model, ("--load-format", "dummy"), {}),
            ("ladder_real", "main", "patched", None, (), {}),
            ("ladder_global2", "global2", "patched", None, (), {"SGLANG_REQ_WAITING_TIMEOUT": "2"}),
            ("ladder_control", "control", "base", None, (), {}),
        ]
        for tag, mode, build, model, extra, env in runs:
            if only and tag not in only:
                continue
            tag = prefix + tag
            if not self.room(240, tag):
                return
            if not self.start(tag, extra=(*extra, *flags, "--max-running-requests", "1"), env=env, build=build, model=model):
                self.record({"cell": tag, "task": "ladder", "failed": True, "reason": "server did not start"})
                continue
            out = self.out / f"{tag}.json"
            cmd = [sys.executable, str(HERE / "prb_ladder.py"), "--base-url", f"http://127.0.0.1:{self.a.port}",
                   "--model", model or self.a.model, "--mode", mode, "--out", str(out)]  # fmt: skip
            if self.a.dry_run:
                cmd += ["--hold-s", "12"]
            with open(self.out / f"{tag}.log", "w") as log:
                try:  # a ladder that does not end is a server that hangs: a failure, not a wait
                    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=420).returncode
                except subprocess.TimeoutExpired:
                    rc = -9
                    log.write("\nTIMEOUT: the ladder did not finish in 420 s\n")
            rows = json.loads(out.read_text())["rows"] if out.exists() else []
            failed = [r["check"] for r in rows if not r["ok"]]
            self.say(f"{tag}: {len(rows) - len(failed)}/{len(rows)} checks passed rc={rc}" + (f" FAILED: {failed}" if failed else ""))
            self.record({"cell": tag, "task": "ladder", "build": build, "rc": rc, "rows": rows})

    def ladder2(self):
        """The real-model ladder and the global combination again, for a later build of the patch."""
        self.ladder(only=("ladder_real", "ladder_global2"), prefix="v2_")

    def calib(self):
        """Closed-loop capacity of each class alone at the batch cap."""
        if not self.room(240, "calib") or not self.start("cap128", extra=CAP):
            return
        chat = self.client("calib_chat", "calib", config(1, chat=([], {}, None)), closed=(128, "chat", 1280))
        self.flush()
        batch = self.client("calib_batch", "calib", config(1, batch=([], {}, None)), closed=(128, "batch", 512))
        if chat and batch:
            m_chat, m_batch = chat["all"]["req_per_s"], batch["all"]["req_per_s"]
            self.say(f"capacity at cap 128: c_chat = {m_chat:.2f} req/s, c_batch = {m_batch:.2f} req/s")
            (self.out / "capacity.json").write_text(json.dumps({"c_chat": m_chat, "c_batch": m_batch}))
            if self.a.c_chat and self.a.c_batch:  # an earlier session's values fix the load levels
                self.say(f"load levels keep the given capacities {self.a.c_chat:.2f} / {self.a.c_batch:.2f} req/s")
            else:
                self.c_chat, self.c_batch = m_chat, m_batch

    def check(self):
        """The client against bench_serving --goodput on the same server and the same request shape."""
        if not self.room(200, "check") or not (self.server_tag == "cap128" or self.start("cap128", extra=CAP)):
            return
        rate = round(0.5 * self.c_chat, 3)
        self.flush()
        self.client("check_client", "check", config(40, chat=([[0, 40, rate]], {}, None), period=40), params={"rate": rate})
        self.flush()
        cell = "check_bench"
        cmd = [sys.executable, "-m", "sglang.bench_serving", "--backend", "sglang", "--host", "127.0.0.1",
               "--port", str(self.a.port), "--model", self.a.model, "--dataset-name", "random-ids",
               "--random-input-len", "256", "--random-output-len", "128", "--random-range-ratio", "1.0",
               "--num-prompts", str(round(rate * 40)), "--request-rate", str(rate), "--seed", "1",
               "--disable-tqdm", "--tag", cell, "--output-file", str(self.out / "cells" / f"{cell}.jsonl"),
               "--goodput", "ttft:2000", "tpot:100"]  # fmt: skip
        if self.a.dry_run:
            return
        with open(self.out / "cells" / f"{cell}.log", "w") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        path = self.out / "cells" / f"{cell}.jsonl"
        if rc == 0 and path.exists():
            r = json.loads(path.read_text().strip().splitlines()[-1])
            keep = ("request_throughput", "output_throughput", "request_goodput", "slo_attainment",
                    "mean_ttft_ms", "p99_ttft_ms", "mean_tpot_ms", "mean_e2e_latency_ms", "completed")  # fmt: skip
            self.record({"cell": cell, "task": "check", "params": {"rate": rate}, **{k: r.get(k) for k in keep}})
            self.say(f"cell {cell}: attain {r.get('slo_attainment', 0) * 100:.1f}% ttft {r.get('mean_ttft_ms', 0):.0f} ms "
                     f"tpot {r.get('mean_tpot_ms', 0):.1f} ms {r.get('output_throughput', 0):.0f} tok/s")  # fmt: skip
        else:
            self.say(f"cell {cell}: bench FAILED rc={rc}")

    # U1: chat bursts above capacity, steady batch, priority to chat.
    def u1_cfg(self, chat_body, batch_body, hangup=None):
        cc, cb = self.c_chat, self.c_batch
        return config(
            self.a.horizon,
            chat=([[0, 40, round(0.3 * cc, 3)], [40, 60, round(1.5 * cc, 3)]], {"priority": 1, **chat_body}, hangup),
            batch=([[0, 60, round(0.5 * cb, 3)]], {"priority": 0, **batch_body}, None),
        )

    # U2: steady chat and a batch burst, first come first served.
    def u2_cfg(self, chat_body, batch_body):
        cc, cb = self.c_chat, self.c_batch
        return config(
            self.a.horizon,
            chat=([[0, 60, round(0.4 * cc, 3)]], chat_body, None),
            batch=([[20, 35, round(2.0 * cb, 3)]], batch_body, None),
        )

    def arms(self, task, cfg_fn, flags, plan):
        """plan: list of (server tag, global timeout or None, [(arm, chat body, batch body, hangup, seeds)])."""
        need = self.a.horizon + 150
        for tag, global_s, runs in plan:
            runs = [(arm, cb, bb, hu, [s for s in seeds if f"{task}_{arm}_s{s}" not in self.skip])
                    for arm, cb, bb, hu, seeds in runs]  # fmt: skip
            if not any(seeds for *_, seeds in runs):
                continue
            if not self.room(need + 120, f"{task} server {tag}"):
                return
            env = {} if global_s is None else {"SGLANG_REQ_WAITING_TIMEOUT": str(global_s)}
            if not self.start(f"{task}_{tag}", extra=(*CAP, *flags), env=env):
                continue
            for arm, chat_body, batch_body, hangup, seeds in runs:
                for seed in seeds:
                    if not self.room(need, f"{task} {arm} seed {seed}"):
                        return
                    self.flush()
                    cfg = cfg_fn(chat_body, batch_body, hangup) if hangup is not None else cfg_fn(chat_body, batch_body)
                    self.client(f"{task}_{arm}_s{seed}", task, cfg, seed=seed, arm=arm,
                                params={"global_timeout_s": global_s, "c_chat": self.c_chat, "c_batch": self.c_batch})  # fmt: skip

    def pilot(self):
        pr = ({"waiting_timeout": CHAT_BOUND}, {"waiting_timeout": BATCH_BOUND})
        self.arms("pilot", self.u1_cfg, PRIORITY, [
            ("off", None, [("none", {}, {}, None, (1,)), ("per_request", *pr, None, (1,))]),
            ("g1.5", CHAT_BOUND, [("global_1.5", {}, {}, None, (1,))]),
        ])  # fmt: skip

    def u1(self):
        """The pilot's three runs are seed 1 of none, per_request and global_1.5; --skip names them.

        The two arms of the acceptance test go first. The no-timeout arm gets a server that has
        never seen the field and runs first on it: the gate is sticky (PRB_PLAN.md §13 S7).
        """
        s3, s2 = (1, 2, 3), (1, 2)
        pr = ({"waiting_timeout": CHAT_BOUND}, {"waiting_timeout": BATCH_BOUND})
        plan = [
            ("off", None, [("per_request", *pr, None, s3)]),
            ("g1.5", 1.5, [("global_1.5", {}, {}, None, s3)]),
            ("off2", None, [("none", {}, {}, None, s2),
                            ("per_request_chat_only", {"waiting_timeout": CHAT_BOUND}, {}, None, s2)]),
            ("g30", 30, [("global_30", {}, {}, None, s2)]),
            ("g5", 5, [("global_5", {}, {}, None, s2)]),
            ("g20", 20, [("global_20", {}, {}, None, s2)]),
        ]  # fmt: skip
        if self.a.hangup:
            plan.append(("hangup", None, [("hang_up", {}, {}, CHAT_BOUND, s2)]))
        self.arms("u1", self.u1_cfg, PRIORITY, plan)

    def u2(self):
        pr = ({"waiting_timeout": CHAT_BOUND}, {"waiting_timeout": BATCH_BOUND})
        self.arms("u2", self.u2_cfg, (), [
            ("off", None, [("none", {}, {}, None, (1,)), ("per_request", *pr, None, (1, 2))]),
            ("g1.5", 1.5, [("global_1.5", {}, {}, None, (1, 2))]),
            ("g30", 30, [("global_30", {}, {}, None, (1,))]),
        ])  # fmt: skip

    # U1b: the same use case with heavier chat bursts (2.0 instead of 1.5 of chat capacity).
    def u1b_cfg(self, chat_body, batch_body):
        cc, cb = self.c_chat, self.c_batch
        return config(
            self.a.horizon,
            chat=([[0, 40, round(0.3 * cc, 3)], [40, 60, round(2.0 * cc, 3)]], {"priority": 1, **chat_body}, None),
            batch=([[0, 60, round(0.5 * cb, 3)]], {"priority": 0, **batch_body}, None),
        )

    def u1b(self):
        pr = ({"waiting_timeout": CHAT_BOUND}, {"waiting_timeout": BATCH_BOUND})
        self.arms("u1b", self.u1b_cfg, PRIORITY, [
            ("off", None, [("per_request", *pr, None, (1, 2))]),
            ("g1.5", 1.5, [("global_1.5", {}, {}, None, (1, 2))]),
            ("off2", None, [("none", {}, {}, None, (1,))]),
        ])  # fmt: skip

    def tp2(self):
        """Two GPUs, tensor parallel: the ladder, the global combination, then many refusals under load."""
        tp = ("--tp-size", "2")
        self.ladder(only=("ladder_real", "ladder_global2"), prefix="tp2_", flags=tp)
        rate = round(1.3 * self.c_chat, 3)
        for tag, global_s, body in (("field", None, {"waiting_timeout": CHAT_BOUND}), ("global", CHAT_BOUND, {})):
            if not self.room(330, f"tp2 load {tag}"):
                return
            env = {} if global_s is None else {"SGLANG_REQ_WAITING_TIMEOUT": str(global_s)}
            if not self.start(f"tp2_load_{tag}", extra=(*CAP, *tp), env=env):
                continue
            self.client(f"tp2_load_{tag}", "tp2", config(60, chat=([[0, 60, rate]], body, None)), arm=tag,
                        params={"rate": rate, "tp_size": 2})  # fmt: skip

    def tp2b(self):
        """Two GPUs with a small batch cap, so that the same load overflows and many requests are refused."""
        flags = ("--tp-size", "2", "--max-running-requests", "32")
        rate = round(1.3 * self.c_chat, 3)
        for tag, global_s, body in (("field", None, {"waiting_timeout": CHAT_BOUND}), ("global", CHAT_BOUND, {})):
            if not self.room(240, f"tp2b load {tag}"):
                return
            env = {} if global_s is None else {"SGLANG_REQ_WAITING_TIMEOUT": str(global_s)}
            if not self.start(f"tp2b_load_{tag}", extra=flags, env=env):
                continue
            self.client(f"tp2b_load_{tag}", "tp2", config(60, chat=([[0, 60, rate]], body, None)), arm=f"{tag}, cap 32",
                        params={"rate": rate, "tp_size": 2, "max_running_requests": 32})  # fmt: skip

    def u3(self):
        """No-op control at 0.8 of capacity, each arm on a fresh server process."""
        cc, cb = self.c_chat, self.c_batch
        seg = lambda c: [[0, 90, round(0.4 * c, 3)]]  # noqa: E731
        loose = {"waiting_timeout": 3600}
        runs = [("base_a", "base", {}), ("patched_absent", "patched", {}),
                ("patched_loose", "patched", loose), ("base_b", "base", {})]  # fmt: skip
        for arm, build, body in runs:
            if not self.room(330, f"u3 {arm}"):
                return
            if not self.start(f"u3_{arm}", extra=CAP, build=build):
                continue
            cfg = config(90, chat=(seg(cc), body, None), batch=(seg(cb), body, None), period=90)
            self.client(f"u3_{arm}", "u3", cfg, arm=arm)

    def u4(self):
        """Same bound as a field and as the global knob: chat alone at 1.3 of its capacity."""
        rate = round(1.3 * self.c_chat, 3)
        for tag, global_s, body in (("field", None, {"waiting_timeout": CHAT_BOUND}), ("global", CHAT_BOUND, {})):
            if not self.room(330, f"u4 {tag}"):
                return
            env = {} if global_s is None else {"SGLANG_REQ_WAITING_TIMEOUT": str(global_s)}
            if not self.start(f"u4_{tag}", extra=CAP, env=env):
                continue
            for seed in (1, 2):
                self.flush()
                self.client(f"u4_{tag}_s{seed}", "u4", config(60, chat=([[0, 60, rate]], body, None)), seed=seed, arm=tag,
                            params={"rate": rate})  # fmt: skip

    def t1rep(self):
        """PR-A: two more seeds at three points of P1's load sweep, stock server flags."""
        if self.a.dry_run or not self.room(300, "t1rep") or not self.start("default"):
            return
        for mult in (1.0, 1.6, 2.0):
            rate = round(P1_C0 * mult, 3)
            for seed in (2, 3):
                if not self.room(200, f"t1rep x{mult} seed {seed}"):
                    return
                self.cell(f"t1_rate_x{mult:g}_seed{seed}", "T1-repeat", rate=rate, seed=seed,
                          num_prompts=round(rate * p1_bench.ARRIVAL_SPAN_S),
                          params={"mult": mult, "rate": rate, "seed": seed})  # fmt: skip

    def t4rep(self):
        """PR-A / PR-B motivation: P1's T4 with the fixed benchmark, three seeds per setting."""
        if self.a.dry_run:
            return
        rate = round(P1_C0 * 1.5, 3)
        for timeout in (None, 2, 10):
            tag = f"wt{'off' if timeout is None else timeout}"
            if not self.room(420, f"t4rep {tag}"):
                return
            env = {} if timeout is None else {"SGLANG_REQ_WAITING_TIMEOUT": str(timeout)}
            if not self.start(f"t4_{tag}", extra=CAP, env=env):
                continue
            for seed in (1, 2, 3):
                if not self.room(200, f"t4rep {tag} seed {seed}"):
                    return
                self.cell(f"t4_{tag}_seed{seed}", "T4-repeat", rate=rate, seed=seed,
                          num_prompts=round(rate * p1_bench.ARRIVAL_SPAN_S),
                          params={"waiting_timeout_s": timeout, "rate": rate, "seed": seed,
                                  "max_running_requests": 128})  # fmt: skip

    def run(self):
        need_capacity = {"check", "pilot", "u1", "u1b", "u2", "u3", "u4", "tp2", "tp2b"}
        try:
            for phase in self.a.phases.split(","):
                if phase in need_capacity and not (self.c_chat and self.c_batch):
                    self.say(f"phase {phase}: no capacity estimate, skipped")
                    continue
                self.say(f"===== phase {phase} ({self.left() / 60:.0f} min left) =====")
                getattr(self, phase)()
        finally:
            self.stop_server()
            (self.out / ".finished").touch()
            self.say("runner finished")
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--small-model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--port", type=int, default=30000)
    p.add_argument("--deadline-epoch", type=float, default=0.0)
    p.add_argument("--phases", required=True)
    p.add_argument("--base-tree", default="~/sgl/sglang-base")
    p.add_argument("--patched-tree", default="", help="source tree of the patched build, if not the installed one")
    p.add_argument("--c-chat", type=float, default=0.0)
    p.add_argument("--c-batch", type=float, default=0.0)
    p.add_argument("--horizon", type=int, default=180)
    p.add_argument("--hangup", action="store_true")
    p.add_argument("--skip", default="", help="comma list of cells already run, e.g. u1_none_s1")
    p.add_argument("--dry-run", action="store_true")
    sys.exit(Runner(p.parse_args()).run())


if __name__ == "__main__":
    main()
