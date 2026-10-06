#!/usr/bin/env python3
"""Queue model for choosing PR-B's benchmark scenarios before any GPU time is spent.

  python3 prb_sim.py            # all scenarios, all arms
  python3 prb_sim.py --seeds 20

A server with a fixed number of slots (--max-running-requests) and a waiting queue. A request holds
one slot for its service time. A queued request is dropped when it has waited longer than its bound:
its own `waiting_timeout`, capped by the global one, as in PRB_PLAN.md §2.6. Two classes:

  chat   TTFT objective: wait + prefill <= 2 s
  batch  deadline:       wait + service <= its deadline

Deliberately crude: service time does not depend on batch occupancy, prefill is a constant, and the
scan is continuous instead of once per scheduler step. It answers one question only: which arm can
win in a scenario, and roughly by how much.
"""

import argparse
import heapq
import random
import statistics

SLOTS = 128
CHAT = dict(service=2.0, prefill=0.08, ttft_slo=2.0)  # 128 output tokens at ~15 ms
BATCH = dict(service=4.0, prefill=0.25, deadline=40.0)  # 256 output tokens, 1k prompt


def arrivals(rng, pattern, horizon):
    """pattern: list of (t_start, t_end, rate per second) repeated every `period`."""
    out = []
    for cls, segments, period in pattern:
        t0 = 0.0
        while t0 < horizon:
            for start, end, rate in segments:
                t = t0 + start
                while rate > 0:
                    t += rng.expovariate(rate)
                    if t >= t0 + end or t >= horizon:
                        break
                    out.append((t, cls))
            t0 += period
    out.sort()
    return out


def simulate(reqs, bound, global_s=None, priority=False):
    """bound: {class: seconds or None}. Returns per-class counts and wasted slot-seconds."""
    spec = {"chat": CHAT, "batch": BATCH}
    limit = {}
    for cls in spec:
        own = bound.get(cls)
        if own is None or (global_s is not None and global_s < own):
            own = global_s
        limit[cls] = own
    free = SLOTS
    done = []  # heap of completion times
    queue = {"chat": [], "batch": []}  # FIFO per class; merged by arrival time unless priority
    head = {"chat": 0, "batch": 0}
    stats = {c: dict(sent=0, good=0, dropped=0, late=0) for c in spec}
    wasted = 0.0
    served = 0.0

    def pop(now):
        """Next request to start at `now`, dropping expired ones on the way."""
        while True:
            cands = []
            for cls in spec:
                q, h = queue[cls], head[cls]
                while h < len(q) and limit[cls] is not None and now - q[h] > limit[cls]:
                    stats[cls]["dropped"] += 1
                    h += 1
                head[cls] = h
                if h < len(q):
                    cands.append((q[h], cls))
            if not cands:
                return None
            if priority:
                cls = "chat" if any(c == "chat" for _, c in cands) else "batch"
            else:
                cls = min(cands)[1]
            t_arr = queue[cls][head[cls]]
            head[cls] += 1
            return t_arr, cls

    def start(now):
        nonlocal free, wasted, served
        while free > 0:
            nxt = pop(now)
            if nxt is None:
                return
            t_arr, cls = nxt
            s = spec[cls]
            wait = now - t_arr
            ok = (
                wait + s["prefill"] <= s["ttft_slo"]
                if cls == "chat"
                else wait + s["service"] <= s["deadline"]
            )
            stats[cls]["good" if ok else "late"] += 1
            served += s["service"]
            if not ok:
                wasted += s["service"]
            free -= 1
            heapq.heappush(done, now + s["service"])

    for t, cls in reqs:
        while done and done[0] <= t:
            now = heapq.heappop(done)
            free += 1
            start(now)
        stats[cls]["sent"] += 1
        queue[cls].append(t)
        start(t)
    while done:
        now = heapq.heappop(done)
        free += 1
        start(now)
    for cls in spec:  # whatever is still queued at the end never ran
        stats[cls]["dropped"] += len(queue[cls]) - head[cls]
    return stats, wasted, served


def capacity(mix_chat):
    """Requests per second the slots sustain for a given share of chat requests."""
    mean_service = mix_chat * CHAT["service"] + (1 - mix_chat) * BATCH["service"]
    return SLOTS / mean_service


SCENARIOS = {}


def scenario(name):
    def deco(fn):
        SCENARIOS[name] = fn
        return fn

    return deco


@scenario("A  both classes surge together, FCFS (U1 as first planned)")
def _a():
    c = capacity(0.5)
    seg = [(0, 45, 0.3 * c), (45, 60, 1.0 * c)]
    return [("chat", seg, 60), ("batch", seg, 60)], False


@scenario("B  steady chat + batch burst, FCFS (U2 of the 10-06 amendment)")
def _b():
    c_chat = SLOTS / CHAT["service"]
    c_batch = SLOTS / BATCH["service"]
    return [
        ("chat", [(0, 60, 0.4 * c_chat)], 60),
        ("batch", [(20, 35, 2.0 * c_batch)], 60),
    ], False


@scenario("C  chat bursts above capacity + steady batch, priority to chat (U1 of the 10-06 amendment)")
def _c():
    c_chat = SLOTS / CHAT["service"]
    c_batch = SLOTS / BATCH["service"]
    return [
        ("chat", [(0, 40, 0.3 * c_chat), (40, 60, 1.5 * c_chat)], 60),
        ("batch", [(0, 60, 0.5 * c_batch)], 60),
    ], True


@scenario("D  steady overload 1.3x, both classes, FCFS (U2 as first planned)")
def _d():
    c = capacity(0.5)
    seg = [(0, 60, 0.65 * c)]
    return [("chat", seg, 60), ("batch", seg, 60)], False


ARMS = [
    ("none", {}, None),
    ("global 1.5", {}, 1.5),
    ("global 5", {}, 5.0),
    ("global 20", {}, 20.0),
    ("global 30", {}, 30.0),
    ("per-request chat 1.5, batch 30", {"chat": 1.5, "batch": 30.0}, None),
    ("per-request chat 1.5 only", {"chat": 1.5}, None),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--horizon", type=float, default=180.0)
    a = ap.parse_args()
    for name, build in SCENARIOS.items():
        pattern, priority = build()
        print(f"\n## {name}\n")
        print("| arm | chat % | batch % | all % (sd) | dropped % | wasted work % |")
        print("|---|---|---|---|---|---|")
        for arm, bound, global_s in ARMS:
            rows = []
            for seed in range(a.seeds):
                reqs = arrivals(random.Random(seed), pattern, a.horizon)
                st, wasted, served = simulate(reqs, bound, global_s, priority)
                sent = sum(s["sent"] for s in st.values())
                rows.append((
                    100 * st["chat"]["good"] / st["chat"]["sent"],
                    100 * st["batch"]["good"] / st["batch"]["sent"],
                    100 * sum(s["good"] for s in st.values()) / sent,
                    100 * sum(s["dropped"] for s in st.values()) / sent,
                    100 * wasted / max(served, 1e-9),
                ))  # fmt: skip
            m = [statistics.mean(r[i] for r in rows) for i in range(5)]
            sd = statistics.stdev(r[2] for r in rows) if len(rows) > 1 else 0.0
            print(f"| {arm} | {m[0]:.0f} | {m[1]:.0f} | {m[2]:.1f} ({sd:.1f}) | {m[3]:.0f} | {m[4]:.0f} |")


if __name__ == "__main__":
    main()
