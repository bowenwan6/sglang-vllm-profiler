#!/usr/bin/env python3
"""Cost of the scheduler's waiting-timeout scan with a long queue (PRB_PLAN.md §2.7), CPU only.

  CUDA_VISIBLE_DEVICES=99 python scan_cost.py                                  # the installed (patched) tree
  CUDA_VISIBLE_DEVICES=99 PYTHONPATH=~/sgl/sglang-base/python python scan_cost.py   # the unpatched tree

Calls the real Scheduler._poll_timeout_aborts on a scheduler built without a model, the way
test_scheduler_timeouts.py does, over real Req objects that have just entered the queue (so nothing
is ever aborted). Prints microseconds per call, best of 9 repeats.
"""

import json
import sys
import time
import timeit
from collections import deque
from types import SimpleNamespace
from unittest.mock import MagicMock

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.environ import envs  # noqa: E402
from sglang.srt.managers.schedule_batch import Req  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.sampling.sampling_params import SamplingParams  # noqa: E402

PATCHED = "waiting_timeout" in Req.__init__.__code__.co_varnames


def make_req(i, bound):
    kw = {"waiting_timeout": bound} if PATCHED and bound is not None else {}
    try:
        r = Req(f"r{i:07d}", "", [1, 2, 3], SamplingParams(), **kw)
    except Exception:  # the constructor wants more context than a script has
        r = SimpleNamespace(rid=f"r{i:07d}", waiting_timeout=bound, **{f"a{k}": k for k in range(150)})
        r.time_stats = SimpleNamespace(wait_queue_entry_time=0.0)
    r.time_stats.wait_queue_entry_time = time.perf_counter()
    return r


def scheduler(queue, gate):
    s = Scheduler.__new__(Scheduler)
    s.enable_continuous_input_polling = False
    s.result_queue = deque()
    s.waiting_queue = queue
    s.running_batch = SimpleNamespace(reqs=[], is_empty=lambda: True)
    s.last_batch = None
    s.ipc_channels = SimpleNamespace(send_to_tokenizer=MagicMock())
    if PATCHED:
        s._has_req_waiting_timeout = gate
    return s


def cost(n, global_s, bound, gate):
    s = scheduler([make_req(i, bound) for i in range(n)], gate)
    with envs.SGLANG_REQ_WAITING_TIMEOUT.override(global_s):
        assert s._poll_timeout_aborts() == []
        number = 2000 if n == 0 or (global_s <= 0 and not gate) else max(3, 20000 // max(n, 1))
        best = min(timeit.repeat(s._poll_timeout_aborts, number=number, repeat=9)) / number
    return best * 1e6


def main():
    kind = type(make_req(0, None)).__name__
    rows = []
    cases = [("global off, field never seen", -1, None, False), ("global 60 s, no request bound", 60, None, False)]
    if PATCHED:
        cases += [("global off, field seen, no queued request carries one", -1, None, True),
                  ("global off, every queued request carries 3600 s", -1, 3600.0, True),
                  ("global 60 s, every queued request carries 3600 s", 60, 3600.0, True)]  # fmt: skip
    for name, global_s, bound, gate in cases:
        row = {"case": name, **{str(n): round(cost(n, global_s, bound, gate), 2) for n in (0, 1000, 10000)}}
        rows.append(row)
        print(f"{name:58s} " + "  ".join(f"n={n}: {row[str(n)]:9.2f} us" for n in (0, 1000, 10000)), flush=True)
    json.dump({"patched": PATCHED, "queue_item": kind, "python": sys.version.split()[0], "rows": rows}, sys.stdout)
    print()


if __name__ == "__main__":
    main()
