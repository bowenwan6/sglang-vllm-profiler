#!/usr/bin/env python3
"""R6 idle-GPU monitor + R6.1b auto-launcher.

Purpose
-------
We are authorized to use exactly ONE GPU at a time and must never
signal, reset, or interfere with any process we did not launch. This
monitor waits for one GPU to remain continuously idle for at least
600 seconds (10 minutes), then hands its ID off to the R6.1 runner
via `R6_GPU_ID=<id>`. It never selects more than one GPU and never
resets a GPU.

Definitions
-----------
Idle GPU (all three must hold at every 30 s poll):
    * no compute application PIDs from `nvidia-smi --query-compute-apps=pid`
    * memory used <= 500 MiB
    * GPU utilization <= 5 %

Continuous idle: the same GPU satisfies all three at every poll
between the timer start and now. Any failed poll resets that GPU's
timer to zero. A GPU showing residual memory with no visible process
is considered busy (never reset).

Selection: sorted by GPU index ascending. The first GPU whose
continuous-idle streak reaches 600 s wins. Ties broken by lowest ID.

Immediate pre-launch recheck: after a GPU qualifies, the monitor
runs one more live check. If it is no longer idle, the timer is
cleared and monitoring resumes; the qualification is not treated as
a reservation.

Concurrency
-----------
A non-blocking `fcntl.flock` on `raw/monitor.lock` prevents two
copies of this monitor from starting at once.

Records
-------
Every poll appends a JSON snapshot to `raw/monitor.jsonl`. Free-text
status goes to `raw/monitor.log`. The full qualification interval +
selected GPU + pre-launch check is written to
`raw/monitor_selection.json` immediately before the runner is exec'd.
All of these are `.gitignore`d and not committed.

Prohibited (never issued by this monitor)
-----------------------------------------
* `pkill`, `killall`, `fuser -k`, kill-by-name, kill-by-port
* `nvidia-smi --gpu-reset`
* Any signal to any process the monitor did not launch. The runner
  it invokes is authorized to signal only PGIDs it launched itself.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

POLL_INTERVAL_S = 30
IDLE_HOLD_S = 600
MEM_THRESHOLD_MIB = 500
UTIL_THRESHOLD_PCT = 5
STATUS_LOG_INTERVAL_S = 300

ROOT = Path("/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/"
            "image_text_benchmarks/debug_pcg_capture_stream/root_cause")
RAW = ROOT / "results/R6_fix_value_validation/R6.1_correctness/raw"
LOCK = RAW / "monitor.lock"
JSONL = RAW / "monitor.jsonl"
LOG = RAW / "monitor.log"
SELECTION = RAW / "monitor_selection.json"
RUNNER = ROOT / "scripts/run_R6_1_correctness.sh"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str, *, level: str = "INFO") -> None:
    line = f"[{utc_now()}] [{level}] {msg}"
    print(line, flush=True)
    with LOG.open("a") as f:
        f.write(line + "\n")


def nvidia_smi(query: str, *extra: str) -> list[str]:
    r = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}",
         "--format=csv,noheader,nounits", *extra],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {r.stderr.strip()}")
    return [line.strip() for line in r.stdout.strip().splitlines() if line.strip()]


def nvidia_smi_compute_pids(idx: int) -> list[str]:
    r = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits", "-i", str(idx)],
        capture_output=True, text=True, timeout=15,
    )
    if r.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute-apps failed: {r.stderr.strip()}")
    return [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]


def enumerate_gpus() -> list[int]:
    return sorted(int(x) for x in nvidia_smi("index"))


def check_gpu(idx: int) -> dict:
    try:
        mem = int(nvidia_smi("memory.used", "-i", str(idx))[0])
    except Exception as e:
        return {"ok": False, "error": f"mem_query_failed: {e!r}", "idle": False}
    try:
        util = int(nvidia_smi("utilization.gpu", "-i", str(idx))[0])
    except Exception as e:
        return {"ok": False, "error": f"util_query_failed: {e!r}", "idle": False}
    try:
        pids = nvidia_smi_compute_pids(idx)
    except Exception as e:
        return {"ok": False, "error": f"pid_query_failed: {e!r}", "idle": False}
    idle = (len(pids) == 0
            and mem <= MEM_THRESHOLD_MIB
            and util <= UTIL_THRESHOLD_PCT)
    return {"ok": True, "mem_mib": mem, "util_pct": util,
            "compute_pids": pids, "idle": idle}


def acquire_lock() -> "IO[str]":
    RAW.mkdir(parents=True, exist_ok=True)
    fd = open(LOCK, "w")
    try:
        fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("ERROR: another R6 monitor already holds the lock at "
              f"{LOCK}; exiting.", file=sys.stderr)
        sys.exit(75)
    fd.write(f"pid={os.getpid()} started={utc_now()}\n")
    fd.flush()
    return fd


def launch_runner(gpu_id: int, idle_start_epoch: float,
                  qualified_epoch: float, prelaunch_state: dict) -> int:
    payload = {
        "selected_gpu_id": gpu_id,
        "idle_start_utc": datetime.fromtimestamp(
            idle_start_epoch, tz=timezone.utc).isoformat(timespec="seconds"),
        "qualified_utc": datetime.fromtimestamp(
            qualified_epoch, tz=timezone.utc).isoformat(timespec="seconds"),
        "idle_hold_s": IDLE_HOLD_S,
        "poll_interval_s": POLL_INTERVAL_S,
        "mem_threshold_mib": MEM_THRESHOLD_MIB,
        "util_threshold_pct": UTIL_THRESHOLD_PCT,
        "prelaunch_state": prelaunch_state,
        "prelaunch_utc": utc_now(),
        "monitor_pid": os.getpid(),
    }
    SELECTION.write_text(json.dumps(payload, indent=2, sort_keys=True))
    log(f"launching runner: R6_GPU_ID={gpu_id} bash {RUNNER}")
    env = {**os.environ, "R6_GPU_ID": str(gpu_id)}
    r = subprocess.run(["bash", str(RUNNER)], env=env)
    log(f"runner exit code: {r.returncode}")
    return r.returncode


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    _ = acquire_lock()  # intentional: keep fd for process lifetime
    log(f"monitor start pid={os.getpid()} poll={POLL_INTERVAL_S}s "
        f"hold={IDLE_HOLD_S}s mem<={MEM_THRESHOLD_MIB}MiB "
        f"util<={UTIL_THRESHOLD_PCT}%")
    try:
        gpus = enumerate_gpus()
    except Exception as e:
        log(f"nvidia-smi enumeration failed: {e}", level="ERROR")
        return 2
    log(f"visible GPUs: {gpus}")
    idle_since: dict[int, float | None] = {g: None for g in gpus}
    last_status = 0.0
    while True:
        now = time.time()
        poll = {"ts_utc": utc_now(), "ts_epoch": now, "gpus": {}}
        for g in gpus:
            st = check_gpu(g)
            poll["gpus"][str(g)] = st
            if not st.get("idle"):
                if idle_since[g] is not None:
                    log(f"GPU {g} BUSY (mem={st.get('mem_mib')} "
                        f"util={st.get('util_pct')} "
                        f"pids={st.get('compute_pids')}); "
                        f"resetting timer")
                idle_since[g] = None
            else:
                if idle_since[g] is None:
                    idle_since[g] = now
                    log(f"GPU {g} idle streak start "
                        f"(mem={st['mem_mib']} util={st['util_pct']})")
        # Deterministic selection: lowest GPU ID with idle streak >= hold.
        candidate = None
        for g in gpus:
            t = idle_since[g]
            if t is not None and (now - t) >= IDLE_HOLD_S:
                candidate = g
                break
        with JSONL.open("a") as f:
            f.write(json.dumps(poll) + "\n")
        if candidate is not None:
            log(f"GPU {candidate} qualified: continuously idle for "
                f"{int(now - idle_since[candidate])}s. Pre-launch recheck.")
            live = check_gpu(candidate)
            if not live.get("idle"):
                log(f"GPU {candidate} DROPPED IDLE at pre-launch recheck "
                    f"(state={live}); resetting timer and resuming monitor",
                    level="WARN")
                idle_since[candidate] = None
            else:
                log(f"GPU {candidate} pre-launch recheck OK; handing off "
                    f"to runner")
                return launch_runner(candidate, idle_since[candidate], now, live)
        # Concise status at least every STATUS_LOG_INTERVAL_S.
        if now - last_status >= STATUS_LOG_INTERVAL_S:
            streaks = {g: (int(now - t) if t else 0)
                       for g, t in idle_since.items()}
            log(f"status: continuous-idle seconds per GPU = {streaks}")
            last_status = now
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    sys.exit(main())
