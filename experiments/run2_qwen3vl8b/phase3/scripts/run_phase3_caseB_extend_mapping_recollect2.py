#!/usr/bin/env python3
"""
run2 Phase 4 support — Case B SGLang EXTEND mapping re-collect, attempt 2 (flush-aware).

Root cause of attempt-1 failure: Case B is a 2048-token long-prefill workload, so its
EXTEND trace is large and SGLang's async trace flush lags well past the fixed 8 s wait
in collect_extend(); kill_server() terminated the server before the .gz finished
writing, producing a truncated stream (gzip -t / python gzip both EOFError). Case A/C/D
traces are smaller and flushed in time, so they are valid.

Fix here: after the profiler returns, POLL the newest .gz until python-gzip can fully
read it (or a generous timeout), THEN kill the server. graph-off mapping only; GPU 1;
SGLang only; no source changes; no vLLM. Prior truncated dirs are quarantined
(CORRUPT_* / TRUNC_*) and left in place, not deleted.
"""
import glob, gzip, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import run_phase3_extend as E

CASE = "caseB_longprefill"
FLUSH_TIMEOUT = 240  # seconds to wait for the trace .gz to become fully readable


def gz_readable(path):
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def newest_gz(trace_dir):
    cands = [p for p in glob.glob(str(Path(trace_dir) / "*" / "*-EXTEND.trace.json.gz"))
             if "CORRUPT_" not in p and "TRUNC_" not in p]
    return max(cands, key=lambda p: Path(p).stat().st_mtime) if cands else None


def main():
    E.os.chdir(E.LAB)
    E.LOGS.mkdir(parents=True, exist_ok=True)
    E.log("=== Case B EXTEND mapping re-collect attempt 2 (flush-aware, GPU 1) ===")
    u = E.gpu_used()
    if not (0 <= u < 2000):
        E.log(f"STOP: GPU {E.GPU} not idle (used={u} MiB)"); sys.exit(1)

    cfg = E.CASES[CASE]
    trace_dir = E.TRACES / CASE / "sglang_extend_mapping"
    trace_dir.mkdir(parents=True, exist_ok=True)
    flags = list(cfg["sglang_flags"]) + ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
    logpath = E.LOGS / f"{CASE}_sglang_extend_mapping_recollect2_server.log"
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", E.SNAPSHOT,
           "--dtype", "bfloat16", "--port", str(E.SGLANG_PORT), "--tp", "1",
           "--attention-backend", "flashinfer"] + flags
    env = {**E.BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
           "SGLANG_KERNEL_API_LOGDEST": str(E.LOGS / f"{CASE}_extend_mapping_recollect2_kapi_%i.log"),
           "SGLANG_TORCH_PROFILER_DIR": str(trace_dir)}
    E.log(f"  launch SGLang graph-off: {' '.join(flags)}")
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not E.wait_server():
        E.log("  ERROR: server did not come up"); E.kill_server(proc); sys.exit(2)

    loader = E.PrefillLoad(E.read_prompts(CASE, 64), cfg["conc"])
    new_gz = None
    try:
        E.log(f"  start prefill load (max_new_tokens=1, c={cfg['conc']})")
        loader.start(); time.sleep(6)
        E.log(f"  run profiler --profile-by-stage --num-steps {E.NUM_STEPS}")
        pcmd = ["python3", "-m", "sglang.profiler", "--url", f"http://127.0.0.1:{E.SGLANG_PORT}",
                "--num-steps", str(E.NUM_STEPS), "--profile-by-stage", "--output-dir", str(trace_dir)]
        pr = subprocess.run(pcmd, env=E.BASE_ENV, capture_output=True, text=True, timeout=E.PROFILE_TIMEOUT)
        if pr.returncode != 0:
            E.log(f"  ERROR profiler rc={pr.returncode}: {(pr.stdout+pr.stderr)[-400:]}"); sys.exit(2)
        # keep the SERVER ALIVE and poll until the .gz is fully readable (flush-aware)
        deadline = time.time() + FLUSH_TIMEOUT
        while time.time() < deadline:
            time.sleep(5)
            g = newest_gz(trace_dir)
            if g and gz_readable(g):
                new_gz = g
                E.log(f"  trace flushed & readable after wait: {Path(g).name} ({Path(g).stat().st_size/1e6:.1f}MB)")
                break
            sz = Path(g).stat().st_size/1e6 if g else 0
            E.log(f"  waiting for flush... newest={Path(g).name if g else None} size={sz:.1f}MB readable=False")
    except subprocess.TimeoutExpired:
        E.log("  ERROR profiler timeout"); sys.exit(2)
    finally:
        loader.stop_all()
        E.log(f"  prefill load sent ~{loader.sent} requests")
        E.kill_server(proc)

    ok = bool(new_gz) and gz_readable(new_gz)
    meta_path = E.META / f"{CASE}_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {"case": CASE}
    meta["extend_mapping_recollect"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": 2, "fix": "flush-aware: poll gz readability before killing server",
        "gpu": E.GPU, "graph_off": True, "num_steps": E.NUM_STEPS,
        "ok": ok, "gz": new_gz, "size_bytes": Path(new_gz).stat().st_size if new_gz else 0,
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    E.log(f"  metadata updated. ok={ok} gz={new_gz}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
