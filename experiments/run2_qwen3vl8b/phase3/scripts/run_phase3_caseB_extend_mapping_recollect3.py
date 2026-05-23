#!/usr/bin/env python3
"""
run2 Phase 4 support — Case B SGLang EXTEND mapping re-collect, attempt 3 (no-radix + flush poll).

Diagnosis from attempts 1-2: Case B's repeated 64 prompts get prefix-cached
(server log: "#new-token: 1, #cached-token: ~2170"), so there are NO real 2048-token
prefill forward steps to capture — --profile-by-stage caught either truncated trivial
EXTEND files or a valid DECODE-stage trace. The .gz is also written at server shutdown,
not during profiling.

Fix:
  1. --disable-radix-cache so every 2048-token prompt runs a genuine full prefill
     (mapping trace's job is kernel->source mapping, not perf timing, so disabling the
     cache for the mapping capture is acceptable and is recorded here).
  2. graph-off mapping (--disable-cuda-graph --disable-piecewise-cuda-graph).
  3. After the profiler returns, kill the server (graceful), THEN poll the newest
     *-EXTEND*.gz until python-gzip can fully read it (flush completes at shutdown).

GPU 1 only. SGLang only. No vLLM. No SGLang source changes. Prior bad dirs quarantined.
"""
import glob, gzip, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import run_phase3_extend as E

CASE = "caseB_longprefill"
POST_KILL_FLUSH_TIMEOUT = 180


def gz_readable(path):
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1 << 20):
                pass
        return True
    except Exception:
        return False


def newest_stage_gz(trace_dir, stage):
    cands = [p for p in glob.glob(str(Path(trace_dir) / "*" / f"*-{stage}.trace.json.gz"))
             if not any(q in p for q in ("CORRUPT_", "TRUNC_", "DECODEONLY_"))]
    return max(cands, key=lambda p: Path(p).stat().st_mtime) if cands else None


def main():
    E.os.chdir(E.LAB); E.LOGS.mkdir(parents=True, exist_ok=True)
    E.log("=== Case B EXTEND mapping re-collect attempt 3 (--disable-radix-cache, GPU 1) ===")
    u = E.gpu_used()
    if not (0 <= u < 2000):
        E.log(f"STOP: GPU {E.GPU} not idle (used={u} MiB)"); sys.exit(1)

    cfg = E.CASES[CASE]
    trace_dir = E.TRACES / CASE / "sglang_extend_mapping"
    trace_dir.mkdir(parents=True, exist_ok=True)
    flags = list(cfg["sglang_flags"]) + ["--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                                         "--disable-radix-cache"]
    logpath = E.LOGS / f"{CASE}_sglang_extend_mapping_recollect3_server.log"
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", E.SNAPSHOT,
           "--dtype", "bfloat16", "--port", str(E.SGLANG_PORT), "--tp", "1",
           "--attention-backend", "flashinfer"] + flags
    env = {**E.BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
           "SGLANG_KERNEL_API_LOGDEST": str(E.LOGS / f"{CASE}_extend_mapping_recollect3_kapi_%i.log"),
           "SGLANG_TORCH_PROFILER_DIR": str(trace_dir)}
    E.log(f"  launch SGLang graph-off + no-radix: {' '.join(flags)}")
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not E.wait_server():
        E.log("  ERROR: server did not come up"); E.kill_server(proc); sys.exit(2)

    # unique prompts further guard against any residual caching
    loader = E.PrefillLoad(E.read_prompts(CASE, 64), cfg["conc"])
    try:
        E.log(f"  start prefill load (max_new_tokens=1, c={cfg['conc']}, radix-cache OFF)")
        loader.start(); time.sleep(6)
        E.log(f"  run profiler --profile-by-stage --num-steps {E.NUM_STEPS}")
        pcmd = ["python3", "-m", "sglang.profiler", "--url", f"http://127.0.0.1:{E.SGLANG_PORT}",
                "--num-steps", str(E.NUM_STEPS), "--profile-by-stage", "--output-dir", str(trace_dir)]
        pr = subprocess.run(pcmd, env=E.BASE_ENV, capture_output=True, text=True, timeout=E.PROFILE_TIMEOUT)
        if pr.returncode != 0:
            E.log(f"  ERROR profiler rc={pr.returncode}: {(pr.stdout+pr.stderr)[-400:]}"); sys.exit(2)
        time.sleep(5)
    except subprocess.TimeoutExpired:
        E.log("  ERROR profiler timeout"); sys.exit(2)
    finally:
        loader.stop_all()
        E.log(f"  prefill load sent ~{loader.sent} requests")
        E.kill_server(proc)  # graceful drain writes the .gz at shutdown

    # poll for a readable EXTEND gz now that the server has flushed at shutdown
    new_gz, deadline = None, time.time() + POST_KILL_FLUSH_TIMEOUT
    while time.time() < deadline:
        g = newest_stage_gz(trace_dir, "EXTEND")
        if g and gz_readable(g):
            new_gz = g
            E.log(f"  EXTEND flushed & readable: {Path(g).name} ({Path(g).stat().st_size/1e6:.1f}MB)")
            break
        E.log(f"  post-kill wait... extend={Path(g).name if g else None}")
        time.sleep(5)

    decode_g = newest_stage_gz(trace_dir, "DECODE")
    ok = bool(new_gz)
    meta_path = E.META / f"{CASE}_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {"case": CASE}
    meta["extend_mapping_recollect"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "attempt": 3,
        "fix": "--disable-radix-cache to force real long prefill; post-kill flush poll",
        "gpu": E.GPU, "graph_off": True, "radix_cache_disabled": True, "num_steps": E.NUM_STEPS,
        "ok": ok, "extend_gz": new_gz, "decode_gz_seen": decode_g,
        "size_bytes": Path(new_gz).stat().st_size if new_gz else 0,
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    E.log(f"  metadata updated. ok={ok} extend_gz={new_gz} decode_seen={decode_g}")
    sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
