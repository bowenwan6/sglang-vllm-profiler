#!/usr/bin/env python3
"""
Issue #4 v3 — IMG-A runner (plan.md §11.2 matrix, §11.3 phases 1.4 and 2).

Port of the v2 runner (`v2/image_text_benchmarks/scripts/run_image_text_imgA_fixed.py`,
649 lines) to the v3 flag surface. Bracket ordering, drift gating, forbidden-token
guards and artifact layout are carried over unchanged; what is replaced is the
variant matrix, the flag surface, and the addition of per-arm engagement
verification.

Differences from v2, each forced by an upstream change (plan.md §11.1):

  A  transport is selected with ``--mm-feature-transport`` (unset ⇒ cpu), never
     with the deprecated ``SGLANG_USE_CUDA_IPC_TRANSPORT`` env var;
  B  IPC arms are checked for pool CPU-fallback rather than trusted;
  C  ``--cuda-graph-backend-prefill`` replaces ``--enforce-piecewise-cuda-graph``,
     and piecewise arms are checked for the eager-fallback warning, which no
     longer crashes;
  D  the bench client is ``sglang.benchmark.serving``, not the deprecated
     ``sglang.bench_serving`` shim.

Stack is frozen in ../manifest.md. GPU 7 only, never auto-switched.

Modes:
  --mode smoke     Phase 1.4 — 20 prompts, 1 rep per arm, engagement only.
  --mode headline  Phase 2   — 400 prompts, 5 reps, full bracket + drift gate.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from engagement_verify import fetch_server_info, one_line, verify_arm  # noqa: E402

# ----- Frozen paths (see ../manifest.md) -----
SNAPSHOT = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
LAB = Path("/data/sglang-vllm-profiler")
BASE = LAB / "experiments/qwen3vl8b/v3_issue4"
RESULTS = BASE / "results"
LOGS = LAB / "logs/qwen3vl8b/v3_issue4"
SGLANG_SRC_PY = "/data/sglang-fork/python"
VLLM_PYTHON = "/opt/miniconda3/envs/profiling/bin/python"
LIBCUDA = "/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05"

GPU = "7"                 # user-specified; never auto-switch
SGLANG_PORT = 30000
VLLM_PORT = 30001
GPU_IDLE_MIB = 2000
SEED = 1
CONCURRENCY = 1

IMAGE_ARGS_BASE = [
    "--dataset-name", "image",
    "--image-count", "1",
    "--image-resolution", "720p",
    "--image-format", "png",
    "--image-content", "random",
    "--random-input-len", "128",
    "--random-output-len", "128",
    "--random-range-ratio", "1.0",
    "--seed", str(SEED),
]

# (arm_id, framework, mm_transport|None, prefill_backend|None, server_wait_s)
# `None` means "leave the flag unset" — for A0 that is the point: we record what
# it resolves to rather than asserting it.
ARMS = {
    "A0_default":  ("sglang", None,       None,           480),
    "A1_disabled": ("sglang", None,       "disabled",     480),
    "A2_tcp":      ("sglang", None,       "tc_piecewise", 1200),
    "A3_bcg":      ("sglang", None,       "breakable",    900),
    "A4_ipc":      ("sglang", "cuda_ipc", None,           480),
    "A5_ipc_best": ("sglang", "cuda_ipc", None,           900),  # backend filled at runtime
    "V0_vllm":     ("vllm",   None,       None,           900),
    "A0_repeat":   ("sglang", None,       None,           480),
}

SMOKE_ORDER = ["A0_default", "A1_disabled", "A2_tcp", "A3_bcg", "A4_ipc"]
HEADLINE_ORDER = ["A0_default", "A1_disabled", "A2_tcp", "A3_bcg",
                  "A4_ipc", "A5_ipc_best", "V0_vllm", "A0_repeat"]


def _base_env():
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": GPU,
           "HF_HUB_OFFLINE": "1",
           "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK": "1"}
    prev_ld = env.get("LD_PRELOAD", "")
    env["LD_PRELOAD"] = LIBCUDA + (os.pathsep + prev_ld if prev_ld else "")
    # Never carry the deprecated transport env or any kernel-API logging.
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
              "SGLANG_USE_CUDA_IPC_TRANSPORT"):
        env.pop(k, None)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SGLANG_SRC_PY + (os.pathsep + prev if prev else "")
    return env


BASE_ENV = _base_env()


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def gpu_used():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used",
         "--format=csv,noheader,nounits", "-i", GPU],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_server(port, timeout, proc=None):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            log(f"  ERROR: server exited early rc={proc.returncode}")
            return False
        try:
            if urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3).getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def kill_server(proc, patt):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    subprocess.run(["pkill", "-9", "-f", patt], capture_output=True)
    for _ in range(60):
        u = gpu_used()
        if 0 <= u < GPU_IDLE_MIB:
            log(f"  GPU {GPU} freed (used={u} MiB)")
            return True
        time.sleep(3)
    log(f"  WARNING: GPU {GPU} still {gpu_used()} MiB after kill")
    return False


def parse_bench_jsonl(path: Path):
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except Exception:
                pass
    return last


def run_rep(arm, port, rep, env, raw_dir, num_prompts):
    out = raw_dir / f"{arm}_rep{rep}.jsonl"
    out.unlink(missing_ok=True)
    cmd = (["python3", "-m", "sglang.benchmark.serving",
            "--backend", "sglang-oai-chat",
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", SNAPSHOT,
            "--num-prompts", str(num_prompts),
            "--max-concurrency", str(CONCURRENCY)]
           + IMAGE_ARGS_BASE + ["--output-file", str(out)])
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = round(time.time() - t0, 1)
    if res.returncode != 0:
        tail = (res.stdout + res.stderr)[-600:]
        log(f"    rep{rep} FAILED rc={res.returncode}: {tail}")
        return False, {"status": "BENCH_FAILED", "rc": res.returncode,
                       "stderr_tail": tail}
    d = parse_bench_jsonl(out)
    if d is None:
        return False, {"status": "PARSE_ERROR"}
    errors = d.get("errors", []) or []
    n_fail = sum(1 for e in errors if e)
    forbidden = any("No data iterator found for token" in str(e) for e in errors if e)
    ttfts = [t * 1000 for t in (d.get("ttfts") or []) if t is not None]
    m = {
        "completed": d.get("completed"),
        "failures": n_fail,
        "forbidden_token_error_present": forbidden,
        "ttft_p50": percentile(ttfts, 50) if ttfts else d.get("median_ttft_ms"),
        "ttft_p95": percentile(ttfts, 95) if ttfts else None,
        "ttft_p99": percentile(ttfts, 99) if ttfts else d.get("p99_ttft_ms"),
        "tpot_p50": d.get("median_tpot_ms"),
        "tpot_p99": d.get("p99_tpot_ms"),
        "e2e_p50": d.get("median_e2e_latency_ms"),
        "out_tok_s": d.get("output_throughput"),
        "req_s": d.get("request_throughput"),
        "total_input_vision_tokens": d.get("total_input_vision_tokens"),
        "total_input_text_tokens": d.get("total_input_text_tokens"),
        "errors_sample": [str(e) for e in errors[:2] if e],
        "elapsed_s": elapsed,
    }
    log(f"    rep{rep} {elapsed}s completed={m['completed']} fail={n_fail} "
        f"ttft_p50={m['ttft_p50']}ms tpot_p50={m['tpot_p50']}ms")
    return (n_fail == 0 and not forbidden), m


def run_arm(arm, backend_override, num_prompts, reps, raw_dir, warmup):
    fw, transport, backend, wait_s = ARMS[arm]
    if backend_override is not None:
        backend = backend_override
    log("\n" + "=" * 64)
    log(f"ARM {arm}  fw={fw}  transport={transport or 'unset(→cpu)'}  "
        f"prefill_backend={backend or 'unset(default)'}")
    log("=" * 64)

    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"  STOP: GPU {GPU} not idle (used={u} MiB)")
        return {"arm": arm, "status": "GPU_NOT_IDLE", "gpu_mib": u}

    env = {**BASE_ENV}
    port = SGLANG_PORT if fw == "sglang" else VLLM_PORT
    patt = ("sglang.launch_server" if fw == "sglang"
            else "vllm.entrypoints.openai.api_server")

    if fw == "sglang":
        cmd = ["python3", "-m", "sglang.launch_server",
               "--model-path", SNAPSHOT, "--dtype", "bfloat16",
               "--port", str(port), "--tp", "1",
               "--attention-backend", "flashinfer"]
        if transport is not None:
            cmd += ["--mm-feature-transport", transport]
        if backend is not None:
            cmd += ["--cuda-graph-backend-prefill", backend]
    else:
        cmd = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
               "--model", SNAPSHOT, "--dtype", "bfloat16",
               "--port", str(port), "--tensor-parallel-size", "1"]

    LOGS.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{arm}_server.log"
    lf = open(log_path, "w")
    log(f"  launching {fw} port={port} wait≤{wait_s}s")
    log(f"  cmd: {' '.join(cmd[2:] if fw=='sglang' else cmd[1:])}")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {"arm": arm, "framework": fw, "requested_transport": transport,
           "requested_prefill_backend": backend,
           "port": port, "gpu": GPU, "snapshot": SNAPSHOT.split("/")[-1],
           "num_prompts": num_prompts, "warmup": warmup, "reps_planned": reps,
           "concurrency": CONCURRENCY, "seed": SEED,
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}

    try:
        if not wait_server(port, wait_s, proc=proc):
            rec["status"] = "SERVER_NO_START"
            log(f"  ERROR: server did not come up in {wait_s}s")
            return rec

        info = fetch_server_info(port)
        if info is not None:
            (raw_dir / f"{arm}_server_info.json").write_text(json.dumps(info, indent=2))

        if warmup:
            log(f"  warmup {warmup} prompts (discarded)")
            run_rep(arm, port, 0, env, raw_dir, warmup)

        rep_list, any_fail = [], False
        for rep in range(1, reps + 1):
            ok, m = run_rep(arm, port, rep, env, raw_dir, num_prompts)
            rep_list.append(m)
            if m.get("forbidden_token_error_present"):
                log("  forbidden-token error — stopping this arm")
                any_fail = True
                rec["forbidden_token_error_present"] = True
                break
            if not ok or (m.get("failures") or 0) > 0:
                log("  failures>0 — stopping this arm")
                any_fail = True
                break
        rec["reps"] = rep_list
        rec["status"] = "INVALID_FAILURES" if any_fail else "OK"

        # Engagement verdict is computed while the server is still alive so
        # /server_info is reachable; the log scan sees the full run.
        lf.flush()
        verdict = verify_arm(arm, backend, transport, info, log_path)
        rec["engagement"] = verdict
        log(f"  {one_line(verdict)}")
    finally:
        lf.flush()
        lf.close()
        kill_server(proc, patt)

    if rec.get("status") == "OK":
        p50s = [r["ttft_p50"] for r in rec["reps"] if r.get("ttft_p50") is not None]
        if p50s:
            rec["ttft_p50_reps"] = p50s
            rec["ttft_p50_median"] = round(statistics.median(p50s), 3)
            rec["ttft_p50_cv_pct"] = (
                round(100 * statistics.pstdev(p50s) / statistics.mean(p50s), 1)
                if len(p50s) > 1 and statistics.mean(p50s) else 0.0)
        for k in ("ttft_p95", "ttft_p99", "tpot_p50", "tpot_p99",
                  "e2e_p50", "out_tok_s"):
            vals = [r[k] for r in rec["reps"] if r.get(k) is not None]
            rec[f"{k}_median"] = round(statistics.median(vals), 3) if vals else None
        last = rec["reps"][-1] if rec["reps"] else {}
        rec["vision_tok_per_req"] = (last.get("total_input_vision_tokens") or 0) // max(num_prompts, 1)
        rec["text_tok_per_req"] = (last.get("total_input_text_tokens") or 0) // max(num_prompts, 1)
        log(f"  {arm}: ttft_p50_median={rec.get('ttft_p50_median')}ms "
            f"cv={rec.get('ttft_p50_cv_pct')}%")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("smoke", "headline"), required=True)
    ap.add_argument("--arms", default=None,
                    help="comma-separated subset; default = the mode's order")
    ap.add_argument("--out", default=None, help="results subdirectory name")
    a = ap.parse_args()

    if a.mode == "smoke":
        num_prompts, reps, warmup = 20, 1, 0
        order = SMOKE_ORDER
        out_name = a.out or "phase1_engagement_smoke"
    else:
        num_prompts, reps, warmup = 400, 5, 30
        order = HEADLINE_ORDER
        out_name = a.out or "phase2_imgA_headline"
    if a.arms:
        order = [x.strip() for x in a.arms.split(",") if x.strip()]

    outdir = RESULTS / out_name
    raw = outdir / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"mode={a.mode} arms={order} n={num_prompts} reps={reps} warmup={warmup}")
    log(f"out={outdir}")

    results, backend_override_for_a5 = [], None
    for arm in order:
        override = backend_override_for_a5 if arm == "A5_ipc_best" else None
        if arm == "A5_ipc_best" and override is None:
            # Choose the winner of {A2_tcp, A3_bcg} among VERIFIED arms only.
            cands = [r for r in results
                     if r["arm"] in ("A2_tcp", "A3_bcg")
                     and r.get("status") == "OK"
                     and r.get("engagement", {}).get("engagement") == "VERIFIED"
                     and r.get("ttft_p50_median") is not None]
            if not cands:
                log("  A5_ipc_best SKIPPED: neither A2 nor A3 produced a "
                    "VERIFIED number to compose with")
                results.append({"arm": arm, "status": "SKIPPED_NO_VERIFIED_BASE"})
                continue
            best = min(cands, key=lambda r: r["ttft_p50_median"])
            override = ARMS[best["arm"]][2]
            log(f"  A5_ipc_best composes with {best['arm']} "
                f"(backend={override}, ttft_p50={best['ttft_p50_median']}ms)")
        rec = run_arm(arm, override, num_prompts, reps, raw, warmup)
        results.append(rec)
        (outdir / "results.json").write_text(json.dumps(results, indent=2))

    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    log(f"\nwrote {outdir/'results.json'}")

    log("\n===== engagement summary =====")
    for r in results:
        e = r.get("engagement")
        log(f"  {r['arm']:<12} status={r.get('status')}  "
            + (one_line(e) if e else "engagement: n/a"))


if __name__ == "__main__":
    main()
