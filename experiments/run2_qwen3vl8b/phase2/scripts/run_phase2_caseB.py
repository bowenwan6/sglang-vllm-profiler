#!/usr/bin/env python3
"""
run2 Phase 2 — Case B shaping (chunked-prefill sweep).
4 variants: default, chunk_off, chunk_1024, chunk_512.
Screen: 1 rep each. Finalist: top 2 × 3 reps.
GPU 7, SGLang system python3 only.
"""

import json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB = Path("/data/sglang-vllm-profiler")
DS = LAB / "datasets/run2_qwen3vl8b/caseB_longprefill.jsonl"
RAW = LAB / "experiments/run2_qwen3vl8b/phase2/raw"
LOGS = LAB / "logs/run2_qwen3vl8b/phase2"
PORT = 30000
GPU = "7"
BENCH_N = 200
WARMUP = 30
SEED = 1
EXTRA_BODY = '{"temperature": 0, "top_p": 1}'

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}

VARIANTS = {
    "default":    [],
    "chunk_off":  ["--chunked-prefill-size", "-1"],
    "chunk_1024": ["--chunked-prefill-size", "1024"],
    "chunk_512":  ["--chunked-prefill-size", "512"],
}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_server(port, timeout=480):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def gpu_used():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits", "-i", GPU],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def kill_server(proc):
    if proc and proc.poll() is None:
        log("Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    for _ in range(30):
        used = gpu_used()
        if 0 <= used < 2000:
            log(f"GPU {GPU} freed (used={used} MiB)")
            return True
        time.sleep(3)
    log(f"WARNING: GPU {GPU} still {gpu_used()} MiB after shutdown")
    return False


def percentile(data, q):
    d = sorted(v for v in data if v is not None)
    if not d:
        return None
    rank = (len(d) - 1) * q / 100.0
    lo = int(rank)
    frac = rank - lo
    if lo + 1 >= len(d):
        return d[lo]
    return d[lo] + frac * (d[lo + 1] - d[lo])


def run_bench(variant, rep, phase="screen"):
    out = RAW / f"caseB_sglang_{variant}_{phase}_rep{rep}.json"
    meta_out = RAW / f"caseB_sglang_{variant}_{phase}_rep{rep}_meta.json"
    log(f"  bench caseB sglang {variant} {phase} rep{rep}")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang-oai", "--base-url", f"http://127.0.0.1:{PORT}",
        "--dataset-name", "autobench", "--dataset-path", str(DS),
        "--max-concurrency", "1", "--num-prompts", str(BENCH_N),
        "--seed", str(SEED), "--warmup-requests", str(WARMUP),
        "--extra-request-body", EXTRA_BODY, "--output-details",
        "--output-file", str(out),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = round(time.time() - t0, 2)
    nfail = None
    ttft_p50 = None
    if res.returncode == 0 and out.exists():
        try:
            d = json.load(open(out))
            nfail = sum(1 for e in d.get("errors", []) if e)
            ttfts_ms = [t * 1000 for t in d.get("ttfts", []) if t is not None]
            ttft_p50 = percentile(ttfts_ms, 50)
            log(f"  OK {elapsed}s failures={nfail} ttft_p50={ttft_p50:.1f}ms")
        except Exception as ex:
            log(f"  OK {elapsed}s (parse error: {ex})")
    else:
        log(f"  ERROR rc={res.returncode}: {(res.stdout+res.stderr)[-400:]}")
        nfail = -1
    meta = {
        "status": "OK" if res.returncode == 0 else "FAILED",
        "returncode": res.returncode,
        "elapsed_s": elapsed, "failures": nfail,
        "ttft_p50_ms": ttft_p50,
        "case": "caseB_longprefill", "variant": variant, "phase": phase, "rep": rep,
        "framework": "sglang-oai", "port": PORT, "bench_n": BENCH_N,
        "warmup": WARMUP, "concurrency": 1,
        "extra_body": EXTRA_BODY,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": GPU,
    }
    json.dump(meta, open(meta_out, "w"), indent=2)
    if nfail and nfail > 0:
        log(f"  STOP: {nfail} failures — aborting")
        sys.exit(2)
    return ttft_p50, meta["status"] == "OK"


def launch_server(variant):
    extra = VARIANTS[variant]
    cmd = ["python3", "-m", "sglang.launch_server",
           "--model-path", SNAPSHOT, "--dtype", "bfloat16",
           "--port", str(PORT), "--tp", "1", "--attention-backend", "flashinfer"] + extra
    log(f"  launch SGLang {variant}: {' '.join(extra) or '(no extra flags)'}")
    lf = open(LOGS / f"caseB_sglang_{variant}_server.log", "w")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(PORT):
        log(f"  ERROR: server {variant} did not come up"); kill_server(proc); sys.exit(1)
    log(f"  server ready")
    return proc


def main():
    os.chdir(LAB)
    log("=== Phase 2 Case B shaping ===")

    # --- SCREEN round ---
    log("--- Screen round (1 rep each) ---")
    screen_results = {}
    for variant in VARIANTS:
        proc = launch_server(variant)
        try:
            p50, ok = run_bench(variant, 1, "screen")
            screen_results[variant] = p50
        finally:
            kill_server(proc)

    log("Screen results:")
    for v, p50 in sorted(screen_results.items(), key=lambda x: x[1] or 9999):
        log(f"  {v}: {p50:.1f} ms" if p50 else f"  {v}: N/A")

    # Select finalists: top 2; always include default
    ranked = sorted(screen_results.items(), key=lambda x: x[1] or 9999)
    finalists = [v for v, _ in ranked[:2]]
    if "default" not in finalists:
        finalists[-1] = "default"
        finalists = list(dict.fromkeys(finalists))

    # Drop within 5% of default
    def_p50 = screen_results.get("default") or 9999
    filtered = []
    for v in finalists:
        vp = screen_results.get(v) or 9999
        if v == "default" or abs(vp - def_p50) / def_p50 > 0.05:
            filtered.append(v)
    if "default" not in filtered:
        filtered.insert(0, "default")
    finalists = filtered[:2]
    log(f"Finalists: {finalists}")

    # --- FINALIST round (3 reps) ---
    log("--- Finalist round (3 reps each) ---")
    finalist_p50s = {v: [] for v in finalists}
    for variant in finalists:
        proc = launch_server(variant)
        try:
            for rep in range(1, 4):
                p50, ok = run_bench(variant, rep, "finalist")
                finalist_p50s[variant].append(p50)
        finally:
            kill_server(proc)

    import statistics
    finalist_summary = {}
    for v, vals in finalist_p50s.items():
        vals = [x for x in vals if x is not None]
        med = statistics.median(vals) if vals else None
        cv = (statistics.stdev(vals) / med * 100) if (len(vals) >= 2 and med) else None
        finalist_summary[v] = {"median_p50": med, "cv": cv, "vals": vals}
        log(f"  {v}: median={med:.1f}ms cv={cv:.1f}%" if (med and cv) else f"  {v}: {med}")

    # Winner: lowest median p50; prefer default if within 5%
    def_p50f = (finalist_summary.get("default") or {}).get("median_p50") or 9999
    winner = "default"
    for v, s in sorted(finalist_summary.items(), key=lambda x: x[1].get("median_p50") or 9999):
        vp = s.get("median_p50") or 9999
        if v != "default" and vp < def_p50f * 0.95:
            winner = v
            break

    log(f"Case B winner: {winner} (flags: {' '.join(VARIANTS[winner]) or 'none'})")

    result = {
        "case": "caseB_longprefill", "winner": winner,
        "winner_flags": VARIANTS[winner],
        "screen_results": {k: v for k, v in screen_results.items()},
        "finalist_summary": {k: {"median_p50_ms": v["median_p50"], "cv": v["cv"]}
                              for k, v in finalist_summary.items()},
        "phase1_vllm_ttft_p50_ms": 20.8,
        "phase1_vllm_cv_pct": 114.6,
    }
    json.dump(result, open(RAW / "caseB_shaping_result.json", "w"), indent=2)
    log(f"Saved caseB_shaping_result.json")
    log("=== Case B shaping complete ===")


if __name__ == "__main__":
    main()
