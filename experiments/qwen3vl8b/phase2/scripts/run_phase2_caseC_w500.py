#!/usr/bin/env python3
"""
Phase 2 — Case C W500 variance probe (SGLang only, GPU 1).
Determines whether warmup=500 brings SGLang Case C TTFT p50 CV below threshold.
Single SGLang server, default config, 5 reps. No vLLM, no profiler.
"""

import json, os, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB = Path("/data/sglang-vllm-profiler")
DS = LAB / "datasets/qwen3vl8b/caseC_batched.jsonl"
RAW = LAB / "experiments/qwen3vl8b/phase2/raw"
LOGS = LAB / "logs/qwen3vl8b/phase2"
PORT = 30000
GPU = "1"
CONC = 16
BENCH_N = 2000
WARMUP = 500
REPS = 5
SEED = 1
EXTRA_BODY = '{"temperature": 0, "top_p": 1}'

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}


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


def parse_bench(path):
    d = json.load(open(path))
    nfail = sum(1 for e in d.get("errors", []) if e)
    ttfts_ms = [t * 1000 for t in d.get("ttfts", []) if t is not None]
    return {
        "ttft_p50": percentile(ttfts_ms, 50),
        "ttft_p95": percentile(ttfts_ms, 95),
        "ttft_p99": percentile(ttfts_ms, 99),
        "tpot_p50": d.get("median_tpot_ms"),
        "tpot_p99": d.get("p99_tpot_ms"),
        "out_tok_s": d.get("output_throughput"),
        "req_s": d.get("request_throughput"),
        "completed": d.get("completed"),
        "failures": nfail,
    }


def run_rep(rep):
    out = RAW / f"caseC_sglang_w500_rep{rep}.json"
    meta_out = RAW / f"caseC_sglang_w500_rep{rep}_meta.json"
    log(f"  bench caseC sglang w500 rep{rep}/{REPS}")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang-oai", "--base-url", f"http://127.0.0.1:{PORT}",
        "--dataset-name", "autobench", "--dataset-path", str(DS),
        "--max-concurrency", str(CONC), "--num-prompts", str(BENCH_N),
        "--seed", str(SEED), "--warmup-requests", str(WARMUP),
        "--extra-request-body", EXTRA_BODY, "--output-details",
        "--output-file", str(out),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = round(time.time() - t0, 2)
    if res.returncode != 0 or not out.exists():
        log(f"  ERROR rc={res.returncode}: {(res.stdout+res.stderr)[-500:]}")
        sys.exit(1)
    p = parse_bench(out)
    log(f"  OK {elapsed}s completed={p['completed']} failures={p['failures']} "
        f"ttft_p50={p['ttft_p50']:.1f} p95={p['ttft_p95']:.1f} p99={p['ttft_p99']:.1f}ms")
    if p["failures"] and p["failures"] > 0:
        log(f"  STOP: {p['failures']} failures — aborting"); sys.exit(2)
    meta = {
        "status": "OK", "elapsed_s": elapsed, "failures": p["failures"],
        "ttft_p50_ms": p["ttft_p50"], "ttft_p95_ms": p["ttft_p95"], "ttft_p99_ms": p["ttft_p99"],
        "tpot_p50_ms": p["tpot_p50"], "tpot_p99_ms": p["tpot_p99"],
        "out_tok_s": p["out_tok_s"], "req_s": p["req_s"],
        "case": "caseC_batched", "framework": "sglang-oai", "config": "default",
        "warmup": WARMUP, "rep": rep, "concurrency": CONC, "num_prompts": BENCH_N,
        "seed": SEED, "extra_body": EXTRA_BODY, "port": PORT,
        "dataset_path": str(DS),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": GPU, "snapshot_sha": SNAPSHOT.split("/")[-1],
    }
    json.dump(meta, open(meta_out, "w"), indent=2)
    return p


def cv_stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    cv = (statistics.stdev(vals) / med * 100) if (len(vals) >= 2 and med) else 0.0
    return med, cv


def main():
    os.chdir(LAB)
    log("=== Phase 2 Case C W500 variance probe (SGLang, GPU 1) ===")
    if not DS.exists():
        log(f"ERROR: dataset missing {DS}"); sys.exit(1)

    cmd = ["python3", "-m", "sglang.launch_server",
           "--model-path", SNAPSHOT, "--dtype", "bfloat16",
           "--port", str(PORT), "--tp", "1", "--attention-backend", "flashinfer"]
    sg_env = {**BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
              "SGLANG_KERNEL_API_LOGDEST": str(LOGS / "caseC_w500_sglang_%i.log")}
    log("  launch SGLang (caseC_w500, default config)")
    lf = open(LOGS / "caseC_w500_sglang_server.log", "w")
    proc = subprocess.Popen(cmd, env=sg_env, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(PORT):
        log("  ERROR: SGLang did not come up"); kill_server(proc); sys.exit(1)
    log("  SGLang ready")

    reps = []
    try:
        for rep in range(1, REPS + 1):
            reps.append(run_rep(rep))
    finally:
        kill_server(proc)

    p50s = [r["ttft_p50"] for r in reps]
    p95s = [r["ttft_p95"] for r in reps]
    p99s = [r["ttft_p99"] for r in reps]
    med_p50, cv_p50 = cv_stats(p50s)
    med_p95, _ = cv_stats(p95s)
    med_p99, _ = cv_stats(p99s)
    med_tpot50, _ = cv_stats([r["tpot_p50"] for r in reps])
    med_out, _ = cv_stats([r["out_tok_s"] for r in reps])
    med_req, _ = cv_stats([r["req_s"] for r in reps])
    tot_fail = sum(r["failures"] or 0 for r in reps)

    if cv_p50 is None:
        gate = "N/A"
    elif cv_p50 < 5.0:
        gate = "clean (CV<5%)"
    elif cv_p50 <= 10.0:
        gate = "medium (CV 5-10%)"
    else:
        gate = "high-CV (CV>10%) — not cleanly profilable"

    result = {
        "case": "caseC_batched", "framework": "sglang-oai", "config": "default",
        "warmup": WARMUP, "reps": REPS, "concurrency": CONC, "num_prompts": BENCH_N,
        "ttft_p50_ms_per_rep": p50s,
        "ttft_p50_median_ms": med_p50, "ttft_p50_cv_pct": cv_p50,
        "ttft_p95_median_ms": med_p95, "ttft_p99_median_ms": med_p99,
        "tpot_p50_median_ms": med_tpot50,
        "out_tok_s_median": med_out, "req_s_median": med_req,
        "total_failures": tot_fail,
        "gate_verdict": gate,
        "vllm_w300_reference_ms": 189.0, "vllm_w300_cv_pct": 1.9,
        "note": "vLLM not re-run at W500; W300 recheck used as stable reference (not strict same-warmup).",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    json.dump(result, open(RAW / "caseC_w500_result.json", "w"), indent=2)
    log(f"  W500 p50 reps: {[round(x,1) for x in p50s]}")
    log(f"  W500 median p50={med_p50:.1f}ms CV={cv_p50:.1f}% → {gate}")
    log(f"  total failures across reps: {tot_fail}")
    log("Saved caseC_w500_result.json")
    log("=== Case C W500 probe complete ===")


if __name__ == "__main__":
    main()
