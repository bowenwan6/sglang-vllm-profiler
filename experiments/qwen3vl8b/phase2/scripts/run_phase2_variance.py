#!/usr/bin/env python3
"""
Phase 2 — Cases C/D variance gate + vLLM Case B/C recheck.

Variance gate (SGLang default, GPU 7):
  - Case C (512→128, c=16): warmup 30/100/300 × 3 reps
  - Case D (512→512, c=16): warmup 30/100/300 × 3 reps
  If CV 5-10% persists at W300, escalate to 5 reps; report W500 need if still failing.

vLLM recheck (GPU 7):
  - Case B: warmup=300, 5 reps, c=1
  - Case C: warmup=300, 5 reps, c=16
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
DATASETS = LAB / "datasets/qwen3vl8b"
RAW = LAB / "experiments/qwen3vl8b/phase2/raw"
LOGS = LAB / "logs/qwen3vl8b/phase2"
SGLANG_PORT = 30000
VLLM_PORT = 30001
GPU = "7"
VLLM_PYTHON = "/opt/miniconda3/envs/profiling/bin/python"
SEED = 1
EXTRA_BODY = '{"temperature": 0, "top_p": 1}'

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}

VARIANCE_CASES = {
    "caseC_batched": dict(ds="caseC_batched.jsonl", concurrency=16, bench_n=2000),
    "caseD_decode":  dict(ds="caseD_decode.jsonl",  concurrency=16, bench_n=1000),
}
WARMUP_LEVELS = [30, 100, 300]
BASE_REPS = 3
VLLM_RECHECK = {
    "caseB_longprefill": dict(ds="caseB_longprefill.jsonl", concurrency=1,  bench_n=200,  warmup=300, reps=5),
    "caseC_batched":     dict(ds="caseC_batched.jsonl",     concurrency=16, bench_n=2000, warmup=300, reps=5),
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


def kill_server(proc, pattern):
    if proc and proc.poll() is None:
        log("Terminating server...")
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
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


def parse_bench(out_path):
    try:
        d = json.load(open(out_path))
        nfail = sum(1 for e in d.get("errors", []) if e)
        ttfts_ms = [t * 1000 for t in d.get("ttfts", []) if t is not None]
        return {
            "ttft_p50": percentile(ttfts_ms, 50),
            "ttft_p95": percentile(ttfts_ms, 95),
            "ttft_p99": percentile(ttfts_ms, 99),
            "tpot_p50": d.get("median_tpot_ms"),
            "tpot_p99": d.get("p99_tpot_ms"),
            "out_tok_s": d.get("output_throughput"),
            "failures": nfail,
        }
    except Exception as ex:
        log(f"  parse error: {ex}")
        return None


def run_bench_sg(case, cfg, warmup, rep, phase_tag="variance"):
    ds = DATASETS / cfg["ds"]
    out = RAW / f"{case}_sglang_w{warmup}_{phase_tag}_rep{rep}.json"
    meta_out = RAW / f"{case}_sglang_w{warmup}_{phase_tag}_rep{rep}_meta.json"
    log(f"  bench {case} sglang w{warmup} rep{rep}")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang-oai", "--base-url", f"http://127.0.0.1:{SGLANG_PORT}",
        "--dataset-name", "autobench", "--dataset-path", str(ds),
        "--max-concurrency", str(cfg["concurrency"]),
        "--num-prompts", str(cfg["bench_n"]),
        "--seed", str(SEED), "--warmup-requests", str(warmup),
        "--extra-request-body", EXTRA_BODY, "--output-details",
        "--output-file", str(out),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = round(time.time() - t0, 2)
    parsed = parse_bench(out) if res.returncode == 0 and out.exists() else None
    if parsed:
        log(f"  OK {elapsed}s failures={parsed['failures']} ttft_p50={parsed['ttft_p50']:.1f}ms")
        if parsed["failures"] > 0:
            log(f"  STOP: {parsed['failures']} failures"); sys.exit(2)
    else:
        log(f"  ERROR rc={res.returncode}: {(res.stdout+res.stderr)[-400:]}")
        sys.exit(1)
    meta = {
        "status": "OK", "elapsed_s": elapsed, "failures": parsed["failures"],
        "ttft_p50_ms": parsed["ttft_p50"],
        "case": case, "framework": "sglang-oai", "warmup": warmup, "rep": rep,
        "concurrency": cfg["concurrency"], "bench_n": cfg["bench_n"],
        "extra_body": EXTRA_BODY,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": GPU,
    }
    json.dump(meta, open(meta_out, "w"), indent=2)
    return parsed


def run_bench_vllm(case, cfg):
    ds = DATASETS / cfg["ds"]
    warmup = cfg["warmup"]
    reps_n = cfg["reps"]
    results = []
    for rep in range(1, reps_n + 1):
        out = RAW / f"{case}_vllm_w{warmup}_recheck_rep{rep}.json"
        meta_out = RAW / f"{case}_vllm_w{warmup}_recheck_rep{rep}_meta.json"
        log(f"  bench {case} vllm w{warmup} rep{rep}/{reps_n}")
        cmd = [
            "python3", "-m", "sglang.bench_serving",
            "--backend", "vllm", "--base-url", f"http://127.0.0.1:{VLLM_PORT}",
            "--dataset-name", "autobench", "--dataset-path", str(ds),
            "--max-concurrency", str(cfg["concurrency"]),
            "--num-prompts", str(cfg["bench_n"]),
            "--seed", str(SEED), "--warmup-requests", str(warmup),
            "--extra-request-body", EXTRA_BODY, "--output-details",
            "--output-file", str(out),
        ]
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
        elapsed = round(time.time() - t0, 2)
        parsed = parse_bench(out) if res.returncode == 0 and out.exists() else None
        if parsed:
            log(f"  OK {elapsed}s failures={parsed['failures']} ttft_p50={parsed['ttft_p50']:.1f}ms")
            if parsed["failures"] > 0:
                log(f"  STOP: {parsed['failures']} failures"); sys.exit(2)
        else:
            log(f"  ERROR rc={res.returncode}: {(res.stdout+res.stderr)[-400:]}")
            sys.exit(1)
        meta = {
            "status": "OK", "elapsed_s": elapsed, "failures": parsed["failures"],
            "ttft_p50_ms": parsed["ttft_p50"],
            "case": case, "framework": "vllm", "warmup": warmup, "rep": rep,
            "concurrency": cfg["concurrency"], "bench_n": cfg["bench_n"],
            "extra_body": EXTRA_BODY,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "cuda_visible_devices": GPU,
        }
        json.dump(meta, open(meta_out, "w"), indent=2)
        results.append(parsed)
    return results


def launch_sglang(label):
    cmd = ["python3", "-m", "sglang.launch_server",
           "--model-path", SNAPSHOT, "--dtype", "bfloat16",
           "--port", str(SGLANG_PORT), "--tp", "1", "--attention-backend", "flashinfer"]
    log(f"  launch SGLang ({label})")
    lf = open(LOGS / f"{label}_sglang_server.log", "w")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(SGLANG_PORT):
        log("  ERROR: SGLang did not come up"); kill_server(proc, "sglang.launch_server"); sys.exit(1)
    log("  SGLang ready")
    return proc


def launch_vllm(label):
    cmd = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
           "--model", SNAPSHOT, "--dtype", "bfloat16",
           "--port", str(VLLM_PORT), "--tensor-parallel-size", "1"]
    log(f"  launch vLLM ({label})")
    lf = open(LOGS / f"{label}_vllm_server.log", "w")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(VLLM_PORT):
        log("  ERROR: vLLM did not come up"); kill_server(proc, "vllm.entrypoints"); sys.exit(1)
    log("  vLLM ready")
    return proc


def cv_stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    med = statistics.median(vals)
    cv = (statistics.stdev(vals) / med * 100) if (len(vals) >= 2 and med) else 0.0
    return med, cv


def main():
    os.chdir(LAB)
    variance_gate_results = {}

    # ===== VARIANCE GATE: Cases C and D =====
    log("=== Phase 2 Variance Gate (Cases C and D) ===")

    for case, cfg in VARIANCE_CASES.items():
        log(f"--- {case} variance gate ---")
        case_results = {}  # warmup -> {reps: [...], median_p50, cv}

        for warmup in WARMUP_LEVELS:
            log(f"  warmup={warmup}")
            proc = launch_sglang(f"{case}_w{warmup}")
            rep_data = []
            try:
                for rep in range(1, BASE_REPS + 1):
                    parsed = run_bench_sg(case, cfg, warmup, rep)
                    rep_data.append(parsed["ttft_p50"])
            finally:
                kill_server(proc, "sglang.launch_server")

            med, cv = cv_stats(rep_data)
            log(f"  w{warmup}: median_p50={med:.1f}ms cv={cv:.1f}%")

            # Escalate to 5 reps if CV 5-10%
            if cv is not None and 5.0 <= cv <= 10.0:
                log(f"  CV {cv:.1f}% in 5-10% zone — escalating to 5 reps")
                proc = launch_sglang(f"{case}_w{warmup}_ext")
                try:
                    for rep in range(BASE_REPS + 1, 6):
                        parsed = run_bench_sg(case, cfg, warmup, rep, "variance_ext")
                        rep_data.append(parsed["ttft_p50"])
                finally:
                    kill_server(proc, "sglang.launch_server")
                med, cv = cv_stats(rep_data)
                log(f"  w{warmup} (5 reps): median_p50={med:.1f}ms cv={cv:.1f}%")

            case_results[warmup] = {"reps": rep_data, "median_p50_ms": med, "cv_pct": cv}

            # Check if gate passed
            if cv is not None and cv < 5.0:
                log(f"  GATE PASSED at w{warmup}: CV={cv:.1f}% < 5%")
                break

        # Check if W300 still failing
        w300_cv = case_results.get(300, {}).get("cv_pct")
        if w300_cv is not None and w300_cv >= 5.0:
            log(f"  WARNING: {case} CV={w300_cv:.1f}% at W300 — W500 may be needed, reporting only")

        # Determine recommended warmup
        rec_warmup = 300
        for w in WARMUP_LEVELS:
            if case_results.get(w, {}).get("cv_pct", 100) < 5.0:
                rec_warmup = w
                break
        variance_gate_results[case] = {
            "warmup_results": case_results,
            "recommended_warmup": rec_warmup,
            "recommended_reps": 5 if case_results.get(rec_warmup, {}).get("cv_pct", 100) >= 5.0 else 3,
            "gate_passed": w300_cv is None or w300_cv < 5.0,
        }
        log(f"  {case} → recommended warmup={rec_warmup}")

    json.dump(variance_gate_results, open(RAW / "variance_gate_results.json", "w"), indent=2)
    log("Variance gate complete — saved variance_gate_results.json")

    # ===== vLLM RECHECK: Cases B and C =====
    log("=== Phase 2 vLLM Recheck (Cases B and C) ===")
    vllm_recheck_results = {}

    proc = launch_vllm("vllm_recheck")
    try:
        for case, cfg in VLLM_RECHECK.items():
            log(f"--- vLLM recheck {case} warmup={cfg['warmup']} reps={cfg['reps']} ---")
            rep_parsed = run_bench_vllm(case, cfg)
            p50s = [r["ttft_p50"] for r in rep_parsed]
            med, cv = cv_stats(p50s)
            log(f"  {case}: median_p50={med:.1f}ms cv={cv:.1f}%")
            bimodal_resolved = cv is not None and cv < 15.0
            ceiling_m = not bimodal_resolved
            log(f"  bimodal resolved: {bimodal_resolved} | ceiling_M: {ceiling_m}")
            vllm_recheck_results[case] = {
                "warmup": cfg["warmup"], "reps": cfg["reps"],
                "p50s_ms": p50s, "median_p50_ms": med, "cv_pct": cv,
                "bimodal_resolved": bimodal_resolved, "ceiling_M": ceiling_m,
            }
    finally:
        kill_server(proc, "vllm.entrypoints")

    json.dump(vllm_recheck_results, open(RAW / "vllm_recheck_results.json", "w"), indent=2)
    log("vLLM recheck complete — saved vllm_recheck_results.json")
    log("=== Phase 2 variance + vLLM recheck complete ===")


if __name__ == "__main__":
    main()
