#!/usr/bin/env python3
"""
Phase 2 Step 2.4 — vLLM baseline stability recheck.

Verifies whether vLLM TTFT baselines for promoted Phase-3 cases are stable.

Cases:
  B (2048→128, c=1)  — Phase-1 vLLM cv=99.3% ⚠  needs recheck
  C (512→128, c=16)  — Phase-1 vLLM cv=9.9%  ⚠  now promoted, needs recheck

Protocol: warmup=300, 5 reps per case. Single vLLM server for both cases.

Decision per case:
  CV < 10%   → baseline clean, Phase-3 comparison has no ceiling
  CV 10–30%  → noisy, carry ±uncertainty note in downstream conclusions
  CV > 30%   → bimodal/jitter, tag all vLLM comparisons with confidence ceiling M

Usage (from /data/profiling_lab):
    python3 experiments/phase2/scripts/run_phase2_vllm_recheck.py

Outputs:
    experiments/phase2_shaping/vllm_recheck_caseB.json
    experiments/phase2_shaping/vllm_recheck_caseC.json
    logs/phase2/vllm_recheck.log
"""

import json, os, statistics, subprocess, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB      = Path("/data/profiling_lab")
PORT     = 30001
SEED     = 1
WARMUP   = 300
REPS     = 5

CASES = {
    "caseB": dict(
        dataset  = LAB / "datasets/caseB_longprefill.jsonl",
        bench_n  = 200,
        conc     = 1,
        phase1_ttft = 24.1,
        phase1_cv   = 99.3,
        out_json = LAB / "experiments/phase2_shaping/vllm_recheck_caseB.json",
    ),
    "caseC": dict(
        dataset  = LAB / "datasets/caseC_batched.jsonl",
        bench_n  = 2000,
        conc     = 16,
        phase1_ttft = 164.1,
        phase1_cv   = 9.9,
        out_json = LAB / "experiments/phase2_shaping/vllm_recheck_caseC.json",
    ),
}

VLLM_BIN = "/opt/miniconda3/envs/vllm/bin/python"

BASE_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "6",
    "HF_HUB_OFFLINE": "1",
}


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def wait_for_server(port, timeout=420):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(url, timeout=3).getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def kill_server(proc):
    if proc and proc.poll() is None:
        log("  Terminating vLLM server ...")
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log("  Server stopped.")


def run_bench(case_name, cfg, rep):
    tmp_json = LAB / f"experiments/phase2_shaping/vllm_recheck_{case_name}_rep{rep}.json"
    log(f"  bench {case_name} rep{rep} (conc={cfg['conc']}, n={cfg['bench_n']}, warmup={WARMUP})")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "vllm",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--dataset-name", "autobench",
        "--dataset-path", str(cfg["dataset"]),
        "--max-concurrency", str(cfg["conc"]),
        "--num-prompts", str(cfg["bench_n"]),
        "--seed", str(SEED),
        "--warmup-requests", str(WARMUP),
        "--output-file", str(tmp_json),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = time.time() - t0

    combined = result.stdout + result.stderr
    if result.returncode != 0:
        log(f"  ERROR rc={result.returncode}: {combined[-600:]}")
        return None

    metrics = {}
    if tmp_json.exists():
        try:
            data = json.loads(tmp_json.read_text())
            metrics = {
                "ttft_p50_ms":  data.get("median_ttft_ms"),
                "ttft_p99_ms":  data.get("p99_ttft_ms"),
                "ttft_std_ms":  data.get("std_ttft_ms"),
                "ttft_mean_ms": data.get("mean_ttft_ms"),
                "tpot_p50_ms":  data.get("median_tpot_ms"),
                "output_throughput": data.get("output_throughput"),
            }
        except Exception as e:
            log(f"  WARN: JSON parse failed: {e}")

    for line in combined.split("\n"):
        if "Median TTFT" in line:
            log(f"  stdout: {line.strip()}")

    ttft = metrics.get("ttft_p50_ms")
    std  = metrics.get("ttft_std_ms")
    cv   = (std / ttft * 100) if (ttft and std) else None
    log(f"  rep{rep} done: {elapsed:.0f}s  TTFT_p50={ttft:.1f} ms  within-rep CV={cv:.1f}%" if cv else
        f"  rep{rep} done: {elapsed:.0f}s  TTFT_p50={ttft}")
    return metrics


def decide(cv_pct, case_name):
    if cv_pct < 10:
        return "CLEAN", f"CV={cv_pct:.1f}% < 10% — baseline stable, no confidence ceiling"
    elif cv_pct < 30:
        return "NOISY", f"CV={cv_pct:.1f}% (10–30%) — carry ±uncertainty note in downstream {case_name} conclusions"
    else:
        return "CEILING_M", f"CV={cv_pct:.1f}% > 30% — tag all vLLM {case_name} comparisons with confidence ceiling M"


def main():
    log("=== Phase 2 Step 2.4 — vLLM baseline recheck (Cases B and C) ===")
    log(f"Protocol: warmup={WARMUP}, reps={REPS}")

    server_log = LAB / "logs/phase2/vllm_recheck.log"
    cmd = [
        VLLM_BIN, "-m", "vllm.entrypoints.openai.api_server",
        "--model", SNAPSHOT,
        "--dtype", "bfloat16",
        "--port", str(PORT),
        "--tensor-parallel-size", "1",
    ]
    log(f"Launching vLLM → {server_log}")
    f = open(server_log, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=BASE_ENV)

    try:
        if not wait_for_server(PORT, timeout=420):
            log("ERROR: vLLM never became healthy")
            kill_server(proc)
            return
        log("vLLM server healthy.\n")

        all_results = {}

        for case_name, cfg in CASES.items():
            log(f"── Case {case_name} (phase-1 vLLM TTFT={cfg['phase1_ttft']} ms, cv={cfg['phase1_cv']}%) ──")
            rep_ttfts = []

            for rep in range(1, REPS + 1):
                m = run_bench(case_name, cfg, rep)
                if m and m.get("ttft_p50_ms"):
                    rep_ttfts.append(m["ttft_p50_ms"])

            if len(rep_ttfts) >= 2:
                med = statistics.median(rep_ttfts)
                cv  = statistics.stdev(rep_ttfts) / med * 100
                status, note = decide(cv, case_name)
                log(f"\n  {case_name} summary: reps={rep_ttfts}")
                log(f"  median={med:.1f} ms  across-rep CV={cv:.1f}%")
                log(f"  → {status}: {note}")
            else:
                med, cv, status, note = None, None, "INCONCLUSIVE", "insufficient data"
                log(f"  {case_name}: insufficient successful reps")

            result = {
                "case": case_name,
                "warmup": WARMUP,
                "reps": REPS,
                "rep_ttft_p50_ms": rep_ttfts,
                "median_ttft_ms": med,
                "across_rep_cv_pct": round(cv, 2) if cv else None,
                "phase1_ttft_ms": cfg["phase1_ttft"],
                "phase1_cv_pct": cfg["phase1_cv"],
                "status": status,
                "note": note,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            cfg["out_json"].write_text(json.dumps(result, indent=2))
            log(f"  Saved → {cfg['out_json']}\n")
            all_results[case_name] = result

    finally:
        kill_server(proc)

    log("\n=== Step 2.4 complete — summary ===")
    for case_name, r in all_results.items():
        log(f"  {case_name}: {r['status']} — {r['note']}")


if __name__ == "__main__":
    main()
