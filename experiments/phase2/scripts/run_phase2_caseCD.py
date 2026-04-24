#!/usr/bin/env python3
"""
Phase 2 — Cases C & D variance reduction gate.

Client-only sweep: server stays up with Phase-1 default flags throughout.
Tests whether TTFT CV (20-42% in Phase 1) drops below 10% with extended warmup.

Sweep:
  V0  warmup=30  (Phase-1 baseline)  bench_n=2000/1000  3 reps
  V1  warmup=100                     bench_n=2000/1000  3 reps
  V2  warmup=300  bench_n=4000/2000  5 reps

Decision per case:
  CV(V2) ≤ 10%  → promote to Phase 3 with V2 warmup setting
  CV(V2) > 10%  → check per-request TTFT distribution for bimodality → drop

Usage (from /data/profiling_lab):
    python3 experiments/phase2/scripts/run_phase2_caseCD.py
"""

import hashlib, json, os, statistics, subprocess, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB     = Path("/data/profiling_lab")
PORT    = 30000
SEED    = 1

RAW_DIR = LAB / "experiments/phase2_shaping/caseCD"
LOG_DIR = LAB / "logs/phase2"

# Phase-1 references
PHASE1 = {
    "caseC_batched": {"ttft_p50": 243.9, "ttft_cv": 20.6, "vllm_ttft": 164.1},
    "caseD_decode":  {"ttft_p50": 247.0, "ttft_cv":  2.6, "vllm_ttft": 184.8},
}

CASES = {
    "caseC_batched": dict(concurrency=16, dataset="datasets/caseC_batched.jsonl"),
    "caseD_decode":  dict(concurrency=16, dataset="datasets/caseD_decode.jsonl"),
}

VARIANTS = [
    dict(name="V0_warmup30",  warmup=30,  bench_n_C=2000, bench_n_D=1000, reps=3),
    dict(name="V1_warmup100", warmup=100, bench_n_C=2000, bench_n_D=1000, reps=3),
    dict(name="V2_warmup300", warmup=300, bench_n_C=4000, bench_n_D=2000, reps=5),
]

BASE_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "6",
    "HF_HUB_OFFLINE": "1",
}

SERVER_FLAGS = [
    "--model-path", SNAPSHOT,
    "--dtype", "bfloat16",
    "--port", str(PORT),
    "--tp", "1",
    "--attention-backend", "flashinfer",
]


def log(msg):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_for_server(port, timeout=360):
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
        log("  Terminating server ...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log("  Server stopped.")


def run_bench(case_name, variant_name, rep, concurrency, bench_n, warmup, dataset_path):
    out_json  = RAW_DIR / f"{case_name}_{variant_name}_rep{rep}.json"
    meta_json = RAW_DIR / f"{case_name}_{variant_name}_rep{rep}_meta.json"

    log(f"  bench {case_name} {variant_name} rep{rep} (conc={concurrency}, n={bench_n}, warmup={warmup})")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang-oai",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--dataset-name", "autobench",
        "--dataset-path", str(dataset_path),
        "--max-concurrency", str(concurrency),
        "--num-prompts", str(bench_n),
        "--seed", str(SEED),
        "--warmup-requests", str(warmup),
        "--output-file", str(out_json),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = time.time() - t0

    combined = result.stdout + result.stderr
    if result.returncode != 0:
        log(f"  ERROR bench rc={result.returncode}: {combined[-600:]}")
        return None

    metrics = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text())
            metrics = {
                "ttft_p50_ms": data.get("median_ttft_ms"),
                "ttft_p99_ms": data.get("p99_ttft_ms"),
                "ttft_std_ms": data.get("std_ttft_ms"),
                "ttft_mean_ms": data.get("mean_ttft_ms"),
                "tpot_p50_ms": data.get("median_tpot_ms"),
                "output_throughput": data.get("output_throughput"),
                "request_throughput": data.get("request_throughput"),
            }
        except Exception as e:
            log(f"  WARN: JSON parse failed: {e}")

    for line in combined.split("\n"):
        if "Median TTFT" in line or "P99 TTFT" in line:
            log(f"  stdout: {line.strip()}")

    meta = {
        "case": case_name, "variant": variant_name, "rep": rep,
        "warmup": warmup, "bench_n": bench_n, "concurrency": concurrency,
        "elapsed_s": round(elapsed, 1),
        "dataset_sha256": sha256_file(dataset_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    ttft = metrics.get("ttft_p50_ms")
    std  = metrics.get("ttft_std_ms")
    cv   = (std / ttft * 100) if (ttft and std) else None
    log(f"  rep{rep} done: elapsed={elapsed:.0f}s TTFT_p50={ttft} ms std={std} cv={cv:.1f}%" if cv else f"  rep{rep} done: TTFT_p50={ttft}")
    return metrics


def compute_cv(ttfts):
    """CV across reps (std of p50 values / median of p50 values)."""
    if len(ttfts) < 2:
        return None
    med = statistics.median(ttfts)
    std = statistics.stdev(ttfts)
    return std / med * 100 if med else None


def write_summary(all_results):
    """all_results: {case_name: {variant_name: [ttft_p50, ...]}}"""
    path = RAW_DIR / "summary.md"
    lines = [
        "# Phase 2 — Cases C & D Variance Reduction Gate",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()} UTC",
        "\nGoal: Determine if TTFT CV drops ≤10% with extended warmup.\n",
    ]
    for case_name, variants in all_results.items():
        ref = PHASE1[case_name]
        lines += [
            f"## {case_name}",
            f"\nPhase-1 reference: TTFT p50 = {ref['ttft_p50']} ms (CV={ref['ttft_cv']}%) | vLLM = {ref['vllm_ttft']} ms\n",
            "| Variant | warmup | reps | TTFT p50 median (ms) | CV% (across reps) | Decision |",
            "|---|---|---|---|---|---|",
        ]
        warmup_map = {"V0_warmup30": 30, "V1_warmup100": 100, "V2_warmup300": 300}
        reps_map   = {"V0_warmup30": 3,  "V1_warmup100": 3,   "V2_warmup300": 5}
        for vname, ttfts in variants.items():
            if not ttfts:
                lines.append(f"| {vname} | {warmup_map.get(vname,'?')} | — | FAILED | — | — |")
                continue
            med = statistics.median(ttfts)
            cv  = compute_cv(ttfts)
            cv_str = f"{cv:.1f}%" if cv is not None else "—"
            if cv is not None and cv <= 10:
                decision = "✅ CV ≤10% — profilable"
            elif cv is not None and cv <= 20:
                decision = "⚠ CV borderline — monitor"
            else:
                decision = "❌ CV >20% — noisy"
            lines.append(f"| {vname} | {warmup_map.get(vname,'?')} | {reps_map.get(vname,'?')} | {med:.1f} | {cv_str} | {decision} |")

        # Overall decision for this case
        v2_ttfts = variants.get("V2_warmup300", [])
        v2_cv = compute_cv(v2_ttfts) if v2_ttfts else None
        if v2_cv is not None:
            if v2_cv <= 10:
                verdict = f"**PROMOTE** — V2 CV={v2_cv:.1f}% ≤ 10%. Use warmup=300, bench_n doubled for Phase 3."
            else:
                verdict = f"**DROP** — V2 CV={v2_cv:.1f}% > 10%. Not profilable at c=16 with current settings."
        else:
            verdict = "**INCONCLUSIVE** — V2 data missing."
        lines += [f"\n**Case verdict:** {verdict}\n"]

    lines += ["## Decision", "", "_To be filled after analysis._"]
    path.write_text("\n".join(lines) + "\n")
    log(f"\nSummary → {path}")


def main():
    log("=== Phase 2 Cases C & D — variance reduction gate ===")
    log("Server: default flags (no scheduler shaping — client-only sweep)")

    # Single server launch for all C/D runs
    server_log = LOG_DIR / "sglang_caseCD.log"
    env = {
        **BASE_ENV,
        "SGLANG_KERNEL_API_LOGLEVEL": "1",
        "SGLANG_KERNEL_API_LOGDEST": str(LOG_DIR / "sglang_%i.log"),
    }
    cmd = ["python3", "-m", "sglang.launch_server"] + SERVER_FLAGS
    log(f"Launching SGLang → {server_log}")
    f = open(server_log, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)

    try:
        if not wait_for_server(PORT, timeout=360):
            log("ERROR: server never became healthy")
            kill_server(proc)
            return
        log("Server healthy. Starting variance sweep.")

        # {case: {variant: [ttft_p50_values]}}
        all_results = {c: {} for c in CASES}

        for vdict in VARIANTS:
            vname   = vdict["name"]
            warmup  = vdict["warmup"]
            reps    = vdict["reps"]
            log(f"\n── Variant {vname} (warmup={warmup}, reps={reps}) ──")

            for case_name, cfg in CASES.items():
                bench_n = vdict["bench_n_C"] if case_name == "caseC_batched" else vdict["bench_n_D"]
                dataset = LAB / cfg["dataset"]
                ttft_list = []
                for rep in range(1, reps + 1):
                    m = run_bench(
                        case_name, vname, rep,
                        concurrency=cfg["concurrency"],
                        bench_n=bench_n,
                        warmup=warmup,
                        dataset_path=dataset,
                    )
                    if m and m.get("ttft_p50_ms"):
                        ttft_list.append(m["ttft_p50_ms"])

                all_results[case_name][vname] = ttft_list
                if ttft_list:
                    med = statistics.median(ttft_list)
                    cv  = compute_cv(ttft_list)
                    log(f"  {case_name} {vname}: median={med:.1f} ms  CV={cv:.1f}%" if cv else f"  {case_name} {vname}: median={med:.1f} ms")

    finally:
        kill_server(proc)

    write_summary(all_results)
    log("\n=== Cases C & D variance sweep complete ===")


if __name__ == "__main__":
    main()
