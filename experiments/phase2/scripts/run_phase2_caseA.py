#!/usr/bin/env python3
"""
Phase 2 — Case A scheduler-overhead sweep.

Tests whether SGLang's ~50 ms TTFT at c=1 is compressible by scheduler flags.

Candidates:
  A0  baseline (overlap-schedule on, lpm, chunk=8192, stream=1)
  A1  --disable-overlap-schedule
  A2  --schedule-policy fcfs
  A3  --stream-interval 8
  A4  (conditional) best 2-way combo if any single flag moves TTFT ≥10 ms

Usage (from /data/profiling_lab):
    python3 experiments/phase2/scripts/run_phase2_caseA.py

Produces:
    experiments/phase2_shaping/caseA/{candidate}_rep{n}.json
    experiments/phase2_shaping/caseA/{candidate}_rep{n}_meta.json
    experiments/phase2_shaping/caseA/summary.md
    logs/phase2/sglang_caseA_{candidate}.log
    logs/phase2/sglang_%i.log  (L1 kernel-API boundary)
"""

import hashlib, json, os, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB       = Path("/data/profiling_lab")
PORT      = 30000
WARMUP    = 30
SEED      = 1
BENCH_N   = 400   # same as Phase-1 Case A
REPS_SCREENING = 1   # 1 rep per candidate during sweep
REPS_FINALIST  = 3   # 3 reps for structural reconfirm

DATASET   = LAB / "datasets/caseA_short.jsonl"
RAW_DIR   = LAB / "experiments/phase2_shaping/caseA"
LOG_DIR   = LAB / "logs/phase2"

# Phase-1 reference (p50 median ms)
PHASE1_SGLANG_TTFT = 54.6
PHASE1_VLLM_TTFT   = 14.1

# Candidate definitions: name → extra server flags beyond the common base
BASE_FLAGS = [
    "--model-path", SNAPSHOT,
    "--dtype", "bfloat16",
    "--port", str(PORT),
    "--tp", "1",
    "--attention-backend", "flashinfer",
]

CANDIDATES = {
    "A0_baseline":              [],
    "A1_no_overlap":            ["--disable-overlap-schedule"],
    "A2_fcfs":                  ["--schedule-policy", "fcfs"],
    "A3_stream8":               ["--stream-interval", "8"],
}

BASE_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "6",
    "HF_HUB_OFFLINE": "1",
}


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_for_server(port: int, timeout: int = 360) -> bool:
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


def launch_sglang(candidate_name: str, extra_flags: list) -> subprocess.Popen:
    server_log = LOG_DIR / f"sglang_caseA_{candidate_name}.log"
    env = {
        **BASE_ENV,
        "SGLANG_KERNEL_API_LOGLEVEL": "1",
        "SGLANG_KERNEL_API_LOGDEST": str(LOG_DIR / "sglang_%i.log"),
    }
    cmd = ["python3", "-m", "sglang.launch_server"] + BASE_FLAGS + extra_flags
    log(f"  Launching SGLang ({candidate_name}): {' '.join(extra_flags) or '(defaults)'}")
    log(f"  Server log → {server_log}")
    f = open(server_log, "w")
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc


def run_bench(candidate_name: str, rep: int) -> dict | None:
    out_json  = RAW_DIR / f"{candidate_name}_rep{rep}.json"
    meta_json = RAW_DIR / f"{candidate_name}_rep{rep}_meta.json"

    log(f"  bench rep{rep}: {candidate_name}")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", "sglang-oai",
        "--base-url", f"http://127.0.0.1:{PORT}",
        "--dataset-name", "autobench",
        "--dataset-path", str(DATASET),
        "--max-concurrency", "1",
        "--num-prompts", str(BENCH_N),
        "--seed", str(SEED),
        "--warmup-requests", str(WARMUP),
        "--output-file", str(out_json),
    ]
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = time.time() - t0

    combined = result.stdout + result.stderr
    if result.returncode != 0:
        log(f"  ERROR bench rc={result.returncode}: {combined[-600:]}")
        return None

    # Parse metrics from JSON output
    metrics = {}
    if out_json.exists():
        try:
            data = json.loads(out_json.read_text())
            metrics = {
                "ttft_p50_ms": data.get("median_ttft_ms"),
                "ttft_p99_ms": data.get("p99_ttft_ms"),
                "ttft_std_ms": data.get("std_ttft_ms"),
                "tpot_p50_ms": data.get("median_tpot_ms"),
                "output_throughput": data.get("output_throughput"),
                "request_throughput": data.get("request_throughput"),
            }
        except Exception as e:
            log(f"  WARN: could not parse {out_json}: {e}")
            # Try to extract from stdout
            for line in combined.split("\n"):
                if "TTFT" in line or "ttft" in line.lower():
                    log(f"    stdout line: {line}")

    # Also try to parse from bench_serving stdout for key metrics
    for line in combined.split("\n"):
        if "P50 TTFT" in line or "Median TTFT" in line:
            log(f"  stdout: {line.strip()}")

    meta = {
        "candidate": candidate_name,
        "rep": rep,
        "elapsed_s": round(elapsed, 1),
        "dataset_sha256": sha256_file(DATASET),
        "dataset_path": str(DATASET),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    ttft = metrics.get("ttft_p50_ms")
    log(f"  rep{rep} done: elapsed={elapsed:.0f}s TTFT_p50={ttft} ms")
    return metrics


def read_ttft(candidate_name: str, rep: int) -> float | None:
    meta_json = RAW_DIR / f"{candidate_name}_rep{rep}_meta.json"
    if not meta_json.exists():
        return None
    data = json.loads(meta_json.read_text())
    return data.get("metrics", {}).get("ttft_p50_ms")


def run_candidate(candidate_name: str, extra_flags: list, reps: int) -> list[float]:
    """Launch server, run reps, kill server. Returns list of TTFT p50 values."""
    proc = launch_sglang(candidate_name, extra_flags)
    try:
        if not wait_for_server(PORT, timeout=360):
            log(f"  ERROR: server {candidate_name} never became healthy")
            kill_server(proc)
            return []
        log(f"  Server healthy.")
        results = []
        for rep in range(1, reps + 1):
            m = run_bench(candidate_name, rep)
            if m:
                ttft = m.get("ttft_p50_ms")
                if ttft:
                    results.append(ttft)
        return results
    finally:
        kill_server(proc)
        # brief pause to free GPU memory
        time.sleep(10)


def write_summary(results: dict, finalist_results: dict):
    summary_path = RAW_DIR / "summary.md"
    lines = [
        "# Phase 2 — Case A Scheduler-Overhead Sweep",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()} UTC",
        f"\nPhase-1 reference: SGLang TTFT p50 = {PHASE1_SGLANG_TTFT} ms | vLLM = {PHASE1_VLLM_TTFT} ms",
        "\n## Screening results (1 rep each)\n",
        "| Candidate | Flags | TTFT p50 (ms) | Δ vs baseline | Δ vs vLLM |",
        "|---|---|---|---|---|",
    ]
    baseline_ttft = None
    for cname, ttfts in results.items():
        if not ttfts:
            lines.append(f"| {cname} | — | FAILED | — | — |")
            continue
        val = ttfts[0]
        if cname == "A0_baseline":
            baseline_ttft = val
        delta_base = f"{val - (baseline_ttft or PHASE1_SGLANG_TTFT):+.1f}" if baseline_ttft else "—"
        delta_vllm = f"{val - PHASE1_VLLM_TTFT:+.1f}"
        extra = CANDIDATES.get(cname, [])
        flag_str = " ".join(extra) if extra else "(default)"
        lines.append(f"| {cname} | `{flag_str}` | {val:.1f} | {delta_base} ms | {delta_vllm} ms |")

    if finalist_results:
        lines += [
            "\n## Finalist reconfirm (3 reps)\n",
            "| Candidate | TTFT p50 median (ms) | CV% | Verdict |",
            "|---|---|---|---|",
        ]
        for cname, ttfts in finalist_results.items():
            if not ttfts:
                lines.append(f"| {cname} | FAILED | — | — |")
                continue
            med = statistics.median(ttfts)
            cv = (statistics.stdev(ttfts) / med * 100) if len(ttfts) > 1 else 0
            if med <= PHASE1_VLLM_TTFT * 2:
                verdict = "CONFIGURATIONAL — gap closed or near-closed"
            elif med <= PHASE1_SGLANG_TTFT - 10:
                verdict = "PARTIAL — flag moves TTFT but floor remains"
            else:
                verdict = "STRUCTURAL — floor unchanged"
            lines.append(f"| {cname} | {med:.1f} | {cv:.1f}% | {verdict} |")

    lines += [
        "\n## Decision",
        "",
        "_To be filled after analysis — see plan §9 Phase-2 judgment criteria._",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    log(f"\nSummary written → {summary_path}")


def main():
    log("=== Phase 2 Case A — scheduler-overhead sweep ===")
    log(f"Dataset: {DATASET} (sha={sha256_file(DATASET)[:12]}...)")

    screening_results = {}  # candidate → [ttft]
    deltas = {}             # candidate → ttft delta vs baseline

    # — Screening pass (1 rep each) —
    for cname, extra in CANDIDATES.items():
        log(f"\n── Candidate {cname} ──")
        ttfts = run_candidate(cname, extra, reps=REPS_SCREENING)
        screening_results[cname] = ttfts
        if ttfts and cname != "A0_baseline" and screening_results.get("A0_baseline"):
            baseline = screening_results["A0_baseline"][0]
            deltas[cname] = baseline - ttfts[0]   # positive = improvement

    # — Analysis: find best single-flag candidates —
    baseline_ttft = (screening_results.get("A0_baseline") or [PHASE1_SGLANG_TTFT])[0]
    log(f"\n── Screening complete. Baseline TTFT: {baseline_ttft:.1f} ms ──")
    for cname, delta in deltas.items():
        log(f"  {cname}: delta={delta:+.1f} ms vs baseline")

    # Check if any single flag moved ≥10 ms
    big_movers = [(c, d) for c, d in deltas.items() if d >= 10.0]
    finalist_name = None
    finalist_extra = []

    if any(ttft <= PHASE1_VLLM_TTFT * 2
           for ttfts in screening_results.values()
           for ttft in ttfts):
        # Gap collapsed — find best
        best = min(
            [(c, t[0]) for c, t in screening_results.items() if t],
            key=lambda x: x[1]
        )
        finalist_name = best[0]
        finalist_extra = CANDIDATES.get(finalist_name, [])
        log(f"\n  CONFIGURATIONAL signal: {finalist_name} TTFT={best[1]:.1f} ms ≤ 2×vLLM ({PHASE1_VLLM_TTFT*2:.1f} ms)")
    elif big_movers:
        log(f"\n  ≥10 ms movers detected: {big_movers}")
        # Try 2-way combo of best two movers
        top2 = sorted(big_movers, key=lambda x: x[1], reverse=True)[:2]
        if len(top2) >= 2:
            c1, c2 = top2[0][0], top2[1][0]
            combo_name = f"A4_{c1.split('_',1)[1]}_{c2.split('_',1)[1]}_combo"
            combo_extra = CANDIDATES[c1] + CANDIDATES[c2]
            log(f"\n── Candidate {combo_name} (2-way combo) ──")
            combo_ttfts = run_candidate(combo_name, combo_extra, reps=REPS_SCREENING)
            screening_results[combo_name] = combo_ttfts
            CANDIDATES[combo_name] = combo_extra
            if combo_ttfts:
                deltas[combo_name] = baseline_ttft - combo_ttfts[0]
        finalist_name = min(
            [(c, t[0]) for c, t in screening_results.items() if t],
            key=lambda x: x[1]
        )[0]
        finalist_extra = CANDIDATES.get(finalist_name, [])
    else:
        log("\n  No single flag moved TTFT ≥10 ms → structural floor confirmed at screening")
        finalist_name = "A0_baseline"
        finalist_extra = []

    # — Finalist reconfirm (3 reps) —
    finalist_results = {}
    if finalist_name:
        log(f"\n── Finalist reconfirm: {finalist_name} (3 reps) ──")
        ttfts = run_candidate(finalist_name, finalist_extra, reps=REPS_FINALIST)
        finalist_results[finalist_name] = ttfts
        if ttfts:
            med = statistics.median(ttfts)
            cv  = statistics.stdev(ttfts) / med * 100 if len(ttfts) > 1 else 0
            log(f"  Finalist median TTFT: {med:.1f} ms (cv={cv:.1f}%)")

    write_summary(screening_results, finalist_results)
    log("\n=== Case A sweep complete ===")


if __name__ == "__main__":
    main()
