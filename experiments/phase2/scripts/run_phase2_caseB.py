#!/usr/bin/env python3
"""
Phase 2 — Case B chunked-prefill disentanglement sweep.

Tests whether chunked-prefill scheduling adds overhead on top of the
~56 ms structural floor confirmed in Step 2.1.

Candidates (base = default flags, no scheduler inheritance from 2.1):
  B0  --chunked-prefill-size 8192  (default; 2048-tok prompt = 1 chunk)
  B1  --chunked-prefill-size 512   (4 chunks; actually exercises chunked path)
  B2  --chunked-prefill-size 1024  (2 chunks)
  B3  --chunked-prefill-size -1    (disabled; no chunking bookkeeping at all)

Note: plan originally listed {2048, 4096, 8192, -1} but chunk≥2048 for a
2048-tok prompt is functionally identical to 8192 (single chunk). Replaced
2048/4096 with 512/1024 to actually exercise the chunked-prefill code path.

Usage (from /data/profiling_lab):
    python3 experiments/phase2/scripts/run_phase2_caseB.py
"""

import hashlib, json, os, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB     = Path("/data/profiling_lab")
PORT    = 30000
WARMUP  = 30
SEED    = 1
BENCH_N = 200   # same as Phase-1 Case B

DATASET = LAB / "datasets/caseB_longprefill.jsonl"
RAW_DIR = LAB / "experiments/phase2_shaping/caseB"
LOG_DIR = LAB / "logs/phase2"

# Phase-1 references
PHASE1_SGLANG_TTFT = 62.3
PHASE1_VLLM_TTFT   = 24.1

BASE_SERVER_FLAGS = [
    "--model-path", SNAPSHOT,
    "--dtype", "bfloat16",
    "--port", str(PORT),
    "--tp", "1",
    "--attention-backend", "flashinfer",
]

# Candidates: name → extra flags on top of base
CANDIDATES = {
    "B0_chunk8192": ["--chunked-prefill-size", "8192"],   # default
    "B1_chunk512":  ["--chunked-prefill-size", "512"],    # 4 chunks
    "B2_chunk1024": ["--chunked-prefill-size", "1024"],   # 2 chunks
    "B3_disabled":  ["--chunked-prefill-size", "-1"],     # chunking off
}

REPS_SCREENING = 1
REPS_FINALIST  = 3

BASE_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "6",
    "HF_HUB_OFFLINE": "1",
}


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


def launch_sglang(candidate_name, extra_flags):
    server_log = LOG_DIR / f"sglang_caseB_{candidate_name}.log"
    env = {
        **BASE_ENV,
        "SGLANG_KERNEL_API_LOGLEVEL": "1",
        "SGLANG_KERNEL_API_LOGDEST": str(LOG_DIR / "sglang_%i.log"),
    }
    cmd = ["python3", "-m", "sglang.launch_server"] + BASE_SERVER_FLAGS + extra_flags
    log(f"  Launching SGLang ({candidate_name}): {' '.join(extra_flags)}")
    f = open(server_log, "w")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)


def run_bench(candidate_name, rep):
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
            log(f"  WARN: JSON parse failed: {e}")

    for line in combined.split("\n"):
        if "Median TTFT" in line:
            log(f"  stdout: {line.strip()}")

    meta = {
        "candidate": candidate_name,
        "rep": rep,
        "elapsed_s": round(elapsed, 1),
        "dataset_sha256": sha256_file(DATASET),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
    }
    meta_json.write_text(json.dumps(meta, indent=2))
    ttft = metrics.get("ttft_p50_ms")
    log(f"  rep{rep} done: elapsed={elapsed:.0f}s TTFT_p50={ttft} ms")
    return metrics


def run_candidate(candidate_name, extra_flags, reps):
    proc = launch_sglang(candidate_name, extra_flags)
    try:
        if not wait_for_server(PORT, timeout=360):
            log(f"  ERROR: server {candidate_name} never became healthy")
            kill_server(proc)
            return []
        log("  Server healthy.")
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
        time.sleep(10)


def write_summary(screening, finalist_results):
    path = RAW_DIR / "summary.md"
    baseline_ttft = (screening.get("B0_chunk8192") or [PHASE1_SGLANG_TTFT])[0]
    lines = [
        "# Phase 2 — Case B Chunked-Prefill Disentanglement Sweep",
        f"\nGenerated: {datetime.now(timezone.utc).isoformat()} UTC",
        f"\nPhase-1 reference: SGLang TTFT p50 = {PHASE1_SGLANG_TTFT} ms | vLLM = {PHASE1_VLLM_TTFT} ms",
        f"\nCase B: 2048-token prompt, output=128, concurrency=1",
        "\n## Screening results (1 rep each)\n",
        "| Candidate | chunk-prefill-size | Chunks | TTFT p50 (ms) | Δ vs B0 | Δ vs vLLM |",
        "|---|---|---|---|---|---|",
    ]
    chunk_map = {"B0_chunk8192": "8192 (default, 1 chunk)", "B1_chunk512": "512 (4 chunks)",
                 "B2_chunk1024": "1024 (2 chunks)", "B3_disabled": "-1 (disabled)"}
    chunks_map = {"B0_chunk8192": 1, "B1_chunk512": 4, "B2_chunk1024": 2, "B3_disabled": "—"}
    for cname, ttfts in screening.items():
        if not ttfts:
            lines.append(f"| {cname} | — | — | FAILED | — | — |")
            continue
        val = ttfts[0]
        delta_base = f"{val - baseline_ttft:+.1f}" if cname != "B0_chunk8192" else "—"
        delta_vllm = f"{val - PHASE1_VLLM_TTFT:+.1f}"
        lines.append(f"| {cname} | {chunk_map.get(cname,'?')} | {chunks_map.get(cname,'?')} | {val:.1f} | {delta_base} ms | {delta_vllm} ms |")

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
                verdict = "CONFIGURATIONAL — gap closed"
            elif abs(med - baseline_ttft) >= 15:
                verdict = "CHUNKING OVERHEAD — chunked-prefill adds latency"
            else:
                verdict = "STRUCTURAL (same floor) — chunking not responsible for gap"
            lines.append(f"| {cname} | {med:.1f} | {cv:.1f}% | {verdict} |")

    lines += ["\n## Decision", "", "_To be filled after analysis._"]
    path.write_text("\n".join(lines) + "\n")
    log(f"\nSummary → {path}")


def main():
    log("=== Phase 2 Case B — chunked-prefill disentanglement ===")
    log(f"Dataset: {DATASET} (sha={sha256_file(DATASET)[:12]}...)")
    log(f"Note: 2048-tok prompt; chunk sizes 512/1024 force actual chunking; 8192/-1 do not.")

    screening = {}
    for cname, extra in CANDIDATES.items():
        log(f"\n── Candidate {cname} ──")
        ttfts = run_candidate(cname, extra, reps=REPS_SCREENING)
        screening[cname] = ttfts

    baseline = (screening.get("B0_chunk8192") or [PHASE1_SGLANG_TTFT])[0]
    log(f"\n── Screening complete. B0 baseline TTFT: {baseline:.1f} ms ──")
    deltas = {}
    for cname, ttfts in screening.items():
        if ttfts and cname != "B0_chunk8192":
            d = baseline - ttfts[0]
            deltas[cname] = d
            log(f"  {cname}: TTFT={ttfts[0]:.1f} ms  delta={d:+.1f} ms vs B0")

    # Pick finalist: best TTFT; if all within ±5 ms of B0 → structural (same floor), use B0
    all_ttfts = {c: t[0] for c, t in screening.items() if t}
    best_cname = min(all_ttfts, key=lambda c: all_ttfts[c])
    best_ttft  = all_ttfts[best_cname]

    if abs(best_ttft - baseline) < 5:
        finalist = "B0_chunk8192"
        log(f"\n  All candidates within ±5 ms of B0 → STRUCTURAL (same floor as Case A). Finalist = B0.")
    else:
        finalist = best_cname
        log(f"\n  Best candidate {finalist} at {best_ttft:.1f} ms (Δ={baseline-best_ttft:+.1f} ms). Using as finalist.")

    log(f"\n── Finalist reconfirm: {finalist} (3 reps) ──")
    finalist_results = {finalist: run_candidate(finalist, CANDIDATES[finalist], reps=REPS_FINALIST)}

    if finalist_results[finalist]:
        med = statistics.median(finalist_results[finalist])
        cv  = statistics.stdev(finalist_results[finalist]) / med * 100 if len(finalist_results[finalist]) > 1 else 0
        log(f"  Finalist median TTFT: {med:.1f} ms (cv={cv:.1f}%)")

    write_summary(screening, finalist_results)
    log("\n=== Case B sweep complete ===")


if __name__ == "__main__":
    main()
