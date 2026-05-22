#!/usr/bin/env python3
"""
Resume Case A shaping from where interruption left off.
- Loads existing screen results (default/no_overlap/stream8/chunk_off all done)
- Runs chunk_64 screen (1 rep)
- Selects finalists and runs finalist round (3 reps each)
GPU 7.
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
DS = LAB / "datasets/run2_qwen3vl8b/caseA_short.jsonl"
RAW = LAB / "experiments/run2_qwen3vl8b/phase2/raw"
LOGS = LAB / "logs/run2_qwen3vl8b/phase2"
PORT = 30000
GPU = "7"
BENCH_N = 400
WARMUP = 30
SEED = 1
EXTRA_BODY = '{"temperature": 0, "top_p": 1}'

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}

ALL_VARIANTS = {
    "default":    [],
    "no_overlap": ["--disable-overlap-schedule"],
    "stream8":    ["--stream-interval", "8"],
    "chunk_off":  ["--chunked-prefill-size", "-1"],
    "chunk_64":   ["--chunked-prefill-size", "64"],
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
    log(f"WARNING: GPU {GPU} still {gpu_used()} MiB")
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


def parse_json_p50(path):
    try:
        d = json.load(open(path))
        ttfts = [t * 1000 for t in d.get("ttfts", []) if t is not None]
        return percentile(ttfts, 50)
    except Exception:
        return None


def launch_server(variant):
    extra = ALL_VARIANTS[variant]
    cmd = ["python3", "-m", "sglang.launch_server",
           "--model-path", SNAPSHOT, "--dtype", "bfloat16",
           "--port", str(PORT), "--tp", "1", "--attention-backend", "flashinfer"] + extra
    log(f"  launch SGLang {variant}: {' '.join(extra) or '(no extra flags)'}")
    lf = open(LOGS / f"caseA_sglang_{variant}_server.log", "a")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(PORT):
        log(f"  ERROR: server did not come up"); kill_server(proc); sys.exit(1)
    log("  server ready")
    return proc


def run_bench_variant(variant, rep, phase="screen"):
    out = RAW / f"caseA_sglang_{variant}_{phase}_rep{rep}.json"
    meta_out = RAW / f"caseA_sglang_{variant}_{phase}_rep{rep}_meta.json"
    log(f"  bench caseA sglang {variant} {phase} rep{rep}")
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
            p95 = percentile(ttfts_ms, 95)
            p99 = percentile(ttfts_ms, 99)
            log(f"  OK {elapsed}s failures={nfail} p50={ttft_p50:.1f} p95={p95:.1f} p99={p99:.1f}ms")
        except Exception as ex:
            log(f"  OK {elapsed}s (parse: {ex})")
    else:
        log(f"  ERROR rc={res.returncode}: {(res.stdout+res.stderr)[-400:]}")
        nfail = -1
    meta = {
        "status": "OK" if res.returncode == 0 else "FAILED",
        "returncode": res.returncode, "elapsed_s": elapsed, "failures": nfail,
        "ttft_p50_ms": ttft_p50,
        "case": "caseA_short", "variant": variant, "phase": phase, "rep": rep,
        "framework": "sglang-oai", "port": PORT, "bench_n": BENCH_N,
        "warmup": WARMUP, "concurrency": 1,
        "extra_body": EXTRA_BODY,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": GPU,
    }
    json.dump(meta, open(meta_out, "w"), indent=2)
    if nfail and nfail > 0:
        log(f"  STOP: {nfail} failures"); sys.exit(2)
    return ttft_p50


def main():
    os.chdir(LAB)
    log("=== Case A shaping RESUME ===")

    # Load existing screen results
    screen_results = {}
    for v in ALL_VARIANTS:
        p = RAW / f"caseA_sglang_{v}_screen_rep1.json"
        if p.exists():
            p50 = parse_json_p50(p)
            screen_results[v] = p50
            log(f"  loaded {v}: p50={p50:.1f}ms" if p50 else f"  loaded {v}: N/A")

    # Run missing screen variants
    missing = [v for v in ALL_VARIANTS if v not in screen_results]
    log(f"Missing screen variants: {missing}")
    for v in missing:
        proc = launch_server(v)
        try:
            p50 = run_bench_variant(v, 1, "screen")
            screen_results[v] = p50
        finally:
            kill_server(proc)

    log("All screen results:")
    for v, p50 in sorted(screen_results.items(), key=lambda x: x[1] or 9999):
        log(f"  {v}: {p50:.1f}ms" if p50 else f"  {v}: N/A")

    # Select finalists: top 3 by p50; always include default
    ranked = sorted(screen_results.items(), key=lambda x: x[1] or 9999)
    finalists = [v for v, _ in ranked[:3]]
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
    finalists = filtered[:3]
    log(f"Finalists after 5% filter: {finalists}")

    # Finalist round: 3 reps each
    log("--- Finalist round (3 reps each) ---")
    finalist_p50s = {v: [] for v in finalists}
    finalist_details = {v: [] for v in finalists}
    for variant in finalists:
        proc = launch_server(variant)
        try:
            for rep in range(1, 4):
                p50 = run_bench_variant(variant, rep, "finalist")
                finalist_p50s[variant].append(p50)
                # Also grab p95/p99
                f = RAW / f"caseA_sglang_{variant}_finalist_rep{rep}.json"
                try:
                    d = json.load(open(f))
                    ttfts = [t*1000 for t in d.get("ttfts",[]) if t is not None]
                    finalist_details[variant].append({
                        "p50": percentile(ttfts,50), "p95": percentile(ttfts,95),
                        "p99": percentile(ttfts,99)
                    })
                except Exception:
                    pass
        finally:
            kill_server(proc)

    finalist_summary = {}
    for v, vals in finalist_p50s.items():
        vals = [x for x in vals if x is not None]
        med = statistics.median(vals) if vals else None
        cv = (statistics.stdev(vals) / med * 100) if (len(vals) >= 2 and med) else None
        det = finalist_details.get(v, [])
        p95s = [x["p95"] for x in det if x.get("p95")]
        p99s = [x["p99"] for x in det if x.get("p99")]
        finalist_summary[v] = {
            "median_p50_ms": med, "cv": cv,
            "median_p95_ms": statistics.median(p95s) if p95s else None,
            "median_p99_ms": statistics.median(p99s) if p99s else None,
            "vals": vals,
        }
        log(f"  {v}: median_p50={med:.1f}ms cv={cv:.1f}%" if (med and cv is not None) else f"  {v}: {med}")

    # Winner: lowest median p50; prefer default if within 5%
    def_p50f = (finalist_summary.get("default") or {}).get("median_p50_ms") or 9999
    winner = "default"
    for v, s in sorted(finalist_summary.items(), key=lambda x: x[1].get("median_p50_ms") or 9999):
        vp = s.get("median_p50_ms") or 9999
        vc = s.get("cv")
        if v != "default" and vp < def_p50f * 0.95:
            winner = v
            break

    log(f"Case A WINNER: {winner} (flags: {' '.join(ALL_VARIANTS[winner]) or 'none'})")
    log(f"  p50={finalist_summary[winner].get('median_p50_ms'):.1f}ms cv={finalist_summary[winner].get('cv'):.1f}%")

    result = {
        "case": "caseA_short", "winner": winner,
        "winner_flags": ALL_VARIANTS[winner],
        "screen_results": screen_results,
        "finalist_summary": {k: {
            "median_p50_ms": v["median_p50_ms"], "cv": v["cv"],
            "median_p95_ms": v.get("median_p95_ms"), "median_p99_ms": v.get("median_p99_ms")
        } for k, v in finalist_summary.items()},
        "phase1_vllm_ttft_p50_ms": 12.6,
    }
    json.dump(result, open(RAW / "caseA_shaping_result.json", "w"), indent=2)
    log("Saved caseA_shaping_result.json")
    log("=== Case A complete ===")


if __name__ == "__main__":
    main()
