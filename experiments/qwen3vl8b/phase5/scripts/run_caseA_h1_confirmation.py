#!/usr/bin/env python3
"""
Phase 5.2 — Case A H1 CLEAN (uninstrumented) confirmation (GPU 3).

Removes ALL instrumentation (no KAPI logging, no profiler) to isolate the true effect of forcing
SGLang prefill piecewise CUDA graph. Bracket S0_before -> S2 -> S0_after checks baseline drift;
V0_clean_anchor (vLLM) is the clean reference, run last. No SGLang source changes.

STRICT: never set SGLANG_KERNEL_API_LOGLEVEL / SGLANG_KERNEL_API_LOGDEST; no profiler.
"""
import hashlib, json, os, statistics, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
LAB = Path("/data/sglang-vllm-profiler")
DS = LAB / "datasets/qwen3vl8b/caseA_short.jsonl"
BASE = LAB / "experiments/qwen3vl8b/phase5/caseA_h1_confirmation"
RAW = BASE / "raw"
LOGS = LAB / "logs/qwen3vl8b/phase5/caseA_h1_confirmation"
VLLM_PYTHON = "/opt/miniconda3/envs/profiling/bin/python"
GPU = "3"
SGLANG_PORT, VLLM_PORT = 30000, 30001
CONC, BENCH_N, WARMUP, SEED, REPS = 1, 400, 30, 1, 3
EXTRA_BODY = '{"temperature": 0, "top_p": 1}'
# Clean env: NO KAPI vars at all.
BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}
for _k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
    BASE_ENV.pop(_k, None)

VARIANTS = [
    ("S0_before",            "sglang", ["--disable-overlap-schedule"], 480),
    ("S2_enforce_piecewise", "sglang", ["--disable-overlap-schedule", "--enforce-piecewise-cuda-graph"], 900),
    ("S0_after",             "sglang", ["--disable-overlap-schedule"], 480),
    ("V0_clean_anchor",      "vllm",   [], 600),
]


def log(m): print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k); hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def gpu_used():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
                        "-i", GPU], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_server(port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def kill_server(proc, patt):
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    subprocess.run(["pkill", "-9", "-f", patt], capture_output=True)
    for _ in range(40):
        u = gpu_used()
        if 0 <= u < 2000:
            log(f"  GPU {GPU} freed (used={u} MiB)")
            return True
        time.sleep(3)
    log(f"  WARNING: GPU {GPU} still {gpu_used()} MiB")
    return False


def smoke_check(framework, port):
    try:
        if framework == "sglang":
            payload = json.dumps({"text": "The capital of France is",
                                  "sampling_params": {"temperature": 0, "max_new_tokens": 8}}).encode()
            url = f"http://127.0.0.1:{port}/generate"
        else:
            payload = json.dumps({"model": SNAPSHOT, "prompt": "The capital of France is",
                                  "max_tokens": 8, "temperature": 0}).encode()
            url = f"http://127.0.0.1:{port}/v1/completions"
        req = urllib.request.Request(url, data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        txt = resp.get("text", "") if framework == "sglang" else resp["choices"][0].get("text", "")
        return (bool(txt and txt.strip()), txt[:60])
    except Exception as e:
        return (False, f"{type(e).__name__}: {e}")


def ver(framework):
    py = "python3" if framework == "sglang" else VLLM_PYTHON
    pkg = "sglang" if framework == "sglang" else "vllm"
    return subprocess.run([py, "-c", f"import {pkg};print({pkg}.__version__)"],
                          capture_output=True, text=True).stdout.strip()


def run_bench(vid, framework, port, rep, dsha, version):
    out_json = RAW / f"{vid}_rep{rep}.json"
    meta_json = RAW / f"{vid}_rep{rep}_meta.json"
    backend = "sglang-oai" if framework == "sglang" else "vllm"
    cmd = ["python3", "-m", "sglang.bench_serving", "--backend", backend,
           "--base-url", f"http://127.0.0.1:{port}", "--dataset-name", "autobench",
           "--dataset-path", str(DS), "--max-concurrency", str(CONC), "--num-prompts", str(BENCH_N),
           "--seed", str(SEED), "--warmup-requests", str(WARMUP), "--extra-request-body", EXTRA_BODY,
           "--output-details", "--output-file", str(out_json)]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = round(time.time() - t0, 1)
    status, nfail, metrics = "FAILED", None, {}
    if res.returncode == 0:
        try:
            d = json.load(open(out_json))
            nfail = sum(1 for e in d.get("errors", []) if e)
            ttfts = [t * 1000 for t in d.get("ttfts", []) if t is not None]
            metrics = {"completed": d.get("completed"), "failures": nfail,
                       "ttft_p50": percentile(ttfts, 50) if ttfts else d.get("median_ttft_ms"),
                       "ttft_p95": percentile(ttfts, 95) if ttfts else None,
                       "ttft_p99": percentile(ttfts, 99) if ttfts else d.get("p99_ttft_ms"),
                       "tpot_p50": d.get("median_tpot_ms"), "tpot_p99": d.get("p99_tpot_ms"),
                       "out_tok_s": d.get("output_throughput"), "req_s": d.get("request_throughput")}
            status = "OK"
            log(f"    rep{rep} {elapsed}s completed={metrics['completed']} fail={nfail} "
                f"ttft_p50={metrics['ttft_p50']}ms")
        except Exception as ex:
            log(f"    rep{rep} parse error: {ex}")
    else:
        log(f"    rep{rep} bench rc={res.returncode}: {(res.stdout+res.stderr)[-300:]}")
    meta = {"variant": vid, "framework": framework, "rep": rep, "status": status,
            "elapsed_s": elapsed, "failures": nfail, "gpu": GPU, "snapshot_sha": SNAPSHOT.split("/")[-1],
            "dataset_sha256": dsha, "version": version, "warmup": WARMUP, "reps": REPS,
            "num_prompts": BENCH_N, "concurrency": CONC, "extra_request_body": EXTRA_BODY,
            "kapi_logging": False, "profiler": False,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(), "metrics": metrics}
    json.dump(meta, open(meta_json, "w"), indent=2)
    return status == "OK", metrics


def run_variant(vid, framework, flags, wait_s, dsha):
    log(f"\n===== {vid} ({framework}) flags={flags or 'default'} [CLEAN: no KAPI/profiler] =====")
    u = gpu_used()
    if not (0 <= u < 2000):
        log(f"  STOP: GPU {GPU} not idle (used={u} MiB)"); return {"id": vid, "status": "GPU_NOT_IDLE"}
    if framework == "sglang":
        cmd = ["python3", "-m", "sglang.launch_server", "--model-path", SNAPSHOT, "--dtype", "bfloat16",
               "--port", str(SGLANG_PORT), "--tp", "1", "--attention-backend", "flashinfer"] + flags
        port, patt = SGLANG_PORT, "sglang.launch_server"
    else:
        cmd = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", SNAPSHOT,
               "--dtype", "bfloat16", "--port", str(VLLM_PORT), "--tensor-parallel-size", "1"] + flags
        port, patt = VLLM_PORT, "vllm.entrypoints.openai.api_server"
    version = ver(framework)
    lf = open(LOGS / f"{vid}_server.log", "w")
    log(f"  launching (wait≤{wait_s}s, version={version}) ...")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)
    rec = {"id": vid, "framework": framework, "flags": flags, "version": version,
           "kapi_logging": False, "profiler": False, "reps": []}
    try:
        if not wait_server(port, wait_s):
            log(f"  ERROR: {vid} did not come up within {wait_s}s"); rec["status"] = "SERVER_NO_START"
            return rec
        ok, detail = smoke_check(framework, port)
        rec["smoke_ok"], rec["smoke_detail"] = ok, detail
        log(f"  smoke: ok={ok} detail={detail!r}")
        if not ok and vid.startswith("S2"):
            log(f"  {vid} smoke FAILED -> invalid"); rec["status"] = "SMOKE_FAILED"
            return rec
        any_fail = False
        for rep in range(1, REPS + 1):
            okb, metrics = run_bench(vid, framework, port, rep, dsha, version)
            rec["reps"].append(metrics if okb else {"status": "FAILED"})
            if not okb or (metrics.get("failures") or 0) > 0:
                any_fail = True
                log(f"  {vid} rep{rep}: failed/failures>0 -> stopping reps"); break
        rec["status"] = "INVALID_FAILURES" if any_fail else "OK"
    finally:
        kill_server(proc, patt)
    p50s = [r.get("ttft_p50") for r in rec["reps"] if r.get("ttft_p50") is not None]
    if p50s:
        rec["ttft_p50_reps"] = p50s
        rec["ttft_p50_median"] = round(statistics.median(p50s), 3)
        rec["ttft_p50_cv_pct"] = (round(100 * statistics.pstdev(p50s) / statistics.mean(p50s), 1)
                                  if len(p50s) > 1 and statistics.mean(p50s) else 0.0)
        for key in ("ttft_p95", "ttft_p99", "tpot_p50", "tpot_p99", "out_tok_s"):
            vals = [r[key] for r in rec["reps"] if r.get(key) is not None]
            rec[f"{key}_median"] = round(statistics.median(vals), 3) if vals else None
    log(f"  {vid} status={rec['status']} ttft_p50_median={rec.get('ttft_p50_median')} "
        f"cv={rec.get('ttft_p50_cv_pct')}%")
    return rec


def main():
    os.chdir(LAB)
    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    dsha = sha256_file(DS)
    log(f"=== Case A H1 CLEAN confirmation (GPU {GPU}); dataset {dsha[:16]}; NO KAPI / NO profiler ===")
    results = []
    for vid, fw, flags, wait_s in VARIANTS:
        rec = run_variant(vid, fw, flags, wait_s, dsha)
        results.append(rec)
        json.dump(results, open(BASE / "results.json", "w"), indent=2)
        if rec.get("status") == "GPU_NOT_IDLE":
            log("STOP: GPU not idle."); break
    json.dump(results, open(BASE / "results.json", "w"), indent=2)
    log("\n=== clean confirmation done; results.json written ===")


if __name__ == "__main__":
    main()
