#!/usr/bin/env python3
"""
run2 Phase 1 orchestrator — serial SGLang then vLLM baseline on GPU 0.

- SGLang: system python3, editable /sgl-workspace/sglang
- vLLM:   /opt/miniconda3/envs/profiling/bin/python
- Greedy: --extra-request-body '{"temperature":0,"top_p":1}'; ignore_eos default True
- Per-request detail via --output-details (enables p95 + error rate downstream)
- Requires datasets/run2_qwen3vl8b/case*.jsonl to already exist (run gen_datasets.py first)

Usage (from repo root):
    python3 experiments/run2_qwen3vl8b/phase1/scripts/run_phase1.py
"""

import hashlib, json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB = Path("/data/sglang-vllm-profiler")
DATASETS_DIR = LAB / "datasets/run2_qwen3vl8b"
RAW = LAB / "experiments/run2_qwen3vl8b/phase1/raw"
LOGS = LAB / "logs/run2_qwen3vl8b/phase1"
VLLM_PYTHON = "/opt/miniconda3/envs/profiling/bin/python"
SGLANG_PORT, VLLM_PORT = 30000, 30001
GPU = "0"

CASES = {
    "caseA_short":       dict(prompt_len=128,  output_len=128, concurrency=1,  bench_n=400),
    "caseB_longprefill": dict(prompt_len=2048, output_len=128, concurrency=1,  bench_n=200),
    "caseC_batched":     dict(prompt_len=512,  output_len=128, concurrency=16, bench_n=2000),
    "caseD_decode":      dict(prompt_len=512,  output_len=512, concurrency=16, bench_n=1000),
}
REPS, WARMUP, SEED = 3, 30, 1
EXTRA_BODY = '{"temperature": 0, "top_p": 1}'  # explicit greedy (plan §6.1)

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_for_server(port, timeout=480):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3).getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def gpu_used_mib():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPU],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def kill_server(proc):
    if proc and proc.poll() is None:
        log("Terminating server ...")
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    # wait for GPU to free
    for _ in range(30):
        used = gpu_used_mib()
        if 0 <= used < 2000:
            log(f"GPU {GPU} freed (used={used} MiB)")
            return True
        time.sleep(3)
    log(f"WARNING: GPU {GPU} still shows used={gpu_used_mib()} MiB after shutdown")
    return False


def ver_sglang():
    def q(code):
        return subprocess.run(["python3", "-c", code], capture_output=True, text=True).stdout.strip()
    return {"sglang": q("import sglang;print(sglang.__version__)"),
            "torch": q("import torch;print(torch.__version__)"),
            "flashinfer": q("import flashinfer;print(flashinfer.__version__)")}


def ver_vllm():
    def q(code):
        return subprocess.run([VLLM_PYTHON, "-c", code], capture_output=True, text=True).stdout.strip()
    return {"vllm": q("import vllm;print(vllm.__version__)"),
            "torch": q("import torch;print(torch.__version__)"),
            "flashinfer": q("import flashinfer;print(flashinfer.__version__)")}


def run_bench(case, cfg, backend, port, rep, dsha, versions):
    out_json = RAW / f"{case}_{backend}_rep{rep}.json"
    meta_json = RAW / f"{case}_{backend}_rep{rep}_meta.json"
    ds = DATASETS_DIR / f"{case}.jsonl"
    log(f"  bench {case} {backend} rep{rep} conc={cfg['concurrency']} n={cfg['bench_n']}")
    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", backend, "--base-url", f"http://127.0.0.1:{port}",
        "--dataset-name", "autobench", "--dataset-path", str(ds),
        "--max-concurrency", str(cfg["concurrency"]), "--num-prompts", str(cfg["bench_n"]),
        "--seed", str(SEED), "--warmup-requests", str(WARMUP),
        "--extra-request-body", EXTRA_BODY, "--output-details",
        "--output-file", str(out_json),
    ]
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = round(time.time() - t0, 2)
    combined = res.stdout + res.stderr
    if res.returncode != 0:
        log(f"  ERROR bench rc={res.returncode}: {combined[-600:]}")
        meta = {"status": "FAILED", "returncode": res.returncode, "output_tail": combined[-600:]}
    else:
        # error-rate sanity from the produced JSON
        nfail = None
        try:
            d = json.load(open(out_json))
            nfail = sum(1 for e in d.get("errors", []) if e)
            log(f"  OK rep{rep} {elapsed}s completed={d.get('completed')} failures={nfail}")
        except Exception as ex:
            log(f"  OK rep{rep} {elapsed}s (could not read failures: {ex})")
        meta = {"status": "OK", "elapsed_s": elapsed, "failures": nfail}
    meta.update({
        "framework": backend, "case": case, "rep": rep,
        "prompt_len": cfg["prompt_len"], "output_len": cfg["output_len"],
        "concurrency": cfg["concurrency"], "num_prompts": cfg["bench_n"],
        "warmup_requests": WARMUP, "seed": SEED, "port": port,
        "extra_request_body": EXTRA_BODY, "ignore_eos": "default(True)",
        "dataset_path": str(ds), "dataset_sha256": dsha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha": SNAPSHOT.split("/")[-1], "versions": versions,
        "cuda_visible_devices": GPU, "hf_hub_offline": "1",
    })
    json.dump(meta, open(meta_json, "w"), indent=2)
    return meta["status"] == "OK"


def phase(backend, port, launch_cmd, env, server_log, versions, dshas):
    log(f"\n=== {backend} phase ===")
    log(f"  versions: {versions}")
    with open(server_log, "w") as lf:
        proc = subprocess.Popen(launch_cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    log(f"  launched, waiting for /health on :{port} ...")
    if not wait_for_server(port):
        log(f"  ERROR: {backend} did not come up"); kill_server(proc); sys.exit(1)
    log(f"  {backend} ready.")
    ok = True
    try:
        for case, cfg in CASES.items():
            for rep in range(1, REPS + 1):
                if not run_bench(case, cfg, backend, port, rep, dshas[case], versions):
                    ok = False
                    log(f"  ABORT: {case} {backend} rep{rep} failed — stopping per stop-condition")
                    return False
    finally:
        kill_server(proc)
    return ok


def main():
    os.chdir(LAB)
    RAW.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    log("run2 Phase 1 orchestrator")
    for case in CASES:
        if not (DATASETS_DIR / f"{case}.jsonl").exists():
            log(f"ERROR: missing dataset {case}.jsonl — run gen_datasets.py first"); sys.exit(1)
    dshas = {c: sha256_file(DATASETS_DIR / f"{c}.jsonl") for c in CASES}

    # SGLang
    sg_env = {**BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
              "SGLANG_KERNEL_API_LOGDEST": str(LOGS / "sglang_%i.log")}
    sg_cmd = ["python3", "-m", "sglang.launch_server", "--model-path", SNAPSHOT,
              "--dtype", "bfloat16", "--port", str(SGLANG_PORT), "--tp", "1",
              "--attention-backend", "flashinfer"]
    if not phase("sglang-oai", SGLANG_PORT, sg_cmd, sg_env, LOGS / "sglang_server.log", ver_sglang(), dshas):
        sys.exit(1)

    # vLLM
    vl_cmd = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server", "--model", SNAPSHOT,
              "--dtype", "bfloat16", "--port", str(VLLM_PORT), "--tensor-parallel-size", "1"]
    if not phase("vllm", VLLM_PORT, vl_cmd, BASE_ENV, LOGS / "vllm_server.log", ver_vllm(), dshas):
        sys.exit(1)

    log("\n=== Phase 1 complete ===")
    log(f"  raw → {RAW}")


if __name__ == "__main__":
    main()
