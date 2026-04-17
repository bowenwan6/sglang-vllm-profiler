#!/usr/bin/env python3
"""
Phase 1 orchestrator: generate datasets, run SGLang + vLLM benchmarks.

Usage (from /data/profiling_lab):
    python3 experiments/phase1/scripts/run_phase1.py

Produces:
    datasets/case{A,B,C,D}.jsonl
    experiments/phase1/raw/{case}_{framework}_rep{1,2,3}.json
    experiments/phase1/raw/{case}_{framework}_rep{1,2,3}_meta.json
    experiments/phase1/raw/dataset_sha256.txt
    logs/phase1/sglang_server.log
    logs/phase1/vllm_server.log
    logs/phase1/sglang_*.log  (L1 kernel-API boundary trails)
"""

import hashlib, json, os, subprocess, sys, time, urllib.request, signal
from datetime import datetime, timezone
from pathlib import Path

# ─── Constants ───────────────────────────────────────────────────────────────
SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
LAB = Path("/data/profiling_lab")
SGLANG_PORT = 30000
VLLM_PORT   = 30001

# Case matrix per plan §9 Phase 1
CASES = {
    "caseA_short":        dict(prompt_len=128,  output_len=128,  concurrency=1,  dataset_n=600,  bench_n=400),
    "caseB_longprefill":  dict(prompt_len=2048, output_len=128,  concurrency=1,  dataset_n=300,  bench_n=200),
    "caseC_batched":      dict(prompt_len=512,  output_len=128,  concurrency=16, dataset_n=2500, bench_n=2000),
    "caseD_decode":       dict(prompt_len=512,  output_len=512,  concurrency=16, dataset_n=1200, bench_n=1000),
}
REPS        = 3
WARMUP      = 30
SEED        = 1

BASE_ENV = {
    **os.environ,
    "CUDA_VISIBLE_DEVICES": "6",
    "HF_HUB_OFFLINE": "1",
}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_for_server(port: int, timeout: int = 360):
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            code = urllib.request.urlopen(url, timeout=3).getcode()
            if code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def kill_server(proc):
    if proc and proc.poll() is None:
        log("Terminating server ...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        log("Server stopped.")


def get_sglang_version() -> dict:
    try:
        r = subprocess.run(
            ["python3", "-c",
             "import sglang; print(sglang.__version__)"],
            capture_output=True, text=True)
        ver = r.stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        r2 = subprocess.run(
            ["python3", "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True)
        torch_ver = r2.stdout.strip()
    except Exception:
        torch_ver = "unknown"
    try:
        r3 = subprocess.run(
            ["python3", "-c", "import flashinfer; print(flashinfer.__version__)"],
            capture_output=True, text=True)
        fi_ver = r3.stdout.strip()
    except Exception:
        fi_ver = "unknown"
    return {"sglang": ver, "torch": torch_ver, "flashinfer": fi_ver}


def get_vllm_version() -> dict:
    venv = "/opt/miniconda3/envs/vllm/bin/python"
    try:
        r = subprocess.run(
            [venv, "-c", "import vllm; print(vllm.__version__)"],
            capture_output=True, text=True)
        ver = r.stdout.strip()
    except Exception:
        ver = "unknown"
    try:
        r2 = subprocess.run(
            [venv, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True)
        torch_ver = r2.stdout.strip()
    except Exception:
        torch_ver = "unknown"
    return {"vllm": ver, "torch": torch_ver}


# ─── Step 1: Dataset generation ──────────────────────────────────────────────

def generate_datasets():
    log("=== Step 1: Generating datasets ===")
    datasets_dir = LAB / "datasets"
    datasets_dir.mkdir(exist_ok=True)
    sha_file = LAB / "experiments/phase1/raw/dataset_sha256.txt"
    sha_lines = []

    for case_name, cfg in CASES.items():
        out_path = datasets_dir / f"{case_name}.jsonl"
        if out_path.exists():
            log(f"  {case_name}: already exists, skipping generation")
            sha = sha256_file(out_path)
            sha_lines.append(f"{sha}  {out_path}")
            continue

        log(f"  Generating {case_name}: prompt={cfg['prompt_len']} out={cfg['output_len']} n={cfg['dataset_n']}")
        cmd = [
            "python3", "-m", "sglang.auto_benchmark", "convert",
            "--kind", "random",
            "--random-input-len", str(cfg["prompt_len"]),
            "--output-len", str(cfg["output_len"]),   # --output-len, not --random-output-len
            "--random-range-ratio", "1.0",             # 1.0 = fixed length (not uniform 1..N)
            "--num-prompts", str(cfg["dataset_n"]),
            "--tokenizer", SNAPSHOT,
            "--seed", str(SEED),
            "--output", str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
        if result.returncode != 0:
            log(f"  ERROR generating {case_name}:\n{result.stderr}")
            sys.exit(1)
        log(f"  {case_name}: generated {out_path}")

        # validate
        val_cmd = [
            "python3", "-m", "sglang.auto_benchmark", "validate",
            "--dataset-path", str(out_path),
            "--tokenizer", SNAPSHOT,
        ]
        val = subprocess.run(val_cmd, capture_output=True, text=True, env=BASE_ENV)
        if val.returncode != 0:
            log(f"  WARN: validate returned non-zero:\n{val.stderr}")
        else:
            log(f"  {case_name}: validated OK")

        sha = sha256_file(out_path)
        sha_lines.append(f"{sha}  {out_path}")

    with open(sha_file, "w") as f:
        f.write(f"# Generated: {datetime.now(timezone.utc).isoformat()}\n")
        f.write("\n".join(sha_lines) + "\n")
    log(f"  SHA-256 logged to {sha_file}")
    return {
        case_name: sha256_file(datasets_dir / f"{case_name}.jsonl")
        for case_name in CASES
    }


# ─── Step 2: Benchmark runner ────────────────────────────────────────────────

def run_bench(case_name: str, cfg: dict, backend: str, port: int,
              rep: int, dataset_sha: str, versions: dict):
    raw_dir = LAB / "experiments/phase1/raw"
    out_json = raw_dir / f"{case_name}_{backend}_rep{rep}.json"
    meta_json = raw_dir / f"{case_name}_{backend}_rep{rep}_meta.json"
    dataset_path = LAB / "datasets" / f"{case_name}.jsonl"

    log(f"  bench: {case_name} {backend} rep{rep} conc={cfg['concurrency']}")

    cmd = [
        "python3", "-m", "sglang.bench_serving",
        "--backend", backend,
        "--base-url", f"http://127.0.0.1:{port}",
        "--dataset-name", "autobench",
        "--dataset-path", str(dataset_path),
        "--max-concurrency", str(cfg["concurrency"]),
        "--num-prompts", str(cfg["bench_n"]),
        "--seed", str(SEED),
        "--warmup-requests", str(WARMUP),
        "--output-file", str(out_json),
    ]

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, env=BASE_ENV)
    elapsed = time.time() - t0

    combined = result.stdout + result.stderr
    if result.returncode != 0:
        log(f"  ERROR: bench_serving failed (rc={result.returncode})")
        log(f"  output: {combined[-800:]}")
        meta = {
            "status": "FAILED", "returncode": result.returncode,
            "output_tail": combined[-800:],
        }
    else:
        log(f"  OK: rep{rep} finished in {elapsed:.1f}s")
        meta = {"status": "OK", "elapsed_s": round(elapsed, 2)}

    meta.update({
        "framework": backend,
        "case": case_name,
        "rep": rep,
        "prompt_len": cfg["prompt_len"],
        "output_len": cfg["output_len"],
        "concurrency": cfg["concurrency"],
        "num_prompts": cfg["bench_n"],
        "warmup_requests": WARMUP,
        "seed": SEED,
        "port": port,
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha,
        "output_file": str(out_json),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_sha": SNAPSHOT.split("/")[-1],
        "versions": versions,
        "cuda_visible_devices": "6",
        "hf_hub_offline": "1",
    })

    with open(meta_json, "w") as f:
        json.dump(meta, f, indent=2)
    log(f"  meta → {meta_json}")
    return meta["status"] == "OK"


# ─── Step 3: SGLang run ───────────────────────────────────────────────────────

def run_sglang_phase(dataset_shas: dict):
    log("\n=== Step 2: SGLang benchmarks ===")
    logs_dir = LAB / "logs/phase1"
    logs_dir.mkdir(parents=True, exist_ok=True)
    server_log = logs_dir / "sglang_server.log"

    versions = get_sglang_version()
    log(f"  SGLang {versions['sglang']}, torch {versions['torch']}, flashinfer {versions['flashinfer']}")

    sglang_env = {
        **BASE_ENV,
        "SGLANG_KERNEL_API_LOGLEVEL": "1",
        "SGLANG_KERNEL_API_LOGDEST": str(logs_dir / "sglang_%i.log"),
    }

    cmd = [
        "python3", "-m", "sglang.launch_server",
        "--model-path", SNAPSHOT,
        "--dtype", "bfloat16",
        "--port", str(SGLANG_PORT),
        "--tp", "1",
        "--attention-backend", "flashinfer",
    ]
    log(f"  Launching SGLang on port {SGLANG_PORT} ...")
    with open(server_log, "w") as logf:
        proc = subprocess.Popen(cmd, env=sglang_env, stdout=logf, stderr=subprocess.STDOUT)

    log("  Waiting for SGLang health ...")
    if not wait_for_server(SGLANG_PORT, timeout=360):
        log("  ERROR: SGLang did not come up within 360s")
        kill_server(proc)
        sys.exit(1)
    log("  SGLang is ready.")

    try:
        for case_name, cfg in CASES.items():
            sha = dataset_shas[case_name]
            for rep in range(1, REPS + 1):
                run_bench(case_name, cfg, "sglang-oai", SGLANG_PORT, rep, sha, versions)
    finally:
        kill_server(proc)


# ─── Step 4: vLLM run ────────────────────────────────────────────────────────

def run_vllm_phase(dataset_shas: dict):
    log("\n=== Step 3: vLLM benchmarks ===")
    logs_dir = LAB / "logs/phase1"
    server_log = logs_dir / "vllm_server.log"

    versions = get_vllm_version()
    log(f"  vLLM {versions['vllm']}, torch {versions['torch']}")

    vllm_python = "/opt/miniconda3/envs/vllm/bin/python"
    cmd = [
        vllm_python, "-m", "vllm.entrypoints.openai.api_server",
        "--model", SNAPSHOT,
        "--dtype", "bfloat16",
        "--port", str(VLLM_PORT),
        "--tensor-parallel-size", "1",
    ]
    log(f"  Launching vLLM on port {VLLM_PORT} ...")
    with open(server_log, "w") as logf:
        proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=logf, stderr=subprocess.STDOUT)

    log("  Waiting for vLLM health ...")
    if not wait_for_server(VLLM_PORT, timeout=360):
        log("  ERROR: vLLM did not come up within 360s")
        kill_server(proc)
        sys.exit(1)
    log("  vLLM is ready.")

    try:
        for case_name, cfg in CASES.items():
            sha = dataset_shas[case_name]
            for rep in range(1, REPS + 1):
                run_bench(case_name, cfg, "vllm", VLLM_PORT, rep, sha, versions)
    finally:
        kill_server(proc)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.chdir(LAB)
    log("Phase 1 orchestrator starting")
    log(f"  CWD: {Path.cwd()}")
    log(f"  CUDA_VISIBLE_DEVICES=6, HF_HUB_OFFLINE=1")

    dataset_shas = generate_datasets()
    run_sglang_phase(dataset_shas)
    run_vllm_phase(dataset_shas)

    log("\n=== Phase 1 complete ===")
    log(f"  Raw results in: {LAB}/experiments/phase1/raw/")
    log("  Next: run summarize_phase1.py to produce summary.md")


if __name__ == "__main__":
    main()
