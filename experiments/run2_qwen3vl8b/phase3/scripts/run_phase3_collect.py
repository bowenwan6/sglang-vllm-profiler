#!/usr/bin/env python3
"""
run2 Phase 3 — trace collection (NO interpretation).

Per case, collects 4 trace groups:
  1. SGLang mapping  (graph-off, --profile-by-stage)
  2. SGLang formal   (graph-on,  --profile-by-stage)
  3. vLLM prefill_like  (c=1, max_tokens=1 window)
  4. vLLM decode_like   (steady-state decode window)

Case A runs first as a PILOT. If all Case A checks pass, C/B/D run automatically.
If any Case A check fails, STOP (no C/B/D). GPU 1 only, servers strictly serial.
"""

import hashlib, json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = (
    "/root/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-VL-8B-Instruct/snapshots/"
    "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
)
SNAP_SHA = SNAPSHOT.split("/")[-1]
LAB = Path("/data/sglang-vllm-profiler")
DSDIR = LAB / "datasets/run2_qwen3vl8b"
TRACES = LAB / "traces/run2_qwen3vl8b"
LOGS = LAB / "logs/run2_qwen3vl8b/phase3"
META = LAB / "experiments/run2_qwen3vl8b/phase3/metadata"
VLLM_PY = "/opt/miniconda3/envs/profiling/bin/python"
SGLANG_PORT, VLLM_PORT = 30000, 30001
GPU = "1"
NUM_STEPS = 10
PROFILE_TIMEOUT = 300

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}

# Profiling order: A pilot, then C, B, D.
CASES = {
    "caseA_short":       dict(ds="caseA_short.jsonl",       conc=1,  out_len=128, warmup=30,  reps=3, sglang_flags=["--disable-overlap-schedule"], load_n=200,  ceiling_m=False),
    "caseC_batched":     dict(ds="caseC_batched.jsonl",     conc=16, out_len=128, warmup=500, reps=5, sglang_flags=[],                              load_n=1500, ceiling_m=False),
    "caseB_longprefill": dict(ds="caseB_longprefill.jsonl", conc=1,  out_len=128, warmup=300, reps=5, sglang_flags=[],                              load_n=200,  ceiling_m=True),
    "caseD_decode":      dict(ds="caseD_decode.jsonl",      conc=16, out_len=512, warmup=30,  reps=3, sglang_flags=[],                              load_n=1200, ceiling_m=False),
}
ORDER = ["caseA_short", "caseC_batched", "caseB_longprefill", "caseD_decode"]


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_server(port, path="/health", timeout=480):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3).getcode() == 200:
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
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    subprocess.run(["pkill", "-9", "-f", pattern], capture_output=True)
    for _ in range(40):
        u = gpu_used()
        if 0 <= u < 2000:
            log(f"  GPU {GPU} freed (used={u} MiB)")
            return True
        time.sleep(3)
    log(f"  WARNING: GPU {GPU} still {gpu_used()} MiB after shutdown")
    return False


def dir_nonempty(d):
    p = Path(d)
    if not p.is_dir():
        return False, 0
    files = [f for f in p.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    return (len(files) > 0 and total > 0), total


def fatal_in_log(logpath):
    try:
        txt = Path(logpath).read_text(errors="ignore").lower()
        for kw in ["traceback (most recent call last)", "cuda error", "out of memory",
                   "runtimeerror", "illegal memory access"]:
            if kw in txt:
                return kw
    except Exception:
        pass
    return None


# ---------- SGLang ----------
def launch_sglang(case, cfg, graph_off, trace_dir):
    flags = list(cfg["sglang_flags"])
    if graph_off:
        flags += ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
    tag = "mapping" if graph_off else "formal"
    logpath = LOGS / f"{case}_sglang_{tag}_server.log"
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", SNAPSHOT,
           "--dtype", "bfloat16", "--port", str(SGLANG_PORT), "--tp", "1",
           "--attention-backend", "flashinfer"] + flags
    env = {**BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
           "SGLANG_KERNEL_API_LOGDEST": str(LOGS / f"{case}_sglang_{tag}_kapi_%i.log"),
           "SGLANG_TORCH_PROFILER_DIR": str(trace_dir)}
    log(f"  launch SGLang {case} {tag}: {' '.join(flags) or '(default)'}")
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(SGLANG_PORT):
        log(f"  ERROR: SGLang {tag} did not come up")
        kill_server(proc, "sglang.launch_server")
        return None, logpath
    log(f"  SGLang {tag} ready")
    return proc, logpath


def bench_load_bg(case, cfg, backend, port):
    ds = DSDIR / cfg["ds"]
    cmd = ["python3", "-m", "sglang.bench_serving", "--backend", backend,
           "--base-url", f"http://127.0.0.1:{port}", "--dataset-name", "autobench",
           "--dataset-path", str(ds), "--max-concurrency", str(cfg["conc"]),
           "--num-prompts", str(cfg["load_n"]), "--seed", "1",
           "--warmup-requests", str(cfg["warmup"]),
           "--extra-request-body", '{"temperature": 0, "top_p": 1}']
    return subprocess.Popen(cmd, env=BASE_ENV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def collect_sglang(case, cfg, graph_off, trace_dir):
    tag = "mapping" if graph_off else "formal"
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    proc, logpath = launch_sglang(case, cfg, graph_off, trace_dir)
    if proc is None:
        return {"ok": False, "reason": "server_did_not_start", "log": str(logpath)}
    load = None
    try:
        log(f"  start background load (c={cfg['conc']}, n={cfg['load_n']})")
        load = bench_load_bg(case, cfg, "sglang-oai", SGLANG_PORT)
        time.sleep(12)  # reach steady state past warmup
        log(f"  run profiler --profile-by-stage --num-steps {NUM_STEPS}")
        pcmd = ["python3", "-m", "sglang.profiler", "--url", f"http://127.0.0.1:{SGLANG_PORT}",
                "--num-steps", str(NUM_STEPS), "--profile-by-stage",
                "--output-dir", str(trace_dir)]
        pr = subprocess.run(pcmd, env=BASE_ENV, capture_output=True, text=True,
                            timeout=PROFILE_TIMEOUT)
        if pr.returncode != 0:
            log(f"  ERROR profiler rc={pr.returncode}: {(pr.stdout+pr.stderr)[-400:]}")
            return {"ok": False, "reason": "profiler_failed", "log": str(logpath)}
        time.sleep(8)  # flush
    except subprocess.TimeoutExpired:
        log("  ERROR profiler timed out")
        return {"ok": False, "reason": "profiler_timeout", "log": str(logpath)}
    finally:
        if load and load.poll() is None:
            load.terminate()
            try:
                load.wait(timeout=20)
            except subprocess.TimeoutExpired:
                load.kill()
        kill_server(proc, "sglang.launch_server")
    fatal = fatal_in_log(logpath)
    ne, sz = dir_nonempty(trace_dir)
    log(f"  SGLang {tag}: nonempty={ne} size={sz/1e6:.1f}MB fatal={fatal}")
    return {"ok": ne and not fatal, "reason": fatal or ("empty_trace" if not ne else None),
            "size_bytes": sz, "trace_dir": str(trace_dir), "log": str(logpath)}


# ---------- vLLM ----------
def launch_vllm(case, window, trace_dir):
    Path(trace_dir).mkdir(parents=True, exist_ok=True)
    logpath = LOGS / f"{case}_vllm_{window}_server.log"
    pcfg = json.dumps({"profiler": "torch", "torch_profiler_dir": str(Path(trace_dir).resolve())})
    cmd = [VLLM_PY, "-m", "vllm.entrypoints.openai.api_server", "--model", SNAPSHOT,
           "--dtype", "bfloat16", "--port", str(VLLM_PORT), "--tensor-parallel-size", "1",
           "--profiler-config", pcfg]
    log(f"  launch vLLM {case} {window} (profiler dir set)")
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, env=BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(VLLM_PORT):
        log(f"  ERROR: vLLM {window} did not come up")
        kill_server(proc, "vllm.entrypoints")
        return None, logpath
    log(f"  vLLM {window} ready")
    return proc, logpath


def vllm_model_id():
    try:
        d = json.load(urllib.request.urlopen(f"http://127.0.0.1:{VLLM_PORT}/v1/models", timeout=10))
        return d["data"][0]["id"]
    except Exception:
        return SNAPSHOT


def read_prompts(case, n):
    out = []
    with open(DSDIR / CASES[case]["ds"]) as f:
        for line in f:
            out.append(json.loads(line)["prompt"])
            if len(out) >= n:
                break
    return out


def http_post(path, payload, timeout=120):
    req = urllib.request.Request(f"http://127.0.0.1:{VLLM_PORT}{path}",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    return urllib.request.urlopen(req, timeout=timeout)


def collect_vllm_prefill(case, cfg, trace_dir):
    proc, logpath = launch_vllm(case, "prefill_like", trace_dir)
    if proc is None:
        return {"ok": False, "reason": "server_did_not_start", "log": str(logpath)}
    try:
        model = vllm_model_id()
        prompts = read_prompts(case, 8)
        http_post("/start_profile", {}, timeout=30)
        log("  prefill_like: profiler started, sending 8×(max_tokens=1) at c=1")
        for p in prompts:
            http_post("/v1/completions", {"model": model, "prompt": p,
                                          "max_tokens": 1, "temperature": 0}, timeout=120)
        http_post("/stop_profile", {}, timeout=120)
        log("  prefill_like: profiler stopped")
        time.sleep(10)
    except Exception as ex:
        log(f"  ERROR vLLM prefill_like: {ex}")
        kill_server(proc, "vllm.entrypoints")
        return {"ok": False, "reason": f"exception:{ex}", "log": str(logpath)}
    finally:
        kill_server(proc, "vllm.entrypoints")
    fatal = fatal_in_log(logpath)
    ne, sz = dir_nonempty(trace_dir)
    log(f"  vLLM prefill_like: nonempty={ne} size={sz/1e6:.1f}MB fatal={fatal}")
    return {"ok": ne and not fatal, "reason": fatal or ("empty_trace" if not ne else None),
            "size_bytes": sz, "trace_dir": str(trace_dir), "log": str(logpath)}


def collect_vllm_decode(case, cfg, trace_dir):
    proc, logpath = launch_vllm(case, "decode_like", trace_dir)
    if proc is None:
        return {"ok": False, "reason": "server_did_not_start", "log": str(logpath)}
    load = None
    try:
        log(f"  decode_like: start steady load (c={cfg['conc']}), warm to steady state")
        load = bench_load_bg(case, cfg, "vllm", VLLM_PORT)
        time.sleep(20)  # reach steady-state decode
        http_post("/start_profile", {}, timeout=30)
        log("  decode_like: profiler started, capturing ~6s steady decode")
        time.sleep(6)
        http_post("/stop_profile", {}, timeout=120)
        log("  decode_like: profiler stopped")
        time.sleep(10)
    except Exception as ex:
        log(f"  ERROR vLLM decode_like: {ex}")
        return {"ok": False, "reason": f"exception:{ex}", "log": str(logpath)}
    finally:
        if load and load.poll() is None:
            load.terminate()
            try:
                load.wait(timeout=20)
            except subprocess.TimeoutExpired:
                load.kill()
        kill_server(proc, "vllm.entrypoints")
    fatal = fatal_in_log(logpath)
    ne, sz = dir_nonempty(trace_dir)
    log(f"  vLLM decode_like: nonempty={ne} size={sz/1e6:.1f}MB fatal={fatal}")
    return {"ok": ne and not fatal, "reason": fatal or ("empty_trace" if not ne else None),
            "size_bytes": sz, "trace_dir": str(trace_dir), "log": str(logpath)}


# ---------- per-case ----------
def run_case(case):
    cfg = CASES[case]
    log(f"\n===== {case} (c={cfg['conc']}, out={cfg['out_len']}, "
        f"flags={cfg['sglang_flags'] or 'default'}, warmup={cfg['warmup']}, reps={cfg['reps']}) =====")
    croot = TRACES / case
    dsha = sha256_file(DSDIR / cfg["ds"])
    results = {}
    results["sglang_mapping"] = collect_sglang(case, cfg, True,  croot / "sglang_mapping")
    results["sglang_formal"]  = collect_sglang(case, cfg, False, croot / "sglang_formal")
    results["vllm_prefill_like"] = collect_vllm_prefill(case, cfg, croot / "vllm" / "prefill_like")
    results["vllm_decode_like"]  = collect_vllm_decode(case, cfg, croot / "vllm" / "decode_like")

    meta = {
        "case": case, "model_snapshot": SNAP_SHA, "dataset": cfg["ds"], "dataset_sha256": dsha,
        "gpu": GPU, "concurrency": cfg["conc"], "output_len": cfg["out_len"],
        "sglang_flags": cfg["sglang_flags"], "warmup": cfg["warmup"], "reps": cfg["reps"],
        "num_steps": NUM_STEPS, "confidence_ceiling_M": cfg["ceiling_m"],
        "note": ("Case B: both frameworks bimodal -> all cross-framework conclusions carry ceiling M"
                 if cfg["ceiling_m"] else ""),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "traces": {k: {"ok": v["ok"], "reason": v.get("reason"),
                       "size_bytes": v.get("size_bytes"), "trace_dir": v.get("trace_dir")}
                   for k, v in results.items()},
    }
    META.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(META / f"{case}_meta.json", "w"), indent=2)
    all_ok = all(v["ok"] for v in results.values())
    log(f"  {case} ALL_OK={all_ok}")
    for k, v in results.items():
        log(f"    {k}: ok={v['ok']} size={(v.get('size_bytes') or 0)/1e6:.1f}MB reason={v.get('reason')}")
    return all_ok, meta


def main():
    os.chdir(LAB)
    LOGS.mkdir(parents=True, exist_ok=True)
    log("=== run2 Phase 3 trace collection (GPU 1) ===")
    u = gpu_used()
    if not (0 <= u < 2000):
        log(f"STOP: GPU {GPU} not idle (used={u} MiB)"); sys.exit(1)

    summary = {}

    # --- Case A PILOT ---
    log("\n########## CASE A PILOT ##########")
    a_ok, a_meta = run_case("caseA_short")
    summary["caseA_short"] = a_meta
    if not a_ok:
        log("\n!!! Case A pilot gate FAILED — stopping, will NOT run C/B/D !!!")
        json.dump(summary, open(META / "phase3_run_summary.json", "w"), indent=2)
        sys.exit(2)
    log("\n>>> Case A pilot gate PASSED — continuing C, B, D automatically <<<")

    for case in ["caseC_batched", "caseB_longprefill", "caseD_decode"]:
        ok, meta = run_case(case)
        summary[case] = meta
        if not ok:
            log(f"  WARNING: {case} had failed trace group(s) — continuing remaining cases, "
                f"recording failure in metadata")

    json.dump(summary, open(META / "phase3_run_summary.json", "w"), indent=2)
    log("\n=== Phase 3 collection complete ===")


if __name__ == "__main__":
    main()
