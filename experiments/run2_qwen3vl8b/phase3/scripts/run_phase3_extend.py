#!/usr/bin/env python3
"""
run2 Phase 3 — SGLang EXTEND/PREFILL-stage trace supplement (GPU 1, SGLang only).

Why: the original Phase 3 collection armed the profiler during steady decode, so
--profile-by-stage emitted DECODE-only traces. Here we drive a continuous
prefill-only load (/generate with max_new_tokens=1) so the profile window lands on
EXTEND (prefill) forward steps. New traces go to sglang_extend_{mapping,formal};
existing DECODE traces are NOT touched.

No vLLM, no Phase 4 analysis.
"""

import json, os, subprocess, sys, threading, time, urllib.request
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
SGLANG_PORT = 30000
GPU = "1"
NUM_STEPS = 10
PROFILE_TIMEOUT = 300

BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}

# A, B, C, D priority order.
CASES = {
    "caseA_short":       dict(ds="caseA_short.jsonl",       conc=1,  sglang_flags=["--disable-overlap-schedule"]),
    "caseB_longprefill": dict(ds="caseB_longprefill.jsonl", conc=1,  sglang_flags=[]),
    "caseC_batched":     dict(ds="caseC_batched.jsonl",     conc=16, sglang_flags=[]),
    "caseD_decode":      dict(ds="caseD_decode.jsonl",      conc=16, sglang_flags=[]),
}
ORDER = ["caseA_short", "caseB_longprefill", "caseC_batched", "caseD_decode"]


def log(msg):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def wait_server(timeout=480):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{SGLANG_PORT}/health", timeout=3).getcode() == 200:
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
        proc.terminate()
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            proc.kill(); proc.wait()
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    for _ in range(40):
        u = gpu_used()
        if 0 <= u < 2000:
            log(f"  GPU {GPU} freed (used={u} MiB)")
            return True
        time.sleep(3)
    log(f"  WARNING: GPU {GPU} still {gpu_used()} MiB")
    return False


def read_prompts(case, n=64):
    out = []
    with open(DSDIR / CASES[case]["ds"]) as f:
        for line in f:
            out.append(json.loads(line)["prompt"])
            if len(out) >= n:
                break
    return out


def stage_files(d):
    p = Path(d)
    files = [f for f in p.rglob("*.trace.json.gz")]
    stages = set()
    for f in files:
        for s in ("EXTEND", "PREFILL", "DECODE"):
            if s in f.name:
                stages.add(s)
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return files, stages, total


class PrefillLoad:
    """Continuous prefill-only load: /generate with max_new_tokens=1 at given concurrency."""
    def __init__(self, prompts, conc):
        self.prompts = prompts
        self.conc = conc
        self.stop = threading.Event()
        self.threads = []
        self.sent = 0
        self.lock = threading.Lock()

    def _worker(self, wid):
        i = wid
        while not self.stop.is_set():
            p = self.prompts[i % len(self.prompts)]
            i += self.conc
            payload = json.dumps({"text": p,
                                  "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                                  "stream": False}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{SGLANG_PORT}/generate",
                                         data=payload, headers={"Content-Type": "application/json"},
                                         method="POST")
            try:
                urllib.request.urlopen(req, timeout=60).read()
                with self.lock:
                    self.sent += 1
            except Exception:
                pass

    def start(self):
        for w in range(self.conc):
            t = threading.Thread(target=self._worker, args=(w,), daemon=True)
            t.start()
            self.threads.append(t)

    def stop_all(self):
        self.stop.set()
        for t in self.threads:
            t.join(timeout=5)


def collect_extend(case, cfg, graph_off):
    tag = "mapping" if graph_off else "formal"
    trace_dir = TRACES / case / f"sglang_extend_{tag}"
    trace_dir.mkdir(parents=True, exist_ok=True)
    flags = list(cfg["sglang_flags"])
    if graph_off:
        flags += ["--disable-cuda-graph", "--disable-piecewise-cuda-graph"]
    logpath = LOGS / f"{case}_sglang_extend_{tag}_server.log"
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", SNAPSHOT,
           "--dtype", "bfloat16", "--port", str(SGLANG_PORT), "--tp", "1",
           "--attention-backend", "flashinfer"] + flags
    env = {**BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
           "SGLANG_KERNEL_API_LOGDEST": str(LOGS / f"{case}_sglang_extend_{tag}_kapi_%i.log"),
           "SGLANG_TORCH_PROFILER_DIR": str(trace_dir)}
    log(f"  launch SGLang {case} extend_{tag}: {' '.join(flags) or '(default)'}")
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server():
        log(f"  ERROR: server did not come up")
        kill_server(proc)
        return {"ok": False, "reason": "server_did_not_start", "trace_dir": str(trace_dir)}

    loader = PrefillLoad(read_prompts(case, 64), cfg["conc"])
    try:
        log(f"  start continuous prefill load (max_new_tokens=1, c={cfg['conc']})")
        loader.start()
        time.sleep(6)  # ensure server is steadily prefilling
        log(f"  run profiler --profile-by-stage --num-steps {NUM_STEPS}")
        pcmd = ["python3", "-m", "sglang.profiler", "--url", f"http://127.0.0.1:{SGLANG_PORT}",
                "--num-steps", str(NUM_STEPS), "--profile-by-stage", "--output-dir", str(trace_dir)]
        pr = subprocess.run(pcmd, env=BASE_ENV, capture_output=True, text=True, timeout=PROFILE_TIMEOUT)
        if pr.returncode != 0:
            log(f"  ERROR profiler rc={pr.returncode}: {(pr.stdout+pr.stderr)[-400:]}")
            return {"ok": False, "reason": "profiler_failed", "trace_dir": str(trace_dir)}
        time.sleep(8)
    except subprocess.TimeoutExpired:
        log("  ERROR profiler timeout")
        return {"ok": False, "reason": "profiler_timeout", "trace_dir": str(trace_dir)}
    finally:
        loader.stop_all()
        log(f"  prefill load sent ~{loader.sent} requests")
        kill_server(proc)

    files, stages, total = stage_files(trace_dir)
    has_extend = bool(stages & {"EXTEND", "PREFILL"})
    log(f"  extend_{tag}: stages={sorted(stages)} files={len(files)} size={total/1e6:.1f}MB extend={has_extend}")
    return {"ok": has_extend and total > 0, "stages": sorted(stages),
            "has_extend": has_extend, "size_bytes": total, "trace_dir": str(trace_dir),
            "reason": None if has_extend else "no_extend_stage_captured"}


def run_case(case):
    cfg = CASES[case]
    log(f"\n===== {case} EXTEND supplement (c={cfg['conc']}, flags={cfg['sglang_flags'] or 'default'}) =====")
    res = {}
    res["sglang_extend_mapping"] = collect_extend(case, cfg, True)
    res["sglang_extend_formal"] = collect_extend(case, cfg, False)
    # merge into existing case metadata
    meta_path = META / f"{case}_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {"case": case}
    meta.setdefault("extend_supplement", {})
    meta["extend_supplement"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU, "num_steps": NUM_STEPS, "load": "prefill-only /generate max_new_tokens=1",
        "traces": {k: {"ok": v["ok"], "stages": v.get("stages"), "has_extend": v.get("has_extend"),
                       "size_bytes": v.get("size_bytes"), "trace_dir": v.get("trace_dir"),
                       "reason": v.get("reason")} for k, v in res.items()},
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    ok = all(v["ok"] for v in res.values())
    log(f"  {case} EXTEND ok={ok}")
    return ok, res


def main():
    os.chdir(LAB)
    LOGS.mkdir(parents=True, exist_ok=True)
    log("=== run2 Phase 3 SGLang EXTEND supplement (GPU 1) ===")
    u = gpu_used()
    if not (0 <= u < 2000):
        log(f"STOP: GPU {GPU} not idle (used={u} MiB)"); sys.exit(1)

    summary = {}
    for case in ORDER:
        ok, res = run_case(case)
        summary[case] = {k: {"ok": v["ok"], "stages": v.get("stages"),
                             "size_bytes": v.get("size_bytes"), "trace_dir": v.get("trace_dir"),
                             "reason": v.get("reason")} for k, v in res.items()}
        if not ok:
            log(f"  WARNING: {case} EXTEND not fully captured — continuing other cases")
    json.dump(summary, open(META / "phase3_extend_summary.json", "w"), indent=2)
    log("\n=== EXTEND supplement complete ===")


if __name__ == "__main__":
    main()
