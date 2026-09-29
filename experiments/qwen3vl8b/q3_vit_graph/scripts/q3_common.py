#!/usr/bin/env python3
"""Q3 (ViT CUDA graph) — shared helpers, node side.

Every path comes from the environment so the scripts run unchanged on any RADIX
node and never need editing:

  SGL_ROOT        node workspace (default ~/sgl)
  HF_HOME         model cache (default ~/hf)
  Q3_MODEL_REPO   Qwen/Qwen3-VL-8B-Instruct
  Q3_MODEL_REV    0c351dd01ed87e9c1b53cbc748cba10e6187ff3b   (the v2/v3 protocol pin)
  Q3_GPU          CUDA_VISIBLE_DEVICES for the server (default 0; the node exposes one GPU)
  Q3_PORT         server port on 127.0.0.1 (default 30000)
  Q3_OUT          results dir (default <this experiment>/results)
  Q3_LOGS         server logs (default $SGL_ROOT/logs/q3)
  Q3_TRACES       profiler traces (default $SGL_ROOT/traces/q3)
  Q3_STOP_FILE    touch it to make the runner finish the current cell and exit
                  (default $SGL_ROOT/q3/STOP)
"""
from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

HOME = Path.home()
SGL_ROOT = Path(os.environ.get("SGL_ROOT", HOME / "sgl")).expanduser()
HF_HOME = Path(os.environ.get("HF_HOME", HOME / "hf")).expanduser()
MODEL_REPO = os.environ.get("Q3_MODEL_REPO", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_REV = os.environ.get("Q3_MODEL_REV", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
SNAPSHOT = Path(
    os.environ.get(
        "Q3_SNAPSHOT",
        HF_HOME / "hub" / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots" / MODEL_REV,
    )
)
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = Path(os.environ.get("Q3_OUT", EXP / "results")).expanduser()
RAW = OUT / "raw"
LOGS = Path(os.environ.get("Q3_LOGS", SGL_ROOT / "logs" / "q3")).expanduser()
TRACES = Path(os.environ.get("Q3_TRACES", SGL_ROOT / "traces" / "q3")).expanduser()
GPU = os.environ.get("Q3_GPU", "0")
PORT = int(os.environ.get("Q3_PORT", "30000"))
STOP_FILE = Path(os.environ.get("Q3_STOP_FILE", SGL_ROOT / "q3" / "STOP")).expanduser()
STATUS_FILE = OUT / "STATUS.json"

GPU_IDLE_MIB = 2000
SEED = 1
CONCURRENCY = 1
SERVER_WAIT_S = 600
OUTPUT_LEN = 128

# arm -> SGLANG_VIT_ENABLE_CUDA_GRAPH
ARMS = {"off": "0", "on": "1"}

IMG = ["--dataset-name", "image", "--image-count", "1",
       "--image-format", "png", "--image-content", "random"]

# id -> (client flags, text tokens, expected visual tokens incl. the two vision
#        specials as the benchmark counts them, expected ViT patches (= 4 x tokens),
#        note). Patch counts are what the graph key sees.
WORKLOADS = {
    "R0_text":  (["--dataset-name", "random"], 128, 0, 0,
                 "text-only control: the switch must be a no-op here"),
    "R1_256":   (IMG + ["--image-resolution", "256x256"], 128, 66, 256,
                 "smallest real image (= v3 R1_tiny)"),
    "R2_360p":  (IMG + ["--image-resolution", "360p"], 128, 222, 880,
                 "= v3 R2_360p"),
    "R3_512":   (IMG + ["--image-resolution", "512x512"], 128, 258, 1024,
                 "new point between 360p and 640"),
    "R4_640":   (IMG + ["--image-resolution", "640x640"], 128, 402, 1600,
                 "= v3 R6_640, the LM-graph boundary"),
    "R5_720p":  (IMG + ["--image-resolution", "720p"], 128, 882, 3520,
                 "= v3 R3_720p = IMG-A"),
    "R6_1080p": (IMG + ["--image-resolution", "1080p"], 128, 2042, 8160,
                 "= v3 R5_1080p, large-N anchor"),
}
ORDER = list(WORKLOADS)
IMAGE_WORKLOADS = [w for w in ORDER if w != "R0_text"]
MIXED_FLAGS = IMG + ["--image-resolution", "random:256x256-1280x720"]

SERVER_FLAGS = [
    "--dtype", "bfloat16", "--tp", "1", "--host", "127.0.0.1",
    # LM stage exactly as v3 pinned it; it is not under test here.
    "--attention-backend", "flashinfer",
    # The ViT graph supports fa3 / fa4 / triton_attn only; pin instead of resolving.
    "--mm-attention-backend", "fa3",
    # Issue #4's standard condition (and the Hopper default on current main).
    "--mm-feature-transport", "cuda_ipc",
    # LM prefill graph explicitly off on both arms (main auto-disables it for
    # multimodal anyway; saying it keeps the arm what it claims to be).
    "--cuda-graph-backend-prefill", "disabled",
    # Fixed seed => identical prompts across reps; the cache would serve them.
    "--disable-radix-cache",
    "--chunked-prefill-size", "8192",
    # No preprocessing retention: every request runs the full path.
    "--mm-preprocess-cache-size-mb", "0",
    # Per-request queue / forward durations for the decomposition.
    "--enable-request-time-stats-logging",
]

# ---------------------------------------------------------------- logging / status

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg: str) -> None:
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {msg}", flush=True)


def write_status(**fields: Any) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cur: Dict[str, Any] = {}
    if STATUS_FILE.exists():
        try:
            cur = json.loads(STATUS_FILE.read_text())
        except Exception:
            cur = {}
    cur.update(fields)
    cur["updated_utc"] = utc_now()
    STATUS_FILE.write_text(json.dumps(cur, indent=2, default=str))


def stop_requested() -> bool:
    return STOP_FILE.exists()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


# ---------------------------------------------------------------- stats

def percentile(vals: List[float], p: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def median(vals: List[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


# ---------------------------------------------------------------- GPU / server

def gpu_used() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def gpu_name() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "?"


def server_env(arm: str, vit_timing: bool) -> Dict[str, str]:
    env = {**os.environ}
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
              "SGLANG_USE_CUDA_IPC_TRANSPORT", "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
              "LD_PRELOAD"):
        env.pop(k, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": GPU,
        "HF_HOME": str(HF_HOME),
        "HF_HUB_OFFLINE": "1",
        # No embedding retention: every request must run the encoder.
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "SGLANG_VIT_ENABLE_CUDA_GRAPH": ARMS[arm],
        "Q3_VIT_TIMING": "1" if vit_timing else "0",
        "SGLANG_TORCH_PROFILER_DIR": str(TRACES),
    })
    return env


def server_cmd() -> List[str]:
    return ["python3", "-m", "sglang.launch_server",
            "--model-path", str(SNAPSHOT), "--port", str(PORT)] + SERVER_FLAGS


def wait_server(port: int, timeout: float, proc: Optional[subprocess.Popen] = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            log(f"  ERROR: server exited early rc={proc.returncode}")
            return False
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                      timeout=3).getcode() == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def kill_server(proc: Optional[subprocess.Popen]) -> bool:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    for _ in range(60):
        u = gpu_used()
        if 0 <= u < GPU_IDLE_MIB:
            return True
        time.sleep(3)
    log(f"  WARNING: GPU still at {gpu_used()} MiB after kill")
    return False


class ServerHandle:
    def __init__(self, tag: str, arm: str, proc: subprocess.Popen, log_path: Path,
                 info: Optional[Dict[str, Any]], startup_s: float):
        self.tag, self.arm, self.proc, self.log_path = tag, arm, proc, log_path
        self.info, self.startup_s = info, startup_s


def launch_server(tag: str, arm: str, vit_timing: bool,
                  wait_s: float = SERVER_WAIT_S) -> ServerHandle:
    """Launch one server; raises RuntimeError if it does not come up."""
    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        raise RuntimeError(f"GPU not idle before launch (used={u} MiB)")
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}_server.log"
    lf = open(log_path, "w")
    env = server_env(arm, vit_timing)
    cmd = server_cmd()
    log(f"  launch [{tag}] arm={arm} vit_timing={int(vit_timing)} "
        f"SGLANG_VIT_ENABLE_CUDA_GRAPH={env['SGLANG_VIT_ENABLE_CUDA_GRAPH']}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(PORT, wait_s, proc):
        kill_server(proc)
        raise RuntimeError(f"server [{tag}] did not come up in {wait_s}s; see {log_path}")
    startup = round(time.time() - t0, 1)
    info = fetch_server_info(PORT)
    log(f"  up in {startup}s, GPU used {gpu_used()} MiB")
    return ServerHandle(tag, arm, proc, log_path, info, startup)


# ---------------------------------------------------------------- HTTP helpers

def http_post(path: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 120) -> str:
    data = json.dumps(payload or {}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode()


def fetch_server_info(port: int = PORT, timeout: float = 10) -> Optional[Dict[str, Any]]:
    for path in ("/server_info", "/get_server_info"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=timeout) as r:
                return json.load(r)
        except Exception:
            continue
    return None


def dig(obj: Any, key: str) -> Optional[Any]:
    """Depth-first search for a key in nested dicts/lists."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            r = dig(v, key)
            if r is not None:
                return r
    elif isinstance(obj, list):
        for v in obj:
            r = dig(v, key)
            if r is not None:
                return r
    return None


def start_profile(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    http_post("/start_profile", {"output_dir": str(out_dir),
                                 "activities": ["CPU", "GPU"],
                                 "with_stack": False, "record_shapes": False}, timeout=60)


def stop_profile() -> None:
    http_post("/stop_profile", {}, timeout=300)


def trace_files(d: Path) -> set:
    return set(p for p in d.glob("**/*.trace.json*")) if d.exists() else set()


def wait_for_trace(d: Path, before: set, timeout: float = 300) -> Optional[Path]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        new = [p for p in trace_files(d) - before]
        if new:
            p = max(new, key=lambda x: x.stat().st_mtime)
            s1 = p.stat().st_size
            time.sleep(5)
            if p.exists() and p.stat().st_size == s1 and s1 > 0:
                return p
        time.sleep(3)
    return None


# ---------------------------------------------------------------- benchmark client

def bench_cmd(flags: List[str], text_tokens: int, n: int, out_file: Path) -> List[str]:
    return (["python3", "-m", "sglang.benchmark.serving",
             "--backend", "sglang-oai-chat",
             "--base-url", f"http://127.0.0.1:{PORT}",
             "--model", str(SNAPSHOT),
             "--num-prompts", str(n),
             "--max-concurrency", str(CONCURRENCY),
             "--random-input-len", str(text_tokens),
             "--random-output-len", str(OUTPUT_LEN),
             "--random-range-ratio", "1.0",
             "--seed", str(SEED),
             "--output-file", str(out_file), "--output-details", "--disable-tqdm"]
            + list(flags))


def parse_bench_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    last = None
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                last = json.loads(line)
            except Exception:
                pass
    return last


def run_bench(tag: str, flags: List[str], text_tokens: int, n: int,
              env: Dict[str, str], raw_dir: Path) -> Dict[str, Any]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{tag}.jsonl"
    out.unlink(missing_ok=True)
    cmd = bench_cmd(flags, text_tokens, n, out)
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = round(time.time() - t0, 1)
    if res.stderr.strip():
        (raw_dir / f"{tag}.stderr").write_text(res.stderr)
    if res.returncode != 0:
        return {"status": "BENCH_FAILED", "rc": res.returncode,
                "tail": (res.stdout + res.stderr)[-800:], "elapsed_s": elapsed}
    d = parse_bench_jsonl(out)
    if d is None:
        return {"status": "PARSE_ERROR", "elapsed_s": elapsed}
    errors = d.get("errors") or []
    ttfts = [x * 1000 for x in (d.get("ttfts") or []) if x is not None]
    return {
        "status": "OK",
        "completed": d.get("completed"),
        "failures": sum(1 for e in errors if e),
        "ttft_ms": [round(x, 3) for x in ttfts],
        "ttft_p50": percentile(ttfts, 50) if ttfts else d.get("median_ttft_ms"),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_p99": percentile(ttfts, 99) if ttfts else d.get("p99_ttft_ms"),
        "tpot_p50": d.get("median_tpot_ms"),
        "e2e_p50": d.get("median_e2e_latency_ms"),
        "total_input_vision_tokens": d.get("total_input_vision_tokens"),
        "total_input_text_tokens": d.get("total_input_text_tokens"),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------- server-log scanning

RE_CAPTURE = re.compile(r"VIT_CG capture key=(.+?) n_graphs=(\d+) capture_ms=([\d.]+)")
RE_STATS = re.compile(r"VIT_CG_STATS captures=(\d+) replays=(\d+) keys=(\d+) capture_ms_total=([\d.]+)")
RE_TIMING = re.compile(r"VIT_TIMING patches=(\d+) images=(\d+) graph=(\d) "
                       r"cpu_wall_ms=([\d.]+) gpu_span_ms=([\d.]+)")
RE_REQTIME = re.compile(r"ReqTimeStats\(rid=([^,]+), input_len=(\d+).*?\): "
                        r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms")
RE_PREFILL = re.compile(r"Prefill batch.*?#new-token: (\d+).*?cuda graph: (True|False)")
DEGRADATION = [
    (re.compile(r"falling back to non-IPC transport", re.I), "ipc_pool_fallback"),
    (re.compile(r"MmItemMemoryPool has no free chunk", re.I), "mm_pool_exhausted"),
    (re.compile(r"PCG capture stream is not set", re.I), "pcg_eager_fallback"),
    (re.compile(r"ViT CUDA graph does not support attention backend", re.I), "vit_graph_backend_unsupported"),
    (re.compile(r"Traceback \(most recent call last\)"), "traceback"),
]


def log_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def scan_log(path: Path, start: int = 0) -> Dict[str, Any]:
    """Parse the instrumentation and degradation lines written after byte `start`."""
    if not path.exists():
        return {"missing": True}
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read().decode("utf-8", "replace")
    captures = [{"key": m.group(1), "n_graphs": int(m.group(2)), "capture_ms": float(m.group(3))}
                for m in RE_CAPTURE.finditer(text)]
    stats = [{"captures": int(m.group(1)), "replays": int(m.group(2)), "keys": int(m.group(3)),
              "capture_ms_total": float(m.group(4))} for m in RE_STATS.finditer(text)]
    timings = [{"patches": int(m.group(1)), "images": int(m.group(2)), "graph": int(m.group(3)),
                "cpu_wall_ms": float(m.group(4)), "gpu_span_ms": float(m.group(5))}
               for m in RE_TIMING.finditer(text)]
    req_times = [{"rid": m.group(1), "input_len": int(m.group(2)),
                  "queue_ms": float(m.group(3)), "forward_ms": float(m.group(4))}
                 for m in RE_REQTIME.finditer(text)]
    prefill = RE_PREFILL.findall(text)
    # exclude the server's own 1-token readiness probes from the denominator
    real = [(int(n), g) for n, g in prefill if int(n) >= 8]
    degr = {}
    for rx, name in DEGRADATION:
        c = len(rx.findall(text))
        if c:
            degr[name] = c
    return {
        "captures": captures,
        "stats_last": stats[-1] if stats else None,
        "timings": timings,
        "req_times": req_times,
        "prefill_batches": len(real),
        "prefill_graph_true": sum(1 for _, g in real if g == "True"),
        "degradation": degr,
    }


def verify_arm(arm: str, scan: Dict[str, Any], n_requests: int, expect_shapes: int = 1,
               info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """VERIFIED only if the arm demonstrably did what it claims."""
    reasons: List[str] = []
    if scan.get("missing"):
        return {"verdict": "UNVERIFIED", "reasons": ["server log missing"]}
    if info is not None:
        tr = dig(info, "mm_feature_transport")
        if tr is not None and tr != "cuda_ipc":
            reasons.append(f"resolved transport {tr!r} != cuda_ipc")
        pb = dig(info, "cuda_graph_backend_prefill")
        cfg = dig(info, "cuda_graph_config")
        pcfg = cfg.get("prefill", {}).get("backend") if isinstance(cfg, dict) else None
        if (pcfg or pb) not in (None, "disabled"):
            reasons.append(f"resolved prefill graph backend {pcfg or pb!r} != disabled")
        mm = dig(info, "mm_attention_backend")
        if mm not in (None, "fa3"):
            reasons.append(f"resolved mm attention backend {mm!r} != fa3")
    if scan.get("prefill_graph_true", 0):
        reasons.append(f"{scan['prefill_graph_true']} prefill batches ran under an LM graph")
    if scan.get("degradation"):
        reasons.append(f"degradation signals: {scan['degradation']}")
    n_cap = len(scan.get("captures", []))
    st = scan.get("stats_last")
    if arm == "off":
        if n_cap or st:
            reasons.append("ViT graph lines present on the off arm")
    else:
        if expect_shapes >= 0 and n_cap != expect_shapes:
            reasons.append(f"captures={n_cap}, expected {expect_shapes}")
        if st is None:
            reasons.append("no VIT_CG_STATS line")
        else:
            # stats are emitted every 50 replays, so allow that lag
            if st["replays"] < max(0, n_requests - n_cap - 50):
                reasons.append(f"replays={st['replays']} < requests-captures-50={n_requests - n_cap - 50}")
    return {"verdict": "VERIFIED" if not reasons else "UNVERIFIED", "reasons": reasons}


def summarize_timings(timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not timings:
        return {"n": 0}
    return {
        "n": len(timings),
        "patches": median([t["patches"] for t in timings]),
        "graph": timings[-1]["graph"],
        "cpu_wall_ms_p50": median([t["cpu_wall_ms"] for t in timings]),
        "gpu_span_ms_p50": median([t["gpu_span_ms"] for t in timings]),
    }


def summarize_req_times(rts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rts:
        return {"n": 0}
    return {"n": len(rts),
            "queue_ms_p50": median([r["queue_ms"] for r in rts]),
            "forward_ms_p50": median([r["forward_ms"] for r in rts])}
