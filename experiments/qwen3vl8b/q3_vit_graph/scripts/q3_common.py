#!/usr/bin/env python3
"""Q3 (ViT CUDA graph) — shared helpers, node side.

Every path comes from the environment so the scripts run unchanged on any RADIX
node and never need editing:

  SGL_ROOT        node workspace (default ~/sgl)
  HF_HOME         model cache (default ~/hf)
  Q3_MODEL_REPO   Qwen/Qwen3-VL-8B-Instruct
  Q3_MODEL_REV    0c351dd01ed87e9c1b53cbc748cba10e6187ff3b   (the v2/v3 protocol pin)
  Q3_SGLANG_SHA   the SGLang commit the measurement patch was written against
  Q3_GPU          CUDA_VISIBLE_DEVICES for the server (default 0; the node exposes one GPU)
  Q3_PORT         server port on 127.0.0.1 (default 30000)
  Q3_OUT          results dir (default <this experiment>/results)
  Q3_LOGS         server logs (default $SGL_ROOT/logs/q3)
  Q3_TRACES       profiler traces (default $SGL_ROOT/traces/q3)
  Q3_STOP_FILE    touch it to make the runner finish the current cell and exit
                  (default $SGL_ROOT/q3/STOP)
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

PY = sys.executable  # always the interpreter this script runs under (the conda env), never a bare "python3"

HOME = Path.home()
SGL_ROOT = Path(os.environ.get("SGL_ROOT", HOME / "sgl")).expanduser()
HF_HOME = Path(os.environ.get("HF_HOME", HOME / "hf")).expanduser()
MODEL_REPO = os.environ.get("Q3_MODEL_REPO", "Qwen/Qwen3-VL-8B-Instruct")
MODEL_REV = os.environ.get("Q3_MODEL_REV", "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
SGLANG_SHA = os.environ.get("Q3_SGLANG_SHA", "89e1316eae")
SNAPSHOT = Path(
    os.environ.get(
        "Q3_SNAPSHOT",
        HF_HOME / "hub" / f"models--{MODEL_REPO.replace('/', '--')}" / "snapshots" / MODEL_REV,
    )
)
SHAREGPT_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"
HERE = Path(__file__).resolve().parent
EXP = HERE.parent
OUT = Path(os.environ.get("Q3_OUT", EXP / "results")).expanduser()
RAW = OUT / "raw"
CELLS = OUT / "cells"          # git-tracked per-cell summaries: the sweep's resume state
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
BENCH_TIMEOUT_S = int(os.environ.get("Q3_BENCH_TIMEOUT_S", "1500"))
# Output length does not touch TTFT at c=1 (input bytes and the client RNG are unaffected);
# 16 tokens halves the sweep's wall time. The pilot checks this on R1_256 (16 vs 128).
OUTPUT_LEN = 16

# arm -> environment. "off_rot" is the eager arm with the graph arm's unfused rotary
# forced on (D3): it isolates the rotary implementation as a variable.
ARM_ENV = {
    "off":     {"SGLANG_VIT_ENABLE_CUDA_GRAPH": "0", "Q3_VIT_EAGER_ROTARY": "0"},
    "on":      {"SGLANG_VIT_ENABLE_CUDA_GRAPH": "1", "Q3_VIT_EAGER_ROTARY": "0"},
    "off_rot": {"SGLANG_VIT_ENABLE_CUDA_GRAPH": "0", "Q3_VIT_EAGER_ROTARY": "1"},
}
ARMS = {k: v["SGLANG_VIT_ENABLE_CUDA_GRAPH"] for k, v in ARM_ENV.items()}

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
LARGE_WORKLOADS = ["R5_720p", "R6_1080p"]   # near-zero predicted gain: they get one more block (D4)
# 'random:<min_h>x<min_w>-<max_h>x<max_w>' => heights 256..720, widths 256..1280 (landscape up to 720p)
MIXED_FLAGS = IMG + ["--image-resolution", "random:256x256-720x1280"]

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
    # The startup warmup sends a 32x32 image, which would pre-capture the 256-patch
    # graph on the `on` arm before any cell starts; both arms skip it (B1).
    "--skip-server-warmup",
    # Leave room for the mixed stage's ~165 private graph pools (O1); c=1 never
    # touches the KV pool size.
    "--mem-fraction-static", "0.75",
    # Graph mode uses the legacy position-embedding interpolation, which honours
    # this flag; eager uses linspace regardless. With the flag both are linspace (B4).
    "--enable-precise-embedding-interpolation",
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


def clear_stop() -> bool:
    """Remove a leftover STOP file at stage start (O8). Returns True if one was there."""
    if STOP_FILE.exists():
        STOP_FILE.unlink()
        return True
    return False


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.replace(path)


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


def median(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(statistics.median(vals), 3) if vals else None


def mean(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def stdev(vals: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return round(statistics.stdev(vals), 3) if len(vals) >= 2 else None


# ---------------------------------------------------------------- paths

def sharegpt_path() -> Optional[Path]:
    """The ShareGPT file the text-only control samples from (hf download ... --repo-type dataset)."""
    env = os.environ.get("Q3_SHAREGPT")
    if env and Path(env).exists():
        return Path(env)
    hits = sorted(glob.glob(str(HF_HOME / "hub" / "datasets--anon8231489123--ShareGPT_Vicuna_unfiltered"
                                / "snapshots" / "*" / SHAREGPT_FILE)))
    return Path(hits[-1]) if hits else None


def flags_for(wid: str) -> List[str]:
    """Client flags for a workload; the text control gets its dataset file explicitly (B3)."""
    flags = list(WORKLOADS[wid][0])
    if wid == "R0_text":
        p = sharegpt_path()
        if p is not None:
            flags += ["--dataset-path", str(p)]
    return flags


def model_shards_complete() -> Dict[str, Any]:
    """Every safetensors shard named in the index must exist and be non-empty (O6)."""
    idx = SNAPSHOT / "model.safetensors.index.json"
    if not idx.exists():
        single = SNAPSHOT / "model.safetensors"
        return {"complete": single.exists() and single.stat().st_size > 0, "shards": 1 if single.exists() else 0}
    try:
        names = sorted(set(load_json(idx).get("weight_map", {}).values()))
    except Exception as e:  # noqa: BLE001
        return {"complete": False, "error": str(e)}
    missing = [n for n in names if not (SNAPSHOT / n).exists() or (SNAPSHOT / n).stat().st_size == 0]
    return {"complete": not missing and bool(names), "shards": len(names), "missing": missing}


# ---------------------------------------------------------------- GPU / server

def gpu_used() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def gpu_total() -> int:
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def gpu_name() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                        "--format=csv,noheader"], capture_output=True, text=True)
    return r.stdout.strip().splitlines()[0] if r.returncode == 0 and r.stdout.strip() else "?"


def server_env(arm: str, vit_timing: bool, dump_dir: Optional[Path] = None) -> Dict[str, str]:
    env = {**os.environ}
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
              "SGLANG_USE_CUDA_IPC_TRANSPORT", "SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK",
              "LD_PRELOAD", "Q3_VIT_DUMP"):
        env.pop(k, None)
    env.update({
        "CUDA_VISIBLE_DEVICES": GPU,
        "HF_HOME": str(HF_HOME),
        "HF_HUB_OFFLINE": "1",
        # No embedding retention: every request must run the encoder.
        "SGLANG_VLM_CACHE_SIZE_MB": "0",
        "Q3_VIT_TIMING": "1" if vit_timing else "0",
        "SGLANG_TORCH_PROFILER_DIR": str(TRACES),
        **ARM_ENV[arm],
    })
    if dump_dir is not None:
        env["Q3_VIT_DUMP"] = str(dump_dir)
    return env


def server_cmd(drop_flags: Optional[List[str]] = None) -> List[str]:
    """The pinned server command; `drop_flags` removes boolean flags (parity's optional
    default-interpolation arm)."""
    flags = [f for f in SERVER_FLAGS if f not in (drop_flags or [])]
    return [PY, "-m", "sglang.launch_server",
            "--model-path", str(SNAPSHOT), "--port", str(PORT)] + flags


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
                 info: Optional[Dict[str, Any]], startup_s: float, launch_offset: int = 0):
        self.tag, self.arm, self.proc, self.log_path = tag, arm, proc, log_path
        self.info, self.startup_s, self.launch_offset = info, startup_s, launch_offset


def launch_server(tag: str, arm: str, vit_timing: bool, wait_s: float = SERVER_WAIT_S,
                  dump_dir: Optional[Path] = None,
                  drop_flags: Optional[List[str]] = None) -> ServerHandle:
    """Launch one server; raises RuntimeError if it does not come up.

    `launch_offset` is the log size once /health answers: every scan starts there, so the
    startup noise (torchcodec import tracebacks, warmup lines) never reaches a verdict (B2).
    """
    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        raise RuntimeError(f"GPU not idle before launch (used={u} MiB)")
    LOGS.mkdir(parents=True, exist_ok=True)
    log_path = LOGS / f"{tag}_server.log"
    lf = open(log_path, "w")
    env = server_env(arm, vit_timing, dump_dir)
    cmd = server_cmd(drop_flags)
    log(f"  launch [{tag}] arm={arm} vit_timing={int(vit_timing)} "
        f"graph={env['SGLANG_VIT_ENABLE_CUDA_GRAPH']} eager_rotary={env['Q3_VIT_EAGER_ROTARY']}")
    t0 = time.time()
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not wait_server(PORT, wait_s, proc):
        kill_server(proc)
        raise RuntimeError(f"server [{tag}] did not come up in {wait_s}s; see {log_path}")
    startup = round(time.time() - t0, 1)
    info = fetch_server_info(PORT)
    off = log_offset(log_path)
    log(f"  up in {startup}s, GPU used {gpu_used()} MiB, log offset {off}")
    return ServerHandle(tag, arm, proc, log_path, info, startup, off)


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


def resolved_config(info: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    keys = ("mm_feature_transport", "mm_attention_backend", "cuda_graph_backend_prefill",
            "attention_backend", "mem_fraction_static", "skip_server_warmup",
            "enable_precise_embedding_interpolation", "mm_preprocess_cache_size_mb",
            "disable_radix_cache", "chunked_prefill_size")
    out = {k: dig(info, k) for k in keys} if info else {}
    cfg = dig(info, "cuda_graph_config") if info else None
    if isinstance(cfg, dict):
        out["prefill_graph_backend_resolved"] = (cfg.get("prefill") or {}).get("backend")
    return out


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
    cmd = [PY, "-m", "sglang.benchmark.serving",
           "--backend", "sglang-oai-chat",
           "--base-url", f"http://127.0.0.1:{PORT}",
           "--model", str(SNAPSHOT),
           "--num-prompts", str(n),
           "--max-concurrency", str(CONCURRENCY),
           "--random-input-len", str(text_tokens),
           "--random-range-ratio", "1.0",
           "--seed", str(SEED),
           "--output-file", str(out_file), "--output-details", "--disable-tqdm"]
    if "--random-output-len" not in flags:
        cmd += ["--random-output-len", str(OUTPUT_LEN)]
    return cmd + list(flags)


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
    """One benchmark run. Failed requests (error set or ttft <= 0) are excluded from the
    statistics but kept, by index, in `failed_idx` so per-request alignment survives (O7)."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = raw_dir / f"{tag}.jsonl"
    out.unlink(missing_ok=True)
    cmd = bench_cmd(flags, text_tokens, n, out)
    t0 = time.time()
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=BENCH_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        subprocess.run(["pkill", "-9", "-f", "sglang.benchmark.serving"], capture_output=True)
        return {"status": "BENCH_TIMEOUT", "elapsed_s": round(time.time() - t0, 1)}
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
    raw_ttft = d.get("ttfts") or []
    failed_idx = [i for i, t in enumerate(raw_ttft)
                  if (i < len(errors) and errors[i]) or t is None or t <= 0]
    ttft_all = [round(t * 1000, 3) if (t is not None and i not in failed_idx) else None
                for i, t in enumerate(raw_ttft)]
    ttfts = [t for t in ttft_all if t is not None]
    return {
        "status": "OK" if ttfts else "ALL_FAILED",
        "completed": d.get("completed"),
        "failures": len(failed_idx),
        "failed_idx": failed_idx,
        "ttft_ms": ttft_all,                       # index-aligned with the request order
        "ttft_p50": percentile(ttfts, 50),
        "ttft_mean": mean(ttfts),
        "ttft_p90": percentile(ttfts, 90),
        "ttft_p99": percentile(ttfts, 99),
        "tpot_p50": d.get("median_tpot_ms"),
        "e2e_p50": d.get("median_e2e_latency_ms"),
        "total_input_vision_tokens": d.get("total_input_vision_tokens"),
        "total_input_text_tokens": d.get("total_input_text_tokens"),
        "elapsed_s": elapsed,
    }


# ---------------------------------------------------------------- server-log scanning

RE_CAPTURE = re.compile(r"VIT_CG capture key=(.+?) n_graphs=(\d+) capture_ms=([\d.]+)(?: reserved_mib=(\d+))?")
RE_STATS = re.compile(r"VIT_CG_STATS captures=(\d+) replays=(\d+) keys=(\d+) capture_ms_total=([\d.]+)")
RE_CALL = re.compile(r"VIT_CG_CALL replays=(\d+) keys=(\d+)")
RE_TIMING = re.compile(r"VIT_TIMING patches=(\d+) images=(\d+) graph=(\d) "
                       r"cpu_wall_ms=([\d.]+) gpu_span_ms=([\d.]+)")
RE_REQTIME = re.compile(r"ReqTimeStats\(rid=([^,]+), input_len=(\d+).*?\): "
                        r"queue_duration=([\d.]+)ms, forward_duration=([\d.]+)ms")
RE_PREFILL = re.compile(r"Prefill batch.*?#new-token: (\d+).*?cuda graph: (True|False)")
RE_TORCHCODEC = re.compile(r"\[start of libtorchcodec loading traceback\].*?"
                           r"\[end of libtorchcodec loading traceback\]\.?", re.S)
DEGRADATION = [
    (re.compile(r"falling back to non-IPC transport", re.I), "ipc_pool_fallback"),
    (re.compile(r"MmItemMemoryPool has no free chunk", re.I), "mm_pool_exhausted"),
    (re.compile(r"PCG capture stream is not set", re.I), "pcg_eager_fallback"),
    (re.compile(r"ViT CUDA graph does not support attention backend", re.I), "vit_graph_backend_unsupported"),
    (re.compile(r"CUDA out of memory|OutOfMemoryError", re.I), "cuda_oom"),
    (re.compile(r"Traceback \(most recent call last\)"), "traceback"),
]
EFFECTIVE = {  # informational confirmations printed by the server itself (server_runtime-6)
    "mm_backend_fa3_log": re.compile(r"Using fa3 as multimodal attention backend"),
    "hopper_override_log": re.compile(r"Applying profiled qwen3_vl serving defaults"),
    "prefill_graph_disabled_log": re.compile(r"disabling prefill CUDA graph|prefill.*backend=disabled", re.I),
}


def log_offset(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def read_log(path: Path, start: int = 0, end: Optional[int] = None) -> str:
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read() if end is None else f.read(max(0, end - start))
    return data.decode("utf-8", "replace")


def scan_log(path: Path, start: int = 0, end: Optional[int] = None) -> Dict[str, Any]:
    """Parse the instrumentation and degradation lines in the byte range [start, end).
    The torchcodec import tracebacks the server prints at startup are removed first (B2)."""
    if not path.exists():
        return {"missing": True}
    text = RE_TORCHCODEC.sub("", read_log(path, start, end))
    captures = [{"key": m.group(1), "n_graphs": int(m.group(2)), "capture_ms": float(m.group(3)),
                 "reserved_mib": int(m.group(4)) if m.group(4) else None}
                for m in RE_CAPTURE.finditer(text)]
    stats = [{"captures": int(m.group(1)), "replays": int(m.group(2)), "keys": int(m.group(3)),
              "capture_ms_total": float(m.group(4))} for m in RE_STATS.finditer(text)]
    calls = [{"replays": int(m.group(1)), "keys": int(m.group(2))} for m in RE_CALL.finditer(text)]
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
        "stats_first": stats[0] if stats else None,
        "stats_last": stats[-1] if stats else None,
        "n_calls": len(calls),
        "calls_last": calls[-1] if calls else None,
        "timings": timings,
        "n_timing_graph1": sum(1 for t in timings if t["graph"] == 1),
        "req_times": req_times,
        "prefill_batches": len(real),
        "prefill_graph_true": sum(1 for _, g in real if g == "True"),
        "degradation": degr,
        "effective": {k: bool(rx.search(text)) for k, rx in EFFECTIVE.items()},
    }


def verify_arm(arm: str, scan: Dict[str, Any], n_requests: int, expect_shapes: int = 1,
               info: Optional[Dict[str, Any]] = None,
               scan_life: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """VERIFIED only if the arm demonstrably did what it claims, in the scanned window.

    `scan` covers the cell's own window; `scan_life` (optional) covers the server's whole
    life since launch and is only used as fallback evidence for a shape captured before
    the window (pilot: the same server serves every workload).

    on-arm evidence, strongest first, and the one used is recorded in `evidence`:
      per-call   VIT_CG_CALL lines in the window: captures + calls must cover the requests
      stats      no per-call lines anywhere (unpatched runner or simulator): lifetime
                 VIT_CG_STATS must exist and its replay count must be consistent
    Text workloads on either arm must show no graph activity at all (B1).
    """
    reasons: List[str] = []
    if scan.get("missing"):
        return {"verdict": "UNVERIFIED", "reasons": ["server log missing"], "evidence": None}
    life = scan_life or scan
    if info is not None:
        rc = resolved_config(info)
        if rc.get("mm_feature_transport") not in (None, "cuda_ipc"):
            reasons.append(f"resolved transport {rc['mm_feature_transport']!r} != cuda_ipc")
        pb = rc.get("prefill_graph_backend_resolved") or rc.get("cuda_graph_backend_prefill")
        if pb not in (None, "disabled"):
            reasons.append(f"resolved prefill graph backend {pb!r} != disabled")
        if rc.get("mm_attention_backend") not in (None, "fa3"):
            reasons.append(f"resolved mm attention backend {rc['mm_attention_backend']!r} != fa3")
    if scan.get("prefill_batches", 0) < 1:
        reasons.append("no prefill batch reached the server in this window")
    if scan.get("prefill_graph_true", 0):
        reasons.append(f"{scan['prefill_graph_true']} prefill batches ran under an LM graph")
    if scan.get("degradation"):
        reasons.append(f"degradation signals: {scan['degradation']}")
    n_cap = len(scan.get("captures") or [])
    graph_lines = n_cap + scan.get("n_calls", 0) + (1 if scan.get("stats_last") else 0) \
        + scan.get("n_timing_graph1", 0)
    evidence = None
    if arm != "on" or expect_shapes == 0:
        # eager arms, and the text control on the graph arm: no graph activity at all
        if graph_lines:
            reasons.append("ViT graph lines present on a workload that must not use the graph")
        evidence = "no_graph_lines"
    else:
        if expect_shapes >= 0 and n_cap > expect_shapes:   # -1 = any number of shapes (mixed stage)
            reasons.append(f"captures={n_cap} > expected {expect_shapes}")
        if scan.get("n_calls", 0) or life.get("n_calls", 0):
            covered = n_cap + scan.get("n_calls", 0)
            need = max(1, n_requests - 3)          # the client's own warmup requests are not counted exactly
            if covered < need:
                reasons.append(f"graph calls+captures={covered} < requests-3={need}")
            evidence = "per_call"
        else:
            st = life.get("stats_last")
            if st is None:
                reasons.append("no VIT_CG_STATS line (graph never captured on this server)")
            elif st["replays"] < max(0, n_requests - n_cap - 50):
                reasons.append(f"replays={st['replays']} < requests-captures-50={n_requests - n_cap - 50}")
            evidence = "stats_lifetime"
        if scan.get("timings") and scan.get("n_timing_graph1", 0) < len(scan["timings"]):
            reasons.append("VIT_TIMING lines with graph=0 on the on arm")
    return {"verdict": "VERIFIED" if not reasons else "UNVERIFIED", "reasons": reasons,
            "evidence": evidence, "captures": n_cap, "calls": scan.get("n_calls", 0),
            "prefill_batches": scan.get("prefill_batches", 0)}


def summarize_timings(timings: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not timings:
        return {"n": 0}
    return {
        "n": len(timings),
        "patches": median([t["patches"] for t in timings]),
        "graph": timings[-1]["graph"],
        "cpu_wall_ms_p50": median([t["cpu_wall_ms"] for t in timings]),
        "cpu_wall_ms_sd": stdev([t["cpu_wall_ms"] for t in timings]),
        "gpu_span_ms_p50": median([t["gpu_span_ms"] for t in timings]),
    }


def summarize_req_times(rts: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rts:
        return {"n": 0}
    return {"n": len(rts),
            "queue_ms_p50": median([r["queue_ms"] for r in rts]),
            "forward_ms_p50": median([r["forward_ms"] for r in rts])}
