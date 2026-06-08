#!/usr/bin/env python3
"""
v2 / Issue #4 — Stage 4.2 IMG-A formal benchmark runner (fixed-generator path).

Drives `python -m sglang.bench_serving --dataset-name image --backend sglang-oai-chat`
against the **fixed** SGLang generator (upstream commit
`07f326c184 Fix multimodal synthetic benchmark prompt generation to exclude special
tokens (#26864)`), sourced from `/data/sglang-pr` on `main`. Selected via
`PYTHONPATH=/data/sglang-pr/python` — `/sgl-workspace/sglang` is NEVER imported.

Workload IMG-A: single 720p image + short text (~128 raw text tok), c=1.
5 variants in bracket order to bound drift. reps=5, num-prompts=400, warmup=30.
Bracket order: S0_ipc → S2_ipc_pcg → S0_ipc_repeat → V0_vllm → S0_noipc

Run-gating: Stage 4.1 fixed-generator smoke MUST have passed first.

Variants and IPC discipline:
  - `IMG_A_S0_ipc`        : SGLang, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`  (headline)
  - `IMG_A_S2_ipc_pcg`    : SGLang, `SGLANG_USE_CUDA_IPC_TRANSPORT=1` + `--enforce-piecewise-cuda-graph`
  - `IMG_A_S0_ipc_repeat` : SGLang, `SGLANG_USE_CUDA_IPC_TRANSPORT=1`  (bracket repeat of S0)
  - `IMG_A_V0_vllm`       : vLLM clean anchor (IPC env n/a; vLLM has its own Python env)
  - `IMG_A_S0_noipc`      : SGLang, `SGLANG_USE_CUDA_IPC_TRANSPORT` UNSET  (IPC ablation)

Comparison map:
  - IPC benefit (Q3): S0_ipc vs S0_noipc
  - PCG benefit (Q2): S2_ipc_pcg vs bracketed S0_ipc
  - vLLM anchor (Q1): S0_ipc vs V0_vllm (anchor only, not causal)

Outputs (committed):
    experiments/qwen3vl8b/v2/image_text_benchmarks/results_fixed/imgA_results.json
    experiments/qwen3vl8b/v2/image_text_benchmarks/results_fixed/imgA_summary.md

Raw per-rep dumps and server logs (NOT committed unless explicitly approved):
    results_fixed/raw/<variant>_rep<N>.jsonl
    logs/qwen3vl8b/v2/image_text_benchmarks/results_fixed/<variant>_server.log

STRICT CLEAN: never sets SGLANG_KERNEL_API_LOGLEVEL / SGLANG_KERNEL_API_LOGDEST;
no profiler. SGLANG_USE_CUDA_IPC_TRANSPORT=1 is set ONLY for IPC variants.
GPU: 7 (fixed per user spec — never auto-switch).
Never overwrites or touches old results/, smoke/, debug_video_pad/, or v1 paths.
"""
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ----- Fixed paths and constants -----
SNAPSHOT = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
LAB          = Path("/data/sglang-vllm-profiler")
BASE         = LAB / "experiments/qwen3vl8b/v2/image_text_benchmarks"
RESULTS      = BASE / "results_fixed"
RAW          = RESULTS / "raw"
LOGS         = LAB / "logs/qwen3vl8b/v2/image_text_benchmarks/results_fixed"
SGLANG_PR    = Path("/data/sglang-pr")
SGLANG_PR_PY = str(SGLANG_PR / "python")
VLLM_PYTHON  = "/opt/miniconda3/envs/profiling/bin/python"
GPU = "7"  # user-specified; never auto-switch
SGLANG_PORT  = 30000
VLLM_PORT    = 30001
GPU_IDLE_MIB = 2000
SEED = 1

# IMG-A workload params (bracket c=1)
NUM_PROMPTS = 400
WARMUP      = 30
REPS        = 5
CONCURRENCY = 1

IMAGE_ARGS_BASE = [
    "--dataset-name",       "image",
    "--image-count",        "1",
    "--image-resolution",   "720p",
    "--image-format",       "png",
    "--image-content",      "random",
    "--random-input-len",   "128",
    "--random-output-len",  "128",
    "--random-range-ratio", "1.0",
    "--max-concurrency",    str(CONCURRENCY),
    "--num-prompts",        str(NUM_PROMPTS),
    "--warmup-requests",    str(WARMUP),
    "--seed",               str(SEED),
    "--extra-request-body", '{"temperature": 0, "top_p": 1}',
    "--output-details",
]


# ----- Base env: pin GPU, offline HF, strip KAPI and IPC; PYTHONPATH overlays fixed SGLang -----
def _make_base_env():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}
    # Strict: KAPI and IPC always start unset. IPC is added per-variant for IPC-on cases.
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
              "SGLANG_USE_CUDA_IPC_TRANSPORT"):
        env.pop(k, None)
    # PYTHONPATH: prepend the fixed SGLang clone so `import sglang` resolves there.
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SGLANG_PR_PY + (os.pathsep + prev if prev else "")
    return env

_BASE_ENV = _make_base_env()

# Variants: (id, framework, ipc_on, server_flags, server_wait_s)
# Bracket order: S0_ipc -> S2_ipc_pcg -> S0_ipc_repeat -> V0_vllm -> S0_noipc
IMG_A_VARIANTS = [
    ("IMG_A_S0_ipc",        "sglang", True,  [],                                   480),
    ("IMG_A_S2_ipc_pcg",    "sglang", True,  ["--enforce-piecewise-cuda-graph"],   900),
    ("IMG_A_S0_ipc_repeat", "sglang", True,  [],                                   480),
    ("IMG_A_V0_vllm",       "vllm",   False, [],                                   600),
    ("IMG_A_S0_noipc",      "sglang", False, [],                                   480),
]


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def percentile(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def gpu_used():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPU],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_server(port, timeout, proc=None):
    """Wait for /health to return 200. Fail fast if the server process exits early."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            log(f"  ERROR: server process exited early with rc={proc.returncode}")
            return False
        try:
            code = urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=3).getcode()
            if code == 200:
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
            proc.kill()
            proc.wait()
    subprocess.run(["pkill", "-9", "-f", patt], capture_output=True)
    for _ in range(40):
        u = gpu_used()
        if 0 <= u < GPU_IDLE_MIB:
            log(f"  GPU {GPU} freed (used={u} MiB)")
            return True
        time.sleep(3)
    log(f"  WARNING: GPU {GPU} still {gpu_used()} MiB after kill")
    return False


def parse_bench_jsonl(path):
    try:
        lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception as ex:
        log(f"  parse error: {ex}")
        return None


def collect_provenance():
    """Record SGLang clone state and the actual sglang import location."""
    out = {
        "sglang_pr_path": str(SGLANG_PR),
        "pythonpath_prefix": SGLANG_PR_PY,
    }
    try:
        sha = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        out["sglang_pr_head_sha"] = sha
    except Exception as ex:
        out["sglang_pr_head_sha"] = f"ERROR:{ex}"
    try:
        br = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        out["sglang_pr_branch"] = br
    except Exception as ex:
        out["sglang_pr_branch"] = f"ERROR:{ex}"
    try:
        merged = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "merge-base", "--is-ancestor",
             "07f326c184", "HEAD"],
            capture_output=True, text=True)
        out["merged_fix_in_history"] = (merged.returncode == 0)
    except Exception as ex:
        out["merged_fix_in_history"] = f"ERROR:{ex}"
    probe = subprocess.run(
        ["python3", "-c",
         "import sglang, sglang.benchmark.datasets.common as c; "
         "import inspect; "
         "print(sglang.__file__); print(c.__file__); "
         "print('FIX_OK' if 'get_available_multimodal_text_tokens' "
         "in inspect.getsource(c.gen_mm_prompt) else 'FIX_MISSING')"],
        capture_output=True, text=True, env=_BASE_ENV)
    lines = (probe.stdout or "").strip().splitlines()
    out["sglang_dunder_file"] = lines[0] if len(lines) >= 1 else None
    out["common_dunder_file"] = lines[1] if len(lines) >= 2 else None
    out["fix_marker"]         = lines[2] if len(lines) >= 3 else None
    out["probe_stderr_tail"]  = (probe.stderr or "")[-300:]
    out["fixed_path_ok"] = (
        out["sglang_dunder_file"] is not None
        and SGLANG_PR_PY in out["sglang_dunder_file"]
        and out["fix_marker"] == "FIX_OK"
    )
    return out


def run_rep(vid, framework, port, rep, case_env):
    out_jsonl = RAW / f"{vid}_rep{rep}.jsonl"
    out_jsonl.unlink(missing_ok=True)
    cmd = (
        ["python3", "-m", "sglang.bench_serving",
         "--backend", "sglang-oai-chat",
         "--base-url", f"http://127.0.0.1:{port}",
         "--model", SNAPSHOT]
        + IMAGE_ARGS_BASE
        + ["--output-file", str(out_jsonl)]
    )
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=case_env)
    elapsed = round(time.time() - t0, 1)
    if res.returncode != 0:
        log(f"    rep{rep} FAILED rc={res.returncode}: {(res.stdout+res.stderr)[-300:]}")
        return False, {"status": "BENCH_FAILED", "rc": res.returncode}
    d = parse_bench_jsonl(out_jsonl)
    if d is None:
        log(f"    rep{rep} parse error")
        return False, {"status": "PARSE_ERROR"}
    errors = d.get("errors", [])
    n_fail = sum(1 for e in errors if e)
    forbidden_err_present = any(
        "No data iterator found for token" in str(e) for e in errors if e)
    ttfts_raw = d.get("ttfts", [])
    ttfts_ms = [t * 1000 for t in ttfts_raw if t is not None]
    metrics = {
        "completed": d.get("completed"),
        "failures": n_fail,
        "forbidden_token_error_present": forbidden_err_present,
        "ttft_p50": percentile(ttfts_ms, 50) if ttfts_ms else d.get("median_ttft_ms"),
        "ttft_p95": percentile(ttfts_ms, 95) if ttfts_ms else None,
        "ttft_p99": percentile(ttfts_ms, 99) if ttfts_ms else d.get("p99_ttft_ms"),
        "tpot_p50": d.get("median_tpot_ms"),
        "tpot_p99": d.get("p99_tpot_ms"),
        "e2e_p50": d.get("median_e2e_latency_ms"),
        "out_tok_s": d.get("output_throughput"),
        "req_s": d.get("request_throughput"),
        "total_input_vision_tokens": d.get("total_input_vision_tokens"),
        "total_input_text_tokens": d.get("total_input_text_tokens"),
        "errors_sample": [str(e) for e in errors[:2] if e],
        "elapsed_s": elapsed,
    }
    log(f"    rep{rep} {elapsed}s  completed={metrics['completed']}  fail={n_fail}  "
        f"ttft_p50={metrics['ttft_p50']}ms  tpot_p50={metrics['tpot_p50']}ms")
    if forbidden_err_present:
        log(f"    rep{rep} ⚠️ forbidden-token error returned — fix path broken")
    return (n_fail == 0 and not forbidden_err_present), metrics


def run_variant(vid, framework, ipc_on, extra_flags, wait_s):
    log(f"\n{'='*60}")
    log(f"VARIANT: {vid}  fw={framework}  "
        f"IPC={'ON (SGLANG_USE_CUDA_IPC_TRANSPORT=1)' if ipc_on else ('OFF (env unset)' if framework=='sglang' else 'N/A (vLLM)')}  "
        f"flags={extra_flags or 'default'}")
    log(f"{'='*60}")

    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"  STOP: GPU {GPU} not idle (used={u} MiB)")
        return {"id": vid, "status": "GPU_NOT_IDLE", "gpu_mib": u,
                "kapi_logging": False, "profiler": False}

    case_env = {**_BASE_ENV}
    if ipc_on:
        case_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"
    else:
        case_env.pop("SGLANG_USE_CUDA_IPC_TRANSPORT", None)
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        case_env.pop(k, None)
    # vLLM server uses VLLM_PYTHON and does not import sglang at server level. The
    # bench client (python3 -m sglang.bench_serving) DOES honor PYTHONPATH and will
    # load the fixed sglang.benchmark.datasets generator code.

    port = SGLANG_PORT if framework == "sglang" else VLLM_PORT
    patt = ("sglang.launch_server" if framework == "sglang"
            else "vllm.entrypoints.openai.api_server")

    if framework == "sglang":
        srv_cmd = (["python3", "-m", "sglang.launch_server",
                    "--model-path", SNAPSHOT, "--dtype", "bfloat16",
                    "--port", str(port), "--tp", "1",
                    "--attention-backend", "flashinfer"] + extra_flags)
    else:
        srv_cmd = ([VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
                    "--model", SNAPSHOT, "--dtype", "bfloat16",
                    "--port", str(port), "--tensor-parallel-size", "1"] + extra_flags)

    LOGS.mkdir(parents=True, exist_ok=True)
    lf = open(LOGS / f"{vid}_server.log", "w")
    log(f"  launching {framework} port={port} wait≤{wait_s}s ...")
    proc = subprocess.Popen(srv_cmd, env=case_env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {
        "id": vid, "framework": framework, "ipc_on": ipc_on,
        "extra_flags": extra_flags, "port": port,
        "kapi_logging": False, "profiler": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU, "snapshot": SNAPSHOT.split("/")[-1],
        "num_prompts": NUM_PROMPTS, "warmup": WARMUP,
        "concurrency": CONCURRENCY, "seed": SEED,
    }

    try:
        if not wait_server(port, wait_s, proc=proc):
            log(f"  ERROR: server did not come up in {wait_s}s")
            rec["status"] = "SERVER_NO_START"
            return rec

        log(f"  server up. Running {REPS} reps  n={NUM_PROMPTS}  warmup={WARMUP}  c={CONCURRENCY}")
        rep_list = []
        any_fail = False
        for rep in range(1, REPS + 1):
            ok, metrics = run_rep(vid, framework, port, rep, case_env)
            rep_list.append(metrics)
            if metrics.get("forbidden_token_error_present"):
                log(f"  {vid} rep{rep}: forbidden-token error — STOPPING all reps for this variant")
                any_fail = True
                rec["forbidden_token_error_present"] = True
                break
            if not ok or (metrics.get("failures") or 0) > 0:
                log(f"  {vid} rep{rep}: failed/failures>0 — stopping reps")
                any_fail = True
                break

        rec["reps"] = rep_list
        rec["status"] = "INVALID_FAILURES" if any_fail else "OK"

    finally:
        kill_server(proc, patt)

    if rec["status"] == "OK":
        p50s = [r["ttft_p50"] for r in rep_list if r.get("ttft_p50") is not None]
        if p50s:
            rec["ttft_p50_reps"] = p50s
            rec["ttft_p50_median"] = round(statistics.median(p50s), 3)
            rec["ttft_p50_cv_pct"] = (
                round(100 * statistics.pstdev(p50s) / statistics.mean(p50s), 1)
                if len(p50s) > 1 and statistics.mean(p50s) else 0.0
            )
        for key in ("ttft_p95", "ttft_p99", "tpot_p50", "tpot_p99", "e2e_p50", "out_tok_s"):
            vals = [r[key] for r in rep_list if r.get(key) is not None]
            rec[f"{key}_median"] = round(statistics.median(vals), 3) if vals else None
        last = rep_list[-1] if rep_list else {}
        rec["vision_tok_per_req"] = (last.get("total_input_vision_tokens") or 0) // max(NUM_PROMPTS, 1)
        rec["text_tok_per_req"]   = (last.get("total_input_text_tokens") or 0) // max(NUM_PROMPTS, 1)
        log(f"  {vid}: ttft_p50_median={rec.get('ttft_p50_median')}ms  "
            f"cv={rec.get('ttft_p50_cv_pct')}%  "
            f"tpot={rec.get('tpot_p50_median')}ms")

    return rec


def write_summary(results, provenance):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ok_variants = [r for r in results if r.get("status") == "OK"]

    def get_rec(vid):
        return next((r for r in results if r["id"] == vid), {})

    s0    = get_rec("IMG_A_S0_ipc")
    s2    = get_rec("IMG_A_S2_ipc_pcg")
    s0r   = get_rec("IMG_A_S0_ipc_repeat")
    vllm  = get_rec("IMG_A_V0_vllm")
    noipc = get_rec("IMG_A_S0_noipc")

    lines = [
        f"# IMG-A Benchmark Summary — image+text (c=1) — fixed-generator path\n",
        f"> Run: {ts}  GPU={GPU}  seed={SEED}  n={NUM_PROMPTS}  warmup={WARMUP}  "
        f"reps={REPS}  resolution=720p  range_ratio=1.0\n",
        "> SGLang image headline baseline: `SGLANG_USE_CUDA_IPC_TRANSPORT=1` (IPC on).\n",
        "> IPC benefit and PCG benefit reported separately.\n",
        "> vLLM is anchor only — no causal inference.\n",
        "> **Image+text conclusions are separate from text-only Issue #2 findings.**\n",
    ]

    # Provenance block
    lines += [
        "## Fixed-SGLang provenance\n",
        f"- `/data/sglang-pr` HEAD SHA: `{provenance.get('sglang_pr_head_sha')}`",
        f"- `/data/sglang-pr` branch: `{provenance.get('sglang_pr_branch')}`",
        f"- Merged fix `07f326c184` in history: **{provenance.get('merged_fix_in_history')}**",
        f"- `sglang.__file__`: `{provenance.get('sglang_dunder_file')}`",
        f"- `sglang.benchmark.datasets.common.__file__`: `{provenance.get('common_dunder_file')}`",
        f"- Fix marker (`get_available_multimodal_text_tokens` in `gen_mm_prompt`): "
        f"`{provenance.get('fix_marker')}`",
        f"- Fixed-path import gate: **{provenance.get('fixed_path_ok')}**\n",
    ]

    # Headline table
    lines += [
        "## Headline numbers (TTFT p50, median of reps)\n",
        "| variant | ipc | pcg | ttft_p50 median (ms) | CV% | tpot_p50 (ms) | out_tok/s | status |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        status = r.get("status", "?")
        ipc = "on" if r.get("ipc_on") else "off"
        pcg = "on" if "--enforce-piecewise-cuda-graph" in r.get("extra_flags", []) else "off"
        lines.append(
            f"| {r['id']} | {ipc} | {pcg} "
            f"| {r.get('ttft_p50_median','FAIL')} "
            f"| {r.get('ttft_p50_cv_pct','?')}% "
            f"| {r.get('tpot_p50_median','?')} "
            f"| {r.get('out_tok_s_median','?')} "
            f"| {status} |"
        )
    lines.append("")

    # Bracket drift
    lines += ["## Bracket drift (S0_ipc vs S0_ipc_repeat)\n"]
    s0_m  = s0.get("ttft_p50_median")
    s0r_m = s0r.get("ttft_p50_median")
    if s0_m and s0r_m:
        drift_pct = abs(s0r_m - s0_m) / s0_m * 100
        drift_ok = drift_pct <= 5.0
        lines.append(
            f"S0_ipc: {s0_m} ms  |  S0_ipc_repeat: {s0r_m} ms  |  "
            f"drift={drift_pct:.1f}%  |  "
            f"{'✅ ≤5%' if drift_ok else '⚠️ >5% — downgrade absolute numbers to indicative'}"
        )
    else:
        lines.append("⚠ One or both bracket variants failed — drift cannot be assessed.")
    lines.append("")

    # PCG benefit (Q2)
    lines += ["## PCG benefit (Q2): S0_ipc vs S2_ipc_pcg\n"]
    s0_m    = s0.get("ttft_p50_median")
    s2_m    = s2.get("ttft_p50_median")
    s0_tpot = s0.get("tpot_p50_median")
    s2_tpot = s2.get("tpot_p50_median")
    if s0_m and s2_m:
        pcg_delta_pct = (s0_m - s2_m) / s0_m * 100
        tpot_ok = (s2_tpot is None or s0_tpot is None or s2_tpot <= s0_tpot * 1.05)
        pcg_benefit = pcg_delta_pct >= 5.0 and tpot_ok and s2.get("status") == "OK"
        lines.append(
            f"S0_ipc TTFT: {s0_m} ms  |  S2_ipc_pcg TTFT: {s2_m} ms  |  "
            f"delta={pcg_delta_pct:.1f}%  |  "
            f"TPOT S0={s0_tpot} S2={s2_tpot} (worse_ok={tpot_ok})"
        )
        lines.append(f"**PCG benefit on image+IPC baseline: {'YES' if pcg_benefit else 'NO'}**")
        if pcg_benefit:
            lines.append(
                f"  (Rule: ≥5% TTFT improvement with TPOT not worse. "
                f"Δ={pcg_delta_pct:.1f}% meets threshold.)"
            )
        else:
            if pcg_delta_pct < 5.0:
                lines.append(f"  (Δ={pcg_delta_pct:.1f}% < 5% threshold — no material benefit.)")
            elif not tpot_ok:
                lines.append(f"  (TPOT degraded — benefit condition not met.)")
    else:
        lines.append("⚠ S0_ipc or S2_ipc_pcg failed — PCG benefit cannot be assessed.")
    lines.append("")

    # IPC benefit (Q3)
    lines += ["## IPC benefit (Q3): S0_noipc vs S0_ipc\n"]
    noipc_m = noipc.get("ttft_p50_median")
    ipc_m   = s0.get("ttft_p50_median")
    if noipc_m and ipc_m:
        ipc_delta_pct = (noipc_m - ipc_m) / noipc_m * 100
        ipc_benefit = abs(ipc_delta_pct) >= 5.0 and noipc.get("status") == "OK"
        direction = "IPC reduces TTFT" if ipc_delta_pct > 0 else "IPC increases TTFT"
        lines.append(
            f"S0_noipc TTFT: {noipc_m} ms  |  S0_ipc TTFT: {ipc_m} ms  |  "
            f"Δ={ipc_delta_pct:.1f}% ({direction})"
        )
        lines.append(f"**IPC benefit on image+text: {'YES' if ipc_benefit else 'NO'}**")
        if not ipc_benefit:
            lines.append(
                f"  (Δ={abs(ipc_delta_pct):.1f}% "
                f"{'< 5% threshold' if abs(ipc_delta_pct) < 5 else '— TPOT or failure issue'}.)"
            )
    else:
        lines.append("⚠ S0_ipc or S0_noipc failed — IPC benefit cannot be assessed.")
    lines.append("")

    # SGLang vs vLLM gap (Q1)
    lines += ["## SGLang IPC baseline vs vLLM anchor (Q1)\n"]
    vllm_m = vllm.get("ttft_p50_median")
    s0_m   = s0.get("ttft_p50_median")
    if s0_m and vllm_m:
        gap_pct = (s0_m - vllm_m) / vllm_m * 100
        lines.append(
            f"SGLang S0_ipc: {s0_m} ms  |  vLLM V0: {vllm_m} ms  |  "
            f"gap={gap_pct:.1f}% ({'SGLang slower' if gap_pct > 0 else 'SGLang faster'})"
        )
        lines.append("(vLLM is anchor only — not a causal claim about SGLang mechanisms.)")
    else:
        lines.append("⚠ S0_ipc or V0_vllm failed.")
    lines.append("")

    # Token composition
    lines += ["## Token composition (per request)\n"]
    ref = s0 if s0.get("status") == "OK" else (ok_variants[0] if ok_variants else {})
    lines.append(
        f"Vision tokens: {ref.get('vision_tok_per_req','?')}/req  |  "
        f"Text tokens: {ref.get('text_tok_per_req','?')}/req  |  "
        f"Resolution: 720p, image-count=1, range_ratio=1.0, seed={SEED}"
    )
    lines.append("")

    # Failures
    lines += ["## Failure summary\n"]
    total_fail = sum(1 for r in results if r.get("status") != "OK")
    if total_fail == 0:
        lines.append("✅ 0 failures across all variants and reps.")
    else:
        lines.append(f"❌ {total_fail} variants with issues:")
        for r in results:
            if r.get("status") != "OK":
                line = f"  - {r['id']}: status={r.get('status')}"
                if r.get("forbidden_token_error_present"):
                    line += " ⚠️ forbidden-token error returned (fix path broken)"
                lines.append(line)
    lines.append("")

    # Next steps
    lines += ["## Recommendation\n"]
    if total_fail == 0:
        lines.append(
            "All IMG-A variants completed cleanly on the fixed-generator path. "
            "Review results above, then decide on IMG-B / IMG-C:\n"
            "- **IMG-B**: increase text to ~512 tok, check if PCG/IPC benefit holds at longer context.\n"
            "- **IMG-C**: c=16 batched — expect no PCG benefit (Case-C analog).\n"
            "- Requires explicit approval before proceeding."
        )
    else:
        lines.append("IMG-A had failures. Investigate before proceeding to IMG-B / IMG-C.")

    return "\n".join(lines)


def main():
    os.chdir(LAB)
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log("=== Stage 4.2 IMG-A Formal Preflight (fixed-generator path) ===")
    log(f"GPU: {GPU}  (user-specified; not auto-switch)")

    u = gpu_used()
    log(f"GPU {GPU} memory: {u} MiB  (idle threshold: {GPU_IDLE_MIB} MiB)")
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"STOP: GPU {GPU} not idle (used={u} MiB). Aborting.")
        sys.exit(1)

    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        if k in os.environ:
            log(f"STOP: {k} is set — forbidden for clean runs. Aborting.")
            sys.exit(1)
    log("KAPI env: clean")

    # Fixed-SGLang import gate
    prov = collect_provenance()
    log(f"sglang.__file__: {prov.get('sglang_dunder_file')}")
    log(f"common.__file__: {prov.get('common_dunder_file')}")
    log(f"fix marker: {prov.get('fix_marker')}")
    log(f"sglang-pr HEAD: {prov.get('sglang_pr_head_sha')} (branch {prov.get('sglang_pr_branch')})")
    log(f"merged fix 07f326c184 in history: {prov.get('merged_fix_in_history')}")
    if not prov.get("fixed_path_ok"):
        log("STOP: fixed-SGLang import gate failed. "
            "PYTHONPATH does not point at /data/sglang-pr or fix not in gen_mm_prompt.")
        log(f"stderr tail: {prov.get('probe_stderr_tail')}")
        sys.exit(1)
    log("Fixed-SGLang import gate: OK")

    log(f"Snapshot: {SNAPSHOT.split('/')[-1]}")
    log(f"Variants: {[v[0] for v in IMG_A_VARIANTS]}")
    log(f"n={NUM_PROMPTS}  warmup={WARMUP}  reps={REPS}  c={CONCURRENCY}")
    log(f"Image: 720p random png seed={SEED} count=1 input=128 output=128 range_ratio=1.0")
    log("IPC discipline: ON for S0_ipc, S2_ipc_pcg, S0_ipc_repeat; OFF (unset) for S0_noipc; N/A for V0_vllm.")
    log("")

    out_path = RESULTS / "imgA_results.json"
    results = []
    all_ok = True

    for vid, fw, ipc_on, extra_flags, wait_s in IMG_A_VARIANTS:
        rec = run_variant(vid, fw, ipc_on, extra_flags, wait_s)
        results.append(rec)
        out_path.write_text(json.dumps({"provenance": prov, "variants": results}, indent=2))
        if rec.get("status") == "GPU_NOT_IDLE":
            log("STOP: GPU not idle — aborting IMG-A.")
            all_ok = False
            break
        if rec.get("status") != "OK":
            log(f"STOP: {vid} status={rec.get('status')} — stopping; remaining variants skipped.")
            all_ok = False
            break

    out_path.write_text(json.dumps({"provenance": prov, "variants": results}, indent=2))
    log(f"\nimgA_results.json written: {out_path}")

    summary = write_summary(results, prov)
    summary_path = RESULTS / "imgA_summary.md"
    summary_path.write_text(summary)
    log(f"imgA_summary.md written: {summary_path}")

    if all_ok:
        log("\n=== IMG-A COMPLETE: all variants OK ===")
        sys.exit(0)
    else:
        log("\n=== IMG-A INCOMPLETE: see imgA_summary.md ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
