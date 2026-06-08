#!/usr/bin/env python3
"""
v2 / Issue #4 — PCG capture-stream debug runner v2 (E1–E4).

Follow-up to the D1–D4 driver (`run_pcg_capture_stream_debug.py`). That runner
hard-coded a tiny `num_prompts=2 warmup=0 output_len=32` matrix and used
`--dataset-name random` for the text-only case, which failed with
`huggingface_hub.errors.LocalEntryNotFoundError` under `HF_HUB_OFFLINE=1`. This
v2 runner:

  - parameterises per-case `num_prompts`, `warmup`, `output_len`,
    `dataset_kind`, and (for text autobench) `dataset_path`;
  - adds a `text_autobench` case kind that drives
    `--dataset-name autobench --dataset-path <local jsonl>` so the bench
    client never touches the HF Hub (this is the same path Issue #2 used);
  - keeps the same fixed-generator import gate, GPU-7 pinning,
    KAPI/profiler discipline, IPC-on-only-for-IPC-cases, fresh-server-per-
    case lifecycle, early `proc.poll()` exit detection, server-log scanning
    for the PCG assertion, and the 5-way classifier as v1.

Case kinds supported:
  - `text_autobench`  → text-only; `--dataset-name autobench --dataset-path
                        datasets/qwen3vl8b/caseA_short.jsonl`
                        (offline-safe; mirrors #2 Case A)
  - `image`           → image+text; `--dataset-name image
                        --image-count 1 --image-resolution 720p
                        --image-content random --image-format png`

This file is the working tool for E1–E4 per
`debug_pcg_capture_stream/next_debug_plan.md`. The old runner stays
untouched as the D1–D4 historical driver.

Provenance recorded per case:
  - `/data/sglang-pr` HEAD SHA + branch + merged-fix ancestor check
    (`07f326c184`)
  - runtime `sglang.__file__`, `common.__file__`, `FIX_OK` marker
  - exact server command line, env signals (IPC, KAPI, PYTHONPATH prefix)
  - exact bench command line, dataset args, case spec
  - GPU id, model snapshot, timestamp
  - classification + server-log excerpt around the failure region if any

Outputs go under
`debug_pcg_capture_stream/results/E<N>_<label>_results.json` /
`debug_pcg_capture_stream/results/E<N>_<label>_summary.md` (committed) and
`debug_pcg_capture_stream/results/raw/E<N>_<label>_bench.jsonl` /
`logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/
E<N>_<label>_server.log` (NOT committed unless approved).

Usage:
    # E1 — text-only PCG control (the only built-in stage as of writing).
    python3 run_pcg_capture_stream_debug_v2.py E1

Other E-stages will be added to CASE_SPECS as separate dict entries once
explicitly approved.
"""
import json
import os
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
RESULTS      = BASE / "debug_pcg_capture_stream" / "results"
RAW          = RESULTS / "raw"
LOGS         = LAB / "logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream"
SGLANG_PR    = Path("/data/sglang-pr")
SGLANG_PR_PY = str(SGLANG_PR / "python")
GPU = "7"  # user-specified; never auto-switch
SGLANG_PORT  = 30000
GPU_IDLE_MIB = 2000
SEED = 1

# Canonical offline-safe text dataset. Same JSONL Issue #2 Case A used.
CASE_A_DATASET = LAB / "datasets/qwen3vl8b/caseA_short.jsonl"

# Case-classifier signatures.
PCG_ASSERT_SIGN = "PCG capture stream is not set"
FORBIDDEN_TOKEN_SIGN = "No data iterator found for token"


def _make_base_env():
    """Pin GPU, offline HF, strip KAPI + IPC; prepend fixed-SGLang to PYTHONPATH."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
              "SGLANG_USE_CUDA_IPC_TRANSPORT"):
        env.pop(k, None)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SGLANG_PR_PY + (os.pathsep + prev if prev else "")
    return env

_BASE_ENV = _make_base_env()


# ----- Case specifications. Add new entries here for E2/E3/E4. -----
# Each entry: dict with keys
#   stage_id, label, dataset_kind, ipc_on, pcg_on,
#   num_prompts, warmup, output_len, server_wait_s,
#   expected_classification
# For text_autobench cases: dataset_path is required.
# For image cases: image_resolution / image_count / image_format /
#                  image_content defaults match Stage 4.2 IMG-A unless
#                  overridden.

CASE_SPECS = {
    "E1": {
        "stage_id": "E1",
        "label": "text_autobench_PCG_control",
        "description": (
            "Text-only Case A-like + PCG on /data/sglang-pr upstream main. "
            "Confirms whether upstream PCG itself regresses on text-only "
            "Qwen3-VL. Uses --dataset-name autobench with a local JSONL so "
            "the bench client never touches the HF Hub (replaces the D4 "
            "--dataset-name random path that failed in offline mode)."
        ),
        "dataset_kind": "text_autobench",
        "dataset_path": str(CASE_A_DATASET),
        "ipc_on": False,
        "pcg_on": True,
        "num_prompts": 8,
        "warmup": 0,
        "output_len": 128,
        "server_wait_s": 360,  # PCG capture loop typically ~25s + startup ~30s
        "expected_classification": "OK",
    },
    # ----- E2 ladder: image + IPC + PCG, growing sample size -----
    # Same recipe as Stage 4.2 IMG_A_S2_ipc_pcg except num_prompts. The first
    # stage that classifies PCG_CAPTURE_STREAM_ASSERT is the minimal repro
    # size. expected_classification is left as OK so a non-OK outcome flips
    # the runner's exit code to 1 (signal to stop the ladder).
    "E2a": {
        "stage_id": "E2a",
        "label": "image_IPC_PCG_n32",
        "description": (
            "Image+IPC+PCG ladder step 1 (n=32). Probes whether the Stage 4.2 "
            "PCG capture-stream assertion reproduces at a small sample size."
        ),
        "dataset_kind": "image",
        "ipc_on": True,
        "pcg_on": True,
        "num_prompts": 32,
        "warmup": 30,
        "output_len": 128,
        "input_len": 128,
        "image_resolution": "720p",
        "image_content": "random",
        "image_format": "png",
        "image_count": 1,
        "range_ratio": 1.0,
        "server_wait_s": 480,
        "expected_classification": "OK",
    },
    "E2b": {
        "stage_id": "E2b",
        "label": "image_IPC_PCG_n64",
        "description": (
            "Image+IPC+PCG ladder step 2 (n=64). Run only if E2a is OK."
        ),
        "dataset_kind": "image",
        "ipc_on": True,
        "pcg_on": True,
        "num_prompts": 64,
        "warmup": 30,
        "output_len": 128,
        "input_len": 128,
        "image_resolution": "720p",
        "image_content": "random",
        "image_format": "png",
        "image_count": 1,
        "range_ratio": 1.0,
        "server_wait_s": 480,
        "expected_classification": "OK",
    },
    "E2c": {
        "stage_id": "E2c",
        "label": "image_IPC_PCG_n100",
        "description": (
            "Image+IPC+PCG ladder step 3 (n=100). Run only if E2b is OK."
        ),
        "dataset_kind": "image",
        "ipc_on": True,
        "pcg_on": True,
        "num_prompts": 100,
        "warmup": 30,
        "output_len": 128,
        "input_len": 128,
        "image_resolution": "720p",
        "image_content": "random",
        "image_format": "png",
        "image_count": 1,
        "range_ratio": 1.0,
        "server_wait_s": 480,
        "expected_classification": "OK",
    },
    "E2d": {
        "stage_id": "E2d",
        "label": "image_IPC_PCG_n400",
        "description": (
            "Image+IPC+PCG ladder step 4 (n=400). Exact Stage 4.2 IMG-A "
            "S2_ipc_pcg config except this is a single attempt (no rep loop). "
            "Run only if E2c is OK."
        ),
        "dataset_kind": "image",
        "ipc_on": True,
        "pcg_on": True,
        "num_prompts": 400,
        "warmup": 30,
        "output_len": 128,
        "input_len": 128,
        "image_resolution": "720p",
        "image_content": "random",
        "image_format": "png",
        "image_count": 1,
        "range_ratio": 1.0,
        "server_wait_s": 480,
        "expected_classification": "OK",
    },
    # ----- E3: same shape as E2a but IPC off. Tests whether IPC is required -----
    # to trigger the assertion. Matched-size control for E2a (image + PCG +
    # n=32 + warmup=30 + output=128). expected_classification is left as OK
    # so that any non-OK outcome flips the runner exit code.
    "E3": {
        "stage_id": "E3",
        "label": "image_noIPC_PCG_n32",
        "description": (
            "Image+PCG+n32, but SGLANG_USE_CUDA_IPC_TRANSPORT unset. Matched "
            "shape control for E2a; if E3 also asserts, IPC is not a "
            "required trigger and the fault is VLM image + PCG."
        ),
        "dataset_kind": "image",
        "ipc_on": False,
        "pcg_on": True,
        "num_prompts": 32,
        "warmup": 30,
        "output_len": 128,
        "input_len": 128,
        "image_resolution": "720p",
        "image_content": "random",
        "image_format": "png",
        "image_count": 1,
        "range_ratio": 1.0,
        "server_wait_s": 480,
        "expected_classification": "OK",
    },
}


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def gpu_used():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", GPU],
        capture_output=True, text=True)
    try:
        return int(r.stdout.strip().splitlines()[0])
    except Exception:
        return -1


def wait_server(port, timeout, proc=None):
    """Wait for /health. Fail fast on proc.poll() != None."""
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
        time.sleep(3)
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


def server_log_excerpt(path, max_lines=20):
    """Return up to `max_lines` lines centered on the most relevant signature,
    else the last `max_lines` lines."""
    try:
        text = Path(path).read_text()
    except Exception as ex:
        return f"(log unreadable: {ex})"
    lines = text.splitlines()
    for sign in (PCG_ASSERT_SIGN, "AssertionError", "Exception:", "Error:"):
        for i, ln in enumerate(lines):
            if sign in ln:
                lo = max(0, i - max_lines // 2)
                hi = min(len(lines), i + max_lines // 2)
                return "\n".join(lines[lo:hi])
    return "\n".join(lines[-max_lines:])


def collect_provenance():
    """Per-case provenance: SGLang HEAD, runtime sglang.__file__, fix marker."""
    out = {
        "sglang_pr_path": str(SGLANG_PR),
        "pythonpath_prefix": SGLANG_PR_PY,
    }
    try:
        out["sglang_pr_head_sha"] = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception as ex:
        out["sglang_pr_head_sha"] = f"ERROR:{ex}"
    try:
        out["sglang_pr_branch"] = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
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
    out["fixed_path_ok"] = (
        out["sglang_dunder_file"] is not None
        and SGLANG_PR_PY in out["sglang_dunder_file"]
        and out["fix_marker"] == "FIX_OK"
    )
    return out


def build_bench_args(spec):
    """Build the dataset+shape arguments for sglang.bench_serving from a spec.

    The same num_prompts / warmup / seed / max-concurrency / extra-body args
    are used regardless of dataset kind; only the dataset-name / dataset-path /
    image-* / random-input-len differ.
    """
    common = [
        "--max-concurrency",    "1",
        "--num-prompts",        str(spec["num_prompts"]),
        "--warmup-requests",    str(spec["warmup"]),
        "--seed",               str(SEED),
        "--extra-request-body", '{"temperature": 0, "top_p": 1}',
        "--output-details",
    ]
    kind = spec["dataset_kind"]
    if kind == "text_autobench":
        dataset_path = spec.get("dataset_path") or str(CASE_A_DATASET)
        ds_args = [
            "--dataset-name", "autobench",
            "--dataset-path", dataset_path,
        ]
    elif kind == "image":
        ds_args = [
            "--dataset-name",       "image",
            "--image-count",        str(spec.get("image_count", 1)),
            "--image-resolution",   spec.get("image_resolution", "720p"),
            "--image-format",       spec.get("image_format", "png"),
            "--image-content",      spec.get("image_content", "random"),
            "--random-input-len",   str(spec.get("input_len", 128)),
            "--random-output-len",  str(spec["output_len"]),
            "--random-range-ratio", str(spec.get("range_ratio", 1.0)),
        ]
    else:
        raise ValueError(f"unknown dataset_kind: {kind!r}")
    return ds_args + common


def run_case(spec):
    stage_id = spec["stage_id"]
    label = spec["label"]
    log(f"\n{'='*60}")
    log(f"STAGE: {stage_id}  {label}")
    log(f"  dataset_kind={spec['dataset_kind']}  "
        f"IPC={'on' if spec['ipc_on'] else 'off'}  "
        f"PCG={'on' if spec['pcg_on'] else 'off'}  "
        f"num_prompts={spec['num_prompts']}  warmup={spec['warmup']}  "
        f"output_len={spec['output_len']}")
    log(f"{'='*60}")

    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"  STOP: GPU {GPU} not idle (used={u} MiB)")
        return {
            "stage_id": stage_id, "label": label, "spec": spec,
            "classification": "GPU_NOT_IDLE", "gpu_mib_at_start": u,
        }

    case_env = {**_BASE_ENV}
    if spec["ipc_on"]:
        case_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"
    else:
        case_env.pop("SGLANG_USE_CUDA_IPC_TRANSPORT", None)
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        case_env.pop(k, None)

    extra_flags = ["--enforce-piecewise-cuda-graph"] if spec["pcg_on"] else []
    srv_cmd = (["python3", "-m", "sglang.launch_server",
                "--model-path", SNAPSHOT, "--dtype", "bfloat16",
                "--port", str(SGLANG_PORT), "--tp", "1",
                "--attention-backend", "flashinfer"] + extra_flags)
    patt = "sglang.launch_server"

    LOGS.mkdir(parents=True, exist_ok=True)
    server_log = LOGS / f"{stage_id}_{label}_server.log"
    lf = open(server_log, "w")
    log(f"  launching SGLang port={SGLANG_PORT} wait≤{spec['server_wait_s']}s ...")
    proc = subprocess.Popen(srv_cmd, env=case_env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {
        "stage_id": stage_id, "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU, "snapshot": SNAPSHOT.split("/")[-1],
        "spec": spec,
        "server_cmd": srv_cmd,
        "env_signals": {
            "SGLANG_USE_CUDA_IPC_TRANSPORT":
                case_env.get("SGLANG_USE_CUDA_IPC_TRANSPORT"),
            "SGLANG_KERNEL_API_LOGLEVEL":
                case_env.get("SGLANG_KERNEL_API_LOGLEVEL"),
            "SGLANG_KERNEL_API_LOGDEST":
                case_env.get("SGLANG_KERNEL_API_LOGDEST"),
            "PYTHONPATH_prefix":
                case_env.get("PYTHONPATH", "").split(os.pathsep)[0],
            "HF_HUB_OFFLINE": case_env.get("HF_HUB_OFFLINE"),
            "CUDA_VISIBLE_DEVICES": case_env.get("CUDA_VISIBLE_DEVICES"),
        },
        "kapi_logging": False, "profiler": False,
        "server_log_path": str(server_log),
    }

    try:
        ready = wait_server(SGLANG_PORT, spec["server_wait_s"], proc=proc)
        if not ready:
            log_text = ""
            try:
                log_text = server_log.read_text()
            except Exception:
                pass
            if PCG_ASSERT_SIGN in log_text:
                rec["classification"] = "PCG_CAPTURE_STREAM_ASSERT"
            else:
                rec["classification"] = "SERVER_NO_START"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
            log(f"  → classification={rec['classification']}")
            return rec

        log("  server up.")
        bench_args = build_bench_args(spec)
        out_jsonl = RAW / f"{stage_id}_{label}_bench.jsonl"
        RAW.mkdir(parents=True, exist_ok=True)
        out_jsonl.unlink(missing_ok=True)
        cmd = (
            ["python3", "-m", "sglang.bench_serving",
             "--backend", "sglang-oai-chat",
             "--base-url", f"http://127.0.0.1:{SGLANG_PORT}",
             "--model", SNAPSHOT]
            + bench_args
            + ["--output-file", str(out_jsonl)]
        )
        rec["bench_cmd"] = cmd
        log(f"  bench: dataset_kind={spec['dataset_kind']} num-prompts={spec['num_prompts']} "
            f"warmup={spec['warmup']} output_len={spec['output_len']} seed={SEED}")
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=case_env)
        elapsed = round(time.time() - t0, 1)
        rec["bench_rc"] = res.returncode
        rec["bench_elapsed_s"] = elapsed
        log(f"  bench rc={res.returncode} elapsed={elapsed}s")

        try:
            log_text = server_log.read_text()
        except Exception:
            log_text = ""

        if res.returncode != 0:
            stderr_tail = (res.stdout + res.stderr)[-500:]
            rec["bench_stdout_tail"] = stderr_tail
            if PCG_ASSERT_SIGN in log_text:
                rec["classification"] = "PCG_CAPTURE_STREAM_ASSERT"
            elif FORBIDDEN_TOKEN_SIGN in stderr_tail or FORBIDDEN_TOKEN_SIGN in log_text:
                rec["classification"] = "FORBIDDEN_TOKEN_ERROR"
            else:
                rec["classification"] = "OTHER_FAILURE"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
            log(f"  → classification={rec['classification']}")
            return rec

        d = parse_bench_jsonl(out_jsonl)
        if d is None:
            rec["classification"] = "OTHER_FAILURE"
            rec["note"] = "bench rc=0 but JSON parse failed"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
            log(f"  → classification={rec['classification']}")
            return rec

        errors = d.get("errors", [])
        n_fail = sum(1 for e in errors if e)
        generated = d.get("generated_texts", [])
        has_output = any(bool(t and t.strip()) for t in generated)
        rec.update({
            "completed": d.get("completed"),
            "failures": n_fail,
            "has_non_empty_output": has_output,
            "median_ttft_ms": d.get("median_ttft_ms"),
            "total_input_tokens": d.get("total_input_tokens"),
            "total_input_text_tokens": d.get("total_input_text_tokens"),
            "total_input_vision_tokens": d.get("total_input_vision_tokens"),
            "total_output_tokens": d.get("total_output_tokens"),
            "generated_texts_sample": [t[:120] for t in generated[:2]],
            "errors_sample": [str(e) for e in errors[:2] if e],
        })

        if PCG_ASSERT_SIGN in log_text:
            rec["classification"] = "PCG_CAPTURE_STREAM_ASSERT"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
            rec["note"] = "bench reported success but server log contains PCG assert"
        elif any(FORBIDDEN_TOKEN_SIGN in str(e) for e in errors if e):
            rec["classification"] = "FORBIDDEN_TOKEN_ERROR"
        elif n_fail > 0:
            rec["classification"] = "OTHER_FAILURE"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
        elif not has_output:
            rec["classification"] = "OTHER_FAILURE"
            rec["note"] = "no non-empty output"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
        else:
            rec["classification"] = "OK"

        log(f"  → classification={rec['classification']}")
        return rec

    finally:
        kill_server(proc, patt)
        lf.close()


def write_summary(stage_id, rec, provenance):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    spec = rec.get("spec", {})
    lines = [
        f"# PCG debug — {stage_id} ({spec.get('label','')})\n",
        f"> Run: {ts}  GPU={GPU}  seed={SEED}\n",
        f"> Description: {spec.get('description','')}\n",
    ]

    lines += [
        "## Fixed-SGLang provenance\n",
        f"- `/data/sglang-pr` HEAD SHA: `{provenance.get('sglang_pr_head_sha')}`",
        f"- `/data/sglang-pr` branch: `{provenance.get('sglang_pr_branch')}`",
        f"- Merged fix `07f326c184` in history: **{provenance.get('merged_fix_in_history')}**",
        f"- `sglang.__file__`: `{provenance.get('sglang_dunder_file')}`",
        f"- `sglang.benchmark.datasets.common.__file__`: `{provenance.get('common_dunder_file')}`",
        f"- Fix marker: `{provenance.get('fix_marker')}`",
        f"- Fixed-path import gate: **{provenance.get('fixed_path_ok')}**\n",
    ]

    cls = rec.get("classification", "?")
    expected = spec.get("expected_classification", "?")
    match = "✅" if cls == expected else "⚠️"
    lines += [
        "## Verdict\n",
        f"- Classification: **`{cls}`** (expected `{expected}`) {match}",
        f"- bench_rc: {rec.get('bench_rc','n/a')} | "
        f"elapsed: {rec.get('bench_elapsed_s','n/a')} s | "
        f"server_wait_s: {spec.get('server_wait_s','n/a')}",
    ]
    if rec.get("completed") is not None:
        lines.append(
            f"- completed={rec.get('completed')} fails={rec.get('failures')} "
            f"non_empty={rec.get('has_non_empty_output')} "
            f"median_ttft_ms={rec.get('median_ttft_ms')}")
    if rec.get("env_signals"):
        es = rec["env_signals"]
        lines.append(
            "- env signals: "
            f"SGLANG_USE_CUDA_IPC_TRANSPORT={es.get('SGLANG_USE_CUDA_IPC_TRANSPORT')!r}  "
            f"KAPI_LOGLEVEL={es.get('SGLANG_KERNEL_API_LOGLEVEL')!r}  "
            f"KAPI_LOGDEST={es.get('SGLANG_KERNEL_API_LOGDEST')!r}  "
            f"HF_HUB_OFFLINE={es.get('HF_HUB_OFFLINE')!r}  "
            f"PYTHONPATH_prefix={es.get('PYTHONPATH_prefix')!r}  "
            f"CUDA_VISIBLE_DEVICES={es.get('CUDA_VISIBLE_DEVICES')!r}")
    if rec.get("note"):
        lines.append(f"- note: {rec['note']}")
    lines.append("")

    # Spec & commands
    lines += [
        "## Case spec\n",
        "```json",
        json.dumps(spec, indent=2, default=str),
        "```",
        "",
    ]
    if rec.get("server_cmd"):
        lines += ["## Server command\n", "```bash",
                  " ".join(rec["server_cmd"]),
                  "```", ""]
    if rec.get("bench_cmd"):
        lines += ["## Bench command\n", "```bash",
                  " ".join(rec["bench_cmd"]),
                  "```", ""]

    if rec.get("generated_texts_sample"):
        lines += ["## Sample bench output\n"]
        for i, txt in enumerate(rec["generated_texts_sample"][:2]):
            lines.append(f"- req{i+1}: `{txt}`")
        lines.append("")

    if rec.get("errors_sample"):
        lines += ["## Bench errors sample\n"]
        for e in rec["errors_sample"]:
            lines.append(f"- `{e[:300]}`")
        lines.append("")

    if rec.get("server_log_excerpt"):
        lines += [
            "## Server log excerpt (failure region or tail)\n",
            "```text",
            rec["server_log_excerpt"],
            "```",
            "",
        ]

    return "\n".join(lines)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CASE_SPECS:
        avail = ", ".join(sorted(CASE_SPECS))
        print(f"usage: {sys.argv[0]} <stage_id>")
        print(f"available stage_ids: {avail}")
        sys.exit(2)

    stage_id = sys.argv[1]
    spec = CASE_SPECS[stage_id]

    os.chdir(LAB)
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log(f"=== PCG debug v2 — {stage_id} preflight ===")
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

    prov = collect_provenance()
    log(f"sglang.__file__: {prov.get('sglang_dunder_file')}")
    log(f"common.__file__: {prov.get('common_dunder_file')}")
    log(f"fix marker: {prov.get('fix_marker')}")
    log(f"sglang-pr HEAD: {prov.get('sglang_pr_head_sha')} (branch {prov.get('sglang_pr_branch')})")
    log(f"merged fix 07f326c184 in history: {prov.get('merged_fix_in_history')}")
    if not prov.get("fixed_path_ok"):
        log("STOP: fixed-SGLang import gate failed. Aborting.")
        sys.exit(1)
    log("Fixed-SGLang import gate: OK")

    # Sanity: if the spec uses text_autobench, the JSONL must exist locally.
    if spec["dataset_kind"] == "text_autobench":
        dp = Path(spec.get("dataset_path") or str(CASE_A_DATASET))
        if not dp.is_file():
            log(f"STOP: text_autobench dataset path does not exist: {dp}")
            sys.exit(1)
        log(f"Dataset path: {dp} (size={dp.stat().st_size} bytes)")
    log(f"Snapshot: {SNAPSHOT.split('/')[-1]}")
    log(f"Stage: {stage_id}  label={spec['label']}")
    log("")

    rec = run_case(spec)

    results_path = RESULTS / f"{stage_id}_{spec['label']}_results.json"
    results_path.write_text(json.dumps(
        {"provenance": prov, "case": rec}, indent=2, default=str))
    log(f"\n{results_path.name} written: {results_path}")

    summary = write_summary(stage_id, rec, prov)
    summary_path = RESULTS / f"{stage_id}_{spec['label']}_summary.md"
    summary_path.write_text(summary)
    log(f"{summary_path.name} written: {summary_path}")

    cls = rec.get("classification")
    log(f"\n=== {stage_id} DONE — classification={cls} ===")

    # Exit nonzero only if the stage failed in a way the user needs to act on.
    if cls != spec.get("expected_classification"):
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
