#!/usr/bin/env python3
"""
v2 / Issue #4 — PCG capture-stream debug runner (D1–D4).

Isolates the failure mode behind the Stage 4.2 IMG-A `IMG_A_S2_ipc_pcg` crash:

  File "/data/sglang-pr/python/sglang/srt/compilation/cuda_piecewise_backend.py",
       line 171, in __call__
      stream is not None
  AssertionError: PCG capture stream is not set, please check if runtime
                  recompilation happened

Drives a tiny 4-case matrix (≤ 2 requests / case) against the fixed-generator
path (`PYTHONPATH=/data/sglang-pr/python`), with each case running a fresh
SGLang server and killing it before the next case. The per-case classifier
labels each outcome as exactly one of:

  - OK                          — bench finished, 0 failures, non-empty output.
  - PCG_CAPTURE_STREAM_ASSERT   — server log contains the cuda_piecewise_backend
                                  capture-stream AssertionError.
  - FORBIDDEN_TOKEN_ERROR       — bench errors contain
                                  "No data iterator found for token".
  - SERVER_NO_START             — server never reached /health within wait; no
                                  PCG assertion in log.
  - OTHER_FAILURE               — anything else (bench rc != 0 for a non-PCG
                                  reason, parse error, etc).

Cases (per experiment_plan.md):

  D1 — image + IPC on  + PCG on   (reproduce Stage 4.2 crash)
  D2 — image + IPC off + PCG on   (does IPC matter?)
  D3 — image + IPC on  + PCG off  (positive control; mirrors IMG_A_S0_ipc)
  D4 — text + PCG on              (does upstream PCG itself regress?)

Outputs (committed):
    experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/results/
        D1234_summary.md
        D1234_results.json

Logs and raw bench JSONL (NOT committed unless explicitly approved):
    experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/results/raw/
        D<N>_<case>_bench.jsonl
    logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/
        D<N>_<case>_server.log

Constraints:
  - GPU 7 only; never auto-switch.
  - PYTHONPATH=/data/sglang-pr/python (no sanitized monkeypatch; no
    /sgl-workspace/sglang imports).
  - SGLANG_KERNEL_API_LOGLEVEL / SGLANG_KERNEL_API_LOGDEST always unset
    (KAPI forbidden); no profiler.
  - SGLANG_USE_CUDA_IPC_TRANSPORT=1 only for IPC-on cases.
  - Tiny workload (num_prompts=2, warmup=0) — correctness probe, no perf.
  - Per-case `wait_server` polls `/health` AND `proc.poll()`; fails fast on
    early server death.
  - Never writes to `results/`, `results_fixed/`, `smoke/`, `smoke_fixed/`,
    `debug_video_pad/`, or any historical artifact.
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

# Tiny correctness-probe params
NUM_PROMPTS = 2
WARMUP      = 0

# Common bench client args for an image-dataset probe
IMAGE_ARGS = [
    "--dataset-name",       "image",
    "--image-count",        "1",
    "--image-resolution",   "720p",
    "--image-format",       "png",
    "--image-content",      "random",
    "--random-input-len",   "128",
    "--random-output-len",  "32",
    "--random-range-ratio", "1.0",
    "--max-concurrency",    "1",
    "--num-prompts",        str(NUM_PROMPTS),
    "--warmup-requests",    str(WARMUP),
    "--seed",               str(SEED),
    "--extra-request-body", '{"temperature": 0, "top_p": 1}',
    "--output-details",
]

# Common bench client args for a text-only random probe (no images at all)
TEXT_ARGS = [
    "--dataset-name",       "random",
    "--random-input-len",   "128",
    "--random-output-len",  "32",
    "--random-range-ratio", "1.0",
    "--max-concurrency",    "1",
    "--num-prompts",        str(NUM_PROMPTS),
    "--warmup-requests",    str(WARMUP),
    "--seed",               str(SEED),
    "--extra-request-body", '{"temperature": 0, "top_p": 1}',
    "--output-details",
]

# Case-specific signatures we look for in the server log to classify failures.
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


# Debug matrix. Each entry:
#   id, label, dataset_kind, ipc_on, pcg_on, server_wait_s, expected_classification
DEBUG_CASES = [
    ("D1", "image + IPC + PCG (reproduce 4.2 crash)",
     "image", True,  True,  300, "PCG_CAPTURE_STREAM_ASSERT"),
    ("D2", "image + no IPC + PCG (factor IPC out)",
     "image", False, True,  300, "PCG_CAPTURE_STREAM_ASSERT"),
    ("D3", "image + IPC + no PCG (positive control, mirrors IMG_A_S0_ipc)",
     "image", True,  False, 240, "OK"),
    ("D4", "text-only + PCG (does upstream PCG regress generally?)",
     "text",  False, True,  300, "OK"),
]


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
    """Return up to `max_lines` lines centered on the most relevant signature
    if present, else the last `max_lines` of the log."""
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


def run_case(case_id, label, dataset_kind, ipc_on, pcg_on, wait_s):
    log(f"\n{'='*60}")
    log(f"CASE: {case_id}  {label}")
    log(f"  dataset={dataset_kind}  IPC={'on' if ipc_on else 'off'}  "
        f"PCG={'on' if pcg_on else 'off'}  wait≤{wait_s}s")
    log(f"{'='*60}")

    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"  STOP: GPU {GPU} not idle (used={u} MiB)")
        return {
            "id": case_id, "label": label,
            "classification": "GPU_NOT_IDLE",
            "gpu_mib_at_start": u,
            "ipc_on": ipc_on, "pcg_on": pcg_on, "dataset_kind": dataset_kind,
        }

    case_env = {**_BASE_ENV}
    if ipc_on:
        case_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"
    else:
        case_env.pop("SGLANG_USE_CUDA_IPC_TRANSPORT", None)
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        case_env.pop(k, None)

    extra_flags = ["--enforce-piecewise-cuda-graph"] if pcg_on else []
    srv_cmd = (["python3", "-m", "sglang.launch_server",
                "--model-path", SNAPSHOT, "--dtype", "bfloat16",
                "--port", str(SGLANG_PORT), "--tp", "1",
                "--attention-backend", "flashinfer"] + extra_flags)

    LOGS.mkdir(parents=True, exist_ok=True)
    server_log = LOGS / f"{case_id}_server.log"
    lf = open(server_log, "w")
    log(f"  launching SGLang port={SGLANG_PORT} ...")
    proc = subprocess.Popen(srv_cmd, env=case_env, stdout=lf, stderr=subprocess.STDOUT)
    patt = "sglang.launch_server"

    rec = {
        "id": case_id, "label": label,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU, "snapshot": SNAPSHOT.split("/")[-1],
        "ipc_on": ipc_on, "pcg_on": pcg_on, "dataset_kind": dataset_kind,
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
        },
        "kapi_logging": False, "profiler": False,
        "wait_s": wait_s,
    }

    try:
        ready = wait_server(SGLANG_PORT, wait_s, proc=proc)
        if not ready:
            # Classify why the server is not ready.
            log_text = ""
            try:
                log_text = server_log.read_text()
            except Exception:
                pass
            if PCG_ASSERT_SIGN in log_text:
                rec["classification"] = "PCG_CAPTURE_STREAM_ASSERT"
            elif proc.poll() is not None:
                rec["classification"] = "SERVER_NO_START"
            else:
                rec["classification"] = "SERVER_NO_START"
            rec["server_log_excerpt"] = server_log_excerpt(server_log)
            log(f"  → classification={rec['classification']}")
            return rec

        log(f"  server up.")

        out_jsonl = RAW / f"{case_id}_bench.jsonl"
        out_jsonl.unlink(missing_ok=True)
        bench_args = IMAGE_ARGS if dataset_kind == "image" else TEXT_ARGS
        cmd = (
            ["python3", "-m", "sglang.bench_serving",
             "--backend", "sglang-oai-chat",
             "--base-url", f"http://127.0.0.1:{SGLANG_PORT}",
             "--model", SNAPSHOT]
            + bench_args
            + ["--output-file", str(out_jsonl)]
        )
        log(f"  bench: dataset={dataset_kind} num-prompts={NUM_PROMPTS} seed={SEED}")
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, env=case_env)
        elapsed = round(time.time() - t0, 1)
        rec["bench_rc"] = res.returncode
        rec["bench_elapsed_s"] = elapsed
        log(f"  bench rc={res.returncode} elapsed={elapsed}s")

        # Re-read the server log now that the bench has fired prefill, since the
        # PCG assertion may only appear on first request, not on server start.
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

        # PCG assertion may still appear in the server log even if the bench
        # client thinks it succeeded (e.g. fallback retried). Check explicitly.
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


def write_summary(results, provenance):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# PCG capture-stream debug — D1–D4 results\n",
        f"> Run: {ts}  GPU={GPU}  seed={SEED}  num_prompts={NUM_PROMPTS}  "
        f"warmup={WARMUP}\n",
        "> Tiny correctness probe — NO performance conclusions.\n",
    ]

    # Provenance
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
        "## Per-case results\n",
        "| case | label | dataset | IPC | PCG | classification | expected | match |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        ipc = "on" if r.get("ipc_on") else "off"
        pcg = "on" if r.get("pcg_on") else "off"
        cls = r.get("classification", "?")
        expected = r.get("expected_classification", "?")
        match = "✅" if cls == expected else "⚠️"
        lines.append(
            f"| {r['id']} | {r.get('label','')} | {r.get('dataset_kind','?')} "
            f"| {ipc} | {pcg} | `{cls}` | `{expected}` | {match} |"
        )
    lines.append("")

    # Decision matrix application
    cls_map = {r["id"]: r.get("classification") for r in results}
    d1 = cls_map.get("D1")
    d2 = cls_map.get("D2")
    d3 = cls_map.get("D3")
    d4 = cls_map.get("D4")

    lines += ["## Decision-matrix interpretation\n"]
    if d1 == "PCG_CAPTURE_STREAM_ASSERT" and d2 == "PCG_CAPTURE_STREAM_ASSERT" \
            and d3 == "OK" and d4 == "OK":
        lines.append(
            "**VLM + PCG specifically unsupported on this upstream main; "
            "IPC is not a factor.** D1+D2 reproduce the assertion with and "
            "without IPC; D3 confirms the no-PCG image path is healthy; D4 "
            "confirms upstream PCG itself is not broadly regressed.")
        lines.append("\nSuggested next: file upstream SGLang issue (extend HIP "
                     "fallback to CUDA OR loud warning on VLM + enforce-pcg). "
                     "Continue #4 without PCG.")
    elif d1 == "PCG_CAPTURE_STREAM_ASSERT" and d2 == "OK" \
            and d3 == "OK" and d4 == "OK":
        lines.append(
            "**Assertion requires IPC + PCG combination.** PCG alone on image "
            "is fine; adding IPC breaks it.")
        lines.append("\nSuggested next: file upstream issue scoped to "
                     "IPC transport + PCG interaction.")
    elif d1 == "PCG_CAPTURE_STREAM_ASSERT" and d4 == "PCG_CAPTURE_STREAM_ASSERT":
        lines.append(
            "**Broader upstream PCG regression on this HEAD** (not VLM-"
            "specific). Static-audit conclusion must be revised.")
        lines.append("\nSuggested next: file upstream issue about PCG "
                     "regression in general. Pause #4. Do not PR.")
    elif d1 == "OK":
        lines.append(
            "**D1 did not reproduce.** Stage 4.2 crash may be intermittent. "
            "Retry D1 with larger sample (e.g. 8 prompts) before drawing "
            "conclusions.")
    else:
        lines.append("**Outcome did not match any predefined matrix row.** See "
                     "per-case notes and re-audit before reporting.")
    lines.append("")

    # Per-case detail
    lines += ["## Per-case detail\n"]
    for r in results:
        lines.append(f"### {r['id']} — {r.get('label','')}\n")
        lines.append(f"- Classification: **{r.get('classification','?')}** "
                     f"(expected `{r.get('expected_classification','?')}`)")
        lines.append(f"- bench_rc: {r.get('bench_rc','n/a')} | "
                     f"elapsed: {r.get('bench_elapsed_s','n/a')} s | "
                     f"wait_s: {r.get('wait_s','n/a')}")
        if r.get("completed") is not None:
            lines.append(f"- completed={r.get('completed')} fails={r.get('failures')} "
                         f"non_empty={r.get('has_non_empty_output')} "
                         f"vision_tok={r.get('total_input_vision_tokens')} "
                         f"text_tok={r.get('total_input_text_tokens')}")
        if r.get("env_signals"):
            es = r["env_signals"]
            lines.append(
                f"- env: SGLANG_USE_CUDA_IPC_TRANSPORT={es.get('SGLANG_USE_CUDA_IPC_TRANSPORT')!r}  "
                f"KAPI_LOGLEVEL={es.get('SGLANG_KERNEL_API_LOGLEVEL')!r}  "
                f"KAPI_LOGDEST={es.get('SGLANG_KERNEL_API_LOGDEST')!r}  "
                f"PYTHONPATH_prefix={es.get('PYTHONPATH_prefix')!r}")
        if r.get("server_log_excerpt"):
            lines.append("- server log excerpt (head of failure region):")
            lines.append("```text")
            lines.append(r["server_log_excerpt"])
            lines.append("```")
        if r.get("errors_sample"):
            lines.append(f"- bench errors sample: {r['errors_sample']}")
        if r.get("note"):
            lines.append(f"- note: {r['note']}")
        if r.get("generated_texts_sample"):
            lines.append(f"- sample output[0]: `{r['generated_texts_sample'][0]}`")
        lines.append("")

    return "\n".join(lines)


def main():
    os.chdir(LAB)
    RAW.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log("=== PCG capture-stream debug — D1–D4 preflight ===")
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

    log(f"Snapshot: {SNAPSHOT.split('/')[-1]}")
    log(f"Cases: {[c[0] for c in DEBUG_CASES]}")
    log(f"Tiny probe: num_prompts={NUM_PROMPTS} warmup={WARMUP} c=1 seed={SEED}")
    log("")

    out_path = RESULTS / "D1234_results.json"
    results = []
    for cid, label, dk, ipc_on, pcg_on, wait_s, expected in DEBUG_CASES:
        rec = run_case(cid, label, dk, ipc_on, pcg_on, wait_s)
        rec["expected_classification"] = expected
        results.append(rec)
        out_path.write_text(json.dumps(
            {"provenance": prov, "cases": results}, indent=2))

    out_path.write_text(json.dumps(
        {"provenance": prov, "cases": results}, indent=2))
    log(f"\n{out_path.name} written: {out_path}")

    summary = write_summary(results, prov)
    summary_path = RESULTS / "D1234_summary.md"
    summary_path.write_text(summary)
    log(f"{summary_path.name} written: {summary_path}")

    # Exit: non-zero only if the gate (D3 positive control) fails or if any case
    # was OTHER_FAILURE / SERVER_NO_START outside the expected matrix. D1+D2 may
    # legitimately classify as PCG_CAPTURE_STREAM_ASSERT and are still expected.
    unexpected = []
    for r in results:
        cls = r.get("classification")
        if r["id"] == "D3" and cls != "OK":
            unexpected.append(r["id"])
        if cls in {"OTHER_FAILURE", "GPU_NOT_IDLE"}:
            unexpected.append(r["id"])
    if unexpected:
        log(f"\n=== DEBUG MATRIX: completed with unexpected outcomes in {unexpected} ===")
        sys.exit(1)
    log("\n=== DEBUG MATRIX: completed; see D1234_summary.md ===")
    sys.exit(0)


if __name__ == "__main__":
    main()
