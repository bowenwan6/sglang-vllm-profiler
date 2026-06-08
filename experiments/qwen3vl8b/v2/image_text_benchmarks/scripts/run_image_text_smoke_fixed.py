#!/usr/bin/env python3
"""
v2 / Issue #4 — Stage 4.1 fixed-generator smoke runner: image+text, 3 paths.

Drives `python -m sglang.bench_serving --dataset-name image --backend sglang-oai-chat`
against the **fixed** SGLang generator (upstream commit
`07f326c184 Fix multimodal synthetic benchmark prompt generation to exclude special
tokens (#26864)`), sourced from `/data/sglang-pr` on `main`. Selected via
`PYTHONPATH=/data/sglang-pr/python` — `/sgl-workspace/sglang` is NEVER imported by
this runner. No sanitized monkeypatch wrapper is used: the bench client and the
SGLang server both import the merged upstream fix.

Three smoke cases (tiny `--num-prompts 2`):
  - smoke_sglang_ipc     SGLang  `SGLANG_USE_CUDA_IPC_TRANSPORT=1`
  - smoke_sglang_noipc   SGLang  `SGLANG_USE_CUDA_IPC_TRANSPORT` unset
  - smoke_vllm_anchor    vLLM    (no IPC env), bench client still uses fixed SGLang

NO performance conclusions. Purpose: confirm the fixed generator path is wired up
end-to-end (server, bench client, vLLM anchor), 0 failures, GPU cleanup, no
forbidden multimodal special-token error.

Outputs (committed):
    experiments/qwen3vl8b/v2/image_text_benchmarks/smoke_fixed/smoke_results.json
    experiments/qwen3vl8b/v2/image_text_benchmarks/smoke_fixed/smoke_summary.md

Logs (NOT committed):
    logs/qwen3vl8b/v2/image_text_benchmarks/smoke_fixed/<case>_server.log
    smoke_fixed/<case>_bench.jsonl (raw per-case bench output; not committed)

STRICT CLEAN: never sets SGLANG_KERNEL_API_LOGLEVEL / SGLANG_KERNEL_API_LOGDEST;
no profiler.
GPU: 7 (fixed per user spec — never auto-switch).
Never overwrites or touches old smoke/, results/, results/raw/, debug_video_pad/,
or v1 paths.
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
LAB         = Path("/data/sglang-vllm-profiler")
BASE        = LAB / "experiments/qwen3vl8b/v2/image_text_benchmarks"
SMOKE_DIR   = BASE / "smoke_fixed"
LOGS        = LAB / "logs/qwen3vl8b/v2/image_text_benchmarks/smoke_fixed"
SGLANG_PR   = Path("/data/sglang-pr")                       # fixed SGLang clone
SGLANG_PR_PY = str(SGLANG_PR / "python")                    # for PYTHONPATH
VLLM_PYTHON = "/opt/miniconda3/envs/profiling/bin/python"
GPU = "7"                                                   # user-specified; never auto-switch
SGLANG_PORT = 30000
VLLM_PORT   = 30001
GPU_IDLE_MIB = 2000
SEED = 1

# ----- Base env: pin GPU, offline HF, strip KAPI and IPC; PYTHONPATH overlays fixed SGLang -----
def _make_base_env():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}
    # Strict: KAPI and IPC always start unset. IPC is added per-case for SGLang IPC variant.
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
              "SGLANG_USE_CUDA_IPC_TRANSPORT"):
        env.pop(k, None)
    # PYTHONPATH: prepend the fixed SGLang clone so `import sglang` resolves there.
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = SGLANG_PR_PY + (os.pathsep + prev if prev else "")
    return env

_BASE_ENV = _make_base_env()

# Smoke cases: (id, framework, ipc_on, extra_server_flags, server_wait_s)
SMOKE_CASES = [
    ("smoke_sglang_ipc",   "sglang", True,  [], 480),
    ("smoke_sglang_noipc", "sglang", False, [], 480),
    ("smoke_vllm_anchor",  "vllm",   False, [], 600),
]

# bench_serving image params (smoke: tiny, no perf conclusions). Pin text length via
# --random-range-ratio 1.0 (verified 2026-05-30).
IMG_ARGS = [
    "--dataset-name",       "image",
    "--image-count",        "1",
    "--image-resolution",   "720p",
    "--image-format",       "png",
    "--image-content",      "random",
    "--random-input-len",   "128",
    "--random-output-len",  "32",
    "--random-range-ratio", "1.0",
    "--max-concurrency",    "1",
    "--num-prompts",        "2",
    "--warmup-requests",    "1",
    "--seed",               str(SEED),
    "--extra-request-body", '{"temperature": 0, "top_p": 1}',
    "--output-details",
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


def wait_server(port, timeout):
    deadline = time.time() + timeout
    while time.time() < deadline:
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
    # SGLang clone commit SHA
    try:
        sha = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        out["sglang_pr_head_sha"] = sha
    except Exception as ex:
        out["sglang_pr_head_sha"] = f"ERROR:{ex}"
    # Branch name (informational)
    try:
        br = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
        out["sglang_pr_branch"] = br
    except Exception as ex:
        out["sglang_pr_branch"] = f"ERROR:{ex}"
    # Confirm the merged fix is in history (commit 07f326c184)
    try:
        merged = subprocess.run(
            ["git", "-C", str(SGLANG_PR), "merge-base", "--is-ancestor",
             "07f326c184", "HEAD"],
            capture_output=True, text=True)
        out["merged_fix_in_history"] = (merged.returncode == 0)
    except Exception as ex:
        out["merged_fix_in_history"] = f"ERROR:{ex}"
    # Confirm the runtime `import sglang` resolves under SGLANG_PR_PY (not /sgl-workspace)
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
    out["fix_marker"]        = lines[2] if len(lines) >= 3 else None
    out["probe_stderr_tail"]  = (probe.stderr or "")[-300:]
    out["fixed_path_ok"] = (
        out["sglang_dunder_file"] is not None
        and SGLANG_PR_PY in out["sglang_dunder_file"]
        and out["fix_marker"] == "FIX_OK"
    )
    return out


def run_bench_smoke(case_id, port, out_jsonl, case_env):
    out_jsonl.unlink(missing_ok=True)
    cmd = (
        ["python3", "-m", "sglang.bench_serving",
         "--backend", "sglang-oai-chat",
         "--base-url", f"http://127.0.0.1:{port}",
         "--model", SNAPSHOT]
        + IMG_ARGS
        + ["--output-file", str(out_jsonl)]
    )
    log(f"  bench: --base-url http://127.0.0.1:{port} num-prompts=2 warmup=1 resolution=720p seed={SEED}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=case_env)
    elapsed = round(time.time() - t0, 1)
    log(f"  bench rc={res.returncode} elapsed={elapsed}s")
    if res.returncode != 0:
        snippet = (res.stdout + res.stderr)[-500:]
        log(f"  bench output tail: {snippet}")
        return None, elapsed, res.returncode, res.stdout + res.stderr
    d = parse_bench_jsonl(out_jsonl)
    return d, elapsed, 0, res.stdout + res.stderr


def run_case(case_id, framework, ipc_on, extra_flags, wait_s):
    log(f"\n{'='*60}")
    log(f"SMOKE: {case_id}  framework={framework}  ipc={'ON' if ipc_on else 'OFF'}")
    log(f"{'='*60}")

    u = gpu_used()
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"  STOP: GPU {GPU} not idle (used={u} MiB)")
        return {"id": case_id, "status": "GPU_NOT_IDLE", "gpu_mib": u,
                "kapi_logging": False, "profiler": False}

    case_env = {**_BASE_ENV}
    if ipc_on:
        case_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        case_env.pop(k, None)
    # vLLM server uses its own python interpreter, so importing sglang via the
    # PYTHONPATH overlay does not affect it. The bench client (python3 -m
    # sglang.bench_serving) DOES honor PYTHONPATH and will load the fixed code.

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
    lf = open(LOGS / f"{case_id}_server.log", "w")
    log(f"  launching {framework} port={port} wait≤{wait_s}s ...")
    proc = subprocess.Popen(srv_cmd, env=case_env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {
        "id": case_id, "framework": framework, "ipc_on": ipc_on,
        "extra_server_flags": extra_flags, "port": port,
        "kapi_logging": False, "profiler": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU, "snapshot": SNAPSHOT.split("/")[-1],
    }

    try:
        if not wait_server(port, wait_s):
            log(f"  ERROR: server {case_id} did not come up in {wait_s}s")
            rec["status"] = "SERVER_NO_START"
            return rec

        log(f"  server up (port={port})")
        out_jsonl = SMOKE_DIR / f"{case_id}_bench.jsonl"
        d, elapsed, rc, stdout = run_bench_smoke(case_id, port, out_jsonl, case_env)

        rec["bench_elapsed_s"] = elapsed
        rec["bench_rc"] = rc

        if d is None:
            log(f"  STOP: bench failed for {case_id}")
            rec["status"] = "BENCH_FAILED"
            rec["bench_stdout_tail"] = stdout[-400:]
            return rec

        errors = d.get("errors", [])
        n_fail = sum(1 for e in errors if e)
        completed = d.get("completed", 0)
        generated = d.get("generated_texts", [])
        has_output = any(bool(t and t.strip()) for t in generated)
        ttfts_raw = d.get("ttfts", [])

        # Look for the specific forbidden-token failure signature that the fix targets.
        forbidden_err_present = any(
            "No data iterator found for token" in str(e) for e in errors if e)

        rec.update({
            "completed": completed,
            "failures": n_fail,
            "forbidden_token_error_present": forbidden_err_present,
            "has_non_empty_output": has_output,
            "median_ttft_ms": d.get("median_ttft_ms"),
            "total_input_tokens": d.get("total_input_tokens"),
            "total_input_text_tokens": d.get("total_input_text_tokens"),
            "total_input_vision_tokens": d.get("total_input_vision_tokens"),
            "total_output_tokens": d.get("total_output_tokens"),
            "generated_texts_sample": [t[:120] for t in generated[:2]],
            "errors_sample": [str(e) for e in errors[:2]],
            "ttfts_ms": [round(t * 1000, 1) for t in ttfts_raw if t is not None],
        })

        if forbidden_err_present:
            log(f"  {case_id}: forbidden-token error returned — fix path broken")
            rec["status"] = "FORBIDDEN_TOKEN_ERROR"
        elif n_fail > 0:
            log(f"  {case_id}: {n_fail} failures -> status=HAS_FAILURES")
            rec["status"] = "HAS_FAILURES"
        elif completed == 0:
            log(f"  {case_id}: 0 completed -> status=NO_COMPLETIONS")
            rec["status"] = "NO_COMPLETIONS"
        elif not has_output:
            log(f"  {case_id}: no non-empty output -> status=NO_OUTPUT")
            rec["status"] = "NO_OUTPUT"
        else:
            log(f"  {case_id}: OK  completed={completed}  ttft={rec.get('median_ttft_ms')}ms  "
                f"vision_tok={rec.get('total_input_vision_tokens')}  "
                f"text_tok={rec.get('total_input_text_tokens')}")
            rec["status"] = "OK"

    finally:
        kill_server(proc, patt)

    return rec


def write_summary(results, provenance):
    lines = [
        "# Stage 4.1 Fixed-Generator Smoke Summary — image+text\n",
        f"> Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
        f"GPU={GPU}  seed={SEED}  num_prompts=2  warmup=1  resolution=720p  "
        f"range_ratio=1.0\n",
        "> **Purpose:** confirm the fixed-generator path is wired up end-to-end. "
        "NO performance conclusions.\n",
    ]

    all_ok = all(r.get("status") == "OK" for r in results)
    verdict = ("✅ ALL PASS — fixed-generator path validated; safe to proceed to IMG-A"
               if all_ok else
               "❌ FAILURES — do NOT proceed to IMG-A")
    lines += [f"## Overall verdict: {verdict}\n"]

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

    lines += [
        "## Per-case results\n",
        "| case | status | completed | failures | forbidden_token_err | "
        "vision_tok | text_tok | median_ttft_ms | output_non_empty |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r.get('status','?')} | {r.get('completed','?')} "
            f"| {r.get('failures','?')} | {r.get('forbidden_token_error_present','?')} "
            f"| {r.get('total_input_vision_tokens','?')} "
            f"| {r.get('total_input_text_tokens','?')} | {r.get('median_ttft_ms','?')} "
            f"| {r.get('has_non_empty_output','?')} |"
        )
    lines.append("")

    lines += ["## Token composition (smoke, no perf weight)\n"]
    for r in results:
        if r.get("status") == "OK":
            lines.append(
                f"- **{r['id']}**: input={r.get('total_input_tokens','?')} "
                f"(text={r.get('total_input_text_tokens','?')} + "
                f"vision={r.get('total_input_vision_tokens','?')}), "
                f"output={r.get('total_output_tokens','?')}"
            )
    lines.append("")

    lines += ["## Sample outputs\n"]
    for r in results:
        lines.append(f"**{r['id']}** (status={r.get('status')}  ipc={r.get('ipc_on','?')}):")
        for i, txt in enumerate(r.get("generated_texts_sample", [])[:2]):
            lines.append(f"  req{i+1}: `{txt}`")
    lines.append("")

    lines += ["## Stop condition check\n"]
    stops = [f"- {r['id']}: {r.get('status')}" for r in results if r.get("status") != "OK"]
    if stops:
        lines.append("⚠️ STOP CONDITIONS TRIGGERED — do NOT proceed to IMG-A:")
        lines.extend(stops)
    else:
        lines.append("✅ No stop conditions. Fixed-generator smoke pass — IMG-A is the next gated stage.")

    return "\n".join(lines)


def main():
    os.chdir(LAB)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log("=== Stage 4.1 Fixed-Generator Smoke Preflight ===")
    log(f"GPU: {GPU}  (user-specified; not auto-selected)")

    # Preflight: GPU idle check
    u = gpu_used()
    log(f"GPU {GPU} memory: {u} MiB  (idle threshold: {GPU_IDLE_MIB} MiB)")
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"STOP: GPU {GPU} not idle (used={u} MiB). Aborting.")
        sys.exit(1)

    # Preflight: KAPI env check
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        if k in os.environ:
            log(f"STOP: {k} is set — forbidden for clean runs. Aborting.")
            sys.exit(1)
    log("KAPI env: clean")

    # Preflight: fixed-SGLang import gate
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
    log(f"Smoke cases: {[c[0] for c in SMOKE_CASES]}")
    log(f"Image params: 720p random png seed={SEED} count=1 input_len=128 output_len=32 range_ratio=1.0")
    log("")

    results = []
    all_ok = True
    for case_id, framework, ipc_on, extra_flags, wait_s in SMOKE_CASES:
        rec = run_case(case_id, framework, ipc_on, extra_flags, wait_s)
        results.append(rec)
        (SMOKE_DIR / "smoke_results.json").write_text(
            json.dumps({"provenance": prov, "cases": results}, indent=2))
        if rec.get("status") != "OK":
            log(f"\nSTOP: {case_id} status={rec.get('status')}. "
                "Writing summary and halting — do NOT proceed to IMG-A.")
            all_ok = False
            break

    (SMOKE_DIR / "smoke_results.json").write_text(
        json.dumps({"provenance": prov, "cases": results}, indent=2))
    log(f"\nsmoke_results.json written: {SMOKE_DIR / 'smoke_results.json'}")

    summary = write_summary(results, prov)
    (SMOKE_DIR / "smoke_summary.md").write_text(summary)
    log(f"smoke_summary.md written: {SMOKE_DIR / 'smoke_summary.md'}")

    if all_ok:
        log("\n=== SMOKE PASS: all 3 cases OK — fixed-generator path validated ===")
        sys.exit(0)
    else:
        log("\n=== SMOKE FAIL: see smoke_summary.md ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
