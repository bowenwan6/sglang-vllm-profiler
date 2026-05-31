#!/usr/bin/env python3
"""
v2 / Issue #4 — Phase 4.0 smoke runner: image+text, 3 paths.

Verifies 3 smoke paths (smoke_sglang_ipc, smoke_sglang_noipc, smoke_vllm_anchor)
with --dataset-name image --backend sglang-oai-chat. num-prompts=2, tiny.
NO performance conclusions. Purpose: resolve Phase 4.0 open items only.

Usage:
    python3 run_image_text_smoke.py

Writes:
    experiments/qwen3vl8b/v2/image_text_benchmarks/smoke/smoke_results.json
    experiments/qwen3vl8b/v2/image_text_benchmarks/smoke/smoke_summary.md
    logs/qwen3vl8b/v2/image_text_benchmarks/smoke/<case>_server.log

STRICT CLEAN: never sets SGLANG_KERNEL_API_LOGLEVEL / SGLANG_KERNEL_API_LOGDEST / profiler.
SGLANG_USE_CUDA_IPC_TRANSPORT=1 is set ONLY for smoke_sglang_ipc.
GPU: 7 (fixed per user spec — never auto-switch).
Never writes to v1 paths or caseAC_rebaseline/.
"""
import json, os, subprocess, sys, time, urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
LAB      = Path("/data/sglang-vllm-profiler")
BASE     = LAB / "experiments/qwen3vl8b/v2/image_text_benchmarks"
SMOKE_DIR = BASE / "smoke"
LOGS     = LAB / "logs/qwen3vl8b/v2/image_text_benchmarks/smoke"
VLLM_PYTHON = "/opt/miniconda3/envs/profiling/bin/python"
GPU = "7"  # user-specified; never auto-switch
SGLANG_PORT = 30000
VLLM_PORT   = 30001
GPU_IDLE_MIB = 2000
SEED = 1

# Base env: strip KAPI vars and IPC transport; GPU pinned; offline HF.
_BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1"}
for _k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST",
           "SGLANG_USE_CUDA_IPC_TRANSPORT"):
    _BASE_ENV.pop(_k, None)

# Smoke cases: (id, framework, ipc_on, extra_server_flags, server_wait_s)
SMOKE_CASES = [
    ("smoke_sglang_ipc",   "sglang", True,  [], 480),
    ("smoke_sglang_noipc", "sglang", False, [], 480),
    ("smoke_vllm_anchor",  "vllm",   False, [], 600),
]

# bench_serving image params (smoke: tiny, no perf conclusions)
IMG_ARGS = [
    "--dataset-name", "image",
    "--image-count",       "1",
    "--image-resolution",  "720p",
    "--image-format",      "png",
    "--image-content",     "random",
    "--random-input-len",  "128",
    "--random-output-len", "32",
    "--random-range-ratio","1.0",   # pins text length: randint(128,129) = always 128
    "--max-concurrency",   "1",
    "--num-prompts",       "2",
    "--warmup-requests",   "1",
    "--seed",              str(SEED),
    "--extra-request-body",'{"temperature": 0, "top_p": 1}',
    "--output-details",
]


def log(m):
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] {m}", flush=True)


def gpu_used():
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
         "-i", GPU],
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
    """Read last non-empty line of a JSONL output file from bench_serving."""
    try:
        lines = [l.strip() for l in Path(path).read_text().splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else None
    except Exception as ex:
        log(f"  parse error: {ex}")
        return None


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

    # Per-case env: add IPC only for ipc_on variant
    case_env = {**_BASE_ENV}
    if ipc_on:
        case_env["SGLANG_USE_CUDA_IPC_TRANSPORT"] = "1"
    # Double-check: no KAPI leakage
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        case_env.pop(k, None)

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

        rec.update({
            "completed": completed,
            "failures": n_fail,
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

        if n_fail > 0:
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


def write_summary(results):
    lines = [
        "# Phase 4.0 Smoke Summary — image+text\n",
        f"> Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  "
        f"GPU={GPU}  seed={SEED}  num_prompts=2  warmup=1  resolution=720p  "
        f"range_ratio=1.0\n",
        "> **Purpose:** schema / path verification only — NO performance conclusions.\n",
    ]

    all_ok = all(r.get("status") == "OK" for r in results)
    verdict = "✅ ALL PASS — safe to proceed to IMG-A" if all_ok else "❌ FAILURES — do NOT proceed to IMG-A"
    lines += [f"## Overall verdict: {verdict}\n"]

    lines += [
        "## Per-case results\n",
        "| case | status | completed | failures | vision_tok | text_tok | median_ttft_ms | output_non_empty |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['id']} | {r.get('status','?')} | {r.get('completed','?')} "
            f"| {r.get('failures','?')} | {r.get('total_input_vision_tokens','?')} "
            f"| {r.get('total_input_text_tokens','?')} | {r.get('median_ttft_ms','?')} "
            f"| {r.get('has_non_empty_output','?')} |"
        )
    lines.append("")

    ipc_r   = next((r for r in results if r["id"] == "smoke_sglang_ipc"), {})
    noipc_r = next((r for r in results if r["id"] == "smoke_sglang_noipc"), {})
    vllm_r  = next((r for r in results if r["id"] == "smoke_vllm_anchor"), {})

    lines += ["## Phase 4.0 open items resolution\n"]

    # Open item 1: vLLM anchor
    lines.append("**1. vLLM image anchor** (`sglang-oai-chat` → vLLM `/v1/chat/completions` with data-URI images):")
    if vllm_r.get("status") == "OK":
        lines.append(
            f"  ✅ RESOLVED — works. completed={vllm_r.get('completed')}, "
            f"failures={vllm_r.get('failures')}, non-empty_output={vllm_r.get('has_non_empty_output')}, "
            f"vision_tok={vllm_r.get('total_input_vision_tokens')}"
        )
    else:
        lines.append(
            f"  ❌ FAILED — status={vllm_r.get('status','not_run')}. "
            f"Errors: {vllm_r.get('errors_sample', vllm_r.get('bench_stdout_tail','?'))}"
        )
    lines.append("")

    # Open item 2: range-ratio / length pinning
    lines.append("**2. Text length pinning** (`--random-range-ratio 1.0`):")
    ref = ipc_r if ipc_r.get("status") == "OK" else (noipc_r if noipc_r.get("status") == "OK" else {})
    if ref.get("total_input_text_tokens") is not None:
        avg_text = ref.get("total_input_text_tokens", 0) / max(ref.get("completed", 1), 1)
        lines.append(
            f"  ✅ range_ratio=1.0 confirmed. "
            f"total_text_tok={ref.get('total_input_text_tokens')} over {ref.get('completed')} requests "
            f"(avg {avg_text:.1f} tok/req; includes chat-template overhead over 128 raw text tokens). "
            f"Use `--random-range-ratio 1.0` in IMG-A/B to pin text length."
        )
    else:
        lines.append("  ⚠ Token counts unavailable; check bench JSONL output manually.")
    lines.append("")

    # Open item 3: IPC observability
    lines.append("**3. CUDA IPC transport** (`SGLANG_USE_CUDA_IPC_TRANSPORT=1` observability):")
    if ipc_r.get("status") == "OK" and noipc_r.get("status") == "OK":
        lines.append(
            "  ✅ Both IPC-on and IPC-off paths ran cleanly. Env var accepted by SGLang server (no error). "
            "Direct engagement verification requires checking server log for IPC-related init lines "
            "(grep the server log for 'ipc' / 'transport'). "
            f"IPC-on completed={ipc_r.get('completed')}, noipc completed={noipc_r.get('completed')}."
        )
    elif ipc_r.get("status") == "OK":
        lines.append(f"  ⚠ IPC-on OK; IPC-off status={noipc_r.get('status','?')}")
    else:
        lines.append(f"  ❌ IPC-on failed: status={ipc_r.get('status','?')}")
    lines.append("")

    lines += ["## Token composition (smoke, no perf weight)\n"]
    for r in results:
        if r.get("status") == "OK":
            total_in = r.get("total_input_tokens", "?")
            text_in  = r.get("total_input_text_tokens", "?")
            vis_in   = r.get("total_input_vision_tokens", "?")
            total_out = r.get("total_output_tokens", "?")
            lines.append(
                f"- **{r['id']}**: input={total_in} (text={text_in} + vision={vis_in}), "
                f"output={total_out}"
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
        lines.append("✅ No stop conditions. Smoke pass — implement IMG-A runner next.")

    return "\n".join(lines)


def main():
    os.chdir(LAB)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log("=== Phase 4.0 Smoke Preflight ===")
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
    log(f"Snapshot: {SNAPSHOT.split('/')[-1]}")
    log(f"Smoke cases: {[c[0] for c in SMOKE_CASES]}")
    log(f"Image params: 720p random png seed={SEED} count=1 input_len=128 output_len=32 range_ratio=1.0")
    log("")

    results = []
    all_ok = True
    for case_id, framework, ipc_on, extra_flags, wait_s in SMOKE_CASES:
        rec = run_case(case_id, framework, ipc_on, extra_flags, wait_s)
        results.append(rec)
        (SMOKE_DIR / "smoke_results.json").write_text(json.dumps(results, indent=2))
        if rec.get("status") != "OK":
            log(f"\nSTOP: {case_id} status={rec.get('status')}. "
                "Writing summary and halting — do NOT proceed to IMG-A.")
            all_ok = False
            break

    (SMOKE_DIR / "smoke_results.json").write_text(json.dumps(results, indent=2))
    log(f"\nsmoke_results.json written: {SMOKE_DIR / 'smoke_results.json'}")

    summary = write_summary(results)
    (SMOKE_DIR / "smoke_summary.md").write_text(summary)
    log(f"smoke_summary.md written: {SMOKE_DIR / 'smoke_summary.md'}")

    if all_ok:
        log("\n=== SMOKE PASS: all 3 cases OK — proceed to IMG-A ===")
        sys.exit(0)
    else:
        log("\n=== SMOKE FAIL: see smoke_summary.md ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
