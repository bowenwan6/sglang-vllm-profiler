#!/usr/bin/env python3
"""
V2 — tiny serving repro for the `<|video_pad|>` benchmark-generator blocker.

Proves the failure path is real at serving time (not only theoretical):
  Probe A (failing):  one image-only chat request whose TEXT deliberately contains
                      the literal `<|video_pad|>` -> expect HTTP 400 with
                      "No data iterator found for token: <|video_pad|>".
  Probe B (control):  the same request shape with safe text -> expect HTTP 200 and
                      non-empty generated output.

This is NOT a benchmark and produces NO performance numbers. It is a 2-request probe.

IMPORTANT framing: V2 confirms the serving symptom; it does NOT make SGLang serving
the primary bug. The upstream fix target remains the benchmark *generator*
(`gen_mm_prompt`), which should never emit multimodal/control placeholder tokens in
synthetic random text when no matching media payload exists.

STRICT: GPU 7 only (never auto-switch). Clean: no KAPI logging, no profiler. SGLang
source is NOT modified. Writes only under debug_video_pad/results/ and the
debug_video_pad log dir; never touches v1 paths, original smoke/, or IMG-A results.

Usage:
    python3 run_tiny_video_pad_repro.py
"""
import base64
import io
import json
import os
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SNAPSHOT = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
            "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
LAB     = Path("/data/sglang-vllm-profiler")
BASE    = LAB / "experiments/qwen3vl8b/v2/image_text_benchmarks"
RESULTS = BASE / "debug_video_pad" / "results"
LOGS    = LAB / "logs/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad"
GPU = "7"  # user-specified; never auto-switch
SGLANG_PORT = 30000
GPU_IDLE_MIB = 2000
VIDEO_PAD = "<|video_pad|>"
EXPECTED_ERR = "No data iterator found for token: <|video_pad|>"

# Clean env: strip KAPI vars; keep IPC on (headline image config) — IPC is irrelevant to
# this generator-content bug but matches the smoke config used elsewhere.
_BASE_ENV = {**os.environ, "CUDA_VISIBLE_DEVICES": GPU, "HF_HUB_OFFLINE": "1",
             "SGLANG_USE_CUDA_IPC_TRANSPORT": "1"}
for _k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
    _BASE_ENV.pop(_k, None)


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
            if urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3).getcode() == 200:
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


def tiny_png_data_uri():
    """A small synthetic random PNG as a data URI (same construction family as the
    image dataset: random pixels -> PNG -> base64 data URI). Kept small (64x64)."""
    try:
        import numpy as np
        from PIL import Image
        arr = (np.random.rand(64, 64, 3) * 255).astype("uint8")
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="png")
        raw = buf.getvalue()
    except Exception:
        # Fallback: a fixed minimal 1x1 PNG if PIL/numpy unavailable.
        raw = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
            "+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==")
    enc = base64.b64encode(raw).decode("utf-8")
    return f"data:image/png;base64,{enc}"


def post_chat(port, text, image_uri, timeout=120):
    """POST one image-only chat request. Returns (status_code, body_str)."""
    payload = {
        "model": SNAPSHOT,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_uri}},
                {"type": "text", "text": text},
            ],
        }],
        "max_tokens": 8,
        "temperature": 0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return resp.getcode(), resp.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def extract_text(body):
    try:
        d = json.loads(body)
        return d["choices"][0]["message"]["content"]
    except Exception:
        return ""


def main():
    os.chdir(LAB)
    RESULTS.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)

    log("=== V2 tiny video_pad serving repro (GPU 7) ===")

    # KAPI guard
    for k in ("SGLANG_KERNEL_API_LOGLEVEL", "SGLANG_KERNEL_API_LOGDEST"):
        if k in os.environ:
            log(f"STOP: {k} set — forbidden for clean runs. Aborting.")
            sys.exit(1)

    # Preflight: GPU 7 idle
    u = gpu_used()
    log(f"GPU {GPU} memory: {u} MiB (idle threshold {GPU_IDLE_MIB})")
    if not (0 <= u < GPU_IDLE_MIB):
        log(f"STOP: GPU {GPU} not idle (used={u} MiB). Aborting.")
        sys.exit(1)

    image_uri = tiny_png_data_uri()
    log(f"image data URI bytes: {len(image_uri)}")

    srv_cmd = ["python3", "-m", "sglang.launch_server",
               "--model-path", SNAPSHOT, "--dtype", "bfloat16",
               "--port", str(SGLANG_PORT), "--tp", "1",
               "--attention-backend", "flashinfer"]
    patt = "sglang.launch_server"
    lf = open(LOGS / "v2_repro_server.log", "w")
    log("launching SGLang (IPC on, clean) ...")
    proc = subprocess.Popen(srv_cmd, env=_BASE_ENV, stdout=lf, stderr=subprocess.STDOUT)

    rec = {
        "stage": "V2_serving_repro",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU, "snapshot": SNAPSHOT.split("/")[-1],
        "kapi_logging": False, "profiler": False,
        "expected_error_substring": EXPECTED_ERR,
    }

    try:
        if not wait_server(SGLANG_PORT, 480):
            log("STOP: server did not come up in 480s.")
            rec["status"] = "SERVER_NO_START"
            (RESULTS / "V2_serving_repro.json").write_text(json.dumps(rec, indent=2))
            sys.exit(1)
        log("server up.")

        # Probe A — deliberate <|video_pad|> in text (image-only request).
        codeA, bodyA = post_chat(SGLANG_PORT, f"{VIDEO_PAD}describe the image", image_uri)
        a_400 = (codeA == 400)
        a_match = (EXPECTED_ERR in bodyA)
        rec["probe_A_failing"] = {
            "text": f"{VIDEO_PAD}describe the image",
            "status_code": codeA,
            "error_substring_present": a_match,
            "body_head": bodyA[:300],
        }
        log(f"Probe A: status={codeA} expected_err_present={a_match}")

        # Probe B — safe text control (same image-only shape).
        codeB, bodyB = post_chat(SGLANG_PORT, "describe the image", image_uri)
        textB = extract_text(bodyB)
        b_200 = (codeB == 200)
        b_nonempty = bool(textB and textB.strip())
        rec["probe_B_control"] = {
            "text": "describe the image",
            "status_code": codeB,
            "non_empty_output": b_nonempty,
            "output_head": (textB or "")[:200],
        }
        log(f"Probe B: status={codeB} non_empty={b_nonempty}")

        v2_pass = a_400 and a_match and b_200 and b_nonempty
        rec["status"] = "PASS" if v2_pass else "FAIL"
        rec["pass_criteria"] = {
            "A_returns_400": a_400,
            "A_error_matches": a_match,
            "B_returns_200": b_200,
            "B_non_empty": b_nonempty,
        }
        log(f"V2 verdict: {rec['status']}")

    finally:
        kill_server(proc, patt)
        rec["gpu_freed_after"] = (0 <= gpu_used() < GPU_IDLE_MIB)

    (RESULTS / "V2_serving_repro.json").write_text(json.dumps(rec, indent=2))
    log(f"wrote {RESULTS / 'V2_serving_repro.json'}")

    # Markdown summary
    pa = rec.get("probe_A_failing", {})
    pb = rec.get("probe_B_control", {})
    lines = [
        "# V2 — tiny `<|video_pad|>` serving repro",
        "",
        f"> Run: {rec['timestamp_utc']}  GPU={GPU}  clean (no KAPI / no profiler)",
        "> SGLang server, single image-only `/v1/chat/completions` probes (NOT a benchmark).",
        "",
        f"## Verdict: {'PASS' if rec.get('status') == 'PASS' else str(rec.get('status'))}",
        "",
        "| probe | text | status | expectation | met |",
        "|---|---|---|---|---|",
        f"| A (failing) | `{VIDEO_PAD}describe the image` | {pa.get('status_code')} "
        f"| 400 + `{EXPECTED_ERR}` | {pa.get('error_substring_present')} |",
        f"| B (control) | `describe the image` | {pb.get('status_code')} "
        f"| 200 + non-empty | {pb.get('non_empty_output')} |",
        "",
        "## Probe A error body (head)",
        "```",
        pa.get("body_head", ""),
        "```",
        "",
        "## Probe B output (head)",
        "```",
        pb.get("output_head", ""),
        "```",
        "",
        "## Interpretation",
        "",
        "V2 confirms the serving symptom is real: an image-only request whose text "
        "contains `<|video_pad|>` returns HTTP 400, while safe text succeeds. This does "
        "**not** make SGLang serving the primary bug — the server is correctly rejecting "
        "a video placeholder with no video payload. The fix target is the benchmark "
        "**generator** (`gen_mm_prompt`), which must not emit such tokens in synthetic "
        "random text (validated separately in V1).",
    ]
    (RESULTS / "V2_serving_repro.md").write_text("\n".join(lines))
    log(f"wrote {RESULTS / 'V2_serving_repro.md'}")

    sys.exit(0 if rec.get("status") == "PASS" else 1)


if __name__ == "__main__":
    main()
