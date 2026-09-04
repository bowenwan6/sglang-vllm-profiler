#!/usr/bin/env python3
"""
Issue #4 v3 — Phase 1.2 correctness parity.

Two questions, both of which must be settled before any latency number from the
bracket means anything:

  P1  **Cross-framework parity** (the plan's 1.2): do SGLang and vLLM produce
      the same greedy completions for the same fixed text fixture? If they
      diverge, the A0-vs-V0 gap is not a like-for-like comparison and must be
      reported as such.

  P2  **Cross-backend parity on images** (added here): does the prefill CUDA
      graph change the output? `A1_disabled` is the eager reference;
      `A3_bcg` / `A0_default` replay through the graph. On Qwen3-VL-8B
      (`deepstack_visual_indexes = [8,16,24]`, replay width 12288) this is
      exactly the DeepStack replay path that PR #33726 fixes, so a divergence
      here would mean the graph arms are fast but wrong -- the failure mode the
      whole v3 design exists to refuse to publish.

P2 is not in the plan's 1.2 but belongs in it: without it, `A0`/`A3` could pass
engagement verification (right backend, graph genuinely used) and still be
numerically wrong. Engagement and correctness are different questions.

The fixture is generated in-process -- fixed RGB vertical stripes -- so every arm
sees byte-identical input with no download or disk dependency. Greedy decode
(temperature 0, top_p 1, fixed seed).

Usage:
    python3 parity_check.py --serve <arm> ...     # run named arms end to end
    python3 parity_check.py --port 30000 --label X  # probe a live server
"""
from __future__ import annotations

import argparse
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_imgA_v3 import (  # noqa: E402
    ARMS, BASE_ENV, LOGS, SGLANG_PORT, SNAPSHOT, VLLM_PORT, VLLM_PYTHON,
    gpu_used, kill_server, log, wait_server,
)

OUT = Path("/data/sglang-vllm-profiler/experiments/qwen3vl8b/v3_issue4/results/phase1_parity")
MAX_TOKENS = 48


def stripe_image_b64(w=336, h=336, stripe=42):
    from PIL import Image
    img = Image.new("RGB", (w, h))
    px = img.load()
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    for x in range(w):
        c = palette[(x // stripe) % len(palette)]
        for y in range(h):
            px[x, y] = c
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def fixtures():
    b64 = stripe_image_b64()
    return {
        "text_primes": [{"role": "user",
                         "content": "Name the first four prime numbers."}],
        "text_capital": [{"role": "user",
                          "content": "What is the capital of France? Answer in one word."}],
        "image_colors": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "Describe the colors in this image in order."}]}],
        "image_count": [{"role": "user", "content": [
            {"type": "image_url",
             "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "How many distinct colored bands are there?"}]}],
    }


def probe(port, label):
    out = {}
    for name, msgs in fixtures().items():
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=json.dumps({"model": SNAPSHOT, "messages": msgs,
                             "temperature": 0.0, "top_p": 1.0,
                             "max_tokens": MAX_TOKENS, "seed": 0}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            out[name] = d["choices"][0]["message"]["content"]
        except Exception as e:
            out[name] = f"<ERROR {type(e).__name__}: {e}>"
        log(f"    [{label}] {name}: {out[name][:80]!r}")
    return out


def serve_and_probe(arm):
    fw, transport, backend, wait_s = ARMS[arm]
    port = SGLANG_PORT if fw == "sglang" else VLLM_PORT
    patt = ("sglang.launch_server" if fw == "sglang"
            else "vllm.entrypoints.openai.api_server")
    if fw == "sglang":
        cmd = ["python3", "-m", "sglang.launch_server",
               "--model-path", SNAPSHOT, "--dtype", "bfloat16",
               "--port", str(port), "--tp", "1",
               "--attention-backend", "flashinfer"]
        if transport is not None:
            cmd += ["--mm-feature-transport", transport]
        if backend is not None:
            cmd += ["--cuda-graph-backend-prefill", backend]
    else:
        cmd = [VLLM_PYTHON, "-m", "vllm.entrypoints.openai.api_server",
               "--model", SNAPSHOT, "--dtype", "bfloat16",
               "--port", str(port), "--tensor-parallel-size", "1"]

    env = {**BASE_ENV}
    if fw == "vllm":
        env.pop("PYTHONPATH", None)
        env.pop("SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK", None)

    LOGS.mkdir(parents=True, exist_ok=True)
    lf = open(LOGS / f"parity_{arm}_server.log", "w")
    log(f"  launching {arm} ({fw}) port={port} wait<={wait_s}s")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    try:
        if not wait_server(port, wait_s, proc=proc):
            log(f"  ERROR: {arm} did not come up")
            return {"arm": arm, "status": "SERVER_NO_START"}
        return {"arm": arm, "status": "OK", "outputs": probe(port, arm)}
    finally:
        lf.close()
        kill_server(proc, patt)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serve", nargs="+", required=True,
                    help="arm ids to launch and probe, e.g. A1_disabled A3_bcg V0_vllm")
    ap.add_argument("--out", default="phase1_parity")
    a = ap.parse_args()

    outdir = Path(str(OUT).replace("phase1_parity", a.out))
    outdir.mkdir(parents=True, exist_ok=True)
    recs = []
    for arm in a.serve:
        u = gpu_used()
        if not (0 <= u < 2000):
            log(f"  STOP: GPU not idle ({u} MiB)")
            recs.append({"arm": arm, "status": "GPU_NOT_IDLE", "gpu_mib": u})
            break
        recs.append(serve_and_probe(arm))
        (outdir / "parity.json").write_text(json.dumps(recs, indent=2))

    # Compare every arm against the first OK arm (the eager/reference arm).
    ok = [r for r in recs if r.get("status") == "OK"]
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "reference": ok[0]["arm"] if ok else None, "comparisons": []}
    if ok:
        ref = ok[0]
        for other in ok[1:]:
            diffs = [
                {"fixture": k, "reference": ref["outputs"][k], "other": other["outputs"][k]}
                for k in ref["outputs"]
                if ref["outputs"][k] != other["outputs"].get(k)
            ]
            report["comparisons"].append({
                "arm": other["arm"],
                "identical": not diffs,
                "n_fixtures": len(ref["outputs"]),
                "diffs": diffs,
            })
            log(f"  {ref['arm']} vs {other['arm']}: "
                + ("IDENTICAL" if not diffs else f"{len(diffs)} DIVERGENT"))
    (outdir / "parity_report.json").write_text(json.dumps(report, indent=2))
    log(f"wrote {outdir/'parity_report.json'}")


if __name__ == "__main__":
    main()
