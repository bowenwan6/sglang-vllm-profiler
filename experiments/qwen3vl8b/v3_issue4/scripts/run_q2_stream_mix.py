#!/usr/bin/env python3
"""
Issue #4 follow-on — Q2: the composition of the request *stream* (plan.md §12.2).

Everything measured so far is homogeneous and at concurrency 1: every request in
a bracket had the same shape, and all 926 measured prefill batches carried
`#new-seq: 1`. Real serving is a mixed stream at concurrency well above 1, and
the prefill-graph flag is server-wide — it cannot be set per request. So the
operator's question is not "does the graph help this workload" but **"at what
image arrival fraction does enabling it stop paying?"**

Why homogeneous data cannot answer it: graph eligibility is decided **per batch,
on the batch's summed token count** (`prefill_cuda_graph_runner.py:1231-1240`),
and the scheduler applies no image/text separation when forming a prefill batch.
A text request that shares a batch with an image request therefore sits in a
large-N batch and loses the benefit it would have had alone. Co-batching never
occurred in any prior measurement, so a weighted average of the homogeneous
points would **overestimate** the aggregate benefit.

Design: one server per arm, **two bench clients against it at once** — one
text-only, one image — each with its own Poisson arrival rate, so the arrival
fraction is `f = r_image / (r_text + r_image)` at a fixed total rate. This needs
no harness surgery and yields **per-class TTFT for free**, which is the whole
point: an aggregate number would hide the interference being looked for.

Read as three separate quantities, never merged:

  1. text-class TTFT vs f. Under `disabled` this is queueing alone; under
     `breakable`, any *extra* degradation as f rises is the co-batching effect.
  2. image-class TTFT vs f — expected flat, a control that catches confounds.
  3. aggregate vs f, derived, giving the break-even image fraction.

Staged: stage 1 is f in {0, 0.2, 1.0}; the finer fractions only run if stage 1
shows text-class degradation under `breakable` that `disabled` does not show.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_imgA_v3 as R  # noqa: E402
from engagement_verify import fetch_server_info, one_line, verify_arm  # noqa: E402

TRANSPORT = "cuda_ipc"
ARMS = ["disabled", "breakable"]
BLOCKS = 3
FRACTIONS_STAGE1 = [0.0, 0.2, 1.0]
FRACTIONS_STAGE2 = [0.05, 0.5]

TOTAL_REQUESTS = 600          # split across the two classes by f
TOTAL_RATE = 6.0              # requests/s, Poisson, fixed across every cell
TEXT_TOKENS = 512             # a realistic prompt, measured alone by Q1
IMAGE_RES = "720p"
IMAGE_TEXT_TOKENS = 128
WITHIN_PAIR_DRIFT_GATE = 2.0

RUNNING_REQ = re.compile(r"#running-req: (\d+)")


def client_cmd(kind, port, n, rate, out):
    base = ["python3", "-m", "sglang.benchmark.serving",
            "--backend", "sglang-oai-chat",
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", R.SNAPSHOT,
            "--num-prompts", str(n),
            "--request-rate", f"{rate:.4f}",
            "--random-output-len", "128",
            "--random-range-ratio", "1.0",
            "--output-file", str(out)]
    if kind == "text":
        return base + ["--dataset-name", "random",
                       "--random-input-len", str(TEXT_TOKENS),
                       "--seed", str(R.SEED)]
    return base + ["--dataset-name", "image", "--image-count", "1",
                   "--image-resolution", IMAGE_RES, "--image-format", "png",
                   "--image-content", "random",
                   "--random-input-len", str(IMAGE_TEXT_TOKENS),
                   "--seed", str(R.SEED + 1)]


def parse_class(path):
    d = R.parse_bench_jsonl(path)
    if d is None:
        return None
    errs = d.get("errors", []) or []
    ttfts = [x * 1000 for x in (d.get("ttfts") or []) if x is not None]
    return {"completed": d.get("completed"),
            "failures": sum(1 for e in errs if e),
            "ttft_p50": R.percentile(ttfts, 50) if ttfts else d.get("median_ttft_ms"),
            "ttft_p95": R.percentile(ttfts, 95) if ttfts else None,
            "tpot_p50": d.get("median_tpot_ms"),
            "n": len(ttfts)}


def observed_concurrency(log_path):
    """Mean and max #running-req seen in the server log — the load actually
    applied, as opposed to the load aimed for. Poisson arrivals make the
    in-flight count a distribution, not a setting."""
    try:
        vals = [int(v) for v in RUNNING_REQ.findall(log_path.read_text(errors="replace"))]
    except Exception:
        return None
    vals = [v for v in vals if v > 0]
    if not vals:
        return None
    return {"mean": round(statistics.mean(vals), 2), "max": max(vals),
            "p50": R.percentile([float(v) for v in vals], 50), "samples": len(vals)}


def run_block(f, backend, block, raw_dir, gpu_note):
    cell = f"f{f:g}__{backend}__b{block}"
    n_img = int(round(TOTAL_REQUESTS * f))
    n_txt = TOTAL_REQUESTS - n_img
    r_img = TOTAL_RATE * f
    r_txt = TOTAL_RATE - r_img
    R.log("\n" + "-" * 64)
    R.log(f"BLOCK {cell}  f={f}  backend={backend}  "
          f"text={n_txt}@{r_txt:.2f}/s  image={n_img}@{r_img:.2f}/s")

    u = R.gpu_used()
    if not (0 <= u < R.GPU_IDLE_MIB):
        R.log(f"  STOP: GPU {R.GPU} not idle (used={u} MiB)")
        return {"cell": cell, "status": "GPU_NOT_IDLE", "gpu_mib": u}

    env = {**R.BASE_ENV}
    port = R.SGLANG_PORT
    cmd = ["python3", "-m", "sglang.launch_server",
           "--model-path", R.SNAPSHOT, "--dtype", "bfloat16",
           "--port", str(port), "--tp", "1",
           "--attention-backend", "flashinfer",
           "--disable-radix-cache",
           "--chunked-prefill-size", "8192",
           "--mm-feature-transport", TRANSPORT,
           "--cuda-graph-backend-prefill", backend]

    R.LOGS.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = R.LOGS / f"q2_{cell}_server.log"
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {"cell": cell, "image_fraction": f, "backend": backend, "block": block,
           "transport": TRANSPORT, "gpu": R.GPU, "gpu_note": gpu_note,
           "n_text": n_txt, "n_image": n_img,
           "rate_text": round(r_txt, 4), "rate_image": round(r_img, 4),
           "total_rate": TOTAL_RATE, "text_tokens": TEXT_TOKENS,
           "image_resolution": IMAGE_RES,
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if not R.wait_server(port, 600, proc=proc):
            rec["status"] = "SERVER_NO_START"
            return rec
        info = fetch_server_info(port)
        if info is not None:
            (raw_dir / f"{cell}_server_info.json").write_text(json.dumps(info, indent=2))

        # Short warmup on whichever class is present, discarded.
        warm_kind = "text" if n_txt else "image"
        subprocess.run(client_cmd(warm_kind, port, 20, TOTAL_RATE,
                                  raw_dir / f"{cell}_warm.jsonl"),
                       capture_output=True, text=True, env=env)

        # Both classes are launched together so their arrivals genuinely
        # interleave; a sequential run would never co-batch.
        procs, outs = {}, {}
        t0 = time.time()
        for kind, n, rate in (("text", n_txt, r_txt), ("image", n_img, r_img)):
            if n <= 0:
                continue
            outs[kind] = raw_dir / f"{cell}_{kind}.jsonl"
            outs[kind].unlink(missing_ok=True)
            procs[kind] = subprocess.Popen(
                client_cmd(kind, port, n, rate, outs[kind]),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        for kind, pr in procs.items():
            _, err = pr.communicate()
            if pr.returncode != 0:
                R.log(f"  {kind} client FAILED rc={pr.returncode}: {err[-300:]}")
        rec["elapsed_s"] = round(time.time() - t0, 1)

        rec["classes"] = {k: parse_class(v) for k, v in outs.items()}
        bad = [k for k, v in rec["classes"].items()
               if v is None or (v.get("failures") or 0) > 0]
        rec["status"] = "INVALID_FAILURES" if bad else "OK"
        lf.flush()
        rec["observed_concurrency"] = observed_concurrency(log_path)
        v = verify_arm(cell, backend, TRANSPORT, info, log_path)
        rec["engagement"] = v
        R.log(f"  {one_line(v)}")
        R.log(f"  concurrency {rec['observed_concurrency']}")
        for k, m in rec["classes"].items():
            if m:
                R.log(f"  {k:<6} n={m['n']:<4} ttft_p50={m['ttft_p50']}ms "
                      f"p95={m['ttft_p95']}ms tpot={m['tpot_p50']}ms")
    finally:
        lf.flush()
        lf.close()
        R.kill_server(proc, "sglang.launch_server")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, default=1, choices=(1, 2))
    ap.add_argument("--fractions", default=None,
                    help="comma-separated override, e.g. 0,0.2,1.0")
    ap.add_argument("--out", default="q2_stream_mix")
    ap.add_argument("--gpu", default=None)
    a = ap.parse_args()
    gpu_note = ""
    if a.gpu is not None and a.gpu != R.GPU:
        R.log(f"GPU override: {R.GPU} -> {a.gpu}")
        gpu_note = f"overridden from {R.GPU}"
        R.GPU = a.gpu
        R.BASE_ENV = {**R.BASE_ENV, "CUDA_VISIBLE_DEVICES": a.gpu}

    fracs = ([float(x) for x in a.fractions.split(",")] if a.fractions
             else (FRACTIONS_STAGE1 if a.stage == 1 else FRACTIONS_STAGE2))
    outdir = R.RESULTS / a.out
    raw = outdir / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    R.log(f"Q2 stream mix: f={fracs} arms={ARMS} blocks={BLOCKS} "
          f"total_requests={TOTAL_REQUESTS} total_rate={TOTAL_RATE}/s "
          f"text={TEXT_TOKENS}tok image={IMAGE_RES}")
    R.log(f"out={outdir}")

    results = []
    for f in fracs:
        for block in range(1, BLOCKS + 1):
            for backend in ARMS:
                results.append(run_block(f, backend, block, raw, gpu_note))
                (outdir / "results.json").write_text(json.dumps(results, indent=2))
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    R.log(f"\nwrote {outdir / 'results.json'}")

    R.log("\n===== Q2: per-class TTFT by arrival fraction =====")
    ok = [r for r in results
          if r.get("status") == "OK"
          and (r.get("engagement") or {}).get("engagement") == "VERIFIED"]
    for f in fracs:
        for kind in ("text", "image"):
            row = {}
            for backend in ARMS:
                vals = [r["classes"][kind]["ttft_p50"] for r in ok
                        if r["image_fraction"] == f and r["backend"] == backend
                        and r.get("classes", {}).get(kind)]
                if vals:
                    row[backend] = statistics.median(vals)
            if len(row) == 2:
                eff = 100 * (row["breakable"] - row["disabled"]) / row["disabled"]
                R.log(f"  f={f:<5} {kind:<6} disabled={row['disabled']:>7.2f}ms "
                      f"breakable={row['breakable']:>7.2f}ms  graph effect={eff:+6.2f}%")


if __name__ == "__main__":
    main()
