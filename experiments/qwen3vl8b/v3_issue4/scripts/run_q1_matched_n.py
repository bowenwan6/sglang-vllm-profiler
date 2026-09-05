#!/usr/bin/env python3
"""
Issue #4 follow-on — Q1: is a text token equivalent to a visual token?
(plan.md §12.1)

The v3 sweep varied *visual* tokens across seven points and located the
graph's material-win boundary. It varied *text* tokens exactly once, at 720p,
where both cells sat inside the ±3.60% resolution floor. So the claim "the
controlling variable is total prefill tokens, not the image/text ratio" has never
been tested where a difference in token type would show.

This runs three text-only cells whose token counts **match** workloads v3 already
measured with images, so half of every comparison already exists:

    text-208   vs  R1_tiny  (66 visual + 142 text)   measured -16.30%
    text-544   vs  R6_640   (402 visual + 142 text)  measured  -4.54%
    text-1024  vs  R3_720p  (882 visual + 142 text)  measured  +0.80%

Same N, different composition. Agreement closes Q1 on both axes; divergence
narrows the v3 conclusion to the visual axis.

`text-1024` doubles as the realistic-prompt-length answer — the v3 report's
load-bearing -44.83% was measured at 128 text tokens, and real prompts carry
system context and retrieved passages.

**Paired A/B/A/B blocking** (plan.md §12.3). v3's resolution floor was set by
drift across a bracket spanning hours; per-cell variation was only 0.2-1.8%. So
the arms alternate in short blocks — disabled, breakable, disabled, breakable,
... — one repetition each, instead of all of one arm then all of the other. The
comparison then spans minutes. The three `disabled` blocks of a workload must
agree within 2% or the workload is discarded, not averaged.

GPU is pinned per run and recorded in every record.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_imgA_v3 as R  # noqa: E402
from engagement_verify import fetch_server_info, one_line, verify_arm  # noqa: E402

TRANSPORT = "cuda_ipc"
ARMS = ["disabled", "breakable"]
BLOCKS = 3                 # A/B pairs per workload
NUM_PROMPTS = 300
WARMUP = 20
WITHIN_PAIR_DRIFT_GATE = 2.0

# id -> (image resolution or None, text tokens, v3 partner, partner's effect)
#
# This bracket runs on GPU 5 while the v3 partners were measured on GPU 7. Both
# are H200s of the same model on the same host; the owner's call is that the
# difference is immaterial, so no cross-GPU calibration cell is run. Recorded
# here and in manifest.md so the assumption is visible rather than silent.
WORKLOADS = {
    "text-208":  (None,      208,  "R1_tiny", -16.30),
    "text-544":  (None,      544,  "R6_640",   -4.54),
    "text-1024": (None,      1024, "R3_720p",  +0.80),
}
ORDER = ["text-208", "text-544", "text-1024"]


def bench_args(resolution, text_tokens):
    if resolution is None:
        base = ["--dataset-name", "random"]
    else:
        base = ["--dataset-name", "image", "--image-count", "1",
                "--image-resolution", resolution, "--image-format", "png",
                "--image-content", "random"]
    return base + ["--random-input-len", str(text_tokens),
                   "--random-output-len", "128",
                   "--random-range-ratio", "1.0",
                   "--seed", str(R.SEED)]


def run_block(wid, backend, block, raw_dir):
    """One server lifetime = one repetition of one arm."""
    resolution, text_tokens, partner, partner_eff = WORKLOADS[wid]
    cell = f"{wid}__{backend}__b{block}"
    R.log("\n" + "-" * 64)
    R.log(f"BLOCK {cell}  transport={TRANSPORT}  backend={backend}  "
          f"image={resolution or 'none'} text_tokens={text_tokens}")

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
           # Pinned rather than left to resolution: the box resolves it to 8192,
           # comfortably above every prompt here, but it must not drift.
           "--chunked-prefill-size", "8192",
           "--mm-feature-transport", TRANSPORT,
           "--cuda-graph-backend-prefill", backend]

    R.LOGS.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = R.LOGS / f"q1_{cell}_server.log"
    lf = open(log_path, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {"cell": cell, "workload": wid, "backend": backend, "block": block,
           "transport": TRANSPORT, "image_resolution": resolution,
           "text_tokens": text_tokens, "gpu": R.GPU,
           "matches": partner, "partner_effect_pct": partner_eff,
           "num_prompts": NUM_PROMPTS, "warmup": WARMUP,
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if not R.wait_server(port, 600, proc=proc):
            rec["status"] = "SERVER_NO_START"
            R.log("  ERROR: server did not come up")
            return rec
        info = fetch_server_info(port)
        if info is not None:
            (raw_dir / f"{cell}_server_info.json").write_text(json.dumps(info, indent=2))
        if WARMUP:
            _run_rep(cell, port, 0, env, raw_dir, resolution, text_tokens, WARMUP)
        ok, m = _run_rep(cell, port, 1, env, raw_dir, resolution, text_tokens, NUM_PROMPTS)
        rec["metrics"] = m
        rec["status"] = "OK" if ok else "INVALID_FAILURES"
        rec["ttft_p50"] = m.get("ttft_p50")
        lf.flush()
        v = verify_arm(cell, backend, TRANSPORT, info, log_path)
        rec["engagement"] = v
        R.log(f"  {one_line(v)}")
    finally:
        lf.flush()
        lf.close()
        R.kill_server(proc, "sglang.launch_server")
    R.log(f"  {cell}: ttft_p50={rec.get('ttft_p50')}ms")
    return rec


def _run_rep(cell, port, rep, env, raw_dir, resolution, text_tokens, n):
    import time
    out = raw_dir / f"{cell}_rep{rep}.jsonl"
    out.unlink(missing_ok=True)
    cmd = (["python3", "-m", "sglang.benchmark.serving",
            "--backend", "sglang-oai-chat",
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", R.SNAPSHOT,
            "--num-prompts", str(n), "--max-concurrency", "1"]
           + bench_args(resolution, text_tokens) + ["--output-file", str(out)])
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    el = round(time.time() - t0, 1)
    if res.stderr.strip():
        (raw_dir / f"{cell}_rep{rep}.stderr").write_text(res.stderr)
    if res.returncode != 0:
        R.log(f"    rep{rep} FAILED rc={res.returncode}: "
              f"{(res.stdout + res.stderr)[-400:]}")
        return False, {"status": "BENCH_FAILED"}
    d = R.parse_bench_jsonl(out)
    if d is None:
        return False, {"status": "PARSE_ERROR"}
    errs = d.get("errors", []) or []
    nfail = sum(1 for e in errs if e)
    ttfts = [x * 1000 for x in (d.get("ttfts") or []) if x is not None]
    m = {"completed": d.get("completed"), "failures": nfail,
         "ttft_p50": R.percentile(ttfts, 50) if ttfts else d.get("median_ttft_ms"),
         "tpot_p50": d.get("median_tpot_ms"),
         "total_input_text_tokens": d.get("total_input_text_tokens"),
         "total_input_vision_tokens": d.get("total_input_vision_tokens"),
         "elapsed_s": el}
    R.log(f"    rep{rep} {el}s completed={m['completed']} fail={nfail} "
          f"ttft_p50={m['ttft_p50']}ms")
    return nfail == 0, m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workloads", default=None)
    ap.add_argument("--out", default="q1_matched_n")
    ap.add_argument("--gpu", default=None,
                    help="override the pinned GPU; recorded in every record")
    a = ap.parse_args()
    if a.gpu is not None and a.gpu != R.GPU:
        R.log(f"GPU override: {R.GPU} -> {a.gpu}")
        R.GPU = a.gpu
        R.BASE_ENV = {**R.BASE_ENV, "CUDA_VISIBLE_DEVICES": a.gpu}
    order = ([w.strip() for w in a.workloads.split(",")] if a.workloads else ORDER)

    outdir = R.RESULTS / a.out
    raw = outdir / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    R.log(f"Q1 matched-N: workloads={order} arms={ARMS} blocks={BLOCKS} "
          f"n={NUM_PROMPTS} warmup={WARMUP} transport={TRANSPORT}")
    R.log(f"out={outdir}")

    results = []
    for wid in order:
        for block in range(1, BLOCKS + 1):
            for backend in ARMS:          # A, B, A, B, A, B
                results.append(run_block(wid, backend, block, raw))
                (outdir / "results.json").write_text(json.dumps(results, indent=2))
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    R.log(f"\nwrote {outdir / 'results.json'}")

    R.log("\n===== Q1: matched-N comparison =====")
    by = {}
    for r in results:
        if r.get("status") == "OK" and (r.get("engagement") or {}).get(
                "engagement") == "VERIFIED":
            by.setdefault((r["workload"], r["backend"]), []).append(r["ttft_p50"])
    for wid in order:
        d = by.get((wid, "disabled"), [])
        b = by.get((wid, "breakable"), [])
        if len(d) < 2 or len(b) < 2:
            R.log(f"  {wid:<11} insufficient verified blocks "
                  f"(disabled={len(d)}, breakable={len(b)})")
            continue
        drift = 100 * (max(d) - min(d)) / statistics.median(d)
        dv, bv = statistics.median(d), statistics.median(b)
        eff = 100 * (bv - dv) / dv
        _, _, partner, partner_eff = WORKLOADS[wid]
        gate = "PASS" if drift <= WITHIN_PAIR_DRIFT_GATE else "FAIL — discard"
        R.log(f"  {wid:<11} disabled={dv:>7.2f}ms breakable={bv:>7.2f}ms "
              f"effect={eff:+6.2f}%  | {partner} (with image) was "
              f"{partner_eff:+6.2f}%  | within-pair drift {drift:.2f}% [{gate}]")


if __name__ == "__main__":
    main()
