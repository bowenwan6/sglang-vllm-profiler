#!/usr/bin/env python3
"""
Issue #4 v3 — IMG-R ratio sweep (plan.md §11.8).

Answers a question the IMG-A bracket cannot: **is there an image+text mix where
the prefill CUDA graph pays?** IMG-A says it costs +3.4% at one operating point;
one point is not an answer.

Designed around a mechanism, not a search. A prefill CUDA graph saves a roughly
*constant* per-forward launch overhead, while BCG on Qwen3-VL pays a cost that
scales with sequence length — the DeepStack replay bridge copies `input_embeds`
and `input_deepstack_embeds` into registered stable slots, `N × 4096 × 4` on 8B,
plus padding to the captured bucket. Net ≈ C − k·N, positive only for small N.

So the variable is **N, total tokens entering the LM prefill** — and images
inflate it: Qwen3-VL spends one visual token per 32×32 pixels, so 720p is ~900
visual tokens against ~128 of text.

Prediction recorded before running (plan.md §11.8): the graph effect is monotonic
in N and changes sign below N ≈ 400. `R4_720p_longtext` is the falsification
case — it adds *text* at a fixed image, moving the image/text ratio toward text
while N grows. The mechanism says it gets worse; a ratio-driven story says better.

Transport is pinned to `cuda_ipc` on every arm: issue #4 names it as the standard
condition for SGLang image runs, and holding it fixed removes it as a variable.

`tc_piecewise` is excluded here — its eager fallback begins at graph-eligible
call ~6402, so any run long enough for a stable p50 measures eager execution
under a PCG label. See plan.md §11.9 for the protocol that can measure it.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_imgA_v3 as R  # noqa: E402
from engagement_verify import fetch_server_info, one_line, verify_arm  # noqa: E402

# (id, dataset flags, est. vision tokens, text tokens, note)
WORKLOADS = {
    "R0_text": (["--dataset-name", "random"], 0, 128,
                "text-only; does the graph win at all in this harness?"),
    "R1_tiny": (["--dataset-name", "image", "--image-count", "1",
                 "--image-resolution", "256x256", "--image-format", "png",
                 "--image-content", "random"], 64, 128,
                "smallest real image"),
    "R2_360p": (["--dataset-name", "image", "--image-count", "1",
                 "--image-resolution", "360p", "--image-format", "png",
                 "--image-content", "random"], 225, 128,
                "expected crossover region"),
    "R3_720p": (["--dataset-name", "image", "--image-count", "1",
                 "--image-resolution", "720p", "--image-format", "png",
                 "--image-content", "random"], 882, 128,
                "= IMG-A, the known negative"),
    "R4_720p_longtext": (["--dataset-name", "image", "--image-count", "1",
                          "--image-resolution", "720p", "--image-format", "png",
                          "--image-content", "random"], 882, 1024,
                         "falsification: more text at the same image"),
    "R5_1080p": (["--dataset-name", "image", "--image-count", "1",
                  "--image-resolution", "1080p", "--image-format", "png",
                  "--image-content", "random"], 1980, 128,
                 "confirms monotonicity at large N"),
    # Added after R2/R3 showed the saving plateaus to N=364 and has crossed zero
    # by N=1024, leaving the sign change inside an interval the original design
    # never sampled. Square images so height/width order is moot; Qwen3-VL spends
    # one visual token per 32x32 pixels.
    "R6_640": (["--dataset-name", "image", "--image-count", "1",
                "--image-resolution", "640x640", "--image-format", "png",
                "--image-content", "random"], 400, 128,
               "fills the 364->1024 gap (low)"),
    "R7_768": (["--dataset-name", "image", "--image-count", "1",
                "--image-resolution", "768x768", "--image-format", "png",
                "--image-content", "random"], 576, 128,
               "fills the 364->1024 gap (mid)"),
    "R8_896": (["--dataset-name", "image", "--image-count", "1",
                "--image-resolution", "896x896", "--image-format", "png",
                "--image-content", "random"], 784, 128,
               "fills the 364->1024 gap (high)"),
}

# Pairs run back-to-back so each comparison is its own bracket.
SWEEP_ORDER = ["R0_text", "R1_tiny", "R2_360p", "R3_720p",
               "R4_720p_longtext", "R5_1080p"]
GRAPH_ARMS = ["disabled", "breakable"]
DRIFT_WORKLOAD = "R3_720p"

NUM_PROMPTS = 300
WARMUP = 20
REPS = 3
TRANSPORT = "cuda_ipc"


def workload_args(wid, text_tokens):
    flags, _, _, _ = WORKLOADS[wid]
    return list(flags) + [
        "--random-input-len", str(text_tokens),
        "--random-output-len", "128",
        "--random-range-ratio", "1.0",
        "--seed", str(R.SEED),
    ]


def run_rep(cell, port, rep, env, raw_dir, wid, text_tokens, num_prompts):
    import subprocess
    import time
    out = raw_dir / f"{cell}_rep{rep}.jsonl"
    out.unlink(missing_ok=True)
    cmd = (["python3", "-m", "sglang.benchmark.serving",
            "--backend", "sglang-oai-chat",
            "--base-url", f"http://127.0.0.1:{port}",
            "--model", R.SNAPSHOT,
            "--num-prompts", str(num_prompts),
            "--max-concurrency", "1"]
           + workload_args(wid, text_tokens) + ["--output-file", str(out)])
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    elapsed = round(time.time() - t0, 1)
    if res.stderr.strip():
        (raw_dir / f"{cell}_rep{rep}.stderr").write_text(res.stderr)
    if res.returncode != 0:
        R.log(f"    rep{rep} FAILED rc={res.returncode}: "
              f"{(res.stdout + res.stderr)[-500:]}")
        return False, {"status": "BENCH_FAILED", "rc": res.returncode}
    d = R.parse_bench_jsonl(out)
    if d is None:
        return False, {"status": "PARSE_ERROR"}
    errors = d.get("errors", []) or []
    n_fail = sum(1 for e in errors if e)
    ttfts = [x * 1000 for x in (d.get("ttfts") or []) if x is not None]
    m = {
        "completed": d.get("completed"),
        "failures": n_fail,
        "ttft_p50": R.percentile(ttfts, 50) if ttfts else d.get("median_ttft_ms"),
        "ttft_p99": R.percentile(ttfts, 99) if ttfts else d.get("p99_ttft_ms"),
        "tpot_p50": d.get("median_tpot_ms"),
        "e2e_p50": d.get("median_e2e_latency_ms"),
        "total_input_vision_tokens": d.get("total_input_vision_tokens"),
        "total_input_text_tokens": d.get("total_input_text_tokens"),
        "elapsed_s": elapsed,
    }
    R.log(f"    rep{rep} {elapsed}s completed={m['completed']} fail={n_fail} "
          f"ttft_p50={m['ttft_p50']}ms")
    return n_fail == 0, m


def run_cell(wid, backend, raw_dir, tag=""):
    import subprocess
    _, est_vis, text_tokens, note = WORKLOADS[wid]
    cell = f"{wid}__{backend}{tag}"
    R.log("\n" + "=" * 64)
    R.log(f"CELL {cell}  transport={TRANSPORT}  prefill_backend={backend}")
    R.log(f"  workload: {note} (est vision tok {est_vis}, text tok {text_tokens})")
    R.log("=" * 64)

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
           "--mm-feature-transport", TRANSPORT,
           "--cuda-graph-backend-prefill", backend]

    R.LOGS.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = R.LOGS / f"imgR_{cell}_server.log"
    lf = open(log_path, "w")
    R.log(f"  launching sglang port={port}")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    rec = {"cell": cell, "workload": wid, "prefill_backend": backend,
           "transport": TRANSPORT, "est_vision_tokens": est_vis,
           "text_tokens": text_tokens,
           "num_prompts": NUM_PROMPTS, "warmup": WARMUP, "reps_planned": REPS,
           "timestamp_utc": datetime.now(timezone.utc).isoformat()}
    try:
        if not R.wait_server(port, 900, proc=proc):
            rec["status"] = "SERVER_NO_START"
            R.log("  ERROR: server did not come up")
            return rec
        info = fetch_server_info(port)
        if info is not None:
            (raw_dir / f"{cell}_server_info.json").write_text(json.dumps(info, indent=2))
        if WARMUP:
            R.log(f"  warmup {WARMUP} prompts (discarded)")
            run_rep(cell, port, 0, env, raw_dir, wid, text_tokens, WARMUP)
        reps, any_fail = [], False
        for rep in range(1, REPS + 1):
            ok, m = run_rep(cell, port, rep, env, raw_dir, wid, text_tokens, NUM_PROMPTS)
            reps.append(m)
            if not ok:
                any_fail = True
                break
        rec["reps"] = reps
        rec["status"] = "INVALID_FAILURES" if any_fail else "OK"
        lf.flush()
        v = verify_arm(cell, backend, TRANSPORT, info, log_path)
        rec["engagement"] = v
        R.log(f"  {one_line(v)}")
    finally:
        lf.flush()
        lf.close()
        R.kill_server(proc, "sglang.launch_server")

    if rec.get("status") == "OK":
        p50s = [r["ttft_p50"] for r in rec["reps"] if r.get("ttft_p50") is not None]
        if p50s:
            rec["ttft_p50_median"] = round(statistics.median(p50s), 3)
            rec["ttft_p50_reps"] = p50s
            rec["ttft_p50_cv_pct"] = (
                round(100 * statistics.pstdev(p50s) / statistics.mean(p50s), 1)
                if len(p50s) > 1 and statistics.mean(p50s) else 0.0)
        last = rec["reps"][-1] if rec["reps"] else {}
        n = max(NUM_PROMPTS, 1)
        rec["vision_tok_per_req"] = (last.get("total_input_vision_tokens") or 0) // n
        rec["text_tok_per_req"] = (last.get("total_input_text_tokens") or 0) // n
        rec["measured_N"] = rec["vision_tok_per_req"] + rec["text_tok_per_req"]
        R.log(f"  {cell}: ttft_p50_median={rec.get('ttft_p50_median')}ms "
              f"cv={rec.get('ttft_p50_cv_pct')}%  measured_N={rec['measured_N']}")
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workloads", default=None,
                    help="comma-separated subset of the sweep order")
    ap.add_argument("--out", default="phase2b_imgR_sweep")
    a = ap.parse_args()

    order = ([w.strip() for w in a.workloads.split(",") if w.strip()]
             if a.workloads else SWEEP_ORDER)
    outdir = R.RESULTS / a.out
    raw = outdir / "raw"
    outdir.mkdir(parents=True, exist_ok=True)
    R.log(f"IMG-R sweep: workloads={order} arms={GRAPH_ARMS} "
          f"n={NUM_PROMPTS} warmup={WARMUP} reps={REPS} transport={TRANSPORT}")
    R.log(f"out={outdir}")

    results = []
    for wid in order:
        for backend in GRAPH_ARMS:
            results.append(run_cell(wid, backend, raw))
            (outdir / "results.json").write_text(json.dumps(results, indent=2))
    # Drift gate: repeat the reference workload's `disabled` arm at the end.
    if DRIFT_WORKLOAD in order:
        results.append(run_cell(DRIFT_WORKLOAD, "disabled", raw, tag="__repeat"))
    (outdir / "results.json").write_text(json.dumps(results, indent=2))
    R.log(f"\nwrote {outdir/'results.json'}")

    R.log("\n===== graph effect by workload =====")
    by = {r.get("cell"): r for r in results}
    for wid in order:
        d = by.get(f"{wid}__disabled")
        b = by.get(f"{wid}__breakable")
        if not d or not b:
            continue
        if (d.get("engagement", {}).get("engagement") != "VERIFIED"
                or b.get("engagement", {}).get("engagement") != "VERIFIED"):
            R.log(f"  {wid:<20} UNVERIFIED — excluded")
            continue
        dv, bv = d.get("ttft_p50_median"), b.get("ttft_p50_median")
        if dv and bv:
            R.log(f"  {wid:<20} N={d.get('measured_N'):>5}  "
                  f"disabled={dv:>8.2f}ms  breakable={bv:>8.2f}ms  "
                  f"graph effect={100*(bv-dv)/dv:+.2f}%")


if __name__ == "__main__":
    main()
