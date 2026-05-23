#!/usr/bin/env python3
"""
Phase 3 — targeted Case B SGLang EXTEND *formal* (graph-on) retry.

Case B graph-on EXTEND failed 3x earlier (num_steps=10 window landed on fast
graph-replay decode). Here we escalate strategies, stopping at first EXTEND:
  1. c=1, num_steps=50   (canonical conc)
  2. c=1, num_steps=100  (canonical conc)
  3. c=2, num_steps=100  (non-canonical conc)
  4. c=4, num_steps=100  (non-canonical conc)
  5. c=4, num_steps=200  (non-canonical conc)

Graph-on kept throughout (formal). SGLang only, GPU 1, Case B default flags,
prefill-only load (/generate max_new_tokens=1) on the Case B dataset.
Writes to sglang_extend_formal_retry/ (does NOT touch existing traces).
"""

import json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_phase3_extend as E  # reuse helpers: log, wait_server, gpu_used, kill_server, read_prompts, PrefillLoad, stage_files

LAB = Path("/data/sglang-vllm-profiler")
SNAPSHOT = E.SNAPSHOT
RETRY_DIR = LAB / "traces/qwen3vl8b/caseB_longprefill/sglang_extend_formal_retry"
LOGS = E.LOGS
META = E.META
PORT = E.SGLANG_PORT
GPU = E.GPU
CASE = "caseB_longprefill"
FLAGS = []  # Case B canonical: default flags, graph-ON (no graph-disable flags)

STRATEGIES = [
    dict(name="s1_c1_n50",  conc=1, num_steps=50,  canonical=True),
    dict(name="s2_c1_n100", conc=1, num_steps=100, canonical=True),
    dict(name="s3_c2_n100", conc=2, num_steps=100, canonical=False),
    dict(name="s4_c4_n100", conc=4, num_steps=100, canonical=False),
    dict(name="s5_c4_n200", conc=4, num_steps=200, canonical=False),
]


def attempt(strat):
    name, conc, num_steps = strat["name"], strat["conc"], strat["num_steps"]
    out_dir = RETRY_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    logpath = LOGS / f"{CASE}_extend_formal_retry_{name}_server.log"
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", SNAPSHOT,
           "--dtype", "bfloat16", "--port", str(PORT), "--tp", "1",
           "--attention-backend", "flashinfer"] + FLAGS  # graph-ON
    env = {**E.BASE_ENV, "SGLANG_KERNEL_API_LOGLEVEL": "1",
           "SGLANG_KERNEL_API_LOGDEST": str(LOGS / f"{CASE}_extend_formal_retry_{name}_kapi_%i.log"),
           "SGLANG_TORCH_PROFILER_DIR": str(out_dir)}
    E.log(f"--- strategy {name}: c={conc}, num_steps={num_steps}, graph-ON ---")
    lf = open(logpath, "w")
    proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if not E.wait_server():
        E.log("  ERROR: server did not come up"); E.kill_server(proc)
        return {"ok": False, "reason": "server_did_not_start", "dir": str(out_dir)}
    loader = E.PrefillLoad(E.read_prompts(CASE, 64), conc)
    try:
        E.log(f"  start prefill-only load (max_new_tokens=1, c={conc})")
        loader.start()
        time.sleep(6)
        E.log(f"  run profiler --profile-by-stage --num-steps {num_steps}")
        pcmd = ["python3", "-m", "sglang.profiler", "--url", f"http://127.0.0.1:{PORT}",
                "--num-steps", str(num_steps), "--profile-by-stage", "--output-dir", str(out_dir)]
        pr = subprocess.run(pcmd, env=E.BASE_ENV, capture_output=True, text=True, timeout=600)
        if pr.returncode != 0:
            E.log(f"  ERROR profiler rc={pr.returncode}: {(pr.stdout+pr.stderr)[-400:]}")
            return {"ok": False, "reason": "profiler_failed", "dir": str(out_dir)}
        time.sleep(8)
    except subprocess.TimeoutExpired:
        E.log("  ERROR profiler timeout")
        return {"ok": False, "reason": "profiler_timeout", "dir": str(out_dir)}
    finally:
        loader.stop_all()
        E.log(f"  prefill load sent ~{loader.sent} requests")
        E.kill_server(proc)
    files, stages, total = E.stage_files(out_dir)
    has_ext = bool(stages & {"EXTEND", "PREFILL"})
    E.log(f"  {name}: stages={sorted(stages)} files={len(files)} size={total/1e6:.1f}MB extend={has_ext}")
    return {"ok": has_ext, "stages": sorted(stages), "size_bytes": total, "dir": str(out_dir),
            "reason": None if has_ext else "no_extend"}


def main():
    os.chdir(LAB)
    LOGS.mkdir(parents=True, exist_ok=True)
    RETRY_DIR.mkdir(parents=True, exist_ok=True)
    E.log("=== Case B EXTEND formal (graph-on) targeted retry ===")
    u = E.gpu_used()
    if not (0 <= u < 2000):
        E.log(f"STOP: GPU {GPU} not idle (used={u} MiB)"); sys.exit(1)

    attempts = []
    winner = None
    for strat in STRATEGIES:
        res = attempt(strat)
        res["strategy"] = strat
        attempts.append(res)
        if res["ok"]:
            winner = res
            E.log(f">>> EXTEND captured with {strat['name']} (canonical={strat['canonical']}) <<<")
            break

    # Clean up failed DECODE-only attempt dirs to avoid LFS bloat; keep only winner (if any).
    for res in attempts:
        if winner and res["dir"] == winner["dir"]:
            continue
        d = Path(res["dir"])
        if d.exists():
            subprocess.run(["rm", "-rf", str(d)], check=False)

    meta = {
        "case": CASE, "purpose": "targeted graph-on EXTEND formal retry",
        "model_snapshot": E.SNAP_SHA, "dataset": E.CASES[CASE]["ds"], "gpu": GPU,
        "graph_mode": "graph-on (formal)", "server_flags": FLAGS,
        "load": "prefill-only /generate max_new_tokens=1",
        "strategies_tried": [{"name": s["strategy"]["name"], "conc": s["strategy"]["conc"],
                              "num_steps": s["strategy"]["num_steps"],
                              "canonical": s["strategy"]["canonical"],
                              "ok": s["ok"], "stages": s.get("stages"), "reason": s.get("reason")}
                             for s in attempts],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if winner:
        sw = winner["strategy"]
        meta.update({
            "success": True, "winning_strategy": sw["name"],
            "concurrency": sw["conc"], "num_steps": sw["num_steps"],
            "canonical_protocol_match": sw["canonical"],
            "canonical_note": ("matches canonical Case B protocol (c=1)" if sw["canonical"]
                               else f"NON-canonical concurrency c={sw['conc']} used only to land prefill "
                                    "in the profile window; canonical Case B benchmark is c=1"),
            "trace_dir": winner["dir"], "size_bytes": winner["size_bytes"],
        })
    else:
        meta.update({"success": False,
                     "note": "graph-on EXTEND not captured after 5 strategies; graph-off mapping EXTEND remains the prefill-stage source for Case B"})
    META.mkdir(parents=True, exist_ok=True)
    json.dump(meta, open(META / "caseB_extend_formal_retry_meta.json", "w"), indent=2)
    E.log(f"success={bool(winner)} -> caseB_extend_formal_retry_meta.json")
    E.log("=== retry complete ===")


if __name__ == "__main__":
    main()
