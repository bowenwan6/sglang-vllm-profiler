#!/usr/bin/env python3
"""Q3 — ViT CUDA graph on Qwen3-VL-8B: staged runner (node side).

Stages, in order. Each is resumable and writes its own summary under results/:

  check    preflight: snapshot, patch applied, one GPU, torch sees it        -> preflight.json
  parity   both arms answer the same greedy fixtures identically              -> parity.json      (exit 4 on FAIL)
  pilot    one server per arm; per workload: warmup, profiler capture, 30 req -> pilot.json, gate_g1.json (exit 5 on STOP)
  sweep    A/B/B/A blocks per workload, 200 prompts + 20 warmup each          -> raw/sweep/<cell>.json, sweep.json
  mixed    random resolutions 256x256..1280x720, same seed on both arms       -> mixed.json
  status   print STATUS.json

Stop protocol: `touch $Q3_STOP_FILE` (default ~/sgl/q3/STOP). The runner finishes the
current cell, writes its summaries and exits 3. Rerunning the same stage resumes.
Design and pre-registered predictions: ../PLAN.md.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q3_common as C  # noqa: E402
from q3_common import log  # noqa: E402

REPLAY_RESIDUAL_MS = 1.0    # pre-registered residual of the graph arm (launch + 3 copies + eager prologue)
GATE_MIN_PRED_MS = 3.0      # G1: run the sweep only if some size is predicted to gain at least this
SPREAD_GATE_PP = 3.0        # G2: paired-effect spread per workload (Q1's rule)
EXIT_STOPPED, EXIT_PARITY_FAIL, EXIT_GATE_STOP = 3, 4, 5


# ============================================================ check

def cmd_check(_a) -> int:
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(), "hard_failures": [], "warnings": []}
    hard, warn = rec["hard_failures"], rec["warnings"]

    rec["snapshot"] = str(C.SNAPSHOT)
    if not (C.SNAPSHOT / "config.json").exists():
        hard.append(f"model snapshot missing: {C.SNAPSHOT}  "
                    f"(hf download {C.MODEL_REPO} --revision {C.MODEL_REV})")
    env = C.server_env("off", False)
    r = subprocess.run(["python3", "-c",
                        "import os, torch, sglang; print(os.path.dirname(sglang.__file__)); "
                        "print(sglang.__version__); print(torch.__version__, torch.version.cuda, "
                        "torch.cuda.is_available(), torch.cuda.device_count())"],
                       capture_output=True, text=True, env=env)
    if r.returncode != 0:
        hard.append("python cannot import torch/sglang: " + r.stderr[-400:])
    else:
        pkg, ver, torchline = r.stdout.strip().splitlines()[:3]
        rec["sglang_pkg_dir"], rec["sglang_version"], rec["torch"] = pkg, ver, torchline
        if "True" not in torchline:
            hard.append(f"torch.cuda.is_available() is False ({torchline})")
        src = Path(pkg)
        runner = src / "srt/multimodal/vit_cuda_graph_runner.py"
        model = src / "srt/models/qwen3_vl.py"
        rec["patch_applied"] = (runner.exists() and "VIT_CG_STATS" in runner.read_text()
                                and model.exists() and "Q3_VIT_FORWARD" in model.read_text())
        if not rec["patch_applied"]:
            hard.append("instrumentation patch not applied to the installed sglang source "
                        "(git apply patches/q3_vit_instrumentation.patch in the sglang checkout)")
        try:
            rec["sglang_commit"] = subprocess.run(["git", "-C", str(src.parent.parent), "rev-parse", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip()
            rec["sglang_dirty_files"] = subprocess.run(
                ["git", "-C", str(src.parent.parent), "status", "--short", "--untracked-files=no"],
                capture_output=True, text=True).stdout.strip().splitlines()
        except Exception:
            pass
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
    rec["visible_gpus"] = smi
    if len(smi) != 1:
        warn.append(f"{len(smi)} GPUs visible; the design assumes exactly one")
    rec["gpu"] = C.gpu_name()
    rec["gpu_used_mib"] = C.gpu_used()
    if not (0 <= rec["gpu_used_mib"] < C.GPU_IDLE_MIB):
        hard.append(f"GPU not idle: {rec['gpu_used_mib']} MiB used")
    du = shutil.disk_usage(C.SGL_ROOT if C.SGL_ROOT.exists() else Path.home())
    rec["disk_free_gb"] = round(du.free / 1e9, 1)
    if rec["disk_free_gb"] < 50:
        warn.append(f"only {rec['disk_free_gb']} GB free")
    rec["profiler_commit"] = subprocess.run(["git", "-C", str(C.EXP), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    rec["paths"] = {"OUT": str(C.OUT), "LOGS": str(C.LOGS), "TRACES": str(C.TRACES),
                    "STOP_FILE": str(C.STOP_FILE)}
    rec["server_cmd"] = " ".join(C.server_cmd())
    rec["verdict"] = "PASS" if not hard else "FAIL"
    C.save_json(C.OUT / "preflight.json", rec)
    for h in hard:
        log(f"  FAIL: {h}")
    for w in warn:
        log(f"  warn: {w}")
    log(f"preflight {rec['verdict']}  sglang={rec.get('sglang_version')} @ {rec.get('sglang_commit', '?')[:10]} "
        f"patch={rec.get('patch_applied')}  gpu={rec['gpu']}")
    C.write_status(stage="check", verdict=rec["verdict"])
    return 0 if not hard else 2


# ============================================================ parity

def _stripe_png_b64(w: int, h: int, stripe: int = 42) -> str:
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


def _fixtures() -> Dict[str, list]:
    b336, b512 = _stripe_png_b64(336, 336), _stripe_png_b64(512, 512)

    def img(b64, text):
        return [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": text}]}]
    return {
        "text_primes": [{"role": "user", "content": "Name the first four prime numbers."}],
        "text_capital": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
        "image336_colors": img(b336, "Describe the colors in this image in order."),
        "image336_count": img(b336, "How many distinct colored bands are there?"),
        "image512_colors": img(b512, "Describe the colors in this image in order."),
    }


def _probe(label: str) -> Dict[str, str]:
    out = {}
    for name, msgs in _fixtures().items():
        req = urllib.request.Request(
            f"http://127.0.0.1:{C.PORT}/v1/chat/completions",
            data=json.dumps({"model": str(C.SNAPSHOT), "messages": msgs, "temperature": 0.0,
                             "top_p": 1.0, "max_tokens": 48, "seed": 0}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out[name] = json.load(r)["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            out[name] = f"<ERROR {type(e).__name__}: {e}>"
        log(f"    [{label}] {name}: {out[name][:70]!r}")
    return out


def cmd_parity(_a) -> int:
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(), "arms": {}}
    for arm in ("off", "on"):
        C.write_status(stage="parity", arm=arm)
        h = C.launch_server(f"parity_{arm}", arm, vit_timing=False)
        try:
            outputs = _probe(arm)
            scan = C.scan_log(h.log_path, 0)
            # two image shapes (336^2, 512^2) => two captures expected on the on arm
            ver = C.verify_arm(arm, scan, n_requests=len(outputs), expect_shapes=2 if arm == "on" else 0,
                               info=h.info)
            rec["arms"][arm] = {"outputs": outputs, "captures": scan.get("captures"),
                                "verify": ver, "startup_s": h.startup_s, "gpu_used_mib": C.gpu_used()}
        finally:
            C.kill_server(h.proc)
    a, b = rec["arms"]["off"]["outputs"], rec["arms"]["on"]["outputs"]
    diffs = [k for k in a if a[k] != b.get(k)]
    errors = [k for k in a if a[k].startswith("<ERROR") or b.get(k, "").startswith("<ERROR")]
    rec["differing_fixtures"], rec["errored_fixtures"] = diffs, errors
    rec["verdict"] = "PASS" if not diffs and not errors else "FAIL"
    C.save_json(C.OUT / "parity.json", rec)
    log(f"parity {rec['verdict']}  differing={diffs} errored={errors}  "
        f"on-arm verify={rec['arms']['on']['verify']['verdict']} {rec['arms']['on']['verify']['reasons']}")
    C.write_status(stage="parity", verdict=rec["verdict"])
    return 0 if rec["verdict"] == "PASS" else EXIT_PARITY_FAIL


# ============================================================ pilot

def _trace_summary(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        import parse_trace
        res = parse_trace.analyze_file(path)
        return res["summary"]
    except Exception as e:  # noqa: BLE001
        log(f"    trace parse failed: {e}")
        return {"error": str(e)}


def cmd_pilot(a) -> int:
    wids = a.workloads.split(",") if a.workloads else C.ORDER
    raw = C.RAW / "pilot"
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(),
                           "params": {"n_warm": a.n_warm, "n_profile": a.n_profile, "n_measure": a.n_measure,
                                      "replay_residual_ms": REPLAY_RESIDUAL_MS, "workloads": wids},
                           "arms": {}, "cells": {w: {} for w in wids}}
    for arm in ("off", "on"):
        if C.stop_requested():
            log("STOP requested before pilot arm; exiting")
            return EXIT_STOPPED
        h = C.launch_server(f"pilot_{arm}", arm, vit_timing=True)
        env = C.server_env(arm, True)
        rec["arms"][arm] = {"startup_s": h.startup_s,
                            "resolved": {k: C.dig(h.info, k) for k in
                                         ("mm_feature_transport", "mm_attention_backend",
                                          "cuda_graph_backend_prefill", "attention_backend")}
                            if h.info else None}
        try:
            for wid in wids:
                flags, text, exp_tok, exp_patches, note = C.WORKLOADS[wid]
                C.write_status(stage="pilot", arm=arm, workload=wid)
                log(f"  [{arm}] {wid}: {note}")
                off0 = C.log_offset(h.log_path)
                C.run_bench(f"pilot_{arm}_{wid}_warm", flags, text, a.n_warm, env, raw)
                tdir = C.TRACES / "pilot" / arm / wid
                before = C.trace_files(tdir)
                trace_path = None
                try:
                    C.start_profile(tdir)
                    C.run_bench(f"pilot_{arm}_{wid}_prof", flags, text, a.n_profile, env, raw)
                    C.stop_profile()
                    trace_path = C.wait_for_trace(tdir, before)
                    log(f"    trace: {trace_path}")
                except Exception as e:  # noqa: BLE001
                    log(f"    profiler step failed: {e}")
                off2 = C.log_offset(h.log_path)
                meas = C.run_bench(f"pilot_{arm}_{wid}_meas", flags, text, a.n_measure, env, raw)
                scan_meas = C.scan_log(h.log_path, off2)
                scan_all = C.scan_log(h.log_path, off0)
                n_req = a.n_warm + a.n_profile + a.n_measure + 3   # + the client's own warmup requests
                ver = C.verify_arm(arm, scan_all, n_req,
                                   expect_shapes=(1 if (arm == "on" and exp_patches) else 0), info=h.info)
                cell = {"bench": meas, "timings": C.summarize_timings(scan_meas["timings"]),
                        "req_times": C.summarize_req_times(scan_meas["req_times"]),
                        "trace_path": str(trace_path) if trace_path else None,
                        "trace": _trace_summary(trace_path),
                        "captures": scan_all.get("captures"), "stats_last": scan_all.get("stats_last"),
                        "verify": ver, "gpu_used_mib": C.gpu_used()}
                rec["cells"][wid][arm] = cell
                t = cell["timings"]
                log(f"    ttft_p50={meas.get('ttft_p50')} ms  vit cpu_wall={t.get('cpu_wall_ms_p50')} "
                    f"gpu_span={t.get('gpu_span_ms_p50')} patches={t.get('patches')}  "
                    f"verify={ver['verdict']} {ver['reasons']}")
                C.save_json(C.OUT / "pilot.json", rec)   # incremental
        finally:
            C.kill_server(h.proc)

    # ---- predictions (H1) from the eager arm's trace, gate G1
    preds: Dict[str, Any] = {}
    for wid in wids:
        off, on = rec["cells"][wid].get("off", {}), rec["cells"][wid].get("on", {})
        tr_off = ((off.get("trace") or {}).get("Q3_VIT_FORWARD") or {})
        tr_on = ((on.get("trace") or {}).get("Q3_VIT_FORWARD") or {})
        step_off = ((off.get("trace") or {}).get("Q3_STEP_EXTEND") or {})
        U = tr_off.get("unoverlapped_ms_p50")
        pred = None if U is None else round(max(0.0, U - REPLAY_RESIDUAL_MS), 3)
        t_off, t_on = (off.get("bench") or {}).get("ttft_p50"), (on.get("bench") or {}).get("ttft_p50")
        preds[wid] = {
            "patches": (off.get("timings") or {}).get("patches"),
            "vit_cpu_wall_off_ms": tr_off.get("cpu_wall_ms_p50"),
            "vit_gpu_busy_off_ms": tr_off.get("gpu_busy_ms_p50"),
            "vit_unoverlapped_off_ms": U,
            "vit_n_launches_off": tr_off.get("n_launches_p50"),
            "vit_cpu_wall_on_ms": tr_on.get("cpu_wall_ms_p50"),
            "vit_n_graph_launches_on": tr_on.get("n_graph_launches_p50"),
            "prefill_step_off_ms": step_off.get("cpu_wall_ms_p50"),
            "ttft_off_p50": t_off, "ttft_on_p50": t_on,
            "pilot_gain_ms": round(t_off - t_on, 3) if (t_off is not None and t_on is not None) else None,
            "pred_gain_ms": pred,
            "verified": (off.get("verify", {}).get("verdict") == "VERIFIED"
                         and on.get("verify", {}).get("verdict") == "VERIFIED"),
        }
    rec["predictions"] = preds
    reasons = []
    unverified = [w for w, p in preds.items() if not p["verified"]]
    if unverified:
        reasons.append(f"unverified cells: {unverified}")
    par = C.OUT / "parity.json"
    parity = C.load_json(par).get("verdict") if par.exists() else "MISSING"
    if parity != "PASS":
        reasons.append(f"parity {parity}")
    img_preds = [p["pred_gain_ms"] for w, p in preds.items() if w != "R0_text" and p["pred_gain_ms"] is not None]
    max_pred = max(img_preds) if img_preds else None
    if max_pred is None:
        reasons.append("no trace-based prediction available")
    elif max_pred < GATE_MIN_PRED_MS:
        reasons.append(f"max predicted gain {max_pred} ms < {GATE_MIN_PRED_MS} ms")
    gate = {"verdict": "GO" if not reasons else "STOP", "reasons": reasons, "max_pred_gain_ms": max_pred,
            "parity": parity, "timestamp_utc": C.utc_now()}
    rec["gate_g1"] = gate
    C.save_json(C.OUT / "pilot.json", rec)
    C.save_json(C.OUT / "gate_g1.json", gate)

    log("\nPILOT  workload      patches  vit_wall_off  gpu_busy_off  unoverlapped  pred_gain  pilot_gain  ttft_off  ttft_on")
    for wid, p in preds.items():
        log(f"  {wid:<12} {str(p['patches']):>7}  {str(p['vit_cpu_wall_off_ms']):>12}  "
            f"{str(p['vit_gpu_busy_off_ms']):>12}  {str(p['vit_unoverlapped_off_ms']):>12}  "
            f"{str(p['pred_gain_ms']):>9}  {str(p['pilot_gain_ms']):>10}  "
            f"{str(p['ttft_off_p50']):>8}  {str(p['ttft_on_p50']):>7}  {'ok' if p['verified'] else 'UNVERIFIED'}")
    log(f"GATE G1: {gate['verdict']}  {reasons}")
    C.write_status(stage="pilot", verdict=gate["verdict"], gate_reasons=reasons)
    return 0 if gate["verdict"] == "GO" else EXIT_GATE_STOP


# ============================================================ sweep

def _cell_path(cell: str) -> Path:
    return C.RAW / "sweep" / f"{cell}.json"


def run_cell(wid: str, arm: str, block: int, prompts: int, warmup: int) -> Dict[str, Any]:
    flags, text, exp_tok, exp_patches, note = C.WORKLOADS[wid]
    cell = f"{wid}__{arm}__b{block}"
    rec: Dict[str, Any] = {"cell": cell, "workload": wid, "arm": arm, "block": block,
                           "num_prompts": prompts, "warmup": warmup, "timestamp_utc": C.utc_now()}
    t0 = time.time()
    h = C.launch_server(f"sweep_{cell}", arm, vit_timing=False)
    env = C.server_env(arm, False)
    try:
        rec["startup_s"] = h.startup_s
        C.run_bench(f"{cell}_warm", flags, text, warmup, env, C.RAW / "sweep")
        rec["gpu_used_mib_after_warmup"] = C.gpu_used()
        off = C.log_offset(h.log_path)
        meas = C.run_bench(cell, flags, text, prompts, env, C.RAW / "sweep")
        scan_all = C.scan_log(h.log_path, 0)
        scan_meas = C.scan_log(h.log_path, off)
        rec["bench"] = meas
        rec["req_times"] = C.summarize_req_times(scan_meas["req_times"])
        rec["captures"], rec["stats_last"] = scan_all.get("captures"), scan_all.get("stats_last")
        rec["verify"] = C.verify_arm(arm, scan_all, warmup + prompts + 2,
                                     expect_shapes=(1 if (arm == "on" and exp_patches) else 0), info=h.info)
        rec["status"] = meas.get("status", "?") if meas.get("failures", 0) == 0 else "HAS_FAILURES"
    finally:
        C.kill_server(h.proc)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def aggregate_sweep(wids: List[str], blocks: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {"generated_utc": C.utc_now(), "workloads": {}}
    for wid in wids:
        rows = []
        for b in range(1, blocks + 1):
            po, pn = _cell_path(f"{wid}__off__b{b}"), _cell_path(f"{wid}__on__b{b}")
            if not (po.exists() and pn.exists()):
                continue
            o, n = C.load_json(po), C.load_json(pn)
            to, tn = (o.get("bench") or {}).get("ttft_p50"), (n.get("bench") or {}).get("ttft_p50")
            if to is None or tn is None:
                continue
            rows.append({"block": b, "off_p50": to, "on_p50": tn, "saving_ms": round(to - tn, 3),
                         "effect_pct": round((tn - to) / to * 100.0, 2),
                         "tpot_off": (o.get("bench") or {}).get("tpot_p50"),
                         "tpot_on": (n.get("bench") or {}).get("tpot_p50"),
                         "verified": (o.get("verify", {}).get("verdict") == "VERIFIED"
                                      and n.get("verify", {}).get("verdict") == "VERIFIED"),
                         "capture_ms": ((n.get("captures") or [{}])[0].get("capture_ms")
                                        if n.get("captures") else None)})
        if not rows:
            continue
        effs = [r["effect_pct"] for r in rows]
        eff = statistics.median(effs)
        spread = round(max(effs) - min(effs), 2) if len(effs) > 1 else None
        gate = ("PASS" if spread is not None and (spread <= SPREAD_GATE_PP or spread <= abs(eff) / 2)
                else ("SINGLE_BLOCK" if spread is None else "WEAK"))
        out["workloads"][wid] = {"blocks": rows, "n_blocks": len(rows),
                                 "effect_pct_median": round(eff, 2),
                                 "saving_ms_median": round(statistics.median(r["saving_ms"] for r in rows), 3),
                                 "spread_pp": spread, "gate": gate,
                                 "all_verified": all(r["verified"] for r in rows),
                                 "expected_patches": C.WORKLOADS[wid][3]}
    return out


def cmd_sweep(a) -> int:
    wids = a.workloads.split(",") if a.workloads else C.ORDER
    cells = []
    for wid in wids:
        for b in range(1, a.blocks + 1):
            order = ["off", "on"] if b % 2 == 1 else ["on", "off"]   # A/B then B/A cancels linear drift
            for arm in order:
                cells.append((wid, arm, b))
    def _ok(c) -> bool:   # a FAILED / partial cell is re-run on resume
        p = _cell_path(f"{c[0]}__{c[1]}__b{c[2]}")
        try:
            return p.exists() and C.load_json(p).get("status") == "OK"
        except Exception:  # noqa: BLE001
            return False
    done = [c for c in cells if _ok(c)]
    todo = [c for c in cells if c not in done]
    log(f"sweep: {len(cells)} cells, {len(done)} done, {len(todo)} to run  "
        f"(blocks={a.blocks} prompts={a.prompts} warmup={a.warmup})")
    durations: List[float] = []
    consecutive_failures = 0
    for i, (wid, arm, b) in enumerate(todo):
        cell = f"{wid}__{arm}__b{b}"
        if C.stop_requested():
            log(f"STOP requested; {len(todo) - i} cells left")
            C.write_status(stage="sweep", stopped=True, cells_left=len(todo) - i)
            C.save_json(C.OUT / "sweep.json", aggregate_sweep(wids, a.blocks))
            return EXIT_STOPPED
        eta_min = round(statistics.mean(durations) * (len(todo) - i) / 60.0, 1) if durations else None
        C.write_status(stage="sweep", cell=cell, done=len(done) + i, total=len(cells), eta_min=eta_min)
        log(f"\n=== cell {cell}  ({len(done) + i + 1}/{len(cells)}, eta {eta_min} min) ===")
        try:
            rec = run_cell(wid, arm, b, a.prompts, a.warmup)
            consecutive_failures = 0 if rec.get("status") == "OK" else consecutive_failures + 1
        except Exception as e:  # noqa: BLE001
            log(f"  cell failed: {e}\n{traceback.format_exc()[-600:]}")
            rec = {"cell": cell, "workload": wid, "arm": arm, "block": b, "status": "FAILED",
                   "error": str(e), "timestamp_utc": C.utc_now()}
            C.kill_server(None)
            consecutive_failures += 1
        C.save_json(_cell_path(cell), rec)
        durations.append(rec.get("elapsed_s") or 0.0)
        bench = rec.get("bench") or {}
        log(f"  {cell}: ttft_p50={bench.get('ttft_p50')} ms  status={rec.get('status')}  "
            f"verify={rec.get('verify', {}).get('verdict')} {rec.get('verify', {}).get('reasons')}  "
            f"{rec.get('elapsed_s')}s")
        C.save_json(C.OUT / "sweep.json", aggregate_sweep(wids, a.blocks))   # incremental
        if consecutive_failures >= 2:
            log("two consecutive failures; aborting the sweep")
            C.write_status(stage="sweep", aborted="two consecutive failures")
            return 6
    agg = aggregate_sweep(wids, a.blocks)
    C.save_json(C.OUT / "sweep.json", agg)
    log("\nSWEEP  workload      patches  off_p50   on_p50   saving  effect   spread  gate")
    for wid, w in agg["workloads"].items():
        r = w["blocks"]
        log(f"  {wid:<12} {w['expected_patches']:>7}  {statistics.median(x['off_p50'] for x in r):>7.2f}  "
            f"{statistics.median(x['on_p50'] for x in r):>7.2f}  {w['saving_ms_median']:>+6.2f}  "
            f"{w['effect_pct_median']:>+6.2f}%  {str(w['spread_pp']):>6}  {w['gate']}"
            f"{'' if w['all_verified'] else '  UNVERIFIED'}")
    C.write_status(stage="sweep", done=len(cells), total=len(cells), finished=True)
    return 0


# ============================================================ mixed

def _sequence(path: Path, start: int) -> List[Dict[str, Any]]:
    """VIT_TIMING lines in order, each tagged with the capture (if any) that preceded it."""
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read().decode("utf-8", "replace")
    seq, pending = [], None
    for line in text.splitlines():
        m = C.RE_CAPTURE.search(line)
        if m:
            pending = {"key": m.group(1), "capture_ms": float(m.group(3))}
            continue
        m = C.RE_TIMING.search(line)
        if m:
            seq.append({"patches": int(m.group(1)), "graph": int(m.group(3)),
                        "cpu_wall_ms": float(m.group(4)), "gpu_span_ms": float(m.group(5)),
                        "captured": pending is not None,
                        "capture_ms": pending["capture_ms"] if pending else None})
            pending = None
    return seq


def cmd_mixed(a) -> int:
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(),
                           "params": {"prompts": a.prompts, "warmup": a.warmup, "flags": C.MIXED_FLAGS},
                           "arms": {}}
    text_flags, text_tok = C.WORKLOADS["R0_text"][0], C.WORKLOADS["R0_text"][1]
    for arm in ("off", "on"):
        if C.stop_requested():
            return EXIT_STOPPED
        C.write_status(stage="mixed", arm=arm)
        h = C.launch_server(f"mixed_{arm}", arm, vit_timing=True)
        env = C.server_env(arm, True)
        try:
            # text-only warmup so no image shape is captured before the run
            C.run_bench(f"mixed_{arm}_warm", text_flags, text_tok, a.warmup, env, C.RAW / "mixed")
            off = C.log_offset(h.log_path)
            meas = C.run_bench(f"mixed_{arm}", C.MIXED_FLAGS, 128, a.prompts, env, C.RAW / "mixed")
            seq = _sequence(h.log_path, off)
            scan = C.scan_log(h.log_path, off)
            ttfts = meas.get("ttft_ms") or []
            # align by the tail: the client's own warmup request also produces a timing line
            seq_tail = seq[-len(ttfts):] if ttfts and len(seq) >= len(ttfts) else seq
            joined = [{"i": i, "ttft_ms": t, **s} for i, (t, s) in enumerate(zip(ttfts, seq_tail))]
            first = [j for j in joined if j["captured"]]
            rep = [j for j in joined if not j["captured"]]
            rec["arms"][arm] = {
                "bench": {k: v for k, v in meas.items() if k != "ttft_ms"},
                "ttft_ms": ttfts,
                "n_requests": len(joined),
                "n_captures": len(scan.get("captures") or []),
                "capture_ms_total": (scan.get("stats_last") or {}).get("capture_ms_total"),
                "capture_ms_p50": C.median([c["capture_ms"] for c in (scan.get("captures") or [])]),
                "distinct_patch_counts": len({s["patches"] for s in seq_tail}),
                "ttft_first_seen_p50": C.median([j["ttft_ms"] for j in first]),
                "ttft_repeat_p50": C.median([j["ttft_ms"] for j in rep]),
                "vit_wall_first_seen_p50": C.median([j["cpu_wall_ms"] for j in first]),
                "vit_wall_repeat_p50": C.median([j["cpu_wall_ms"] for j in rep]),
                "verify": C.verify_arm(arm, scan, a.prompts + 1, expect_shapes=-1, info=h.info),
                "gpu_used_mib_end": C.gpu_used(),
                "per_request": joined,
            }
            log(f"  [{arm}] ttft p50={meas.get('ttft_p50')} p90={meas.get('ttft_p90')} p99={meas.get('ttft_p99')}  "
                f"captures={rec['arms'][arm]['n_captures']} distinct_shapes={rec['arms'][arm]['distinct_patch_counts']}  "
                f"first_seen_p50={rec['arms'][arm]['ttft_first_seen_p50']} repeat_p50={rec['arms'][arm]['ttft_repeat_p50']}")
        finally:
            C.kill_server(h.proc)
        C.save_json(C.OUT / "mixed.json", rec)
    o, n = rec["arms"].get("off", {}), rec["arms"].get("on", {})
    if o and n:
        # same seed => the i-th request has the same resolution on both arms
        idx_first = {j["i"] for j in n["per_request"] if j["captured"]}
        off_first = [j["ttft_ms"] for j in o["per_request"] if j["i"] in idx_first]
        off_rep = [j["ttft_ms"] for j in o["per_request"] if j["i"] not in idx_first]
        rec["summary"] = {
            "ttft_p50_off": o["bench"].get("ttft_p50"), "ttft_p50_on": n["bench"].get("ttft_p50"),
            "ttft_p99_off": o["bench"].get("ttft_p99"), "ttft_p99_on": n["bench"].get("ttft_p99"),
            "mean_ttft_off": C.median([sum(o["ttft_ms"]) / len(o["ttft_ms"])]) if o["ttft_ms"] else None,
            "mean_ttft_on": C.median([sum(n["ttft_ms"]) / len(n["ttft_ms"])]) if n["ttft_ms"] else None,
            "hit_rate": round(1 - n["n_captures"] / max(1, n["n_requests"]), 3),
            "first_seen_penalty_ms_p50": (C.median([j["ttft_ms"] for j in n["per_request"] if j["captured"]])
                                          or 0) - (C.median(off_first) or 0),
            "repeat_gain_ms_p50": (C.median(off_rep) or 0)
                                  - (C.median([j["ttft_ms"] for j in n["per_request"] if not j["captured"]]) or 0),
        }
        log(f"MIXED summary: {json.dumps(rec['summary'])}")
    C.save_json(C.OUT / "mixed.json", rec)
    C.write_status(stage="mixed", finished=True)
    return 0


# ============================================================ main

def cmd_status(_a) -> int:
    print(C.STATUS_FILE.read_text() if C.STATUS_FILE.exists() else "{}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("check")
    sub.add_parser("parity")
    p = sub.add_parser("pilot")
    p.add_argument("--n-warm", type=int, default=5)
    p.add_argument("--n-profile", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument("--workloads", help="comma-separated subset of " + ",".join(C.ORDER))
    s = sub.add_parser("sweep")
    s.add_argument("--blocks", type=int, default=2)
    s.add_argument("--prompts", type=int, default=200)
    s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--workloads")
    m = sub.add_parser("mixed")
    m.add_argument("--prompts", type=int, default=300)
    m.add_argument("--warmup", type=int, default=5)
    sub.add_parser("status")
    a = ap.parse_args()
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.RAW.mkdir(parents=True, exist_ok=True)
    C.LOGS.mkdir(parents=True, exist_ok=True)
    C.write_status(stage=a.stage, started_utc=C.utc_now(), pid=__import__("os").getpid())
    rc = {"check": cmd_check, "parity": cmd_parity, "pilot": cmd_pilot,
          "sweep": cmd_sweep, "mixed": cmd_mixed, "status": cmd_status}[a.stage](a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
