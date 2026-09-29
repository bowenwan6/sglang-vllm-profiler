#!/usr/bin/env python3
"""Q3 — ViT CUDA graph on Qwen3-VL-8B: staged runner (node side).

Stages, in order. Each is resumable and writes its own summary under results/:

  check    preflight: pinned SHA, patch applied, model shards, ShareGPT, conda python, GPU   -> preflight.json   (exit 2 on FAIL)
  parity   both arms encode the same fixtures: ViT outputs within tolerance, text compared    -> parity.json      (exit 4 on FAIL)
  pilot    4 servers (off, on, off_rot, off without timing); per workload: warmup, trace, 30 -> pilot.json, gate_g1.json (exit 5 on STOP)
  sweep    A/B/B/A blocks per workload (3, 4 for 720p/1080p), 200 prompts + 20 warmup       -> cells/<cell>.json, sweep.json
  mixed    random resolutions 256x256..1280x720, same seed on both arms                      -> mixed.json
  status   print STATUS.json

Stop protocol: `touch $Q3_STOP_FILE` (default ~/sgl/q3/STOP). The runner finishes the
current block, writes its summaries and exits 3. A leftover STOP file is cleared when a
stage starts. Rerunning the same stage resumes at block granularity.
Design and pre-registered predictions: ../PLAN.md (with the 2026-09-29 amendment).
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import threading
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
BLOCKS_DEFAULT, BLOCKS_LARGE = 3, 4
DUMP_REL_FRO_TOL = 2e-2     # parity: relative Frobenius error of the encoder output, bf16 arms
TEXT_MARGIN_BENIGN_NAT = 0.5
EXIT_STOPPED, EXIT_PARITY_FAIL, EXIT_GATE_STOP = 3, 4, 5


def _stage_start(stage: str) -> None:
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.RAW.mkdir(parents=True, exist_ok=True)
    C.LOGS.mkdir(parents=True, exist_ok=True)
    if C.clear_stop():
        log("cleared a leftover STOP file")
    C.write_status(stage=stage, started_utc=C.utc_now(), pid=os.getpid(), stopped=False, finished=False)


# ============================================================ check

def cmd_check(_a) -> int:
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(), "hard_failures": [], "warnings": []}
    hard, warn = rec["hard_failures"], rec["warnings"]

    rec["python"] = C.PY
    if "sgl-profiler" not in C.PY and "sgl-profiler" not in os.environ.get("CONDA_PREFIX", ""):
        hard.append(f"python is not the sgl-profiler conda env: {C.PY} (source conda.sh && conda activate sgl-profiler)")
    rec["snapshot"] = str(C.SNAPSHOT)
    if not (C.SNAPSHOT / "config.json").exists():
        hard.append(f"model snapshot missing: {C.SNAPSHOT}  "
                    f"(hf download {C.MODEL_REPO} --revision {C.MODEL_REV})")
    else:
        rec["model_shards"] = C.model_shards_complete()
        if not rec["model_shards"].get("complete"):
            hard.append(f"model download incomplete: {rec['model_shards']}")
    sg = C.sharegpt_path()
    rec["sharegpt"] = str(sg) if sg else None
    if sg is None:
        hard.append(f"ShareGPT file missing for the text control "
                    f"(hf download {C.SHAREGPT_REPO} {C.SHAREGPT_FILE} --repo-type dataset)")
    env = C.server_env("off", False)
    r = subprocess.run([C.PY, "-c",
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
        files = {
            "runner": (src / "srt/multimodal/vit_cuda_graph_runner.py", "VIT_CG_CALL"),
            "model": (src / "srt/models/qwen3_vl.py", "Q3_VIT_DUMP"),
            "vision": (src / "srt/layers/attention/vision.py", "Q3_VIT_EAGER_ROTARY"),
        }
        applied = {k: (p.exists() and marker in p.read_text()) for k, (p, marker) in files.items()}
        rec["patch_applied"] = applied
        if not all(applied.values()):
            hard.append(f"instrumentation patch not (fully) applied: {applied} "
                        "(git apply patches/q3_vit_instrumentation.patch in the sglang checkout)")
        repo = src.parent.parent
        try:
            sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                 capture_output=True, text=True).stdout.strip()
            rec["sglang_commit"] = sha
            if not sha.startswith(C.SGLANG_SHA) and os.environ.get("Q3_ALLOW_SHA_MISMATCH") != "1":
                hard.append(f"sglang HEAD {sha[:12]} != pinned {C.SGLANG_SHA} "
                            "(SGLANG_REF=89e1316eae in setup_node.sh; Q3_ALLOW_SHA_MISMATCH=1 to override)")
            dirty = subprocess.run(["git", "-C", str(repo), "status", "--short", "--untracked-files=no"],
                                   capture_output=True, text=True).stdout.strip().splitlines()
            rec["sglang_dirty_files"] = dirty
            allowed = {"python/sglang/srt/multimodal/vit_cuda_graph_runner.py",
                       "python/sglang/srt/models/qwen3_vl.py",
                       "python/sglang/srt/layers/attention/vision.py"}
            extra = [d for d in dirty if d.split()[-1] not in allowed]
            if extra:
                warn.append(f"unexpected modified files in the sglang checkout: {extra}")
        except Exception as e:  # noqa: BLE001
            warn.append(f"could not read the sglang git state: {e}")
    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines()
    rec["visible_gpus"] = smi
    if len(smi) != 1:
        warn.append(f"{len(smi)} GPUs visible; the design assumes exactly one")
    rec["gpu"] = C.gpu_name()
    rec["gpu_total_mib"] = C.gpu_total()
    rec["gpu_used_mib"] = C.gpu_used()
    if not (0 <= rec["gpu_used_mib"] < C.GPU_IDLE_MIB):
        hard.append(f"GPU not idle: {rec['gpu_used_mib']} MiB used")
    du = shutil.disk_usage(C.SGL_ROOT if C.SGL_ROOT.exists() else Path.home())
    rec["disk_free_gb"] = round(du.free / 1e9, 1)
    if rec["disk_free_gb"] < 50:
        warn.append(f"only {rec['disk_free_gb']} GB free")
    rec["profiler_commit"] = subprocess.run(["git", "-C", str(C.EXP), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    rec["profiler_branch"] = subprocess.run(["git", "-C", str(C.EXP), "branch", "--show-current"],
                                            capture_output=True, text=True).stdout.strip()
    rec["paths"] = {"OUT": str(C.OUT), "CELLS": str(C.CELLS), "LOGS": str(C.LOGS),
                    "TRACES": str(C.TRACES), "STOP_FILE": str(C.STOP_FILE)}
    rec["server_cmd"] = " ".join(C.server_cmd())
    rec["output_len"] = C.OUTPUT_LEN
    rec["verdict"] = "PASS" if not hard else "FAIL"
    C.save_json(C.OUT / "preflight.json", rec)
    for h in hard:
        log(f"  FAIL: {h}")
    for w in warn:
        log(f"  warn: {w}")
    log(f"preflight {rec['verdict']}  sglang={rec.get('sglang_version')} @ {str(rec.get('sglang_commit', '?'))[:10]} "
        f"patch={rec.get('patch_applied')}  gpu={rec['gpu']}")
    C.write_status(stage="check", verdict=rec["verdict"], finished=True)
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
    # image fixtures first, in a fixed order: dump files are matched by order
    return {
        "image336_colors": img(b336, "Describe the colors in this image in order."),
        "image336_count": img(b336, "How many distinct colored bands are there?"),
        "image512_colors": img(b512, "Describe the colors in this image in order."),
        "text_primes": [{"role": "user", "content": "Name the first four prime numbers."}],
        "text_capital": [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
    }


def _probe(label: str) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name, msgs in _fixtures().items():
        req = urllib.request.Request(
            f"http://127.0.0.1:{C.PORT}/v1/chat/completions",
            data=json.dumps({"model": str(C.SNAPSHOT), "messages": msgs, "temperature": 0.0,
                             "top_p": 1.0, "max_tokens": 48, "seed": 0,
                             "logprobs": True, "top_logprobs": 2}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                d = json.load(r)
            ch = d["choices"][0]
            toks = []
            for t in ((ch.get("logprobs") or {}).get("content") or []):
                top = sorted((x.get("logprob", 0.0) for x in (t.get("top_logprobs") or [])), reverse=True)
                margin = (top[0] - top[1]) if len(top) >= 2 else None
                toks.append({"token": t.get("token"), "logprob": t.get("logprob"), "margin": margin})
            out[name] = {"content": ch["message"]["content"], "tokens": toks}
        except Exception as e:  # noqa: BLE001
            out[name] = {"content": f"<ERROR {type(e).__name__}: {e}>", "tokens": []}
        log(f"    [{label}] {name}: {out[name]['content'][:70]!r}")
    return out


def _compare_dumps(d_off: Path, d_on: Path) -> Dict[str, Any]:
    """Relative error of the encoder outputs, dump by dump (same fixture order on both arms)."""
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {"error": f"torch not importable: {e}"}
    fo, fn = sorted(d_off.glob("vit_*.pt")), sorted(d_on.glob("vit_*.pt"))
    rows = []
    for i, (a, b) in enumerate(zip(fo, fn)):
        ta, tb = torch.load(a).float(), torch.load(b).float()
        if ta.shape != tb.shape:
            rows.append({"i": i, "shape_off": list(ta.shape), "shape_on": list(tb.shape), "rel_fro": None})
            continue
        diff = ta - tb
        rows.append({"i": i, "shape": list(ta.shape),
                     "rel_fro": float(diff.norm() / ta.norm().clamp_min(1e-12)),
                     "rel_max": float(diff.abs().max() / ta.abs().max().clamp_min(1e-12)),
                     "max_abs": float(diff.abs().max())})
    return {"n_off": len(fo), "n_on": len(fn), "rows": rows}


def _text_divergence(off: Dict[str, Any], on: Dict[str, Any]) -> Dict[str, Any]:
    to, tn = [t["token"] for t in off.get("tokens", [])], [t["token"] for t in on.get("tokens", [])]
    if to == tn and off.get("content") == on.get("content"):
        return {"identical": True}
    k = next((i for i, (x, y) in enumerate(zip(to, tn)) if x != y), min(len(to), len(tn)))
    margin = off["tokens"][k]["margin"] if k < len(off.get("tokens", [])) else None
    return {"identical": False, "first_divergence": k, "off_margin_nat": margin,
            "benign": (margin is not None and margin < TEXT_MARGIN_BENIGN_NAT)}


def cmd_parity(a) -> int:
    _stage_start("parity")
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(), "arms": {}, "tolerance_rel_fro": DUMP_REL_FRO_TOL}
    arms = [("off", "off", None), ("on", "on", None)]
    if getattr(a, "with_default_interp", False):
        arms.append(("on_default_interp", "on", ["--enable-precise-embedding-interpolation"]))
    for slot, arm, drop in arms:
        C.write_status(stage="parity", arm=slot)
        dump_dir = C.RAW / "parity" / slot
        shutil.rmtree(dump_dir, ignore_errors=True)
        h = C.launch_server(f"parity_{slot}", arm, False, C.SERVER_WAIT_S, dump_dir, drop)
        try:
            outputs = _probe(slot)
            scan = C.scan_log(h.log_path, h.launch_offset)
            ver = C.verify_arm(arm, scan, n_requests=len(outputs), expect_shapes=2 if arm == "on" else 0,
                               info=h.info)
            rec["arms"][slot] = {"outputs": outputs, "captures": scan.get("captures"),
                                 "verify": ver, "startup_s": h.startup_s, "gpu_used_mib": C.gpu_used(),
                                 "resolved": C.resolved_config(h.info), "effective": scan.get("effective"),
                                 "dump_dir": str(dump_dir), "n_dumps": len(list(dump_dir.glob("vit_*.pt")))}
        finally:
            C.kill_server(h.proc)
    off, on = rec["arms"]["off"], rec["arms"]["on"]
    rec["dumps"] = _compare_dumps(Path(off["dump_dir"]), Path(on["dump_dir"]))
    if "on_default_interp" in rec["arms"]:
        rec["dumps_default_interp_vs_off"] = _compare_dumps(Path(off["dump_dir"]),
                                                            Path(rec["arms"]["on_default_interp"]["dump_dir"]))
    rec["text"] = {k: _text_divergence(off["outputs"][k], on["outputs"].get(k, {})) for k in off["outputs"]}
    errored = [k for k, v in off["outputs"].items() if str(v["content"]).startswith("<ERROR")] + \
              [k for k, v in on["outputs"].items() if str(v["content"]).startswith("<ERROR")]
    reasons = []
    if errored:
        reasons.append(f"errored fixtures: {sorted(set(errored))}")
    rows = rec["dumps"].get("rows") or []
    if rec["dumps"].get("error"):
        reasons.append(rec["dumps"]["error"])
    elif len(rows) < 3:
        reasons.append(f"expected 3 image dumps per arm, got off={rec['dumps'].get('n_off')} on={rec['dumps'].get('n_on')}")
    bad = [r for r in rows if r.get("rel_fro") is None or r["rel_fro"] > DUMP_REL_FRO_TOL]
    if bad:
        reasons.append(f"encoder outputs differ beyond {DUMP_REL_FRO_TOL}: {bad}")
    for v in (off["verify"], on["verify"]):
        if v["verdict"] != "VERIFIED":
            reasons.append(f"engagement: {v['reasons']}")
    non_benign = [k for k, v in rec["text"].items() if not v.get("identical") and not v.get("benign")]
    rec["text_warnings"] = non_benign
    rec["reasons"] = reasons
    rec["verdict"] = "PASS" if not reasons else "FAIL"
    C.save_json(C.OUT / "parity.json", rec)
    log(f"parity {rec['verdict']}  reasons={reasons}  text divergences (non-benign)={non_benign}  "
        f"dump rel_fro={[round(r.get('rel_fro') or -1, 4) for r in rows]}")
    C.write_status(stage="parity", verdict=rec["verdict"], finished=True)
    return 0 if rec["verdict"] == "PASS" else EXIT_PARITY_FAIL


# ============================================================ pilot

def _trace_summary(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    try:
        import parse_trace
        return parse_trace.analyze_file(path)["summary"]
    except Exception as e:  # noqa: BLE001
        log(f"    trace parse failed: {e}")
        return {"error": str(e)}


def _g(d: Optional[Dict[str, Any]], *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


PILOT_SERVERS = [  # (slot, arm, vit_timing, purpose)
    ("off", "off", True, "eager: W_v from VIT_TIMING, G_v from the trace, TTFT"),
    ("on", "on", True, "graph arm: TTFT, graph-launch evidence"),
    ("off_rot", "off_rot", True, "eager with the graph arm's unfused rotary: dG_rot (D3)"),
    ("off_notiming", "off", False, "eager without timing syncs: LM W_l/G_l for the overlap term (D2)"),
]


def cmd_pilot(a) -> int:
    _stage_start("pilot")
    par = C.OUT / "parity.json"
    parity = C.load_json(par).get("verdict") if par.exists() else "MISSING"
    if parity != "PASS":
        log(f"parity is {parity}; the pilot does not run without it (B7)")
        return EXIT_PARITY_FAIL
    wids = a.workloads.split(",") if getattr(a, "workloads", None) else C.ORDER
    raw = C.RAW / "pilot"
    servers = [s for s in PILOT_SERVERS
               if not (s[0] == "off_rot" and getattr(a, "skip_rot", False))
               and not (s[0] == "off_notiming" and getattr(a, "skip_notiming", False))]
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(),
                           "params": {"n_warm": a.n_warm, "n_profile": a.n_profile, "n_measure": a.n_measure,
                                      "replay_residual_ms": REPLAY_RESIDUAL_MS, "workloads": wids,
                                      "output_len": C.OUTPUT_LEN, "servers": [s[0] for s in servers]},
                           "arms": {}, "cells": {w: {} for w in wids}}
    for slot, arm, timing, purpose in servers:
        if C.stop_requested():
            log("STOP requested before a pilot server; exiting")
            C.write_status(stage="pilot", stopped=True)
            return EXIT_STOPPED
        h = C.launch_server(f"pilot_{slot}", arm, timing)
        env = C.server_env(arm, timing)
        rec["arms"][slot] = {"arm": arm, "vit_timing": timing, "purpose": purpose, "startup_s": h.startup_s,
                             "resolved": C.resolved_config(h.info)}
        try:
            for wid in wids:
                flags, text, exp_tok, exp_patches, note = C.flags_for(wid), *C.WORKLOADS[wid][1:]
                C.write_status(stage="pilot", server=slot, workload=wid)
                log(f"  [{slot}] {wid}: {note}")
                off0 = C.log_offset(h.log_path)
                C.run_bench(f"pilot_{slot}_{wid}_warm", flags, text, a.n_warm, env, raw)
                tdir = C.TRACES / "pilot" / slot / wid
                before = C.trace_files(tdir)
                trace_path = None
                try:
                    C.start_profile(tdir)
                    C.run_bench(f"pilot_{slot}_{wid}_prof", flags, text, a.n_profile, env, raw)
                    C.stop_profile()
                    trace_path = C.wait_for_trace(tdir, before)
                    log(f"    trace: {trace_path}")
                except Exception as e:  # noqa: BLE001
                    log(f"    profiler step failed: {e}")
                meas, scan_meas, extra = None, None, {}
                n_req = a.n_warm + a.n_profile + 3
                if timing:
                    off2 = C.log_offset(h.log_path)
                    meas = C.run_bench(f"pilot_{slot}_{wid}_meas", flags, text, a.n_measure, env, raw)
                    scan_meas = C.scan_log(h.log_path, off2)
                    n_req += a.n_measure
                    if slot == "off" and wid == "R1_256":
                        # D6: output length must not move TTFT at c=1
                        extra["bench_out128"] = C.run_bench(f"pilot_off_R1_256_out128",
                                                            flags + ["--random-output-len", "128"],
                                                            text, a.n_measure, env, raw)
                        n_req += a.n_measure
                scan_win = C.scan_log(h.log_path, off0)
                scan_life = C.scan_log(h.log_path, h.launch_offset)
                ver = C.verify_arm(arm, scan_win, n_req,
                                   expect_shapes=(1 if (arm == "on" and exp_patches) else 0),
                                   info=h.info, scan_life=scan_life)
                cell = {"bench": meas,
                        "timings": C.summarize_timings((scan_meas or {}).get("timings") or []),
                        "req_times": C.summarize_req_times((scan_meas or scan_win).get("req_times") or []),
                        "trace_path": str(trace_path) if trace_path else None,
                        "trace": _trace_summary(trace_path),
                        "captures": scan_win.get("captures"), "stats_last": scan_life.get("stats_last"),
                        "verify": ver, "gpu_used_mib": C.gpu_used(), **extra}
                rec["cells"][wid][slot] = cell
                t = cell["timings"]
                log(f"    ttft_p50={(meas or {}).get('ttft_p50')} ms  vit cpu_wall={t.get('cpu_wall_ms_p50')} "
                    f"gpu_span={t.get('gpu_span_ms_p50')} patches={t.get('patches')}  "
                    f"verify={ver['verdict']}[{ver.get('evidence')}] {ver['reasons']}")
                C.save_json(C.OUT / "pilot.json", rec)   # incremental
        finally:
            C.kill_server(h.proc)

    # ---- predictions (H1, amended form) from the eager arms; gate G1
    preds: Dict[str, Any] = {}
    for wid in wids:
        cells = rec["cells"][wid]
        off, on = cells.get("off", {}), cells.get("on", {})
        rot, nt = cells.get("off_rot"), cells.get("off_notiming")
        tr_v = _g(off, "trace", "Q3_VIT_FORWARD") or {}
        tr_on = _g(on, "trace", "Q3_VIT_FORWARD") or {}
        W_v = _g(off, "timings", "cpu_wall_ms_p50")               # unprofiled CPU wall (D1)
        G_v = tr_v.get("gpu_busy_ms_p50")
        W_prof = tr_v.get("cpu_wall_ms_p50")
        U_trace = tr_v.get("unoverlapped_ms_p50")
        if W_v is not None and G_v is not None:
            U_star, src = round(W_v - G_v, 3), "vit_timing_minus_trace_gpu"
        else:
            U_star, src = U_trace, "trace_unoverlapped"
        dG_rot = None
        if rot is not None:
            g_rot = _g(rot, "trace", "Q3_VIT_FORWARD", "gpu_busy_ms_p50")
            if g_rot is not None and G_v is not None:
                dG_rot = round(g_rot - G_v, 3)
        overlap = None
        if nt is not None:
            st, vt = _g(nt, "trace", "Q3_STEP_EXTEND") or {}, _g(nt, "trace", "Q3_VIT_FORWARD") or {}
            if all(x is not None for x in (st.get("cpu_wall_ms_p50"), vt.get("cpu_wall_ms_p50"),
                                           st.get("gpu_busy_ms_p50"), vt.get("gpu_busy_ms_p50"), G_v)):
                W_l = st["cpu_wall_ms_p50"] - vt["cpu_wall_ms_p50"]
                G_l = st["gpu_busy_ms_p50"] - vt["gpu_busy_ms_p50"]
                overlap = round(min(G_v, max(0.0, W_l - G_l)), 3)
        rot_term = dG_rot or 0.0
        pred_lo = None if U_star is None else round(U_star - REPLAY_RESIDUAL_MS - rot_term, 3)
        pred_point = None if pred_lo is None else round(pred_lo + (overlap or 0.0), 3)
        pred_hi = None if W_v is None else round(W_v - REPLAY_RESIDUAL_MS - rot_term, 3)
        t_off, t_on = _g(off, "bench", "ttft_p50"), _g(on, "bench", "ttft_p50")
        preds[wid] = {
            "patches": _g(off, "timings", "patches") or C.WORKLOADS[wid][3],
            "W_v_ms": W_v, "G_v_ms": G_v, "U_star_ms": U_star, "U_source": src,
            "U_trace_ms": U_trace, "profiler_inflation_ms": (round(W_prof - W_v, 3)
                                                           if (W_prof is not None and W_v is not None) else None),
            "dG_rot_ms": dG_rot, "overlap_ms": overlap,
            "pred_lo_ms": pred_lo, "pred_point_ms": pred_point, "pred_hi_ms": pred_hi,
            "vit_n_launches_off": tr_v.get("n_launches_p50"),
            "vit_n_launches_on": tr_on.get("n_launches_p50"),
            "vit_n_graph_launches_on": tr_on.get("n_graph_launches_p50"),
            "vit_cpu_wall_on_ms": _g(on, "timings", "cpu_wall_ms_p50"),
            "vit_gpu_span_off_ms": _g(off, "timings", "gpu_span_ms_p50"),
            "prefill_step_crit_ms": _g(nt, "trace", "Q3_STEP_EXTEND", "crit_ms_p50") if nt else None,
            "ttft_off_p50": t_off, "ttft_on_p50": t_on,
            "pilot_gain_ms": round(t_off - t_on, 3) if (t_off is not None and t_on is not None) else None,
            "verified": all(_g(cells.get(s), "verify", "verdict") == "VERIFIED" for s in cells),
        }
    rec["predictions"] = preds
    reasons, warnings = [], []
    unverified = [w for w, p in preds.items() if not p["verified"]]
    if unverified:
        reasons.append(f"unverified cells: {unverified}")
    for wid in wids:
        if wid == "R0_text":
            continue
        p = preds[wid]
        ngl, nl_on, nl_off = p["vit_n_graph_launches_on"], p["vit_n_launches_on"], p["vit_n_launches_off"]
        if ngl is None or ngl < 1:
            reasons.append(f"{wid}: no cudaGraphLaunch inside Q3_VIT_FORWARD on the on arm (got {ngl})")
        if nl_on is not None and nl_off is not None and nl_off > 0 and nl_on > 0.2 * nl_off:
            reasons.append(f"{wid}: on-arm kernel launches {nl_on} not far below off-arm {nl_off}")
    img_preds = [p["pred_point_ms"] for w, p in preds.items() if w != "R0_text" and p["pred_point_ms"] is not None]
    max_pred = max(img_preds) if img_preds else None
    if max_pred is None:
        reasons.append("no prediction available (VIT_TIMING and trace both missing)")
    elif max_pred < GATE_MIN_PRED_MS:
        reasons.append(f"max predicted gain {max_pred} ms < {GATE_MIN_PRED_MS} ms")
    o128 = _g(rec["cells"].get("R1_256", {}).get("off", {}), "bench_out128", "ttft_p50")
    o16 = _g(rec["cells"].get("R1_256", {}).get("off", {}), "bench", "ttft_p50")
    if o128 is not None and o16:
        rel = abs(o128 - o16) / o16
        rec["output_len_check"] = {"ttft_out16": o16, "ttft_out128": o128, "rel_diff": round(rel, 4)}
        if rel > 0.05:
            warnings.append(f"D6: TTFT differs between 16 and 128 output tokens by {rel:.1%}; "
                            "run the sweep with --output-len 128 to stay comparable")
    gate = {"verdict": "GO" if not reasons else "STOP", "reasons": reasons, "warnings": warnings,
            "max_pred_point_ms": max_pred, "parity": parity, "timestamp_utc": C.utc_now()}
    rec["gate_g1"] = gate
    C.save_json(C.OUT / "pilot.json", rec)
    C.save_json(C.OUT / "gate_g1.json", gate)
    log("\nPILOT  workload      patches   W_v     G_v    U*     dG_rot  overlap  pred[lo/pt/hi]        pilot_gain  ttft_off  ttft_on")
    for wid, p in preds.items():
        log(f"  {wid:<12} {str(p['patches']):>7}  {str(p['W_v_ms']):>6}  {str(p['G_v_ms']):>6}  {str(p['U_star_ms']):>5}  "
            f"{str(p['dG_rot_ms']):>6}  {str(p['overlap_ms']):>7}  "
            f"{str(p['pred_lo_ms'])}/{str(p['pred_point_ms'])}/{str(p['pred_hi_ms']):<10} "
            f"{str(p['pilot_gain_ms']):>10}  {str(p['ttft_off_p50']):>8}  {str(p['ttft_on_p50']):>7}  "
            f"{'ok' if p['verified'] else 'UNVERIFIED'}")
    log(f"GATE G1: {gate['verdict']}  {reasons}  warnings={warnings}")
    C.write_status(stage="pilot", verdict=gate["verdict"], gate_reasons=reasons, finished=True)
    return 0 if gate["verdict"] == "GO" else EXIT_GATE_STOP


# ============================================================ sweep

def _cell_path(cell: str) -> Path:
    return C.CELLS / f"{cell}.json"


def _cell_ok(cell: str) -> bool:
    p = _cell_path(cell)
    try:
        return p.exists() and C.load_json(p).get("status") == "OK"
    except Exception:  # noqa: BLE001
        return False


def run_cell(wid: str, arm: str, block: int, prompts: int, warmup: int,
             output_len: Optional[int]) -> Dict[str, Any]:
    flags, text, exp_tok, exp_patches, note = C.flags_for(wid), *C.WORKLOADS[wid][1:]
    if output_len:
        flags = flags + ["--random-output-len", str(output_len)]
    cell = f"{wid}__{arm}__b{block}"
    rec: Dict[str, Any] = {"cell": cell, "workload": wid, "arm": arm, "block": block,
                           "num_prompts": prompts, "warmup": warmup,
                           "output_len": output_len or C.OUTPUT_LEN, "timestamp_utc": C.utc_now()}
    t0 = time.time()
    h = C.launch_server(f"sweep_{cell}", arm, False)
    env = C.server_env(arm, False)
    try:
        rec["startup_s"] = h.startup_s
        rec["resolved"] = C.resolved_config(h.info)
        C.run_bench(f"{cell}_warm", flags, text, warmup, env, C.RAW / "sweep")
        rec["gpu_used_mib_after_warmup"] = C.gpu_used()
        off = C.log_offset(h.log_path)
        meas = C.run_bench(cell, flags, text, prompts, env, C.RAW / "sweep")
        scan_all = C.scan_log(h.log_path, h.launch_offset)
        scan_meas = C.scan_log(h.log_path, off)
        rec["bench"] = {k: v for k, v in meas.items() if k != "ttft_ms"}
        rec["req_times"] = C.summarize_req_times(scan_meas.get("req_times") or [])
        rec["captures"], rec["stats_last"] = scan_all.get("captures"), scan_all.get("stats_last")
        rec["effective"] = scan_all.get("effective")
        rec["verify"] = C.verify_arm(arm, scan_all, warmup + prompts + 2,
                                     expect_shapes=(1 if (arm == "on" and exp_patches) else 0), info=h.info)
        rec["status"] = "OK" if (meas.get("status") == "OK" and meas.get("failures", 0) == 0) \
            else meas.get("status", "HAS_FAILURES")
    finally:
        C.kill_server(h.proc)
    rec["elapsed_s"] = round(time.time() - t0, 1)
    return rec


def aggregate_sweep(wids: List[str], blocks_for) -> Dict[str, Any]:
    out: Dict[str, Any] = {"generated_utc": C.utc_now(), "workloads": {}}
    for wid in wids:
        rows = []
        for b in range(1, blocks_for(wid) + 1):
            po, pn = _cell_path(f"{wid}__off__b{b}"), _cell_path(f"{wid}__on__b{b}")
            if not (po.exists() and pn.exists()):
                continue
            o, n = C.load_json(po), C.load_json(pn)
            to, tn = _g(o, "bench", "ttft_p50"), _g(n, "bench", "ttft_p50")
            if to is None or tn is None or to <= 0:
                continue
            rows.append({"block": b, "off_p50": to, "on_p50": tn, "saving_ms": round(to - tn, 3),
                         "effect_pct": round((tn - to) / to * 100.0, 2),
                         "off_mean": _g(o, "bench", "ttft_mean"), "on_mean": _g(n, "bench", "ttft_mean"),
                         "tpot_off": _g(o, "bench", "tpot_p50"), "tpot_on": _g(n, "bench", "tpot_p50"),
                         "verified": (_g(o, "verify", "verdict") == "VERIFIED"
                                      and _g(n, "verify", "verdict") == "VERIFIED"),
                         "status_ok": (o.get("status") == "OK" and n.get("status") == "OK"),
                         "capture_ms": ((n.get("captures") or [{}])[0].get("capture_ms")
                                        if n.get("captures") else None)})
        if not rows:
            continue
        effs = [r["effect_pct"] for r in rows]
        savs = [r["saving_ms"] for r in rows]
        eff = statistics.median(effs)
        spread = round(max(effs) - min(effs), 2) if len(effs) > 1 else None
        gate = ("PASS" if spread is not None and (spread <= SPREAD_GATE_PP or spread <= abs(eff) / 2)
                else ("SINGLE_BLOCK" if spread is None else "WEAK"))
        sd = statistics.stdev(savs) if len(savs) >= 2 else None
        out["workloads"][wid] = {
            "blocks": rows, "n_blocks": len(rows),
            "effect_pct_median": round(eff, 2),
            "saving_ms_median": round(statistics.median(savs), 3),
            "saving_ms_mean": round(sum(savs) / len(savs), 3),
            "saving_ms_sd": round(sd, 3) if sd is not None else None,
            "saving_ms_se": round(sd / math.sqrt(len(savs)), 3) if sd is not None else None,
            "spread_pp": spread, "gate": gate,
            "all_verified": all(r["verified"] and r["status_ok"] for r in rows),
            "expected_patches": C.WORKLOADS[wid][3],
        }
    return out


def cmd_sweep(a) -> int:
    _stage_start("sweep")
    wids = a.workloads.split(",") if getattr(a, "workloads", None) else C.ORDER

    def blocks_for(w: str) -> int:
        return a.blocks_large if w in C.LARGE_WORKLOADS else a.blocks

    blocks = [(wid, b) for wid in wids for b in range(1, blocks_for(wid) + 1)]
    # a block is done only when both of its cells are OK; a half block is re-run (O2)
    todo = []
    for wid, b in blocks:
        ok_off, ok_on = _cell_ok(f"{wid}__off__b{b}"), _cell_ok(f"{wid}__on__b{b}")
        if ok_off and ok_on:
            continue
        for arm in ("off", "on"):
            _cell_path(f"{wid}__{arm}__b{b}").unlink(missing_ok=True)
        todo.append((wid, b))
    n_done = len(blocks) - len(todo)
    log(f"sweep: {len(blocks)} blocks ({len(blocks) * 2} cells), {n_done} done, {len(todo)} to run  "
        f"(blocks={a.blocks}/{a.blocks_large} prompts={a.prompts} warmup={a.warmup} output_len={a.output_len or C.OUTPUT_LEN})")
    durations: List[float] = []
    consecutive_failures = 0
    for i, (wid, b) in enumerate(todo):
        if C.stop_requested():
            log(f"STOP requested; {len(todo) - i} blocks left")
            C.write_status(stage="sweep", stopped=True, blocks_left=len(todo) - i)
            C.save_json(C.OUT / "sweep.json", aggregate_sweep(wids, blocks_for))
            return EXIT_STOPPED
        order = ["off", "on"] if b % 2 == 1 else ["on", "off"]   # A/B then B/A cancels linear drift
        eta_min = round(statistics.mean(durations) * (len(todo) - i) / 60.0, 1) if durations else None
        t0 = time.time()
        for arm in order:
            cell = f"{wid}__{arm}__b{b}"
            C.write_status(stage="sweep", cell=cell, blocks_done=n_done + i, blocks_total=len(blocks), eta_min=eta_min)
            log(f"\n=== cell {cell}  (block {n_done + i + 1}/{len(blocks)}, eta {eta_min} min) ===")
            try:
                rec = run_cell(wid, arm, b, a.prompts, a.warmup, a.output_len)
                consecutive_failures = 0 if rec.get("status") == "OK" else consecutive_failures + 1
            except Exception as e:  # noqa: BLE001
                log(f"  cell failed: {e}\n{traceback.format_exc()[-600:]}")
                rec = {"cell": cell, "workload": wid, "arm": arm, "block": b, "status": "FAILED",
                       "error": str(e), "timestamp_utc": C.utc_now()}
                C.kill_server(None)
                consecutive_failures += 1
            C.save_json(_cell_path(cell), rec)
            bench = rec.get("bench") or {}
            log(f"  {cell}: ttft_p50={bench.get('ttft_p50')} ms  status={rec.get('status')}  "
                f"verify={_g(rec, 'verify', 'verdict')}[{_g(rec, 'verify', 'evidence')}] {_g(rec, 'verify', 'reasons')}  "
                f"{rec.get('elapsed_s')}s")
            if consecutive_failures >= 2:
                log("two consecutive failed cells; aborting the sweep")
                C.write_status(stage="sweep", aborted="two consecutive failed cells")
                C.save_json(C.OUT / "sweep.json", aggregate_sweep(wids, blocks_for))
                return 6
        durations.append(time.time() - t0)
        C.save_json(C.OUT / "sweep.json", aggregate_sweep(wids, blocks_for))   # incremental, per block
    agg = aggregate_sweep(wids, blocks_for)
    C.save_json(C.OUT / "sweep.json", agg)
    log("\nSWEEP  workload      patches  off_p50   on_p50   saving   se     effect   spread  gate")
    for wid, w in agg["workloads"].items():
        r = w["blocks"]
        log(f"  {wid:<12} {w['expected_patches']:>7}  {statistics.median(x['off_p50'] for x in r):>7.2f}  "
            f"{statistics.median(x['on_p50'] for x in r):>7.2f}  {w['saving_ms_median']:>+6.2f}  "
            f"{str(w['saving_ms_se']):>5}  {w['effect_pct_median']:>+6.2f}%  {str(w['spread_pp']):>6}  {w['gate']}"
            f"{'' if w['all_verified'] else '  UNVERIFIED'}")
    C.write_status(stage="sweep", blocks_done=len(blocks), blocks_total=len(blocks), finished=True)
    return 0


# ============================================================ mixed

def _sequence(path: Path, start: int) -> List[Dict[str, Any]]:
    """VIT_TIMING lines in order, each tagged with the capture (if any) that preceded it."""
    text = C.RE_TORCHCODEC.sub("", C.read_log(path, start))
    seq, pending = [], None
    for line in text.splitlines():
        m = C.RE_CAPTURE.search(line)
        if m:
            pending = {"key": m.group(1), "capture_ms": float(m.group(3)),
                       "reserved_mib": int(m.group(4)) if m.group(4) else None}
            continue
        m = C.RE_TIMING.search(line)
        if m:
            seq.append({"patches": int(m.group(1)), "graph": int(m.group(3)),
                        "cpu_wall_ms": float(m.group(4)), "gpu_span_ms": float(m.group(5)),
                        "captured": pending is not None,
                        "capture_ms": pending["capture_ms"] if pending else None,
                        "reserved_mib": pending["reserved_mib"] if pending else None})
            pending = None
    return seq


class MemGuard:
    """Samples GPU memory during the mixed run; kills the client if the device is nearly full (O1)."""

    def __init__(self, limit_frac: float = 0.95):
        self.total = C.gpu_total()
        self.limit = limit_frac * self.total if self.total > 0 else None
        self.peak, self.tripped, self._stop = 0, False, threading.Event()

    def _run(self):
        while not self._stop.is_set():
            u = C.gpu_used()
            self.peak = max(self.peak, u)
            if self.limit and u > self.limit:
                self.tripped = True
                subprocess.run(["pkill", "-9", "-f", "sglang.benchmark.serving"], capture_output=True)
            self._stop.wait(10)

    def __enter__(self):
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.t.join(timeout=15)


def cmd_mixed(a) -> int:
    _stage_start("mixed")
    rec: Dict[str, Any] = {"timestamp_utc": C.utc_now(),
                           "params": {"prompts": a.prompts, "warmup": a.warmup, "flags": C.MIXED_FLAGS,
                                      "expected_distinct_keys": 165, "expected_hit_rate": 0.45},
                           "arms": {}}
    text_flags, text_tok = C.flags_for("R0_text"), C.WORKLOADS["R0_text"][1]
    for arm in ("off", "on"):
        if C.stop_requested():
            C.write_status(stage="mixed", stopped=True)
            return EXIT_STOPPED
        C.write_status(stage="mixed", arm=arm)
        h = C.launch_server(f"mixed_{arm}", arm, True)
        env = C.server_env(arm, True)
        try:
            # text-only warmup so no image shape is captured before the run
            C.run_bench(f"mixed_{arm}_warm", text_flags, text_tok, a.warmup, env, C.RAW / "mixed")
            off = C.log_offset(h.log_path)
            with MemGuard() as mg:
                meas = C.run_bench(f"mixed_{arm}", C.MIXED_FLAGS, 128, a.prompts, env, C.RAW / "mixed")
            seq = _sequence(h.log_path, off)
            scan = C.scan_log(h.log_path, off)
            ttft_all = meas.get("ttft_ms") or []
            # align by the tail: the client's own warmup request also produces a timing line
            seq_tail = seq[-len(ttft_all):] if ttft_all and len(seq) >= len(ttft_all) else seq
            joined = [{"i": i, "ttft_ms": t, **s} for i, (t, s) in enumerate(zip(ttft_all, seq_tail))
                      if t is not None]
            first = [j for j in joined if j["captured"]]
            rep = [j for j in joined if not j["captured"]]
            rec["arms"][arm] = {
                "bench": {k: v for k, v in meas.items() if k != "ttft_ms"},
                "ttft_ms": ttft_all,
                "n_requests": len(joined),
                "n_captures": len(scan.get("captures") or []),
                "capture_ms_total": _g(scan, "stats_last", "capture_ms_total"),
                "capture_ms_p50": C.median([c["capture_ms"] for c in (scan.get("captures") or [])]),
                "capture_ms_max": max([c["capture_ms"] for c in (scan.get("captures") or [])], default=None),
                "distinct_patch_counts": len({s["patches"] for s in seq_tail}),
                "ttft_first_seen_p50": C.median([j["ttft_ms"] for j in first]),
                "ttft_repeat_p50": C.median([j["ttft_ms"] for j in rep]),
                "vit_wall_first_seen_p50": C.median([j["cpu_wall_ms"] for j in first]),
                "vit_wall_repeat_p50": C.median([j["cpu_wall_ms"] for j in rep]),
                "verify": C.verify_arm(arm, scan, a.prompts + 1, expect_shapes=(-1 if arm == "on" else 0),
                                       info=h.info),
                "gpu_used_mib_end": C.gpu_used(), "gpu_used_mib_peak": mg.peak,
                "mem_guard_tripped": mg.tripped,
                "reserved_mib_last_capture": (scan.get("captures") or [{}])[-1].get("reserved_mib")
                if scan.get("captures") else None,
                "per_request": joined,
            }
            log(f"  [{arm}] ttft p50={meas.get('ttft_p50')} mean={meas.get('ttft_mean')} p99={meas.get('ttft_p99')}  "
                f"captures={rec['arms'][arm]['n_captures']} distinct_shapes={rec['arms'][arm]['distinct_patch_counts']}  "
                f"first_seen_p50={rec['arms'][arm]['ttft_first_seen_p50']} repeat_p50={rec['arms'][arm]['ttft_repeat_p50']}  "
                f"peak_mem={mg.peak} MiB tripped={mg.tripped}")
        finally:
            C.kill_server(h.proc)
        C.save_json(C.OUT / "mixed.json", rec)
    o, n = rec["arms"].get("off", {}), rec["arms"].get("on", {})
    if o and n and o.get("per_request") and n.get("per_request"):
        idx_first = {j["i"] for j in n["per_request"] if j["captured"]}
        off_by_i = {j["i"]: j["ttft_ms"] for j in o["per_request"]}
        on_by_i = {j["i"]: j["ttft_ms"] for j in n["per_request"]}
        common = sorted(set(off_by_i) & set(on_by_i))
        first_c = [i for i in common if i in idx_first]
        rep_c = [i for i in common if i not in idx_first]
        rec["summary"] = {
            "n_common": len(common),
            "ttft_mean_off": C.mean([off_by_i[i] for i in common]),
            "ttft_mean_on": C.mean([on_by_i[i] for i in common]),
            "ttft_p50_off": o["bench"].get("ttft_p50"), "ttft_p50_on": n["bench"].get("ttft_p50"),
            "ttft_p99_off": o["bench"].get("ttft_p99"), "ttft_p99_on": n["bench"].get("ttft_p99"),
            "hit_rate_by_key": round(1 - n["n_captures"] / max(1, n["n_requests"]), 3),
            "first_seen_penalty_ms_p50": (C.median([on_by_i[i] for i in first_c]) or 0)
                                         - (C.median([off_by_i[i] for i in first_c]) or 0),
            "repeat_gain_ms_p50": (C.median([off_by_i[i] for i in rep_c]) or 0)
                                  - (C.median([on_by_i[i] for i in rep_c]) or 0),
            "mean_penalty_ms": None,
        }
        if rec["summary"]["ttft_mean_off"] is not None and rec["summary"]["ttft_mean_on"] is not None:
            rec["summary"]["mean_penalty_ms"] = round(rec["summary"]["ttft_mean_on"] - rec["summary"]["ttft_mean_off"], 3)
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
    pa = sub.add_parser("parity")
    pa.add_argument("--with-default-interp", action="store_true",
                    help="also run the graph arm without --enable-precise-embedding-interpolation (upstream-issue evidence)")
    p = sub.add_parser("pilot")
    p.add_argument("--n-warm", type=int, default=5)
    p.add_argument("--n-profile", type=int, default=5)
    p.add_argument("--n-measure", type=int, default=30)
    p.add_argument("--workloads", help="comma-separated subset of " + ",".join(C.ORDER))
    p.add_argument("--skip-rot", action="store_true", help="skip the off_rot server (no dG_rot term)")
    p.add_argument("--skip-notiming", action="store_true", help="skip the off_notiming server (no overlap term)")
    s = sub.add_parser("sweep")
    s.add_argument("--blocks", type=int, default=BLOCKS_DEFAULT)
    s.add_argument("--blocks-large", type=int, default=BLOCKS_LARGE, help="blocks for " + ",".join(C.LARGE_WORKLOADS))
    s.add_argument("--prompts", type=int, default=200)
    s.add_argument("--warmup", type=int, default=20)
    s.add_argument("--output-len", type=int, default=None, help="override OUTPUT_LEN (e.g. 128 if the D6 check fails)")
    s.add_argument("--workloads")
    m = sub.add_parser("mixed")
    m.add_argument("--prompts", type=int, default=300)
    m.add_argument("--warmup", type=int, default=5)
    sub.add_parser("status")
    a = ap.parse_args()
    rc = {"check": cmd_check, "parity": cmd_parity, "pilot": cmd_pilot,
          "sweep": cmd_sweep, "mixed": cmd_mixed, "status": cmd_status}[a.stage](a)
    sys.exit(rc)


if __name__ == "__main__":
    main()
