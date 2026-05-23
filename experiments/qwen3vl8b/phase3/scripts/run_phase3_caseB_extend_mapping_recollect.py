#!/usr/bin/env python3
"""
Phase 4 support — re-collect Case B SGLang EXTEND **mapping** (graph-off) trace.

Why: the original Phase-3 EXTEND-supplement mapping trace for Case B
(sglang_extend_mapping/) was found CORRUPT during Phase-4 triage
(gzip -t: "unexpected end of file"; the analyze script raised EOFError). The
corrupt timestamped dir was quarantined (renamed CORRUPT_*). Case B graph-on
EXTEND formal remains unavailable (prior 8-attempt deviation) and is NOT retried
here. This script re-runs ONLY the graph-off mapping capture, reusing the proven
collect_extend() mechanism, so Case B prefill-stage triage has a usable trace.

GPU 1 only. SGLang only. No vLLM. No source changes.
"""
import json, sys
from datetime import datetime, timezone
import run_phase3_extend as E

CASE = "caseB_longprefill"


def main():
    E.os.chdir(E.LAB)
    E.LOGS.mkdir(parents=True, exist_ok=True)
    E.log("=== Case B EXTEND mapping re-collect (graph-off, GPU 1) ===")
    u = E.gpu_used()
    if not (0 <= u < 2000):
        E.log(f"STOP: GPU {E.GPU} not idle (used={u} MiB)")
        sys.exit(1)

    cfg = E.CASES[CASE]
    res = E.collect_extend(CASE, cfg, graph_off=True)
    E.log(f"  result: ok={res['ok']} stages={res.get('stages')} size={res.get('size_bytes',0)/1e6:.1f}MB")

    # record outcome in metadata without clobbering existing fields
    meta_path = E.META / f"{CASE}_meta.json"
    meta = json.load(open(meta_path)) if meta_path.exists() else {"case": CASE}
    meta.setdefault("extend_mapping_recollect", {})
    meta["extend_mapping_recollect"] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "original sglang_extend_mapping gz corrupt (EOFError); quarantined as CORRUPT_*",
        "gpu": E.GPU, "graph_off": True, "num_steps": E.NUM_STEPS,
        "ok": res["ok"], "stages": res.get("stages"), "size_bytes": res.get("size_bytes"),
        "trace_dir": res.get("trace_dir"),
    }
    json.dump(meta, open(meta_path, "w"), indent=2)
    E.log(f"  metadata updated: {meta_path}")
    if not res["ok"]:
        E.log("  WARNING: re-collect did not capture EXTEND")
        sys.exit(2)
    E.log("=== re-collect complete ===")


if __name__ == "__main__":
    main()
