#!/usr/bin/env python3
"""Re-run the mixed stage's engagement verification from the saved server logs.

The first mixed run recorded the graph arm as UNVERIFIED only because `expect_shapes=-1`
("any number of shapes") was compared as a capture count. This re-applies `verify_arm` from
the fixed `q3_common` to the synced server logs and rewrites the two `verify` entries in
mixed.json, keeping the original ones under `verify_original`.

Usage: reverify_mixed.py [--results DIR] [--logs DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import q3_common as C  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE.parent / "results")
    ap.add_argument("--logs", type=Path, default=None, help="dir with mixed_<arm>_server.log (default results/raw/logs/q3)")
    a = ap.parse_args()
    logs = a.logs or (a.results / "raw" / "logs" / "q3")
    p = a.results / "mixed.json"
    rec = json.loads(p.read_text())
    n_req = rec["params"]["prompts"] + 1
    for arm in ("off", "on"):
        log_path = logs / f"mixed_{arm}_server.log"
        if not log_path.exists():
            print(f"{arm}: log missing at {log_path}")
            continue
        text = log_path.read_text(errors="replace")
        # the warmup is text-only, so the window starts at the first image request: the first
        # VIT_TIMING line (both arms log it); everything before is startup + text warmup
        start = text.find("VIT_TIMING")
        start = 0 if start < 0 else len(text[:start].encode())
        scan = C.scan_log(log_path, start)
        ver = C.verify_arm(arm, scan, n_req, expect_shapes=(-1 if arm == "on" else 0))
        rec["arms"][arm].setdefault("verify_original", rec["arms"][arm].get("verify"))
        rec["arms"][arm]["verify"] = {**ver, "reverified_from": str(log_path.name)}
        print(f"{arm}: {ver['verdict']} {ver['reasons']} evidence={ver.get('evidence')} "
              f"captures={ver.get('captures')} calls={ver.get('calls')} prefill_batches={ver.get('prefill_batches')}")
    p.write_text(json.dumps(rec, indent=2, default=str))
    print(f"rewrote {p}")


if __name__ == "__main__":
    main()
