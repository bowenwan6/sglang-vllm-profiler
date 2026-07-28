#!/usr/bin/env python3
"""R6.1 verdict computation. Applies pre-declared verdict rules (see
`results/R6_fix_value_validation/R6.1_correctness/protocol.md`) to the
per-leg JSON captures produced by R6_1_client.py, and to the per-server
log-scan tallies produced by the orchestrator.

Verdict rules (pre-declared in R6.1 protocol; NOT to be edited after
observing results):

  PASS iff ALL of:
    (a) leg_a_fork_default_run1.requests[*].response_text ==
        leg_a_fork_default_run2.requests[*].response_text  (bit-identical)
    (b) leg_b_fork_pcg_image.requests[*].response_text ==
        leg_a_fork_default_run1.requests[*].response_text  (bit-identical)
    (c) leg_c_stock_default_image.requests[*].response_text ==
        leg_a_fork_default_run1.requests[*].response_text  (bit-identical)
    (d) leg_d_stock_pcg_text.requests[*].response_text ==
        leg_dprime_fork_pcg_text.requests[*].response_text  (bit-identical)
    (e) mixed_safety_summary.assertions == 0 AND
        mixed_safety_summary.fallbacks == 0 AND
        mixed_safety_summary.inference_recompiles == 0 AND
        mixed_safety_summary.request_failures == 0

  FAIL iff ANY of:
    (a) fails, or (c) fails, or (d) fails, or (e) fails, or
    any leg has an HTTP error / request exception, or
    any per-server log recorded an unhandled traceback.

  AMBIGUOUS / R7_REQUIRED iff:
    (a), (c), (d), (e) all pass BUT (b) fails.
    (Cross-backend / cross-config difference on the image path that is
    not accompanied by a text-path divergence — cannot be automatically
    attributed to normal PCG-vs-eager bf16 noise; needs R7.)

  Additional diagnostic (never changes verdict on its own):
    (f) baseline stock_default_text_only vs stock_pcg_text_only ==
        used to characterise "normal" stock PCG-vs-eager delta if any.
        If (f) diverges AND (b) diverges by similar magnitude, R7 has a
        starting hypothesis; still AMBIGUOUS.

The script writes:
    verdict.md   — human-readable per-leg comparison + final verdict
    verdict.json — machine-readable structure with the same verdict
"""
from __future__ import annotations

import argparse
import difflib
import json
import sys
from pathlib import Path
from typing import Any


def load_leg(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return {"missing": False, "path": str(path),
            "data": json.loads(path.read_text())}


def texts_of(leg: dict[str, Any]) -> list[str | None]:
    if leg.get("missing"):
        return []
    return [r.get("response_text") for r in leg["data"].get("requests", [])]


def any_request_error(leg: dict[str, Any]) -> str | None:
    if leg.get("missing"):
        return f"leg_missing:{leg.get('path')}"
    for r in leg["data"].get("requests", []):
        if r.get("error") is not None:
            return f"request_error:idx={r.get('idx')}:{r.get('error')}"
        if r.get("http_status") not in (200, None):
            return f"http_{r.get('http_status')}"
    return None


def compare_texts(a: list[str | None], b: list[str | None]
                  ) -> tuple[bool, list[dict[str, Any]]]:
    per_prompt = []
    all_ok = True
    for i, (x, y) in enumerate(zip(a, b)):
        equal = x == y
        if not equal:
            all_ok = False
        first_diff = None
        if not equal and x is not None and y is not None:
            for j, (cx, cy) in enumerate(zip(x, y)):
                if cx != cy:
                    first_diff = j
                    break
            if first_diff is None:
                first_diff = min(len(x), len(y))
        per_prompt.append({
            "idx": i, "equal": equal, "len_a": len(x) if x is not None else None,
            "len_b": len(y) if y is not None else None,
            "first_diff_offset": first_diff,
        })
    if len(a) != len(b):
        all_ok = False
        per_prompt.append({"note": f"count_mismatch: len(a)={len(a)} len(b)={len(b)}"})
    return all_ok, per_prompt


def diff_snippet(a: str, b: str, n_lines: int = 3) -> str:
    if a is None or b is None:
        return "(one side is null)"
    aa = a.splitlines(keepends=False)
    bb = b.splitlines(keepends=False)
    diff = list(difflib.unified_diff(aa, bb, lineterm="",
                                     fromfile="A", tofile="B", n=n_lines))
    return "\n".join(diff[:40])


def render_leg_result(name: str, leg: dict[str, Any]) -> list[str]:
    lines = [f"### Leg `{name}`", ""]
    if leg.get("missing"):
        lines.append(f"- **MISSING**: `{leg['path']}`")
        return lines
    meta = leg["data"].get("meta", {})
    reqs = leg["data"].get("requests", [])
    lines.append(f"- source: `{leg['path']}`")
    lines.append(f"- mode: `{meta.get('mode')}`  fixture_sha256: "
                 f"`{meta.get('fixture_sha256')}`")
    lines.append(f"- requests: {len(reqs)}")
    for r in reqs:
        status = "OK" if r.get("error") is None and r.get("http_status") == 200 else "ERR"
        lines.append(f"  - idx={r['idx']} kind={r['kind']} status={status} "
                     f"len={len(r['response_text']) if r.get('response_text') else 'n/a'} "
                     f"latency={r.get('latency_s'):.3f}s")
    return lines


def render_compare(name: str, a_leg: dict[str, Any], b_leg: dict[str, Any],
                   ok: bool, per_prompt: list[dict[str, Any]]
                   ) -> list[str]:
    lines = [f"### Compare `{name}`",
             f"- match_all: **{ok}**"]
    for pp in per_prompt:
        if "note" in pp:
            lines.append(f"- {pp['note']}")
        else:
            lines.append(f"- idx={pp['idx']} equal={pp['equal']} "
                         f"len_a={pp['len_a']} len_b={pp['len_b']} "
                         f"first_diff_offset={pp['first_diff_offset']}")
    if not ok and not a_leg.get("missing") and not b_leg.get("missing"):
        for i, (x, y) in enumerate(zip(texts_of(a_leg), texts_of(b_leg))):
            if x != y:
                lines.append(f"\n<details><summary>diff for idx={i}</summary>\n\n```diff")
                lines.append(diff_snippet(x or "", y or ""))
                lines.append("```\n</details>")
                break
    return lines


def load_safety(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return {"missing": False, **json.loads(path.read_text())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path,
                    help="directory containing leg_*.json and safety_summary.json")
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    ind = args.in_dir
    leg_a1 = load_leg(ind / "leg_a_fork_default_run1.json")
    leg_a2 = load_leg(ind / "leg_a_fork_default_run2.json")
    leg_b  = load_leg(ind / "leg_b_fork_pcg_image.json")
    leg_c  = load_leg(ind / "leg_c_stock_default_image.json")
    leg_d  = load_leg(ind / "leg_d_stock_pcg_text.json")
    leg_dp = load_leg(ind / "leg_dprime_fork_pcg_text.json")
    leg_f1 = load_leg(ind / "leg_f_stock_default_text.json")
    leg_f2 = load_leg(ind / "leg_f_stock_pcg_text.json")
    safety = load_safety(ind / "safety_summary.json")
    selection_path = ind / "monitor_selection.json"
    selection = (json.loads(selection_path.read_text())
                 if selection_path.exists() else None)

    per_leg_errs = {
        "a1": any_request_error(leg_a1), "a2": any_request_error(leg_a2),
        "b":  any_request_error(leg_b),  "c":  any_request_error(leg_c),
        "d":  any_request_error(leg_d),  "dp": any_request_error(leg_dp),
        "f1": any_request_error(leg_f1), "f2": any_request_error(leg_f2),
    }

    a_ok, a_per = compare_texts(texts_of(leg_a1), texts_of(leg_a2))
    b_ok, b_per = compare_texts(texts_of(leg_b),  texts_of(leg_a1))
    c_ok, c_per = compare_texts(texts_of(leg_c),  texts_of(leg_a1))
    d_ok, d_per = compare_texts(texts_of(leg_d),  texts_of(leg_dp))
    f_ok, f_per = compare_texts(texts_of(leg_f1), texts_of(leg_f2))

    e_ok = (not safety.get("missing")
            and safety.get("assertions", 1) == 0
            and safety.get("fallbacks", 1) == 0
            and safety.get("inference_recompiles", 1) == 0
            and safety.get("request_failures", 1) == 0)

    any_err = any(v is not None for v in per_leg_errs.values())
    verdict = "PASS"
    reasons: list[str] = []

    if any_err:
        verdict = "FAIL"
        for k, v in per_leg_errs.items():
            if v is not None:
                reasons.append(f"leg_{k}_error: {v}")

    if not a_ok:
        verdict = "FAIL"; reasons.append("(a) fork-default same-run repeat NOT bit-identical")
    if not c_ok:
        verdict = "FAIL"; reasons.append("(c) stock-default != fork-default (fix affects PCG-off path)")
    if not d_ok:
        verdict = "FAIL"; reasons.append("(d) stock-PCG text != fork-PCG text (fix perturbs text-only PCG)")
    if not e_ok:
        verdict = "FAIL"; reasons.append(f"(e) mixed-safety subtest failed: {safety}")

    if verdict == "PASS" and not b_ok:
        verdict = "AMBIGUOUS"
        reasons.append("(b) fork-PCG image != fork-default image; "
                       "cannot auto-attribute to normal PCG-vs-eager noise → R7_REQUIRED")

    md_lines: list[str] = [
        f"# R6.1 verdict — **{verdict}**",
        "",
        "> Verdict rules were pre-declared in "
        "[`protocol.md`](protocol.md) BEFORE any leg was run. This "
        "file computes verdicts from the raw JSON captures under `raw/` "
        "and the safety-log tally under `raw/safety_summary.json`.",
        "",
    ]
    if selection is not None:
        md_lines.extend([
            "## GPU selection (from `raw/monitor_selection.json`)",
            "",
            f"- **Selected GPU ID:** {selection.get('selected_gpu_id')}",
            f"- Idle streak start (UTC): `{selection.get('idle_start_utc')}`",
            f"- Qualified (UTC): `{selection.get('qualified_utc')}`",
            f"- Idle hold requirement: {selection.get('idle_hold_s')} s "
            f"(mem ≤ {selection.get('mem_threshold_mib')} MiB, "
            f"util ≤ {selection.get('util_threshold_pct')} %, "
            f"0 compute PIDs, polled every "
            f"{selection.get('poll_interval_s')} s)",
            f"- Final pre-launch check (UTC): `{selection.get('prelaunch_utc')}` "
            f"→ `{selection.get('prelaunch_state')}`",
            "",
        ])
    else:
        md_lines.extend([
            "## GPU selection",
            "",
            "- `raw/monitor_selection.json` not found — this run was "
            "not driven by `monitor_idle_gpu.py`. Selected GPU is "
            f"whatever was passed via `R6_GPU_ID`.",
            "",
        ])
    if reasons:
        md_lines.append("**Reasons for non-PASS:**")
        for r in reasons:
            md_lines.append(f"- {r}")
        md_lines.append("")

    md_lines.extend([
        "## Diagnostic (does not change verdict on its own)",
        "",
        f"- **(f) baseline stock text default vs stock text PCG**: match_all=**{f_ok}**",
    ])
    for pp in f_per:
        if "note" in pp:
            md_lines.append(f"  - {pp['note']}")
        else:
            md_lines.append(f"  - idx={pp['idx']} equal={pp['equal']} "
                            f"first_diff_offset={pp['first_diff_offset']}")

    md_lines.append("")
    md_lines.append("## Per-leg detail")
    md_lines.append("")
    for name, leg in [("a1_fork_default_run1", leg_a1),
                      ("a2_fork_default_run2", leg_a2),
                      ("b_fork_pcg_image", leg_b),
                      ("c_stock_default_image", leg_c),
                      ("d_stock_pcg_text", leg_d),
                      ("dp_fork_pcg_text", leg_dp),
                      ("f1_stock_default_text", leg_f1),
                      ("f2_stock_pcg_text", leg_f2)]:
        md_lines.extend(render_leg_result(name, leg))
        md_lines.append("")

    md_lines.append("## Comparisons")
    md_lines.append("")
    md_lines.extend(render_compare("a: fork-default run1 vs run2", leg_a1, leg_a2, a_ok, a_per))
    md_lines.append("")
    md_lines.extend(render_compare("b: fork-PCG image vs fork-default image", leg_b, leg_a1, b_ok, b_per))
    md_lines.append("")
    md_lines.extend(render_compare("c: stock-default image vs fork-default image", leg_c, leg_a1, c_ok, c_per))
    md_lines.append("")
    md_lines.extend(render_compare("d: stock-PCG text vs fork-PCG text", leg_d, leg_dp, d_ok, d_per))
    md_lines.append("")

    md_lines.append("## Mixed-safety subtest (e)")
    md_lines.append("")
    md_lines.append(f"- source: `{safety.get('path')}`")
    md_lines.append(f"- request_failures: {safety.get('request_failures')}")
    md_lines.append(f"- assertions: {safety.get('assertions')}")
    md_lines.append(f"- fallbacks: {safety.get('fallbacks')}")
    md_lines.append(f"- inference_recompiles: {safety.get('inference_recompiles')}")
    md_lines.append(f"- other_notes: {safety.get('notes')}")

    args.out_md.write_text("\n".join(md_lines) + "\n")
    args.out_json.write_text(json.dumps({
        "verdict": verdict,
        "reasons": reasons,
        "compare": {"a": a_ok, "b": b_ok, "c": c_ok, "d": d_ok, "e": e_ok,
                    "f_diagnostic": f_ok},
        "leg_errors": per_leg_errs,
        "safety": safety,
        "selection": selection,
    }, indent=2, sort_keys=True))
    print(f"VERDICT={verdict}")
    for r in reasons:
        print(f"REASON: {r}")
    return 0 if verdict == "PASS" else (1 if verdict == "FAIL" else 2)


if __name__ == "__main__":
    sys.exit(main())
