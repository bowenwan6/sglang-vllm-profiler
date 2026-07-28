#!/usr/bin/env python3
"""R6.1 amended verdict (Protocol Amendment A, 2026-07-28).

Computes the three-tier verdict from a raw/ directory populated by
`scripts/run_R6_1_amended.sh`. CPU-only; requires transformers for
the Qwen3-VL tokenizer.

Inputs expected under --in-dir (all optional; missing files -> that
comparison's verdict falls back to FAIL_MISSING or per-tier N/A):

  # Matched cold-cache repeats (each is client output of one fresh
  # server's first leg)
  cold/stock_default_image_A.json          cold/stock_default_image_B.json
  cold/fork_default_image_A.json           cold/fork_default_image_B.json
  cold/stock_pcg_text_A.json               cold/stock_pcg_text_B.json
  cold/fork_pcg_text_A.json                cold/fork_pcg_text_B.json
  cold/fork_pcg_image_A.json               cold/fork_pcg_image_B.json

  # Direct stock-PCG image negative control
  neg/stock_pcg_image.json
  neg/stock_pcg_image_classification.json  # {result, reason, log_snippet}

  # Fork-PCG mixed-modality interleaved safety subtest
  fork_pcg_interleaved.json
  fork_pcg_interleaved_safety.json         # {startup_warmup_recompiles,
                                             #  post_ready_recompiles,
                                             #  per_leg_recompiles,
                                             #  assertions, fallbacks,
                                             #  request_failures}

  launch_context.json                      # runner-emitted launch identity

Writes:
  verdict_amended.md
  verdict_amended.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SNAP_PATH = ("/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/"
             "snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b")
ENVELOPE_FLOOR_K = 2


def load_leg(p: Path) -> list[dict[str, Any]] | None:
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("requests", [])


def load_json(p: Path) -> dict[str, Any] | None:
    if not p.exists():
        return None
    return json.loads(p.read_text())


def texts_of(reqs: list[dict[str, Any]] | None) -> list[str | None]:
    if reqs is None:
        return []
    return [r.get("response_text") for r in reqs]


def levenshtein(a: list[int], b: list[int]) -> int:
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b) + 1))
    for i, ai in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, bj in enumerate(b, 1):
            cur[j] = min(cur[j - 1] + 1, prev[j] + 1,
                         prev[j - 1] + (0 if ai == bj else 1))
        prev = cur
    return prev[-1]


def common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def compare_pair(a: str | None, b: str | None,
                 enc) -> dict[str, Any]:
    ta = enc(a) if a is not None else []
    tb = enc(b) if b is not None else []
    cp = common_prefix(ta, tb)
    denom = max(len(ta), len(tb), 1)
    return {
        "char_len_a": len(a) if a else 0,
        "char_len_b": len(b) if b else 0,
        "tok_len_a": len(ta),
        "tok_len_b": len(tb),
        "tok_common_prefix": cp,
        "tok_first_diff": (cp if cp < min(len(ta), len(tb)) else (
            cp if len(ta) != len(tb) else None)),
        "tok_levenshtein": levenshtein(ta, tb),
        "exact_equal": a == b,
        "token_equal": ta == tb,
    }


def compute_envelope(cmp_ab: list[dict[str, Any]]) -> list[int]:
    """Per-prompt envelope max = max(k, tok_levenshtein) of the
    same-config repeat A vs B.
    """
    return [max(ENVELOPE_FLOOR_K, x["tok_levenshtein"]) for x in cmp_ab]


def cross_inside_envelope(cross: list[dict[str, Any]],
                          envelope_max: list[int]) -> list[dict[str, Any]]:
    out = []
    for i, x in enumerate(cross):
        env = envelope_max[i] if i < len(envelope_max) else ENVELOPE_FLOOR_K
        out.append({
            **x,
            "envelope_max": env,
            "inside_envelope": x["tok_levenshtein"] <= env,
        })
    return out


def summarise_leg_reqs(reqs: list[dict[str, Any]] | None) -> dict[str, Any]:
    if reqs is None:
        return {"missing": True}
    return {
        "missing": False,
        "n": len(reqs),
        "all_http_200": all(r.get("http_status") == 200 for r in reqs),
        "any_error": any(r.get("error") for r in reqs),
        "response_char_lens": [len(r.get("response_text") or "") for r in reqs],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True, type=Path)
    ap.add_argument("--out-md", required=True, type=Path)
    ap.add_argument("--out-json", required=True, type=Path)
    args = ap.parse_args()

    ind = args.in_dir
    cold = ind / "cold"
    neg = ind / "neg"

    tokenizer = AutoTokenizer.from_pretrained(SNAP_PATH)
    enc = lambda s: tokenizer.encode(s, add_special_tokens=False)

    # ------ Load raw ------
    matched = {
        "stock_default_image": (load_leg(cold / "stock_default_image_A.json"),
                                load_leg(cold / "stock_default_image_B.json")),
        "fork_default_image":  (load_leg(cold / "fork_default_image_A.json"),
                                load_leg(cold / "fork_default_image_B.json")),
        "stock_pcg_text":      (load_leg(cold / "stock_pcg_text_A.json"),
                                load_leg(cold / "stock_pcg_text_B.json")),
        "fork_pcg_text":       (load_leg(cold / "fork_pcg_text_A.json"),
                                load_leg(cold / "fork_pcg_text_B.json")),
        "fork_pcg_image":      (load_leg(cold / "fork_pcg_image_A.json"),
                                load_leg(cold / "fork_pcg_image_B.json")),
    }
    neg_reqs = load_leg(neg / "stock_pcg_image.json")
    neg_class = load_json(neg / "stock_pcg_image_classification.json")
    fpi_reqs = load_leg(ind / "fork_pcg_interleaved.json")
    fpi_safety = load_json(ind / "fork_pcg_interleaved_safety.json")
    launch_ctx = load_json(ind / "launch_context.json")

    # ------ Envelopes (same-config repeat A vs B) ------
    envelopes: dict[str, dict[str, Any]] = {}
    for key, (rA, rB) in matched.items():
        if rA is None or rB is None:
            envelopes[key] = {"missing": True}
            continue
        cmps = [compare_pair(a, b, enc)
                for a, b in zip(texts_of(rA), texts_of(rB))]
        env = compute_envelope(cmps)
        envelopes[key] = {
            "missing": False, "n_prompts": len(cmps),
            "per_prompt_repeat_lev": [c["tok_levenshtein"] for c in cmps],
            "per_prompt_envelope_max": env,
        }

    # ------ Cross-config comparisons ------
    def a_texts(key: str) -> list[str | None]:
        rA, _ = matched.get(key, (None, None))
        return texts_of(rA)

    def cross(name: str, key_a: str, key_b: str,
              envelope_key: str) -> dict[str, Any]:
        ta = a_texts(key_a)
        tb = a_texts(key_b)
        if not ta or not tb:
            return {"name": name, "missing": True,
                    "key_a": key_a, "key_b": key_b}
        cmps = [compare_pair(x, y, enc) for x, y in zip(ta, tb)]
        env = envelopes.get(envelope_key, {}).get(
            "per_prompt_envelope_max", [ENVELOPE_FLOOR_K] * len(cmps))
        cmps = cross_inside_envelope(cmps, env)
        return {"name": name, "missing": False, "key_a": key_a,
                "key_b": key_b, "envelope_key": envelope_key,
                "per_prompt": cmps,
                "all_inside_envelope": all(c["inside_envelope"] for c in cmps)}

    crosses = [
        cross("stock_default_vs_fork_default__image_cold",
              "stock_default_image", "fork_default_image",
              # envelope = max of the two same-config repeats' envelopes
              # (we approximate by using the LARGER of the two envelope_max
              # arrays per prompt; caller inspects both to decide)
              "stock_default_image"),
        cross("stock_pcg_vs_fork_pcg__text_cold",
              "stock_pcg_text", "fork_pcg_text",
              "stock_pcg_text"),
        cross("fork_default_vs_fork_pcg__image_cold",
              "fork_default_image", "fork_pcg_image",
              "fork_default_image"),
    ]
    # Enrich each cross with the union-envelope (max of both sides' envelopes)
    for c in crosses:
        if c["missing"]: continue
        e_a = envelopes.get(c["key_a"], {}).get("per_prompt_envelope_max", [])
        e_b = envelopes.get(c["key_b"], {}).get("per_prompt_envelope_max", [])
        union = [max(x, y) if x is not None and y is not None else max(x or ENVELOPE_FLOOR_K, y or ENVELOPE_FLOOR_K)
                 for x, y in zip(e_a, e_b)] if e_a and e_b else e_a or e_b
        # Recompute inside-envelope with the union envelope
        for i, pp in enumerate(c["per_prompt"]):
            env = union[i] if i < len(union) else ENVELOPE_FLOOR_K
            pp["union_envelope_max"] = env
            pp["inside_envelope"] = pp["tok_levenshtein"] <= env
        c["all_inside_envelope"] = all(
            pp["inside_envelope"] for pp in c["per_prompt"])

    # ------ Tier 1: SAFETY_SUPERIORITY ------
    neg_result = (neg_class or {}).get("result", "MISSING")
    fpi_safety_ok = fpi_safety is not None and all(
        (fpi_safety.get(k) or 0) == 0
        for k in ("assertions", "fallbacks", "request_failures"))
    fpi_per_leg = (fpi_safety or {}).get("per_leg_recompiles", {})
    fpi_no_inflight_recompile = (
        fpi_safety is not None and
        all(v == 0 for v in (fpi_per_leg.values() if isinstance(fpi_per_leg, dict) else []))
    )
    fpi_reqs_ok = fpi_reqs is not None and all(
        r.get("http_status") == 200 and r.get("error") is None for r in fpi_reqs)

    safety_reasons: list[str] = []
    if neg_result != "EXPECTED_STOCK_FAILURE":
        safety_reasons.append(f"negative_control.result={neg_result} (need EXPECTED_STOCK_FAILURE)")
    if not fpi_reqs_ok:
        safety_reasons.append("fork-PCG interleaved leg had non-200 or error")
    if not fpi_safety_ok:
        safety_reasons.append(f"fork-PCG interleaved safety metrics not all zero: {fpi_safety}")
    if not fpi_no_inflight_recompile:
        safety_reasons.append("fork-PCG interleaved had inflight-leg recompile events")
    safety_verdict = "PASS" if not safety_reasons else "FAIL"

    # ------ Tier 2: CORRECTNESS ------
    corr_reasons: list[str] = []
    for c in crosses:
        if c["missing"]:
            corr_reasons.append(f"{c['name']}: missing raw data")
            continue
        if not c["all_inside_envelope"]:
            offenders = [pp for pp in c["per_prompt"] if not pp["inside_envelope"]]
            corr_reasons.append(
                f"{c['name']}: {len(offenders)} prompt(s) outside envelope; "
                f"per-prompt(lev, env): "
                f"{[(pp['tok_levenshtein'], pp['union_envelope_max']) for pp in c['per_prompt']]}"
            )
    correctness_verdict = "PASS" if not corr_reasons else "FAIL"

    # ------ Overall ------
    if safety_verdict == "PASS" and correctness_verdict == "PASS":
        overall = "PASS"
        tier_claimed = "SAFETY_SUPERIORITY + CORRECTNESS"
    elif safety_verdict == "PASS":
        overall = "SAFETY_PASS_CORRECTNESS_AMBIGUOUS"
        tier_claimed = "SAFETY_SUPERIORITY"
    else:
        overall = "FAIL"
        tier_claimed = "NONE"

    # ------ Render markdown ------
    L = ["# R6.1 amended verdict — **{}**".format(overall), ""]
    L.append(f"> Evaluated under [`protocol_amendment_A_direct_fix_comparison.md`](../protocol_amendment_A_direct_fix_comparison.md). Rules were pre-declared before any leg ran.")
    L.append("")
    if launch_ctx:
        L.append("## Launch context")
        L.append("")
        for k in ("selected_gpu_id", "attempt_dir", "host_libcuda",
                  "ld_preload", "cuda_visible_devices", "nvidia_driver",
                  "sglang_stock_head", "sglang_fork_head"):
            if k in launch_ctx:
                L.append(f"- `{k}`: {launch_ctx[k]}")
        L.append("")

    L.append("## Tier 1 — SAFETY_SUPERIORITY: **{}**".format(safety_verdict))
    L.append("")
    L.append(f"- Negative control (stock-PCG image): **{neg_result}**")
    if neg_class and neg_class.get("reason"):
        L.append(f"  - reason: `{neg_class['reason']}`")
    L.append(f"- fork-PCG interleaved leg: all_http_200="
             f"{fpi_reqs_ok}, safety metrics zero={fpi_safety_ok}, "
             f"no inflight recompiles={fpi_no_inflight_recompile}")
    if fpi_safety:
        L.append(f"- fork-PCG safety: `{json.dumps(fpi_safety, sort_keys=True)}`")
    if safety_reasons:
        L.append("")
        L.append("**Failure reasons:**")
        for r in safety_reasons:
            L.append(f"- {r}")
    L.append("")

    L.append("## Tier 2 — CORRECTNESS: **{}**".format(correctness_verdict))
    L.append("")
    L.append("### Matched cold-cache repeat envelopes (per-prompt tok Levenshtein)")
    L.append("")
    for key, e in envelopes.items():
        if e.get("missing"):
            L.append(f"- `{key}`: **missing**")
        else:
            L.append(f"- `{key}`: repeat_lev={e['per_prompt_repeat_lev']}, "
                     f"envelope_max={e['per_prompt_envelope_max']}")
    L.append("")
    L.append("### Cross-config comparisons")
    L.append("")
    for c in crosses:
        L.append(f"#### `{c['name']}`")
        if c["missing"]:
            L.append("- **missing raw data**")
            continue
        L.append(f"- All inside envelope: **{c['all_inside_envelope']}**")
        L.append("")
        L.append("| Prompt | tok_lev | union_env | inside? | char==/tok== |")
        L.append("|---|---|---|---|---|")
        for i, pp in enumerate(c["per_prompt"]):
            L.append(f"| {i} | {pp['tok_levenshtein']} | "
                     f"{pp['union_envelope_max']} | "
                     f"{pp['inside_envelope']} | "
                     f"{pp['exact_equal']} / {pp['token_equal']} |")
        L.append("")
    if corr_reasons:
        L.append("**Failure reasons:**")
        for r in corr_reasons:
            L.append(f"- {r}")
        L.append("")

    L.append("## Overall verdict: **{}**".format(overall))
    L.append("")
    L.append(f"- Evidence tier claimed: **{tier_claimed}**")
    L.append("")

    args.out_md.write_text("\n".join(L) + "\n")
    args.out_json.write_text(json.dumps({
        "overall_verdict": overall,
        "evidence_tier_claimed": tier_claimed,
        "safety_superiority": {"verdict": safety_verdict,
                               "reasons": safety_reasons,
                               "negative_control_result": neg_result,
                               "fpi_reqs_ok": fpi_reqs_ok,
                               "fpi_safety_ok": fpi_safety_ok,
                               "fpi_no_inflight_recompile": fpi_no_inflight_recompile,
                               "fpi_safety": fpi_safety},
        "correctness": {"verdict": correctness_verdict,
                        "reasons": corr_reasons,
                        "envelopes": envelopes,
                        "cross_comparisons": crosses},
        "launch_context": launch_ctx,
    }, indent=2, sort_keys=True))

    print(f"OVERALL_VERDICT={overall}")
    print(f"TIER_CLAIMED={tier_claimed}")
    print(f"SAFETY={safety_verdict}")
    print(f"CORRECTNESS={correctness_verdict}")
    return 0 if overall == "PASS" else (2 if overall.startswith("SAFETY_PASS") else 1)


if __name__ == "__main__":
    sys.exit(main())
