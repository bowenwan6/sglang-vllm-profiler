#!/usr/bin/env python3
"""Verdict computation for the Qwen3.5-4B BCG DeepStack validation
(predeclared 2×2 design).

Reads an attempt directory laid out by ``runner.sh``:

    attempt_dir/
        metadata.json
        raw/
            instrumentation_<config>.jsonl
            client_records_<config>.json
            server_stderr_<config>.log
            ...

and emits ``verdict.json`` + ``verdict.md`` + ``summary.md``.

Predeclared 2×2 configurations (validation_plan.md Amendment 2,
2026-08-01):

    - eager_normal          (BCG off, DeepStack computed normally)
    - eager_zero_deepstack  (BCG off, DeepStack zeroed by hook)
    - bcg_normal            (BCG on,  DeepStack computed normally)
    - bcg_zero_deepstack    (BCG on,  DeepStack zeroed by hook)

Verdict labels (source of truth — matches hypothesis.md §5):

    PASS_BCG_CORRECT
    FEATURE_GAP_EAGER_FALLBACK
    FAIL_BCG_DEEPSTACK
    AMBIGUOUS
    INFRA_FAILURE

Historical guard: refuses to score across mismatched launches. Every
raw file must carry a ``launch_id`` (or the launch_id is inferred from
the client record); a file whose launch_id disagrees with the
attempt's is rejected and NOT averaged in.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

VALID_VERDICTS = {
    "PASS_BCG_CORRECT",
    "FEATURE_GAP_EAGER_FALLBACK",
    "FAIL_BCG_DEEPSTACK",
    "AMBIGUOUS",
    "INFRA_FAILURE",
}

REQUIRED_ARMS = (
    "eager_normal",
    "eager_zero_deepstack",
    "bcg_normal",
    "bcg_zero_deepstack",
)

# Server-side warning that signals a placeholder-vs-image count mismatch.
IMAGE_PLACEHOLDER_WARNING_RE = re.compile(
    r"More image data items provided than corresponding tokens found in the prompt"
)


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    return json.loads(path.read_text())


def load_metadata(attempt_dir: Path) -> dict:
    return load_json(attempt_dir / "metadata.json")


def load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    if not path.is_file():
        return out
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:  # noqa: BLE001
                continue
    return out


def _read_text(path: Path) -> str:
    try:
        return path.read_text(errors="replace")
    except Exception:
        return ""


def collect_config_records(
    attempt_dir: Path, launch_id: str
) -> dict[str, dict[str, Any]]:
    """Return {config_label: {"client": ..., "instrumentation": ...,
    "server_stderr": ..., "placeholder_warning": bool}}."""
    raw = attempt_dir / "raw"
    out: dict[str, dict[str, Any]] = {}
    if not raw.is_dir():
        return out

    for cf in sorted(raw.glob("client_records_*.json")):
        label = cf.stem[len("client_records_") :]
        try:
            data = json.loads(cf.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"REJECT: {cf.name} unreadable: {exc}", file=sys.stderr)
            continue
        recorded = (data.get("summary") or {}).get("launch_id")
        if recorded != launch_id:
            print(
                f"REJECT: {cf.name} launch_id={recorded!r} != {launch_id!r}",
                file=sys.stderr,
            )
            continue
        out.setdefault(label, {})["client"] = data

    for inst in sorted(raw.glob("instrumentation_*.jsonl")):
        label = inst.stem[len("instrumentation_") :]
        events = load_jsonl(inst)
        # Filter by launch_id when present.
        events = [
            e for e in events
            if e.get("launch_id") in (None, "", launch_id)
        ]
        out.setdefault(label, {})["instrumentation"] = events

    for se in sorted(raw.glob("server_stderr_*.log")):
        label = se.stem[len("server_stderr_") :]
        text = _read_text(se)
        out.setdefault(label, {})["server_stderr_prefix"] = text[:2000]
        out.setdefault(label, {})["placeholder_warning"] = bool(
            IMAGE_PLACEHOLDER_WARNING_RE.search(text)
        )

    return out


def _client_records(bundle: dict) -> list[dict]:
    return ((bundle.get("client") or {}).get("records") or [])


def _first_img_record(records: list[dict]) -> dict | None:
    for r in records:
        if r.get("kind") == "P_IMG" and r.get("role") == "scored":
            return r
    return None


def _second_img_record(records: list[dict]) -> dict | None:
    hits = [r for r in records if r.get("kind") == "P_IMG" and r.get("role") == "scored"]
    return hits[1] if len(hits) >= 2 else None


def _extract_output_ids(rec: dict | None) -> list[int] | None:
    if rec is None:
        return None
    ids = rec.get("output_token_ids")
    if isinstance(ids, list) and ids:
        return list(ids)
    # Fallback: parse http_response.
    resp = rec.get("http_response")
    if resp is None:
        return None
    if isinstance(resp, list) and resp:
        resp = resp[0]
    if not isinstance(resp, dict):
        return None
    ids = resp.get("output_ids")
    if isinstance(ids, list) and ids:
        return list(ids)
    meta = resp.get("meta_info") or {}
    otl = meta.get("output_token_logprobs")
    if isinstance(otl, list) and otl:
        out = []
        for e in otl:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                out.append(int(e[1]))
            elif isinstance(e, dict) and "token_id" in e:
                out.append(int(e["token_id"]))
        if out:
            return out
    return None


def _extract_output_text(rec: dict | None) -> str | None:
    if rec is None:
        return None
    resp = rec.get("http_response")
    if resp is None:
        return None
    if isinstance(resp, list) and resp:
        resp = resp[0]
    if isinstance(resp, dict):
        return resp.get("text")
    return None


def _extract_output_logprobs(rec: dict | None) -> list[float] | None:
    """Per-generated-token greedy logprob (element 0 of the tuple)."""
    if rec is None:
        return None
    resp = rec.get("http_response")
    if resp is None:
        return None
    if isinstance(resp, list) and resp:
        resp = resp[0]
    if not isinstance(resp, dict):
        return None
    meta = resp.get("meta_info") or {}
    otl = meta.get("output_token_logprobs")
    if not isinstance(otl, list):
        return None
    out = []
    for e in otl:
        if isinstance(e, (list, tuple)) and e:
            try:
                out.append(float(e[0]))
            except Exception:
                out.append(math.nan)
        elif isinstance(e, dict) and "logprob" in e:
            try:
                out.append(float(e["logprob"]))
            except Exception:
                out.append(math.nan)
    return out or None


def _compare_id_sequences(a: list[int] | None, b: list[int] | None) -> dict:
    if not a or not b:
        return {
            "a_len": len(a) if a else 0,
            "b_len": len(b) if b else 0,
            "equal": False,
            "common_prefix_len": 0,
            "first_diff_index": None,
        }
    n = min(len(a), len(b))
    prefix = 0
    for i in range(n):
        if a[i] == b[i]:
            prefix += 1
        else:
            break
    equal = a == b
    first_diff = None if equal else prefix
    return {
        "a_len": len(a),
        "b_len": len(b),
        "equal": equal,
        "common_prefix_len": prefix,
        "first_diff_index": first_diff,
    }


def _compare_logprob_sequences(a: list[float] | None, b: list[float] | None) -> dict:
    if not a or not b:
        return {"available": False}
    n = min(len(a), len(b))
    diffs = [abs(a[i] - b[i]) for i in range(n) if not (math.isnan(a[i]) or math.isnan(b[i]))]
    if not diffs:
        return {"available": False}
    return {
        "available": True,
        "compared_tokens": len(diffs),
        "l1_mean_abs_diff": sum(diffs) / len(diffs),
        "l1_max_abs_diff": max(diffs),
    }


def _per_config_evidence(label: str, bundle: dict) -> dict:
    client = bundle.get("client") or {}
    records = client.get("records") or []
    inst = bundle.get("instrumentation") or []

    img_scored = [r for r in records if r.get("kind") == "P_IMG" and r.get("role") == "scored"]
    txt_scored = [r for r in records if r.get("kind") == "P_TXT" and r.get("role") == "scored"]

    total_bcg_execute = sum(1 for e in inst if e.get("event") == "bcg_execute_body_enter")
    total_bcg_error = sum(1 for e in inst if e.get("event") == "bcg_execute_body_error")
    total_mr_forward = sum(1 for e in inst if e.get("event") == "model_runner_forward_enter")
    total_ds_events = sum(1 for e in inst if e.get("event") == "lm_forward_input_deepstack")
    total_ds_zeroed = sum(1 for e in inst if e.get("event") == "lm_forward_input_deepstack_zeroed")
    hook_install_errors = sum(
        1 for e in inst if e.get("event") == "lm_forward_pre_hook_install_error"
    )
    errors_seen = sum(1 for e in inst if e.get("event", "").endswith("_error"))

    ds_events = [e for e in inst if e.get("event") == "lm_forward_input_deepstack"]
    first_ds_summary = (
        ds_events[0].get("input_deepstack_embeds") if ds_events else None
    )
    # Max non-zero fraction seen on ANY image request.
    max_nonzero = 0.0
    for e in ds_events:
        s = e.get("input_deepstack_embeds") or {}
        if isinstance(s, dict) and s.get("nonzero_frac") is not None:
            try:
                max_nonzero = max(max_nonzero, float(s["nonzero_frac"]))
            except Exception:
                pass

    zeroed_events = [e for e in inst if e.get("event") == "lm_forward_input_deepstack_zeroed"]
    zero_verified = False
    if zeroed_events:
        after = zeroed_events[-1].get("after") or {}
        zero_verified = (
            after.get("nonzero_frac", None) == 0.0
            and after.get("abs_sum", None) == 0.0
        )

    return {
        "records": len(records),
        "img_scored": len(img_scored),
        "txt_scored": len(txt_scored),
        "path_counts": {
            "bcg_execute_body_enter": total_bcg_execute,
            "bcg_execute_body_error": total_bcg_error,
            "model_runner_forward_enter": total_mr_forward,
            "lm_forward_input_deepstack": total_ds_events,
            "lm_forward_input_deepstack_zeroed": total_ds_zeroed,
            "lm_forward_pre_hook_install_error": hook_install_errors,
        },
        "first_deepstack_summary": first_ds_summary,
        "max_deepstack_nonzero_frac": max_nonzero,
        "zero_replacement_verified": zero_verified,
        "errors_seen": errors_seen,
        "placeholder_warning": bool(bundle.get("placeholder_warning")),
        "first_scored_image_output_ids": _extract_output_ids(_first_img_record(records)),
        "first_scored_image_text": _extract_output_text(_first_img_record(records)),
        "first_scored_image_placeholder_count": (
            (_first_img_record(records) or {}).get("placeholder_provided_count")
        ),
        "first_scored_image_image_count": (
            (_first_img_record(records) or {}).get("image_provided_count")
        ),
        "second_scored_image_output_ids": _extract_output_ids(_second_img_record(records)),
    }


def compute_verdict(
    metadata: dict, per_config: dict[str, dict[str, Any]]
) -> tuple[str, dict]:
    evidence: dict[str, Any] = {
        "per_config_summary": {},
        "notes": [],
        "comparisons": {},
        "arms_present": sorted(per_config.keys()),
        "arms_required": list(REQUIRED_ARMS),
    }

    if metadata.get("preflight_status") == "FAIL":
        evidence["preflight_status"] = "FAIL"
        evidence["notes"].append("preflight_status=FAIL in metadata")
        return "INFRA_FAILURE", evidence

    # Per-arm summaries.
    for label, bundle in per_config.items():
        evidence["per_config_summary"][label] = _per_config_evidence(label, bundle)

    missing = [a for a in REQUIRED_ARMS if a not in per_config]
    if missing:
        evidence["notes"].append(
            f"missing predeclared arms: {missing}. Cannot score full 2x2."
        )
        return "AMBIGUOUS", evidence

    # Extract records per arm.
    def _first_img(label: str) -> dict | None:
        return _first_img_record(_client_records(per_config[label]))

    def _second_img(label: str) -> dict | None:
        return _second_img_record(_client_records(per_config[label]))

    en = _first_img("eager_normal")
    ez = _first_img("eager_zero_deepstack")
    bn = _first_img("bcg_normal")
    bz = _first_img("bcg_zero_deepstack")

    if any(x is None for x in (en, ez, bn, bz)):
        evidence["notes"].append(
            "at least one arm has no scored image record; ambiguous."
        )
        return "AMBIGUOUS", evidence

    # --- Layer 1: image/placeholder alignment ---------------------------
    for label in REQUIRED_ARMS:
        s = evidence["per_config_summary"][label]
        if s.get("placeholder_warning"):
            evidence["notes"].append(
                f"arm {label} emitted the SGLang "
                "'More image data items provided than corresponding tokens' "
                "warning. Image placeholder alignment failed. AMBIGUOUS."
            )
            return "AMBIGUOUS", evidence
        if (s.get("first_scored_image_placeholder_count") or 0) != (
            s.get("first_scored_image_image_count") or 0
        ):
            evidence["notes"].append(
                f"arm {label} placeholder count "
                f"{s.get('first_scored_image_placeholder_count')} != image count "
                f"{s.get('first_scored_image_image_count')}. AMBIGUOUS."
            )
            return "AMBIGUOUS", evidence

    # --- Layer 2: DeepStack telemetry ------------------------------------
    eager_normal_summary = evidence["per_config_summary"]["eager_normal"]
    eager_zero_summary = evidence["per_config_summary"]["eager_zero_deepstack"]
    bcg_normal_summary = evidence["per_config_summary"]["bcg_normal"]
    bcg_zero_summary = evidence["per_config_summary"]["bcg_zero_deepstack"]

    for label, s in (
        ("eager_normal", eager_normal_summary),
        ("bcg_normal", bcg_normal_summary),
    ):
        if s["path_counts"]["lm_forward_input_deepstack"] == 0:
            evidence["notes"].append(
                f"arm {label} recorded ZERO lm_forward_input_deepstack "
                "events on image requests. DeepStack presence unverified."
            )
            return "AMBIGUOUS", evidence
        if s["max_deepstack_nonzero_frac"] == 0.0:
            evidence["notes"].append(
                f"arm {label} observed a DeepStack tensor whose "
                "nonzero_frac == 0. Fixture failed to exercise DeepStack."
            )
            return "AMBIGUOUS", evidence

    for label, s in (
        ("eager_zero_deepstack", eager_zero_summary),
        ("bcg_zero_deepstack", bcg_zero_summary),
    ):
        if not s["zero_replacement_verified"]:
            evidence["notes"].append(
                f"arm {label} did not verify a zero DeepStack replacement "
                "(no lm_forward_input_deepstack_zeroed with nonzero_frac == 0)."
            )
            return "AMBIGUOUS", evidence

    # --- Layer 3: BCG execution path -------------------------------------
    # Every scored image request in bcg_normal + bcg_zero_deepstack must
    # have entered BCG replay.
    for label, s in (("bcg_normal", bcg_normal_summary), ("bcg_zero_deepstack", bcg_zero_summary)):
        # There is no perfect per-request correlation. We check that the
        # arm has at least one bcg_execute_body_enter with contains_mm=True.
        inst = per_config[label].get("instrumentation") or []
        mm_bcg = [
            e for e in inst
            if e.get("event") == "bcg_execute_body_enter"
            and (e.get("forward_batch") or {}).get("contains_mm_inputs")
        ]
        if not mm_bcg:
            evidence["notes"].append(
                f"arm {label} shows zero bcg_execute_body_enter with "
                "contains_mm_inputs=true. BCG did not serve any image "
                "prefill. FEATURE_GAP_EAGER_FALLBACK."
            )
            return "FEATURE_GAP_EAGER_FALLBACK", evidence

    # --- Layer 4: BCG errors ---------------------------------------------
    for label, s in (("bcg_normal", bcg_normal_summary), ("bcg_zero_deepstack", bcg_zero_summary)):
        if s["path_counts"]["bcg_execute_body_error"] > 0:
            evidence["notes"].append(
                f"arm {label} recorded bcg_execute_body_error events; "
                "BCG replay raised. FAIL_BCG_DEEPSTACK."
            )
            return "FAIL_BCG_DEEPSTACK", evidence

    # --- Layer 5: token / logprob comparisons ---------------------------
    en_ids = _extract_output_ids(en)
    en2_ids = _extract_output_ids(_second_img("eager_normal"))
    ez_ids = _extract_output_ids(ez)
    bn_ids = _extract_output_ids(bn)
    bz_ids = _extract_output_ids(bz)

    en_lp = _extract_output_logprobs(en)
    en2_lp = _extract_output_logprobs(_second_img("eager_normal"))
    ez_lp = _extract_output_logprobs(ez)
    bn_lp = _extract_output_logprobs(bn)
    bz_lp = _extract_output_logprobs(bz)

    cmps: dict[str, Any] = {
        "eager_repeat_noise": {
            "ids": _compare_id_sequences(en_ids, en2_ids),
            "logprobs": _compare_logprob_sequences(en_lp, en2_lp),
        },
        "eager_zero_vs_eager_normal": {
            "ids": _compare_id_sequences(en_ids, ez_ids),
            "logprobs": _compare_logprob_sequences(en_lp, ez_lp),
        },
        "bcg_normal_vs_eager_normal": {
            "ids": _compare_id_sequences(en_ids, bn_ids),
            "logprobs": _compare_logprob_sequences(en_lp, bn_lp),
        },
        "bcg_zero_vs_eager_zero": {
            "ids": _compare_id_sequences(ez_ids, bz_ids),
            "logprobs": _compare_logprob_sequences(ez_lp, bz_lp),
        },
        "bcg_normal_vs_bcg_zero": {
            "ids": _compare_id_sequences(bn_ids, bz_ids),
            "logprobs": _compare_logprob_sequences(bn_lp, bz_lp),
        },
        "eager_normal_vs_eager_zero_texts": {
            "eager_normal_text": _extract_output_text(en),
            "eager_zero_text": _extract_output_text(ez),
            "bcg_normal_text": _extract_output_text(bn),
            "bcg_zero_text": _extract_output_text(bz),
        },
    }
    evidence["comparisons"] = cmps

    # Ablation sensitivity: eager_zero must diverge from eager_normal
    # beyond the eager-repeat noise envelope.
    ablation_diverges = (
        cmps["eager_zero_vs_eager_normal"]["ids"]["equal"] is False
        or (
            cmps["eager_zero_vs_eager_normal"]["logprobs"].get("available")
            and cmps["eager_zero_vs_eager_normal"]["logprobs"]["l1_max_abs_diff"] >
            max(
                cmps["eager_repeat_noise"]["logprobs"].get("l1_max_abs_diff", 0.0),
                1e-3,
            )
        )
    )
    if not ablation_diverges:
        evidence["notes"].append(
            "eager_zero_deepstack output does not measurably differ from "
            "eager_normal (tokens equal AND logprob distance within eager "
            "repeat noise). Fixture / prompt does not diagnostically "
            "exercise DeepStack under this configuration. AMBIGUOUS."
        )
        return "AMBIGUOUS", evidence

    # BCG replay comparisons.
    bn_matches_en = cmps["bcg_normal_vs_eager_normal"]["ids"]["equal"]
    bn_matches_bz = cmps["bcg_normal_vs_bcg_zero"]["ids"]["equal"]
    bcg_zero_matches_eager_zero = cmps["bcg_zero_vs_eager_zero"]["ids"]["equal"]

    if bn_matches_en and not bn_matches_bz:
        evidence["notes"].append(
            "bcg_normal matches eager_normal AND diverges from bcg_zero "
            "(so DeepStack was retained under BCG). PASS_BCG_CORRECT."
        )
        return "PASS_BCG_CORRECT", evidence

    if (not bn_matches_en) and bn_matches_bz and bcg_zero_matches_eager_zero:
        evidence["notes"].append(
            "bcg_normal diverges from eager_normal, matches bcg_zero, and "
            "bcg_zero matches eager_zero. Strong zero-DeepStack signature "
            "under BCG. FAIL_BCG_DEEPSTACK."
        )
        return "FAIL_BCG_DEEPSTACK", evidence

    if (not bn_matches_en) and bn_matches_bz:
        evidence["notes"].append(
            "bcg_normal diverges from eager_normal and matches bcg_zero, "
            "but bcg_zero does NOT match eager_zero — attribution partial. "
            "FAIL_BCG_DEEPSTACK is likely but not fully proven; recording "
            "AMBIGUOUS to preserve rigour."
        )
        return "AMBIGUOUS", evidence

    if bn_matches_en and bn_matches_bz:
        evidence["notes"].append(
            "bcg_normal matches both eager_normal AND bcg_zero. If the "
            "ablation is diagnostic, this three-way match implies the "
            "fixture does not perturb the greedy top-1 sequence enough to "
            "distinguish DeepStack presence via tokens alone. AMBIGUOUS."
        )
        return "AMBIGUOUS", evidence

    evidence["notes"].append(
        "bcg_normal diverges from eager_normal in a pattern that does not "
        "match the zero-DeepStack signature (bcg_normal != bcg_zero, "
        "bcg_normal != eager_normal). Attribution unclear. AMBIGUOUS."
    )
    return "AMBIGUOUS", evidence


def write_verdict(attempt_dir: Path, verdict: str, evidence: dict) -> None:
    if verdict not in VALID_VERDICTS:
        raise SystemExit(f"invalid verdict: {verdict}")
    payload = {"verdict": verdict, "evidence": evidence}
    (attempt_dir / "verdict.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    md_lines = [
        f"# Verdict — {verdict}",
        "",
        "Machine verdict produced by `scripts/verdict.py`.",
        "See `validation_plan.md` §1 for the verdict shape and §6 for the rules.",
        "",
        "## Notes",
        "",
    ]
    for note in evidence.get("notes", []):
        md_lines.append(f"- {note}")
    md_lines.extend([
        "",
        "## Per-config execution-path counts",
        "",
        "```json",
        json.dumps(evidence.get("per_config_summary", {}), indent=2, sort_keys=True, default=str),
        "```",
        "",
        "## Cross-arm comparisons",
        "",
        "```json",
        json.dumps(evidence.get("comparisons", {}), indent=2, sort_keys=True, default=str),
        "```",
        "",
    ])
    (attempt_dir / "verdict.md").write_text("\n".join(md_lines))


def write_summary(attempt_dir: Path, verdict: str, evidence: dict) -> None:
    summary = f"Qwen3.5-4B BCG DeepStack attempt {attempt_dir.name}: verdict {verdict}. "
    notes = evidence.get("notes") or []
    if notes:
        summary += " ".join(notes)
    (attempt_dir / "summary.md").write_text(summary.rstrip() + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    attempt_dir = args.attempt_dir
    if not attempt_dir.is_dir():
        print(f"FATAL: attempt-dir does not exist: {attempt_dir}", file=sys.stderr)
        return 1

    if args.dry_run and not (attempt_dir / "metadata.json").is_file():
        metadata = {
            "launch_context": {"prelaunch_utc": "1970-01-01T00-00-00Z"},
            "dry_run": True,
        }
    else:
        metadata = load_metadata(attempt_dir)

    launch_ctx = metadata.get("launch_context") or {}
    launch_id = (
        launch_ctx.get("prelaunch_utc")
        or metadata.get("prelaunch_utc")
    )
    if not launch_id:
        print("FATAL: metadata.json missing launch_context.prelaunch_utc", file=sys.stderr)
        return 1

    per_config = collect_config_records(attempt_dir, launch_id)

    if args.dry_run and not per_config:
        verdict = "INFRA_FAILURE"
        evidence = {
            "launch_id": launch_id,
            "notes": [
                "skeleton dry-run: no evidence files present in attempt-dir. "
                "Emitting INFRA_FAILURE as the safest placeholder."
            ],
            "per_config_summary": {},
            "comparisons": {},
        }
        write_verdict(attempt_dir, verdict, evidence)
        write_summary(attempt_dir, verdict, evidence)
        print(f"dry-run: wrote {attempt_dir/'verdict.json'} (verdict={verdict})")
        return 0

    verdict, evidence = compute_verdict(metadata, per_config)
    evidence["launch_id"] = launch_id
    write_verdict(attempt_dir, verdict, evidence)
    write_summary(attempt_dir, verdict, evidence)
    print(f"wrote {attempt_dir/'verdict.json'} (verdict={verdict})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
