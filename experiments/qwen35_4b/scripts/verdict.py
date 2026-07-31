#!/usr/bin/env python3
"""Verdict computation for the Qwen3.5-4B BCG DeepStack validation.

Reads an attempt directory laid out by ``runner.sh``:

    attempt_dir/
        metadata.json
        raw/
            instrumentation_<config>.jsonl
            client_records_<config>.json
            server_stderr_<config>.log
            ...

and emits ``verdict.json`` + ``verdict.md`` per
``validation_plan.md`` §1.

Verdict labels (source of truth):

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


def collect_config_records(
    attempt_dir: Path, launch_id: str
) -> dict[str, dict[str, Any]]:
    """Return {config_label: {"client": <records>, "instrumentation": [events]}}."""
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
            e
            for e in events
            if e.get("launch_id") in (None, "", launch_id)
        ]
        out.setdefault(label, {})["instrumentation"] = events

    return out


def classify_execution_path(events: list[dict], client_rid: str) -> dict:
    """Given instrumentation events for one client request id, return whether
    it hit BCG replay, eager, or was unclassifiable.

    The client's `rid` becomes the server's `forward_batch.rid`, but our
    instrumentation does not carry client_request_id directly. We
    correlate by simple temporal window: events between two client-send
    boundaries are attributed to that request. Since the client is
    strictly serial (concurrency 1), this ordering is faithful.

    A simpler and more robust classification: count TOTAL bcg_execute_body
    entries vs total model_runner forwards; if both are > 0 in the
    per-config window, we know BCG fired for at least one request.

    This function returns whether the SINGLE request likely hit BCG,
    based on the events already sliced for it.
    """
    saw_bcg_execute = any(
        e.get("event") == "bcg_execute_body_enter" for e in events
    )
    saw_replay_layer = any(
        e.get("event") == "bcg_replay_layer_forward_enter" for e in events
    )
    saw_mr_forward = any(
        e.get("event") == "model_runner_forward_enter" for e in events
    )
    ds_summaries = [
        e for e in events
        if e.get("event") == "lm_forward_input_deepstack"
    ]

    return {
        "bcg_execute": saw_bcg_execute,
        "replay_layer": saw_replay_layer,
        "model_runner_forward": saw_mr_forward,
        "deepstack_events": len(ds_summaries),
        "deepstack_nonzero_max": max(
            [
                (e.get("input_deepstack_embeds") or {}).get("nonzero_frac", 0.0)
                for e in ds_summaries
            ],
            default=0.0,
        ),
        "deepstack_checksum_first": (
            (ds_summaries[0].get("input_deepstack_embeds") or {}).get("sha256_16")
            if ds_summaries else None
        ),
    }


def compare_token_sequences(a: dict | None, b: dict | None) -> dict:
    """Compare two http_response payloads' generated text / token ids.

    SGLang's /generate returns a list of dicts each with ``text`` and
    (when return_logprob) ``meta_info.output_token_logprobs``. For
    greedy sampling the ``text`` alone is enough to detect divergence.
    """
    def _extract_text(resp: dict | None) -> str | None:
        if not resp:
            return None
        if isinstance(resp, list) and resp:
            resp = resp[0]
        if not isinstance(resp, dict):
            return None
        return resp.get("text")

    ta = _extract_text(a)
    tb = _extract_text(b)
    return {
        "a_text": ta,
        "b_text": tb,
        "equal": ta == tb,
        "a_len": len(ta) if ta else 0,
        "b_len": len(tb) if tb else 0,
    }


def compute_verdict(
    metadata: dict, per_config: dict[str, dict[str, Any]]
) -> tuple[str, dict]:
    """Score the verdict from the collected evidence.

    See validation_plan.md §6.
    """
    evidence: dict[str, Any] = {
        "per_config_summary": {},
        "notes": [],
    }

    # Preflight failure?
    if metadata.get("preflight_status") == "FAIL":
        evidence["preflight_status"] = "FAIL"
        return "INFRA_FAILURE", evidence

    # Assemble per-config summaries.
    for label, bundle in per_config.items():
        client = bundle.get("client") or {}
        inst = bundle.get("instrumentation") or []
        records = (client.get("records") or [])
        img_scored = [r for r in records if r.get("kind") == "P_IMG" and r.get("role") == "scored"]
        txt_scored = [r for r in records if r.get("kind") == "P_TXT" and r.get("role") == "scored"]

        # Path counts across the entire config window.
        total_bcg_execute = sum(1 for e in inst if e.get("event") == "bcg_execute_body_enter")
        total_replay_layer = sum(1 for e in inst if e.get("event") == "bcg_replay_layer_forward_enter")
        total_mr_forward = sum(1 for e in inst if e.get("event") == "model_runner_forward_enter")
        total_ds_events = sum(1 for e in inst if e.get("event") == "lm_forward_input_deepstack")
        total_ds_zeroed = sum(1 for e in inst if e.get("event") == "lm_forward_input_deepstack_zeroed")
        errors = [e for e in inst if e.get("event", "").endswith("_error")]

        ds_events = [e for e in inst if e.get("event") == "lm_forward_input_deepstack"]
        first_ds_summary = (
            ds_events[0].get("input_deepstack_embeds") if ds_events else None
        )

        evidence["per_config_summary"][label] = {
            "records": len(records),
            "img_scored": len(img_scored),
            "txt_scored": len(txt_scored),
            "path_counts": {
                "bcg_execute_body_enter": total_bcg_execute,
                "bcg_replay_layer_forward_enter": total_replay_layer,
                "model_runner_forward_enter": total_mr_forward,
                "lm_forward_input_deepstack": total_ds_events,
                "lm_forward_input_deepstack_zeroed": total_ds_zeroed,
            },
            "first_deepstack_summary": first_ds_summary,
            "errors_seen": len(errors),
            "first_error": errors[0] if errors else None,
        }

    # Verdict rules.
    def _has_bcg_normal_image(label: str = "bcg_normal") -> bool:
        s = evidence["per_config_summary"].get(label) or {}
        return s.get("path_counts", {}).get("bcg_execute_body_enter", 0) > 0

    def _client_records(label: str) -> list[dict]:
        client = (per_config.get(label) or {}).get("client") or {}
        return client.get("records") or []

    def _first_img(records: list[dict]) -> dict | None:
        for r in records:
            if r.get("kind") == "P_IMG" and r.get("role") == "scored":
                return r
        return None

    def _img_text(label: str) -> str | None:
        r = _first_img(_client_records(label))
        if r is None:
            return None
        return compare_token_sequences(r.get("http_response"), None)["a_text"]

    eager_normal_txt = _img_text("eager_normal")
    eager_zero_txt = _img_text("eager_zero_deepstack")
    bcg_normal_txt = _img_text("bcg_normal")

    evidence["greedy_texts"] = {
        "eager_normal_image": eager_normal_txt,
        "eager_zero_deepstack_image": eager_zero_txt,
        "bcg_normal_image": bcg_normal_txt,
    }

    # Was BCG actually exercised for image requests in bcg_normal?
    bcg_had_image_bcg_execute = _has_bcg_normal_image("bcg_normal")

    # Errors on BCG image request?
    bcg_errors = (evidence["per_config_summary"].get("bcg_normal") or {}).get("errors_seen", 0)

    # Report deepstack nonzero on eager_normal - if fixture failed to
    # produce nonzero DeepStack, we can't attribute anything.
    eager_normal_summary = evidence["per_config_summary"].get("eager_normal", {})
    ds_first = eager_normal_summary.get("first_deepstack_summary") or {}
    ds_nonzero_frac = ds_first.get("nonzero_frac", 0.0)

    if ds_nonzero_frac == 0.0 and eager_normal_summary.get("img_scored", 0) > 0:
        evidence["notes"].append(
            "eager_normal image request produced input_deepstack_embeds "
            "with nonzero_frac == 0; the fixture is not exercising DeepStack. "
            "Verdict AMBIGUOUS."
        )
        return "AMBIGUOUS", evidence

    # Scenario 1: BCG replay never fired for image → feature gap.
    if not bcg_had_image_bcg_execute and (
        evidence["per_config_summary"].get("bcg_normal", {}).get("img_scored", 0) > 0
    ):
        evidence["notes"].append(
            "bcg_normal config was configured but BCG replay was never "
            "invoked for the scored image request. Image request went "
            "through eager path."
        )
        return "FEATURE_GAP_EAGER_FALLBACK", evidence

    # Scenario 2: BCG image request produced a matched greedy divergence.
    if bcg_errors > 0:
        evidence["notes"].append(
            "bcg_normal image request produced an error / illegal memory event."
        )
        return "FAIL_BCG_DEEPSTACK", evidence

    if bcg_normal_txt and eager_normal_txt:
        if bcg_normal_txt == eager_normal_txt:
            evidence["notes"].append(
                "bcg_normal image greedy text equals eager_normal — "
                "and BCG replay was exercised. Correct."
            )
            return "PASS_BCG_CORRECT", evidence
        # bcg_normal diverges. Check zero-deepstack signature.
        if eager_zero_txt and bcg_normal_txt == eager_zero_txt:
            evidence["notes"].append(
                "bcg_normal image greedy text differs from eager_normal "
                "AND matches eager_zero_deepstack — strong attribution "
                "for a missing DeepStack contribution."
            )
            return "FAIL_BCG_DEEPSTACK", evidence
        evidence["notes"].append(
            "bcg_normal image text diverges from eager_normal but does "
            "not match eager_zero_deepstack. Attribution unclear."
        )
        return "AMBIGUOUS", evidence

    # Missing scored data.
    evidence["notes"].append(
        "insufficient scored image requests to score verdict."
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
        "## Greedy texts",
        "",
        "```json",
        json.dumps(evidence.get("greedy_texts", {}), indent=2, sort_keys=True, default=str),
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
        # Dry-run with an empty attempt-dir: produce INFRA_FAILURE and
        # a small self-check verdict payload.
        verdict = "INFRA_FAILURE"
        evidence = {
            "launch_id": launch_id,
            "notes": [
                "skeleton dry-run: no evidence files present in attempt-dir. "
                "Emitting INFRA_FAILURE as the safest placeholder."
            ],
            "per_config_summary": {},
            "greedy_texts": {},
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
