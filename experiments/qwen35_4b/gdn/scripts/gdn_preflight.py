#!/usr/bin/env python3
"""GDN-specific preflight, layered on top of the DeepStack sub-track's
``preflight_provenance.py``.

CPU-only. Does NOT touch a GPU. In ``--dry-run`` mode does NOT hit the
network. When run without ``--dry-run`` it queries the HF API for the
Qwen3.5-4B config and verifies that the GDN-relevant fields
(``layers_block_type``, ``linear_num_key_heads`` etc.) are present and
non-degenerate as required by ``provenance.md`` §3.

Exit codes:
    0 — all hard pins agree.
    1 — at least one hard pin disagrees.
    2 — usage / IO error.
    3 — a required check could not run AND ``--strict`` is set.

Usage
-----

    python3 scripts/gdn_preflight.py --dry-run           # skips HF fetch
    python3 scripts/gdn_preflight.py                     # full run
    python3 scripts/gdn_preflight.py --strict            # non-zero on any warn
    python3 scripts/gdn_preflight.py --json > preflight.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

# --- Pins (mirror gdn/provenance.md §2 and §3) ------------------------

PINNED_MODEL_ID = "Qwen/Qwen3.5-4B"
PINNED_MODEL_SHA = "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
PINNED_MODEL_ARCH = ["Qwen3_5ForConditionalGeneration"]
PINNED_MODEL_TYPE = "qwen3_5"
PINNED_LM_CLASS = "Qwen3_5ForCausalLM"  # non-MoE

# Fields under config.text_config on the HF config.json for Qwen3.5.
# SGLang internally exposes `config.layers_block_type` translated from
# HF's `text_config.layer_types` (HF values: {"linear_attention",
# "full_attention"} → SGLang values: {"linear_attention", "attention"}).
# See source_audit.md §1 for the translation.
REQUIRED_GDN_CONFIG_FIELDS = (
    "num_hidden_layers",
    "layer_types",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_key_head_dim",
    "linear_value_head_dim",
    "linear_conv_kernel_dim",
    "rms_norm_eps",
    "hidden_act",
)

# HF layer_types values that count as "linear-attention / GDN".
LINEAR_ATTENTION_TYPES = {"linear_attention"}
# HF layer_types values that count as standard attention.
FULL_ATTENTION_TYPES = {"full_attention", "attention"}

# Ratios for which the fused
# fused_qkvzba_split_reshape_cat_contiguous path is hit (see
# source_audit.md §3 items 3.4 vs 3.5).
FUSED_PATH_RATIOS = (1, 2, 4)


def _http_get_json(url: str, timeout: float = 10.0) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:  # noqa: S310
            return json.loads(fh.read().decode("utf-8"))
    except Exception:
        return None


def _http_get_text(url: str, timeout: float = 10.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as fh:  # noqa: S310
            return fh.read().decode("utf-8")
    except Exception:
        return None


def load_model_metadata(dry_run: bool) -> dict:
    """Fetch and validate the HF model metadata for Qwen/Qwen3.5-4B."""
    label = "hf_model_metadata"
    if dry_run:
        return {
            "check": label,
            "status": "SKIPPED_DRY_RUN",
            "pinned_id": PINNED_MODEL_ID,
            "pinned_sha": PINNED_MODEL_SHA,
        }
    api_url = f"https://huggingface.co/api/models/{PINNED_MODEL_ID}"
    data = _http_get_json(api_url)
    if data is None:
        return {"check": label, "status": "ERROR_FETCH", "url": api_url}
    got_sha = data.get("sha")
    got_arch = (data.get("config") or {}).get("architectures")
    got_type = (data.get("config") or {}).get("model_type")
    ok = (
        got_sha == PINNED_MODEL_SHA
        and got_arch == PINNED_MODEL_ARCH
        and got_type == PINNED_MODEL_TYPE
    )
    return {
        "check": label,
        "status": "OK" if ok else "MISMATCH",
        "got_sha": got_sha,
        "pinned_sha": PINNED_MODEL_SHA,
        "got_arch": got_arch,
        "pinned_arch": PINNED_MODEL_ARCH,
        "got_type": got_type,
        "pinned_type": PINNED_MODEL_TYPE,
    }


def load_gdn_config(dry_run: bool) -> dict:
    """Fetch the raw config.json and record GDN-specific fields."""
    label = "gdn_config_fields"
    if dry_run:
        return {
            "check": label,
            "status": "SKIPPED_DRY_RUN",
            "required_fields": list(REQUIRED_GDN_CONFIG_FIELDS),
        }
    cfg_url = (
        f"https://huggingface.co/{PINNED_MODEL_ID}/raw/{PINNED_MODEL_SHA}/config.json"
    )
    text = _http_get_text(cfg_url)
    if text is None:
        return {"check": label, "status": "ERROR_FETCH", "url": cfg_url}
    try:
        cfg = json.loads(text)
    except json.JSONDecodeError as e:
        return {"check": label, "status": "ERROR_JSON", "error": str(e)}

    # Qwen3.5 nests the language-model config under text_config; the
    # GDN fields all live there. If text_config is missing we still
    # attempt the top-level lookup as a fallback so future config
    # shapes are not silently misread.
    text_cfg = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    observed = {k: text_cfg.get(k, "<MISSING>") for k in REQUIRED_GDN_CONFIG_FIELDS}

    missing = [k for k, v in observed.items() if v == "<MISSING>"]

    # Hybrid check: layer_types (HF field) must contain both a linear-
    # attention entry and a full-attention / standard-attention entry.
    lbt = observed.get("layer_types")
    n_attn = n_linear = None
    hybrid_ok = False
    if isinstance(lbt, list) and lbt:
        n_attn = sum(1 for t in lbt if t in FULL_ATTENTION_TYPES)
        n_linear = sum(1 for t in lbt if t in LINEAR_ATTENTION_TYPES)
        hybrid_ok = n_attn > 0 and n_linear > 0

    # Fused-path check.
    nvh = observed.get("linear_num_value_heads")
    nkh = observed.get("linear_num_key_heads")
    heads_ratio = None
    fused_path = None
    if isinstance(nvh, int) and isinstance(nkh, int) and nkh > 0 and nvh % nkh == 0:
        heads_ratio = nvh // nkh
        fused_path = heads_ratio in FUSED_PATH_RATIOS

    status = "OK"
    if missing:
        status = "MISSING_FIELDS"
    elif not hybrid_ok:
        status = "NOT_HYBRID"

    return {
        "check": label,
        "status": status,
        "observed": observed,
        "n_full_attention_layers": n_attn,
        "n_linear_attention_layers": n_linear,
        "heads_ratio": heads_ratio,
        "fused_split_path": fused_path,
        "missing_fields": missing,
        "hybrid_ok": hybrid_ok,
        "config_source": "text_config" if isinstance(cfg.get("text_config"), dict) else "top_level",
    }


def env_gdn_flags() -> dict:
    """Record env vars that affect GDN capture behaviour."""
    return {
        "check": "gdn_env_flags",
        "status": "OK",
        "SGLANG_GDN_QKVZ_BA_ALT_STREAM": os.environ.get(
            "SGLANG_GDN_QKVZ_BA_ALT_STREAM"
        ),
        "SGLANG_KERNEL_API_LOGLEVEL": os.environ.get("SGLANG_KERNEL_API_LOGLEVEL"),
        "SGLANG_KERNEL_API_LOGDEST": os.environ.get("SGLANG_KERNEL_API_LOGDEST"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "LD_PRELOAD": os.environ.get("LD_PRELOAD"),
    }


def frozen_checkout_present() -> dict:
    """Best-effort locate the frozen SGLang checkout without importing torch."""
    # We do not want to shell out to git here; just record whether an
    # expected checkout path exists under a set of scratchpad prefixes
    # that the DeepStack sub-track uses. The main runner (runner.sh)
    # does the authoritative check.
    candidates = []
    for base in (
        os.environ.get("QWEN35_FROZEN_SGLANG_PATH"),
        "/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/sglang_checkout/sglang",
    ):
        if base and Path(base).exists():
            candidates.append(base)
    return {
        "check": "frozen_checkout_present",
        "status": "OK" if candidates else "MISSING",
        "candidates": candidates,
    }


def run(dry_run: bool, strict: bool, want_json: bool) -> int:
    results = [
        load_model_metadata(dry_run),
        load_gdn_config(dry_run),
        env_gdn_flags(),
        frozen_checkout_present(),
    ]
    payload = {"preflight": "gdn", "results": results, "strict": strict}
    if want_json:
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        for r in results:
            print(f"[{r['status']}] {r['check']}: {json.dumps(r, sort_keys=True)}")
    hard_bad = [
        r
        for r in results
        if r["status"] in {"MISMATCH", "MISSING_FIELDS", "NOT_HYBRID", "ERROR_FETCH"}
    ]
    if hard_bad:
        return 1
    if strict and any(r["status"] not in {"OK", "SKIPPED_DRY_RUN"} for r in results):
        return 3
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dry-run", action="store_true", help="skip network calls"
    )
    p.add_argument("--strict", action="store_true", help="non-zero on any warn")
    p.add_argument("--json", action="store_true", help="machine-readable output")
    args = p.parse_args(argv)
    return run(dry_run=args.dry_run, strict=args.strict, want_json=args.json)


if __name__ == "__main__":
    raise SystemExit(main())
