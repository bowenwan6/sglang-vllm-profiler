#!/usr/bin/env python3
"""
Issue #4 v3 — step 0.4 engagement verifier.

The piece v2 did not have, and the one thing that makes v3 trustworthy.

Every lever in this experiment now has a *silent degradation* path (plan.md
§11.1 changes A-D): the multimodal transport falls back to CPU per-tensor when
the IPC pool is full, the piecewise backend falls back to eager per-subgraph
when the capture stream is missing, and the deprecated flag surface still
accepts our old flags while meaning something else. None of these crash. A
clean-looking latency number can therefore describe a configuration we did not
ask for.

So no arm's number is quotable unless this module says VERIFIED.

Evidence used, in order of strength:

  1. **Resolved configuration** from ``GET /server_info`` -- documented upstream
     as "the resolution result: what the launcher was given, with every decision
     resolution made applied over it". Compared against what the arm requested.
  2. **Behavioural** prefill-graph engagement: the scheduler's per-prefill log
     line ends with ``cuda graph: True|False``
     (``metrics_reporter.py:655``). A prefill-graph arm whose batches report
     False was not using the graph, whatever the config says.
  3. **Degradation signals** in the server log:
       - ``PCG capture stream is not set``    (cuda_piecewise_backend.py:168)
       - ``falling back to non-IPC transport`` (transport/cuda_ipc.py:174)
       - any ``DeprecationWarning``/``FutureWarning`` naming a flag we set.
"""
from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

# Degradation signals: (regex, short reason, which requests it invalidates)
PCG_EAGER_FALLBACK = re.compile(r"PCG capture stream is not set", re.I)
IPC_POOL_FALLBACK = re.compile(r"falling back to non-IPC transport", re.I)
MM_POOL_EXHAUSTED = re.compile(r"MmItemMemoryPool has no free chunk", re.I)
# Measurement instrumentation on the pinned stack (see ../manifest.md §7):
# PCG_STATS <tag> eligible=<n> eager_fallback=<n> eager_shapes=<n>
PCG_STATS = re.compile(
    r"PCG_STATS \w+ eligible=(\d+) eager_fallback=(\d+) eager_shapes=(\d+)"
)
DEPRECATION = re.compile(
    r"(is deprecated|DeprecationWarning|FutureWarning).{0,200}", re.I | re.S
)
# Prefill stats line (metrics_reporter.py:625-656). `#new-token` is captured so
# the server's own 1-token readiness probes can be excluded from the denominator:
# they are not benchmark work and they never run under a graph, so counting them
# drags a perfectly healthy arm toward the failure floor.
PREFILL_GRAPH = re.compile(
    r"Prefill batch.*?#new-token: (\d+), #cached-token: (\d+).*?"
    r"cuda graph: (True|False)", re.I
)
# Smallest captured prefill bucket is 4 tokens; 8 keeps a clear margin above the
# 1-token probes without excluding any real request.
BENCH_BATCH_MIN_TOKENS = 8

# Direct proof of what was actually captured, independent of /server_info
# (cuda_graph_runner: "Capture target prefill CUDA graph begin. backend=<x>").
CAPTURE_BEGIN = re.compile(
    r"Capture target prefill CUDA graph begin\. backend=(\w+)", re.I
)

# Flags whose deprecation would prove we are still driving an old surface.
FLAGS_WE_SET = (
    "--mm-feature-transport",
    "--cuda-graph-backend-prefill",
    "sglang.benchmark.serving",
    "SGLANG_USE_CUDA_IPC_TRANSPORT",
    "--enforce-piecewise-cuda-graph",
)


def fetch_server_info(port: int, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """Resolved server configuration, or None if the endpoint is unavailable."""
    for path in ("/server_info", "/get_server_info"):
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=timeout
            ) as r:
                return json.loads(r.read().decode())
        except Exception:
            continue
    return None


def _dig(obj: Any, key: str) -> Optional[Any]:
    """Depth-first search for `key` anywhere in a nested dict/list."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            found = _dig(v, key)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _dig(v, key)
            if found is not None:
                return found
    return None


def resolved_prefill_backend(info: Dict[str, Any]) -> Optional[str]:
    """Best-effort extraction of the resolved prefill CUDA-graph backend.

    Tries the structured cuda_graph_config first, then the flat server-arg.
    Returns a lowercase string, or None when nothing could be read -- and None
    is treated as UNVERIFIED by `verify_arm`, never as agreement.
    """
    cfg = _dig(info, "cuda_graph_config")
    if isinstance(cfg, dict):
        prefill = cfg.get("prefill")
        if isinstance(prefill, dict) and prefill.get("backend") is not None:
            return str(prefill["backend"]).lower().split(".")[-1]
    if isinstance(cfg, str) and cfg:
        try:
            parsed = json.loads(cfg)
            prefill = parsed.get("prefill", {})
            if isinstance(prefill, dict) and prefill.get("backend"):
                return str(prefill["backend"]).lower().split(".")[-1]
        except Exception:
            pass
    flat = _dig(info, "cuda_graph_backend_prefill")
    if flat is not None:
        return str(flat).lower().split(".")[-1]
    return None


def resolved_transport(info: Dict[str, Any]) -> Optional[str]:
    """Resolved multimodal feature transport.

    ``mm_feature_transport`` is Optional and *unset means cpu*
    (base_processor.py:247-252 coerces anything outside the literal set to
    "cpu"), so an explicit null is reported as "cpu" rather than unknown.
    """
    val = _dig(info, "mm_feature_transport")
    if val is None:
        # Distinguish "key absent" from "key present and null".
        return "cpu" if _key_present(info, "mm_feature_transport") else None
    return str(val).lower()


def _key_present(obj: Any, key: str) -> bool:
    if isinstance(obj, dict):
        if key in obj:
            return True
        return any(_key_present(v, key) for v in obj.values())
    if isinstance(obj, list):
        return any(_key_present(v, key) for v in obj)
    return False


def scan_server_log(path: Path) -> Dict[str, Any]:
    """Count degradation signals and prefill-graph engagement in a server log."""
    if not path.exists():
        return {"log_missing": True}
    text = path.read_text(errors="replace")
    all_batches = [
        (int(n), int(c), f.lower() == "true")
        for n, c, f in PREFILL_GRAPH.findall(text)
    ]
    bench = [(n, c, f) for n, c, f in all_batches if n >= BENCH_BATCH_MIN_TOKENS]
    probes = [(n, c, f) for n, c, f in all_batches if n < BENCH_BATCH_MIN_TOKENS]
    n_true = sum(1 for _, _, f in bench if f)
    cache_hit_batches = sum(1 for _, c, _ in all_batches if c > 0)
    tok_new = sum(n for n, _, _ in all_batches)
    tok_cached = sum(c for _, c, _ in all_batches)
    captures = CAPTURE_BEGIN.findall(text)
    pcg_stats = PCG_STATS.findall(text)
    pcg_last = (
        {"eligible": int(pcg_stats[-1][0]),
         "eager_fallback": int(pcg_stats[-1][1]),
         "eager_shapes": int(pcg_stats[-1][2])}
        if pcg_stats else None
    )
    deprecations = [
        m.group(0)[:200]
        for m in DEPRECATION.finditer(text)
        if any(f in m.group(0) for f in FLAGS_WE_SET)
    ]
    return {
        "log_missing": False,
        "pcg_eager_fallback": len(PCG_EAGER_FALLBACK.findall(text)),
        "ipc_pool_fallback": len(IPC_POOL_FALLBACK.findall(text)),
        "mm_pool_exhausted": len(MM_POOL_EXHAUSTED.findall(text)),
        "deprecations_naming_our_flags": deprecations[:5],
        "captured_prefill_backend": captures[-1].lower() if captures else None,
        "pcg_stats_last": pcg_last,
        "prefill_batches_logged": len(bench),
        "prefill_graph_true": n_true,
        "prefill_graph_true_pct": (
            round(100.0 * n_true / len(bench), 1) if bench else None
        ),
        "probe_batches_excluded": len(probes),
        "total_prefill_batches": len(all_batches),
        "batches_with_cached_tokens": cache_hit_batches,
        "tokens_new": tok_new,
        "tokens_cached": tok_cached,
        "cached_token_share_pct": (
            round(100.0 * tok_cached / (tok_new + tok_cached), 1)
            if (tok_new + tok_cached) else None
        ),
    }


# vLLM failure signatures. The vLLM arm is a cross-framework *anchor*: we set no
# lever on it, so "did the lever engage" is not a question that applies. What must
# still be true is that it served the workload without falling over.
VLLM_FAILURE = re.compile(
    r"(Traceback \(most recent call last\)|CUDA out of memory|"
    r"Engine (?:core )?(?:process )?(?:failed|died|crashed)|"
    r"AsyncEngineDeadError)", re.I
)


def verify_arm(
    arm_id: str,
    requested_backend: Optional[str],
    requested_transport: Optional[str],
    server_info: Optional[Dict[str, Any]],
    log_path: Path,
    graph_engagement_floor_pct: float = 99.0,
    framework: str = "sglang",
) -> Dict[str, Any]:
    """Return the arm's engagement verdict.

    `requested_backend` / `requested_transport` are None for arms that
    deliberately leave the flag unset; for those we only *record* what the
    default resolved to (that recording is the point of A0_default), and do not
    fail on a mismatch that cannot exist.

    `framework="vllm"` switches to the anchor criteria. vLLM exposes no
    `/server_info`, has no SGLang prefill-graph backend and no SGLang multimodal
    transport, so every SGLang-specific check would fail on it vacuously and the
    arm could never verify -- which would make Q1 unanswerable by construction
    rather than by evidence. What is checked instead is that it served the
    workload cleanly.
    """
    reasons: List[str] = []
    scan = scan_server_log(log_path)

    if scan.get("log_missing"):
        return {
            "arm": arm_id,
            "engagement": "UNVERIFIED",
            "reasons": [f"server log absent: {log_path}"],
            "scan": scan,
        }

    if framework != "sglang":
        text = log_path.read_text(errors="replace")
        hits = VLLM_FAILURE.findall(text)
        if hits:
            reasons.append(
                f"{len(hits)} failure signature(s) in the {framework} server log, "
                f"first: {hits[0][:120]!r}"
            )
        return {
            "arm": arm_id,
            "engagement": "VERIFIED" if not reasons else "UNVERIFIED",
            "reasons": reasons,
            "framework": framework,
            "anchor_only": True,
            "requested_prefill_backend": None,
            "resolved_prefill_backend": None,
            "requested_mm_transport": None,
            "resolved_mm_transport": None,
            "scan": scan,
        }

    if server_info is None:
        reasons.append("/server_info unreachable — resolved config unknown")
        res_backend = res_transport = None
    else:
        res_backend = resolved_prefill_backend(server_info)
        res_transport = resolved_transport(server_info)
        if res_backend is None:
            reasons.append("resolved prefill backend not readable from /server_info")
        elif requested_backend is not None and res_backend != requested_backend:
            reasons.append(
                f"resolved prefill backend {res_backend!r} != requested "
                f"{requested_backend!r}"
            )
        if res_transport is None:
            reasons.append("resolved mm transport not readable from /server_info")
        elif requested_transport is not None and res_transport != requested_transport:
            reasons.append(
                f"resolved mm transport {res_transport!r} != requested "
                f"{requested_transport!r}"
            )

    effective_backend = requested_backend or res_backend

    # Independent capture-time evidence. `/server_info` reports the resolution;
    # this reports what the runner actually captured. They must agree.
    captured = scan.get("captured_prefill_backend")
    if effective_backend in ("breakable", "tc_piecewise", "full"):
        if captured is None:
            reasons.append(
                "no 'Capture target prefill CUDA graph begin' line — the prefill "
                "graph was never captured"
            )
        elif captured != effective_backend:
            reasons.append(
                f"captured prefill backend {captured!r} != effective "
                f"{effective_backend!r}"
            )
    elif effective_backend == "disabled" and captured is not None:
        reasons.append(
            f"a prefill graph was captured (backend={captured!r}) on a "
            "'disabled' arm"
        )

    # Silent partial-eager on any piecewise arm: the number is not a PCG number.
    if scan["pcg_eager_fallback"] and effective_backend == "tc_piecewise":
        stats = scan.get("pcg_stats_last")
        if stats:
            # With the counters present the degradation is measurable, so report
            # the fraction rather than an unbounded suspicion.
            frac = (100.0 * stats["eager_fallback"] / stats["eligible"]
                    if stats["eligible"] else 100.0)
            reasons.append(
                f"PCG eager fallback on {frac:.2f}% of graph-eligible calls "
                f"({stats['eager_fallback']}/{stats['eligible']}, "
                f"{stats['eager_shapes']} distinct shapes) — arm ran partially "
                "eager"
            )
        else:
            reasons.append(
                "'PCG capture stream is not set' fired — arm ran partially eager. "
                "The warning is print_warning_once (@lru_cache), so its count is "
                "capped at 1 and the extent of the degradation is UNBOUNDED from "
                "the log"
            )

    # Silent CPU fallback on any GPU-transport arm.
    if (requested_transport or res_transport) in ("cuda_ipc", "cuda_vmm"):
        if scan["ipc_pool_fallback"] or scan["mm_pool_exhausted"]:
            reasons.append(
                f"multimodal transport fell back to CPU "
                f"({scan['ipc_pool_fallback']} fallback, "
                f"{scan['mm_pool_exhausted']} pool-exhaustion signals)"
            )

    # Proof we are still on an old surface.
    if scan["deprecations_naming_our_flags"]:
        reasons.append(
            "deprecation warning names a flag we set: "
            + scan["deprecations_naming_our_flags"][0]
        )

    # Workload integrity: a prefix cache serving the repeated prompt set turns
    # most reps into cache lookups, so the lever under test barely executes. The
    # sub-threshold batches get filed as "probes" and the arm still reads as
    # healthy -- which is exactly how the first headline attempt passed while
    # measuring almost no prefill. Two independent signals for it.
    total = scan.get("total_prefill_batches") or 0
    probes_n = scan.get("probe_batches_excluded") or 0
    bench_n = scan.get("prefill_batches_logged") or 0
    cached_n = scan.get("batches_with_cached_tokens") or 0
    if total and probes_n > max(10, 0.25 * bench_n):
        reasons.append(
            f"{probes_n} of {total} prefill batches carried < "
            f"{BENCH_BATCH_MIN_TOKENS} new tokens against {bench_n} benchmark "
            "batches — the workload was largely served without prefill "
            "(prefix cache still on?)"
        )
    # Counted in *tokens*, not batches: a few tokens of shared chat-template
    # prefix appear on nearly every batch and are harmless, so a batch-count rule
    # would fire on a perfectly good run. What matters is how much of the
    # prompt work was skipped.
    cached_share = scan.get("cached_token_share_pct")
    if cached_share is not None and cached_share > 20.0:
        reasons.append(
            f"{cached_share}% of prefill tokens came from cache "
            f"({scan['tokens_cached']} cached vs {scan['tokens_new']} new) — "
            "prefix caching is serving the repeated prompt set, so the reps are "
            "not independent measurements of prefill"
        )

    # Behavioural check: a graph-on arm whose prefill batches ran without a graph.
    pct = scan["prefill_graph_true_pct"]
    if effective_backend in ("breakable", "tc_piecewise", "full"):
        if not scan["prefill_batches_logged"]:
            reasons.append("no prefill batch lines logged — cannot confirm graph use")
        elif pct is not None and pct < graph_engagement_floor_pct:
            reasons.append(
                f"only {pct}% of prefill batches ran under a CUDA graph "
                f"(floor {graph_engagement_floor_pct}%)"
            )
    elif effective_backend == "disabled":
        if pct is not None and pct > 0:
            reasons.append(
                f"{pct}% of prefill batches ran under a CUDA graph on a "
                "'disabled' arm — the flag did not take effect"
            )

    return {
        "arm": arm_id,
        "engagement": "VERIFIED" if not reasons else "UNVERIFIED",
        "reasons": reasons,
        "requested_prefill_backend": requested_backend,
        "resolved_prefill_backend": res_backend,
        "requested_mm_transport": requested_transport,
        "resolved_mm_transport": res_transport,
        "scan": scan,
    }


def one_line(verdict: Dict[str, Any]) -> str:
    if verdict.get("anchor_only"):
        return (
            f"engagement: {verdict['engagement']} (cross-framework anchor — no "
            "SGLang lever set; checked for clean serving only"
            + ("" if verdict["engagement"] == "VERIFIED"
               else ": " + "; ".join(verdict["reasons"]))
            + ")"
        )
    if verdict["engagement"] == "VERIFIED":
        return (
            f"engagement: VERIFIED "
            f"(backend={verdict.get('resolved_prefill_backend')}, "
            f"transport={verdict.get('resolved_mm_transport')}, "
            f"captured={verdict['scan'].get('captured_prefill_backend')}, "
            f"graph={verdict['scan'].get('prefill_graph_true_pct')}% of "
            f"{verdict['scan'].get('prefill_batches_logged')} bench prefill "
            f"batches, {verdict['scan'].get('probe_batches_excluded')} probes "
            f"excluded)"
        )
    return "engagement: UNVERIFIED (" + "; ".join(verdict["reasons"]) + ")"


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--port", type=int, default=None,
                    help="live server port to query /server_info")
    ap.add_argument("--server-info-json", type=Path, default=None,
                    help="previously captured /server_info payload")
    ap.add_argument("--arm", default="ad_hoc")
    ap.add_argument("--requested-backend", default=None)
    ap.add_argument("--requested-transport", default=None)
    a = ap.parse_args()

    info = None
    if a.server_info_json and a.server_info_json.exists():
        info = json.loads(a.server_info_json.read_text())
    elif a.port:
        info = fetch_server_info(a.port)

    v = verify_arm(a.arm, a.requested_backend, a.requested_transport, info, a.log)
    print(json.dumps(v, indent=2))
    print(one_line(v))
