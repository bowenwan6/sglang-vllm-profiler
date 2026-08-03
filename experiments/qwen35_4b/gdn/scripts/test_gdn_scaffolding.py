#!/usr/bin/env python3
"""CPU-only smoke tests for the GDN sub-track's scaffolding.

Run with:

    python3 experiments/qwen35_4b/gdn/scripts/test_gdn_scaffolding.py

Prints one line per test. Exits non-zero on the first failure. No torch,
no network, no GPU. The point is to catch obvious breakage before any
runner is invoked against a real server.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))

import generate_gdn_prompts as gp  # noqa: E402
import gdn_preflight as gpreflight  # noqa: E402


def _ok(name: str, note: str = "") -> None:
    print(f"[PASS] {name}" + (f"  ({note})" if note else ""))


def _fail(name: str, msg: str) -> None:
    print(f"[FAIL] {name}: {msg}", file=sys.stderr)
    sys.exit(1)


def test_fixture_regeneration_bit_identical() -> None:
    a = gp.dump_jsonl(gp.build_records())
    b = gp.dump_jsonl(gp.build_records())
    if a != b:
        _fail(
            "fixture_regeneration_bit_identical",
            "two dump_jsonl calls returned different bytes",
        )
    _ok("fixture_regeneration_bit_identical", f"{len(a)} bytes")


def test_fixture_matches_manifest() -> None:
    fixture = REPO / gp.FIXTURE_REL
    manifest = REPO / gp.MANIFEST_REL
    if not fixture.exists() or not manifest.exists():
        _fail(
            "fixture_matches_manifest",
            f"missing files: fixture={fixture.exists()} manifest={manifest.exists()}",
        )
    got = hashlib.sha256(fixture.read_bytes()).hexdigest()
    expected = json.loads(manifest.read_text())["sha256"]
    if got != expected:
        _fail(
            "fixture_matches_manifest",
            f"fixture sha {got} != manifest sha {expected}",
        )
    _ok("fixture_matches_manifest", got)


def test_fixture_records_shape() -> None:
    fixture = REPO / gp.FIXTURE_REL
    lines = fixture.read_text().splitlines()
    if len(lines) != len(gp.SEEDS) * 2:
        _fail(
            "fixture_records_shape",
            f"expected {len(gp.SEEDS) * 2} records, got {len(lines)}",
        )
    ids = {json.loads(line)["id"] for line in lines}
    if len(ids) != len(lines):
        _fail("fixture_records_shape", "duplicate ids in fixture")
    _ok("fixture_records_shape", f"{len(lines)} records")


def test_fixture_lengths_hit_targets() -> None:
    fixture = REPO / gp.FIXTURE_REL
    for line in fixture.read_text().splitlines():
        rec = json.loads(line)
        got = len(rec["text"])
        if got != rec["target_chars"]:
            _fail(
                "fixture_lengths_hit_targets",
                f"record {rec['id']}: text len {got} != target {rec['target_chars']}",
            )
    _ok("fixture_lengths_hit_targets")


def test_generator_check_command_ok() -> None:
    script = HERE / "generate_gdn_prompts.py"
    rc = subprocess.run(
        [sys.executable, str(script), "--check", "--strict"],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("generator_check_command_ok", rc.stderr.strip() or rc.stdout.strip())
    _ok("generator_check_command_ok", rc.stdout.strip())


def test_generator_print_matches_file() -> None:
    fixture = REPO / gp.FIXTURE_REL
    script = HERE / "generate_gdn_prompts.py"
    rc = subprocess.run(
        [sys.executable, str(script), "--print"],
        capture_output=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("generator_print_matches_file", rc.stderr.decode())
    if rc.stdout != fixture.read_bytes():
        _fail("generator_print_matches_file", "stdout bytes != fixture bytes")
    _ok("generator_print_matches_file")


def test_preflight_dry_run_returns_zero() -> None:
    rc = gpreflight.run(dry_run=True, strict=False, want_json=True)
    if rc != 0:
        _fail("preflight_dry_run_returns_zero", f"rc={rc}")
    _ok("preflight_dry_run_returns_zero")


def test_preflight_dry_run_reports_skipped() -> None:
    buf = io.StringIO()
    real_stdout, sys.stdout = sys.stdout, buf
    try:
        gpreflight.run(dry_run=True, strict=False, want_json=True)
    finally:
        sys.stdout = real_stdout
    payload = json.loads(buf.getvalue())
    statuses = {r["check"]: r["status"] for r in payload["results"]}
    for chk in ("hf_model_metadata", "gdn_config_fields", "frozen_sglang_head", "sglang_module_file"):
        if statuses.get(chk) != "SKIPPED_DRY_RUN":
            _fail(
                "preflight_dry_run_reports_skipped",
                f"expected SKIPPED_DRY_RUN for {chk}, got {statuses.get(chk)}",
            )
    _ok("preflight_dry_run_reports_skipped")


def test_preflight_required_fields_include_full_attention_interval() -> None:
    if "full_attention_interval" not in gpreflight.REQUIRED_GDN_CONFIG_FIELDS:
        _fail(
            "preflight_required_fields_include_full_attention_interval",
            f"missing from {gpreflight.REQUIRED_GDN_CONFIG_FIELDS}",
        )
    _ok("preflight_required_fields_include_full_attention_interval")


def test_preflight_libcuda_preload_detects_mismatch() -> None:
    saved = os.environ.get("LD_PRELOAD")
    try:
        os.environ["LD_PRELOAD"] = "/definitely/wrong/libcuda.so.1"
        res = gpreflight.libcuda_preload()
        if res["status"] != "MISMATCH":
            _fail(
                "preflight_libcuda_preload_detects_mismatch",
                f"expected MISMATCH, got {res}",
            )
    finally:
        if saved is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = saved
    _ok("preflight_libcuda_preload_detects_mismatch")


def test_preflight_libcuda_preload_matches_pinned() -> None:
    saved = os.environ.get("LD_PRELOAD")
    try:
        os.environ["LD_PRELOAD"] = gpreflight.PINNED_LIBCUDA_PRELOAD
        res = gpreflight.libcuda_preload()
        if res["status"] != "OK":
            _fail(
                "preflight_libcuda_preload_matches_pinned",
                f"expected OK, got {res}",
            )
    finally:
        if saved is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = saved
    _ok("preflight_libcuda_preload_matches_pinned")


def test_preflight_libcuda_preload_accepts_nsys_injected() -> None:
    """Under nsys profile, LD_PRELOAD is replaced with nsys instrumentation
    libs. Preflight must accept this as OK (nsys owns CUDA interception)."""
    saved = os.environ.get("LD_PRELOAD")
    try:
        os.environ["LD_PRELOAD"] = (
            "/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/libnvperf_target.so:"
            "/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/libnvperf_host.so:"
            "/opt/nvidia/nsight-systems-cli/2026.3.1/target-linux-x64/libToolsInjectionProxy64.so"
        )
        res = gpreflight.libcuda_preload()
        if res["status"] != "OK":
            _fail(
                "preflight_libcuda_preload_accepts_nsys_injected",
                f"expected OK under nsys LD_PRELOAD; got {res}",
            )
        if not res.get("nsys_injected"):
            _fail(
                "preflight_libcuda_preload_accepts_nsys_injected",
                f"expected nsys_injected marker; got {res}",
            )
    finally:
        if saved is None:
            os.environ.pop("LD_PRELOAD", None)
        else:
            os.environ["LD_PRELOAD"] = saved
    _ok("preflight_libcuda_preload_accepts_nsys_injected")


def test_preflight_libcuda_preload_missing_is_missing() -> None:
    saved = os.environ.get("LD_PRELOAD")
    try:
        os.environ.pop("LD_PRELOAD", None)
        res = gpreflight.libcuda_preload()
        if res["status"] != "MISSING":
            _fail(
                "preflight_libcuda_preload_missing_is_missing",
                f"expected MISSING, got {res}",
            )
    finally:
        if saved is not None:
            os.environ["LD_PRELOAD"] = saved
    _ok("preflight_libcuda_preload_missing_is_missing")


def test_preflight_lib_versions_returns_ok() -> None:
    # Best-effort: modules may be present or absent, but structure must hold.
    res = gpreflight.lib_versions()
    if res["status"] != "OK":
        _fail("preflight_lib_versions_returns_ok", f"got {res}")
    for mod in ("torch", "sgl_kernel", "flashinfer"):
        if mod not in res["versions"]:
            _fail("preflight_lib_versions_returns_ok", f"missing {mod} in versions")
    _ok("preflight_lib_versions_returns_ok")


def test_preflight_nsys_version_is_present() -> None:
    # nsys is installed on this host per the audit.
    res = gpreflight.nsys_version()
    if res["status"] not in ("OK", "MISSING"):
        _fail("preflight_nsys_version_is_present", f"got {res}")
    _ok("preflight_nsys_version_is_present", res["status"])


def test_preflight_frozen_sglang_head_dry_run() -> None:
    res = gpreflight.frozen_sglang_head(dry_run=True)
    if res["status"] != "SKIPPED_DRY_RUN":
        _fail("preflight_frozen_sglang_head_dry_run", f"got {res}")
    if res.get("pinned_sha") != gpreflight.PINNED_FROZEN_SGLANG_SHA:
        _fail(
            "preflight_frozen_sglang_head_dry_run",
            "pinned_sha absent or wrong",
        )
    _ok("preflight_frozen_sglang_head_dry_run")


def test_preflight_frozen_sglang_head_live() -> None:
    """When the frozen checkout is present, it must match the pin."""
    res = gpreflight.frozen_sglang_head(dry_run=False)
    if res["status"] == "MISSING":
        # If the frozen checkout isn't on this host, skip gracefully.
        _ok("preflight_frozen_sglang_head_live", "no frozen checkout on host — skipped")
        return
    if res["status"] != "OK":
        _fail("preflight_frozen_sglang_head_live", f"got {res}")
    if res.get("got_sha") != gpreflight.PINNED_FROZEN_SGLANG_SHA:
        _fail(
            "preflight_frozen_sglang_head_live",
            f"HEAD {res.get('got_sha')} != pin {gpreflight.PINNED_FROZEN_SGLANG_SHA}",
        )
    _ok("preflight_frozen_sglang_head_live", res["got_sha"][:12])


def test_preflight_env_flags_recorded() -> None:
    payload = gpreflight.env_gdn_flags()
    for key in (
        "SGLANG_GDN_QKVZ_BA_ALT_STREAM",
        "SGLANG_KERNEL_API_LOGLEVEL",
        "SGLANG_KERNEL_API_LOGDEST",
        "CUDA_VISIBLE_DEVICES",
        "LD_PRELOAD",
    ):
        if key not in payload:
            _fail(
                "preflight_env_flags_recorded",
                f"missing env key {key} in payload",
            )
    _ok("preflight_env_flags_recorded")


def test_materialise_stretch_and_truncate() -> None:
    short = "abcdef"  # 6 chars
    # Stretch: request 32 chars from a 6-char seed, must be exactly 32.
    stretched = gp.materialise(short, 32)
    if len(stretched) != 32:
        _fail("materialise_stretch", f"len={len(stretched)}")
    # Truncate: request 4 chars, must be exactly 4 and a prefix of seed.
    truncated = gp.materialise(short, 4)
    if truncated != short[:4]:
        _fail("materialise_truncate", f"got {truncated!r}")
    # Long seed exact-length passthrough.
    long_seed = "x" * 1000
    same = gp.materialise(long_seed, 1000)
    if same != long_seed:
        _fail("materialise_exact", "seed of exact target length was modified")
    _ok("materialise_stretch_and_truncate")


import gdn_client  # noqa: E402


def test_client_extract_selected_logprobs_from_val_key() -> None:
    meta = {"output_token_logprobs_val": [-0.1, -0.2, -0.3]}
    got = gdn_client._extract_selected_logprobs(meta)
    if got != [-0.1, -0.2, -0.3]:
        _fail("client_extract_selected_logprobs_from_val_key", str(got))
    _ok("client_extract_selected_logprobs_from_val_key")


def test_client_extract_selected_logprobs_from_tuple_form() -> None:
    meta = {"output_token_logprobs": [[-0.1, 42], [-0.2, 43]]}
    got = gdn_client._extract_selected_logprobs(meta)
    if got != [-0.1, -0.2]:
        _fail("client_extract_selected_logprobs_from_tuple_form", str(got))
    _ok("client_extract_selected_logprobs_from_tuple_form")


def test_client_extract_selected_logprobs_missing() -> None:
    got = gdn_client._extract_selected_logprobs({})
    if got is not None:
        _fail("client_extract_selected_logprobs_missing", str(got))
    _ok("client_extract_selected_logprobs_missing")


def test_client_extract_top_logprobs() -> None:
    payload = [[[-0.1, 1, "a"], [-0.2, 2, "b"]], [[-0.05, 3, "c"]]]
    got = gdn_client._extract_top_logprobs({"output_top_logprobs_val": payload})
    if got != payload:
        _fail("client_extract_top_logprobs", str(got))
    _ok("client_extract_top_logprobs")


def test_client_issue_batch_hard_fails_on_partial_response() -> None:
    saved = gdn_client.http_post_json
    try:
        # Mock: return only 1 entry when we sent 3 prompts.
        gdn_client.http_post_json = lambda url, payload, timeout=None: (
            200,
            [{"text": "only-one", "output_ids": [1, 2], "meta_info": {}}],
            None,
        )
        try:
            gdn_client.issue_batch("http://x", ["p1", "p2", "p3"], 4, ["a", "b", "c"])
        except RuntimeError as exc:
            if "partial batch" not in str(exc):
                _fail(
                    "client_issue_batch_hard_fails_on_partial_response",
                    f"wrong exc: {exc}",
                )
        else:
            _fail(
                "client_issue_batch_hard_fails_on_partial_response",
                "expected RuntimeError, none raised",
            )
    finally:
        gdn_client.http_post_json = saved
    _ok("client_issue_batch_hard_fails_on_partial_response")


def test_client_issue_batch_extracts_all_fields() -> None:
    saved = gdn_client.http_post_json
    try:
        canned = [
            {
                "text": "hello world",
                "output_ids": [10, 20, 30],
                "meta_info": {
                    "output_token_logprobs_val": [-0.1, -0.2, -0.3],
                    "output_top_logprobs_val": [[[-0.5, 99, None]], [[-0.6, 100, None]], [[-0.7, 101, None]]],
                    "completion_tokens": 3,
                    "finish_reason": {"type": "length", "length": 3},
                    "prompt_tokens": 5,
                },
            }
        ]
        gdn_client.http_post_json = lambda url, payload, timeout=None: (
            200,
            canned,
            None,
        )
        results = gdn_client.issue_batch("http://x", ["hi"], 3, ["r0"])
        if len(results) != 1:
            _fail("client_issue_batch_extracts_all_fields", "wrong result count")
        r = results[0]
        for key in (
            "output_ids",
            "output_logprobs",
            "output_top_logprobs",
            "output_len_tokens",
            "finish_reason",
            "raw_meta_info",
            "e2e_ms",
        ):
            if key not in r:
                _fail("client_issue_batch_extracts_all_fields", f"missing key {key}")
        if r["output_ids"] != [10, 20, 30]:
            _fail("client_issue_batch_extracts_all_fields", f"output_ids={r['output_ids']}")
        if r["output_logprobs"] != [-0.1, -0.2, -0.3]:
            _fail("client_issue_batch_extracts_all_fields", f"logprobs={r['output_logprobs']}")
        if r["output_len_tokens"] != 3:
            _fail("client_issue_batch_extracts_all_fields", f"len={r['output_len_tokens']}")
        if r["finish_reason"] != {"type": "length", "length": 3}:
            _fail("client_issue_batch_extracts_all_fields", f"finish={r['finish_reason']}")
    finally:
        gdn_client.http_post_json = saved
    _ok("client_issue_batch_extracts_all_fields")


def test_client_request_body_sets_return_logprob_and_top_k() -> None:
    captured: dict = {}
    saved = gdn_client.http_post_json

    def mock_post(url, payload, timeout=None):
        captured["url"] = url
        captured["payload"] = payload
        return (
            200,
            [{"text": "", "output_ids": [], "meta_info": {}}],
            None,
        )

    try:
        gdn_client.http_post_json = mock_post
        gdn_client.issue_batch("http://x", ["hi"], 4, ["r0"])
    finally:
        gdn_client.http_post_json = saved
    body = captured.get("payload") or {}
    if body.get("return_logprob") is not True:
        _fail(
            "client_request_body_sets_return_logprob_and_top_k",
            f"return_logprob={body.get('return_logprob')!r}",
        )
    if body.get("top_logprobs_num") != 1:
        _fail(
            "client_request_body_sets_return_logprob_and_top_k",
            f"top_logprobs_num={body.get('top_logprobs_num')!r}",
        )
    _ok("client_request_body_sets_return_logprob_and_top_k")


def test_client_probe_tokenizer_returns_counts() -> None:
    saved = gdn_client.http_post_json
    try:
        gdn_client.http_post_json = lambda url, payload, timeout=None: (
            200,
            {"tokens": [[1, 2], [3, 4, 5]], "count": [2, 3], "max_model_len": 100},
            None,
        )
        counts, source = gdn_client.probe_tokenizer("http://x", ["hi", "hello"])
    finally:
        gdn_client.http_post_json = saved
    if source != "server_tokenize":
        _fail("client_probe_tokenizer_returns_counts", f"source={source}")
    if counts != [2, 3]:
        _fail("client_probe_tokenizer_returns_counts", f"counts={counts}")
    _ok("client_probe_tokenizer_returns_counts")


def test_client_probe_tokenizer_returns_single_count() -> None:
    saved = gdn_client.http_post_json
    try:
        gdn_client.http_post_json = lambda url, payload, timeout=None: (
            200,
            {"tokens": [1, 2, 3], "count": 3, "max_model_len": 100},
            None,
        )
        counts, source = gdn_client.probe_tokenizer("http://x", ["hi"])
    finally:
        gdn_client.http_post_json = saved
    if source != "server_tokenize" or counts != [3]:
        _fail(
            "client_probe_tokenizer_returns_single_count",
            f"source={source} counts={counts}",
        )
    _ok("client_probe_tokenizer_returns_single_count")


def test_client_probe_tokenizer_falls_back_on_error() -> None:
    saved = gdn_client.http_post_json
    try:
        gdn_client.http_post_json = lambda url, payload, timeout=None: (
            500,
            None,
            "server bug",
        )
        counts, source = gdn_client.probe_tokenizer("http://x", ["hi"])
    finally:
        gdn_client.http_post_json = saved
    if source != "fallback_char_heuristic" or counts is not None:
        _fail(
            "client_probe_tokenizer_falls_back_on_error",
            f"source={source} counts={counts}",
        )
    _ok("client_probe_tokenizer_falls_back_on_error")


def test_client_dry_run_probe() -> None:
    fixtures_dir = REPO / "experiments/qwen35_4b/gdn/fixtures"
    script = HERE / "gdn_client.py"
    rc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--server",
            "http://127.0.0.1:0",
            "--arm",
            "A0",
            "--prompt-len",
            "128",
            "--batch-size",
            "4",
            "--new-tokens",
            "16",
            "--fixtures-dir",
            str(fixtures_dir),
            "--output",
            "/tmp/gdn_dry.jsonl",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("client_dry_run_probe", rc.stderr.strip() or rc.stdout.strip())
    payload = json.loads(rc.stdout)
    for key in (
        "arm",
        "server",
        "prompt_len_target_tokens",
        "prompt_len_target_chars",
        "batch_size",
        "n_seeds_used",
        "n_fixture_records",
    ):
        if key not in payload:
            _fail("client_dry_run_probe", f"missing key {key} in dry-run probe")
    if payload["n_seeds_used"] != (2 + 8) * 4:
        _fail(
            "client_dry_run_probe",
            f"n_seeds_used={payload['n_seeds_used']} != (warmup+timed)*batch_size",
        )
    _ok("client_dry_run_probe", f"prompt_chars={payload['prompt_len_target_chars']}")


def test_client_materialise_matches_generator() -> None:
    # The client's materialise() must be byte-identical to the
    # generator's materialise() so timed prompts are deterministic
    # cross-machine.
    seed = "abcdef"
    if gdn_client.materialise_prompt(seed, 40) != gp.materialise(seed, 40):
        _fail(
            "client_materialise_matches_generator",
            "client stretch != generator stretch",
        )
    if gdn_client.materialise_prompt(seed, 3) != gp.materialise(seed, 3):
        _fail(
            "client_materialise_matches_generator",
            "client truncate != generator truncate",
        )
    _ok("client_materialise_matches_generator")


def test_runner_dry_run_shell_exec() -> None:
    runner = HERE / "gdn_runner.sh"
    if not runner.is_file():
        _fail("runner_dry_run_shell_exec", f"missing {runner}")
    rc = subprocess.run(
        [
            "bash",
            str(runner),
            "--dry-run",
            "--gpu-id",
            "0",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("runner_dry_run_shell_exec", rc.stderr.strip() or rc.stdout.strip())
    if "gdn_runner: --dry-run" not in rc.stdout:
        _fail(
            "runner_dry_run_shell_exec",
            f"expected dry-run banner in stdout; got {rc.stdout!r}",
        )
    _ok("runner_dry_run_shell_exec")


def test_runner_rejects_missing_gpu_id() -> None:
    runner = HERE / "gdn_runner.sh"
    rc = subprocess.run(
        ["bash", str(runner)],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 64:
        _fail(
            "runner_rejects_missing_gpu_id",
            f"expected exit 64 (usage), got {rc.returncode}. stderr={rc.stderr!r}",
        )
    if "FATAL: no --gpu-id" not in rc.stderr:
        _fail(
            "runner_rejects_missing_gpu_id",
            f"expected FATAL banner; got {rc.stderr!r}",
        )
    _ok("runner_rejects_missing_gpu_id")


def test_runner_rejects_gpu_outside_allowlist() -> None:
    runner = HERE / "gdn_runner.sh"
    # 99 is outside the {0..7} allowlist (Amendment 1 default).
    rc = subprocess.run(
        ["bash", str(runner), "--gpu-id", "99"],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 64:
        _fail(
            "runner_rejects_gpu_outside_allowlist",
            f"expected exit 64, got {rc.returncode}. stderr={rc.stderr!r}",
        )
    if "not in allowlist" not in rc.stderr:
        _fail(
            "runner_rejects_gpu_outside_allowlist",
            f"expected allowlist FATAL; got {rc.stderr!r}",
        )
    _ok("runner_rejects_gpu_outside_allowlist")


def test_runner_dry_run_context_blob_valid_json() -> None:
    runner = HERE / "gdn_runner.sh"
    rc = subprocess.run(
        [
            "bash",
            str(runner),
            "--dry-run",
            "--gpu-id",
            "1",
            "--arm",
            "A1",
            "--prompt-len",
            "512",
            "--batch-size",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail(
            "runner_dry_run_context_blob_valid_json",
            rc.stderr.strip() or rc.stdout.strip(),
        )
    # Extract path from stdout ("gdn_runner: context blob written to <path>").
    ctx_path = None
    for line in rc.stdout.splitlines():
        if line.startswith("gdn_runner: context blob written to "):
            ctx_path = line[len("gdn_runner: context blob written to ") :]
            break
    if ctx_path is None or not Path(ctx_path).is_file():
        _fail(
            "runner_dry_run_context_blob_valid_json",
            f"context blob path not found in stdout: {rc.stdout!r}",
        )
    payload = json.loads(Path(ctx_path).read_text())
    if payload.get("arm") != "A1":
        _fail(
            "runner_dry_run_context_blob_valid_json",
            f"arm={payload.get('arm')!r}, expected A1",
        )
    flags = payload.get("arm_flags", [])
    expected_flags = {
        "--cuda-graph-backend-prefill=breakable",
        "--cuda-graph-backend-decode=disabled",
    }
    missing = expected_flags - set(flags)
    if missing:
        _fail(
            "runner_dry_run_context_blob_valid_json",
            f"A1 arm_flags missing {missing}; got {flags!r}",
        )
    _ok("runner_dry_run_context_blob_valid_json", ctx_path)


def test_runner_env_exports_precede_preflight() -> None:
    """T1 fix: env exports must be BEFORE the preflight write.

    Regression guard for audit B4 — LD_PRELOAD / CUDA_VISIBLE_DEVICES
    must be visible to gdn_preflight.py.
    """
    runner = (HERE / "gdn_runner.sh").read_text()
    preflight_ix = runner.find("python3 \"$HERE/gdn_preflight.py\"")
    ld_preload_ix = runner.find('export LD_PRELOAD=')
    cuda_ix = runner.find('export CUDA_VISIBLE_DEVICES=')
    if preflight_ix < 0 or ld_preload_ix < 0 or cuda_ix < 0:
        _fail(
            "runner_env_exports_precede_preflight",
            f"missing markers: preflight={preflight_ix} ld={ld_preload_ix} cuda={cuda_ix}",
        )
    if ld_preload_ix >= preflight_ix:
        _fail(
            "runner_env_exports_precede_preflight",
            f"LD_PRELOAD export at {ld_preload_ix} not before preflight at {preflight_ix}",
        )
    if cuda_ix >= preflight_ix:
        _fail(
            "runner_env_exports_precede_preflight",
            f"CUDA_VISIBLE_DEVICES export at {cuda_ix} not before preflight at {preflight_ix}",
        )
    _ok("runner_env_exports_precede_preflight")


def test_runner_writes_gpu_post_and_metadata() -> None:
    """T1 fix: the runner writes gpu_post.txt and metadata.json.

    Static-analysis regression guard for audit B6 + R1.
    """
    runner = (HERE / "gdn_runner.sh").read_text()
    if "GPU_RETURNED_CLEAN" not in runner:
        _fail("runner_writes_gpu_post_and_metadata", "missing GPU_RETURNED_CLEAN literal")
    if "GPU_STILL_HOLDS_" not in runner:
        _fail("runner_writes_gpu_post_and_metadata", "missing GPU_STILL_HOLDS_ literal")
    if "gpu_post.txt" not in runner:
        _fail("runner_writes_gpu_post_and_metadata", "missing gpu_post.txt write")
    if "metadata.json" not in runner:
        _fail("runner_writes_gpu_post_and_metadata", "missing metadata.json write")
    if "written_by" not in runner or "gdn_runner.sh" not in runner:
        _fail(
            "runner_writes_gpu_post_and_metadata",
            "metadata payload missing written_by field",
        )
    if 'exit 77' not in runner:
        _fail(
            "runner_writes_gpu_post_and_metadata",
            "missing exit 77 on GPU-not-clean path",
        )
    _ok("runner_writes_gpu_post_and_metadata")


def test_runner_mkdir_before_preflight_write() -> None:
    """T1 fix: RESULTS_DIR must be mkdir-p'd before the preflight write."""
    runner = (HERE / "gdn_runner.sh").read_text()
    mkdir_ix = runner.find('mkdir -p "$RESULTS_DIR"')
    preflight_ix = runner.find("python3 \"$HERE/gdn_preflight.py\"")
    if mkdir_ix < 0:
        _fail(
            "runner_mkdir_before_preflight_write",
            "missing mkdir -p $RESULTS_DIR",
        )
    if mkdir_ix >= preflight_ix:
        _fail(
            "runner_mkdir_before_preflight_write",
            f"mkdir at {mkdir_ix} not before preflight at {preflight_ix}",
        )
    _ok("runner_mkdir_before_preflight_write")


def test_gdn_instrumentation_install_is_noop() -> None:
    import gdn_instrumentation as gi
    # install() should not raise and should not import torch or sglang
    # (we don't have those in the CPU env necessarily).
    gi.install()
    log = Path(f"/tmp/qwen35_gdn_instrumentation_{os.getpid()}.json")
    if not log.is_file():
        _fail("gdn_instrumentation_install_is_noop", f"missing log {log}")
    payload = json.loads(log.read_text())
    if payload.get("installs_hooks") is not False:
        _fail(
            "gdn_instrumentation_install_is_noop",
            f"installs_hooks must be False for baseline; got {payload!r}",
        )
    _ok("gdn_instrumentation_install_is_noop")


import gdn_correctness as gc  # noqa: E402
import gdn_verdict as gv  # noqa: E402


def test_correctness_gate1_pass_on_identical_arms() -> None:
    labelled = {
        "A0": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
        "A1": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "PASS":
        _fail("correctness_gate1_pass_on_identical_arms", json.dumps(v))
    _ok("correctness_gate1_pass_on_identical_arms")


def test_correctness_gate1_fail_on_token_divergence() -> None:
    labelled = {
        "A0": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
        "A1": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 4],  # last token differs
                "output_logprobs": [-0.1, -0.2, -0.9],
            }
        ],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "FAIL":
        _fail("correctness_gate1_fail_on_token_divergence", json.dumps(v))
    _ok("correctness_gate1_fail_on_token_divergence")


def test_correctness_gate1_fail_on_logprob_over_tolerance() -> None:
    labelled = {
        "A0": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
        "A1": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],  # tokens equal
                "output_logprobs": [-0.1, -0.2, -0.9],  # delta 0.6 > 0.05
            }
        ],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "FAIL":
        _fail(
            "correctness_gate1_fail_on_logprob_over_tolerance", json.dumps(v)
        )
    _ok("correctness_gate1_fail_on_logprob_over_tolerance")


def test_correctness_gate1_requires_A0_reference() -> None:
    labelled = {
        "A1": [{"prompt_source_id": "p0", "output_ids": [1]}],
        "A2": [{"prompt_source_id": "p0", "output_ids": [1]}],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "FAIL":
        _fail("correctness_gate1_requires_A0_reference", json.dumps(v))
    if "requires an 'A0' arm" not in v.get("reason", ""):
        _fail(
            "correctness_gate1_requires_A0_reference",
            f"expected missing-A0 reason; got {v.get('reason')!r}",
        )
    _ok("correctness_gate1_requires_A0_reference")


def test_correctness_gate2_pairwise_isolation() -> None:
    labelled = {
        "alone": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
        "batched": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
    }
    v = gc.gate_verdict(2, labelled, tolerance=0.05)
    if v["overall"] != "PASS":
        _fail("correctness_gate2_pairwise_isolation", json.dumps(v))
    _ok("correctness_gate2_pairwise_isolation")


def test_correctness_gate3_pairwise_chunking() -> None:
    labelled = {
        "small_chunk": [
            {
                "prompt_source_id": "p0",
                "output_ids": [4, 5],
                "output_logprobs": [-0.1, -0.2],
            }
        ],
        "single_chunk": [
            {
                "prompt_source_id": "p0",
                "output_ids": [4, 5],
                "output_logprobs": [-0.1, -0.2],
            }
        ],
    }
    v = gc.gate_verdict(3, labelled, tolerance=0.05)
    if v["overall"] != "PASS":
        _fail("correctness_gate3_pairwise_chunking", json.dumps(v))
    _ok("correctness_gate3_pairwise_chunking")


def test_gate1_fails_when_output_ids_missing() -> None:
    """T4: whitespace fallback removed — missing output_ids => hard FAIL."""
    labelled = {
        "A0": [
            {
                "prompt_source_id": "p0",
                "output_text": "hello world",
                "output_logprobs": [-0.1, -0.2],
            }
        ],
        "A1": [
            {
                "prompt_source_id": "p0",
                "output_text": "hello world",
                "output_logprobs": [-0.1, -0.2],
            }
        ],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "FAIL":
        _fail("gate1_fails_when_output_ids_missing", json.dumps(v))
    reason_seen = str(v)
    if "MISSING_OUTPUT_IDS" not in reason_seen:
        _fail(
            "gate1_fails_when_output_ids_missing",
            f"expected MISSING_OUTPUT_IDS in reasons; got {reason_seen}",
        )
    _ok("gate1_fails_when_output_ids_missing")


def test_gate1_fails_when_logprobs_missing() -> None:
    """T4: missing output_logprobs no longer treated as within tolerance."""
    labelled = {
        "A0": [{"prompt_source_id": "p0", "output_ids": [1, 2]}],
        "A1": [{"prompt_source_id": "p0", "output_ids": [1, 2]}],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "FAIL":
        _fail("gate1_fails_when_logprobs_missing", json.dumps(v))
    if "MISSING_LOGPROBS" not in str(v):
        _fail(
            "gate1_fails_when_logprobs_missing",
            "expected MISSING_LOGPROBS in reasons",
        )
    _ok("gate1_fails_when_logprobs_missing")


def test_gate1_uses_noise_floor_scaled_tolerance() -> None:
    """T4: max(0.05, 3 * noise_floor). With noise_floor=0.02, tol=0.06."""
    tol = gc.resolve_tolerance(base=0.05, noise_floor=0.02)
    if abs(tol - 0.06) > 1e-9:
        _fail("gate1_uses_noise_floor_scaled_tolerance", f"tol={tol}")
    # A pair with delta 0.055 passes under tol=0.06 but fails under tol=0.05.
    labelled = {
        "A0": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            }
        ],
        "A1": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.355],  # delta 0.055
            }
        ],
    }
    v_high = gc.gate_verdict(1, labelled, tolerance=0.06)
    v_low = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v_high["overall"] != "PASS":
        _fail(
            "gate1_uses_noise_floor_scaled_tolerance",
            f"expected PASS at tol=0.06; got {v_high}",
        )
    if v_low["overall"] != "FAIL":
        _fail(
            "gate1_uses_noise_floor_scaled_tolerance",
            f"expected FAIL at tol=0.05; got {v_low}",
        )
    _ok("gate1_uses_noise_floor_scaled_tolerance", f"3*0.02={tol}")


def test_gate1_requires_all_repeat_samples_to_match() -> None:
    """T4: aggregate across all timed records per prompt id.

    One PASS record and one FAIL record for the same prompt id should
    aggregate to FAIL (previously only the first record was compared).
    """
    labelled = {
        "A0": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            },
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],
                "output_logprobs": [-0.1, -0.2, -0.3],
            },
        ],
        "A1": [
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 3],  # PASS pair with A0[0]
                "output_logprobs": [-0.1, -0.2, -0.3],
            },
            {
                "prompt_source_id": "p0",
                "output_ids": [1, 2, 4],  # FAIL pair (token diff)
                "output_logprobs": [-0.1, -0.2, -0.3],
            },
        ],
    }
    v = gc.gate_verdict(1, labelled, tolerance=0.05)
    if v["overall"] != "FAIL":
        _fail("gate1_requires_all_repeat_samples_to_match", json.dumps(v))
    _ok("gate1_requires_all_repeat_samples_to_match")


def test_resolve_tolerance_uses_floor() -> None:
    if gc.resolve_tolerance(0.05, 0.0) != 0.05:
        _fail("resolve_tolerance_uses_floor", "noise_floor=0 should return base")
    if gc.resolve_tolerance(0.05, 0.03) != 0.09:
        _fail("resolve_tolerance_uses_floor", "3 * 0.03 = 0.09 > 0.05")
    if gc.resolve_tolerance(0.05, -0.5) != 0.05:
        _fail("resolve_tolerance_uses_floor", "negative noise_floor should be clamped")
    _ok("resolve_tolerance_uses_floor")


def test_correctness_dry_run_command_ok() -> None:
    script = HERE / "gdn_correctness.py"
    rc = subprocess.run(
        [sys.executable, str(script), "--gate", "1", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("correctness_dry_run_command_ok", rc.stderr.strip() or rc.stdout.strip())
    payload = json.loads(rc.stdout)
    if payload.get("overall") != "PASS":
        _fail("correctness_dry_run_command_ok", json.dumps(payload))
    _ok("correctness_dry_run_command_ok")


def _gs_all_pass():
    return {
        "per_gate": {i: {"overall": "PASS"} for i in (1, 2, 3, 4)},
        "all_pass": True,
        "any_missing": False,
        "any_failed": False,
        "partial": False,
    }


def _gs_gate2_failed():
    return {
        "per_gate": {
            1: {"overall": "PASS"},
            2: {"overall": "FAIL"},
            3: {"overall": "PASS"},
            4: {"overall": "PASS"},
        },
        "all_pass": False,
        "any_missing": False,
        "any_failed": True,
        "partial": False,
    }


def _gs_gate2_missing():
    return {
        "per_gate": {
            1: {"overall": "PASS"},
            2: {"overall": "MISSING"},
            3: {"overall": "PASS"},
            4: {"overall": "PASS"},
        },
        "all_pass": False,
        "any_missing": True,
        "any_failed": False,
        "partial": True,
    }


def test_verdict_all_gates_pass_no_perf_gap_is_ambiguous() -> None:
    # Without perf findings, the verdict should be AMBIGUOUS (perf side
    # has no signal to declare either way).
    perf_summary = {
        "findings": [],
        "any_H_A": False,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": False,
    }
    v = gv.decide(_gs_all_pass(), perf_summary, infra_failure=False)
    if v != "AMBIGUOUS":
        _fail("verdict_all_gates_pass_no_perf_gap_is_ambiguous", v)
    _ok("verdict_all_gates_pass_no_perf_gap_is_ambiguous")


def test_verdict_gate_failure_wins_over_perf() -> None:
    perf_summary = {
        "findings": [{"any": "thing"}],
        "any_H_A": True,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": True,
    }
    v = gv.decide(_gs_gate2_failed(), perf_summary, infra_failure=False)
    if v != "FAIL_BCG_GDN_CORRECTNESS":
        _fail("verdict_gate_failure_wins_over_perf", v)
    _ok("verdict_gate_failure_wins_over_perf")


def test_verdict_partial_sweep_when_gate_missing_but_others_pass() -> None:
    """T5 fix: MISSING gate with no FAIL emits PARTIAL_SWEEP, not
    FAIL_BCG_GDN_CORRECTNESS."""
    perf_summary = {
        "findings": [],
        "any_H_A": False,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": False,
    }
    v = gv.decide(_gs_gate2_missing(), perf_summary, infra_failure=False)
    if v != "PARTIAL_SWEEP":
        _fail("verdict_partial_sweep_when_gate_missing_but_others_pass", v)
    _ok("verdict_partial_sweep_when_gate_missing_but_others_pass")


def test_verdict_notable_gap_when_gates_pass_and_h_a() -> None:
    perf_summary = {
        "findings": [{"any": "thing"}],
        "any_H_A": True,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": True,
    }
    v = gv.decide(_gs_all_pass(), perf_summary, infra_failure=False)
    if v != "PASS_BCG_GDN_NOTABLE_GAP":
        _fail("verdict_notable_gap_when_gates_pass_and_h_a", v)
    _ok("verdict_notable_gap_when_gates_pass_and_h_a")


def test_verdict_no_gap_when_gates_pass_findings_no_hypothesis() -> None:
    perf_summary = {
        "findings": [{"any": "thing"}],
        "any_H_A": False,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": False,
    }
    v = gv.decide(_gs_all_pass(), perf_summary, infra_failure=False)
    if v != "PASS_BCG_GDN_NO_GAP":
        _fail("verdict_no_gap_when_gates_pass_findings_no_hypothesis", v)
    _ok("verdict_no_gap_when_gates_pass_findings_no_hypothesis")


def test_verdict_infra_failure_wins_over_everything() -> None:
    perf_summary = {
        "findings": [],
        "any_H_A": True,
        "any_H_B": True,
        "any_H_C": True,
        "any_bcg_gap": True,
    }
    v = gv.decide(_gs_all_pass(), perf_summary, infra_failure=True)
    if v != "INFRA_FAILURE":
        _fail("verdict_infra_failure_wins_over_everything", v)
    _ok("verdict_infra_failure_wins_over_everything")


def test_verdict_h_a_ignores_A2() -> None:
    """T5 fix: H_A scoped to BCG-enabled arms {A1, A3} only.

    Synthetic perf rows where only A2 shows kernel-count inflation must
    NOT trigger a PASS_BCG_GDN_NOTABLE_GAP verdict — the audit's B8
    scoping bug would have wrongly attributed a decode-CG artefact
    to the BCG hypothesis.
    """
    rows = []
    for _ in range(3):
        rows.append(
            {"arm": "A0", "prompt_len": "128", "batch": "1",
             "kernel_count_gdn": "100", "p95_launch_gap_us": "5.0",
             "graph_breaks": "0"}
        )
    for _ in range(3):
        rows.append(
            {"arm": "A2", "prompt_len": "128", "batch": "1",
             "kernel_count_gdn": "500", "p95_launch_gap_us": "5.0",
             "graph_breaks": "0"}
        )
    perf = gv._score_perf(rows)
    if perf["any_H_A"]:
        _fail(
            "verdict_h_a_ignores_A2",
            f"H_A should be False when only A2 inflates: {perf}",
        )
    if perf["any_bcg_gap"]:
        _fail(
            "verdict_h_a_ignores_A2",
            f"any_bcg_gap should be False; got {perf}",
        )
    _ok("verdict_h_a_ignores_A2")


def test_verdict_h_a_triggers_on_A1() -> None:
    """T5 sanity: H_A does trigger for a BCG-enabled arm."""
    rows = []
    for _ in range(3):
        rows.append(
            {"arm": "A0", "prompt_len": "128", "batch": "1",
             "kernel_count_gdn": "100", "p95_launch_gap_us": "5.0",
             "graph_breaks": "0"}
        )
    for _ in range(3):
        rows.append(
            {"arm": "A1", "prompt_len": "128", "batch": "1",
             "kernel_count_gdn": "500", "p95_launch_gap_us": "5.0",
             "graph_breaks": "0"}
        )
    perf = gv._score_perf(rows)
    if not perf["any_H_A"]:
        _fail("verdict_h_a_triggers_on_A1", f"expected H_A True, got {perf}")
    _ok("verdict_h_a_triggers_on_A1")


def test_verdict_dry_run_command_ok() -> None:
    script = HERE / "gdn_verdict.py"
    rc = subprocess.run(
        [sys.executable, str(script), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("verdict_dry_run_command_ok", rc.stderr.strip() or rc.stdout.strip())
    payload = json.loads(rc.stdout)
    if payload.get("verdict") not in gv.VERDICT_LABELS:
        _fail("verdict_dry_run_command_ok", f"unknown label {payload.get('verdict')}")
    _ok("verdict_dry_run_command_ok", payload["verdict"])


def test_verdict_labels_are_exact_strings() -> None:
    expected = {
        "PASS_BCG_GDN_NOTABLE_GAP",
        "PASS_BCG_GDN_NO_GAP",
        "FAIL_BCG_GDN_CORRECTNESS",
        "PARTIAL_SWEEP",
        "AMBIGUOUS",
        "INFRA_FAILURE",
    }
    if set(gv.VERDICT_LABELS) != expected:
        _fail(
            "verdict_labels_are_exact_strings",
            f"got {set(gv.VERDICT_LABELS)}, expected {expected}",
        )
    _ok("verdict_labels_are_exact_strings")


import extract_nsys_metrics as gextract  # noqa: E402


def test_extract_dry_run_emits_header_and_row() -> None:
    """Dry-run produces the CSV header + one placeholder row without
    needing nsys installed."""
    out = Path("/tmp/gdn_extract_dry.csv")
    if out.exists():
        out.unlink()
    rc = gextract.extract(
        nsys_rep=Path("/nonexistent"),
        arm="A0",
        prompt_len=128,
        batch=1,
        records=None,
        output_csv=out,
        dry_run=True,
    )
    if rc != 0:
        _fail("extract_dry_run_emits_header_and_row", f"rc={rc}")
    if not out.exists():
        _fail("extract_dry_run_emits_header_and_row", "no output file")
    with out.open() as fh:
        rdr = csv.DictReader(fh)
        rows = list(rdr)
        if rdr.fieldnames is None or tuple(rdr.fieldnames) != gextract.COLUMNS:
            _fail(
                "extract_dry_run_emits_header_and_row",
                f"columns={rdr.fieldnames}",
            )
    if not rows:
        _fail("extract_dry_run_emits_header_and_row", "no rows")
    if rows[0].get("extractor_status") != "dry_run":
        _fail(
            "extract_dry_run_emits_header_and_row",
            f"status={rows[0].get('extractor_status')!r}",
        )
    _ok("extract_dry_run_emits_header_and_row")


def test_extract_missing_nsys_rep_reports_missing() -> None:
    """A missing .nsys-rep must not crash — it emits a row with
    extractor_status=MISSING_NSYS_REP so the verdict runner can
    still process the sweep."""
    out = Path("/tmp/gdn_extract_missing.csv")
    if out.exists():
        out.unlink()
    rc = gextract.extract(
        nsys_rep=Path("/nonexistent/foo.nsys-rep"),
        arm="A1",
        prompt_len=128,
        batch=1,
        records=None,
        output_csv=out,
        dry_run=False,
    )
    if rc != 0:
        _fail("extract_missing_nsys_rep_reports_missing", f"rc={rc}")
    with out.open() as fh:
        row = next(csv.DictReader(fh))
    if row["extractor_status"] != "MISSING_NSYS_REP":
        _fail(
            "extract_missing_nsys_rep_reports_missing",
            f"status={row['extractor_status']!r}",
        )
    if "nsys_rep not found" not in row["extractor_warnings"]:
        _fail(
            "extract_missing_nsys_rep_reports_missing",
            f"warnings={row['extractor_warnings']!r}",
        )
    _ok("extract_missing_nsys_rep_reports_missing")


def test_extract_columns_match_validation_plan_section_5() -> None:
    """The column set must match validation_plan.md §5 exactly.

    Regression guard: any downstream change to gdn_verdict.py's CSV
    reader will break if this drifts.
    """
    expected = (
        "arm",
        "prompt_len",
        "batch",
        "request_id",
        "kernel_count_total",
        "kernel_count_gdn",
        "kernel_count_attn",
        "kernel_count_other",
        "cudagraphlaunch_count",
        "cudalaunchkernel_count",
        "ttft_ms",
        "prefill_throughput_toks_per_s",
        "p50_launch_gap_us",
        "p95_launch_gap_us",
        "p99_launch_gap_us",
        "graph_breaks",
        "attribution",
        "nsys_source",
        "extractor_status",
        "extractor_warnings",
    )
    if gextract.COLUMNS != expected:
        _fail("extract_columns_match_validation_plan_section_5", str(gextract.COLUMNS))
    _ok("extract_columns_match_validation_plan_section_5")


def test_extract_parses_kernel_and_api_csv_sections() -> None:
    """Unit-level parse of the nsys stats CSV shapes.

    Covers both the older ``Instances``/``Count`` column names and the
    newer nsys 2026.x cuda_api_sum column ``Num Calls``. Kernel-launch
    APIs summed across cudaLaunchKernel + cudaLaunchKernelExC +
    cuLaunchKernelEx to reflect torch/cuda 13 emitting multiple
    launch APIs in the same run.
    """
    kern_raw = """\
# Generated header
**cuda_gpu_kern_sum**
Time (%),Total Time (ns),Instances,Avg (ns),Min (ns),Max (ns),StdDev (ns),Name
50.0,1000,100,10,5,20,3,gemm_kernel
50.0,1000,80,12,6,22,4,softmax_kernel
"""
    # nsys 2026.x header uses "Num Calls" not "Instances" for cuda_api_sum.
    api_raw = """\
Time (%),Total Time (ns),Num Calls,Avg (ns),Med (ns),Min (ns),Max (ns),StdDev (ns),Name
14.3,1316291719,238500,5519.0,5338.0,1562,4218503,9460.2,cudaLaunchKernelExC
8.0,736402678,104566,7042.5,5156.0,2918,23910012,112772.5,cudaLaunchKernel
0.1,9732301,363,26810.7,23903.0,18058,55591,6567.9,cudaGraphLaunch
0.0,442001689,1000,5000.0,5000.0,2000,10000,500.0,cuLaunchKernelEx
"""
    kern_rows = gextract._parse_csv_section(kern_raw)
    api_rows = gextract._parse_csv_section(api_raw)
    kc = gextract._kernel_counts(kern_rows)
    if kc["kernel_count_total"] != 180:
        _fail("extract_parses_kernel_and_api_csv_sections", f"total={kc}")
    launch, graph = gextract._api_counts(api_rows)
    # 238500 + 104566 + 1000 = 344066 (sum of all launch APIs).
    if launch != 344066:
        _fail(
            "extract_parses_kernel_and_api_csv_sections",
            f"launch_sum expected 344066, got {launch}",
        )
    if graph != 363:
        _fail(
            "extract_parses_kernel_and_api_csv_sections",
            f"cudaGraphLaunch expected 363, got {graph}",
        )
    _ok("extract_parses_kernel_and_api_csv_sections")


def test_extract_row_count_handles_column_variants() -> None:
    if gextract._row_count({"Num Calls": "1234"}) != 1234:
        _fail("extract_row_count_handles_column_variants", "Num Calls")
    if gextract._row_count({"Instances": "500"}) != 500:
        _fail("extract_row_count_handles_column_variants", "Instances")
    if gextract._row_count({"Count": "42"}) != 42:
        _fail("extract_row_count_handles_column_variants", "Count")
    if gextract._row_count({"Other": "999"}) is not None:
        _fail("extract_row_count_handles_column_variants", "Other should be None")
    _ok("extract_row_count_handles_column_variants")


def test_extract_launch_names_include_all_variants() -> None:
    for name in ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernelEx", "cuLaunchKernel"):
        if name not in gextract._LAUNCH_API_NAMES:
            _fail("extract_launch_names_include_all_variants", f"missing {name}")
    _ok("extract_launch_names_include_all_variants")


def test_extract_launch_gap_quantiles() -> None:
    """Launch-gap computation from cuda_api_trace timestamps."""
    trace_rows = [
        {"Start (ns)": "1000", "Name": "cudaLaunchKernel"},
        {"Start (ns)": "2000", "Name": "cudaLaunchKernel"},
        {"Start (ns)": "4000", "Name": "cudaLaunchKernel"},
        {"Start (ns)": "8000", "Name": "cudaLaunchKernel"},
        {"Start (ns)": "16000", "Name": "cudaLaunchKernel"},
    ]
    gaps = gextract._launch_gaps_us(trace_rows)
    # Sorted starts → gaps [1000, 2000, 4000, 8000] ns → [1.0, 2.0, 4.0, 8.0] us
    if gaps != [1.0, 2.0, 4.0, 8.0]:
        _fail("extract_launch_gap_quantiles", f"gaps={gaps}")
    if gextract._quantile(gaps, 0.5) not in (2.0, 4.0):
        _fail("extract_launch_gap_quantiles", f"p50={gextract._quantile(gaps, 0.5)}")
    if gextract._quantile(gaps, 0.95) != 8.0:
        _fail("extract_launch_gap_quantiles", f"p95={gextract._quantile(gaps, 0.95)}")
    _ok("extract_launch_gap_quantiles")


def test_extract_client_metadata_aggregates_e2e() -> None:
    """Client records → ttft_ms (coarse from e2e) + throughput."""
    tmp = Path("/tmp/gdn_extract_records.jsonl")
    tmp.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"e2e_ms": 100.0, "prompt_actual_token_count": 128},
                {"e2e_ms": 120.0, "prompt_actual_token_count": 128},
            ]
        )
        + "\n"
    )
    meta = gextract._client_metadata(tmp)
    if meta.get("e2e_ms_mean") != 110.0:
        _fail("extract_client_metadata_aggregates_e2e", str(meta))
    if meta.get("prompt_actual_token_count_mean") != 128.0:
        _fail("extract_client_metadata_aggregates_e2e", str(meta))
    _ok("extract_client_metadata_aggregates_e2e")


def test_extract_dry_run_cli() -> None:
    """End-to-end CLI dry-run — no nsys required."""
    out = Path("/tmp/gdn_extract_cli.csv")
    if out.exists():
        out.unlink()
    rc = subprocess.run(
        [
            sys.executable,
            str(HERE / "extract_nsys_metrics.py"),
            "--dry-run",
            "--arm",
            "A0",
            "--prompt-len",
            "128",
            "--batch",
            "1",
            "--output-csv",
            str(out),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail("extract_dry_run_cli", f"rc={rc.returncode} stderr={rc.stderr}")
    if not out.exists():
        _fail("extract_dry_run_cli", "no output csv")
    _ok("extract_dry_run_cli")


def test_nsys_capture_requires_separator() -> None:
    script = HERE / "nsys_capture.sh"
    rc = subprocess.run(
        ["bash", str(script), "--gpu-id", "0"],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 64:
        _fail(
            "nsys_capture_requires_separator",
            f"expected exit 64, got {rc.returncode}",
        )
    if "requires '--'" not in rc.stderr:
        _fail(
            "nsys_capture_requires_separator",
            f"expected `--` FATAL; got {rc.stderr!r}",
        )
    _ok("nsys_capture_requires_separator")


def test_nsys_capture_dry_run_prints_plan() -> None:
    script = HERE / "nsys_capture.sh"
    rc = subprocess.run(
        [
            "bash",
            str(script),
            "--dry-run",
            "--",
            "--dry-run",
            "--gpu-id",
            "1",
            "--arm",
            "A1",
            "--prompt-len",
            "128",
            "--batch-size",
            "4",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 0:
        _fail(
            "nsys_capture_dry_run_prints_plan",
            rc.stderr.strip() or rc.stdout.strip(),
        )
    if "would exec:" not in rc.stdout:
        _fail(
            "nsys_capture_dry_run_prints_plan",
            f"expected `would exec:` banner; got {rc.stdout!r}",
        )
    if "output stem would be" not in rc.stdout:
        _fail(
            "nsys_capture_dry_run_prints_plan",
            f"expected output-stem line; got {rc.stdout!r}",
        )
    _ok("nsys_capture_dry_run_prints_plan")


def test_nsys_capture_rejects_gpu_outside_allowlist() -> None:
    script = HERE / "nsys_capture.sh"
    rc = subprocess.run(
        [
            "bash",
            str(script),
            "--",
            "--gpu-id",
            "99",
            "--arm",
            "A1",
            "--prompt-len",
            "128",
            "--batch-size",
            "4",
            "--results-dir",
            "/tmp/nowhere",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if rc.returncode != 64:
        _fail(
            "nsys_capture_rejects_gpu_outside_allowlist",
            f"expected exit 64, got {rc.returncode}. stderr={rc.stderr!r}",
        )
    if "not in allowlist" not in rc.stderr:
        _fail(
            "nsys_capture_rejects_gpu_outside_allowlist",
            f"expected allowlist FATAL; got {rc.stderr!r}",
        )
    _ok("nsys_capture_rejects_gpu_outside_allowlist")


TESTS = (
    test_fixture_regeneration_bit_identical,
    test_fixture_matches_manifest,
    test_fixture_records_shape,
    test_fixture_lengths_hit_targets,
    test_generator_check_command_ok,
    test_generator_print_matches_file,
    test_preflight_dry_run_returns_zero,
    test_preflight_dry_run_reports_skipped,
    test_preflight_required_fields_include_full_attention_interval,
    test_preflight_libcuda_preload_detects_mismatch,
    test_preflight_libcuda_preload_matches_pinned,
    test_preflight_libcuda_preload_accepts_nsys_injected,
    test_preflight_libcuda_preload_missing_is_missing,
    test_preflight_lib_versions_returns_ok,
    test_preflight_nsys_version_is_present,
    test_preflight_frozen_sglang_head_dry_run,
    test_preflight_frozen_sglang_head_live,
    test_preflight_env_flags_recorded,
    test_materialise_stretch_and_truncate,
    test_client_extract_selected_logprobs_from_val_key,
    test_client_extract_selected_logprobs_from_tuple_form,
    test_client_extract_selected_logprobs_missing,
    test_client_extract_top_logprobs,
    test_client_issue_batch_hard_fails_on_partial_response,
    test_client_issue_batch_extracts_all_fields,
    test_client_request_body_sets_return_logprob_and_top_k,
    test_client_probe_tokenizer_returns_counts,
    test_client_probe_tokenizer_returns_single_count,
    test_client_probe_tokenizer_falls_back_on_error,
    test_client_dry_run_probe,
    test_client_materialise_matches_generator,
    test_runner_dry_run_shell_exec,
    test_runner_rejects_missing_gpu_id,
    test_runner_rejects_gpu_outside_allowlist,
    test_runner_dry_run_context_blob_valid_json,
    test_runner_env_exports_precede_preflight,
    test_runner_writes_gpu_post_and_metadata,
    test_runner_mkdir_before_preflight_write,
    test_gdn_instrumentation_install_is_noop,
    test_correctness_gate1_pass_on_identical_arms,
    test_correctness_gate1_fail_on_token_divergence,
    test_correctness_gate1_fail_on_logprob_over_tolerance,
    test_correctness_gate1_requires_A0_reference,
    test_correctness_gate2_pairwise_isolation,
    test_correctness_gate3_pairwise_chunking,
    test_gate1_fails_when_output_ids_missing,
    test_gate1_fails_when_logprobs_missing,
    test_gate1_uses_noise_floor_scaled_tolerance,
    test_gate1_requires_all_repeat_samples_to_match,
    test_resolve_tolerance_uses_floor,
    test_correctness_dry_run_command_ok,
    test_verdict_all_gates_pass_no_perf_gap_is_ambiguous,
    test_verdict_gate_failure_wins_over_perf,
    test_verdict_partial_sweep_when_gate_missing_but_others_pass,
    test_verdict_notable_gap_when_gates_pass_and_h_a,
    test_verdict_no_gap_when_gates_pass_findings_no_hypothesis,
    test_verdict_infra_failure_wins_over_everything,
    test_verdict_h_a_ignores_A2,
    test_verdict_h_a_triggers_on_A1,
    test_verdict_dry_run_command_ok,
    test_verdict_labels_are_exact_strings,
    test_extract_dry_run_emits_header_and_row,
    test_extract_missing_nsys_rep_reports_missing,
    test_extract_columns_match_validation_plan_section_5,
    test_extract_parses_kernel_and_api_csv_sections,
    test_extract_row_count_handles_column_variants,
    test_extract_launch_names_include_all_variants,
    test_extract_launch_gap_quantiles,
    test_extract_client_metadata_aggregates_e2e,
    test_extract_dry_run_cli,
    test_nsys_capture_requires_separator,
    test_nsys_capture_dry_run_prints_plan,
    test_nsys_capture_rejects_gpu_outside_allowlist,
)


def main() -> int:
    for t in TESTS:
        t()
    print(f"[SUMMARY] {len(TESTS)}/{len(TESTS)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
