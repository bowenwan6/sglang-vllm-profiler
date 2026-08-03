#!/usr/bin/env python3
"""CPU-only smoke tests for the GDN sub-track's scaffolding.

Run with:

    python3 experiments/qwen35_4b/gdn/scripts/test_gdn_scaffolding.py

Prints one line per test. Exits non-zero on the first failure. No torch,
no network, no GPU. The point is to catch obvious breakage before any
runner is invoked against a real server.
"""

from __future__ import annotations

import hashlib
import io
import json
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
    if statuses.get("hf_model_metadata") != "SKIPPED_DRY_RUN":
        _fail(
            "preflight_dry_run_reports_skipped",
            f"expected SKIPPED_DRY_RUN for hf_model_metadata, got {statuses}",
        )
    if statuses.get("gdn_config_fields") != "SKIPPED_DRY_RUN":
        _fail(
            "preflight_dry_run_reports_skipped",
            f"expected SKIPPED_DRY_RUN for gdn_config_fields, got {statuses}",
        )
    _ok("preflight_dry_run_reports_skipped")


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


TESTS = (
    test_fixture_regeneration_bit_identical,
    test_fixture_matches_manifest,
    test_fixture_records_shape,
    test_fixture_lengths_hit_targets,
    test_generator_check_command_ok,
    test_generator_print_matches_file,
    test_preflight_dry_run_returns_zero,
    test_preflight_dry_run_reports_skipped,
    test_preflight_env_flags_recorded,
    test_materialise_stretch_and_truncate,
)


def main() -> int:
    for t in TESTS:
        t()
    print(f"[SUMMARY] {len(TESTS)}/{len(TESTS)} passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
