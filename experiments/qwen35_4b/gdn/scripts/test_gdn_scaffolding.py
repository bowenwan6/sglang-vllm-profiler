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


import gdn_client  # noqa: E402


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
    rc = subprocess.run(
        ["bash", str(runner), "--gpu-id", "3"],
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
    if not any(
        flag == "--disable-cuda-graph" for flag in payload.get("arm_flags", [])
    ):
        _fail(
            "runner_dry_run_context_blob_valid_json",
            f"A1 arm_flags should include --disable-cuda-graph; got "
            f"{payload.get('arm_flags')!r}",
        )
    _ok("runner_dry_run_context_blob_valid_json", ctx_path)


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
        "alone": [{"prompt_source_id": "p0", "output_ids": [1, 2, 3]}],
        "batched": [{"prompt_source_id": "p0", "output_ids": [1, 2, 3]}],
    }
    v = gc.gate_verdict(2, labelled, tolerance=0.05)
    if v["overall"] != "PASS":
        _fail("correctness_gate2_pairwise_isolation", json.dumps(v))
    _ok("correctness_gate2_pairwise_isolation")


def test_correctness_gate3_pairwise_chunking() -> None:
    labelled = {
        "small_chunk": [{"prompt_source_id": "p0", "output_ids": [4, 5]}],
        "single_chunk": [{"prompt_source_id": "p0", "output_ids": [4, 5]}],
    }
    v = gc.gate_verdict(3, labelled, tolerance=0.05)
    if v["overall"] != "PASS":
        _fail("correctness_gate3_pairwise_chunking", json.dumps(v))
    _ok("correctness_gate3_pairwise_chunking")


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


def test_verdict_all_gates_pass_no_perf_gap_is_ambiguous() -> None:
    # Without perf findings, the verdict should be AMBIGUOUS (perf side
    # has no signal to declare either way).
    gate_summary = {
        "per_gate": {i: {"overall": "PASS"} for i in (1, 2, 3, 4)},
        "all_pass": True,
    }
    perf_summary = {
        "findings": [],
        "any_H_A": False,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": False,
    }
    v = gv.decide(gate_summary, perf_summary, infra_failure=False)
    if v != "AMBIGUOUS":
        _fail("verdict_all_gates_pass_no_perf_gap_is_ambiguous", v)
    _ok("verdict_all_gates_pass_no_perf_gap_is_ambiguous")


def test_verdict_gate_failure_wins_over_perf() -> None:
    gate_summary = {
        "per_gate": {1: {"overall": "PASS"}, 2: {"overall": "FAIL"}, 3: {"overall": "PASS"}, 4: {"overall": "PASS"}},
        "all_pass": False,
    }
    perf_summary = {
        "findings": [{"any": "thing"}],
        "any_H_A": True,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": True,
    }
    v = gv.decide(gate_summary, perf_summary, infra_failure=False)
    if v != "FAIL_BCG_GDN_CORRECTNESS":
        _fail("verdict_gate_failure_wins_over_perf", v)
    _ok("verdict_gate_failure_wins_over_perf")


def test_verdict_notable_gap_when_gates_pass_and_h_a() -> None:
    gate_summary = {
        "per_gate": {i: {"overall": "PASS"} for i in (1, 2, 3, 4)},
        "all_pass": True,
    }
    perf_summary = {
        "findings": [{"any": "thing"}],
        "any_H_A": True,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": True,
    }
    v = gv.decide(gate_summary, perf_summary, infra_failure=False)
    if v != "PASS_BCG_GDN_NOTABLE_GAP":
        _fail("verdict_notable_gap_when_gates_pass_and_h_a", v)
    _ok("verdict_notable_gap_when_gates_pass_and_h_a")


def test_verdict_no_gap_when_gates_pass_findings_no_hypothesis() -> None:
    gate_summary = {
        "per_gate": {i: {"overall": "PASS"} for i in (1, 2, 3, 4)},
        "all_pass": True,
    }
    perf_summary = {
        "findings": [{"any": "thing"}],
        "any_H_A": False,
        "any_H_B": False,
        "any_H_C": False,
        "any_bcg_gap": False,
    }
    v = gv.decide(gate_summary, perf_summary, infra_failure=False)
    if v != "PASS_BCG_GDN_NO_GAP":
        _fail("verdict_no_gap_when_gates_pass_findings_no_hypothesis", v)
    _ok("verdict_no_gap_when_gates_pass_findings_no_hypothesis")


def test_verdict_infra_failure_wins_over_everything() -> None:
    gate_summary = {
        "per_gate": {i: {"overall": "PASS"} for i in (1, 2, 3, 4)},
        "all_pass": True,
    }
    perf_summary = {
        "findings": [],
        "any_H_A": True,
        "any_H_B": True,
        "any_H_C": True,
        "any_bcg_gap": True,
    }
    v = gv.decide(gate_summary, perf_summary, infra_failure=True)
    if v != "INFRA_FAILURE":
        _fail("verdict_infra_failure_wins_over_everything", v)
    _ok("verdict_infra_failure_wins_over_everything")


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
        "AMBIGUOUS",
        "INFRA_FAILURE",
    }
    if set(gv.VERDICT_LABELS) != expected:
        _fail(
            "verdict_labels_are_exact_strings",
            f"got {set(gv.VERDICT_LABELS)}, expected {expected}",
        )
    _ok("verdict_labels_are_exact_strings")


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
            "3",
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
    test_preflight_env_flags_recorded,
    test_materialise_stretch_and_truncate,
    test_client_dry_run_probe,
    test_client_materialise_matches_generator,
    test_runner_dry_run_shell_exec,
    test_runner_rejects_missing_gpu_id,
    test_runner_rejects_gpu_outside_allowlist,
    test_runner_dry_run_context_blob_valid_json,
    test_gdn_instrumentation_install_is_noop,
    test_correctness_gate1_pass_on_identical_arms,
    test_correctness_gate1_fail_on_token_divergence,
    test_correctness_gate1_fail_on_logprob_over_tolerance,
    test_correctness_gate1_requires_A0_reference,
    test_correctness_gate2_pairwise_isolation,
    test_correctness_gate3_pairwise_chunking,
    test_correctness_dry_run_command_ok,
    test_verdict_all_gates_pass_no_perf_gap_is_ambiguous,
    test_verdict_gate_failure_wins_over_perf,
    test_verdict_notable_gap_when_gates_pass_and_h_a,
    test_verdict_no_gap_when_gates_pass_findings_no_hypothesis,
    test_verdict_infra_failure_wins_over_everything,
    test_verdict_dry_run_command_ok,
    test_verdict_labels_are_exact_strings,
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
