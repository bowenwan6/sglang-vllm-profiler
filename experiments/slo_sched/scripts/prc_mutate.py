#!/usr/bin/env python3
"""Mutation check for PR-C's unit cases: every listed regression must turn a unit file red.

Upstream admits a unit case only if some future diff would make it fail
(.claude/rules/unit-test-admission.md). This applies one regression at a time to the
goodput code in a checkout, runs the unit file that should notice it the way CI does (one
process, `python <file> -f`) and restores the file afterwards.

  python prc_mutate.py <sglang checkout>      # exit 0 if no mutation survives
"""

import os
import pathlib
import subprocess
import sys

UTILS = "python/sglang/srt/observability/utils.py"
COLLECTOR = "python/sglang/srt/observability/metrics_collector.py"
MANAGER = "python/sglang/srt/managers/tokenizer_manager.py"
HOOK = "python/sglang/srt/arg_groups/validation_hook.py"
T_UTILS = "test/registered/unit/observability/test_metrics_utils.py"
T_COLLECTOR = "test/registered/unit/observability/test_stat_loggers_di.py"
T_MANAGER = "test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py"
T_ARGS = "test/registered/unit/server_args/test_server_args.py"

# (file, unit file that must notice, what breaks, text in the file, replacement)
MUTATIONS = [
    (UTILS, T_UTILS, "good if any SLO is met", "return all(observed_ms", "return any(observed_ms"),
    (UTILS, T_UTILS, "seconds compared with milliseconds", '"ttft": ttft * 1000,', '"ttft": ttft,'),
    (UTILS, T_UTILS, "unlisted SLOs applied as 0",
     "for name, slo_ms in goodput_slo.items())",
     "for name, slo_ms in ((n, goodput_slo.get(n, 0)) for n in GOODPUT_SLO_METRICS))"),
    (COLLECTOR, T_COLLECTOR, "aborted request is scored",
     "if self.goodput_slo and time_to_first_token is not None:", "if self.goodput_slo:"),
    (COLLECTOR, T_COLLECTOR, "scored without --goodput-slo",
     "if self.goodput_slo and time_to_first_token is not None:", "if time_to_first_token is not None:"),
    (COLLECTOR, T_COLLECTOR, "undefined TPOT fails the SLO",
     "tpot=time_per_output_token or 0.0,", 'tpot=time_per_output_token or float("inf"),'),
    (COLLECTOR, T_COLLECTOR, "counter without the is_streaming label",
     "self.num_good_requests_total.labels(**stream_labels).inc(1)",
     "self.num_good_requests_total.labels(**labels).inc(1)"),
    (COLLECTOR, T_COLLECTOR, "TPOT SLO checked against TTFT",
     "tpot=time_per_output_token or 0.0,", "tpot=time_to_first_token,"),
    (MANAGER, T_MANAGER, "TTFT passed for aborts and PD prefill",
     "            time_to_first_token = None\n",
     "            time_to_first_token = state.time_stats.get_first_token_latency()\n"),
    (HOOK, T_ARGS, "unknown key accepted", "assert name in GOODPUT_SLO_METRICS, (", "assert True, ("),
    (HOOK, T_ARGS, "zero accepted", "assert slo_ms > 0,", "assert slo_ms >= 0,"),
    (HOOK, T_ARGS, "validation not called", "    validate_goodput_slo(cfg.goodput_slo)\n", ""),
]  # fmt: skip


def main():
    root = pathlib.Path(sys.argv[1]).expanduser()
    names = sorted({m[0] for m in MUTATIONS})
    tests = sorted({m[1] for m in MUTATIONS})
    originals = {name: (root / name).read_text() for name in names}
    env = dict(os.environ, PYTHONPATH=str(root / "python"))

    def run_test(test):
        return subprocess.run(
            [sys.executable, "-W", "ignore", str(root / test), "-f"],
            env=env, capture_output=True, text=True, cwd=root,
        )  # fmt: skip

    for test in tests:
        if run_test(test).returncode != 0:
            print(f"{test} does not pass before any mutation")
            return 2
    survivors = []
    try:
        for name, test, what, old, new in MUTATIONS:
            original = originals[name]
            if original.count(old) != 1:
                print(f"NOT APPLICABLE | {what} (text found {original.count(old)} times)")
                survivors.append(what)
                continue
            (root / name).write_text(original.replace(old, new))
            try:
                result = run_test(test)
            finally:
                (root / name).write_text(original)
            hits = [line.split(" (")[0] for line in result.stderr.splitlines()
                    if line.startswith(("FAIL:", "ERROR:"))]  # fmt: skip
            red = result.returncode != 0
            print(f"{'red  ' if red else 'GREEN'} | {what} | {'; '.join(sorted(set(hits)))}", flush=True)
            if not red:
                survivors.append(what)
    finally:
        for name, original in originals.items():
            (root / name).write_text(original)
    restored = all(run_test(test).returncode == 0 for test in tests)
    print(f"restored files pass: {restored}")
    print(f"MUTATIONS {len(MUTATIONS)}, survivors {len(survivors)}: {survivors or 'none'}")
    return 0 if restored and not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
