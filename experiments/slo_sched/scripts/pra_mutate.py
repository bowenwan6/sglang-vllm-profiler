#!/usr/bin/env python3
"""Mutation check for PR-A's unit file: every listed regression must turn it red.

Upstream admits a unit case only if some future diff would make it fail
(.claude/rules/unit-test-admission.md). This applies one regression at a time to
python/sglang/benchmark/goodput.py in a checkout, runs the unit file the way CI does
(one process, `python <file> -f`) and restores the file afterwards.

  python pra_mutate.py <sglang checkout>      # exit 0 if no mutation survives
"""

import os
import pathlib
import subprocess
import sys

TARGET = "python/sglang/benchmark/goodput.py"
TEST = "test/registered/unit/bench/test_bench_serving_goodput.py"

# (what breaks, text in goodput.py, replacement)
MUTATIONS = [
    ("good if any SLO is met", "sum(all(result.values()) for result in slo_results)",
     "sum(any(result.values()) for result in slo_results)"),
    ("attainment over completed requests", "num_requests = max(len(outputs), 1)",
     "num_requests = max(sum(o.success for o in outputs), 1)"),
    ("tpot divides by output_len", "/ (output.output_len - 1)", "/ output.output_len"),
    ("no single-token guard", "        if output.output_len > 1\n        else 0.0\n",
     "        if True\n        else 0.0\n"),
    ("failed request is scored",
     "    if not output.success:\n        return {name: False for name in goodput_slos}\n", ""),
    ("unlisted SLOs applied as 0",
     "return {name: observed_ms[name] <= slo for name, slo in goodput_slos.items()}",
     "return {name: observed_ms[name] <= goodput_slos.get(name, 0) for name in GOODPUT_SLO_METRICS}"),
    ("seconds compared with milliseconds", '"ttft": output.ttft * 1000,', '"ttft": output.ttft,'),
    ("goodput per request, not per second", "request_goodput=num_good / dur_s,",
     "request_goodput=num_good / num_requests,"),
    ("int() instead of float()", "slo_ms = float(value)", "slo_ms = float(int(value))"),
    ("unknown key accepted", "if not sep or name not in GOODPUT_SLO_METRICS:", "if not sep:"),
    ("non-numeric value skipped",
     "            raise ValueError(\n"
     '                f"Invalid --goodput value in {pair!r}: expected milliseconds."\n'
     "            ) from None\n",
     "            continue\n"),
    ("duplicate accepted",
     '        if name in slos:\n            raise ValueError(f"Duplicate --goodput metric {name!r}.")\n',
     ""),
    ("non-positive accepted", "if not math.isfinite(slo_ms) or slo_ms <= 0:",
     "if not math.isfinite(slo_ms):"),
    ("non-finite accepted", "if not math.isfinite(slo_ms) or slo_ms <= 0:", "if slo_ms <= 0:"),
    ("no flag raises", "    if not slo_pairs:\n        return None\n",
     "    if slo_pairs == []:\n        return None\n"),
]  # fmt: skip


def main():
    root = pathlib.Path(sys.argv[1]).expanduser()
    target, test = root / TARGET, root / TEST
    original = target.read_text()
    env = dict(os.environ, PYTHONPATH=str(root / "python"))

    def run_test():
        return subprocess.run(
            [sys.executable, "-W", "ignore", str(test), "-f"],
            env=env, capture_output=True, text=True, cwd=root,
        )  # fmt: skip

    if run_test().returncode != 0:
        print("the unit file does not pass before any mutation")
        return 2
    survivors = []
    try:
        for name, old, new in MUTATIONS:
            if original.count(old) != 1:
                print(f"NOT APPLICABLE | {name} (text found {original.count(old)} times)")
                survivors.append(name)
                continue
            target.write_text(original.replace(old, new))
            result = run_test()
            hits = [line.split(" (")[0] for line in result.stderr.splitlines()
                    if line.startswith(("FAIL:", "ERROR:"))]  # fmt: skip
            red = result.returncode != 0
            print(f"{'red  ' if red else 'GREEN'} | {name} | {'; '.join(sorted(set(hits)))}", flush=True)
            if not red:
                survivors.append(name)
    finally:
        target.write_text(original)
    restored = run_test().returncode == 0
    print(f"restored file passes: {restored}")
    print(f"MUTATIONS {len(MUTATIONS)}, survivors {len(survivors)}: {survivors or 'none'}")
    return 0 if restored and not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
