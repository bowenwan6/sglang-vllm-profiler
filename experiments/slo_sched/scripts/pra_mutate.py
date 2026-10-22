#!/usr/bin/env python3
"""Mutation check for PR-A's unit file: every listed regression must turn it red.

Upstream admits a unit case only if some future diff would make it fail
(.claude/rules/unit-test-admission.md). This applies one regression at a time to the
goodput code in a checkout, runs the unit file the way CI does (one process,
`python <file> -f`) and restores the file afterwards.

  python pra_mutate.py <sglang checkout>      # exit 0 if no mutation survives
"""

import os
import pathlib
import subprocess
import sys

GOODPUT = "python/sglang/benchmark/goodput.py"
SERVING = "python/sglang/benchmark/serving.py"
TEST = "test/registered/unit/bench/test_bench_serving_goodput.py"

# (file, what breaks, text in the file, replacement)
MUTATIONS = [
    (GOODPUT, "good if any SLO is met", "if len(met) == len(goodput_slos):", "if met:"),
    (GOODPUT, "attainment over completed requests", "slo_attainment=num_good / len(outputs),",
     "slo_attainment=num_good / sum(o.success for o in outputs),"),
    (GOODPUT, "tpot divides by output_len", "/ (output.output_len - 1)", "/ output.output_len"),
    (GOODPUT, "no single-token guard", "if output.output_len > 1:", "if True:"),
    (GOODPUT, "failed request is scored",
     "        if not output.success:\n            continue\n", ""),
    (GOODPUT, "unlisted SLOs applied as 0",
     "met = [name for name, slo in goodput_slos.items() if observed_ms[name] <= slo]",
     "met = [name for name in GOODPUT_SLO_METRICS if observed_ms[name] <= goodput_slos.get(name, 0)]"),
    (GOODPUT, "seconds compared with milliseconds", '"ttft": output.ttft * 1000,', '"ttft": output.ttft,'),
    (GOODPUT, "goodput per request, not per second", "request_goodput=num_good / dur_s,",
     "request_goodput=num_good / len(outputs),"),
    (GOODPUT, "per-SLO attainment counts good requests only",
     "        for name in met:\n            num_met[name] += 1\n"
     "        if len(met) == len(goodput_slos):\n            num_good += 1\n",
     "        if len(met) == len(goodput_slos):\n            num_good += 1\n"
     "            for name in met:\n                num_met[name] += 1\n"),
    (SERVING, "unknown key accepted", "    if name not in GOODPUT_SLO_METRICS:\n", "    if False:\n"),
    (SERVING, "value not validated", "return name, _finite_positive_float(value)",
     "return name, float(value)"),
]  # fmt: skip


def main():
    root = pathlib.Path(sys.argv[1]).expanduser()
    test = root / TEST
    originals = {name: (root / name).read_text() for name in (GOODPUT, SERVING)}
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
        for name, what, old, new in MUTATIONS:
            original = originals[name]
            if original.count(old) != 1:
                print(f"NOT APPLICABLE | {what} (text found {original.count(old)} times)")
                survivors.append(what)
                continue
            (root / name).write_text(original.replace(old, new))
            try:
                result = run_test()
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
    restored = run_test().returncode == 0
    print(f"restored files pass: {restored}")
    print(f"MUTATIONS {len(MUTATIONS)}, survivors {len(survivors)}: {survivors or 'none'}")
    return 0 if restored and not survivors else 1


if __name__ == "__main__":
    sys.exit(main())
