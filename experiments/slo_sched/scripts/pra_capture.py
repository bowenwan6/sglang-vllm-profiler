#!/usr/bin/env python3
"""Run bench_serving in-process and save what each request measured.

--output-details keeps ttft and the inter-token intervals of a request but not its
end-to-end latency, and ttft + sum(itl) is not that latency: the benchmark reads the
clock again after later chunks. A recount built from the details can therefore differ
from the benchmark by a request that sits microseconds from a threshold. This wrapper
keeps the per-request records the benchmark itself scores (success, ttft, latency,
output_len) and the duration it divides by, so that a recount can be exact.

  python pra_capture.py <capture.json> <bench_serving arguments ...>
"""

import json
import sys

from sglang.benchmark import serving


def main():
    capture_path = sys.argv[1]
    sys.argv = [sys.argv[0], *sys.argv[2:]]
    captured = {}
    calculate_metrics = serving.calculate_metrics

    def capturing_calculate_metrics(*args, **kwargs):
        captured["dur_s"] = kwargs["dur_s"]
        captured["requests"] = [
            {"success": o.success, "ttft": o.ttft, "latency": o.latency, "output_len": o.output_len}
            for o in kwargs["outputs"]
        ]  # fmt: skip
        return calculate_metrics(*args, **kwargs)

    serving.calculate_metrics = capturing_calculate_metrics
    try:
        serving.cli_main()
    finally:
        with open(capture_path, "w") as f:
            json.dump(captured, f)


if __name__ == "__main__":
    main()
