"""Cost of the goodput check per finished request: observe_one_finished_request with and without SLOs."""
import json
import statistics
import time

from sglang.srt.observability.metrics_collector import TokenizerMetricsCollector
from sglang.srt.runtime_context import get_context

LABELS = {"model_name": "m", "engine_type": "unified"}
N = 200_000


class A(TokenizerMetricsCollector):
    pass


def build(goodput_slo):
    import prometheus_client

    for c in list(prometheus_client.REGISTRY._collector_to_names):
        prometheus_client.REGISTRY.unregister(c)
    with get_context().override_server_args(prompt_tokens_buckets=None, generation_tokens_buckets=None):
        return TokenizerMetricsCollector(labels=LABELS, goodput_slo=goodput_slo)


def bench(goodput_slo):
    c = build(goodput_slo)
    t = time.perf_counter()
    for _ in range(N):
        c.observe_one_finished_request(
            LABELS, 200, 100, 0, 2.5, False,
            is_streaming=True, time_per_output_token=0.02, time_to_first_token=0.2,
        )
    return (time.perf_counter() - t) / N * 1e6


rows = {"none": [], "three": []}
for _ in range(5):
    rows["none"].append(bench(None))
    rows["three"].append(bench({"ttft": 500, "tpot": 50, "e2el": 3000}))
out = {k: {"median_us": round(statistics.median(v), 3), "runs_us": [round(x, 3) for x in v]} for k, v in rows.items()}
out["delta_us"] = round(out["three"]["median_us"] - out["none"]["median_us"], 3)
print(json.dumps(out))
