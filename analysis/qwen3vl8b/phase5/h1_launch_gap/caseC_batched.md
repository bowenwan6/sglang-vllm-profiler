# Phase 5.1 — H1 launch/graph-coverage metrics · caseC_batched

Read-only offline analysis of existing graph-on formal (SGLang) + vLLM traces. SGLang rows are from FORMAL traces only (not graph-off mapping).

| window | kernels | launch ops | GPU µs | graph % | eager % | uncl % | GEMM % | GPU idle % | idle µs | launch CPU µs |
|---|---|---|---|---|---|---|---|---|---|---|
| sglang_formal_DECODE | 1216 | 997 | 186589.3 | 0.0 | 99.9 | 0.1 | 85.6 | 40.6 | 127218.5 | 8441.7 |
| sglang_formal_EXTEND | 8732 | 5578 | 78243.4 | 35.7 | 62.1 | 2.2 | 75.0 | 94.1 | 1202334.5 | 46556.6 |
| vllm_prefill_like | 4320 | 3904 | 101081.2 | 0.0 | 99.6 | 0.4 | 83.3 | 68.6 | 220700.8 | 19008.0 |
| vllm_decode_like | 485351 | 26302 | 5854682.4 | 72.7 | 26.7 | 0.6 | 77.9 | 6.3 | 395598.0 | 541643.6 |


## Notes / limits

- graph/eager classified by the kernel's correlated CPU launch op (cudaGraphLaunch=graph, cudaLaunchKernel*=eager, no correlation=unclassified).
- idle gap is on the dominant GPU stream (union-of-intervals); cross-stream overlap not modeled.
- per-forward-step critical-path segmentation NOT computed (num_steps>1 windows are not cleanly separable from the trace alone) -> reported as window-level span/busy/idle, and per-step critical path is AMBIGUOUS/unavailable.
- launch_op_total_cpu_us is the summed CPU duration of launch runtime ops (a proxy); true critical-path CPU launch-gap across threads is AMBIGUOUS and not asserted.
