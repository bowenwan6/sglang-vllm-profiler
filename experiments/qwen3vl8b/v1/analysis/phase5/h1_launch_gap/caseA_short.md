# Phase 5.1 — H1 launch/graph-coverage metrics · caseA_short

Read-only offline analysis of existing graph-on formal (SGLang) + vLLM traces. SGLang rows are from FORMAL traces only (not graph-off mapping).

| window | kernels | launch ops | GPU µs | graph % | eager % | uncl % | GEMM % | GPU idle % | idle µs | launch CPU µs |
|---|---|---|---|---|---|---|---|---|---|---|
| sglang_formal_DECODE | 580 | 566 | 6337.2 | 0.0 | 99.7 | 0.3 | 74.6 | 95.3 | 129097.2 | 4702.0 |
| sglang_formal_EXTEND | 5870 | 5630 | 49345.8 | 0.0 | 99.4 | 0.6 | 83.5 | 96.2 | 1236177.6 | 44156.5 |
| vllm_prefill_like | 4328 | 1320 | 49137.7 | 81.7 | 18.0 | 0.3 | 77.3 | 65.0 | 89948.8 | 11848.4 |
| vllm_decode_like | 640230 | 23398 | 5697403.4 | 93.1 | 6.5 | 0.4 | 81.1 | 7.8 | 480341.9 | 648496.1 |


## Notes / limits

- graph/eager classified by the kernel's correlated CPU launch op (cudaGraphLaunch=graph, cudaLaunchKernel*=eager, no correlation=unclassified).
- idle gap is on the dominant GPU stream (union-of-intervals); cross-stream overlap not modeled.
- per-forward-step critical-path segmentation NOT computed (num_steps>1 windows are not cleanly separable from the trace alone) -> reported as window-level span/busy/idle, and per-step critical path is AMBIGUOUS/unavailable.
- launch_op_total_cpu_us is the summed CPU duration of launch runtime ops (a proxy); true critical-path CPU launch-gap across threads is AMBIGUOUS and not asserted.
