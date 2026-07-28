# R6.1 Amendment B verdict — **SAFETY_SUPERIORITY_PASS**

> Evaluated under [`../protocol_amendment_B_repeated_shape_safety.md`](../protocol_amendment_B_repeated_shape_safety.md). Rules pre-declared before Attempt 04 ran.

## Launch context

- `selected_gpu_id`: `0`
- `attempt_dir`: `attempt_04_repeated_shape_gpu0`
- `host_libcuda`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `ld_preload`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `cuda_visible_devices`: `0`
- `nvidia_driver`: `595.71.05`
- `sglang_stock_head`: `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- `sglang_fork_head`: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`

## stock-PCG side

- server_log: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_04_repeated_shape_gpu0/raw/stock/server.log`
- server_log_total_lines: 44328
- server_ready_line: 35424
- assertion count: **1** (lines [44322])
- deepstack-recompile trigger lines: [172, 8914, 17647, 26674, 26679, 26850, 26851, 35435, 35440, 35467, 35621, 35622, 44151]
- startup/warmup recompiles: 3 (lines [8752, 17476, 26671])
- post-ready recompiles: 1 (lines [35429])
- fallback lines: 0 ([])
- prefill batches captured: 5
- unique prefill shapes: ['total=1023,new_seq=1', 'total=21439,new_seq=21', 'total=78,new_seq=1', 'total=8228,new_seq=9']
- per-shape occurrence counts: {'total=78,new_seq=1': 1, 'total=1023,new_seq=1': 2, 'total=8228,new_seq=9': 1, 'total=21439,new_seq=21': 1}
- assertion contexts:
  - assertion at line 44322; last prefill batch before: {'line': 44229, 'new_seq': 1, 'new_token': 1, 'cached_token': 1022}
- bench.jsonl aggregate_completed: None / aggregate_num_prompts: None / generated_texts count: None

## fork-PCG side

- server_log: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_04_repeated_shape_gpu0/raw/fork/server.log`
- server_log_total_lines: 44594
- server_ready_line: 44379
- assertion count: **0** (lines [])
- deepstack-recompile trigger lines: [172, 8914, 17647, 26294, 26299, 26302, 26307, 26475, 26476, 35055, 35060, 35087, 35088, 35244, 35245, 43774]
- startup/warmup recompiles: 4 (lines [8752, 17476, 26291, 35049])
- post-ready recompiles: 0 (lines [])
- fallback lines: 0 ([])
- prefill batches captured: 39
- unique prefill shapes: ['total=1020,new_seq=1', 'total=1021,new_seq=1', 'total=1022,new_seq=1', 'total=1023,new_seq=1', 'total=1024,new_seq=1', 'total=1025,new_seq=1', 'total=1026,new_seq=1', 'total=1027,new_seq=1', 'total=1028,new_seq=1', 'total=1029,new_seq=1', 'total=12276,new_seq=12', 'total=2046,new_seq=2', 'total=5115,new_seq=5', 'total=78,new_seq=1', 'total=9207,new_seq=9']
- per-shape occurrence counts: {'total=78,new_seq=1': 1, 'total=1023,new_seq=1': 8, 'total=2046,new_seq=2': 1, 'total=5115,new_seq=5': 1, 'total=9207,new_seq=9': 1, 'total=12276,new_seq=12': 1, 'total=1021,new_seq=1': 4, 'total=1024,new_seq=1': 2, 'total=1027,new_seq=1': 4, 'total=1029,new_seq=1': 4, 'total=1028,new_seq=1': 1, 'total=1022,new_seq=1': 2, 'total=1026,new_seq=1': 2, 'total=1025,new_seq=1': 4, 'total=1020,new_seq=1': 3}
- bench.jsonl aggregate_completed: 32 / aggregate_num_prompts: None / generated_texts count: 32

## Overall verdict: **SAFETY_SUPERIORITY_PASS**


