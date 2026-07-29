# R6.3 verdict — **PASS**

## Launch context
- `selected_gpu_id`: `2`
- `attempt_dir`: `attempt_gpu2`
- `host_libcuda`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `ld_preload`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `nvidia_driver`: `595.71.05`
- `sglang_stock_head`: `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- `sglang_fork_head`: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`

## R6.3a — Fresh IMG-A rebaseline (720p 1 image, 128->128, c=1, n=400 × 3 reps)

| variant | reps | mean_ttft_ms | median_ttft_ms | CV% | assertions | fallbacks | post_ready_recompiles |
|---|---|---|---|---|---|---|---|
| stock_default | 3/3 | 94.348 | 86.500 | 15.51 | 0 | 0 | 0 |
| fork_pcg | 3/3 | 87.011 | 76.771 | 21.85 | 0 | 0 | 0 |

- fork_pcg - stock_default mean TTFT delta: -7.336 ms (ratio 0.9222)

## R6.3b — Workload sweep (text_tokens × image_res × concurrency)

| cell | stock_default mean_ttft_ms | fork_pcg mean_ttft_ms | fork/stock ratio | winning? | stock_safe | fork_safe |
|---|---|---|---|---|---|---|
| `_server_fork_pcg` | — | — | — | ❌ | True | True |
| `_server_stock_default` | — | — | — | ❌ | True | True |
| `cell_t128_r360p_c1` | 70.875 | 63.375 | 0.8942 | ✅ | True | True |
| `cell_t128_r360p_c4` | 131.557 | 114.948 | 0.8738 | ✅ | True | True |
| `cell_t128_r720p_c1` | 113.070 | 110.793 | 0.9799 | ✅ | True | True |
| `cell_t128_r720p_c4` | 169.013 | 141.065 | 0.8346 | ✅ | True | True |
| `cell_t2048_r360p_c1` | 111.299 | 115.792 | 1.0404 | ❌ | True | True |
| `cell_t2048_r360p_c4` | 243.051 | 261.893 | 1.0775 | ❌ | True | True |
| `cell_t2048_r720p_c1` | 166.549 | 175.201 | 1.0519 | ❌ | True | True |
| `cell_t2048_r720p_c4` | 280.347 | 407.504 | 1.4536 | ❌ | True | True |
| `cell_t512_r360p_c1` | 60.575 | 58.048 | 0.9583 | ✅ | True | True |
| `cell_t512_r360p_c4` | 127.841 | 98.419 | 0.7699 | ✅ | True | True |
| `cell_t512_r720p_c1` | 100.095 | 94.744 | 0.9465 | ✅ | True | True |
| `cell_t512_r720p_c4` | 160.049 | 182.670 | 1.1413 | ❌ | True | True |

**Winning cells** (7): ['cell_t128_r360p_c1', 'cell_t128_r360p_c4', 'cell_t128_r720p_c1', 'cell_t128_r720p_c4', 'cell_t512_r360p_c1', 'cell_t512_r360p_c4', 'cell_t512_r720p_c1']

## R6.3c — Mixed-modality safety (fork-PCG interleaved text+image)
- request_failures: 0
- server assertions: 0
- server fallbacks: 0
- server post_ready_recompiles: 0

## Overall verdict: **PASS**
