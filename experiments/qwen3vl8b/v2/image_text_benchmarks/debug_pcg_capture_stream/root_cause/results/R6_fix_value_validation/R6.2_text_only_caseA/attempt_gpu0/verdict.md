# R6.2 verdict — **FAIL**

## Launch context
- `selected_gpu_id`: `0`
- `attempt_dir`: `attempt_gpu0`
- `host_libcuda`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `ld_preload`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `cuda_visible_devices`: `0`
- `nvidia_driver`: `595.71.05`
- `sglang_stock_head`: `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- `sglang_fork_head`: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`

## Predeclared gates
- fork_pcg mean TTFT / stock_pcg mean TTFT: 0.9617 (require ≤ 1.05)
- drift stock_default vs stock_default_repeat: 3.050% (require ≤ 3.0%)
- per-variant CV ≤ 6% and safety zeros

## Per-variant summary

| variant | reps completed | mean_ttft_ms | median_ttft_ms | CV% | mean_tpot_ms | output_throughput | assertions | fallbacks | post_ready_recompiles |
|---|---|---|---|---|---|---|---|---|---|
| stock_default | 5/5 | 26.862 | 26.870 | 5.91 | 5.132 | 188.449 | 0 | 0 | 0 |
| stock_pcg | 5/5 | 18.353 | 18.292 | 2.29 | 5.129 | 190.948 | 0 | 0 | 0 |
| fork_pcg | 5/5 | 17.650 | 17.530 | 2.02 | 5.134 | 190.980 | 0 | 0 | 0 |
| stock_default_repeat | 5/5 | 27.681 | 27.473 | 2.51 | 5.128 | 188.357 | 0 | 0 | 0 |

## Per-rep detail

### stock_default
| rep | completed | mean_ttft_ms | median_ttft_ms | mean_tpot_ms | output_throughput |
|---|---|---|---|---|---|
| 1 | 400 | 26.870 | 26.674 | 5.130 | 188.496 |
| 2 | 400 | 25.342 | 25.124 | 5.135 | 188.791 |
| 3 | 400 | 25.378 | 25.197 | 5.132 | 188.895 |
| 4 | 400 | 27.625 | 26.796 | 5.132 | 188.259 |
| 5 | 400 | 29.095 | 27.732 | 5.132 | 187.807 |

### stock_pcg
| rep | completed | mean_ttft_ms | median_ttft_ms | mean_tpot_ms | output_throughput |
|---|---|---|---|---|---|
| 1 | 400 | 19.027 | 17.785 | 5.131 | 190.682 |
| 2 | 400 | 18.429 | 17.157 | 5.132 | 190.791 |
| 3 | 400 | 17.973 | 16.797 | 5.125 | 191.232 |
| 4 | 400 | 18.041 | 17.092 | 5.130 | 191.000 |
| 5 | 400 | 18.292 | 17.217 | 5.127 | 191.033 |

### fork_pcg
| rep | completed | mean_ttft_ms | median_ttft_ms | mean_tpot_ms | output_throughput |
|---|---|---|---|---|---|
| 1 | 400 | 18.277 | 17.057 | 5.134 | 190.779 |
| 2 | 400 | 17.530 | 16.646 | 5.134 | 191.001 |
| 3 | 400 | 17.383 | 16.456 | 5.135 | 190.997 |
| 4 | 400 | 17.490 | 16.617 | 5.134 | 190.989 |
| 5 | 400 | 17.572 | 16.442 | 5.131 | 191.134 |

### stock_default_repeat
| rep | completed | mean_ttft_ms | median_ttft_ms | mean_tpot_ms | output_throughput |
|---|---|---|---|---|---|
| 1 | 400 | 28.871 | 27.781 | 5.129 | 187.967 |
| 2 | 400 | 27.056 | 26.214 | 5.127 | 188.537 |
| 3 | 400 | 27.400 | 26.084 | 5.129 | 188.436 |
| 4 | 400 | 27.607 | 26.672 | 5.129 | 188.323 |
| 5 | 400 | 27.473 | 26.424 | 5.126 | 188.524 |

## Failure reasons
- drift stock_default vs repeat 3.05% > 3%

## Overall verdict: **FAIL**
