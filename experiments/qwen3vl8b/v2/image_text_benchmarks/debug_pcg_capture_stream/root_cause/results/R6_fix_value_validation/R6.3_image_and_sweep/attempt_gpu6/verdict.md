# R6.3 verdict — **FAIL**

## Launch context
- `selected_gpu_id`: `6`
- `attempt_dir`: `attempt_gpu6`
- `host_libcuda`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `ld_preload`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `nvidia_driver`: `595.71.05`
- `sglang_stock_head`: `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- `sglang_fork_head`: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`

## R6.3a — Fresh IMG-A rebaseline (720p 1 image, 128->128, c=1, n=400 × 3 reps)

| variant | reps | mean_ttft_ms | median_ttft_ms | CV% | assertions | fallbacks | post_ready_recompiles |
|---|---|---|---|---|---|---|---|
| stock_default | 3/3 | 98.381 | 92.321 | 13.24 | 0 | 0 | 0 |
| fork_pcg | 3/3 | 93.230 | 84.287 | 22.44 | 0 | 0 | 0 |

- fork_pcg - stock_default mean TTFT delta: -5.151 ms (ratio 0.9476)

## R6.3b — Workload sweep (text_tokens × image_res × concurrency)

| cell | stock_default mean_ttft_ms | fork_pcg mean_ttft_ms | fork/stock ratio | winning? | stock_safe | fork_safe |
|---|---|---|---|---|---|---|
| `_server_fork_pcg` | — | — | — | ❌ | True | True |
| `_server_stock_default` | — | — | — | ❌ | True | True |
| `cell_t128_r224p_c1` | — | — | — | ❌ | True | True |
| `cell_t128_r224p_c4` | — | — | — | ❌ | True | True |
| `cell_t128_r720p_c1` | 107.082 | 109.925 | 1.0265 | ❌ | True | True |
| `cell_t128_r720p_c4` | 183.497 | 182.591 | 0.9951 | ✅ | True | True |
| `cell_t2048_r224p_c1` | — | — | — | ❌ | True | True |
| `cell_t2048_r224p_c4` | — | — | — | ❌ | True | True |
| `cell_t2048_r720p_c1` | 143.162 | 147.960 | 1.0335 | ❌ | True | True |
| `cell_t2048_r720p_c4` | 379.357 | 402.617 | 1.0613 | ❌ | True | True |
| `cell_t512_r224p_c1` | — | — | — | ❌ | True | True |
| `cell_t512_r224p_c4` | — | — | — | ❌ | True | True |
| `cell_t512_r720p_c1` | 86.339 | 89.561 | 1.0373 | ❌ | True | True |
| `cell_t512_r720p_c4` | 181.545 | 179.402 | 0.9882 | ✅ | True | True |

**Winning cells** (2): ['cell_t128_r720p_c4', 'cell_t512_r720p_c4']

## R6.3c — Mixed-modality safety (fork-PCG interleaved text+image)
- request_failures: 100
- server assertions: 0
- server fallbacks: 0
- server post_ready_recompiles: 0

## Overall verdict: **FAIL**
- R6.3c mixed safety failed: {'missing': False, 'server': {'total_lines': 0, 'ready_line': None, 'assertions': 0, 'fallbacks': 0, 'startup_recompiles': 0, 'post_ready_recompiles': 0}, 'client': {'completed': 0, 'ended_utc': '2026-07-29T07:09:21+00:00', 'fixture_sha256': '79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967', 'n_image': 50, 'n_text': 50, 'n_total': 100, 'request_failures': 100, 'started_utc': '2026-07-29T07:09:21+00:00'}, 'request_failures': 100, 'completed': 0}
