# R6.3 confirmation verdict — **PASS**

## Launch context
- `selected_gpu_id`: `2`
- `attempt_dir`: `attempt_gpu2_confirm`
- `cells`: `['cell_t128_r360p_c1', 'cell_t128_r360p_c4', 'cell_t128_r720p_c1', 'cell_t128_r720p_c4', 'cell_t512_r360p_c1', 'cell_t512_r360p_c4', 'cell_t512_r720p_c1']`
- `reps_per_cell_per_variant`: `3`
- `host_libcuda`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `ld_preload`: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- `nvidia_driver`: `595.71.05`
- `sglang_stock_head`: `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- `sglang_fork_head`: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`

## Per-cell confirmation

| cell | stock reps_ok/inv | fork reps_ok/inv | stock mean_ttft_ms | fork mean_ttft_ms | fork/stock | verdict |
|---|---|---|---|---|---|---|
| `cell_t128_r360p_c1` | 3/0 | 3/0 | 64.228 | 63.478 | 0.9883 | CONFIRMED_WIN |
| `cell_t128_r360p_c4` | 3/0 | 3/0 | 108.275 | 88.525 | 0.8176 | CONFIRMED_WIN |
| `cell_t128_r720p_c1` | 3/0 | 3/0 | 103.115 | 93.834 | 0.9100 | CONFIRMED_WIN |
| `cell_t128_r720p_c4` | 3/0 | 3/0 | 188.878 | 200.496 | 1.0615 | NOT_CONFIRMED |
| `cell_t512_r360p_c1` | 3/0 | 3/0 | 65.136 | 59.524 | 0.9138 | CONFIRMED_WIN |
| `cell_t512_r360p_c4` | 3/0 | 3/0 | 166.333 | 129.837 | 0.7806 | CONFIRMED_WIN |
| `cell_t512_r720p_c1` | 3/0 | 3/0 | 98.256 | 114.861 | 1.1690 | NOT_CONFIRMED |

## Per-rep detail (mean TTFT ms)

### `cell_t128_r360p_c1` — verdict **CONFIRMED_WIN**
- **stock_default**: rep1=71.489, rep2=56.746, rep3=64.450 | mean=64.228 CV%=11.48
- **fork_pcg**: rep1=65.026, rep2=43.352, rep3=82.055 | mean=63.478 CV%=30.56

### `cell_t128_r360p_c4` — verdict **CONFIRMED_WIN**
- **stock_default**: rep1=98.492, rep2=99.312, rep3=127.021 | mean=108.275 CV%=15.00
- **fork_pcg**: rep1=82.793, rep2=83.671, rep3=99.112 | mean=88.525 CV%=10.37

### `cell_t128_r720p_c1` — verdict **CONFIRMED_WIN**
- **stock_default**: rep1=115.426, rep2=105.330, rep3=88.588 | mean=103.115 CV%=13.15
- **fork_pcg**: rep1=118.763, rep2=83.002, rep3=79.736 | mean=93.834 CV%=23.07

### `cell_t128_r720p_c4` — verdict **NOT_CONFIRMED**
- **stock_default**: rep1=199.845, rep2=180.841, rep3=185.948 | mean=188.878 CV%=5.21
- **fork_pcg**: rep1=178.895, rep2=211.222, rep3=211.371 | mean=200.496 CV%=9.33

### `cell_t512_r360p_c1` — verdict **CONFIRMED_WIN**
- **stock_default**: rep1=63.928, rep2=68.130, rep3=63.349 | mean=65.136 CV%=4.01
- **fork_pcg**: rep1=58.216, rep2=57.657, rep3=62.699 | mean=59.524 CV%=4.64

### `cell_t512_r360p_c4` — verdict **CONFIRMED_WIN**
- **stock_default**: rep1=119.052, rep2=171.256, rep3=208.690 | mean=166.333 CV%=27.07
- **fork_pcg**: rep1=109.962, rep2=100.633, rep3=178.916 | mean=129.837 CV%=32.93

### `cell_t512_r720p_c1` — verdict **NOT_CONFIRMED**
- **stock_default**: rep1=101.972, rep2=98.519, rep3=94.278 | mean=98.256 CV%=3.92
- **fork_pcg**: rep1=95.831, rep2=151.244, rep3=97.508 | mean=114.861 CV%=27.44

## Server safety (shared across cells per variant)
- **stock_default**: assertions=0, fallbacks=0, post_ready_recompiles=0, ready_line=99, total_lines=8998
- **fork_pcg**: assertions=0, fallbacks=0, post_ready_recompiles=0, ready_line=470, total_lines=9628

**CONFIRMED_WIN cells** (5): ['cell_t128_r360p_c1', 'cell_t128_r360p_c4', 'cell_t128_r720p_c1', 'cell_t512_r360p_c1', 'cell_t512_r360p_c4']
**NOT_CONFIRMED cells** (2): ['cell_t128_r720p_c4', 'cell_t512_r720p_c1']
**AMBIGUOUS cells** (0): []

## Overall verdict: **PASS**
