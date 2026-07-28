# R6.1 amended verdict — **FAIL**

> Evaluated under [`protocol_amendment_A_direct_fix_comparison.md`](../protocol_amendment_A_direct_fix_comparison.md). Rules were pre-declared before any leg ran.

## Launch context

- `selected_gpu_id`: 0
- `attempt_dir`: attempt_03_amended_A_gpu0
- `host_libcuda`: /usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
- `ld_preload`: /usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
- `cuda_visible_devices`: 0
- `nvidia_driver`: 595.71.05
- `sglang_stock_head`: da802ddcafe55e25b3e1db86b1e0444afc3e05bc
- `sglang_fork_head`: 986c89e69c25882ab6f3d396f8eb306f38f2c8d2

## Tier 1 — SAFETY_SUPERIORITY: **FAIL**

- Negative control (stock-PCG image): **STOCK_NOW_SURVIVES**
  - reason: `stock-PCG served all 3 image requests with HTTP 200; historical failure does not reproduce on da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- fork-PCG interleaved leg: all_http_200=True, safety metrics zero=True, no inflight recompiles=True
- fork-PCG safety: `{"assertions": 0, "fallbacks": 0, "markers": [{"kind": "SERVER_READY", "label": "fork_pcg_interleaved", "ts": "2026-07-28T13:49:14.549Z"}, {"kind": "LEG_START", "label": "fork_pcg_interleaved", "ts": "2026-07-28T13:49:14.678Z"}, {"kind": "LEG_END", "label": "fork_pcg_interleaved", "ts": "2026-07-28T13:49:17.208Z"}], "notes": "post_ready_recompiles conservatively attributed to the last recorded LEG_START; refine in R7 if needed.", "per_leg_recompiles": {"fork_pcg_interleaved": 0}, "phase_markers": "/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_03_amended_A_gpu0/raw/fork_pcg_interleaved_phase_markers.txt", "post_ready_recompile_lines": [], "post_ready_recompiles": 0, "request_failures": 0, "server_log": "/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_03_amended_A_gpu0/raw/fork_pcg_interleaved_server.log", "server_log_total_lines": 591, "server_ready_line": 575, "startup_warmup_recompile_lines": [30, 53, 160, 195], "startup_warmup_recompiles": 4}`

**Failure reasons:**
- negative_control.result=STOCK_NOW_SURVIVES (need EXPECTED_STOCK_FAILURE)

## Tier 2 — CORRECTNESS: **PASS**

### Matched cold-cache repeat envelopes (per-prompt tok Levenshtein)

- `stock_default_image`: repeat_lev=[0, 0, 0], envelope_max=[2, 2, 2]
- `fork_default_image`: repeat_lev=[0, 0, 0], envelope_max=[2, 2, 2]
- `stock_pcg_text`: repeat_lev=[0, 0, 0], envelope_max=[2, 2, 2]
- `fork_pcg_text`: repeat_lev=[0, 42, 0], envelope_max=[2, 42, 2]
- `fork_pcg_image`: repeat_lev=[0, 0, 0], envelope_max=[2, 2, 2]

### Cross-config comparisons

#### `stock_default_vs_fork_default__image_cold`
- All inside envelope: **True**

| Prompt | tok_lev | union_env | inside? | char==/tok== |
|---|---|---|---|---|
| 0 | 0 | 2 | True | True / True |
| 1 | 0 | 2 | True | True / True |
| 2 | 0 | 2 | True | True / True |

#### `stock_pcg_vs_fork_pcg__text_cold`
- All inside envelope: **True**

| Prompt | tok_lev | union_env | inside? | char==/tok== |
|---|---|---|---|---|
| 0 | 0 | 2 | True | True / True |
| 1 | 42 | 42 | True | False / False |
| 2 | 0 | 2 | True | True / True |

#### `fork_default_vs_fork_pcg__image_cold`
- All inside envelope: **True**

| Prompt | tok_lev | union_env | inside? | char==/tok== |
|---|---|---|---|---|
| 0 | 0 | 2 | True | True / True |
| 1 | 0 | 2 | True | True / True |
| 2 | 0 | 2 | True | True / True |

## Overall verdict: **FAIL**

- Evidence tier claimed: **NONE**

