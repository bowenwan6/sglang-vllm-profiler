# R6.1 verdict — **FAIL**

> Verdict rules were pre-declared in [`protocol.md`](protocol.md) BEFORE any leg was run. This file computes verdicts from the raw JSON captures under `raw/` and the safety-log tally under `raw/safety_summary.json`.

## Launch context (from `raw/launch_context.json`)

- **Launched by:** direct runner invocation (no idle-GPU monitor)
- **Selected GPU ID:** 0
- Attempt dir: `attempt_02_host_libcuda_595_gpu0`
- Host libcuda pinned: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- LD_PRELOAD: `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
- CUDA_VISIBLE_DEVICES: `0`
- Pre-launch UTC: `2026-07-28T12:39:29+00:00`
- Pre-launch state: `{'compute_pids': [], 'mem_mib': 4, 'util_pct': 0}`
- NVIDIA driver: `595.71.05`
- Stock SGLang HEAD: `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`
- Fork SGLang HEAD: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`
- Hostname: `17b86cda54f9`

**Reasons for non-PASS:**
- (a) fork-default same-run repeat NOT bit-identical
- (d) stock-PCG text != fork-PCG text (fix perturbs text-only PCG)
- (e) mixed-safety subtest failed: {'missing': False, 'assertion_line_numbers': [], 'assertions': 0, 'fallback_line_numbers': [], 'fallbacks': 0, 'inference_only_recompile_count': 0, 'inference_only_recompile_line_numbers': [], 'inference_recompiles': 4, 'notes': "manually reproduced via corrected tally (commit 73d438d). The pre-declared 'inference_recompiles' field is intentionally kept as the total recompile count per protocol §6 which acknowledges the metric conflates warmup and inference-time recompiles. The phase-split fields (warmup_recompiles_count, inference_only_recompile_count) are supplementary evidence added to attempt-02 to characterise which category the 4 recompiles fall into: all 4 occur at lines <server_ready_line (i.e. all pre-server-ready warmup). R6.1 verdict.py still applies the strict pre-declared rule ('inference_recompiles == 0' required for PASS).", 'path': '/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/safety_summary.json', 'recompile_line_numbers_all': [30, 53, 158, 193], 'request_failures': 0, 'server_log': '/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/fork-pcg_server.log', 'server_log_total_lines': 613, 'server_ready_line': 570, 'warmup_recompile_line_numbers': [30, 53, 158, 193], 'warmup_recompiles_count': 4}

## Diagnostic (does not change verdict on its own)

- **(f) baseline stock text default vs stock text PCG**: match_all=**False**
  - idx=0 equal=True first_diff_offset=None
  - idx=1 equal=False first_diff_offset=75
  - idx=2 equal=True first_diff_offset=None

## Per-leg detail

### Leg `a1_fork_default_run1`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_a_fork_default_run1.json`
- mode: `image`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=image status=OK len=194 latency=0.317s
  - idx=1 kind=image status=OK len=267 latency=0.374s
  - idx=2 kind=image status=OK len=298 latency=0.421s

### Leg `a2_fork_default_run2`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_a_fork_default_run2.json`
- mode: `image`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=image status=OK len=194 latency=0.250s
  - idx=1 kind=image status=OK len=267 latency=0.369s
  - idx=2 kind=image status=OK len=295 latency=0.405s

### Leg `b_fork_pcg_image`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_b_fork_pcg_image.json`
- mode: `image`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=image status=OK len=194 latency=0.665s
  - idx=1 kind=image status=OK len=267 latency=0.373s
  - idx=2 kind=image status=OK len=291 latency=0.415s

### Leg `c_stock_default_image`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_c_stock_default_image.json`
- mode: `image`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=image status=OK len=194 latency=1.072s
  - idx=1 kind=image status=OK len=267 latency=0.377s
  - idx=2 kind=image status=OK len=298 latency=0.426s

### Leg `d_stock_pcg_text`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_d_stock_pcg_text.json`
- mode: `text`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=text status=OK len=107 latency=0.173s
  - idx=1 kind=text status=OK len=387 latency=0.449s
  - idx=2 kind=text status=OK len=2 latency=0.094s

### Leg `dp_fork_pcg_text`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_dprime_fork_pcg_text.json`
- mode: `text`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=text status=OK len=107 latency=0.128s
  - idx=1 kind=text status=OK len=372 latency=0.441s
  - idx=2 kind=text status=OK len=2 latency=0.027s

### Leg `f1_stock_default_text`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_f_stock_default_text.json`
- mode: `text`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=text status=OK len=107 latency=0.140s
  - idx=1 kind=text status=OK len=387 latency=0.454s
  - idx=2 kind=text status=OK len=2 latency=0.038s

### Leg `f2_stock_pcg_text`

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/leg_f_stock_pcg_text.json`
- mode: `text`  fixture_sha256: `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
- requests: 3
  - idx=0 kind=text status=OK len=107 latency=4.302s
  - idx=1 kind=text status=OK len=417 latency=0.479s
  - idx=2 kind=text status=OK len=2 latency=0.025s

## Comparisons

### Compare `a: fork-default run1 vs run2`
- match_all: **False**
- idx=0 equal=True len_a=194 len_b=194 first_diff_offset=None
- idx=1 equal=True len_a=267 len_b=267 first_diff_offset=None
- idx=2 equal=False len_a=298 len_b=295 first_diff_offset=241

<details><summary>diff for idx=2</summary>

```diff
--- A
+++ B
@@ -5,4 +5,4 @@
 - **Green** (in the center)
 - **Blue** (on the right)
 
-These three vertical stripes are of equal width and each is a single, uniform solid color with no patterns or gradients.
+These three vertical stripes are of equal width and each is a solid, uniform color without any patterns or gradients.
```
</details>

### Compare `b: fork-PCG image vs fork-default image`
- match_all: **False**
- idx=0 equal=True len_a=194 len_b=194 first_diff_offset=None
- idx=1 equal=True len_a=267 len_b=267 first_diff_offset=None
- idx=2 equal=False len_a=291 len_b=298 first_diff_offset=241

<details><summary>diff for idx=2</summary>

```diff
--- A
+++ B
@@ -5,4 +5,4 @@
 - **Green** (in the center)
 - **Blue** (on the right)
 
-These three vertical stripes are of equal width and each is a solid, uniform color with no patterns or gradients.
+These three vertical stripes are of equal width and each is a single, uniform solid color with no patterns or gradients.
```
</details>

### Compare `c: stock-default image vs fork-default image`
- match_all: **True**
- idx=0 equal=True len_a=194 len_b=194 first_diff_offset=None
- idx=1 equal=True len_a=267 len_b=267 first_diff_offset=None
- idx=2 equal=True len_a=298 len_b=298 first_diff_offset=None

### Compare `d: stock-PCG text vs fork-PCG text`
- match_all: **False**
- idx=0 equal=True len_a=107 len_b=107 first_diff_offset=None
- idx=1 equal=False len_a=387 len_b=372 first_diff_offset=75
- idx=2 equal=True len_a=2 len_b=2 first_diff_offset=None

<details><summary>diff for idx=1</summary>

```diff
--- A
+++ B
@@ -1,7 +1,7 @@
 The three primary additive colors used in digital displays are:
 
-1. **Red**
-2. **Green**
+1. **Red**  
+2. **Green**  
 3. **Blue**
 
-These are often abbreviated as **RGB**. Additive color mixing combines light of these primary colors to create a wide range of hues. When all three are combined at full intensity, they produce white light — the basis for how screens (like TVs, monitors, and smartphones) display color.
+These are known as the **RGB** color model, which is based on the additive mixing of light. When combined in various intensities, they can produce a wide spectrum of colors, making them fundamental to how screens (like TVs, monitors, and smartphones) display images.
```
</details>

## Mixed-safety subtest (e)

- source: `/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/results/R6_fix_value_validation/R6.1_correctness/attempt_02_host_libcuda_595_gpu0/raw/safety_summary.json`
- request_failures: 0
- assertions: 0
- fallbacks: 0
- inference_recompiles: 4
- other_notes: manually reproduced via corrected tally (commit 73d438d). The pre-declared 'inference_recompiles' field is intentionally kept as the total recompile count per protocol §6 which acknowledges the metric conflates warmup and inference-time recompiles. The phase-split fields (warmup_recompiles_count, inference_only_recompile_count) are supplementary evidence added to attempt-02 to characterise which category the 4 recompiles fall into: all 4 occur at lines <server_ready_line (i.e. all pre-server-ready warmup). R6.1 verdict.py still applies the strict pre-declared rule ('inference_recompiles == 0' required for PASS).
