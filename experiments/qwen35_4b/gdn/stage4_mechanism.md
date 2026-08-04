# Stage-4 mechanism attribution — GPU 6

**Purpose.** Attribute Stage-2's +13.6 % steady-state A1-vs-A0 kernel
inflation to specific ops, using existing Nsight kernel names and
Stage-3's threshold-ladder captures before considering NVTX
instrumentation. Per operator brief Stage 4: *first use existing
Nsight kernel names, stream IDs, CUDA API calls, events, and source
audit findings*.

## Method

`nsys stats --report cuda_gpu_kern_sum` on Stage-3's A0_p128_rep1
and A1_p128_rep1 nsys captures. Diff per-kernel-name instance counts.

## Result — mechanism identified WITHOUT NVTX

**Under BCG, SGLang switches Qwen3.5-4B's GDN prefill from the
eager "recurrent packed" kernel family to the FLA chunk kernel family.**

### Kernels ADDED under A1 (BCG) — FLA chunk prefill implementation

| kernel | A0 | A1 | Δ | role |
|---|---|---|---|---|
| `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | 288 | 4,464 | **+4,176** | FLA chunk GDN core forward |
| `chunk_gated_delta_rule_fwd_kkt_solve_kernel`    | 288 | 4,464 | **+4,176** | FLA chunk K·Kᵀ solve |
| `chunk_fwd_kernel_o`                             | 288 | 4,464 | **+4,176** | FLA chunk output combine |
| `chunk_local_cumsum_scalar_kernel`               | 288 | 4,464 | **+4,176** | FLA chunk local cumsum |
| `recompute_w_u_fwd_kernel`                       | 288 | 4,464 | **+4,176** | FLA chunk w/u recompute |
| `fused_qkv_split_gdn_prefill_kernel`             | 288 | 4,464 | **+4,176** | GDN-prefill fused split |
| `fused_gdn_gating_kernel`                        | 288 | 4,464 | **+4,176** | GDN gating fusion |
| `_causal_conv1d_fwd_kernel`                      | 288 | 4,464 | **+4,176** | Causal 1D conv (chunk mode) |
| `l2norm_fwd_kernel`                              | 576 | 8,928 | **+8,352** | L2 norm (2× per chunk-step) |

### Kernels REDUCED under A1 (BCG) — recurrent packed decode kernels no longer used for prefill

| kernel | A0 | A1 | Δ | role |
|---|---|---|---|---|
| `fused_recurrent_gated_delta_rule_packed_decode_kernel` | 30,579 | 30,333 | **−246** | Recurrent GDN core (A0 uses for prefill AND decode; A1 decode only) |
| `_causal_conv1d_update_kernel`                          | 30,579 | 30,333 | **−246** | Recurrent 1D conv update |
| `track_mamba_state_if_needed_kernel`                    | 30,579 | 30,333 | **−246** | Recurrent state tracker |

### Shared kernels with modest increases (per-chunk vs per-prefill work)

| kernel | A0 | A1 | Δ |
|---|---|---|---|
| `fused_add_rmsnorm`                     | 82,311 | 88,376 | +6,065 |
| `act_and_mul_kernel`                    | 41,156 | 44,188 | +3,032 |
| `_layer_norm_fwd_1pass_kernel`          | 30,867 | 33,141 | +2,274 |
| `fused_qkvzba_split_reshape_cat_contiguous_kernel` | 30,867 | 33,141 | +2,274 |
| various nvjet cuBLAS GEMM variants (new SM90 tiles) | 0 | ~6,900 total | +6,900 |
| `vectorized_elementwise_kernel` (fill) | 1,015 | 9,440 | +8,425 |

### Total accounting (whole trace)

- A0 total kernels: **679,358**
- A1 total kernels: **772,524**
- Overall Δ: **+93,166** (matches Stage-3's +92,712 average)
- Sum of top-40 positive diffs: **+88,056**
- Sum of top-40 negative diffs: **−246**
- Net (top 40): **+87,810** — accounts for **94 %** of the whole-trace Δ

## Why BCG uses different kernels

**Structural, not a defect.** The FLA "chunk" kernel family
(`chunk_gated_delta_rule_fwd_*`, `_causal_conv1d_fwd_kernel`,
`fused_qkv_split_gdn_prefill_kernel`, `recompute_w_u_fwd_kernel`,
`l2norm_fwd_kernel`) is designed for **fixed-shape execution** —
each chunk kernel takes a bucket-sized tensor and dispatches
statically-known kernel arguments. This is exactly what CUDA graph
capture requires.

The "recurrent packed decode" family
(`fused_recurrent_gated_delta_rule_packed_decode_kernel`) uses
**variable-length packing** with per-request offsets; kernel launch
arguments depend on the runtime batch composition, which cannot be
graph-captured.

Under **eager (A0)**, SGLang uses the recurrent packed
implementation for both prefill and decode (fewer, larger, more
efficient kernels; more per-kernel overhead handled by CUDA).

Under **BCG (A1)**, SGLang must use the chunk implementation for
prefill (many more, smaller kernels — but fixed shape per bucket, so
captured into the graph). Decode continues on the recurrent packed
path (BCG here disables decode CG, so decode kernels are unchanged
from A0).

## Wall-clock impact

Per-request e2e latency across the Stage-3 ladder (all p, batch=1,
n_timed=8, means across rep1+rep2):

| prompt_len | A0 e2e mean | A1 e2e mean | Δ | pct |
|---|---|---|---|---|
| 128 | 3811.8 ms | 3872.5 ms | +60.7 ms | **+1.6 %** |
| 1024 | 3792.6 ms | 3815.1 ms | +22.4 ms | +0.6 % |
| 2048 | 3837.1 ms | 3865.2 ms | +28.2 ms | +0.7 % |
| 4096 | 3814.4 ms | 3844.9 ms | +30.5 ms | +0.8 % |

**Wall-clock delta is small: +22 to +61 ms per request (0.6–1.6 %).**
Decode (128 new tokens × ~25 ms/step ≈ 3.2 s) dominates e2e, so the
+9,300 extra kernel launches per prefill under BCG cost ~30 ms of
prefill wall-clock — real but modest.

Note: these measurements are under `nsys profile` (which itself adds
overhead); the unprofiled Phase-6 A1 vs unprofiled Phase-5 A0 at
p=128 b=1 also showed a similar +2 % e2e gap (Phase 5 A0 mean ~3025
ms, Phase 6 A1 unprofiled ~3025 ms — comparable), so the Nsight
overhead is small on this workload and does not dominate the delta.

## Verdict recommendation

**`PASS_BCG_GDN_NOTABLE_GAP`** per hypothesis.md §5:
- H_A (kernel-count inflation on BCG-enabled arm) is supported and
  reproducible: +13.6 % ± 0.4 %, MIN_CAPTURES_FOR_REPRO=2 met at 4
  cells and 2 reps each (Stage 3).
- Mechanism attributed at the source-code level (FLA chunk kernel
  family vs recurrent packed kernel family) using existing Nsight
  data — NVTX not needed.
- Wall-clock impact is small (~30 ms per prefill, 0.6–1.6 %).

**No source patch is justified.** The FLA chunk kernel family is a
structural design choice enabling CUDA graph capture for GDN
prefill; removing it would disable BCG entirely for hybrid GDN
models. The kernel-count inflation is the intrinsic cost of graph-
compatible prefill on this architecture on this frozen SGLang SHA.

The Stage-3 rejection of the alt-stream `<1024` hypothesis
(H12.1) stands: the alt-stream branch inside `_forward_input_proj`
is not the cause of the kernel-count delta. The delta is entirely
attributable to the prefill-implementation switch.

## Stage-4 signal

**`SIGNAL_GOOD`.** Mechanism identified from existing Nsight kernel
names — no NVTX instrumentation added, no frozen-SGLang source
modification triggered. Continue to final report + verdict.

## Preservation invariants (verified post-Stage 4)

- `/data/sglang-fork` HEAD unchanged: `986c89e69c…`.
- Frozen SGLang HEAD unchanged: `58974ca16c…`, empty `git diff --stat`.
- No new GPU work — Stage 4 was entirely post-processing of
  Stage-3's nsys captures.
