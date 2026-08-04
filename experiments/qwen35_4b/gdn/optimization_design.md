# GDN prefill under BCG — upstream optimization design & feasibility plan

**Status.** Design document. No source edits. Every claim below cites the
current frozen SGLang tree at `/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/sglang_checkout/sglang`
(HEAD `58974ca16c…`). Repo-relative paths omit the checkout prefix.

**Scope.** Optimize the **BCG (Breakable-CUDA-Graph) prefill path** of
`Qwen3.5-4B` GDN linear-attention layers. Empirical baseline: Stage-2/3
measured a reproducible **+13.6 % ± 0.4 %** kernel-count inflation on A1
vs A0 (`experiments/qwen35_4b/gdn/results/gdn_stage3_gpu6_20260803T234412Z/stage3_summary.md`).
Stage-4 attribution (`experiments/qwen35_4b/gdn/stage4_mechanism.md`)
found the inflation is dominated by 9 Triton kernels in the **FLA chunk
kernel family**, each fired ~4,176 additional times per Stage-2 trace.

**Non-goals.** No code changes yet. No claims not grounded in the source
tree. No proposals that would break BCG/CUDA-graph capture or hybrid GDN
correctness.

---

## Deliverable 1 — Complete execution path of FLA chunk-based GDN prefill

### Entry point and dispatcher

* `GDNAttnBackend.forward_extend` (`python/sglang/srt/layers/attention/linear/gdn_backend.py:465`)
  is the per-layer entry for GDN prefill. For the non-`target_verify`
  path it does the following at `:661-679`:
  1. `g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)` — 1 launch.
  2. `self.kernel_dispatcher.extend(q, k, v, g, beta, …)` — routes to
     `TritonGDNKernel.extend` under the default Triton backend.
* Before that at `:558-568` the packed conv is applied: `causal_conv1d_fn`
  (defined in `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py:19`,
  kernel `_causal_conv1d_fwd_kernel`) — 1 launch, the same kernel the
  Stage-4 audit counted as `+4,176`.
* At `:572-581` `fused_qkv_split_gdn_prefill(mixed_qkv, …)` splits the
  packed post-conv tensor into contiguous `[1, T, H, D]` Q/K/V for the
  chunk kernels (`python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py:314`
  kernel, `:356` launcher) — 1 launch, grid `(seq_len,)`.
* `TritonGDNKernel.extend` (`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:169-199`)
  calls exactly one function:

  ```python
  chunk_gated_delta_rule(
      q, k, v, g, beta,
      initial_state=recurrent_state,
      cu_seqlens=query_start_loc,
      head_first=False,
      use_qk_l2norm_in_kernel=True,
      initial_state_indices=cache_indices,
  )
  ```

  This is the FLA autograd wrapper defined in
  `python/sglang/kernels/ops/attention/fla/chunk.py:127+` and is
  decorated `@torch.compiler.disable` at the module level — see comment
  in `chunk.py:127-130`. That means `torch.compile` cannot fuse across
  the launches inside it; any fusion must be inside the FLA package.

### The FLA chunk pipeline (per prefill, per GDN layer)

`chunk_gated_delta_rule_fwd` (`python/sglang/kernels/ops/attention/fla/chunk.py:36-84`)
is the actual pipeline. The complete kernel dependency graph for a
single prefill of a single GDN layer:

```
                   ┌──────────────────────────────┐
                   │  fused_gdn_gating_kernel     │  (gdn_backend :662)
                   │  →  g[1,B,H], beta[1,B,H]    │  1 launch, grid=(B,1,cdiv(H,8))
                   └───────────────┬──────────────┘
                                   │  g
                                   ▼
                   ┌──────────────────────────────┐
                   │  chunk_local_cumsum_scalar_  │  (cumsum.py :159)
                   │  kernel  →  g_cum[1,T,H]      │  1 launch, grid=(NT, B*H)
                   └───────────────┬──────────────┘
                                   │  g_cum
   [use_qk_l2norm_in_kernel=True]  │
   ┌──────────────────────┐        │
   │  l2norm_fwd_kernel   │        │
   │  on q  → q_norm      │◄──┐    │
   └──────────────────────┘   │    │
   ┌──────────────────────┐   │    │       (chunk.py :108-110)
   │  l2norm_fwd_kernel   │   │    │       2 separate launches
   │  on k  → k_norm      │◄──┤    │       grid each = (cdiv(T,16),)
   └──────────────────────┘   │    │
                              ▼    ▼
   ┌────────────────────────────────────────────────────────┐
   │  chunk_gated_delta_rule_fwd_kkt_solve_kernel           │  (chunk_fwd.py :40)
   │  →  A[NT,BT,BT]                                         │  1 launch, grid=(NT,B*H)
   │  Already fuses kkt + solve_tril; see comment            │  (was originally 2 launches;
   │  chunk_fwd.py :349-357.                                 │   now 1)
   └───────────────────────┬────────────────────────────────┘
                           │  A
                           ▼
   ┌────────────────────────────────────────────────────────┐
   │  recompute_w_u_fwd_kernel                              │  (wy_fast.py :23, launcher :131)
   │  →  w[1,T,H,K], u[1,T,H,V]                              │  1 launch, grid=(NT,B*H)
   │  Produces BOTH w and u in one launch (per-BV/BK loops)  │
   └───────────────────────┬────────────────────────────────┘
                           │  w, u
                           ▼
   ┌────────────────────────────────────────────────────────┐
   │  chunk_gated_delta_rule_fwd_kernel_h_blockdim64        │  (chunk_delta_h.py :53)
   │  →  h[N,NT,H,K,V], v_new[1,T,H,V]                        │  1 launch, grid=(cdiv(V,BV), N*H)
   │  INPLACE_UPDATE=True (chunk_delta_h.py :377),           │  bucketized NT bucketing
   │  bucketed by NT_BUCKET (0/1/2) at chunk_delta_h.py :380 │  (graph-friendly!)
   └───────────────────────┬────────────────────────────────┘
                           │  h, v_new
                           ▼
   ┌────────────────────────────────────────────────────────┐
   │  chunk_fwd_kernel_o                                     │  (chunk_o.py :30, launcher :115)
   │  →  o[1,T,H,V]                                          │  1 launch, grid=(cdiv(V,BV),NT,B*H)
   │  BT = min(64, max(16, next_power_of_2(T)))              │  (chunk_o.py :138)
   └────────────────────────────────────────────────────────┘
```

### Per-layer launch count (chunk pipeline only)

Purely counting the FLA-side launches in `chunk_gated_delta_rule_fwd`:

| # | Kernel | Source | Launches / prefill / GDN layer |
|---|---|---|---|
| 1 | `l2norm_fwd_kernel` (q) | `fla/l2norm.py:55` | 1 |
| 2 | `l2norm_fwd_kernel` (k) | `fla/l2norm.py:55` | 1 |
| 3 | `chunk_local_cumsum_scalar_kernel` | `fla/cumsum.py:22` | 1 |
| 4 | `chunk_gated_delta_rule_fwd_kkt_solve_kernel` | `fla/chunk_fwd.py:40` | 1 |
| 5 | `recompute_w_u_fwd_kernel` | `fla/wy_fast.py:23` | 1 |
| 6 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `fla/chunk_delta_h.py:53` | 1 |
| 7 | `chunk_fwd_kernel_o` | `fla/chunk_o.py:30` | 1 |

Adding the SGLang-side surroundings from `forward_extend`:

| # | Kernel | Source | Launches / prefill / GDN layer |
|---|---|---|---|
| 8 | `_causal_conv1d_fwd_kernel` | `mamba/causal_conv1d_triton.py:19` | 1 |
| 9 | `fused_qkv_split_gdn_prefill_kernel` | `attention/triton_gdn_fused_proj.py:314` | 1 |
| 10 | `fused_gdn_gating_kernel` | `fla/fused_gdn_gating.py:11` | 1 |

**10 Triton launches per GDN layer per prefill on this path.** For
Qwen3.5-4B (24 GDN layers) with ~36 BCG segment replays per prefill (per
`stage3_summary.md`), the linear-attention contribution is bounded by
the graph-capture bucket rather than by T directly — every kernel above
except `_causal_conv1d_fwd_kernel` grids on `NT` or `cdiv(T, BT)` where
`T = padded_bucket`.

### Cross-check against Stage-4 kernel diff

Stage-4 (`experiments/qwen35_4b/gdn/stage4_mechanism.md`) counted **9**
distinct FLA-family kernels with roughly-equal per-trace Δ (+4,176 each,
except `l2norm_fwd_kernel` at +8,352 — 2× because q + k). That matches
this pipeline exactly: `chunk_local_cumsum_scalar_kernel`,
`chunk_gated_delta_rule_fwd_kkt_solve_kernel`, `recompute_w_u_fwd_kernel`,
`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`, `chunk_fwd_kernel_o`,
`l2norm_fwd_kernel` (×2), `_causal_conv1d_fwd_kernel`,
`fused_qkv_split_gdn_prefill_kernel`, `fused_gdn_gating_kernel` = 10
kernel names on this path, one of them (l2norm) fired twice, giving the
9 unique kernel names × ~4,176 + 1 × 8,352 Stage-4 saw.

---

## Deliverable 2 — Fusion opportunities

### Already-applied fusions (baseline)

* **kkt + solve_tril fused.** Explicit comment at
  `python/sglang/kernels/ops/attention/fla/chunk_fwd.py:349-357`:
  “Fuses kernels 1+2 into a single kernel, reducing from 3 to 2 kernel
  launches and eliminating the HBM round-trip for the intermediate A
  matrix.” No further fusion win here.
* **recompute_w_u produces both w and u in one launch.**
  `wy_fast.py:43-108` shows a single kernel body that iterates over BV
  to write `u` and over BK to write `w`. Not split.
* **`fused_qkv_split_gdn_prefill_kernel` already exists.**
  `attention/triton_gdn_fused_proj.py:356-407` collapses three
  `aten::copy_` launches into one when `qkv_dim ≤ MAX_FUSED_QKV_SPLIT_DIM`.

### Candidate fusions

The table below rates each candidate by launch reduction, data-dependency
risk, and estimated implementation cost. All references are to the FLA
package inside SGLang (`python/sglang/kernels/ops/attention/fla/`).

| # | Fusion | Kernels merged | Data-dependency notes | Launch Δ / GDN layer | Impl. cost |
|---|---|---|---|---|---|
| F1 | **`l2norm_fwd_kernel` × 2 → single q+k fused l2norm** | 2 launches of `l2norm.py:55` | q, k are independent tensors of shape `[1,T,H,K]` — one kernel can write both with grid `(cdiv(T,16), 2)` and program-ID 1 selecting output. No dep on any other kernel. | −1 (10 → 9) | **Low.** ~40 LOC in `fla/l2norm.py`; add a `l2norm_fwd_pair` helper that keeps the fused kernel signature and stride-checks q and k. Behavior byte-identical to two independent l2norms on shared eps. |
| F2 | **`fused_gdn_gating_kernel` + `chunk_local_cumsum_scalar_kernel` → gate-then-cumsum** | Merge `fused_gdn_gating.py:11` output `g` directly into the cumsum body at `cumsum.py:22`. | `g` is the ONLY output of gating that cumsum needs. `beta_output` is a second gating output that stays separate. Cumsum currently takes `g[1,B,H]` and reduces along a BT-window. Programs would need to compute `g[i_t*BT + j]` on the fly for j∈[0,BT). This requires re-loading `A_log`, `a`, `b`, `dt_bias` inside the cumsum kernel — expands its live-tensor set. | −1 (9 → 8) | **Medium.** Complicates cumsum. Also breaks the pattern where gating is used by non-chunk kernels (target_verify path also calls gating in the recurrent update via `fused_sigmoid_gating_delta_rule_update` at `gdn_triton.py:153-167`). Would need a chunk-specific variant. |
| F3 | **`fused_qkv_split_gdn_prefill_kernel` + `l2norm_fwd_kernel` (q,k) → split-and-normalize** | Merge `triton_gdn_fused_proj.py:314` split with the two l2norm launches (F1). Split writes q and k; the same kernel could normalize while writing. | Feasible: the split kernel already handles per-token stride and writes q/k separately. Adding an l2norm computation requires an in-program reduction over K per row; row-major access is natural. The split kernel today uses `BLOCK_SIZE = next_power_of_2(qkv_dim)` — 1D grid over tokens. To normalize q and k it would need to switch to a 2D layout so reductions stay within a single program. | −2 (10 → 8, subsumes F1) | **Medium-high.** Rewrites the split kernel’s access pattern (BLOCK_SIZE currently spans the entire packed row). Alters the SGLang-owned kernel, not FLA. |
| F4 | **`_causal_conv1d_fwd_kernel` + `fused_qkv_split_gdn_prefill_kernel` → conv-and-split** | The conv output feeds directly into the split. Both are SGLang-owned. | Conv is CUDA-not-Triton and has its own launch shape (grid over dims × width). Fusing across the CUDA/Triton boundary is not practical without a full port. | −1 (10 → 9) | **High.** Cross-language fusion; would require porting `_causal_conv1d_fwd_kernel` to Triton or a hand-written CUDA fused kernel. Correctness surface is enormous (batched packed conv + variable-length). Not recommended. |
| F5 | **`chunk_gated_delta_rule_fwd_kernel_h_blockdim64` + `chunk_fwd_kernel_o` → h,o fused** | Merge `chunk_delta_h.py:53` (writes `h`, `v_new`) with `chunk_o.py:30` (reads `q`, `k`, `v_new`, `h`, writes `o`). | Two hard blockers. (a) **h is a required inter-chunk state and must be materialized in HBM** — `chunk_fwd_kernel_o` reads `h` at chunk `i_t` while `chunk_delta_h` writes `h` at chunks `[i_t+1 .. NT]`. Different program-IDs in the two kernels see the same HBM slots; there is no producer-consumer relation on the SM that can bypass HBM. (b) The two kernels grid differently: h uses `(cdiv(V,BV), N*H)` (chunk_delta_h.py:351), o uses `(cdiv(V,BV), NT, B*H)` (chunk_o.py:149) — different program-ID cardinality, incompatible without dropping the o kernel’s parallelism across NT. | 0 in the best case (would keep 2 launches), and would sacrifice o’s NT parallelism. | **High and risky.** Not viable within a single-kernel Triton launch. |
| F6 | **`recompute_w_u_fwd_kernel` + `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` → w,u,h fused** | w,u are computed per chunk-block and immediately consumed by the h-kernel to advance state. | Same dependency shape as F5. h needs w and u for all chunks up to `i_t` before it can advance to `i_t+1`. Within a single kernel this would serialize the entire NT dimension on-chip, killing the NT parallelism h enjoys. | 0 useful launches saved. | **High and risky.** Not viable. |

### Ordered summary of fusion opportunities

* **F1 (l2norm q+k → one)** — cleanest, biggest ratio of value/cost.
  Removes 1 of the 10 launches unambiguously; no shape or dispatch
  change; identical numeric result. **Recommended.**
* **F3 (split+l2norm)** — subsumes F1 for a −2 net gain but touches an
  SGLang-owned kernel with a working input-block layout that is not
  amenable to per-row reductions.
* **F2 (gate+cumsum)** — worth exploring after F1 lands; blocked on
  matching the target_verify path.
* **F4, F5, F6** — either cross-language or defeated by the intrinsic
  producer-consumer data dependency through `h`. Not viable as
  single-kernel fusions.

**Expected launch-count reduction ceiling with viable fusions
(F1+F2+F3):** 3 launches per GDN layer per prefill (10 → 7). At 24 GDN
layers × 10 prefills per Stage-2 trace, this is ~720 fewer launches out
of the +9,300/req delta — **~7.7 %** of the inflation. Modest but real
and it targets exactly the kernels the mechanism attribution found.

---

## Deliverable 3 — Could the recurrent packed implementation become CUDA-graph compatible?

### The two recurrent kernels in the tree

* **`fused_recurrent_gated_delta_rule_packed_decode_kernel`**
  (`python/sglang/kernels/ops/attention/fla/fused_recurrent.py:186`).
  Launcher at `:268-402`, grid at `:373`
  = `(NV, B * HV)`. Body assumes **exactly one token per program**
  (T=1); no inner loop over T. Called from `TritonGDNKernel.packed_decode`
  (`python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:121-132`).
  This is a **decode-only** kernel.
* **`fused_recurrent_gated_delta_rule_fwd_kernel`**
  (`python/sglang/kernels/ops/attention/fla/fused_recurrent.py:15-121`).
  Decorated `@triton.jit(do_not_specialize=["T"])`. Launcher at `:124-181`,
  grid at `:151` = `(NK, NV, N * HV)` where `N = len(cu_seqlens) - 1`.
  Contains a runtime for-loop `for _ in range(0, T)` at `:80` where `T`
  is computed inside the kernel at `:50` from `cu_seqlens` as `T = eos - bos`.
  **This is the variable-length recurrent prefill kernel** and is the
  only recurrent candidate that could carry a prefill.

`TritonGDNKernel.extend` currently **does not** dispatch either
recurrent kernel — it always calls `chunk_gated_delta_rule`
(`gdn_triton.py:188-198`). So the question is not “switch back to a
kernel BCG used to use for prefill,” it is “could the varlen recurrent
kernel replace the chunk pipeline under BCG?”

### Runtime-dependent arguments

Enumerating `fused_recurrent_gated_delta_rule_fwd_kernel` signature
(`fused_recurrent.py:15-41`):

| Arg | Kind | Runtime-dependent? |
|---|---|---|
| `q, k, v, g, beta` | tensor ptr | tensor address — **must be static-slot** to be graph-safe |
| `o, h0, ht` | tensor ptr | tensor address — **must be static-slot** |
| `cu_seqlens` | tensor ptr | tensor address; values inside are read to compute per-program `bos, eos, T` |
| `scale` | float | scalar; graph-safe if fixed per layer (it is) |
| `T` | int (do_not_specialize) | value used both as loop bound and stride multiplier — **runtime-varying** |
| `B, H, HV, K, V, BK, BV` | `tl.constexpr` | graph-safe only if constant per capture (they are, per bucket) |
| `USE_INITIAL_STATE, STORE_FINAL_STATE, IS_BETA_HEADWISE, USE_QK_L2NORM_IN_KERNEL, IS_VARLEN, IS_KDA` | `tl.constexpr` | graph-safe only if constant per capture |
| grid dim `(NK, NV, N * HV)` | derived | **N varies with batch composition** — graph-safe only per bucket |

### What actually breaks graph capture

1. **Runtime for-loop with per-program trip count.** Line `:80`
   `for _ in range(0, T)` where `T = tl.load(cu_seqlens+i_n+1) - tl.load(cu_seqlens+i_n)`.
   CUDA graphs record kernel launches, not internal control flow, so
   this by itself is **not** the blocker: within a single graph replay
   the kernel is a single launch and Triton emits the loop naturally.
   *But* the per-request T varies with the actual live batch on every
   replay, which means the SAME captured launch fires a different amount
   of work each time — that is legal for CUDA graphs (they just replay
   the launch), but the launch performance is data-dependent.
2. **Grid dimension `N * HV`.** N is “number of active requests” for
   this replay. Under BCG the grid MUST be constant for the captured
   launch. This is the real hard blocker. `chunk_gated_delta_rule` sidesteps
   it because its grid depends on `NT` (which is a function of the
   padded bucket `T`, a constant per capture) rather than on N directly.
3. **`do_not_specialize=["T"]` interacts with Triton JIT caching.**
   The captured kernel binary was compiled for whatever T the capture
   saw. Different T triggers a JIT recompile — a recompile inside a
   captured graph is not allowed. With `do_not_specialize=["T"]`, T is
   passed as a runtime argument and no recompile is triggered — good.
   The other `constexpr` args (B, H, HV, etc.) do recompile on change,
   which is fine since they are truly constant per bucket.
4. **`initial_state` and per-request state indexing.** The kernel’s
   `p_h0 = h0 + i_nh * V * K + …` at `:77` is contiguous-slot indexing.
   The chunk kernels use `initial_state_indices` for the pool-slot
   remap; the recurrent kernel does not. Under BCG the state pool is
   allocated static-slot per bucket, and the packed_decode kernel does
   support `ssm_state_indices` (`fused_recurrent.py:222`), so this is a
   surmountable API change but does require rewriting the recurrent
   prefill kernel — see below.

### Would bucketized graph capture help?

Yes, at a cost that has to be paid up-front:

* **Idea.** Follow the pattern already used by `chunk_delta_h.py:380`
  (`NT_BUCKET = 0 if NT <= 32 else (1 if NT <= 128 else 2)`) but at the
  batch-composition level. Capture one graph per `(B_bucket, T_bucket)`
  pair; pad the recurrent kernel’s grid to `B_bucket * HV` and mask off
  the unused N slots via `initial_state_indices` = −1 (see
  `packed_decode` kernel `:225-228`, which already treats `state_idx < 0`
  as “write zeros and return”). This mirrors the trick packed_decode
  already applies for empty request slots.
* **Cost.** Each additional bucket multiplies the number of Triton
  compilations and the number of captured graphs. BCG already buckets
  by padded token count; adding a request-count dimension expands the
  bucket cross-product. On Qwen3.5-4B with `bs≤32` this is manageable
  (~4 buckets × existing T-buckets); on larger deployments it grows.
* **Missing invariants that must be added to make it work.**
  1. Add a `state_indices` argument to the varlen recurrent kernel
     (mirroring `ssm_state_indices` in packed_decode at
     `fused_recurrent.py:222`) so the pool’s real slot layout survives.
     Non-trivial: also touches `p_h0` and `p_ht` at `:77` and `:120`.
  2. Add a graceful-empty-slot early-return (line `:225-228` template)
     for masked-off N slots.
  3. Add per-request maximum-T reduction: the current loop iterates T
     per program based on that program’s `i_n`. Under bucket padding
     the wasted programs would iterate `T=0` — already legal.

### Comparison to chunk-kernel fusion (Deliverable 2)

| Axis | Chunk fusion (e.g. F1) | Bucketized recurrent varlen |
|---|---|---|
| Launches saved / GDN layer / prefill | 1 – 3 | 8 – 10 (collapses whole chunk pipeline into 1 kernel) |
| Perf headroom | Low (~7 % of the +13.6 % inflation) | High (could eliminate most of it — the inflation IS the chunk-family expansion) |
| Complexity | Low (F1) to medium (F2/F3) | **High.** New kernel API, per-slot indexing rewrite, batch-count bucketing in the BCG capture loop, correctness regression risk on hybrid GDN state |
| Correctness risk | Minimal (F1 is numerically identical) | **High.** Serial recurrent state update is harder to test-match vs the chunk-parallel version; small numerical differences from operator order will show up in end-to-end evals |
| Upstream acceptance | High for F1 (obvious win, tiny diff) | **Low.** Adds a new capture-shape dimension, changes the FLA kernel signature, and is only motivated by a small (0.6-1.6 %) wall-clock win — no upstream would accept the complexity without matching wall-clock evidence |
| Reversibility | Full (single-file kernel change) | Poor. Capture-loop changes bleed into MambaPool, BCG segment planning, and warmup |
| Debuggability under regression | Easy (unit-test the l2norm kernel) | Hard (requires end-to-end BCG capture + replay under multiple bucket combos) |

**Verdict for Deliverable 3.** The recurrent packed implementation *could*
be made CUDA-graph compatible in principle, via bucketized capture over
`(B_bucket, T_bucket)` plus a rewrite of
`fused_recurrent_gated_delta_rule_fwd_kernel` to support pool-slot
indexing. It is **not recommended** for upstreaming: the engineering
surface is much larger than the chunk-fusion path, and the wall-clock
prize is bounded by Stage-4’s +22 to +61 ms/request (0.6-1.6 %).

---

## Deliverable 4 — Ranked optimization opportunities

Legend: **Perf** = expected wall-clock reduction of the Stage-3 delta
(baseline +22-61 ms/req); **Effort** = engineer-days at the frozen
SGLang SHA; **Risk** = correctness/regression risk; **Upstream** =
probability of clean acceptance in FLA or SGLang upstream; **Layer** =
which project the change lands in.

| Rank | Optimization | Perf | Effort | Risk | Upstream | Layer |
|---|---|---|---|---|---|---|
| **1** | **F1: fuse `l2norm_fwd_kernel(q)` + `l2norm_fwd_kernel(k)` into a single kernel launched by `chunk_gated_delta_rule_fwd`** at `chunk.py:108-110` | **Small–moderate (−1 of 10 launches per GDN layer per prefill; ~10 % of chunk-family launches, ~1.5-2 % of the +9,300/req delta)** | **1-2 days** | **Very low** (byte-identical output; single-file change in `fla/l2norm.py` + 2-line call-site change in `fla/chunk.py`) | **High** — matches FLA’s existing “fuse 3→2” pattern in `chunk_fwd.py:349-357`; small self-contained PR | **FLA (upstream)** — SGLang picks it up on next FLA sync |
| 2 | F2: fold `fused_gdn_gating` output into `chunk_local_cumsum_scalar_kernel` (chunk-specific variant) | Small (−1 launch) | 3-5 days | Low-medium (target_verify path still needs the standalone gating kernel; must not regress that) | Medium — adds a specialized variant | SGLang (owns `fla/fused_gdn_gating.py`, `fla/cumsum.py` copies) |
| 3 | F3: fold `l2norm(q,k)` into `fused_qkv_split_gdn_prefill_kernel` (subsumes F1 for a −2 net) | Small (−2 launches) | 5-8 days | Medium (rewrites split-kernel access pattern) | Medium — SGLang-only kernel, but the reviewers will want unit-perf evidence | SGLang (`triton_gdn_fused_proj.py`) |
| 4 | Autotune enablement in `wy_fast.py` (the config block at `:14-21` is commented out) — could pick better `num_warps`/`num_stages` for Qwen3.5-4B’s N-head shape | Small (unknown; pure autotune, no launch-count change) | 1 day | Low | Medium (FLA maintainers prefer autotune off by default; SGLang could ship a static config override) | FLA or SGLang override |
| 5 | Bucketized-N recurrent-varlen prefill (Deliverable 3) | Moderate–high (could remove up to 8-10 launches per GDN layer per prefill) | 3-6 weeks | **High** (new kernel API, capture-shape dimension) | **Low** (justification is 0.6-1.6 % wall-clock) | SGLang + FLA + BCG capture loop |
| 6 | Direct fusion of chunk-h + chunk-o (F5) | 0 net launches saved | 2-4 weeks | High (data dep through h) | Low | FLA |

---

## Deliverable 5 — Recommendation

**Recommendation: implement F1 — fuse the two `l2norm_fwd_kernel`
launches invoked by `chunk_gated_delta_rule_fwd` when
`use_qk_l2norm_in_kernel=True`.**

### Rationale against the requested criteria

* **Highest upstream acceptance probability.** F1 is a self-contained
  change to `python/sglang/kernels/ops/attention/fla/l2norm.py` plus a
  2-line replacement at `python/sglang/kernels/ops/attention/fla/chunk.py:108-110`
  (`q = l2norm_fwd(q); k = l2norm_fwd(k)` → `q, k = l2norm_fwd_pair(q, k)`).
  It matches an established FLA fusion pattern (see the identical
  motivating comment at `chunk_fwd.py:349-357`).
* **Measurable perf benefit.** Removes exactly one Triton launch per
  GDN layer per prefill under BCG. On Qwen3.5-4B (24 GDN layers) that
  is 24 fewer launches per prefill; scaled to the Stage-3 test
  configuration (~36 replays/prefill, 10 prefills/trace) it is ~8,600
  fewer launches per trace — directly measurable via the same nsys
  extractor used in Stage-3.
* **Limited implementation risk.** Both l2norm launches operate on
  independent tensors of shape `[1, T, H*K]` (q and k reshape identically
  at `chunk.py:107` where `l2norm_fwd(q)` reshapes to `(-1, D)`). A
  single fused kernel can launch with grid `(cdiv(T*H, 16), 2)` and use
  `tl.program_id(1)` to select between the q and k pointers. Behavior
  is byte-identical to two independent launches with shared eps.
* **Suitable as a standalone PR.** No API breaks. No new capture shapes.
  No changes to `MambaPool`, `GDNAttnBackend`, or `TritonGDNKernel`
  above `chunk_gated_delta_rule`. Existing unit tests for l2norm still
  cover single-tensor use; add one pair test.

### Implementation sketch (design only — no code)

1. **New helper in `fla/l2norm.py`**:
   * `l2norm_fwd_pair(x1, x2, eps=1e-6, output_dtype=None) -> (y1, y2)`.
   * Requires `x1.shape == x2.shape` and matching stride profile
     (both are `[1, T, H, K]` here; both flatten to `[-1, D]` with same D).
   * New kernel `l2norm_fwd_kernel_pair` similar to `l2norm_fwd_kernel`
     at `l2norm.py:55` but with an extra `tl.program_id(1)` selecting
     between x1 (write y1) and x2 (write y2). Grid: `(cdiv(T, BT), 2)`.
   * Fall through to the existing single-tensor path when `D > 512`
     (matches the current `if D <= 512` branch at `l2norm.py:92`).
2. **Call-site change in `fla/chunk.py:107-110`**:
   ```python
   if use_qk_l2norm_in_kernel:
       q, k = l2norm_fwd_pair(q, k)
   ```
3. **Test additions**: a numerical parity test comparing
   `l2norm_fwd_pair(q, k)` against `(l2norm_fwd(q), l2norm_fwd(k))` for
   the shapes Qwen3.5-4B exercises (K=128, various T).

### Why this is the right single-PR pick

The Stage-4 audit already showed the +13.6 % delta is structural — the
FLA chunk kernel family is the right implementation to keep, because it
is the only implementation that is graph-capture-friendly for hybrid
GDN. The best return on upstream effort is therefore incremental fusion
inside that family, matching FLA’s own approach
(`chunk_fwd.py:349-357`). Among such fusions F1 is the one where the
producer (l2norm) and consumer (kkt_solve) are already independent, and
where the two kernels being fused have identical program-per-token
structure. The other fusions either serialize a data-dependent chain
(F5/F6), touch multi-language kernels (F4), or require variant
maintenance across the target-verify path (F2). F1 is the low-risk,
upstream-compatible pick.

---

## References

* `python/sglang/srt/layers/attention/linear/gdn_backend.py:465-679` — GDN backend extend path.
* `python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py:41-199` — Triton GDN kernel dispatcher.
* `python/sglang/kernels/ops/attention/fla/chunk.py:36-110` — FLA chunk_gated_delta_rule_fwd pipeline.
* `python/sglang/kernels/ops/attention/fla/chunk_fwd.py:40, :339-410` — kkt+solve fused kernel and launcher, including motivating fusion comment at :349-357.
* `python/sglang/kernels/ops/attention/fla/wy_fast.py:23-108, :131` — recompute_w_u kernel and launcher.
* `python/sglang/kernels/ops/attention/fla/chunk_delta_h.py:53, :351, :380` — h-forward kernel, grid, NT bucketing.
* `python/sglang/kernels/ops/attention/fla/chunk_o.py:30, :115-149` — output-combine kernel and launcher.
* `python/sglang/kernels/ops/attention/fla/cumsum.py:22, :159-198` — chunk local cumsum kernel and launcher.
* `python/sglang/kernels/ops/attention/fla/l2norm.py:24, :55, :73-119` — l2norm kernel variants and launcher.
* `python/sglang/kernels/ops/attention/fla/fused_gdn_gating.py:11-75` — gating kernel and launcher.
* `python/sglang/kernels/ops/attention/fla/fused_recurrent.py:15-181` — varlen recurrent fwd kernel (candidate for Deliverable 3).
* `python/sglang/kernels/ops/attention/fla/fused_recurrent.py:185-402` — packed_decode kernel (T=1 decode).
* `python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py:314-407` — SGLang-owned fused QKV split.
* `python/sglang/kernels/ops/mamba/causal_conv1d_triton.py:19` — packed conv kernel.
* `experiments/qwen35_4b/gdn/stage4_mechanism.md` — empirical kernel-count attribution.
* `experiments/qwen35_4b/gdn/results/gdn_stage3_gpu6_20260803T234412Z/stage3_summary.md` — Stage-3 threshold ladder result.
