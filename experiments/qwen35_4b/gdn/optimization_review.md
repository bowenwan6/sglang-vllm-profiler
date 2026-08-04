# F1 (l2norm q+k fusion) — Stage-1 plan review

**Status.** Stage-1 review only. No code changes, no new GPU experiments.
Companion to `optimization_design.md` (commit `cc21e0e`).

**Reviewers.** Three independent reviewers were spun off in parallel
against the frozen SGLang checkout at `/tmp/claude-0/.../sglang_checkout/sglang`
(HEAD `58974ca16c…`):

* Kernel feasibility (Triton correctness, shape/dtype, numerical parity)
* Performance skepticism (wall-clock realism, Amdahl reality check,
  alternative candidates)
* Integration & correctness (call-site coverage, alternate paths,
  test strategy, upstream maintenance)

---

## Verdict — **PLAN_ACCEPT** (with 3 caveats)

Ready to execute Stage 2. See “Caveats & scope adjustments” below.

---

## What was verified

### Feasibility (kernel-level)

* **The two launches are truly consecutive and independently fusible.**
  `python/sglang/kernels/ops/attention/fla/chunk.py:108-110` calls
  `q = l2norm_fwd(q)` then `k = l2norm_fwd(k)` back-to-back inside
  `ChunkGatedDeltaRuleFunction.forward`. No intervening op; q and k are
  independent tensors.
* **Q and K have identical shape, dtype, and stride at the call site.**
  Qwen3.5-4B / Qwen3-Next builds RadixLinearAttention with
  `num_q_heads == num_k_heads` and `head_q_dim == head_k_dim`
  (`python/sglang/srt/models/qwen3_next.py:238-242`;
  `python/sglang/srt/configs/qwen3_next.py:205-208`,
  `linear_key_head_dim=128` default gated by `gdn_backend.py:87-88`).
  Both tensors are fresh contiguous allocations from
  `fused_qkv_split_gdn_prefill_kernel`
  (`python/sglang/kernels/ops/attention/triton_gdn_fused_proj.py:373-387`).
* **The D≤512 branch is the actual production path.** D = 128, so the
  `l2norm_fwd_kernel` block-BT variant fires
  (`python/sglang/kernels/ops/attention/fla/l2norm.py:92-107`; BT=16,
  num_warps=8, num_stages=3). The `l2norm_fwd_kernel1` per-row branch
  at `:108-117` does not fire in production.
* **Numerical parity is bit-exact.** Per-row reductions are independent;
  same eps default (1e-6, `l2norm.py:74`), same fp32 upcast, same tile
  order — fused vs split produce bit-identical outputs on the same
  input.
* **Fused-kernel implementation is a small delta.** Grid becomes
  `(cdiv(T*H, BT), 2)` with `pid1 ∈ {0,1}` selecting between the
  (x_q,y_q) and (x_k,y_k) pointer pairs. Two `tl.make_block_ptr`s per
  program, gated by a compile-time branch on `pid1`
  (`tl.static_range` / `tl.constexpr`), not a runtime `if`.
* **No autograd backward to keep in sync.** SGLang's `L2NormFunction`
  has only forward (`l2norm.py:122-127`); no `l2norm_bwd` anywhere in
  the SGLang tree. Upstream FLA does keep a backward, so if we
  upstream, we must add a paired backward — but SGLang inference is
  unaffected.

### Integration (call-site coverage)

The design report cited only `chunk.py:108-110`. The audit found the
paired-l2norm pattern in **5 production sites** and 4 benchmark sites:

| # | Site | Notes |
|---|---|---|
| 1 | `fla/chunk.py:109-110` | GDN chunk-prefill, Triton dispatcher; guarded by `use_qk_l2norm_in_kernel=True` (invariantly True on the SGLang GDN path) |
| 2 | `fla/kda.py:1155-1156` | KDA (Kimi delta attention) chunk-prefill; same guard |
| 3 | `srt/layers/attention/linear/kernels/gdn_flashinfer.py:295-296` | GDN flashinfer backend — unconditional pre-norm before calling the fused kernel with `use_qk_l2norm_in_kernel=False` |
| 4 | `srt/layers/attention/linear/kernels/gdn_cutedsl.py:133-134` | GDN CuteDSL backend, via `self._l2norm_fn` |
| 5 | `srt/layers/attention/linear/kernels/kda_cutedsl.py:121-122` | KDA CuteDSL backend, via `self._l2norm_fn` |
| — | `python/sglang/kernels/ops/attention/fla/bench_gdn_prefill.py:212/213, 396/397` | Benchmark harness |
| — | `python/sglang/kernels/ops/attention/fla/bench_gdn_prefill_cutedsl.py:146/147, 272/273` | Benchmark harness |

No single-tensor call sites exist in the SGLang tree — every `l2norm_fwd`
call is one of a paired q+k invocation. The paired helper improves
value/cost across all 5 backends, not only the Triton chunk path.

Alternate GDN dispatch paths that are **unaffected** because the
l2norm is fused INSIDE another kernel:

* `decode` (`gdn_triton.py:153-167`) — `fused_sigmoid_gating_delta_rule_update`
* `packed_decode` (`gdn_triton.py:46-136`) — packed decode kernel
* `target_verify` (`gdn_triton.py:201-241`) — same recurrent kernel
* ReplaySSM verify / target_verify (`gdn_backend.py:700-836`)

### Performance envelope

* **Kernel-count reduction.** Removes 1 launch/GDN-layer/prefill on the
  chunk path. Qwen3.5-4B has 24 GDN layers; Stage-3 recorded ~10
  prefills/trace with ~4,176 per-kernel launches counted, so the fused
  helper eliminates **~4,176 kernels per trace** — measurable via
  `nsys stats --report cuda_gpu_kern_sum`, the same extractor used in
  Stage-3.
* **Wall-clock reduction.** Stage-3's per-launch amortized cost is
  ~3 μs (30 ms / 9,300 launches). Fusion saves ~0.5-2 ms of prefill
  wall-clock per request — **below the Stage-3 within-cell rep
  variability of 0.36 %** and roughly 0.05 % of e2e. Not distinguishable
  from noise on end-to-end measurement.
* **BCG replay reality.** Removing one kernel *node* from a captured
  graph does not remove one `cudaGraphLaunch`; it saves the node's
  in-replay dispatch (~1-3 μs on H100) plus the kernel's GPU time. This
  matches the arithmetic — no hidden multiplier.

---

## What remains uncertain

1. **Actual measurable e2e win: none expected.** The perf reviewer is
   correct that 0.5-2 ms/req is below noise. If the goal is a
   user-visible latency reduction, F1 will not deliver it. If the goal
   is an incremental, clean upstream PR that removes a demonstrable
   number of launches and clears the ground for follow-on fusion, F1
   is the correct pick.
2. **Upstream FLA receptiveness.** The file is annotated
   `# Adapt from https://github.com/fla-org/flash-linear-attention/blob/main/fla/modules/l2norm.py`
   (`l2norm.py:1`). Upstreaming to FLA lets SGLang inherit at next sync
   with zero drift, but FLA maintainers' response time is unknown.
   Dual-track (SGLang-local first, upstream in parallel) mitigates
   this.
3. **Interleaved-grid register pressure.** At num_warps=8 the fused
   kernel materializes both q-block and k-block pointers per program.
   Should be a compile-time constexpr branch on `pid1` so only one path
   is emitted per program, but this needs to be verified when the
   kernel is written — a small microbenchmark on kernel latency (fused
   vs 2× split) will catch any regression.

---

## Expected benefit — honest summary

| Metric | Expected | Detection method |
|---|---|---|
| Launches removed / trace (Stage-3 rig) | ~4,176 (1 launch × 24 layers × ~10 prefills × 17.4 replays/layer) | nsys `cuda_gpu_kern_sum` diff, matches Stage-4 methodology |
| Prefill wall-clock / request | −0.5 to −2 ms (0.03-0.07 % of e2e) | Below Stage-3's 0.36 % noise floor; not detectable |
| Per-launch kernel time | ~3-5 μs (savings for one fused launch vs two) | Isolated microbenchmark, not e2e |
| Code footprint | ~50 LOC in `fla/l2norm.py` + 1-2 line change at 5 call sites | Diff review |
| Upstream PR complexity | Low; matches FLA's existing 3→2 kkt+solve fusion pattern | — |

---

## Risks

**Very low overall.** Enumerated:

* **Correctness (numerical).** Very low. Bit-exact under the identical-tile,
  same-eps design; regression test can catch any deviation.
* **Correctness (shape/stride).** Low. Q and K are guaranteed same shape by
  the model config. Prefer the general (Option-b) fused kernel signature
  that accepts separate T pointers, so a future GQA variant on this
  path is a silent no-regression rather than a silent shape bug.
* **Regression on adjacent kernels.** Very low. The fusion is a
  local kernel replacement, not an ordering/scheduling change.
* **Register pressure at num_warps=8.** Low. Compile-time branch on
  `pid1` keeps per-program register footprint at parity with the
  single-tensor kernel. Verify with `TRITON_INTERPRET` and a smoke
  benchmark before landing.
* **Overselling in the PR description.** Real. The wall-clock story is
  a null result under this measurement setup. PR text should lead with
  launch-count reduction and the fusion-pattern-precedent, not e2e ms.

---

## Recommended validation sequence

Ordered milestones for Stage 2. Each has a commit + push and a
milestone report. Stop and clean up if any milestone yields
`EXEC_REJECT`.

1. **Hot-path confirmation.** Re-run Stage-4 kernel diff on a fresh
   A1 capture; confirm `l2norm_fwd_kernel` count is 8,352 ± noise at
   `p=128 b=1` on the frozen SHA. If not, investigate before touching
   code.
2. **Prototype fused kernel in a dev checkout of SGLang** (NOT the
   frozen `sglang_checkout`; branch off `main` of a working copy) —
   `l2norm_fwd_pair(q, k)` helper only, no call-site change yet.
3. **Bit-exact correctness test.** Numerical parity vs
   `(l2norm_fwd(q), l2norm_fwd(k))` on the shapes exercised in prod
   (T ∈ {1, 64, 128, 4096}; H=16; K=128; bf16 and fp16). Assert
   element-wise equality.
4. **Kernel-latency microbenchmark.** Compare fused vs 2×split on
   the D≤512 kernel; expect ≥30 % launch-overhead reduction per pair
   on H100. If regression, stop and diagnose (register pressure,
   occupancy).
5. **Integrate call-site change at `fla/chunk.py:108-110` first.**
   Re-run the Stage-3 A0/A1 harness on `p=128 b=1` and confirm
   `l2norm_fwd_kernel` count halves on A1. Wall-clock delta expected
   to be null (within noise) — that is the honest expected result.
6. **Extend to the other 4 production call sites** (kda.py,
   gdn_flashinfer, gdn_cutedsl, kda_cutedsl). Regression-test each.
7. **Decide: keep or revert.** Keep if all correctness tests pass and
   the kernel-count reduction is confirmed at the exact predicted
   magnitude (no surprises). Revert if any correctness test fails or
   if kernel-latency microbenchmark shows a regression larger than the
   launch-overhead saving.
8. **Upstream to FLA in parallel** (independent PR, no dependency on
   the SGLang change landing first).

---

## Caveats & scope adjustments to the plan

Three amendments to the plan as originally described in
`optimization_design.md`:

1. **Extend fusion to all 5 production call sites, not just
   `fla/chunk.py:108-110`.** The audit found `kda.py:1155-1156`,
   `gdn_flashinfer.py:295-296`, `gdn_cutedsl.py:133-134`, and
   `kda_cutedsl.py:121-122` share the identical paired pattern.
   Amortizing the helper across all 5 sites improves the
   value/cost ratio at minimal marginal effort.
2. **Prefer general (Option-b) kernel signature** — pass `(x_q, y_q,
   T_q, x_k, y_k, T_k, D, eps, BT, BD)` and use `tl.constexpr`-branched
   `pid1` to pick the pair. Handles any future variant with differing
   T without a rewrite; costs 4 extra kernel args over the tight
   Option-a form.
3. **Frame the PR as "launch-count reduction under CUDA-graph
   replay"** — not "prefill speedup". The e2e wall-clock win is a
   null result at this measurement resolution and honesty preserves
   upstream trust. Cite the existing 3→2 fusion at
   `chunk_fwd.py:349-357` as precedent.

---

## Alternatives considered and why F1 is still the pick

* **Rank 4 — `wy_fast.py` autotune enablement.** Zero launch-count
  change but potentially non-trivial per-launch improvement. Unknown
  magnitude; the perf reviewer suggested this could beat F1. Correct
  in principle, but "could" is not evidence; F1 has a known, provable
  kernel-count reduction. Autotune enablement should be a **separate
  parallel PR** — no reason to trade one for the other.
* **F3 — fold l2norm into `fused_qkv_split_gdn_prefill_kernel`.**
  Saves 2 launches instead of 1. Higher effort (5-8 days vs 1-2) and
  higher risk (touches an SGLang-owned Triton kernel with a wildly
  different access pattern — currently 1-D grid with
  `BLOCK_SIZE = next_power_of_2(qkv_dim)`). Worth pursuing as a
  **follow-on** once F1 is landed and the l2norm_fwd_pair helper is
  battle-tested. F3 without F1 forecloses on the reusable helper path.
* **Bucketized-N recurrent varlen (Deliverable 3).** Excluded — 3-6
  weeks, high risk, motivated by 0.6-1.6 % wall-clock. Not a
  standalone PR candidate.

---

## Final recommendation

**Proceed to Stage 2. Execute F1 with the 3 scope adjustments above.**

The plan is technically clean, numerically safe, and integrationally
low-risk. The honest wall-clock ceiling is small, but the
launch-count reduction is provable and matches a precedent FLA
already established. Expanded to all 5 call sites, the fix has a
better value/cost ratio than the design report claimed. Upstream
route is FLA-first with a local SGLang wrapper as a fallback if
upstream is slow.

Do not oversell the wall-clock result. Do land the fix.
