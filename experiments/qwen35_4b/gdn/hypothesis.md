# Hypothesis and Acceptance Criteria — Qwen3.5-4B GDN Prefill-BCG

> Established facts vs source-level observations vs runtime hypotheses
> vs predeclared acceptance criteria. Mirrors the structure of the
> closed DeepStack sub-track's `../hypothesis.md` so future readers
> can navigate consistently.

## 1. Established facts (source-level only)

Every item is verifiable in [`source_audit.md`](source_audit.md)
against upstream SGLang `main` at frozen SHA
`58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`.

- **[F1]** `Qwen/Qwen3.5-4B` is a hybrid stack: some decoder layers
  are `Qwen3_5AttentionDecoderLayer`, others are
  `Qwen3_5LinearDecoderLayer` (whose `linear_attn` submodule is
  `Qwen3_5GatedDeltaNet`). Layer selection is per-layer via
  `config.layers_block_type[idx]`
  (`models/qwen3_5.py:1225-1228, 1359-1373`).
- **[F2]** `Qwen3_5GatedDeltaNet.forward` runs five ordered stages
  (`models/qwen3_5.py:620-685`): input projection (two parallel
  `ColumnParallelLinear`s, optionally on alt-stream), fused
  split/reshape, `RadixLinearAttention` call, `RMSNormGated`,
  output projection.
- **[F3]** The alt-stream branch in `_forward_input_proj`
  (`models/qwen3_5.py:551-585`) is gated on `get_is_capture_mode()`,
  `seq_len < DUAL_STREAM_TOKEN_THRESHOLD (= 1024 outside TC
  piecewise, 0 under TC piecewise)`, and `_gdn_use_alt_stream`.
  **No explicit short-circuit exists for BCG**; the alt-stream
  branch is therefore active on short prefills captured under BCG.
- **[F4]** `RadixLinearAttention` (`layers/radix_linear_attention.py`)
  is a stateful primitive. Its state (mamba-style KV,
  `conv1d` weights, `A_log`, `dt_bias`) lives in the mamba
  checkpoint pool (`mem_cache/mamba_checkpoint_pool.py`), not in
  the BCG buffer registry.
- **[F5]** The BCG buffer registry (`cuda_graph_buffer_registry.py:867-877`)
  registers `input_embeds` for multimodal models only; **no GDN-
  specific slot** (no slot for mamba state, `A_log`, `dt_bias`, or
  any GDN intermediate) is registered.
- **[F6]** `Qwen3_5ForConditionalGeneration` and
  `Qwen3_5MoeForConditionalGeneration` are on the multimodal BCG
  allowlist (`configs/model_config.py:1845-1848`). They are **not**
  on the piecewise-CUDA-graph allowlist (`1836-1841`).
- **[F7]** Full-decode CUDA-graph is a distinct capture code path
  (`cuda_graph_runner.py`, one-token replay); the four-arm matrix
  in `validation_plan.md` isolates it from prefill BCG.

## 2. Source-level observations that are not yet runtime evidence

- **[O1]** Because [F3] holds, GDN's alt-stream branch is active
  inside prefill BCG capture at `seq_len < 1024`. Cross-stream
  syncs (`alt_stream.wait_stream`, `torch.cuda.stream`) get baked
  into the captured graph and may not replay with the same
  overlap the eager path achieves — this is a candidate for the
  "measurable launch-overhead" the pivot brief asks the profile
  to identify.
- **[O2]** Because [F5] holds, `RadixLinearAttention`'s state
  pointers are not stabilised by the BCG buffer registry. If the
  mamba pool reuses stable slots across requests, correctness is
  preserved by that pool's own contract; if not, BCG replay may
  read from stale state. This is an open question resolvable only
  by direct observation on the mamba pool at replay time.
- **[O3]** Because [F6] and the DeepStack sub-track's `NOT_APPLICABLE_QWEN35`
  closure both hold, prefill BCG is enabled for `Qwen/Qwen3.5-4B`
  on the shipped configuration — no monkey-patch is required.
- **[O4]** The fused kernel `fused_qkvzba_split_reshape_cat_contiguous`
  is not always used (falls back to Python split/reshape/cat if
  `num_v_heads/num_k_heads ∉ {1,2,4}`); the fallback path has
  many small launches. Whether Qwen3.5-4B hits the fused path is
  a config-derivable fact (see `provenance.md` §2) that the
  audit records at run time.

## 3. Runtime hypotheses (unverified — this is the point of the sweep)

Enumerated so the validation plan predeclares what it must
disprove or confirm.

- **[H_A]** On `Qwen/Qwen3.5-4B` under BCG-enabled prefill (arm
  `A1` or `A3`), Nsight shows GDN layers dispatch strictly more
  kernels per prefill token than the eager reference (arm `A0`)
  by a measurable and repeatable margin at at least one
  (prompt_len, batch) cell, attributable to at least one op from
  the GDN op focus list in `source_audit.md` §3.
- **[H_B]** Under BCG-enabled prefill, at least one GDN op falls
  **outside** the captured graph (visible in Nsight as a
  host-launched call between two `cudaGraphLaunch` entries within
  a single prefill), producing a repeated graph break.
- **[H_C]** GDN's alt-stream cross-stream sync (op 3.3 in
  `source_audit.md`) shows a measurable CPU launch gap under BCG
  that is absent (or smaller) under eager.
- **[H_D]** No BCG-specific effect on GDN kernel counts, launch
  gaps, or graph-break count; the observed TTFT/throughput
  difference between arms tracks the standard CUDA-graph capture
  overhead / benefit envelope. GDN is not the bottleneck under
  BCG at any tested cell.
- **[H_E]** GDN produces a **correctness** divergence between
  arms (token mismatch, order-dependent output, chunk-dependent
  output, or bucket-dependent output). This closes the sweep as
  `FAIL_BCG_GDN_CORRECTNESS` and pre-empts any perf claim.

Exactly one of `{[H_A]∨[H_B]∨[H_C], [H_D], [H_E]}` is expected to
hold. The 4-arm sweep with predeclared correctness gates must
distinguish among them with direct evidence, not by elimination.

## 4. Unverified assumptions to be checked at run time

- **[A1]** Current upstream SGLang at the frozen SHA builds and
  serves `Qwen/Qwen3.5-4B` end-to-end on a single GPU in BF16
  with prefill BCG enabled (verified by inspecting server logs
  for `cuda graph: True` on prefill and by the runner's
  per-request instrumentation from the DeepStack sub-track).
- **[A2]** `config.layers_block_type` on the loaded model is
  non-empty and non-degenerate (has both `"attention"` and
  `"linear_attention"` entries), so the hybrid stack is really
  hybrid. Any all-`"attention"` model degenerates the
  investigation to standard-attention-under-BCG (out of scope).
- **[A3]** `Qwen/Qwen3.5-4B` hits the fused
  `fused_qkvzba_split_reshape_cat_contiguous` path
  (`num_v_heads/num_k_heads ∈ {1, 2, 4}`). If it does not, the
  Python fallback becomes the primary op path and the source
  audit's focus list shifts accordingly (recorded, not
  reinterpreted).
- **[A4]** `mp.set_start_method('spawn')` still applies for
  SGLang worker subprocesses, so the DeepStack sub-track's
  `sitecustomize.py` bootstrap remains necessary to propagate
  runner-side instrumentation into the workers.
- **[A5]** Nsight Systems (`nsys`) is available on the target GPU
  host and the SGLang server process can be attached without
  altering its steady-state behaviour beyond a documented Nsight
  overhead.

## 5. Acceptance criteria — machine verdict (predeclared)

The validation runner must produce a verdict of exactly one of the
following. **These verdict labels are the source of truth**; the
implementation and any tooling must use these exact strings.

- **`PASS_BCG_GDN_NOTABLE_GAP`** — Every cell clears all four
  correctness gates (`validation_plan.md` §4) AND at least one of
  `[H_A]`, `[H_B]`, `[H_C]` holds with quantitative Nsight
  support: named GDN op, named cell, named metric that exceeds
  the arm-comparison threshold, reproducible across at least two
  captures. This closes the investigation with a documented BCG
  limitation on GDN. **No upstream fix is implemented on this
  branch.**
- **`PASS_BCG_GDN_NO_GAP`** — Every cell clears all four
  correctness gates AND none of `[H_A]`, `[H_B]`, `[H_C]` holds.
  BCG on GDN is not the bottleneck at any tested cell.
- **`FAIL_BCG_GDN_CORRECTNESS`** — At least one correctness gate
  fails. Supports `[H_E]`. No perf claim is admissible; the
  investigation closes on the correctness gap and pivots to a
  correctness follow-up (or closes as `FEATURE_GAP` if the
  divergence traces to a bounded feature the arm exposes).
- **`AMBIGUOUS`** — Nsight variance is too large to separate
  arms (95 % CI intervals overlap on the arm-comparison metric),
  or correctness gates pass but instrumentation disagrees with
  itself.
- **`INFRA_FAILURE`** — Environment / GPU / preflight failure
  (unchanged from the DeepStack sub-track's `INFRA_FAILURE`
  criteria).

**Any perf number reported without every correctness gate having
passed is a protocol violation, not a finding.** Post-hoc rewrite
of these tiers requires an explicit "Amendment N" block in this
file.

## 6. Explicit rules carried from the pivot brief (2026-08-03)

Recorded here so they are not re-litigated:

- **No upstream source modification** until the baseline profile
  pins one specific BCG limitation.
- **No checkpoint mutation** to force GDN behavior; use
  `Qwen/Qwen3.5-4B` as shipped.
- **Correctness first, perf second**: every arm must clear all
  four correctness gates before its Nsight metrics contribute to
  a verdict.
- **Do not assume recurrent-state handling is faulty.** A
  targeted deep-dive on any specific GDN op is triggered only by
  a repeated graph break or a measurable launch-overhead
  bottleneck in the baseline profile.
- **GPU allowlist `{0..7}`** per `validation_plan.md` Amendment 1
  (2026-08-03); a GPU qualifies only when compute processes = 0,
  memory ≤ 500 MiB, and utilisation ≤ 5 %. Never signal foreign
  PIDs.

## 7. Amendments

### Amendment 2 (2026-08-03) — `PARTIAL_SWEEP` verdict + `H_A` scoping

Landed as Phase-4 T5 of the execution plan, motivated by the audit's
blocking gaps B8 and B9.

- **`PARTIAL_SWEEP` added to the verdict label set** in §5. Defined as:
  "investigation stopped short with insufficient evidence for any
  verdict; at least one predeclared gate was MISSING (never run) but
  no gate returned FAIL". `PARTIAL_SWEEP` is emitted by
  `scripts/gdn_verdict.py`'s `decide()` when `any_missing=True` and
  `any_failed=False` — previously a MISSING gate silently collapsed to
  `FAIL_BCG_GDN_CORRECTNESS`, making a scaffolding gap
  indistinguishable from a real correctness failure. **Precedence**:
  `INFRA_FAILURE > FAIL_BCG_GDN_CORRECTNESS > PARTIAL_SWEEP >
  PASS_BCG_GDN_NOTABLE_GAP > PASS_BCG_GDN_NO_GAP > AMBIGUOUS`. A real
  gate FAIL still wins over PARTIAL_SWEEP (even if other gates were
  missing, one FAIL is decisive evidence).
- **`H_A` scoped to `{A1, A3}` only.** Previously
  `scripts/gdn_verdict.py:_score_perf` iterated `("A1", "A2", "A3")`
  for the kernel-count-inflation test. A2 (`eager_dcg`) uses eager
  prefill, so any GDN-side kernel divergence there cannot be
  attributed to BCG — allowing A2 to trigger `H_A` risked a false
  positive `PASS_BCG_GDN_NOTABLE_GAP`. The scoping now matches §4.1's
  `[H_A]` definition ("under BCG-enabled prefill (arm `A1` or `A3`)").
  Confirmed by `test_verdict_h_a_ignores_A2` in `scripts/test_gdn_scaffolding.py`.
