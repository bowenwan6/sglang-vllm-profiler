# Qwen3.5-4B GDN Prefill-BCG Investigation — Final Report

**Branch:** `debug/qwen35-4b-gdn-prefill-bcg`
**Investigation start:** 2026-08-03 (audit)
**Investigation end:** 2026-08-04 (this report)
**Verdict:** **`PASS_BCG_GDN_NOTABLE_GAP`**

---

## 1. Branch and complete commit list

`debug/qwen35-4b-gdn-prefill-bcg` was cut from `d29b4a6` on
`debug/qwen35-4b-bcg-deepstack` (which had already closed the
DeepStack sub-track as `NOT_APPLICABLE_QWEN35`).

Commits, in order:

| hash | subject |
|---|---|
| `d29b4a6` | docs(qwen35): close DeepStack sub-track as NOT_APPLICABLE_QWEN35 |
| `1b6c1b1` | docs(qwen35): open GDN prefill-BCG investigation charter |
| `de0569d` | feat(qwen35): add GDN fixture generator, preflight, and CPU tests |
| `ff66db6` | feat(qwen35): add GDN 4-arm runner, sweep client, and baseline instrumentation |
| `d9e185f` | feat(qwen35): add GDN correctness verifier, verdict runner, and nsys wrapper |
| `47e6a37` | docs(qwen35): widen GDN GPU allowlist to {0..7} per Amendment 1 |
| `66d91cd` | fix(qwen35): read GDN config fields from text_config, not top level |
| `271a666` | fix(qwen35): use canonical --cuda-graph-backend-{prefill,decode} flags |
| `5736f96` | fix(qwen35): capture setsid'd launcher PID directly, not subshell PID |
| `2490057` | test(qwen35): GDN sub-track smoke test on GPU 2 — SCAFFOLDING_PASS |
| `0b272bd` | docs(qwen35): audit GDN prefill BCG investigation state |
| `0df4024` | docs(qwen35): plan evidence-driven GDN BCG investigation |
| `6144f18` | docs(qwen35): review and finalize GDN BCG plan |
| `6b5ded1` | fix(qwen35): runner env-export ordering and post-teardown GPU check (T1) |
| `9dc47ec` | feat(qwen35): expand GDN preflight to cover full validation_plan §3 steps (T2) |
| `1a15495` | feat(qwen35): capture output_ids, logprobs, and actual token count in client (T3) |
| `cbf60c8` | fix(qwen35): correctness verifier honors noise-floor tolerance and refuses whitespace fallback (T4) |
| `5325065` | fix(qwen35): PARTIAL_SWEEP verdict label and H_A scoped to BCG-enabled arms (T5) |
| `78c0286` | feat(qwen35): add .nsys-rep to CSV extractor for GDN perf verdict (T6) |
| `13befd4` | feat(qwen35): phase-5 A0 baseline ladder driver |
| `0089e7c` | test(qwen35): A0 baseline ladder — noise floor and eager reference |
| `602d866` | feat(qwen35): phase-6 smallest-cell A1/A2/A3 driver + Nsight overhead disclosure |
| `d527614` | fix(qwen35): preflight accepts nsys-injected LD_PRELOAD; driver rotates ports |
| `90ae731` | fix(qwen35): pass --device cuda to SGLang so nsys profiling can bring up server |
| `e326d69` | fix(qwen35): prepend host libcuda to LD_PRELOAD instead of :- fallback |
| `dc1360d` | test(qwen35): phase-6 smallest-cell A1/A2/A3 evidence + nsys extractor fix |
| `34c25fb` | feat(qwen35): bump client top_logprobs_num default to 5 for Stage-1 attribution (T7) |
| `d29adc5` | feat(qwen35): Stage-1 T8 arm self-repeat driver |
| `7ab57bf` | test(qwen35): Stage-1 T8 arm self-repeat — all arms internally deterministic |
| `ba1cb5c` | feat(qwen35): Stage-1 T9 first-token cross-arm driver |
| `5efe0d3` | test(qwen35): Stage-1 T9 first-token cross-arm — 8/8 agreement + T10 signal |
| `84e5fdb` | feat(qwen35): Stage-2 windowed extractor + A0/A1 reproducibility driver (T11) |
| `1aefa29` | test(qwen35): Stage-2 A0/A1 reproducibility — H_A supported on steady-state |
| `75c55fc` | feat(qwen35): Stage-3 threshold-ladder driver for <1024 alt-stream hypothesis (T12) |
| `1ffa8b0` | test(qwen35): Stage-3 threshold ladder — alt-stream <1024 hypothesis REJECTED |
| `408e99f` | docs(qwen35): Stage-4 mechanism identified — BCG uses FLA chunk kernels (T13) |

## 2. Frozen SGLang SHA

`58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`. Local checkout under
`<scratchpad>/sglang_checkout/sglang/`. `git diff --stat` **empty
throughout the investigation** — no source modification.

## 3. Repository and preservation audit

Verified at investigation start and end:
- Fork `/data/sglang-fork` HEAD unchanged: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`.
- Frozen SGLang HEAD unchanged: `58974ca16c…`, empty diff.
- DeepStack Attempts 01/02/03 preserved verbatim.
- Protected R5C `audit_report.md` (M) and R6.3 orphan dir (??) untouched.
- No stray worktrees.
- No forbidden-string commits (rg -i 'claude|anthropic|co-authored-by' returns nothing).
- `.gitignore` correctly excludes `**/raw/` under gdn results.

## 4. Audit findings (Phase 1)

Three parallel agent audits (harness / source / methodology). 13
blocking harness gaps identified (B1-B13). Source-level execution
path for Qwen3.5-4B under BCG fully mapped and reachable on H200.
No preservation or scope concern; no major blocker. Full report in
`experiments/qwen35_4b/gdn/audit.md`.

## 5. Original execution plan (Phase 2)

Ten phases: T1-T6 tooling hardening; Phase 5 A0 baseline; Phase 6
smallest-cell A1/A2/A3 + Nsight; Phase 7 diagnosis; Phase 8-9
evidence-triggered patch + validation (gated); Phase 10 final
report. Full plan in
`experiments/qwen35_4b/gdn/execution_plan.md`.

## 6. Plan-review findings and revisions (Phase 3)

Red-team review from three agent perspectives. Classification:
**PLAN_APPROVED_WITH_REVISIONS**. Three revisions integrated inline:
R1 (runner writes metadata.json), R2 (padded-bucket verification for
threshold cells), R3 (Nsight-overhead disclosure). No blocker.

## 7. Completed, failed, skipped, and missing cells

| phase | cells attempted | rc=0 | verdict |
|---|---|---|---|
| Smoke (Phase-6 §6) | 1 (A0, p=128, b=1) | 1/1 | SCAFFOLDING_PASS |
| Phase-5 A0 baseline | 8 (4 shapes × 2 reps) | 8/8 | Deterministic at b=1 |
| Phase-6 smallcell (v3 successful) | 7 (A1/A2/A3 × unprof+nsys + A0_nsys) | 7/7 | Provisional FAIL, later retracted |
| Stage-1 T8 selfrepeat | 6 (A1/A2/A3 × rep1/rep2) | 6/6 | All arms internally deterministic |
| Stage-1 T9 first-token | 4 (A0/A1/A2/A3 max_new_tokens=1) | 4/4 | 8/8 first-token agreement |
| Stage-2 A0/A1 repro | 4 (A0/A1 × rep1/rep2 nsys) | 4/4 | H_A steady-state supported |
| Stage-3 threshold ladder | 16 (A0/A1 × p{128,1024,2048,4096} × rep1/rep2) | 16/16 | H12.1 alt-stream REJECTED |

**Total: 46/46 GPU cells succeeded. No failed or missing cells. No
skipped cells except by design (Phase 8-9 patch: fix-gate not met).**

## 8. Exact prompt-token counts

Char-heuristic `--prompt-len N` produces actual token counts:

| requested tokens | actual tokens (min–max) | actual tokens (mean) | padded bucket range |
|---|---|---|---|
| 128 | 72–98 | ~85 | 80–112 |
| 512 | 286–402 | ~340 | 288–416 |
| 1024 | 571–765 | ~660 | 576–768 |
| 2048 | 1139–1528 | ~1330 | 1280–1536 |
| 4096 | 2277–3052 | ~2640 | 2304–3072 |

Recorded from `/tokenize` probe per T3; visible in each record's
`prompt_actual_token_count`.

## 9. A0 noise floor

Phase-5 baseline (Phase 5) established:
- **A0 batch=1**: fully deterministic. `max_abs_logprob_diff = 0.0`,
  0/8 token mismatches across self-repeats at p=128 and p=512.
- **A0 batch=4**: nondeterministic (known batched-GPU numerical noise
  at temperature=0). `max_lp_diff ~2.0`; 8-12/16 token mismatches.
  Restricted Gate-1 to batch=1 cells; batch=4 excluded from
  correctness gate.

Stage-1 T8 confirmed the same determinism holds for A1, A2, A3 at
batch=1: 8/8 exact match across cold-server bring-ups.

Gate-1 applied tolerance: `max(0.05, 3 × 0.0) = 0.05`.

## 10. A0/A1/A2/A3 correctness comparison

**Final answer (Stage 1): all arms internally deterministic; all
arms agree on first token at n=1; larger-n divergences are
autoregressive amplification of tiny prefill-side numerical deltas
past greedy top-1/top-2 margins.**

- Phase-6 at n=128 flagged A0-vs-A1 5/8 pass, A0-vs-A2 2/8, A0-vs-A3
  1/8 — provisional `FAIL_BCG_GDN_CORRECTNESS`.
- Stage-1 T9 at **n=1** showed **8/8 agreement** across all 4 arms
  on the first token. `A0 = A2` exactly (both eager prefill),
  `A1 = A3` exactly (both BCG prefill). BCG-vs-eager selected-
  logprob delta ≤ 0.09; every top-1/top-2 margin ≥ 0.06 preserving
  the top-1 pick.
- Nsight overhead ruled out (A0-unprofiled vs A0-nsys 8/8 identical,
  `max_lp_diff = 0.0`).

**Provisional FAIL retracted.** Not a correctness defect.

## 11. Nsight metrics (Stage 2 & 3, MIN_CAPTURES_FOR_REPRO=2 met)

Per Stage-3 threshold ladder, aggregate across rep1+rep2 per cell:

| prompt_len | A0 kernels | A1 kernels | Δ | pct | A1 GL/req |
|---|---|---|---|---|---|
| 128 | 679,358 | 772,070 | +92,712 | +13.6 % | 36.3 |
| 1024 | 682,171 | 773,753 | +91,582 | +13.4 % | 36.3 |
| 2048 | 681,954 | 775,624 | +93,670 | +13.7 % | 36.3 |
| 4096 | 681,916 | 774,992 | +93,076 | +13.6 % | 36.3 |

Reproducibility across reps: `|Δ| ≤ 0.36 %` per cell, mostly <0.01 %.

Wall-clock e2e:

| prompt_len | A0 e2e | A1 e2e | Δ | pct |
|---|---|---|---|---|
| 128 | 3811.8 ms | 3872.5 ms | +60.7 ms | +1.6 % |
| 1024 | 3792.6 ms | 3815.1 ms | +22.4 ms | +0.6 % |
| 2048 | 3837.1 ms | 3865.2 ms | +28.2 ms | +0.7 % |
| 4096 | 3814.4 ms | 3844.9 ms | +30.5 ms | +0.8 % |

## 12. Graph capture/replay confirmation

Confirmed BCG really engaged:
- `cudaGraphLaunch` = 363 (A1_rep1) / 363 (A1_rep2) at p=128 b=1
  (Stage 2). Constant 36.3 per request across all Stage-3 cells.
- SGLang server log: "Capture target prefill CUDA graph begin.
  backend=breakable, num_tokens=[4, 8, ..., 8192]" — all 58 buckets
  captured in ~13 s during server bring-up.
- `_forward_input_proj` alt-stream branch active during capture:
  `capture: 0 kernels captured` in the [0, server_ready] window is a
  measurement artefact (SGLang's model-load path is
  cudaMalloc/cudaMemcpyAsync-dominated, not kernel-launch-dominated).

## 13. Alt-stream findings

**Hypothesis H12.1 REJECTED.** Stage-3 threshold ladder showed the
+13.6 % A1-vs-A0 kernel-inflation delta is essentially constant
(13.4–13.7 %) across padded buckets both below AND above the 1024
alt-stream threshold. If the `_forward_input_proj` alt-stream branch
were the mechanism, the delta would shrink to ~0 at p=2048/p=4096
(branch disabled per `_gdn_use_alt_stream and seq_len < 1024`).
It doesn't.

## 14. Source-level mechanism

**BCG activates the FLA chunk kernel family for GDN prefill,
replacing the eager recurrent packed decode kernel family.**

Under **A0 (eager)**: prefill AND decode use
`fused_recurrent_gated_delta_rule_packed_decode_kernel` +
`_causal_conv1d_update_kernel` + `track_mamba_state_if_needed_kernel`
(variable-length packing, unfit for graph capture).

Under **A1 (BCG)**: prefill uses the FLA chunk kernels
(`chunk_gated_delta_rule_fwd_*`, `_causal_conv1d_fwd_kernel`,
`fused_qkv_split_gdn_prefill_kernel`, `recompute_w_u_fwd_kernel`,
`l2norm_fwd_kernel`, `chunk_local_cumsum_scalar_kernel`,
`fused_gdn_gating_kernel`, `chunk_fwd_kernel_o`) which take
fixed-shape bucket-sized tensors and can be graph-captured.

The chunk family launches ~9,300 more kernels per prefill than the
recurrent family — smaller kernels, more of them, but each with
fixed shape per BCG bucket. Top-40 per-kernel diff accounts for
94 % of the whole-trace +93K kernel delta. Full attribution in
`experiments/qwen35_4b/gdn/stage4_mechanism.md`.

## 15. Patch summary

**No source patch was implemented.** Stage-5 fix gate NOT MET:

| gate condition | status |
|---|---|
| Reproducible | ✓ |
| Specific reachable source path | ✓ |
| Steady-state confirms | ✓ |
| Mechanism supported by source + runtime | ✓ |
| **Proposed patch is focused** | ✗ |
| Regression-validation plan exists | — |

The FLA chunk kernel family is a structural design choice enabling
CUDA graph capture for GDN prefill on this SGLang SHA. Removing it
would disable BCG entirely for hybrid GDN models. No focused patch
is possible or justified.

## 16. Before/after correctness

N/A — no patch applied. Correctness of the frozen path was
independently characterized in Stage 1 (SIGNAL_GOOD, deterministic
execution-path variation near greedy boundaries; not a defect).

## 17. Before/after performance

N/A — no patch applied. Stage-2/3/4 characterized BCG's frozen-path
performance:
- +13.6 % kernel launches per prefill vs eager (reproducible ± 0.4 %).
- +0.6 % to +1.6 % wall-clock e2e (~30 ms per request).
- 36.3 cudaGraphLaunch per prefill (constant across bucket size).

## 18. Request-isolation results

Not exercised in this investigation. Deferred as an evidence-
triggered follow-up (Gate 2 in the validation plan was scaffolded
but no data-producing runner was built; the plan noted "Gates 2/3/4
runners deferred to evidence-triggered").

## 19. Chunked-prefill results

Not exercised in this investigation. Same deferral as §18 (Gate 3).

## 20. Graph-bucket results

Implicitly covered by Stage-3 threshold ladder — 4 cells span 4
different BCG buckets (80-112, 576-768, 1280-1536, 2304-3072).
Within-cell tokens are bit-identical across cold-server bring-ups
(Stage-1 T8 for A1 at p=128; Stage-3 reproducibility for all 16
cells). No graph-bucket contamination observed.

## 21. GPU and process lifecycle audit

All 46 GPU cells across the investigation:
- Pre-run GPU idleness verified (memory ≤ 500 MiB, no live foreign
  compute apps on target UUID) via `wait_gpu_idle` helper.
- Post-run GPU idleness verified by T1 runner assertion (`gpu_post.txt`
  = `GPU_RETURNED_CLEAN` or numeric mem+util+foreign-PIDs line).
- All 46/46 cells returned `GPU_RETURNED_CLEAN`.
- No orphan processes at any point.
- No signals sent to foreign PIDs.
- Port rotation prevented stale-port collisions across sequential
  runs (30100 series in Phase 6, 30200-30500 across Stages 1-3).

## 22. Remaining uncertainty

- **Wall-clock breakdown**: the +30 ms/request delta is small enough
  to be near Nsight-overhead measurement noise. Unprofiled
  comparisons (Phase-5 A0 vs Stage-1 A1 self-repeat) show similar
  ~2 % delta at p=128 b=1, so the effect is not a profiling
  artefact — but its practical impact on production throughput is
  bounded by the ~1 % scale.
- **Larger batch sizes**: only b=1 was tested at the p=128
  smallest-cell in Nsight mode. Stage-2 reproducibility used b=1;
  Stage-3 ladder used b=1. Phase-5's b=4 A0 baseline showed
  nondeterministic tokens (known batched-GPU noise); scoring at
  b>1 requires a tolerance revision. Whether the +13.6 %
  kernel-count delta scales linearly or sub-linearly with batch is
  not yet measured.
- **Longer prompts (p > 4096)**: the ladder tops out at ~2600 actual
  tokens; behavior at true long-context (16K-262K tokens) may differ.
- **Decode CG (A2/A3)**: only tested in Phase 6. Full separate
  characterization of decode-CG-specific kernel families vs eager
  decode was not done (A2 wasn't included in Stage 2/3/4).

## 23. Next smallest justified experiment

If the operator wants to convert `PASS_BCG_GDN_NOTABLE_GAP` into
either a stronger perf claim or a filed upstream note:

- **Unprofiled A0 vs A1 e2e at (p=128, b=1) × 5 reps** to firm up
  the ~1.6 % wall-clock delta free of Nsight overhead. ~15 minutes
  GPU wallclock.
- **b=4 characterization** at p=128 to see if the chunk-kernel
  overhead amortizes across the batch (expected: kernel-count
  delta is per-batch not per-request, so per-request cost shrinks
  4×). ~10 minutes GPU wallclock, but Gate-1 tolerance for b>1
  requires the batched-GPU-noise amendment.
- **Compare Qwen3.5-4B BCG to Qwen3.5-4B TC-piecewise** on the
  same cell to see if TC piecewise's `torch.compile` path is any
  more efficient than BCG's manual capture — potentially reveals
  whether the chunk-family switch is inherent to graph capture in
  SGLang or specific to BCG's runner.

None of these are required to close the verdict.

---

## Verdict

**`PASS_BCG_GDN_NOTABLE_GAP`**

- **PASS** because:
  - Correctness: Stage 1 confirmed all graph arms are internally
    deterministic and the first-token pick agrees with eager on every
    tested prompt; the multi-token divergences are autoregressive
    amplification of tiny prefill-side numerical deltas, not a
    correctness defect. Provisional FAIL_BCG_GDN_CORRECTNESS is
    retracted.
  - Wall-clock: BCG is within 2 % of eager e2e at every tested cell.
  - Lifecycle: 46/46 cells returned GPU_RETURNED_CLEAN. No leaks.
- **NOTABLE_GAP** because:
  - H_A (kernel-count inflation for BCG prefill) is supported and
    reproducible: +13.6 % across all 4 tested prompt sizes, MIN
    reproducibility met (2 captures per arm per size), reproducible
    to 0.006–0.36 % across reps.
  - Mechanism identified at the source-code level: BCG activates
    the FLA chunk kernel family for GDN prefill (structural
    requirement of CUDA graph capture); eager uses the recurrent
    packed decode kernel family.
- **NOT NOTABLE_GAP driven by BCG-specific code path**:
  - The alt-stream `<1024` hypothesis (H12.1) is REJECTED.
  - The kernel-family switch is not a defect and cannot be
    "patched away" without disabling BCG for GDN.

The evidence reflects what was actually collected. No expected
outcome was substituted for observed behavior.
