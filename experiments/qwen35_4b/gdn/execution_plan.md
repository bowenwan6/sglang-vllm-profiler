# GDN Prefill-BCG — Execution Plan

**Date:** 2026-08-03
**Predecessor:** [`audit.md`](audit.md) at commit `0b272bd`.
**Goal:** turn the scaffolded harness into a scored diagnosis with a defensible
verdict, using the smallest experiments that can decide each question. Do not
run the full 64-cell sweep without a clear hypothesis.

## 0. Guiding principles

1. **Correctness gates block performance conclusions.** Every perf number is
   inadmissible until Gate 1 (token/logprob equivalence) passes for the cell
   in question, using a noise-floor–calibrated tolerance.
2. **Smallest distinguishing experiment first.** Baseline = A0 ladder only.
   Comparison = smallest cell only. Escalate only on evidence.
3. **Every tooling patch is a separate commit** with a passing test.
4. **No frozen SGLang source modification** until Phases 5-7 have identified a
   specific mechanism.
5. **Preservation invariants unchanged** (fork read-only at `986c89e69c`;
   frozen SGLang at `58974ca16c` with empty diff; DeepStack attempts verbatim).
6. **GPU pool at execution time: `{6, 7}` idle**; recheck per launch, foreign-
   PID guard on target UUID, PGID-scoped cleanup, no signalling foreign PIDs.

---

## 1. Mandatory tooling work (Phase 4)

Ordered by dependency — each step lands as its own commit with an updated CPU
test and a `git push`.

### T1 — Runner env-export ordering + preflight `mkdir` + post-teardown assertion

Fixes B4, B6, and part of B5. Blocking because every preflight run currently
reports `LD_PRELOAD: null` and `CUDA_VISIBLE_DEVICES: null` despite the runner
setting them, and post-teardown GPU state is not verified.

Files:
- `experiments/qwen35_4b/gdn/scripts/gdn_runner.sh`:
  - Move the three `export …` lines (currently 285-286 and the `PYTHONPATH`
    concatenation at 284) **above** the preflight call at line 227.
  - Add `mkdir -p "$RESULTS_DIR"` before the preflight invocation.
  - After teardown trap fires, append one of two lines to
    `$RESULTS_DIR/gpu_post.txt`:
    - `GPU_RETURNED_CLEAN` if target GPU's memory ≤ 500 MiB, util ≤ 5 %, and no
      compute apps in `nvidia-smi --query-compute-apps=pid,gpu_uuid` for that
      UUID whose PID exists in `/proc`.
    - `GPU_STILL_HOLDS_<N>_MIB` otherwise. Runner exits non-zero (77) so the
      sweep driver treats the cell as failed.

Test:
- `test_runner_dry_run_context_blob_valid_json` extended to assert the reorder
  puts env exports before preflight (via a dry-run marker log line).
- New `test_gpu_post_check_reports_clean_on_empty_gpu` — mock-friendly by
  wrapping the `nvidia-smi` call in a helper so a test-only shim can inject a
  synthetic response.

Commit style: `fix(qwen35): runner env-export ordering and post-teardown GPU check`

### T2 — Preflight coverage (frozen HEAD, sglang.__file__, model revision, LD_PRELOAD, lib versions)

Completes B5. Ensures every scored cell has a provenance snapshot that would
detect drift.

Files:
- `experiments/qwen35_4b/gdn/scripts/gdn_preflight.py`:
  - Add `check_frozen_sglang_head()` — shells `git -C <frozen> rev-parse HEAD`;
    hard-fail on mismatch with pinned `58974ca16c…`.
  - Add `check_sglang_module_file()` — `python3 -c "import sglang;
    print(sglang.__file__)"` under the frozen `PYTHONPATH`; hard-fail if the
    resolved path is not inside the frozen checkout.
  - Add `check_libcuda_preload()` — verify `LD_PRELOAD` targets
    `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`; WARN on mismatch (ABORT
    only under `--strict-env`).
  - Add `check_lib_versions()` — record `torch.__version__`,
    `sgl_kernel.__version__`, `flashinfer.__version__` in the JSON output.
  - Add `check_nsys_version()` — best-effort `nsys --version`; WARN if missing.
  - Add `full_attention_interval` to `REQUIRED_GDN_CONFIG_FIELDS`.
  - Wire model-revision hard-fail into `load_model_metadata()`.

Test:
- `test_preflight_dry_run_reports_skipped` extended to require all six new
  check names in the payload.
- `test_preflight_required_fields_include_full_attention_interval` — asserts
  the constant.

Commit style: `feat(qwen35): expand GDN preflight to cover full validation_plan §3 steps`

### T3 — Client logprob + output_ids + finish_reason + `/tokenize` + hard-fail I/O

Fixes B1a, B12, B13. Unlocks Gate 1.

Files:
- `experiments/qwen35_4b/gdn/scripts/gdn_client.py`:
  - Set `"return_logprob": True` and `"top_logprobs_num": 1` on the request
    body.
  - Extract from server response: `output_ids` (from `meta_info.output_ids` or
    the top-level `output_ids` key), `output_logprobs` (from
    `meta_info.output_token_logprobs` — the list of `[logprob, token_id]`
    tuples SGLang returns), `finish_reason` (from `meta_info.finish_reason`).
    Preserve the raw `meta_info` under a `raw_meta_info` record field for
    audit.
  - Add `POST /tokenize` call before the sweep: single call with each unique
    materialised prompt; record `prompt_actual_token_count` per record.
  - If `/tokenize` unavailable (SGLang has `/get_server_info` but tokenize is
    at `/tokenize`; test at server-ready time), fall back to recording
    `prompt_actual_token_count: null` and set a top-level `tokenizer_source:
    "fallback_char_heuristic"` flag; the noise-floor pilot fails hard if
    fallback is in effect.
  - Replace `contextlib.suppress(OSError)` at line 259-260 with explicit
    error-raise; a JSONL write failure must be a client failure.
  - After `issue_batch`, hard-fail if `len(results) != len(prompts)` (partial
    batch response). Exit code `76`.
  - Add per-request record fields: `prompt_actual_token_count`,
    `tokenizer_source`, `finish_reason`, `output_ids`,
    `output_top_logprobs` (top-1 alt-token id + logprob).

Test:
- New `test_client_captures_logprobs_and_output_ids` — mocks the HTTP call
  with a canned response; asserts every new field is present in the written
  record.
- New `test_client_fails_hard_on_partial_batch_response` — canned mismatch.
- New `test_client_records_tokenizer_source_when_fallback` — canned failure of
  the /tokenize probe.

Commit style: `feat(qwen35): capture output_ids, logprobs, and actual token count in client`

### T4 — Correctness verifier hardening (`--noise-floor`, tolerance formula, fallback hard-fail)

Fixes B1b and B7. Ensures Gate 1 can actually fail.

Files:
- `experiments/qwen35_4b/gdn/scripts/gdn_correctness.py`:
  - Add `--noise-floor FLOAT` CLI arg (default `0.0`); actual tolerance is
    `max(BASE_LOGPROB_TOLERANCE, 3 × noise_floor)` (`BASE = 0.05`).
  - `_token_stream` — when `output_ids` is missing, **return an empty list AND
    set a `used_output_ids: false` flag on the pair**. The pair `verdict` must
    become `FAIL` (not `PASS`) if `used_output_ids` is False.
  - Change `within_logprob_tolerance` semantics: `None` no longer counts as
    within. If `output_logprobs` is absent on either side, the pair fails
    with a `MISSING_LOGPROBS` reason.
  - Compare **all** timed records per `prompt_source_id`, not just the first;
    aggregate to per-prompt PASS/FAIL by requiring all samples to match.

Test:
- Existing tests keep passing.
- New `test_gate1_fails_when_output_ids_missing` — synthetic records with no
  ids; expect FAIL.
- New `test_gate1_fails_when_logprobs_missing` — expect FAIL.
- New `test_gate1_uses_noise_floor_scaled_tolerance` — synthetic with
  `--noise-floor 0.02` and a delta of 0.04 (below `max(0.05, 3×0.02=0.06)`);
  expect PASS. With delta 0.07, expect FAIL.
- New `test_gate1_requires_all_repeat_samples_to_match` — one PASS record and
  one FAIL record for same prompt id → aggregated FAIL.

Commit style: `fix(qwen35): correctness verifier honors noise-floor tolerance and refuses whitespace fallback`

### T5 — Verdict runner: `H_A` scoping + `PARTIAL_SWEEP` label

Fixes B8 and B9. Prevents false positives from A2 and prevents scaffolding
gaps from being labelled as correctness failures.

Files:
- `experiments/qwen35_4b/gdn/scripts/gdn_verdict.py`:
  - Add `PARTIAL_SWEEP` to `VERDICT_LABELS`.
  - `_score_gates` returns `{"partial": True}` when any gate has `overall ==
    "MISSING"` (as distinct from `overall == "FAIL"`).
  - `decide()`: if `partial` is True and no gate has actually failed, return
    `PARTIAL_SWEEP` instead of `FAIL_BCG_GDN_CORRECTNESS`.
  - `_score_perf`: iterate `("A1", "A3")` for `H_A` — not `("A1", "A2", "A3")`.
  - Add per-cell status enum output (`completed | failed | skipped | missing |
    not_yet_run`) as recorded in the sweep-driver summary; verdict runner
    reads it if available.
- `experiments/qwen35_4b/gdn/hypothesis.md`:
  - Amendment 2 (2026-08-03) documenting `PARTIAL_SWEEP` label + `H_A`
    scoping fix.

Test:
- `test_verdict_labels_are_exact_strings` updated with `PARTIAL_SWEEP`.
- New `test_verdict_partial_sweep_when_gate_missing_but_others_pass`.
- New `test_verdict_h_a_ignores_A2` — synthetic perf rows with A2 kernel
  inflation only; expect no `H_A` finding.

Commit style: `fix(qwen35): PARTIAL_SWEEP verdict label and H_A scoped to BCG-enabled arms`

### T6 — Nsight extractor `.nsys-rep → CSV`

Fixes B2. Without this, every scored sweep collapses to `AMBIGUOUS`.

Files:
- `experiments/qwen35_4b/gdn/scripts/extract_nsys_metrics.sh`:
  - Input: one `.nsys-rep` file + `--arm --prompt-len --batch --output-csv`.
  - Output: CSV row(s) with columns per `validation_plan.md §5`:
    `arm, prompt_len, batch, request_id, kernel_count_total,
    kernel_count_gdn, kernel_count_attn, cudagraphlaunch_count,
    cudalaunchkernel_count, ttft_ms, prefill_throughput_toks_per_s,
    p50_launch_gap_us, p95_launch_gap_us, p99_launch_gap_us, graph_breaks`.
  - Implementation: use `nsys stats` reports (`cuda_api_sum`,
    `cuda_gpu_kern_sum`, `nvtx_gpu_proj_trace`) and post-process with `awk`
    or python one-liner.
  - Without NVTX ranges (Phase-4 default), `kernel_count_gdn` /
    `kernel_count_attn` fall back to a `unknown` bucket and the CSV records
    `attribution: coarse`.
  - `graph_breaks` count comes from CUDA API `cudaLaunchKernel` events that
    are **not** enclosed in a `cudaGraphLaunch`.
- `experiments/qwen35_4b/gdn/scripts/nsys_capture.sh`:
  - After the profile completes, call `extract_nsys_metrics.sh` on the
    resulting `.nsys-rep` and write the CSV alongside.
- `experiments/qwen35_4b/gdn/scripts/test_gdn_scaffolding.py`:
  - New `test_extract_nsys_metrics_produces_expected_columns` — dry-run
    against a *fake* `.nsys-rep` marker; assert CLI arg parsing and column
    header emission at minimum. Real extraction validated at first live
    Nsight capture (Phase 6).

Commit style: `feat(qwen35): add .nsys-rep to CSV extractor for GDN perf verdict`

**Deferred to evidence-triggered:**
- **Gate-2 isolation runner** (needs a Gate-1 baseline first).
- **Gate-3 `--chunked-prefill-size` runner** (needs to know whether it's the
  right question; deferred to Phase 7 if H12.1 is inconclusive).
- **Gate-4 bucket-boundary picker** (needs server log parser; deferred).
- **NVTX-tagged instrumentation** (only if coarse metrics leave the mechanism
  ambiguous).
- **Sweep orchestrator** (not needed for the smallest-cell ladder).
- **Real-process lifecycle test** (opportunistic).

---

## 2. Baseline — A0 ladder (Phase 5)

**Objective:** measure eager baseline latency, kernel counts, and noise floor
at each cell shape so Gate-1 tolerance is defensible. Establish that A0 is
deterministic within the accepted threshold.

**Preconditions:** Phase-4 T1-T5 landed; Phase-4 T6 landed but extractor is
best-effort at this stage (baseline needs Nsight to disclose overhead, not to
compute a verdict).

**Target cells (4):**

| Cell | prompt_len_target | batch | Rationale |
|---|---|---|---|
| c1 | 128 | 1 | Smallest — hits the alt-stream branch (padded bucket < 1024). |
| c2 | 128 | 4 | Batch effect at small shape. |
| c3 | 512 | 1 | Larger prompt still in the alt-stream regime. |
| c4 | 512 | 4 | Batch × moderate prompt. |

Deferred: p ∈ {2048, 8192} — expensive, will only run once we have a
specific hypothesis about the >1024 threshold behavior.

**Per-cell protocol:**

1. Verify target GPU idle (compute apps = 0, memory ≤ 500 MiB, util ≤ 5 %).
2. Preflight (T2-hardened) — hard-fail on any mismatch.
3. Launch A0 (`--cuda-graph-backend-prefill=disabled
   --cuda-graph-backend-decode=disabled`), n_warmup=2, n_timed=8.
4. Records include `output_ids`, `output_logprobs`, `prompt_actual_token_count`.
5. **Self-repeat once**: same cell, same GPU, same server bring-up procedure,
   second time. Two record files per cell.
6. Post-teardown GPU idleness check (T1-hardened).
7. Compute per-cell noise floor: max-abs logprob delta and token-equality
   check between the two record files, per prompt.
8. Commit per-cell metadata + records + noise-floor summary.

**Cell wallclock estimate:** ~90-120 s bring-up + 10 requests × ~3 s + teardown
+ repeat = **~5-10 min per cell** × 4 cells × 2 (self-repeat) = **~40-80 min**
for the full baseline.

**Early-stop rules:**
- If any A0 self-repeat token comparison fails on any prompt → stop; the
  baseline is nondeterministic and no BCG comparison is meaningful.
- If preflight hard-fails on any cell → stop; investigate provenance drift.
- If GPU 6 and 7 both become occupied → stop; wait for one to free.

**Deliverable:** `experiments/qwen35_4b/gdn/results/gdn_a0_baseline_<ts>/`
with per-cell subdirs + a top-level `baseline_summary.md` recording the noise
floors and the applied Gate-1 tolerance per cell.

**Commit style:** `test(qwen35): A0 baseline ladder — noise floor and eager reference`

---

## 3. Smallest-cell A1/A2/A3 comparison (Phase 6)

**Objective:** verify Gate 1 (correctness) on the three non-eager arms at the
smallest cell whose actual token count is close to 128, then take one Nsight
capture per arm to see if H12.1 (alt-stream in BCG replay) shows a repeatable
BCG-side signal.

**Preconditions:** Phase 5 complete with A0 self-repeat passing at each cell;
tolerance calibrated.

**Target cell:** cell `c1` from the baseline (prompt=128, batch=1) — actual
token count will be recorded and used to label the cell.

**Per-arm protocol:**

1. Verify GPU idle.
2. Preflight.
3. Launch arm (A1 `bcg_eager`, A2 `eager_dcg`, A3 `bcg_dcg` — separate cold
   server per arm).
4. Confirm from server log:
   - `cuda_graph_backend_prefill` value matches expectation.
   - `cuda_graph_backend_decode` value matches expectation.
   - For BCG arms: `capture_num_tokens` bucket list is present.
   - Warmup completed without error.
5. Client records with logprobs + output_ids.
6. Post-teardown GPU idleness check.
7. Under a fresh server bring-up, take one Nsight capture at the same cell
   using `nsys_capture.sh`; extract CSV via `extract_nsys_metrics.sh`.
8. Post-teardown check again.

**Correctness scoring:** Gate 1 comparison A0 vs A1, A0 vs A2, A0 vs A3, using
the tolerance = `max(0.05, 3 × noise_floor_c1)` established in Phase 5.

**Metrics recorded per arm from Nsight CSV:**

- Token-level: tokens equal (Gate 1 already computed).
- Kernel counts (coarse until NVTX is added).
- `cudagraphlaunch_count` — must be > 0 for BCG arms A1, A3; must be 0 for
  A0, A2 on prefill side.
- `graph_breaks` — `cudaLaunchKernel` events between `cudaGraphLaunch`.
  Expected: 24-32 breaks per BCG prefill (one per GDN layer eager break).
- `p50/p95/p99_launch_gap_us`.
- `ttft_ms`.

**Early-stop rules:**
- Any arm fails Gate 1 → provisional verdict `FAIL_BCG_GDN_CORRECTNESS`,
  reduce to smallest reproducing prompt+batch, commit evidence, then stop
  performance work and diagnose the correctness gap.
- No `cudagraphlaunch_count > 0` for A1/A3 → BCG did not actually engage;
  investigate config resolution before continuing.
- Nsight overhead makes A0-with-nsys diverge from Phase-5 A0-without-nsys by
  more than the applied tolerance → Nsight capture is unreliable at this
  cell; retry with `--sample=none` and reduced trace channels.

**Deliverable:** `experiments/qwen35_4b/gdn/results/gdn_smallcell_<ts>/` with
per-arm subdirs (records + nsys CSV + preflight + gpu_post) and a
`smallcell_summary.md` listing gate outcomes + arm metric comparison.

**Commit style:** `test(qwen35): smallest-cell A1/A2/A3 correctness and coarse Nsight`

---

## 4. Performance diagnosis (Phase 7)

**Objective:** if correctness gates pass, decide whether the alt-stream
hypothesis (H12.1) is supported by evidence, or whether the perf profile is
inside expected CUDA-graph capture overhead.

**Preconditions:** Phase 6 complete with Gate 1 = PASS on all three arms;
Nsight extraction verified.

**Extended cells (up to 4):**

| Cell id | prompt_actual (target) | batch | Purpose |
|---|---|---|---|
| d1 | ~128 | 1 | Reused from Phase 6. |
| d2 | ~512 | 1 | Second short prompt in alt-stream regime. |
| d3 | ~900 | 1 | Just below the 1024-token alt-stream threshold. |
| d4 | ~1200 | 1 | Just above — alt-stream disabled; expect BCG signal to shrink. |

Each cell runs A0 + A1 (or A3) sequentially with Nsight capture on both.
Skip A2 unless a decode-side confounder appears.

**Arm-comparison thresholds** (from `gdn_verdict.py`):
- `H_A` (kernel inflation): mean kernel count on A1/A3 ≥ 1.10 × A0 AND > 2σ.
- `H_B` (graph break): `graph_breaks > 0` under BCG arms on a request whose
  prefill fits one bucket, across ≥ 2 captures. **Note:** in the frozen tree,
  GDN core is eager, so `graph_breaks > 0` is *expected* per prefill (24-32
  breaks). `H_B` needs re-scoping — actual signal is *unexpected* graph
  breaks (i.e. above the layer-count baseline).
- `H_C` (launch-gap): p95 launch gap ≥ 2 × A0's p95.

**Interpretation matrix:**

| Cells d1-d3 | Cell d4 | Interpretation |
|---|---|---|
| BCG arm shows `H_C` support | BCG arm gap shrinks | H12.1 supported: alt-stream branch is the mechanism. |
| BCG arm shows `H_C` support | d4 also shows gap | Mechanism is broader than alt-stream; investigate further. |
| No `H_A/H_B/H_C` support at any cell | — | Verdict `PASS_BCG_GDN_NO_GAP`. Investigation closes. |
| Signal too noisy | — | Consider Phase-4 NVTX addition. |

**NVTX escalation:** add per-op NVTX ranges (10 GDN ops from `source_audit.md
§3`) only if the coarse d1-d4 metrics leave the mechanism ambiguous. NVTX
addition is a new, separately-committed instrumentation swap; the baseline
no-op `gdn_instrumentation.py` is preserved.

**Deliverable:** `experiments/qwen35_4b/gdn/results/gdn_diagnosis_<ts>/` with
per-cell subdirs + a `diagnosis_summary.md` that either supports one of the
three hypotheses or documents insufficient signal.

**Commit style:** `test(qwen35): performance diagnosis at threshold ladder`

---

## 5. Evidence-triggered source patch (Phase 8)

**Gate for opening a source patch:**

- H12.1 (or another mechanism) is reproducible across ≥ 2 Nsight captures per
  cell, on ≥ 2 cells.
- Gate 1 continues to PASS on all arms.
- Alternate hypotheses ruled out (e.g. profiler artefact ruled out by an
  unprofiled repeated comparison).
- The mechanism has a source-level attribution to a specific set of lines.
- The proposed patch is smaller than a broad refactor.

**If gate not met**, do not open a source patch on this branch. Emit
`PASS_BCG_GDN_NO_GAP` or `AMBIGUOUS` per evidence.

**If gate met**, candidate patches to consider (do not preselect):

- **Candidate A** (narrowest): Disable the alt-stream branch during BCG
  capture and replay, mirroring the TC piecewise short-circuit at
  `qwen3_5.py:551-556`. One-line change plus a comment.
- **Candidate B**: Keep alt-stream but make it capture-safe using
  fixed streams + `cudaEventRecord` + `cudaStreamWaitEvent` with correct
  replay dependencies. Larger patch, unknown regression risk.
- **Candidate C**: Add a graph break around the alt-stream region so it runs
  eagerly under BCG while the surrounding projections stay captured.
- **Candidate D**: Something the profile reveals we haven't thought of yet.

Any patch selection requires:
- Exact observed mechanism written up.
- Exact source lines to touch.
- Why the alternative candidates are worse.
- Expected correctness impact + regression risks.
- Validation matrix (Phase 9).

**Commit style:** `docs(qwen35): patch-design memo for alt-stream BCG interaction` +
`fix(sglang, in-repo): <narrow patch>` (in the frozen SGLang checkout,
patched as a `.patch` file under `experiments/qwen35_4b/gdn/patches/`).
**Note:** the frozen checkout itself remains unmodified in git-terms; the
patch is stored as a text artefact, applied at server-launch time.

---

## 6. Validation (Phase 9)

**Preconditions:** patch applied at launch time; baseline + smallest-cell +
extended cells rerun.

**Matrix:**

- Rerun originally slow cell → confirm speedup.
- A0 noise floor rerun → confirm still deterministic.
- A1/A2/A3 smallest-cell rerun → confirm Gate 1 still PASS.
- One additional short prompt.
- One long prompt (p=2048, b=1).
- Batch 1 and 4.
- Chunked vs unchunked prefill — needs Gate-3 runner (deferred T-item).
- Graph-bucket reuse — needs Gate-4 runner (deferred).
- Request-order isolation — needs Gate-2 runner (deferred).

**If any correctness gate regresses** or performance improvement doesn't hold
at a second cell: revert the patch commit (new commit, not `git revert`
history rewrite) and reclassify.

**Commit style:** `test(qwen35): post-patch validation matrix`

---

## 7. Final report + verdict (Phase 10)

Produce `experiments/qwen35_4b/gdn/final_report.md` covering all 23 sections
from the operating model. End with exactly one:

- `FAIL_BCG_GDN_CORRECTNESS`
- `PASS_BCG_GDN_NOTABLE_GAP`
- `PASS_BCG_GDN_NO_GAP`
- `PARTIAL_SWEEP`

Verdict reflects evidence collected, not expected outcome.

**Commit style:** `docs(qwen35): final report and verdict for GDN BCG investigation`

---

## 8. Ordering, commit boundaries, expected artefacts

| Phase | Commits (expected) | Artefacts |
|---|---|---|
| 4 T1 | 1 | runner + preflight + test updates |
| 4 T2 | 1 | preflight coverage + tests |
| 4 T3 | 1 | client + client tests |
| 4 T4 | 1 | correctness verifier + tests |
| 4 T5 | 1 | verdict runner + hypothesis Amendment 2 + tests |
| 4 T6 | 1 | nsys extractor + tests |
| 5 | 4 (one per cell) | per-cell metadata, records, self-repeat records, noise-floor summary |
| 6 | 3 (one per arm) or 1 grouped | per-arm records, nsys CSVs, comparison summary |
| 7 | 1-2 | diagnosis summary + optional NVTX-instrumentation swap |
| 8 | 1 (patch-design memo) + 1 (patch file) | patch memo, .patch file |
| 9 | 1 | validation matrix |
| 10 | 1 | final report |

Every phase transition emits a signal. Every commit is pushed immediately.
Every commit includes an updated `plan.md` §8.1 log entry when the milestone
warrants.

---

## 9. Early-stop rules (across phases)

| Trigger | Action |
|---|---|
| Frozen SGLang `git diff --stat` non-empty | `SIGNAL_BAD`; stop; investigate before doing anything else. |
| /data/sglang-fork HEAD moved | `SIGNAL_BAD`; stop. |
| GPU 6 and 7 both occupied at launch | Poll every ~10 min; if > 2 h with no free GPU, `SIGNAL_AMBIGUOUS`, notify user. |
| A0 self-repeat FAIL | Stop baseline; noise-floor exceeds threshold; diagnose. |
| Preflight hard-fail | Stop; investigate. |
| Gate 1 FAIL on any arm | Verdict `FAIL_BCG_GDN_CORRECTNESS`; commit evidence; stop perf; diagnose correctness. |
| Nsight overhead invalidates a comparison | Reduce trace channels; if still noisy, mark cell inconclusive and continue. |
| Any evidence contradicts the central premise | `SIGNAL_BAD`; stop. |

---

## 10. Rollback strategy

- No amend, reset, force-push, or history rewrite. Ever.
- If a tooling commit introduces a regression, land a new commit that reverses
  the specific change with a clear message; do not `git revert` if the
  original commit contained useful adjacent work.
- Every source-side patch lives as a `.patch` file in
  `experiments/qwen35_4b/gdn/patches/`; the frozen SGLang checkout stays
  git-clean.

---

## 11. Source-modification gate (repeat, prominently)

**Frozen SGLang source is not modified until Phase 8 opens with all
preconditions met.**

Preconditions repeated:
- Runtime evidence identifies a specific mechanism.
- The source path is confirmed (already done, §3 of audit).
- Profiler evidence supports the mechanism across ≥ 2 captures × ≥ 2 cells.
- Alternate hypotheses ruled out.
- Proposed patch is narrow.
- Validation plan can detect regressions.

Before any patch: `SIGNAL_AMBIGUOUS` (before-patch decision, asks user IF the
patch scope is meaningfully large) or `SIGNAL_GOOD` (small internal decision).

---

## 12. Final verdict rules

- `FAIL_BCG_GDN_CORRECTNESS` — any Gate-1/2/3/4 hard failure with a specific
  reproducing case.
- `PASS_BCG_GDN_NOTABLE_GAP` — Gate 1 PASS on smallest cell + at least one
  H_A/H_B/H_C supported on ≥ 2 cells × ≥ 2 captures, with mechanism
  attribution (either coarse or NVTX-tagged).
- `PASS_BCG_GDN_NO_GAP` — Gate 1 PASS on smallest cell + no H_A/H_B/H_C
  support on any cell.
- `PARTIAL_SWEEP` — investigation stopped short with insufficient evidence
  for any verdict; document what was completed and what remains.

Nothing else.

---

## 13. Plan review (Phase 3, 2026-08-03)

Internal red-team of §§0-12 above from the three audit-agent perspectives.
Each subsection lists the questions the operating model requires that
perspective to ask, and this lead agent's answer. Corrections and safeguards
that emerged from the review are applied inline (revisions R1-R3 below).

### 13.1 Agent 1 (repository / harness) questions

- **Can the harness prove which server process it launched?**
  After T1 the runner writes `preflight.json` AFTER exporting `LD_PRELOAD` and
  `CUDA_VISIBLE_DEVICES`, so those fields will reflect reality. **However, the
  runner does not currently write `metadata.json` — the smoke's was hand-
  composed by the operator.** Without runner-written metadata, per-cell
  provenance for a batched sweep is unreliable. **Revision R1** applies (§13.4).
- **Can teardown kill the complete process tree?**
  Yes — after the setsid PGID fix (`5736f96`), `kill -TERM -"$SERVER_PGID"`
  targets the whole session; the smoke verified 0 MiB post-teardown.
- **Can logprob and token parsing silently fail?**
  After T3 (hard-fail on partial-batch, no `contextlib.suppress`) and T4 (hard-
  fail on missing `output_ids` / `output_logprobs`) — no. Both layers of
  Gate 1's false-pass hole are closed.
- **Are prompt-length labels tokenizer-exact?**
  After T3's `/tokenize` call, `prompt_actual_token_count` is recorded per
  record; the plan uses `prompt_actual_token_count` (not the target) for
  cell labels and bucket assignment.
- **Can repeated requests accidentally reuse results or caches?**
  Cold server per arm (fresh bring-up), and each self-repeat in Phase 5 is
  itself a fresh bring-up. SGLang's radix cache is per-server, so it's empty
  at start of each cell.
- **Can missing cells be mistaken for passing cells?**
  After T5's `PARTIAL_SWEEP` label + `_score_gates` returning `partial: True`
  on any MISSING gate — no. A stopped sweep cannot claim a scored verdict.

### 13.2 Agent 2 (SGLang / BCG source) questions

- **Is the suspected code path reachable on Qwen3.5-4B?**
  Yes — 24 GDN layers × alt-stream branch predicate always True when the
  padded bucket is `< 1024`. `models/qwen3_5.py:551-585`.
- **Is it reachable under BCG specifically?**
  Yes. `_gdn_use_alt_stream = True` unconditionally on CUDA; `get_is_capture_mode()`
  is True during BCG capture AND replay; no BCG-specific short-circuit exists.
  See audit §3.3 for citations.
- **Is the sequence-length threshold based on actual tokens or padded shape?**
  **Padded bucket size.** `seq_len = hidden_states.shape[0]` at
  `qwen3_5.py:560` is post-`_pad_to_bucket` under BCG. This means d3 (~900
  actual tokens) and d4 (~1200 actual tokens) in §4 must be verified against
  the server's actual `capture_num_tokens` bucket list — not the requested
  count. If the bucket list has (e.g.) `[…, 960, 1024, …]`, then a 900-token
  prompt might pad up to 960 (branch fires: 960 < 1024) and a 1200-token
  prompt might pad up to 1280 (branch skipped: 1280 > 1024). **Revision R2**
  applies (§13.4).
- **Does `get_is_capture_mode()` distinguish all relevant graph modes?**
  It returns True for BCG capture, BCG replay, and eager-in-capture-scope.
  It's False for Full and TC piecewise contexts. Sufficient for our arms —
  A0 sees it False, A1/A3 see it True.
- **Is the alt-stream branch actually used on the target hardware?**
  Yes on H200 (SM90); `_gdn_use_alt_stream` reduces to `_is_cuda = True`.
- **Could TC piecewise and BCG intentionally require different behavior?**
  TC piecewise disables the alt-stream branch because dynamo cannot trace
  side-stream fork/join (would graph-break the compiled callable). BCG
  captures raw CUDA API calls via `torch.cuda.CUDAGraph`, so tracer
  constraints don't apply. Whether the omission of a BCG short-circuit is
  intentional or oversight is not evident from source; the patch memo in
  Phase 8 must acknowledge both possibilities.
- **Is recurrent state involved in prefill in the assumed way?**
  Yes — state pool tensors pinned, `cache_indices` recomputed per replay by
  the eager `GDNAttnBackend.forward_extend` between BCG segments. Low
  correctness risk (audit R13.3).
- **Could the suspected wait be required for correctness?**
  R13.4 in the audit — the alt-stream join `current_stream.wait_stream(alt_stream)`
  ensures `in_proj_ba` completes before the fused split consumes it. If the
  BCG capture hook silently drops the join under any driver condition, we
  would see a subtle correctness divergence. **Gate 1 covers this** — token
  divergence would show up in the equality check. This is exactly why Gate 1
  is a blocking prerequisite for any perf conclusion.

### 13.3 Agent 3 (validation / methodology) questions

- **Does the experiment vary only one independent variable at a time?**
  Yes: A0 vs A1 differs only in prefill backend; A0 vs A2 differs only in
  decode backend; A0 vs A3 differs in both — that's why the smallest-cell
  test runs all three separately.
- **Is the sample size sufficient for the proposed verdict?**
  Kernel counts are quasi-deterministic (σ often 0), so `H_A`'s `> 2σ` clause
  degenerates to just `≥ 10% mean gap` — acceptable for a quantised metric.
  For `p95_launch_gap`, the plan reports the metric *per request* (many
  launches per request) then aggregates across 8 timed requests — that's
  the right structure. `MIN_CAPTURES_FOR_REPRO=2` enforces reproducibility
  across cells.
- **Is the noise-floor threshold justified?**
  `max(0.05, 3 × noise_floor)` is a standard 3σ rule; noise_floor is measured
  per cell so tolerance is cell-specific. Defensible.
- **Can Nsight overhead change the execution path?**
  Yes — Nsight slows CPU, potentially shifting stream ordering. The plan's
  §3 mentions this but does not scaffold the disclosure step. **Revision R3**
  applies (§13.4): add explicit A0-unprofiled-vs-A0-profiled comparison at
  the smallest cell before drawing perf conclusions.
- **Are warm-up and steady-state measurements separated?**
  Yes — 2 warm rounds discarded + 8 timed.
- **Are graph capture cost and graph replay cost separated?**
  Yes — capture happens at server startup (recorded in server bring-up time);
  Nsight timing per timed request is replay-only.
- **Does the profile identify graph replay rather than only server activity?**
  T6's CSV extractor emits `cudagraphlaunch_count` per request. Phase-6
  preflight asserts this is > 0 for A1/A3 and 0 for A0/A2 on the prefill
  side. If those assertions fail, the arm didn't actually engage BCG.
- **Can a single measurement incorrectly drive a performance conclusion?**
  No — `MIN_CAPTURES_FOR_REPRO=2` requires ≥ 2 captures per cell for any
  `H_A/H_B/H_C` support; and Phase 7 requires ≥ 2 cells with the signal.

### 13.4 Revisions applied

- **R1 (T1 addendum).** Runner writes `$RESULTS_DIR/metadata.json`
  automatically at the end of each cell, containing: attempt_id, arm,
  arm_flags, prompt/batch/timing config, gpu_id, gpu_uuid, gpu_pre_state,
  gpu_post_state (from T1's post-teardown check), frozen_sglang_sha, model
  pins, fixture sha, tokenizer_source (from T3), preflight statuses, and
  the runner exit code. Removes reliance on operator hand-composition.
- **R2 (Phase 6/7 addendum).** Before running d3 and d4, parse the server
  bring-up log (from any of the A0 baseline runs) for the
  `capture_num_tokens` bucket list; select `prompt_actual_token_count` values
  that map to buckets straddling the 1024 threshold. Record the derived
  bucket size alongside each timed record. Cell labels use the *padded*
  bucket, not the target prompt length.
- **R3 (Phase 6 addendum).** After Phase 6's Nsight captures land, take one
  A0 sub-capture at the smallest cell with `nsys` disabled (regular runner
  invocation) and compare token equality + e2e latency against the Phase-5
  A0 baseline for the same cell. If the delta exceeds the Phase-5 noise
  floor, Nsight is perturbing the code path enough to compromise arm
  comparison — reduce trace channels and rerun.

### 13.5 Lead-agent synthesis

- **Accepted:** most of §§0-12. The tooling order T1-T6 is dependency-correct
  and each patch is small enough for a single commit. The 4-cell A0 ladder
  + smallest-cell 3-arm comparison + evidence-triggered escalation is the
  minimum-cost path to a defensible verdict.
- **Corrections:** three revisions above (R1-R3), all applied inline as
  addenda to the affected phases. R1 slots into Phase-4 T1. R2 slots into
  Phases 6 and 7. R3 slots into Phase 6.
- **Removed assumptions:** none — the source-side reachability is verified
  in audit §3 (facts, not hypotheses).
- **Added safeguards:** runner-written metadata (R1), padded-bucket
  verification for the threshold-ladder cells (R2), explicit Nsight-overhead
  disclosure (R3).
- **Remaining uncertainty:**
  1. R13.4 (alt-stream capture join integrity) — covered by Gate 1 but not
     independently testable without live evidence.
  2. Whether coarse Nsight metrics (kernel counts, cudagraphlaunch_count,
     launch gaps without NVTX attribution) will be enough to attribute the
     mechanism. If not, Phase 7 escalates to NVTX-tagged instrumentation.
  3. Whether Qwen3.5-4B's actual `capture_num_tokens` bucket list places a
     natural boundary near 1024 — verified only at first server bring-up
     (Phase 5).
- **Major blocker check:** none. All revisions are within scope, reversible,
  and don't require frozen-source modification.

### 13.6 Classification

**PLAN_APPROVED_WITH_REVISIONS.**

Revisions R1-R3 are integrated above. Autonomous execution continues to
Phase 4 (tooling hardening).

