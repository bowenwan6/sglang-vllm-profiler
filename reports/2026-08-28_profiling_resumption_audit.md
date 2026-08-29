# Profiling resumption audit after the BCG fix

Date: 2026-08-28 UTC · Evidence appendix and claim-to-source ledger:
[`2026-08-28_profiling_resumption_audit_sources.md`](2026-08-28_profiling_resumption_audit_sources.md)

> ### Update — 2026-08-29
>
> This audit's stated limitation ("Live GitHub API access was unavailable, so
> the current DeepStack PR state remains unverified") is now **resolved**, and
> **WP1 is substantially complete**. Verified against the live API:
>
> - The upstream PR **does** exist: [sgl-project/sglang#33726](https://github.com/sgl-project/sglang/pull/33726),
>   opened 2026-08-05, **approved** by the maintainer 2026-08-06, +294/−12 across
>   7 files. The audit's inference that the docs were stale rather than the PR
>   missing was correct.
> - The branch was 263 upstream commits behind and **conflicted**; that has been
>   merged (`c31e6fe315`) and the PR now reports `mergeable: true`.
> - A post-merge dense smoke (M10) was run because upstream #35451 rewired the
>   closure the fix lives in.
>
> Current BCG state lives in
> [`upstream_handoff.md`](../experiments/qwen3vl_bcg_deepstack_fix/upstream_handoff.md);
> WP0's document reconciliation was applied to `README.md`, `plan.md`, and
> `project_memory.md` on the same date. **WP2–WP5 are unchanged and remain the
> plan of record.**

## Executive answer

You are already back on the profiler repository's `main` branch. Local Git has
`HEAD`, `main`, and the recorded `origin/main` at `d18d0e7`; every named debug
branch is merged into `main`. Do not switch or delete anything yet, because the
working tree contains user-owned changes and raw experiment evidence.

The BCG bug is technically fixed and well validated, but the handoff is not
administratively closed. The fix passed dense Qwen3-VL correctness, ten
no-regression gates, adversarial review, and a Qwen3-VL MoE smoke. The SGLang
fork branch is pushed and clean, but it advanced after the last submission docs;
those docs still say “PR not opened” and describe an obsolete branch snapshot.

The main profiling program has a solid completed core, but two roadmap promises
remain unfinished: headline-quality image+text IMG-A data and the actual
Qwen3.5 SGLang-vs-vLLM transfer comparison.

## Open-issue disposition

The GitHub tracker currently lists #1, #3, #4, #5, and #9 as open. Repository
evidence supports the following disposition:

| Issue | Repository reality | Recommended tracker action |
|---|---|---|
| [#9 — Qwen3.5 DeepStack under multimodal prefill BCG](https://github.com/bowenwan6/sglang-vllm-profiler/issues/9) | The Qwen3.5 question is answered: `NOT_APPLICABLE_QWEN35`. Every released checkpoint inspected has an empty DeepStack index list. The live-fire failure belonged to Qwen3-VL under a test-only BCG allowlist and was fixed on a separate follow-up track. | **Close now as completed**, with the Qwen3.5 verdict and a cross-link to the Qwen3-VL fix evidence. |
| [#4 — Qwen3-VL image+text + CUDA IPC](https://github.com/bowenwan6/sglang-vllm-profiler/issues/4) | Partial. The fixed-generator S0 IPC arm is complete (5/5, 2,000 requests, 64.8 ms TTFT p50); PCG crashed; repeat/vLLM/no-IPC controls did not run. | **Keep open and make this the immediate profiling priority.** |
| [#3 — Qwen3.5 clean profiling](https://github.com/bowenwan6/sglang-vllm-profiler/issues/3) | Core deliverable not run. The DeepStack and GDN studies answer different correctness/mechanism questions; neither is the promised SGLang-vs-vLLM Case-A/Case-C transfer benchmark. | **Keep open; run in parallel with #4 after a common environment pin.** |
| [#5 — selective/default-on Qwen3-VL PCG](https://github.com/bowenwan6/sglang-vllm-profiler/issues/5) | Not complete. Text-only evidence supports selective PCG, image evidence is incomplete, and the newer fork prototype is BCG—not PCG. | **Keep open; amend it to compare PCG and BCG explicitly, then decide policy after #4.** |
| [#1 — tracking parent](https://github.com/bowenwan6/sglang-vllm-profiler/issues/1) | Foundation #2 is done; #9 is ready to close; #3/#4/#5 remain. | **Keep open, post a refreshed checklist now, close last.** |

The key distinction is that TC-piecewise PCG and breakable CUDA graph BCG are
different execution paths. The Qwen3-VL BCG DeepStack replay-slot fix does not
complete the PCG benchmark or selective-PCG policy issue.

## What is complete

| Workstream | Status | Evidence-backed result |
|---|---|---|
| Qwen3-VL phases 0-5 | Complete for scoped A/C | Equivalence, baseline, shaping, traces, triage, and clean validation completed. |
| Default-overlap rebaseline (#2) | Complete / PASS | Case A: 21.94 ms SGLang vs 13.12 ms vLLM; PCG intervention 14.04 ms (-36%), TPOT flat. |
| Case-C boundary | Complete | No material batched gap and no Case-A-like PCG gain. |
| Image generator bug | Complete upstream | Special-token generator fix merged; fixed-path smoke passed. |
| PCG capture-stream investigation | Correctness/safety complete | Crash root cause fixed; performance value remains promising but not final. |
| Qwen3.5 DeepStack investigation | Closed | `NOT_APPLICABLE_QWEN35`; shipped Qwen3.5 checkpoints do not exercise DeepStack. |
| Qwen3.5 GDN BCG investigation | Complete | `PASS_BCG_GDN_NOTABLE_GAP`; +13.6% launches, <=2% wall-clock delta, no correctness bug. |
| Qwen3-VL BCG DeepStack fix | Technically complete | FAIL -> PASS, ten gates passed, dense and MoE validation completed. |

Primary local evidence: [README.md](/data/sglang-vllm-profiler/README.md:39),
[active plan](/data/sglang-vllm-profiler/plan.md:12),
[BCG validation report](/data/sglang-vllm-profiler/experiments/qwen3vl_bcg_deepstack_fix/fix_prototype/validation_report.md:24),
[MoE smoke](/data/sglang-vllm-profiler/experiments/qwen3vl_bcg_deepstack_fix/results/m9_moe_smoke_gpu1_20260816T092836Z/report.md:1), and
[GDN final report](/data/sglang-vllm-profiler/experiments/qwen35_4b/gdn/final_report.md:337).

## What remains

### 1. Fix the source-of-truth documents first

The top-level README, active plan, project memory, image-text status, and BCG
submission package disagree with later results. The most serious stale claims
say Qwen3.5 runtime validation is still pending, Issue #4 has not run, and the
Qwen3-VL BCG fix is still only planning. See
[README.md](/data/sglang-vllm-profiler/README.md:33),
[plan.md](/data/sglang-vllm-profiler/plan.md:232), and
[project_memory.md](/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/project_memory.md:50).

### 2. Close the BCG upstream handoff

The old package says the PR was not opened and records a four-commit/five-file
branch. The live SGLang branch now contains later safety, test, cleanup, and
Qwen3-VL allowlist commits and a seven-file diff. Establish current upstream PR
state, refresh the package, rerun the current CPU suite, and repeat the MoE
smoke on a production-matched stack if feasible. The existing MoE report itself
flags the old-container kernel mismatch as a pre-merge caveat.

Do not assume the BCG fix also resolves the older TC-piecewise PCG
capture-stream path; they are different graph backends. Upstream PR #30872 did
merge multimodal BCG and the `input_embeds` replay slot, which is the pattern the
DeepStack fix extends ([SGLang PR #30872](https://github.com/sgl-project/sglang/pull/30872)).

### 3. Finish Issue #4 image+text IMG-A

The fixed-generator run completed only `S0_ipc`: 5/5 reps, 2,000 requests,
64.8 ms TTFT p50. PCG crashed and the repeat, vLLM, and no-IPC controls were
skipped. First run a small current-upstream PCG regression smoke; then complete
`S0_ipc_repeat -> V0_vllm -> S0_noipc` even if PCG remains excluded. Do not
start IMG-B/C until IMG-A has drift, framework-anchor, and IPC controls. See the
[image-text status](/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/README.md:8).

### 4. Run the real Qwen3.5 transfer comparison

The GDN track answered an internal SGLang BCG-vs-eager question. It did not
perform the roadmap's clean SGLang-vs-vLLM Case-A/Case-C transfer check. That
cross-framework experiment remains outstanding and should use a freshly pinned,
version-aligned environment.

### 5. Decide Issue #5 deliberately

The SGLang fork now contains a Qwen3-VL BCG allowlist commit, while the profiler
roadmap still calls selective/default-on graph enablement merely planned. Update
the protocol to distinguish BCG from PCG and decide whether BCG becomes the
production candidate. Do not silently replace the old PCG arm with BCG.

## Safe branch and workspace status

- Profiler repo: already on `main`; no checkout needed.
- All named profiler topic branches are merged into `main`.
- Audit-start dirty state: two modified tracked files and 30 pre-existing
  untracked entries. This audit added the report plus its internal source file
  as two further untracked files; it did not alter the pre-existing changes.
- Raw BCG/log candidates total about 29.3 MiB; one orphan result directory is
  about 0.64 MiB.
- Never use broad `git clean`: its dry-run includes `.agents/`, `.codex/`, and
  experiment evidence.
- Do not stage `.claude/settings.local.json`; repository policy forbids it.
- `/data/sglang-fork` is a separate repository, currently clean on
  `fix/bcg-deepstack-replay-slot` at `639cef9`, synchronized with its fork remote
  in the August 16 local snapshot. Preserve it until upstream handoff is clear.

## Recommended order

1. Preserve/inventory dirty artifacts and refresh Git refs when network access
   is available.
2. Update the status documents so there is one accurate source of truth.
3. Close the BCG handoff without deleting the pushed fix branch.
4. Pin a current reproducible SGLang/vLLM environment.
5. Finish IMG-A controls and current-upstream PCG smoke.
6. Run the Qwen3.5 cross-framework transfer check.
7. Decide selective/default-on BCG/PCG policy from those results.
8. Treat clean B/D baselines and GDN L2Norm fusion as optional follow-ups.

## Detailed execution plan and acceptance gates

### WP0 — Tracker and documentation reconciliation (half day, no GPU)

1. Post the final #9 comment and close it as `NOT_APPLICABLE_QWEN35`. State that
   no shipped Qwen3.5 checkpoint exercises DeepStack; preserve Attempt 03 as a
   Qwen3-VL latent-defect exhibit; link the separate FAIL-to-PASS fix evidence.
2. Update #1 with a checklist: #2 complete, #9 closed, #4 partial/active, #3
   pending, #5 pending evidence and policy decision.
3. Correct the stale top-level README, active plan, project memory, image-text
   status, and BCG submission documents. Label historical SHAs and results; do
   not rewrite them as current measurements.
4. Inventory the dirty workspace and preserve raw evidence. Do not stage the
   local settings file or use broad cleanup commands.

**Gate:** one current status statement agrees across GitHub, README, active
plan, and experiment READMEs.

### WP1 — Finish the BCG handoff (parallel engineering lane, 1–2 days)

1. Establish whether an upstream SGLang PR already exists. If not, refresh the
   old submission package against the live `639cef9` branch, which contains
   later fail-closed, test-pruning, cleanup, and allowlist changes.
2. Review the current seven-file functional diff against a freshly fetched
   upstream parent. Rerun the focused CPU tests from the live branch—not the
   obsolete four-commit instructions.
3. Repeat dense smoke and preferably the MoE smoke on a production-matched
   SGLang/kernel/container stack. Record that BCG really engaged, eager and BCG
   image/text outputs agree, and malformed DeepStack input fails closed.
4. Open or update the upstream PR, then cross-link it from #5 and the profiler
   report. Preserve the fork branch until disposition is known.

**Gate:** current tests pass, the production-matched smoke is documented, and
the upstream disposition is either a live PR, merged PR, or an explicit
rejection/defer decision.

### WP2 — Complete #4 IMG-A first (1–2 GPU days)

1. Freeze a new environment manifest: SGLang SHA, vLLM version/SHA, model
   revision, benchmark harness revision, CUDA/driver/torch/kernel versions,
   attention backend, GPU, launch flags, and IPC environment.
2. Run Phase-0 correctness and a tiny current-upstream image+PCG smoke. If the
   historical capture-stream assertion is gone, restore the PCG arm; if not,
   record the exact current-upstream failure and exclude it transparently.
3. Complete the original non-PCG bracket in order:
   `S0_ipc_repeat -> V0_vllm -> S0_noipc`. Keep the existing S0 IPC result only
   if provenance and stack match; otherwise rerun the entire bracket.
4. Require zero request failures, identical workload identity, fixed warmups,
   five repetitions, drift at or below the protocol threshold, and clean runs
   without profiler/KAPI instrumentation. Report TTFT, TPOT, throughput, CV,
   failures, and whether TTFT includes preprocessing/vision encoding.
5. Answer three questions separately: SGLang vs vLLM, IPC-on vs IPC-off, and
   PCG-on vs default. Add BCG only as a clearly labeled protocol amendment; it
   must not replace the PCG arm silently.
6. Expand next to IMG-C as the batched boundary. IMG-B is useful but secondary;
   multi-image IMG-D remains optional.

**Gate to close #4:** headline-quality IMG-A has a vLLM anchor and IPC
ablation; PCG has either valid clean data or a current-upstream exclusion with
reproduction; text-only and image conclusions are separated.

### WP3 — Execute the actual #3 Qwen3.5 transfer study (1–2 GPU days)

1. Confirm the model/revision and current framework support before freezing
   the protocol. Start with `Qwen/Qwen3.5-4B` only if both frameworks can serve
   the same checkpoint and API semantics.
2. Repeat Phase-0 parity: tokenizer, model revision, greedy settings, prompt
   construction, output length, and functional output checks.
3. Run clean text-only Case A (128->128, concurrency 1) and Case C
   (512->128, concurrency 16) using SGLang default, the canonical supported
   graph intervention, and a vLLM anchor. For Qwen3.5, do not assume the old
   Qwen3-VL PCG lever is valid; its current supported route is BCG unless a
   source audit proves otherwise.
4. Use the #2 discipline: production defaults as headline, five repetitions
   for Case A, interleaved blocks for noisy Case C, no profiler/KAPI, exact
   flags and warmups, and explicit out-of-scope labels for any mismatch.
5. Only profile after a clean gap reproduces. Add image+text as a second phase
   only after #4's harness/protocol is stable and vLLM parity is confirmed.

**Gate to close #3:** a separate Qwen3.5 report answers whether the Qwen3-VL
first-token finding transfers, whether graph coverage explains it, and whether
the Case-C boundary persists. The GDN report remains a separate appendix, not a
substitute.

### WP4 — Resolve #5 with a policy matrix (after #4; 2–4 engineering days)

Build one explicit matrix across backend, modality, and load:

| Path | Text c=1 | Text c=16 | Image c=1 | Image c=16 |
|---|---:|---:|---:|---:|
| Production default | correctness + performance | same | same | same |
| TC-piecewise PCG | benefit already established | known boundary/regression check | pending #4 | pending IMG-C |
| BCG + DeepStack replay fix | smoke/benchmark needed | benchmark needed | dense + MoE correctness exists; perf needed | benchmark needed |

Then implement only the smallest policy supported by the matrix:

- Default-on is acceptable only if dense and MoE correctness pass for text,
  image, mixed image/text sequences, capture-bucket changes, malformed input,
  and eager fallback, with no material regression in the batched boundary.
- If benefit is confined to stable text c=1, choose selective enablement with
  observable hit/miss/fallback reasons.
- If PCG remains unsafe for image input, keep its multimodal auto-disable and
  treat BCG as a separate production candidate rather than renaming it PCG.
- Close #5 only after the policy, implementation, tests, benchmark evidence,
  and upstream disposition are recorded—or after a documented decision not to
  enable either backend by default.

### WP5 — Close the umbrella #1 (half day)

Restructure the final report into production baseline, ablations, Qwen3.5
transfer, image+text/IPC, PCG-versus-BCG policy, and upstream disposition. Close
#1 only when #3, #4, #5, and #9 have final dispositions; optional B/D and L2Norm
work must not block closure.

## Research limitation

The audit reviewed the complete status/verdict/next-step corpus across 185
tracked Markdown files (27,910 lines), with full reads of the primary plans,
reports, validation packages, and active result summaries. Live GitHub API
access was unavailable, so the current DeepStack PR state remains unverified;
local Git and repository evidence are authoritative through the August 16
remote snapshot.
