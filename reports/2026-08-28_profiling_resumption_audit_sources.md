# Deep research source report: return from BCG work to profiling mainline

Audience: repository owner / experiment operator

Date: 2026-08-28 UTC

Scope: local `sglang-vllm-profiler`, its Git history and Markdown evidence, the
adjacent `/data/sglang-fork`, and publicly accessible GitHub evidence. No branch
switch, fetch, cleanup, benchmark, commit, push, issue, or PR mutation was made.

## Direct answer

The profiler repository is already on `main` at `d18d0e7`, matching the locally
recorded `origin/main`. All named investigation branches are ancestors of
`main`; the Qwen3-VL BCG profiler branch was merged by `4c6b054`. The working
tree is not clean: at audit start, two tracked files were modified and 30
pre-existing untracked entries were present, including approximately 29.3 MiB
of BCG logs/raw artifacts. The two research-report files created by this audit
are additional untracked files. Therefore
there is no profiler branch switch left to perform, and cleanup must not be done
blindly.

The original profiling result is complete for its declared A/C scope. The clean
production-default rebaseline established a Case-A first-token/prefill gap
(21.94 ms SGLang vs 13.12 ms vLLM); forced PCG reduced SGLang to 14.04 ms while
TPOT stayed flat. Case C showed no corresponding batched benefit. Phases 0-5 and
v2 issue #2 are complete. See `README.md:41-57,85-95` and `plan.md:12-42`.

The BCG DeepStack bug work is technically complete enough to preserve and hand
off: pre-fix `FAIL_BCG_DEEPSTACK`, post-fix `PASS_BCG_CORRECT`, ten no-regression
gates, adversarial review, dense Qwen3-VL validation, and a later MoE smoke with
BCG demonstrably engaged. The profiler documentation says the upstream PR was
not opened, but that statement is from 2026-08-05 and is stale relative to the
live fork. `/data/sglang-fork` is clean and synchronized with its fork remote at
`639cef9` (2026-08-16), and its fix branch now includes additional fail-closed,
test-pruning, comment-cleanup, and Qwen3-VL BCG-allowlist commits. Current PR
status could not be independently verified: public search found no matching
PR, local networking cannot resolve GitHub, and the documentation predates the
later branch activity.

The profiling mainline is not finished. The highest-value remaining work is:

1. Reconcile stale status documents before executing anything. `README.md`,
   root `plan.md`, `v2/project_memory.md`, the image-text README, and both BCG
   submission documents contradict later commits and experiment outcomes.
2. Close the BCG handoff explicitly: establish the current upstream/fork diff,
   identify whether a PR exists, refresh the test/commit list, run the current
   branch's CPU suite, and ideally repeat the MoE smoke on a production-matched
   stack. Preserve the branch; do not delete it or re-create it from old docs.
3. Resume Issue #4 at IMG-A. First verify whether merged upstream PCG work fixes
   the historical capture-stream assertion on a freshly pinned SGLang SHA.
   Regardless, finish the non-PCG bracket `S0_ipc_repeat -> V0_vllm ->
   S0_noipc` so the existing 64.8 ms S0 datum gains drift, vLLM, and IPC
   controls. Do not proceed to IMG-B/C until IMG-A is headline-quality.
4. Run the actual Issue #3 Qwen3.5 cross-framework transfer check. The completed
   Qwen3.5 GDN study is an SGLang internal BCG-vs-eager mechanism study, not the
   promised clean SGLang-vs-vLLM Case-A/Case-C transfer experiment.
5. Decide Issue #5 using the new evidence. The fork now prototypes Qwen3-VL BCG
   allowlisting, but the old roadmap still describes selective/default-on PCG
   as merely planned. Treat BCG and PCG as separate backends/levers, and do not
   infer that the BCG fix resolves the TC-piecewise PCG benchmark path.
6. Keep optional work optional: clean B/D baselines if a four-workload headline
   is needed; Qwen3.5 GDN L2Norm fusion only if kernel-launch reduction is worth
   pursuing despite an expected 0.03-0.07% end-to-end effect.

Issue disposition derived from the evidence: #9 is ready to close as completed
with `NOT_APPLICABLE_QWEN35`; #4 is partial and should be the immediate profiling
priority; #3's cross-framework transfer study is unrun; #5 remains a policy and
upstream-integration decision and must distinguish PCG from BCG; #1 remains open
as the umbrella and closes last. The dependency order is documentation/#9
closure first, then #4 and #3 in parallel after a shared environment pin, then
#5, then #1. The BCG upstream handoff can proceed as a parallel engineering
lane and feeds the #5 decision.

## Evidence reconciliation

### Profiler Git state

- `HEAD -> main -> origin/main` at `d18d0e7` in the local remote snapshot.
- `main` contains every named topic branch. Ahead counts are 57 (Qwen3.5
  DeepStack), 17 (GDN), 4 (Qwen3-VL BCG fix), 78 (PCG capture-stream fix), and
  76 (plan cleanup), with zero commits unique to those branches.
- The last local remote-reference update is 2026-08-16. A network refresh was
  unavailable, so synchronization claims are bounded to that snapshot.
- Audit-start dirty state: `.claude/settings.local.json` modified; R5C audit
  modified; 30 pre-existing untracked entries including one orphan R6.3
  directory and BCG raw/log artifacts. `CLAUDE.md` forbids staging the local
  settings file and raw outputs without explicit approval.

### What profiling established

- Phase 0 equivalence, Phase 1 baseline, Phase 2 shaping/variance, Phase 3
  traces, Phase 4 triage, and scoped Phase 5 A/C validation are complete
  (`README.md:85-95`).
- Clean Case A: 21.94 ms SGLang default vs 13.12 ms vLLM; PCG intervention
  14.04 ms, -36%, TPOT flat (`README.md:41-53`).
- Clean Case C: no material default gap and no Case-A-like PCG benefit
  (`README.md:54-57`).
- Shared `nvjet_sm90_*` FP8 GEMM is dominant absolute cost, not the gap source
  (`README.md:46-48`).
- Old Phase-1 four-case ratios and the KAPI-contaminated Case-C result must not
  return as headlines (`plan.md:44-58`).

### What BCG work established

- The replay path omitted request-specific `input_deepstack_embeds`; the fix
  adds an optional stable slot, capture wiring, validated replay copy/zeroing,
  and model capability gating (`submission_package.md:50-119`).
- Dense Qwen3-VL moved from `FAIL_BCG_DEEPSTACK` to `PASS_BCG_CORRECT` and ten
  no-regression gates passed (`fix_prototype/validation_report.md:24-49`).
- The later Qwen3-VL-30B-A3B MoE smoke produced byte-identical eager/BCG image
  and text completions, with substantive prefills reporting `cuda graph: True`
  (`results/m9.../report.md:1-96`).
- That MoE run bypassed an old-container kernel-version mismatch and explicitly
  asks for an up-to-date devbox confirmation before merge
  (`results/m9.../report.md:98-126`).
- The old submission files say “PR not opened” and describe a four-commit,
  five-file branch. The live fork has since grown to a seven-file diff plus
  later functional commits and merges; those documents are not submission-safe.

### What remains in the research roadmap

- Image+text Issue #4 remains partial. S0 IPC completed (5/5, 2,000 requests,
  64.8 ms TTFT p50), S2 PCG crashed, and S0 repeat/vLLM/no-IPC were skipped
  (`image_text_benchmarks/README.md:8-36`).
- Qwen3.5 GDN investigation is complete with
  `PASS_BCG_GDN_NOTABLE_GAP`: +13.6% kernel launches, within 2% wall clock,
  mechanism attributed to FLA chunk kernels, not a correctness bug
  (`gdn/final_report.md:337-362`). Its remaining b>1, long-context, and fuller
  decode-CG questions are documented but optional (`:293-333`).
- The L2Norm pair-fusion prototype remains unvalidated. Later successful CUDA
  work using `LD_PRELOAD` weakens the original “no Python process can init CUDA”
  blocker, but does not itself validate the L2Norm prototype.

## Contradictions that must be fixed

1. `README.md:33-37,111-112` says Qwen3.5 DeepStack runtime validation is
   pending; the branch was closed `NOT_APPLICABLE_QWEN35`, a Qwen3-VL latent bug
   was proven, and the fix was later implemented.
2. `README.md:150-157` says Issue #4 has no benchmark runs pending smoke; smoke
   and a partial IMG-A formal run already exist.
3. `plan.md:71,232-244` calls the Qwen3.5 correctness sub-track active/pending;
   it is complete, as is the GDN investigation.
4. `plan.md:1175-1384` calls the Qwen3-VL BCG fix “planning, active” and lists
   pre-fix reproduction/fix-prototype steps that have already completed.
5. `v2/project_memory.md:52-76` says there is no runner/run/server yet; this is
   an obsolete May checkpoint.
6. BCG submission documents list old SHAs and omit the August 15-16 branch
   changes and MoE validation.

## Recommended execution order

### P0 — Safe resumption and source of truth

- Remain on profiler `main`; no checkout required.
- Inventory and deliberately archive/ignore the current untracked evidence.
  Never run broad `git clean`; it would also target `.agents/` and `.codex/`.
- Refresh local refs when network is available, then update `README.md`,
  `plan.md`, `project_memory.md`, image-text status, and BCG handoff docs in one
  documentation-only pass.

### P1 — Close BCG handoff

- Determine actual upstream PR state.
- Rebase or merge current upstream deliberately; review the seven-file diff and
  all post-August-5 commits rather than using the old four-commit instructions.
- Run current CPU tests and production-matched dense/MoE smoke. If accepted,
  open/update the upstream PR and return `/data/sglang-fork` to a provenance-safe
  branch or use worktrees for profiling.

### P2 — Resume main profiling

- Pin one current SGLang SHA, matching kernel/torch/driver stack, vLLM version,
  model revision, and harness revision.
- Run a tiny IMG-A PCG regression smoke on current upstream.
- Finish non-PCG IMG-A controls in the original bracket order. Decide whether
  to add a BCG arm as a new protocol amendment; do not silently substitute it
  for PCG.
- Only after clean IMG-A, expand to IMG-C (batched boundary) and optionally
  IMG-B.
- In parallel or next, execute the promised Qwen3.5 SGLang-vs-vLLM transfer
  check. Keep the GDN optimization sub-track separate.

## Limitations and stopping rule

The local Markdown corpus contains 185 tracked files and 27,910 lines. The audit
scanned the entire status/verdict/next-step corpus and fully read the primary
plans, reports, validation packages, and active result summaries. Raw historical
per-attempt prose was sampled where it could alter a current claim. GitHub live
API/CLI access was unavailable (`gh` absent; shell DNS blocked); public web
retrieval confirmed upstream PR #30872 was merged, but could not establish the
current state of the user's DeepStack PR. Research stopped after every material
done/remaining claim had local primary evidence and the only consequential gap
(current PR status) was explicitly bounded.

## Claim-to-source ledger

| Claim family | Primary source | Date/update | Access notes |
|---|---|---|---|
| Profiling phases and clean A/C findings | `README.md`, `plan.md` | 2026-07-31 / 2026-08-04 | Local tracked files |
| Image+text partial status | `experiments/qwen3vl8b/v2/image_text_benchmarks/README.md`, `results_fixed/imgA_summary.md` | 2026-06-08 | Local tracked files |
| PCG fix audit | `R6_FINAL_CONCLUSION.md` | 2026-07-29 | Local tracked file |
| BCG fix design/tests | `submission_package.md`, `validation_report.md`, `review_report.md` | 2026-08-05 | Local tracked files; submission snapshot stale |
| BCG MoE smoke | `results/m9.../report.md` | 2026-08-16 | Local tracked file |
| Current BCG fork branch | `/data/sglang-fork` Git refs/history/diff | local snapshot 2026-08-16 | Read-only Git inspection |
| Qwen3.5 GDN outcome | `experiments/qwen35_4b/gdn/final_report.md` | 2026-08-04 | Local tracked file |
| Upstream multimodal BCG baseline | GitHub SGLang PR #30872 | merged 2026-07-28 | Public first-party GitHub page |
| Profiler GitHub repository | `github.com/bowenwan6/sglang-vllm-profiler` | web cache inconsistent | Local refs preferred |
