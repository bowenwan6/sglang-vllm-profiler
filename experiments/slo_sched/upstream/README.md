# Upstream packages for the S1 track

What would go to `sgl-project/sglang`, kept here as patches and draft texts. Nothing in this directory
has been posted; PRs and comments are opened by Bowen only.

| Package | State | Fork branch | Files |
|---|---|---|---|
| **PR-A** — `--goodput` in `bench_serving` | **ready to open, goes first** (Bowen, 2026-10-13); one commit on `upstream/main` @ `e775fa7253`, verified on a node at that head after Bowen's review (JITServe cited, shorter text, code in the file's own idiom) | `pr/bench-goodput` @ `0dcb7e0bae` | [`pr_a/PR_BODY.md`](pr_a/PR_BODY.md) (the text for GitHub), [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md) (how to open it, what was verified, likely questions), [`pr_a/0001-bench-goodput.patch`](pr_a/0001-bench-goodput.patch) |
| Benchmark fix — a request the server aborts in-stream is counted as failed | **held, and now without a competitor**: #40881 was closed unmerged by its author on 2026-10-09 after 16 days without a review, so the comment drafted for it is moot and the fix (chat and native path) could go out as its own PR after PR-A — Bowen's call | `feat/bench-goodput` @ `9eb681bef6` (second commit) | [`bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch`](bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch) |
| **PR-B** — per-request `waiting_timeout` | built and measured; draft description ready for review; two commits on `upstream/main` @ `b524de2de6`, **not rebased yet and not yet checked against upstream's `.claude/rules`** (PR-A needed a restructuring for them) | `pr/req-waiting-timeout` @ `6baf9e5d7e` | [`pr_b/PR_DESCRIPTION.md`](pr_b/PR_DESCRIPTION.md), [`pr_b/0001-req-waiting-timeout.patch`](pr_b/0001-req-waiting-timeout.patch), evidence in [`../results/prb_results.md`](../results/prb_results.md) |

Apply a patch to a checkout of `sgl-project/sglang`: `git am < 0001-….patch`.

## Before opening (checked 2026-10-13)

- `upstream/main` moved 486 commits between 10-06 and 10-12 (`b524de2de6` to `7224fce1b5`); none touched
  `benchmark/serving.py` or the docs page. PR-A is rebased and verified on `e775fa7253` (two commits
  later; one of them changed a line of the docs page, no conflict). PR-B still sits on the
  old base.
- Upstream's own rules decide what reviewers ask for: `.claude/rules/*.md` in the sglang repository
  (`general-code-style`, `comment-style`, `unit-test-admission`, `no-dataclasses`,
  `no-getattr-defensive`) and the code-style list in the contribution guide. On 10-10 upstream deleted
  6444 lines of CPU tests "with no bug-catching signal" (#43503). PR-A was restructured for these rules;
  [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md) lists what changed. PR-B needs the same pass before
  it is opened: its unit cases against the admission rule, the scheduler edits against
  `large-class-style`.
- No other open PR adds goodput or a per-request waiting bound. #40881 (benchmark stream errors) was
  closed by its author on 10-09; neighbours of PR-B: #34457 (PD waiting timeout, stale), #42453 (dLLM
  waiting timeout).
- Review path (`.github/MAINTAINER.md`): a bot assigns a Merge Oncall only for files in an oncall area,
  a listed user adds the `run-ci` label, and each file protected by `CODEOWNERS` needs one Codeowner's
  approval. For PR-A only the docs page has Codeowners and nobody is assigned; the people who merged
  serving-benchmark PRs lately are named in `pr_a/PR_DESCRIPTION.md`. PR-B touches
  `python/sglang/srt/managers` (@merrymercy @Ying1123 @hnyls2002 @xiezhq-hermann) and
  `python/sglang/srt/entrypoints/openai` (@JustinTong0323).
- PR-B is two commits; upstream squashes on merge, so they can stay as they are.
- Tests run on a RADIX node, not on the Mac (Bowen, 2026-10-13); `scripts/pra_node.sh` is the flow for
  one PR head.

The node sessions run `exp/slo-node`, which carries PR-B and both benchmark commits on the pin
`734cf3cf3b`, because measurements with a waiting timeout need the fix; `feat/bench-goodput` is the same
without PR-B and serves as the unpatched build; it also keeps the first version of the goodput patch
(`d132f6739e`), the one sessions P1 and P2 measured.
