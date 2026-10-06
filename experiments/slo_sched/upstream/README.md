# Upstream packages for the S1 track

What would go to `sgl-project/sglang`, kept here as patches and draft texts. Nothing in this directory
has been posted; PRs and comments are opened by Bowen only.

| Package | State | Fork branch | Files |
|---|---|---|---|
| **PR-A** — `--goodput` in `bench_serving` | ready to open; one commit on `upstream/main` @ `b524de2de6` | `pr/bench-goodput` @ `fd40750d7b` | [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md), [`pr_a/0001-bench-goodput.patch`](pr_a/0001-bench-goodput.patch) |
| Benchmark fix — a request the server aborts in-stream is counted as failed | **held**: upstream #40881 (open) already fixes the chat path; ours adds the native path. A comment for #40881 is drafted in the workspace context store | `feat/bench-goodput` @ `9eb681bef6` (second commit) | [`bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch`](bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch) |
| **PR-B** — per-request `waiting_timeout` | built and measured; draft description ready for review; two commits on `upstream/main` @ `b524de2de6` | `pr/req-waiting-timeout` @ `6baf9e5d7e` | [`pr_b/PR_DESCRIPTION.md`](pr_b/PR_DESCRIPTION.md), [`pr_b/0001-req-waiting-timeout.patch`](pr_b/0001-req-waiting-timeout.patch), evidence in [`../results/prb_results.md`](../results/prb_results.md) |

Apply a patch to a checkout of `sgl-project/sglang`: `git am < 0001-….patch`.

## Before opening (checked 2026-10-06 23:01 UTC)

- Both PR-ready branches still merge cleanly with `upstream/main` @ `6b737fd4c6` (41 commits after their
  base; only `schedule_batch.py` and one line of `scheduler.py` changed among the files they touch).
  Rebase on the day they are opened and re-run the unit files.
- No other open PR adds goodput or a per-request waiting bound. Neighbours: #40881 (benchmark stream
  errors, chat path, unreviewed), #34457 (PD waiting timeout, stale), #42453 (dLLM waiting timeout).
- Review path upstream (`.github/MAINTAINER.md`): a bot assigns a Merge Oncall, a maintainer adds the
  `run-ci` label, and each modified file needs one Codeowner's approval. PR-A's file
  (`python/sglang/benchmark/serving.py`) has no Codeowner entry. PR-B touches
  `python/sglang/srt/managers` (@merrymercy @Ying1123 @hnyls2002 @xiezhq-hermann) and
  `python/sglang/srt/entrypoints/openai` (@JustinTong0323).
- PR-B is two commits; upstream squashes on merge, so they can stay as they are.

The node sessions run `exp/slo-node`, which carries PR-B and both benchmark commits on the pin
`734cf3cf3b`, because measurements with a waiting timeout need the fix; `feat/bench-goodput` is the same
without PR-B and serves as the unpatched build.
