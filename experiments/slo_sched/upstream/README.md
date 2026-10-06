# Upstream packages for the S1 track

What would go to `sgl-project/sglang`, kept here as patches and draft texts. Nothing in this directory
has been posted; PRs and comments are opened by Bowen only.

| Package | State | Fork branch | Files |
|---|---|---|---|
| **PR-A** — `--goodput` in `bench_serving` | ready to open; one commit on `upstream/main` @ `b524de2de6` | `pr/bench-goodput` @ `fd40750d7b` | [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md), [`pr_a/0001-bench-goodput.patch`](pr_a/0001-bench-goodput.patch) |
| Benchmark fix — a request the server aborts in-stream is counted as failed | **held**: upstream #40881 (open) already fixes the chat path; ours adds the native path. A comment for #40881 is drafted in the workspace context store | `feat/bench-goodput` @ `9eb681bef6` (second commit) | [`bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch`](bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch) |
| **PR-B** — per-request `waiting_timeout` | in progress | `feat/req-waiting-timeout` | [`../PRB_PLAN.md`](../PRB_PLAN.md) |

Apply a patch to a checkout of `sgl-project/sglang`: `git am < 0001-….patch`.

The node sessions run the branch that carries both benchmark commits (`feat/bench-goodput`), because
measurements with a waiting timeout need the fix.
