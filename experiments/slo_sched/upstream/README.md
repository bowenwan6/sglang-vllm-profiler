# Upstream packages for the S1 track

What would go to `sgl-project/sglang`, kept here as patches and draft texts. Nothing in this directory
has been posted; PRs and comments are opened by Bowen only.

| Package | State | Fork branch | Files |
|---|---|---|---|
| **PR-A** — `--goodput` in `bench_serving` | ready to open; one commit on `upstream/main` @ `b524de2de6` | `pr/bench-goodput` @ `fd40750d7b` | [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md), [`pr_a/0001-bench-goodput.patch`](pr_a/0001-bench-goodput.patch) |
| Benchmark fix — a request the server aborts in-stream is counted as failed | **held**: upstream #40881 (open) already fixes the chat path; ours adds the native path. A comment for #40881 is drafted in the workspace context store | `feat/bench-goodput` @ `9eb681bef6` (second commit) | [`bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch`](bench_stream_abort_fix/0001-bench-count-in-stream-aborts-as-failed.patch) |
| **PR-B** — per-request `waiting_timeout` | built and measured; draft description ready for review; two commits on `upstream/main` @ `b524de2de6` | `pr/req-waiting-timeout` @ `6baf9e5d7e` | [`pr_b/PR_DESCRIPTION.md`](pr_b/PR_DESCRIPTION.md), [`pr_b/0001-req-waiting-timeout.patch`](pr_b/0001-req-waiting-timeout.patch), evidence in [`../results/prb_results.md`](../results/prb_results.md) |

Apply a patch to a checkout of `sgl-project/sglang`: `git am < 0001-….patch`.

The node sessions run `exp/slo-node`, which carries PR-B and both benchmark commits on the pin
`734cf3cf3b`, because measurements with a waiting timeout need the fix; `feat/bench-goodput` is the same
without PR-B and serves as the unpatched build.
