# Upstream packages for the S1 track

Patches and texts of the S1 pull requests to `sgl-project/sglang`, kept as a record. PRs and comments are
opened by Bowen only.

| Package | State | Fork branch | Files |
|---|---|---|---|
| **PR-A** — `--goodput` in `bench_serving` | **merged**: opened by Bowen on 2026-10-13 as [sgl-project/sglang#43967](https://github.com/sgl-project/sglang/pull/43967) (head `f96397071e`, same tree as the node-verified `0dcb7e0bae`), approved and merged by JustinTong0323 on 2026-10-18 | `pr/bench-goodput` @ `f96397071e` | [`pr_a/PR_BODY.md`](pr_a/PR_BODY.md) (the text for GitHub), [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md) (what was verified, likely questions), [`pr_a/0001-bench-goodput.patch`](pr_a/0001-bench-goodput.patch) |
| **PR-C** — server-side goodput counter (`--goodput-slo`, `sglang:num_good_requests_total`) | **open**: opened by Bowen on 2026-10-21 as [sgl-project/sglang#45643](https://github.com/sgl-project/sglang/pull/45643), verified on a node on 2026-10-13 (36 unit files, 12 of 12 mutations, pre-commit, upstream's GPU metrics test, counter equal to an exact recount in 12 live phases); independent of PR-A | `pr/goodput-metric` @ `47bd7f463e` on `upstream/main` @ `fc404bbbb3` | [`pr_c/`](pr_c/PR_DESCRIPTION.md), text in [`pr_c/PR_BODY.md`](pr_c/PR_BODY.md) |
| **PR-B** — per-request `waiting_timeout` | **dropped by Bowen on 2026-10-13.** It is JITServe's `waiting_time` parameter and nothing more: the global timeout already drops late requests, the field only makes the bound per request, and it helped in one chosen scenario (priority scheduling under overload) | `pr/req-waiting-timeout` @ `6baf9e5d7e` (on `b524de2de6`) | evidence in [`../results/prb_results.md`](../results/prb_results.md) |

Apply a patch to a checkout of `sgl-project/sglang`: `git am < 0001-….patch`.

## Review path (checked 2026-10-13)

- Upstream's own rules decide what reviewers ask for: `.claude/rules/*.md` in the sglang repository
  (`general-code-style`, `comment-style`, `unit-test-admission`, `no-dataclasses`,
  `no-getattr-defensive`) and the code-style list in the contribution guide. On 10-10 upstream deleted
  6444 lines of CPU tests "with no bug-catching signal" (#43503). PR-A and PR-C were written for these
  rules; [`pr_a/PR_DESCRIPTION.md`](pr_a/PR_DESCRIPTION.md) lists what changed.
- `.github/MAINTAINER.md`: a bot assigns a Merge Oncall only for files in an oncall area, a listed user
  adds the `run-ci` label, and each file protected by `CODEOWNERS` needs one Codeowner's approval.
- Tests run on a RADIX node, not on the Mac (Bowen, 2026-10-13); `scripts/pra_node.sh` and
  `scripts/prc_node.sh` are the flows for one PR head.

The node sessions P1–P3 ran `exp/slo-node`, which carries PR-B and both benchmark commits on the pin
`734cf3cf3b`, because measurements with a waiting timeout need the in-stream abort counted as a failure
(`results/pra_usage.md` §3); `feat/bench-goodput` is the same without PR-B and served as the unpatched
build; it also keeps the first version of the goodput patch (`d132f6739e`), the one sessions P1 and P2
measured.
