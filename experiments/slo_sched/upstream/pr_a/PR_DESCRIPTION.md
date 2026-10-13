# PR-A — ready to open (Bowen opens it; nothing has been posted upstream)

- **Title:** `[Benchmark] Report goodput and SLO attainment in bench_serving`
- **Body:** [`PR_BODY.md`](PR_BODY.md), exactly as it goes to GitHub (upstream's template, all five
  sections and its "Review and Merge Process" block).
- **Branch:** `bowenwan6/sglang` `pr/bench-goodput` @ `74e78591a3` — one commit on `upstream/main`
  @ `7224fce1b5` (2026-10-12), ahead 1 / behind 0 when pushed (2026-10-13).
  Patch: [`0001-bench-goodput.patch`](0001-bench-goodput.patch).
- **Diff:** 4 files, +246: `python/sglang/benchmark/goodput.py` (new, 118), `serving.py` (+36),
  `test/registered/unit/bench/test_bench_serving_goodput.py` (new, 72), `bench_serving.mdx` (+20).

## How to open it

Web: <https://github.com/sgl-project/sglang/compare/main...bowenwan6:sglang:pr/bench-goodput?expand=1>.
The title is pre-filled from the commit subject; replace the body with `PR_BODY.md`; leave "Allow
edits by maintainers" ticked; create it as a normal pull request, not a draft.

Command line, from the workspace root:

```bash
gh pr create --repo sgl-project/sglang --base main --head bowenwan6:pr/bench-goodput --title "[Benchmark] Report goodput and SLO attainment in bench_serving" --body-file sglang-vllm-profiler/experiments/slo_sched/upstream/pr_a/PR_BODY.md
```

What happens next, from upstream's `MAINTAINER.md`, `CODEOWNERS` and the history of similar PRs:

- Lint runs at once. The test workflows stop at "Missing required label run-ci" until someone listed in
  `.github/CI_PERMISSIONS.json` comments `/tag-and-rerun-ci`; `bowenwan6` is not listed. As the author
  Bowen can only use `/rerun-failed-ci`.
- GitHub requests a review from the `/docs` Codeowners (@wisclmy0611 @zijiexia @Richardczl98
  @JustinTong0323 @sogalin) because of the docs page. The other three files have no Codeowner, and
  `python/sglang/benchmark` is in no Merge Oncall area, so the bot assigns nobody: #40881, #39889,
  #39398 and #30918 all had no assignee.
- Who merged serving-benchmark PRs lately: @Fridge003 (#30918, #39398), @BBuf (#39889),
  @merrymercy with @xiezhq-hermann's approval (#39284), @b8zhong (#33663). If nothing moves in a few
  days, ask one of them for `run-ci` and a review in the PR or in Slack `#pull-request`. #40881 (same
  file, no docs change, so no reviewer was requested) got no maintainer answer in 16 days; its author
  closed it two days after pinging @BBuf.
- After review starts, add commits instead of force-pushing; upstream squashes on merge.
- If `upstream/main` moves under the files (it did not between 10-06 and 10-12 for `serving.py`),
  rebase and repeat the node check with `scripts/pra_node.sh`.

## What changed since the version measured in sessions P1 and P2

The arithmetic is the same line for line; the structure follows upstream's own rules, which had not
been read when the first version was written (`.claude/rules/*.md` and the code-style section of the
contribution guide):

| Rule | Before | Now |
|---|---|---|
| files under ~2000 lines, new components in new files | +134 lines in `serving.py` (2807 lines) | `goodput.py`, +36 lines of wiring in `serving.py`; the layout `steady_state.py` uses (#30918) |
| pure functions, small functions | three new fields on `BenchmarkMetrics`, 14 lines inside `calculate_metrics` | both untouched; `calculate_goodput()` returns a frozen `msgspec.Struct` whose field names are the result keys |
| `msgspec.Struct`, not `@dataclass` | fields added to an existing dataclass | one new Struct |
| keyword arguments for calls with two or more arguments | one positional call | all by keyword |
| unit-test admission: every case answers "what future diff turns this red?" | 8 cases | 3 cases (table-driven); a mutation run shows each guards something |

The result keys, the printed block and the option are unchanged, so the P1/P2 result files are what
this head would have written.

## Verified on a node (2026-10-13, `node-radixark-16-0000`, 1× H200, 26 min, no extension)

`scripts/pra_node.sh`, then `scripts/pra_recheck.sh` for the last head. Small outputs are in
[`../../results/pra/final_head/`](../../results/pra/final_head/); the raw run is in `results/raw/pra/`
(git-ignored, 110 MB).

| Check | Result |
|---|---|
| Environment | python 3.12.15, torch 2.14.1+cu130, transformers 5.19.0, flashinfer 0.7.1, sgl_kernel 0.4.9 |
| Unit files, one process each as CI runs them | 6 of 6 pass: ours (3 cases) and five neighbours that import the benchmark (2 + 1 + 10 + 4 + 44 cases) |
| Mutation check (`scripts/pra_mutate.py`) | 15 of 15 regressions turn the file red |
| ruff format, ruff check, isort, codespell at upstream's pins | pass |
| pre-commit on the four files | 16 hooks pass; 14 have no matching file; `no-commit-to-branch` skipped on purpose (it guards the name of the local branch) |
| `--goodput` in `--help` of both entry points | present |
| Live server, 8 runs with the flag and 1 without | 50 of 50 checks: result fields, printed block and recount agree; without the flag no block and no field |

The live-server runs and the first pass of every other check ran at `15f3c7007f`. The pushed head
`74e78591a3` differs from it in one comment of the test file (checked on the node: no file under
`python/` or `docs/` differs); the unit file, the mutation check and pre-commit were repeated on it.

Capacity at concurrency 256 was 40.52 req/s (40.2 on main of 10-05), and the sweep agrees with the one
of P1/P2: 100.0, 99.8, 99.7, 98.5, 52.2 and 17.2 % attainment against 100.0, 99.8, 99.8–100.0,
93.4–96.5, 36.9–71.8 and 16.0–23.1 %.

## Questions a reviewer may raise

| Question | Position taken | Alternative |
|---|---|---|
| Why `if not hasattr(args, "goodput")` when the rules discourage `hasattr`? | `run_benchmark` receives namespaces built by hand in tests and other scripts; the function already fills in 19 fields this way, the latest in #33663 (2026-08-06, after the rule was added) | set the field in every caller that builds a namespace |
| Why not fields on `BenchmarkMetrics`, as vLLM does? | keeps `calculate_metrics` (wrapped by `steady_state_serving.py` through its signature) and the dataclass unchanged | three `Optional` fields and one parameter; 20 more lines in `serving.py` |
| Why a new test file? | no HTTP stub is needed, unlike `test_bench_serving_prompt_len.py`, whose fixture starts a server in `setUp` | a second class in that file |
| Flat result keys or a nested `goodput` object? | flat, like vLLM's `request_goodput` | `result["goodput"] = {...}`, like `cache_report` |
| `<=` or `<` at the SLO? | `<=`, as in vLLM | — |
| An `itl` key? | not included; vLLM has `ttft`, `tpot`, `e2el` | one more entry in the tuple and the observed-values dict |
| Requests the server aborts in-stream count as good | true and independent of this PR (see the body); our fix for the chat and the native path is held in `../bench_stream_abort_fix/` | open it as its own PR now that #40881 is closed |
