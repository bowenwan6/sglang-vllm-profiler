# PR-A — opened 2026-10-13 as sgl-project/sglang#43967, merged 2026-10-18

- **Title:** `[Benchmark] Report goodput and SLO attainment in bench_serving`
- **Body:** [`PR_BODY.md`](PR_BODY.md), exactly as it goes to GitHub: a one-line summary, then
  upstream's five template sections and its checklist. It cites JITServe (NSDI '26,
  arXiv 2504.20068) in the motivation. The template's "Review and Merge Process" boilerplate is left
  out, as in Bowen's #26864 and #33726.
- **Branch:** `bowenwan6/sglang` `pr/bench-goodput` @ `0dcb7e0bae` — one commit on `upstream/main`
  @ `e775fa7253` (2026-10-12). When pushed (2026-10-13 02:27 UTC) upstream was four commits further,
  none of them in the benchmark package, its tests or the docs page; the branch merges cleanly.
  Patch: [`0001-bench-goodput.patch`](0001-bench-goodput.patch).
- **Diff:** 4 files, +190: `python/sglang/benchmark/goodput.py` (new, 68), `serving.py` (+40),
  `test/registered/unit/bench/test_bench_serving_goodput.py` (new, 62), `bench_serving.mdx` (+20).

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
- Whom to ask if nothing moves in a few days: **@JustinTong0323 first** — he reviewed and merged Bowen's
  earlier benchmark PR #26864 and approved #33726. Others who merged serving-benchmark PRs lately:
  @Fridge003 (#30918, #39398), @BBuf (#39889), @merrymercy with @xiezhq-hermann's approval (#39284),
  @b8zhong (#33663). #40881 (same file, no docs change, so no reviewer was requested) got no
  maintainer answer in 16 days and was closed by its author.
- After review starts, add commits instead of force-pushing; upstream squashes on merge.
- If `upstream/main` changes `benchmark/serving.py` or the docs page before it is opened, rebase and
  repeat the node check with `scripts/pra_node.sh`.

## How the code got here

Three versions; the arithmetic is the same in all of them.

1. *Measured in sessions P1 and P2* (`d132f6739e`, kept on the fork as `feat/bench-goodput`): +134
   lines in `serving.py`, fields on `BenchmarkMetrics`, an 8-case unit file.
2. *First pass against upstream's rules* (`74e78591a3`, 10-13 01:20 UTC): scoring in its own module,
   3 unit cases.
3. *After Bowen's review* (`0dcb7e0bae`, this head): written the way the file already does things.

| What upstream's rules or the file's own habits ask for | In this head |
|---|---|
| files under about 2000 lines, new components in new files (`serving.py` has 2807) | `goodput.py`, 68 lines; `serving.py` +40 |
| no duplicated code | each `--goodput` entry is parsed by an argparse `type` that reuses the file's `_finite_positive_float`; a bad entry is an argparse error, not a traceback |
| follow the nearest existing feature | wired like `--cache-report`: read from `args` inside `benchmark()`, default-filled in `run_benchmark`; `benchmark()`, `calculate_metrics` and `BenchmarkMetrics` keep their signatures |
| `msgspec.Struct` for new containers | `GoodputMetrics`, frozen; its field names are the result keys |
| type-only imports | `if TYPE_CHECKING:` for `RequestFuncOutput`, as in 667 files of the package; no import inside a function |
| comments state what the code cannot | three one-line docstrings, two trailing field comments, one comment (why a single-token response has TPOT 0); three short comments in the test |
| every unit case answers "what future diff turns this red?" | 2 table-driven cases; 11 regressions applied one at a time each turn the file red |

Behaviour that differs from version 1: a repeated key takes the last value (it was an error), and a
malformed entry is reported by argparse. Result keys, the printed block and every number are computed
as before.

## Verified on a node (2026-10-13, `node-radixark-16-0000`, 1× H200)

Second node session of this PR: 01:48–02:26 UTC, no extension, credits 93 / 93 before and after.
`scripts/pra_node.sh` at `0dcb7e0bae`, then `scripts/pra_verify.py` once more with an exact recount.
Small outputs: [`../../results/pra/final_head/`](../../results/pra/final_head/); raw:
`results/raw/pra/` (git-ignored).

| Check | Result |
|---|---|
| Environment | python 3.12.15, torch 2.14.1+cu130, transformers 5.19.0, flashinfer 0.7.1 |
| Unit files, one process each as CI runs them | 6 of 6 pass: ours (2 cases) and five neighbours that import the benchmark (2 + 1 + 10 + 4 + 44 cases) |
| Mutation check (`scripts/pra_mutate.py`) | 11 of 11 regressions turn the file red |
| ruff format, ruff check, isort, codespell at upstream's pins | pass |
| pre-commit on the four files | 16 hooks pass; 14 have no matching file; `no-commit-to-branch` skipped on purpose (it guards the name of the local branch) |
| `--goodput` in `--help` of both entry points | present |
| Live server, exact recount: 8 runs with the flag, 1 without | 59 of 59 checks: result fields = printed block = recount of the records the benchmark scored; without the flag no block and no field |

**The recount had to change.** The first pass at this head compared with a recount from
`--output-details`, as every earlier session did, and 49 of 50 checks agreed: at 2.0 × capacity the
TPOT-alone count was 2216 reported against 2217 recounted. `--output-details` has no end-to-end
latency, so that recount uses `ttft + sum(itl)`; the benchmark reads the clock again after later
chunks. The request in question has two output tokens and a recounted TPOT of 99.9797 ms: 20 µs more
latency puts it over the 100 ms line. `scripts/pra_capture.py` now runs the benchmark in-process and
keeps the per-request `success`, `ttft`, `latency` and `output_len` it scores; with those the recount
is exact and all nine runs agree. The 23 + 9 earlier runs agreed under the approximate recount;
nothing suggests the reported numbers were ever off.

Capacity at concurrency 256: 40.94 req/s (40.2 on main of 10-05). The sweep matches P1/P2: 100.0,
99.8, 99.6, 93.0, 63.5 and 18.8 % attainment against 100.0, 99.8, 99.8–100.0, 93.4–96.5, 36.9–71.8
and 16.0–23.1 %.

## Questions a reviewer may raise

| Question | Position taken | Alternative |
|---|---|---|
| Why `if not hasattr(args, "goodput")` when the rules discourage `hasattr`? | `run_benchmark` receives namespaces built by hand in tests and other scripts and fills in 19 fields this way, `cache_report` among them; the latest was added in #33663 (2026-08-06, after the rule) | set the field in every caller that builds a namespace |
| Why not fields on `BenchmarkMetrics`, as vLLM does? | keeps `calculate_metrics` (wrapped by `steady_state_serving.py` through its signature) and the dataclass unchanged | three `Optional` fields and one parameter |
| Why a new test file? | no HTTP stub is needed, unlike `test_bench_serving_prompt_len.py`, whose fixture starts a server in `setUp` | a second class in that file |
| Flat result keys or a nested object? | flat, like vLLM's `request_goodput` | `result["goodput"] = {...}`, like `cache_report` |
| A repeated key | the last one wins, as with any repeated option | an argparse error |
| `<=` or `<` at the SLO? | `<=`, as in vLLM | — |
| Why JITServe and not DistServe, which vLLM's help text cites? | JITServe is the work this came out of; both use the term the same way | cite both |
| Requests the server aborts in-stream count as good | true and independent of this PR (one sentence in the body) | — |
