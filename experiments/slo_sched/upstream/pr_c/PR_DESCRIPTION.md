# PR-C — server-side goodput counter (internal notes)

The text for GitHub is [`PR_BODY.md`](PR_BODY.md). Nothing here has been posted; Bowen opens the PR.

- **Title:** `feat(metrics): count requests that meet latency SLOs for goodput` (same as the commit subject)
- **Head:** fork branch `pr/goodput-metric` @ `f0593274f9`, one commit on `upstream/main` @ `fc404bbbb3`
  (2026-10-12), +193/−8 in 12 files. Patch: [`0001-goodput-metric.patch`](0001-goodput-metric.patch).
- **Open it:** <https://github.com/sgl-project/sglang/compare/main...bowenwan6:sglang:pr/goodput-metric?expand=1>,
  paste the title and `PR_BODY.md`.
- **Does not depend on PR-A (#43967).** The docs do not mention `bench_serving --goodput`, which is not on `main`.

## What it is

JITServe's SLO tracker, reduced to what an engine can export without scheduling changes: with
`--goodput-slo '{"ttft": 500, "tpot": 50}'` the tokenizer-side collector counts, per finished request, whether
every listed SLO was met (`sglang:num_good_requests_total`). `rate()` of it is request goodput; divided by
`sglang:num_requests_total` it is SLO attainment among finished requests.

## How the code follows upstream's precedents

| Piece | Copied from |
|---|---|
| launch argument as a JSON dict, `type_parser=json.loads` | `--extra-metric-labels`, same file (six dict arguments upstream are JSON) |
| validation with asserts in `validation_hook.py`, called next to the bucket rules | `validate_buckets_rule` |
| SLOs handed to the collector as a constructor argument | the three `bucket_*` arguments |
| one more optional keyword on `observe_one_finished_request`, `None` for aborts and PD prefill | `time_per_output_token`, #40275 (merged 09-29) |
| unit cases beside `TestRequestTimePerOutputToken` and in the manager's TPOT gating table | #40275 |
| one line pair in the GPU metrics test | #40275 |
| a row in `server_arguments.mdx` | required by `test_server_args_docs_coverage.py` |

The pure check (`meets_goodput_slo`) and the key names live in `observability/utils.py`, which both the collector
and the validation hook can import without a cycle.

## Verified on a node (2026-10-13, 1× H200, 31 min, not charged)

At `1a9681a340` (the head before one docs paragraph was added):

- 36 CPU unit files, one process each: all pass (the changed ones, all of `unit/observability` and
  `unit/server_args`, and the runtime-context and server-args ratchets).
- Mutation check: 12 of 12 regressions turn a unit file red (`scripts/prc_mutate.py`).
- Pinned ruff / isort / codespell and pre-commit on the 12 files: pass.
- `--goodput-slo` appears in `launch_server --help`.
- Upstream's GPU test `TestEnableMetrics.test_metrics_1gpu` (Qwen3-0.6B): passes; with the wiring line removed
  it fails with `KeyError: 'sglang:num_good_requests_total'`.
- Live server (Qwen3-8B), `scripts/prc_verify.py`: the counter's growth equals an exact recount from each
  response's own `meta_info` latencies in 12 phases over five servers, 76 of 76 checks, no request within
  1e-6 ms of an SLO. The first attempt of this step crashed in my script (a method named like an attribute
  of the base class); the node summary file still shows that FAIL, the rerun is in `live_verdict.json`.
- Cost of the check: 16.5 → 19.3 µs per `observe_one_finished_request` (`scripts/prc_overhead.py`).

At the final head `f0593274f9`: it differs from the above in `docs/docs/references/production_metrics.mdx`
only (checked on the node); the six changed or directly related unit files and pre-commit were run again and pass.

Small evidence files: `results/prc/final_head/`. Raw output: `results/raw/prc/` (Mac only, git-ignored).

## Found on the way

- A request rejected by `--max-queued-requests` is in neither `sglang:num_requests_total` nor the latency
  histograms (600 sent, 536 rejected, `num_requests_total` +64). So good / `num_requests_total` is attainment
  among finished requests and reads 100 % on a server that sheds most of its load; the docs section says so and
  names `sglang:http_requests_total` as the other denominator. This is existing behaviour, not changed here.
- The same run shows the in-stream abort again: 286 streaming requests got the rejection inside a 200 response,
  250 non-streaming ones got HTTP 503.

## Not covered

- PD mode on a live server (the prefill exclusion is covered by the unit case only), the Ray collector backend,
  embedding and scoring requests, more than one GPU.
- CI's own lanes run only after the PR is open and a listed user adds `run-ci`.

## What a reviewer may ask

- **Why not derive it from the histograms?** A joint count over several latencies is not derivable from
  separate histograms; a single SLO only when it is a bucket boundary. Measured example in the PR text
  (96.8 % and 94.7 % alone, 91.7 % together).
- **Why milliseconds when the metrics are in seconds?** SLOs are stated in ms in `bench_serving --goodput`
  (#43967, vLLM) and in the router's `x-sgl-ttft-slo-ms`. Switching to seconds is a one-line change.
- **Why JSON instead of `ttft:500`?** Every dict-valued launch argument upstream is JSON.
- **Per-SLO counters?** Left out to keep the PR small; easy to add as `sglang:num_slo_met_requests_total{slo=...}`.
- **Per-request SLOs (headers)?** Out of scope; the router has SLO headers, the engine has no SLO field.

Code owners that will be asked: `/python/sglang/srt/observability` @merrymercy @fzyzcjy @sufeng-buaa,
`/python/sglang/srt/managers` @merrymercy @Ying1123 @hnyls2002 @xiezhq-hermann, `/docs` (the docs owners).
