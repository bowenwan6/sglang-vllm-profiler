# PCG capture-stream debug — current status (post D1–D4)

> Updated 2026-06-08 after Step 6 of the original PCG debug plan. Goal of this
> file: record exactly what D1–D4 did and did not prove, what hypotheses
> remain, and where the next debug stages plug in. No new GPU experiments have
> been run since `6965058 test(v2): run image-text pcg debug stage D1234`.

## 1. Verified facts (do not re-litigate)

- Generator fix is in scope and verified: `/data/sglang-pr` on `main` (HEAD
  `62c505a196`) contains merged commit
  `07f326c184 Fix multimodal synthetic benchmark prompt generation to exclude
  special tokens (#26864)`. V1 audit and V2 serving repro both PASS. Stage 4.1
  fixed-generator smoke PASS. Stage 4.2 `IMG_A_S0_ipc` clean (5/5 reps, 0
  failures, TTFT p50 64.8 ms, no forbidden-token errors). Across this whole
  debug, the fixed-path import gate has stayed green
  (`sglang.__file__ = /data/sglang-pr/python/sglang/__init__.py`, `FIX_OK`).
- **Stage 4.2 formal `IMG_A_S2_ipc_pcg` crash is real.** Server-side
  `AssertionError: PCG capture stream is not set` in
  `srt/compilation/cuda_piecewise_backend.py:171`, fired during rep 1 of a
  `n=400, warmup=30, output_len=128, IPC on, PCG on` Qwen3-VL image+text run.
  Server log
  (`logs/qwen3vl8b/v2/image_text_benchmarks/results_fixed/IMG_A_S2_ipc_pcg_server.log`)
  records the crash; the partial run record is at
  `results_fixed/imgA_summary.md`. This is not a generator bug.
- Upstream auto-disables PCG for multimodal models (`server_args.py:1374-1376`),
  and `--enforce-piecewise-cuda-graph` explicitly bypasses every auto-disable
  rule (`server_args.py:1342-1346`, comment: *"Skip auto-disable when enforce
  flag is set (for testing)"*). HIP has a graceful fallback at
  `cuda_piecewise_backend.py:163-169` to eager execution on the same code
  path; CUDA deliberately keeps the assertion.

## 2. What D1–D4 actually showed

The Step 6 matrix used a tiny `num_prompts=2, warmup=0` correctness probe
against a fresh server per case, fixed-generator path,
`PYTHONPATH=/data/sglang-pr/python`. Results in
`results/D1234_summary.md` / `D1234_results.json`:

| case | dataset | IPC | PCG | classification | takeaway |
|---|---|---|---|---|---|
| D1 | image | on | on | **`OK`** | image + PCG + IPC at 2 prompts did **not** trigger the assertion |
| D2 | image | off | on | **`OK`** | image + PCG without IPC at 2 prompts also did **not** trigger the assertion |
| D3 | image | on | off | **`OK`** (expected) | positive control passes; runner plumbing is healthy |
| D4 | text-only `random` | off | on | **`OTHER_FAILURE`** | bench-client failure, **not** a server crash (see §4) |

For D1 and D2 the server log shows PCG capture completing in ~20 s and both
prompts being served via `cuda graph: True` prefill batches with `200 OK`
responses. For D4 the server-side PCG capture also completed cleanly
(`Capture piecewise CUDA graph end. Time elapsed: 22.90 s`) and the harness
smoke probe succeeded (`Prefill batch ... cuda graph: True` + `200 OK` +
`The server is fired up and ready to roll!`).

## 3. Why D1/D2 *not* reproducing does **not** invalidate the Stage 4.2 crash

The Stage 4.2 `IMG_A_S2_ipc_pcg` crash and the D1/D2 success differ in three
serving-side dimensions, any of which can shift the runtime past the captured
graph state:

1. **Request count.** Stage 4.2 ran 30 warmup + 400 measured requests (430
   total). D1/D2 ran 0 warmup + 2 measured (2 total). A 215× difference.
2. **Output length and decode load.** Stage 4.2 used `output_len=128` (each
   request decodes 128 tokens after prefill). D1/D2 used `output_len=32` (4×
   shorter). The PCG path is captured for `EXTEND` mode prefill, but the
   ratio of decode-vs-prefill cudagraphs taken in flight differs.
3. **Server steady-state cache + KV pool.** With 430 requests at c=1, the
   radix cache and KV pool reach a settled regime quite different from the
   first 2 requests. Internal pool size, `cached_token` hit rate, and
   per-request `#new-token` distribution all shift.

The static-audit hypothesis (Dynamo runtime recompile on the multimodal
forward path) is **shape-driven**, not "fires on first prefill." With only 2
seeded prompts both inputs happen to fall into the same captured size bucket
(vision=882 + text≈140 → ~1022 tokens, well within the captured size 1024).
With 430 prompts the input lengths drift across multiple captured sizes
(input bucket boundaries at 960, 1024 from the server log
`piecewise_cuda_graph_tokens=[…, 960, 1024, 1280, …]`), and Dynamo can
trigger a recompile for shapes that the warmup-and-capture phase did not
record into the cudagraph for that specific FX submod.

So **the D1/D2 success says only**: "at small sample size, image + PCG (± IPC)
does not crash on the first prefill on this upstream main HEAD." It does
**not** say: "image + PCG never crashes." The Stage 4.2 crash remains the
authoritative observation; D1/D2 narrow the trigger to a sample-size /
sequence regime that the tiny probe does not cover.

## 4. Why D4 is invalid for ruling on text-only PCG

D4 wanted to answer: "does upstream `main` PCG itself regress on text-only?"
The matrix says `OTHER_FAILURE`, but the failure is **client-side**:

```
huggingface_hub.errors.LocalEntryNotFoundError: An error happened while
trying to locate the file on the Hub and we cannot find the requested files
in the local cache.
```

The bench client (`python -m sglang.bench_serving --dataset-name random`)
loads the model tokenizer via `huggingface_hub` to synthesize random prompts;
under `HF_HUB_OFFLINE=1` and the local HF cache layout this resolves to a Hub
lookup that fails. Meanwhile the SGLang server side completed PCG capture
(22.9 s) and served a successful prefill plus `200 OK` chat completion
(visible in the server log excerpt in `D1234_summary.md`).

Net: D4 weakly suggests upstream main text-only PCG **starts** correctly and
**handles at least one prefill**, but the bench probe never ran the actual
2-prompt benchmark loop. To formally rule on "text-only PCG regresses or
not" we need a bench path that does not touch HF Hub. The simplest is
`--dataset-name autobench --dataset-path datasets/qwen3vl8b/caseA_short.jsonl`,
which is exactly what Issue #2's Case A runner uses
(`experiments/qwen3vl8b/v2/caseAC_rebaseline/run_caseAC_rebaseline.py:159-163`)
and is fully offline — the JSONL is checked into this repo and consumed by
the `autobench` reader without any tokenizer download.

## 5. Remaining hypotheses (post-D1–D4)

- **H1 — VLM image + PCG requires sequence/shape variance to trigger Dynamo
  recompile.** At 2 prompts no recompile happens; at 400 prompts (Stage 4.2)
  the input-length distribution + decode load exposes shape variations that
  cause a Dynamo recompile, and the recompiled subgraph reaches
  `CUDAPiecewiseBackend.__call__` out of capture phase → assertion. Static
  audit's underlying mechanism (multimodal forward + Dynamo recompile) is
  still the leading explanation; only the "fires on first prefill" sub-claim
  was wrong.
- **H2 — IPC + PCG interaction only matters at larger scale.** Stage 4.2 had
  IPC on; D1 had IPC on but did not trigger. We cannot yet say IPC matters at
  the 400-prompt scale; need a size-matched IPC-off control once we have a
  reproducing size.
- **H3 — Upstream `main` text-only PCG may still be fine, but D4 did not
  test it cleanly.** Need an offline-safe text-only run before drawing any
  text-vs-image conclusion.
- **H4 — Stage 4.2 crash could be intermittent at the same 400-prompt
  config.** Possible but lower probability given Stage 4.2 was a deterministic
  hit at rep 1. Worth confirming with one exact replay of the same
  config/seed before committing to any upstream issue.

## 6. Next stages overview (detail in `next_debug_plan.md`)

- **E1** — text-only PCG control on `/data/sglang-pr` main, using
  `autobench + caseA_short.jsonl` (no HF Hub access). Targets H3.
- **E2** — image + IPC + PCG reproduce ladder at growing
  `num_prompts ∈ {32, 64, 100/200, 400}` to find the smallest size that
  triggers the assertion. Targets H1 and H4.
- **E3** — image + **no IPC** + PCG at the smallest reproducing size from E2.
  Targets H2 (does dropping IPC at that size still crash?).
- **E4** — optional exact formal replay of Stage 4.2's S2 config (n=400,
  warmup=30, output=128, seed=1) if E2 ladder is inconclusive. Targets H4
  directly.
- **E5** — decision step (no GPU): map E1–E4 outcomes to actions
  (upstream issue scope / continue #4 without PCG / further debug).

`scripts/run_pcg_capture_stream_debug.py` works as-is for E1 (after fixing
the dataset path) and for E2 (after parameterising `num_prompts`,
`warmup`, `output_len`). The detailed runner changes and per-stage configs
go in `next_debug_plan.md`.

## 7. Out-of-scope here

- IMG-B / IMG-C remain paused.
- No upstream SGLang PR work until E1–E3 (and possibly E4) classify the
  trigger. No upstream issue filing yet either — the description we would
  file should be backed by a deterministic minimal repro, which we do not
  have at debug-level until E2 lands.
- No modification to `/data/sglang-pr` source. If a code-change experiment
  is ever approved, it goes on a fresh worktree, not in place.
- No further generator-related work — that blocker is closed.
