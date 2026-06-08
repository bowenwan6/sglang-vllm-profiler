# PCG capture-stream debug — experiment plan (D1–D6)

> **Status: PLAN ONLY — no GPU experiments have been executed.** Companion to
> [`static_audit.md`](static_audit.md) (which classifies this as expected
> unsupported behavior under `--enforce-piecewise-cuda-graph` on a VLM, based
> on source reading alone). The matrix below converts that hypothesis into
> falsifiable runs.

## Constraints

- GPU 7 only, never auto-switch.
- All experiments use the fixed-generator path (`PYTHONPATH=/data/sglang-pr/python`,
  `/data/sglang-pr` HEAD `62c505a196`, merged generator fix `07f326c184` in
  history). Stage 4.1 smoke already passed on this path; no need to re-verify
  generator correctness here.
- Do **not** modify `/data/sglang-pr` source. If a code-change experiment is
  ever approved, it goes on a fresh branch in a separate worktree, not in
  place.
- No KAPI (`SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST` must be
  unset). No profiler.
- Each stage uses a **tiny** workload (≤ 8 requests). The goal is repro /
  isolation, not performance.
- Each stage launches a fresh server, runs the bench, and kills the server.
  Server is never co-resident with the next stage's server.
- Stop on first crash within each stage (do not retry). Stages execute
  sequentially in the order D1 → D2 → D3 → D4 (and only D5 / D6 if needed).
- Outputs under
  `experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/results/`
  and logs under
  `logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/`.
  Do not overwrite `results_fixed/`, `smoke_fixed/`, or any historical
  artifact.
- Commit only processed summaries + small aggregate JSON. Do not commit raw
  per-rep JSONL or server logs unless explicitly approved.

## Common request recipe (unless stated otherwise)

| field | value |
|---|---|
| dataset | `image` (synthetic, fixed generator) for image stages; `random` for text stage |
| backend | `sglang-oai-chat` |
| image | 720p, png, random, image-count 1 (image stages only) |
| `--random-input-len` | 128 |
| `--random-output-len` | 32 |
| `--random-range-ratio` | 1.0 |
| `--max-concurrency` | 1 |
| `--num-prompts` | 2 (tiny) |
| `--warmup-requests` | 0 |
| `--seed` | 1 |
| `--extra-request-body` | `{"temperature": 0, "top_p": 1}` |

Server flags (common): `--model-path <Qwen3-VL-8B-Instruct snapshot>
--dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer`.
PCG-on cases additionally pass `--enforce-piecewise-cuda-graph`. IPC cases
set `SGLANG_USE_CUDA_IPC_TRANSPORT=1`. KAPI vars always unset.

## Per-case classification

The debug runner labels every outcome as one of:

- `PCG_CAPTURE_STREAM_ASSERT` — server crashed with the
  `cuda_piecewise_backend.py:170` assertion (the specific failure under
  investigation).
- `FORBIDDEN_TOKEN_ERROR` — generator regression returned (bench client errors
  contain `No data iterator found for token`). Should not happen on the fixed
  path; included as a safety check.
- `SERVER_NO_START` — server died early or `/health` never responded within
  the per-case wait. Includes any non-PCG startup crash.
- `OK` — bench client returned cleanly, 0 failures, non-empty output.
- `OTHER_FAILURE` — anything else (bench rc != 0 for a non-PCG reason, parse
  error, etc).

## Stages

### D1 — Reproduce current failing combo (sanity)

| component | setting |
|---|---|
| model | Qwen3-VL-8B-Instruct (multimodal) |
| dataset | image + text (720p, image-count 1) |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | `1` |
| `--enforce-piecewise-cuda-graph` | yes |

Expected: `PCG_CAPTURE_STREAM_ASSERT` (matches the Stage 4.2 crash exactly).

Decision rule:
- If `PCG_CAPTURE_STREAM_ASSERT` → static-audit hypothesis is reproducible;
  proceed to D2 to factor out IPC.
- If `OK` → the Stage 4.2 crash was non-deterministic (rep-state or
  warmup-related). Treat as anomalous, retry with `num_prompts=8` before
  drawing any conclusion.
- If any other class → stop and re-audit; the picture is not what we think.

### D2 — Drop IPC, keep PCG (does IPC matter?)

| component | setting |
|---|---|
| model | Qwen3-VL-8B-Instruct |
| dataset | image + text |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | **unset** |
| `--enforce-piecewise-cuda-graph` | yes |

Expected: `PCG_CAPTURE_STREAM_ASSERT` again (assertion does not consult IPC
state; the fault is VLM-forward-vs-PCG, not IPC-vs-PCG).

Decision rule:
- If `PCG_CAPTURE_STREAM_ASSERT` → IPC is not a factor. Image+VLM+PCG is the
  failing combo on this upstream main.
- If `OK` → unexpected; IPC is a contributing factor. Re-audit IPC transport
  code's interaction with PCG (would invalidate the static hypothesis).

### D3 — Image+IPC, PCG OFF (positive control)

| component | setting |
|---|---|
| model | Qwen3-VL-8B-Instruct |
| dataset | image + text |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | `1` |
| `--enforce-piecewise-cuda-graph` | no |

Expected: `OK` (mirrors `IMG_A_S0_ipc`, which already completed 5/5 reps
cleanly; D3 is a tiny redundant check that the debug runner's plumbing is
sane).

Decision rule:
- If `OK` → debug runner is healthy. D1 + D2 + D3 jointly localize the fault
  to "VLM + PCG" regardless of IPC.
- If `PCG_CAPTURE_STREAM_ASSERT` → impossible by construction (PCG is off);
  treat as a debug-runner bug.
- If `SERVER_NO_START` or `OTHER_FAILURE` → unrelated regression appeared;
  stop and investigate before reporting D1/D2.

### D4 — Text-only random + PCG (does PCG itself regress?)

| component | setting |
|---|---|
| model | Qwen3-VL-8B-Instruct |
| dataset | **`random` (text-only)** — `--dataset-name random --random-input-len 128 --random-output-len 32` |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | unset (text path doesn't need IPC; matches #2 conditions) |
| `--enforce-piecewise-cuda-graph` | yes |

Expected: `OK`. The same model loaded but no `pixel_values`, so
`general_mm_embed_routine` either short-circuits to the pure-text path or is
never entered; the Dynamo recompile that fires on the mm path should not
occur.

Decision rule:
- If `OK` → fault localized to **VLM image path + PCG**. Upstream PCG itself
  on the same upstream main HEAD is not broadly regressed.
- If `PCG_CAPTURE_STREAM_ASSERT` → **broader upstream PCG regression**, not
  multimodal-specific. Static-audit conclusion must be revised. File an
  upstream SGLang issue independent of #4 and pause this debug.
- If `SERVER_NO_START` → text-only on this upstream main has a separate
  startup issue; report and stop.

### D5 — OPTIONAL: older SGLang text+PCG comparison

Run only if D4 has surprising behavior. Use `/sgl-workspace/sglang` (the older
SGLang at commit `0c8049d9b`, used by #2) on a text-only PCG run. **Never use
this for any #4 headline number** — it's just an upstream-drift comparator.

Skip by default. Includable later under explicit approval.

### D6 — OPTIONAL: reduced workload shape

If D1 unexpectedly fails to reproduce, try variations to isolate the runtime
shape contribution:

- Lower resolution (e.g. `--image-resolution 360p` if supported) to reduce
  prefill tokens from ~1024 to ~256.
- Shorter output (`--random-output-len 8`).
- Fewer prompts (`--num-prompts 1`).

Skip by default. Includable later under explicit approval.

## Output layout (per stage)

```text
experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/
  results/
    D1_summary.md
    D1_results.json
    raw/                        # raw bench JSONL, NOT committed
      D1_<case>_bench.jsonl
    (D2, D3, D4, …)

logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/
  D1_<case>_server.log          # NOT committed
  (D2, D3, D4, …)
```

The per-stage summary records:

- SGLang clone HEAD SHA + `sglang.__file__` + fix marker (re-checked per
  stage for provenance — must always be `FIX_OK` on
  `/data/sglang-pr/python/`).
- Server args at runtime (`disable_piecewise_cuda_graph`,
  `enforce_piecewise_cuda_graph`, `attention_backend`,
  `piecewise_cuda_graph_tokens`, multimodal flag).
- Env at runtime (`SGLANG_USE_CUDA_IPC_TRANSPORT`, KAPI vars confirmed
  unset).
- Exact bench client command (sanitized for committed log).
- Case classification (one of the 5 above).
- Traceback head from the server log if the case failed.
- Elapsed time from server-up to case verdict.

## Commit / push discipline

- Commit + push after each completed stage with message
  `test(v2): run image-text pcg debug stage D<N>`.
- Only commit `results/D<N>_summary.md` and `results/D<N>_results.json`.
- Do **not** commit `results/raw/` or `logs/.../debug_pcg_capture_stream/`.
- Do **not** stage `.claude/settings.local.json`, report line-ending diffs,
  unrelated working-tree drift, or scheduled-task locks.
- No `Co-Authored-By`. No Claude / Anthropic / AI in commit subjects, bodies,
  or trailers.

## Decision matrix once D1-D4 are in

| D1 | D2 | D3 | D4 | Interpretation | Suggested next action |
|---|---|---|---|---|---|
| ASSERT | ASSERT | OK | OK | VLM + PCG specifically unsupported on this upstream main. IPC not a factor. | File upstream SGLang issue: extend HIP fallback to CUDA OR make `--enforce-piecewise-cuda-graph` warn loudly on VLMs. Continue #4 without PCG. Do not PR. |
| ASSERT | OK | OK | OK | PCG fails only when IPC is also on. Image+VLM+PCG alone is fine. | File upstream issue focused on IPC transport + PCG interaction. Possibly investigate a minimal IPC code-path fix for a PR. |
| ASSERT | ASSERT | OK | ASSERT | Broader upstream PCG regression on this HEAD (not VLM-specific). | File upstream issue about PCG regression in general. Pause #4. Do **not** PR until the regression has a clear minimal fix. |
| OK | ANY | ANY | ANY | The Stage 4.2 crash is intermittent. | Retry D1 with larger sample (e.g. 8 prompts) before drawing any conclusion. |
| any other | … | … | … | Out-of-hypothesis. Stop and re-audit before continuing. | — |

## Out-of-scope (this debug)

- Performance comparisons (smoke / IMG-A perf numbers). Use Stage 4.2's
  `IMG_A_S0_ipc` clean data and any later recovery run for headlines.
- IMG-B / IMG-C runs.
- vLLM anchor comparison.
- Generator changes / `gen_mm_prompt` re-validation.
- Any code change inside `/data/sglang-pr` before the decision matrix
  declares a fix path.

After D1-D4: write
[`conclusion.md`](conclusion.md) summarizing the matrix outcome and the
specific next decision (issue / PR / continue-without-PCG). Stop there;
do not start SGLang branch / PR work without explicit approval.
