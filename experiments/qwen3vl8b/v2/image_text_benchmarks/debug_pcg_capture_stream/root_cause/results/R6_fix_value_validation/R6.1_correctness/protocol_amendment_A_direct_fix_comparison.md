# R6.1 Protocol Amendment A — direct fix comparison (2026-07-28)

> **This amendment supersedes** the parts of
> [`protocol.md`](protocol.md) explicitly named below. All other parts
> of `protocol.md` remain in force. The historical protocol is not
> rewritten (attempts 01 and 02 were evaluated under it and their
> verdicts stand as recorded).
>
> Attempt 03 and onward execute under this amendment. Any further
> change to the sections below requires a **new** amendment file
> (`protocol_amendment_B_*.md`), not an edit to this one, so the
> evolution stays auditable.

## Scope and objective

R6.1 attempts 01 and 02 established solid operational safety
evidence for the clean-Y fix but could not deliver an unambiguous
correctness verdict because:

1. The pre-declared `inference_recompiles` metric mixed startup /
   warmup recompiles with true inference-time recompiles (protocol
   §6 acknowledged this imprecision).
2. The pre-declared "bit-identical" rule compared **different cache
   states** (cold prompt 2 vs warm prompt 2 within one server) as
   if they were a determinism baseline.
3. The historical stock-PCG-crashes-on-first-image failure was
   never re-exercised on the current frozen stock SHA `da802dd`
   during attempts 01 or 02.

This amendment rebuilds R6.1 around a **three-tier evidence
ladder** so that (a) the fix's operational safety superiority over
`stock-PCG` is proven directly, (b) text non-regression is proven
under matched cache state, and (c) any workload performance win
that survives independent confirmation can be claimed at its own
tier rather than folded into a monolithic PASS.

Definitions (fixed for the rest of R6):

- **`stock-default`** — installed SGLang at `da802ddca...` in
  `/sgl-workspace/sglang`, PCG auto-disabled for Qwen3-VL by the
  upstream multimodal gate. Default server flags.
- **`stock-PCG`** — same installed SGLang, launched with
  `--enforce-piecewise-cuda-graph`. This is the **previous
  behavior** that crashes when the first image reaches the model
  (see Issue #4 debug).
- **`fork-PCG`** — final clean-Y / static-buffer fork at
  `986c89e69...`, sourced via `PYTHONPATH=/data/sglang-fork/python`,
  launched with `--enforce-piecewise-cuda-graph`. This is the
  **modification under evaluation**.

The frozen environment tuple in [`../R6.0_provenance.md`](../R6.0_provenance.md)
(including amendments A1 / A2 / A3) still governs stock/fork SHAs,
model snapshot, dataset SHAs, and the `LD_PRELOAD` host-libcuda pin.

## §2.1 Fix the recompile measurement

The runner must emit **phase markers** so log analysis can
attribute each recompile to a specific phase, not to the whole
run.

### Phase markers (emitted by the runner into each server log)

- `R6_MARK SERVER_READY <server_variant>` — written the moment the
  HTTP readiness probe first succeeds for that server.
- `R6_MARK LEG_START <leg_id>` — written immediately before the
  first client request of a leg (regardless of client vs server
  process; the runner writes the marker to the server log via a
  tiny sentinel request or via a side-channel file the tally reads).
- `R6_MARK LEG_END <leg_id>` — written immediately after the last
  client request of a leg completes.

Implementation note: sglang.launch_server does not expose a hook
to write arbitrary lines into its own stdout, so the runner writes
marker lines to a side-channel file
`raw/<server_variant>_phase_markers.txt` (one line per marker
with the UTC timestamp). The tally script joins these markers
against `raw/<server_variant>_server.log` by timestamp.

### Reported recompile counts (independent, in `safety_summary.json`)

- `startup_warmup_recompiles` — recompiles at server-log lines
  emitted at or before the `SERVER_READY` timestamp. **Never
  fails the safety gate.**
- `post_ready_recompiles` — recompiles between `SERVER_READY` and
  the last recorded `LEG_END` for this server. **Fails the
  safety gate if any recompile falls inside a `LEG_START` →
  `LEG_END` interval** (i.e. actually during a measured leg).
- Per-leg recompile counts: for each leg on this server, the
  number of recompile log lines with timestamp in
  `[LEG_START, LEG_END]`.

**Gate rule:** only recompiles inside a `[LEG_START, LEG_END]`
interval that is itself after `SERVER_READY` may fail the gate.
A recompile between `SERVER_READY` and the first `LEG_START`
(e.g. warmup that continues after HTTP ready) is reported but
does not fail.

## §2.2 Cache-matched correctness controls

**Radix caching stays enabled in the primary production-shaped
test** — the original bug occurred on this path and we must
validate the fix under the same conditions operators run.

`--disable-radix-cache` may only be used as a **diagnostic
ablation**, i.e. an additional leg specifically labelled
"radix_off" whose result cannot substitute for the primary
matched-cache verdict.

### Matched-repeat structure

For every correctness comparison whose verdict rule is
"outputs should match", the runner must run **matched pairs**
launched on **fresh servers** (fresh server = new
`sglang.launch_server` process, so radix cache starts empty).
Within a matched pair both requests are the **first** leg on
their respective fresh server, so both hit the cold cache.

Minimum matched repeats added by this amendment:

| Repeat ID | Variant | Purpose |
|---|---|---|
| `stock_default_image_cold_x2` | stock-default × 2 fresh servers | cold-cache determinism baseline for image on stock-default |
| `fork_default_image_cold_x2` | fork-default × 2 fresh servers | cold-cache determinism baseline for image on fork-default |
| `stock_pcg_text_cold_x2` | stock-PCG × 2 fresh servers | cold-cache determinism baseline for text on stock-PCG |
| `fork_pcg_text_cold_x2` | fork-PCG × 2 fresh servers | cold-cache determinism baseline for text on fork-PCG |
| `fork_pcg_image_cold_x2` | fork-PCG × 2 fresh servers | cold-cache determinism baseline for image on fork-PCG |

Legacy legs that previously ran "second-leg-on-same-server" (e.g.
attempt 02's `a2`, `f1`, `f2`, `dp`) are demoted to
**diagnostic-only** (kept for provenance and warm-cache
characterisation) and cannot be used as the determinism baseline.

Cache-flush endpoints (e.g. `POST /flush_cache`) are permitted as
a substitute for a fresh server **only if a same-server matched
pair with a cache flush between them can be shown to reproduce
the fresh-server baseline within noise**. Until that is
demonstrated on this stack, fresh servers are the only accepted
mechanism.

### Comparison metrics (added, in addition to exact-equality)

Every correctness comparison records, per-prompt:

- Exact text equality (existing).
- Generated token IDs via the model tokenizer.
- First differing token index (or `None`).
- Common-prefix token count.
- Normalized token Levenshtein distance
  (`levenshtein / max(len_a, len_b)`).
- Response character length, response token length.
- Finish reason.
- Top-k / logprob data **only if** a preflight verifies the local
  OpenAI-chat API supports the request (`logprobs: true` +
  `top_logprobs: N`). If the preflight fails, this field is
  omitted with a recorded reason.

Semantic (LLM-as-judge) evaluation is **supplementary only** and
cannot substitute for token-level equality in the pre-declared
verdict rules.

## §2.3 Direct stock-PCG image negative control

The runner adds one new leg: `neg_stock_pcg_image` — the exact
same fixture and image prompts as `leg_b_fork_pcg_image`, but
served by a fresh `stock-PCG` server.

Classification (recorded verbatim into
`raw/negative_control_classification.txt`):

- **`EXPECTED_STOCK_FAILURE`** — stock-PCG's server log shows
  `AssertionError: PCG capture stream is not set` OR the first
  image request returns a non-200 / connection error / timeout,
  matching the historical Issue #4 signature. The negative
  control **reproduces** the previous behavior.
- **`STOCK_NOW_SURVIVES`** — stock-PCG serves the image legs
  cleanly with HTTP 200 and no assertion. This **changes the
  value claim** for the fix; requires investigation before any
  `SAFETY_SUPERIORITY_PASS`.
- **`UNRELATED_FAILURE`** — stock-PCG fails for a reason
  other than the historical capture-stream assertion (e.g.
  OOM, driver error, our own preflight rejection). Cannot
  support the superiority claim; requires investigation.

### Isolated crash handling

An **expected** stock-PCG server crash must be scoped to that
server's recorded PGID. The runner:

1. Records the PGID at server launch (same PGID discipline as
   R6.0 Amendment A2).
2. When the negative control leg fails, the runner waits for the
   server process to exit on its own (up to a bounded timeout);
   if the process is still alive, sends `SIGTERM` **only** to
   that recorded PGID after re-verifying ownership.
3. Reads the server log tail to classify the outcome.
4. Continues to the next server (fork-PCG) with GPU pre-check.

No `pkill` / `killall` / broad cleanup at any point. The rest of
R6.0 Amendment A2's cleanup contract stands unchanged.

## §2.4 Predeclared verdicts

The three tiers are evaluated independently. The overall R6.1
verdict is the strongest tier that both PASSES and does not
carry an unresolved dependency.

### Tier 1 — `SAFETY_SUPERIORITY_PASS`

PASSES iff **all** of:

- `neg_stock_pcg_image` produces `EXPECTED_STOCK_FAILURE`
  (stock-PCG reproduces the historical first-image capture-
  stream failure).
- `leg_b_fork_pcg_image` completes the same request sequence with
  every request HTTP 200.
- The fork-PCG mixed-modality interleaved leg
  (`leg_e_fork_pcg_interleaved`) completes with:
  - `request_failures == 0`
  - `assertions == 0` (grep `AssertionError: PCG capture stream is not set`)
  - `fallbacks == 0` (grep `Falling back to eager execution`)
  - `post_ready_recompiles inside any [LEG_START, LEG_END] == 0`
    (per §2.1; warmup recompiles between server boot and
    `SERVER_READY` do not fail this gate; recompiles between
    `SERVER_READY` and the first `LEG_START` do not fail it
    either).

FAILS iff:

- `neg_stock_pcg_image` produces `STOCK_NOW_SURVIVES` (the
  historical failure no longer reproduces; the value claim for
  the fix must be re-derived).
- `neg_stock_pcg_image` produces `UNRELATED_FAILURE` (the
  negative control did not exercise the historical failure
  path; investigation required).
- The fork-PCG interleaved leg surfaces any of the four safety
  signals above.

### Tier 2 — `CORRECTNESS_PASS`

PASSES iff **all** of:

- `stock-default vs fork-default` (matched cold-cache image
  repeats) is either bit-identical or the divergence lies
  within the `stock_default_image_cold_x2` determinism envelope.
- `stock-PCG vs fork-PCG text` (matched cold-cache text repeats)
  divergence does not exceed the union of
  `stock_pcg_text_cold_x2` and `fork_pcg_text_cold_x2` envelopes.
- `fork-default vs fork-PCG image` (matched cold-cache image
  repeats) divergence does not exceed the union of
  `fork_default_image_cold_x2` and `fork_pcg_image_cold_x2`
  envelopes.
- **No stable fork-specific token divergence remains
  unexplained** — every fork-vs-stock divergence has an
  attribution in a same-config determinism envelope.

FAILS iff any of those hold with a divergence outside the
matched envelope.

### Overall R6.1 outcome

- `PASS` iff **both** `SAFETY_SUPERIORITY_PASS` **and**
  `CORRECTNESS_PASS`.
- `SAFETY_PASS_CORRECTNESS_AMBIGUOUS` iff
  `SAFETY_SUPERIORITY_PASS` but `CORRECTNESS_PASS` cannot be
  reached (divergences exceed envelopes but no fork-specific
  stable divergence was proven either).
- `FAIL` iff `SAFETY_SUPERIORITY_PASS` fails.

**Performance claims** (`WORKLOAD_PERFORMANCE_WIN`, R6.3 territory)
require the overall R6.1 outcome to be `PASS`, not
`SAFETY_PASS_CORRECTNESS_AMBIGUOUS`.

### Envelope definition

For each `_cold_x2` matched repeat, on each prompt, compute the
per-prompt token Levenshtein distance between the two runs.
The **envelope** for that (variant, modality, prompt) is
`[0, max(k, tok_lev)]` where `k` is a small pre-declared
minimum floor (default `k=2` tokens to allow for genuinely
tied greedy decoding with occasional last-token variance).

A cross-config comparison for the same (modality, prompt) is
**inside the envelope** iff its tok_lev ≤ envelope max.

This intentionally lets cross-config differences slide *as long
as* the corresponding same-config repeat also varies by at least
that much. If the same-config repeat is bit-identical and the
cross-config comparison is not, the cross-config comparison is
**outside** the envelope and must be attributed.

## §2.5 Reporting

Every R6.1 attempt (03 onward) records, in a machine-readable
`raw/verdict_amended.json` (in addition to the legacy
`verdict.json`):

- `attempt_id`, `attempt_dir`, `stock_sha`, `fork_sha`,
  `fixture_sha`, `dataset_sha`, `nvidia_driver`, `host_libcuda`,
  `attempt_ts_utc_start`, `attempt_ts_utc_end`, `gpu_id`.
- `negative_control` — the stock-PCG image classification result.
- `safety_metrics` — `startup_warmup_recompiles`,
  `post_ready_recompiles`, `per_leg_recompiles`, `assertions`,
  `fallbacks`, `request_failures`.
- `envelope` — per (variant, modality, prompt) matched-repeat
  Levenshtein envelope.
- `cross_config_comparisons` — for each cross-config comparison,
  per-prompt Levenshtein + inside-envelope boolean.
- `tier_verdicts` — `safety_superiority`, `correctness` — each
  `PASS` / `FAIL` / a specific reason string.
- `overall_verdict` — `PASS` / `SAFETY_PASS_CORRECTNESS_AMBIGUOUS`
  / `FAIL`.
- `evidence_tier_claimed` — `SAFETY_SUPERIORITY` or stronger.

Human-readable `verdict_amended.md` summarises the same, with
per-leg detail and per-comparison tables.

Downstream tiers (R6.2 non-regression, R6.3 workload search)
proceed only if the overall R6.1 outcome is `PASS`. In
`SAFETY_PASS_CORRECTNESS_AMBIGUOUS` the fix can still be claimed
at the `SAFETY_SUPERIORITY` tier for framing purposes, but R6.2
and R6.3 must not run without a further amendment authorising
them under that state.

## Amendment audit

Any change to §2.1–§2.5 requires a new amendment file
(`protocol_amendment_B_*.md`, `protocol_amendment_C_*.md`, …).
The current file **does not modify** `protocol.md`, so the
attempt 01 / 02 verdicts remain evaluable under their original
rules. Attempts 03+ evaluate under §2.1–§2.5 of this file, in
addition to the parts of `protocol.md` not superseded above.
