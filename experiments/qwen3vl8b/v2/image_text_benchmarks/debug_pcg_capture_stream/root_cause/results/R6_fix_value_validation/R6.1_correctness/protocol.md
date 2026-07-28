# R6.1 — Correctness gate + mixed-modality safety protocol

> Written and committed **before** any leg is executed. The verdict
> rules in §7 are pre-declared and enforced by
> [`scripts/R6_1_verdict.py`](../../../scripts/R6_1_verdict.py). Any
> revision to §7 after seeing results requires a `docs(v2): amend R6.1
> verdict rules` commit and re-classification as `R7_REQUIRED`.

## 1. Purpose

Formally attribute (or rule out) each of the following about the clean-Y
fork at HEAD `986c89e69`:

- Does the fix perturb the PCG-off code path? (must be no-op)
- Does the fix perturb the text-only PCG path? (must not regress)
- Does the fix produce silently corrupted image outputs? (must not)
- Does the fork-PCG server survive interleaved text ↔ image traffic
  without any crash, assertion, eager fallback, or inference-time
  Dynamo recompile of `qwen3_vl.forward`?

Correctness is claim (1) in the R6 goal set (see `plan.md` §5b).
R6.2 / R6.3 / R6.4 / R6.5 are blocked on this gate passing.

## 2. Frozen inputs

- **Provenance tuple**: exactly the (stock, fork, snapshot, env,
  dataset) frozen in
  [`../R6.0_provenance.md`](../R6.0_provenance.md). The runner
  re-verifies the two SHAs before every launch and refuses to proceed
  if they drift.
- **Fixture image**: [`fixtures/R6.1_fixture.png`](fixtures/R6.1_fixture.png)
  — a 1280×720 deterministic PNG with three vertical solid-color bars
  (muted red / muted green / muted blue on white background).
  Regeneration recipe: [`fixtures/gen_fixture.py`](fixtures/gen_fixture.py).
  SHA-256 pinned in
  [`fixtures/R6.1_fixture.sha256`](fixtures/R6.1_fixture.sha256):
  `79c47c91070abcbae0dbc8bd983ec5b5f3bf37f450d535ac220b95e0fb74c967`
  (4333 bytes). Determinism verified 2026-07-28 by regenerating and
  byte-comparing.
- **Prompt set**: [`fixtures/prompts.json`](fixtures/prompts.json)
  — 3 image prompts, 3 text-only prompts, sampling
  `{temperature: 0, top_p: 1, seed: 42, max_tokens: 128}` (greedy).
- **GPU**: R6.1a does not select a GPU. R6.1b receives an explicit
  approved GPU ID from the user; the runner refuses to launch
  without it and re-checks idle (`memory.used ≤ 500 MiB`,
  `utilization.gpu ≤ 5%`, `compute-apps == 0`) before every server.
- **Servers**: `python3 -m sglang.launch_server`, one at a time
  (never co-resident), fully torn down between variants with
  GPU-memory-drain wait.
- **Flags per variant** (canonical, from R6.0):

  | Variant | Extra flags | Import path |
  |---|---|---|
  | stock-default | — | default (`/sgl-workspace/sglang/...`) |
  | stock-PCG | `--enforce-piecewise-cuda-graph` | default |
  | fork-default | — | `PYTHONPATH=/data/sglang-fork/python` |
  | fork-PCG | `--enforce-piecewise-cuda-graph` | `PYTHONPATH=/data/sglang-fork/python` |

  Common: `--dtype bfloat16 --tp 1 --attention-backend flashinfer
  --port 30003`. `SGLANG_USE_CUDA_IPC_TRANSPORT=1`.
  `TORCH_LOGS=recompiles` is set for every R6.1 leg so the server log
  captures Dynamo recompiles (this is the correctness gate; the
  overhead is acceptable at this scale). KAPI / profiler env vars
  are unset before every launch.

## 3. Legs

| Leg | Server | Client mode | What it exercises |
|---|---|---|---|
| **a** | fork-default | `image` × 2 sequential runs on same server | same-backend deterministic repeat (rules out client / sampling non-determinism BEFORE any cross-backend comparison) |
| **b** | fork-PCG | `image` | isolates PCG-vs-eager numerical delta introduced by the fix (compared against leg a) |
| **c** | stock-default | `image` | fix-off code path is unaffected (compared against leg a) |
| **d** | stock-PCG | `text` | stock's text-only PCG baseline (avoids known image crash) |
| **d'** | fork-PCG | `text` | fix does not perturb the text-only PCG path (compared against leg d) |
| **e** | fork-PCG (same server as b, d') | `interleaved` `text→image→text→image→text` | mixed-modality operational safety subtest — the ONE claim stock has no equivalent for |
| **f** | stock-default + stock-PCG on `text` only | diagnostic | characterises stock's own PCG-vs-eager delta on the text path. Does NOT change the verdict on its own; used only to enrich R7 investigation if verdict is AMBIGUOUS. |

**Server-launch ordering** (serialized, one at a time):

1. stock-default → runs leg c + leg f₁
2. fork-default → runs leg a run 1 + leg a run 2
3. stock-PCG → runs leg d + leg f₂
4. fork-PCG → runs leg b + leg d' + leg e + safety-tally (leg e)

Each variant tears down fully (SIGTERM → SIGKILL fallback → GPU-memory
drain up to 60 s) before the next launches.

## 4. Fixture image content (canonical description)

The fixture is a 1280×720 PNG with **three solid vertical color bands**
covering the full height:

- Left third (0–426 px): muted red `RGB(220, 40, 40)`.
- Middle third (427–853 px): muted green `RGB(40, 180, 60)`.
- Right third (854–1279 px): muted blue `RGB(40, 80, 200)`.

No text, no other shapes, no antialiasing artefacts (single-pixel-set
loop). Any output describing the image should mention "three colored
bars / vertical color bars / red / green / blue" or similar. Cross-leg
matches are byte-comparison on the model's generated text — the model's
actual descriptive accuracy is not part of the verdict; only that
different servers on the same input produce the same output (or a
divergence attributable to a matched control).

## 5. Prompts (canonical, from `fixtures/prompts.json`)

Image prompts (used in legs a, b, c, and image-turns of leg e):

1. `"Describe this image in one sentence."`
2. `"What colors are visible in the image?"`
3. `"How many distinct regions of solid color are present, and what colors are they?"`

Text-only prompts (used in legs d, d', f₁, f₂, and text-turns of leg e):

1. `"In one sentence, what is the color red typically associated with in western cultural symbolism?"`
2. `"List three primary additive colors used in digital displays."`
3. `"What is 12 multiplied by 8? Give only the numeric answer."`

Interleaved order for leg e (5 requests on the fork-PCG server):

- text 0 → image 0 → text 1 → image 1 → text 2

## 6. Metrics collected per leg

For every request in every leg, the client (`scripts/R6_1_client.py`)
records:

- `idx`, `kind ∈ {image, text}`, verbatim prompt.
- `response_text` (the model's assistant message content, string).
- `finish_reason`, `usage`, `latency_s`, `http_status`, `error`.

The server-side log tail (only the safety-relevant tail — no full log
committed) is grepped for:

- `AssertionError: PCG capture stream is not set` → `assertions` count.
- `Falling back to eager execution` → `fallbacks` count.
- `Recompiling function.*qwen3_vl` (`TORCH_LOGS=recompiles` output) →
  `inference_recompiles` count. Note: this count includes any warmup
  recompiles the fork's clean-Y compile pass produces intentionally;
  R6.1 verdict does not currently distinguish warmup vs inference
  recompiles by line. R7 will refine if needed.

## 7. Pre-declared verdict rules

These rules are the authority for the R6.1 verdict and are enforced by
[`scripts/R6_1_verdict.py`](../../../scripts/R6_1_verdict.py). Any post-
hoc revision requires a `docs(v2): amend R6.1 verdict rules` commit
and R7_REQUIRED classification.

**PASS** iff **ALL** of the following hold:

- **(a)** `texts(leg_a_fork_default_run1) == texts(leg_a_fork_default_run2)`
  bit-identical for every prompt.
- **(b)** `texts(leg_b_fork_pcg_image) == texts(leg_a_fork_default_run1)`
  bit-identical for every prompt.
- **(c)** `texts(leg_c_stock_default_image) == texts(leg_a_fork_default_run1)`
  bit-identical for every prompt.
- **(d)** `texts(leg_d_stock_pcg_text) == texts(leg_dprime_fork_pcg_text)`
  bit-identical for every prompt.
- **(e)** `safety_summary.assertions == 0` AND
  `safety_summary.fallbacks == 0` AND
  `safety_summary.inference_recompiles == 0` AND
  `safety_summary.request_failures == 0`.
- No leg has an HTTP error, request exception, or non-200 response.

**FAIL** iff **ANY** of the following hold:

- Any client leg has a request error / non-200 HTTP status / request
  exception.
- (a) not bit-identical → sampling / determinism broken; whole gate
  invalidated regardless of other legs.
- (c) not bit-identical → the fix is not a no-op on the PCG-off path
  (unacceptable regardless of PCG behaviour).
- (d) not bit-identical → the fix perturbs the text-only PCG path.
- (e) any assertion / fallback / inference recompile / request failure.

**AMBIGUOUS / R7_REQUIRED** iff **BOTH**:

- (a), (c), (d), (e) all pass, **but**
- (b) is not bit-identical.

In this case the divergence is on the image path only, and could be
either normal PCG-vs-eager bf16 kernel-numerics noise or a residual
correctness bug. Per user directive, this must **not** be automatically
labelled as either — it is AMBIGUOUS and must be handed off to R7 for
matched-control investigation.

Diagnostic (**never** changes the verdict on its own): leg **f** —
stock-default text-only vs stock-PCG text-only. If (f) diverges, we
have partial evidence that "PCG-vs-eager on this model produces some
observable delta." If (f) is bit-identical, then leg b's divergence
under AMBIGUOUS cannot be explained even by stock's own PCG behaviour
and R7 must investigate first.

## 8. Amendment rule

Anything in §2 (frozen inputs), §3 (legs), §4/5 (fixture / prompts), or
§7 (verdict rules) that changes after the runner is first executed
requires:

1. A `docs(v2): amend R6.1 <what>` commit explaining the change and
   the reason.
2. Re-execution of every affected leg.
3. Explicit classification as `R7_REQUIRED` if the amendment was
   triggered by observing results.

The runner is deliberately non-parameterised for the correctness
inputs — the fixture path, prompt file, port, model snapshot, and
provenance SHAs are all hard-coded in
[`scripts/run_R6_1_correctness.sh`](../../../scripts/run_R6_1_correctness.sh)
so that a change is visible as a diff, not a silent flag flip.
