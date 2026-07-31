# Validation Plan — Qwen3.5-4B BCG DeepStack

> **Purpose.** Design a set of experiments that, run on current upstream
> SGLang `main` (`5f9b0db1` at plan time; re-verify at run time) with
> `Qwen/Qwen3.5-4B`, produces objective, per-request evidence sufficient
> to pick exactly one of the four outcomes in `plan.md` §7.5.
>
> **Non-goal.** This plan does not design a fix. It proves or disproves
> the problem before any fix is discussed. It also does not run any
> GPU workload; every runner it names is CPU-plannable, but GPU
> execution requires an explicitly authorised GPU ID.

## 1. Predeclared verdict shape

The plan must, at the end, emit exactly one verdict:

| Verdict | Trigger |
|---|---|
| **`PASS_H_B`** | BCG is available in server logs but `can_run_graph` (or equivalent) rejects image requests, so image requests demonstrably run through the eager path. Correctness is trivially preserved; DeepStack replay-side gap is not exercised. Upstream defect claim closed. |
| **`PASS_H_D`** | Image requests demonstrably execute BCG replay AND greedy tokens / logits / hidden states match the eager reference within the matched-noise envelope from the text-only control. Some code path we did not find in `source_audit.md` handles DeepStack. Upstream defect claim closed. |
| **`FAIL_H_A`** | Image requests execute BCG replay AND greedy tokens diverge from the eager reference beyond the matched-noise envelope, with tensor-level or hidden-state evidence isolating the divergence to the DeepStack-relevant layers (0–2) or to a zero-DeepStack signature. Upstream defect claim upgraded. |
| **`FAIL_H_C`** | Image requests crash with an assertion (`PCG capture stream is not set`, illegal memory), or emit `Falling back to eager execution` after warmup completed. Upstream defect claim upgraded. |
| **`AMBIGUOUS`** | Divergence exists but attribution fails (bf16 noise floor swallows the signal, no matched control succeeded, or eager reference itself was nondeterministic). |
| **`INFRA_FAILURE`** | Environment cannot be brought to the frozen provenance, a shared GPU disqualifies the run, or the runner aborts on a foreign-PID guard. Recorded, does not count for or against the hypothesis. |

The verdict is **declared before observing any result**. Post-hoc
softening requires an explicit written amendment recorded in
`hypothesis.md` §5 and re-committed.

## 2. Prerequisites the plan requires from every run

Every attempt records, before executing any request:

1. **Upstream SGLang HEAD SHA** at run time. If different from
   `provenance.md` §1, the discrepancy is logged and the run is
   labelled with the new SHA.
2. **HF model revision** actually loaded (from
   `sglang.launch_server`'s startup log or an equivalent snapshot
   probe). Must equal `provenance.md` §2.
3. **Framework versions** (torch, flashinfer, sgl_kernel) and
   `libcuda.so` path via `ldconfig`. Must match `provenance.md` §3
   or an explicit override flag.
4. **Server flags emitted verbatim** (BCG backend selection, allowlist
   membership, `enforce-piecewise-cuda-graph`, cache flags).
5. **`nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv`
   snapshot** at start / periodically / end — must show no foreign PID
   on the authorised GPU. Foreign PID mid-run → run classified
   `INFRA_FAILURE`.
6. **PGID** of every server / bench process the runner launched.
   Cleanup signals **only** those PGIDs and only after verifying every
   PID inside was recorded by the runner.

## 3. Evidence layers required (in order of increasing strength)

Any single layer in isolation is insufficient. The plan requires all
layers below Layer 0 for every attempt; Layer 3 is the minimum for a
FAIL verdict.

### Layer 0 — Execution-path evidence (required for every attempt)

We must know whether each request actually used BCG replay.

- **Server-log evidence.** Enable and grep:
  - `TORCH_LOGS=recompiles_verbose` (only for at least one confirmation
    run per configuration; production runs may keep it off).
  - The prefill-CG runner's per-request `cuda graph: True` / `cuda
    graph: False` line (the same signal used in PR #30872's validation).
  - The BCG capture-session banner (must be visible at server startup).
- **Runtime marker.** A tiny per-branch instrumentation patch,
  restricted to this branch and never staged into upstream, that logs
  from within `_execute_body_capture` and from within
  `replay_layer_forward` whether it took the BCG path and whether the
  incoming `layer_kwargs` contained `input_deepstack_embeds`. The patch
  is kept as a `.patch` file under `scripts/` and applied only for the
  duration of a run.
- **Reject the run's evidence** for any request whose BCG path we
  cannot classify — do not average it in.

### Layer 1 — Input-side evidence (required)

We must know that the request's DeepStack tensor was populated with
non-trivially-nonzero data.

- Before sending, the runner computes the tokenised prompt and image
  embedding via the same processor, or, equivalently, the runner
  requests SGLang to echo `input_deepstack_embeds.abs().sum()` back
  (via a small profiler-side hook — server-scoped, never committed
  upstream). A near-zero value on a real image is itself informative
  (means [F4] in `hypothesis.md` produced only zeros for this input,
  and the whole test is a no-op — must be excluded from a FAIL verdict).

### Layer 2 — Output-token evidence (required)

- **Greedy tokens** (`temperature=0`, `top_p=1`, deterministic sampler
  seed) from the BCG-active configuration must be recorded verbatim.
- **Reference greedy tokens** from the matched eager configuration on
  the same request must be recorded verbatim.
- **Diff** — first-differing token offset, count of differing tokens,
  and full sequence diff. Any diff at all in a greedy setup is
  meaningful (subject to the noise envelope below).

### Layer 3 — Tensor / hidden-state / logits evidence (required for FAIL)

Because bf16 numerics permit small divergences in a "same" setup,
Layer 2 alone cannot cleanly attribute a mismatch. FAIL requires at
least one of:

- **Per-layer hidden-state RMS comparison** at layers 0, 1, 2, 3, and
  the final layer, between eager and BCG runs on the identical batch.
  If the divergence is concentrated in layers 0–2, that is the
  DeepStack signature; if it appears only at the final layer, it is
  likely something else.
- **Logit-level comparison**: top-k logits at each decoded position,
  KL divergence, cosine similarity. Available if we can use
  `return_logprob=True` on both paths.
- **`input_deepstack_embeds`-in-graph probe.** With the branch-local
  instrumentation, tag the DeepStack tensor with a canary pattern
  before submitting; after the run, check whether the layer-body
  captured graph "saw" that pattern (via a lightweight test hook that
  hashes the read address's contents at replay time).

### Layer 4 — Statistical envelope (required)

- **Matched noise envelope.** Run the eager reference against itself
  (two independent replicas of the eager configuration on the same
  input). The measured token / hidden-state / logit divergence between
  those two runs is the noise floor. A BCG-vs-eager divergence larger
  than that noise floor by a documented factor (e.g., ≥ 3× the
  layer-0 RMS noise on the same prompt) is beyond noise.
- **CV bound on every reported metric.** ≤ 6 % on latency; ≤ 3× the
  layer-0 noise-floor RMS on hidden-state divergence.

## 4. Configurations under test

Every configuration serves `Qwen/Qwen3.5-4B` at the pinned revision.

| # | Label | Server flags (key) | Purpose |
|---|---|---|---|
| C0 | `eager_ref` | prefill CUDA graph disabled; radix cache disabled; explicit greedy sampler | Correctness reference. Two independent replicas C0a, C0b give the noise floor. |
| C1 | `bcg_default` | prefill CUDA graph = default breakable backend; no override flag | Production-likely BCG-on default for a Qwen3.5-4B multimodal server. |
| C2 | `bcg_enforce` | prefill CUDA graph = breakable + `--enforce-piecewise-cuda-graph` | Same code path as C1 with the historical Qwen3-VL enforce flag; used to distinguish [H_B] (auto-fallback) from [H_A] (silent DeepStack loss) — if C1 falls back but C2 does not, we know the fallback path exists and works. |
| C3 | `text_only_control` | Any of C0 / C1 / C2 on text-only requests | Confirms the framework serves text-only correctly on the same server; isolates DeepStack-specific regressions from generic BCG regressions. |
| C4 | `warm_cache_replica` | C1 with the same cache-warmed state as the request under test | Guards against warm/cold cache confounds. |

Each configuration runs the same request set (see §5 fixtures) with
the same PRNG seed, same greedy sampling, same batch-of-1 pacing,
and back-to-back with a scripted delay so KV-cache state does not
drift.

## 5. Fixtures

### 5.1 Deterministic prompts

- **Image prompt (P_IMG).** A short instruction ("Describe the
  colours in this image.") applied to a **byte-pinned PNG fixture**
  (`fixtures/image_bands.png`, 1280×720, three deterministic
  vertical colour bands with SHA-256 recorded in the fixture manifest).
  Chosen to make DeepStack tiles nonzero and easy to reason about.
- **Text prompt (P_TXT).** A single short instruction with no image;
  reused from `datasets/qwen3vl8b/caseA_short.jsonl` (an existing,
  SHA-256-pinned dataset). Used only as the C3 control.
- **Multi-token prompt (P_LONG).** ≥ 512 text tokens plus the same
  image, to exercise a shape not covered by the single-shape DeepStack
  warmup (`capture_num_tokens[-1]` per [F7]). This is critical
  because the source-level hypothesis specifically implicates non-max
  shapes.

Every fixture is committed under `experiments/qwen35_4b/fixtures/`
with a `manifest.json` that pins SHA-256, dimensions, and MIME type.
Scaffolding for this manifest lands in Part 5.

### 5.2 Request schedule

- Serial (max concurrency 1) to eliminate scheduler interference.
- Warmup: 30 requests using P_TXT only (never DeepStack) so any DeepStack
  branch trigger is provably the first-ever tensor-valued call, matching
  the historical failure mode.
- Measured: 30 requests per configuration, first 5 discarded as JIT /
  cache warmup.
- **Cold-cache repeats.** For each `C1_measured[i]`, run C0 immediately
  before with the same request against a fresh server to keep prefix
  cache and radix state matched.
- **Configuration order.** Randomised across attempts (but recorded)
  so no configuration systematically gets warmer buffers.

## 6. What each verdict requires as evidence

- **`PASS_H_B`** ← Layer 0 shows every image request took the eager
  path (`cuda graph: False`) despite BCG being available AND Layer 2
  shows greedy tokens match C0 exactly on every request.
- **`PASS_H_D`** ← Layer 0 shows every image request took the BCG path
  (`cuda graph: True`) AND Layer 2 shows greedy tokens match C0 within
  the C0a↔C0b noise envelope AND Layer 3 shows layer-0/1/2 hidden-state
  RMS divergence is inside the noise envelope AND Layer 1 confirms
  `input_deepstack_embeds` was non-trivially nonzero.
- **`FAIL_H_A`** ← Layer 0 shows every image request took the BCG path
  AND Layer 2 shows greedy tokens differ AND Layer 3 shows
  layer-0/1/2 divergence exceeds the noise envelope by the documented
  factor OR the layer-0/1/2 hidden-state pattern matches a zero-DeepStack
  signature (i.e. the divergence is exactly what one would expect if
  DeepStack were replaced by zeros).
- **`FAIL_H_C`** ← Any image request produces a server assertion or
  `Falling back to eager execution` log line at inference time (i.e.
  after warmup completed).
- **`AMBIGUOUS`** ← Any required layer's evidence is missing or
  degraded.
- **`INFRA_FAILURE`** ← Provenance mismatch not waived, foreign PID
  seen mid-run, or setup failure before any measurement.

## 7. Confounders and how the plan controls them

| Confounder | Control |
|---|---|
| bf16 nondeterminism | C0a↔C0b noise envelope in Layer 4; never claim FAIL below the envelope. |
| Warm-vs-cold KV cache | Fresh server per matched pair; matched request order; radix cache flag identical across paired runs. |
| Silent BCG-off | Server-log per-request `cuda graph:` line + branch-local instrumentation; every request classified before averaging. |
| Framework version drift | Provenance preflight (§2). |
| Prompt/image tokenisation drift | Byte-pinned image fixture, SHA-pinned text prompts, tokenisation echoed and hashed. |
| Shared-GPU noise | Foreign-PID guard aborts to `INFRA_FAILURE`; per-metric CV bound (§3). |
| Speculative decoding / MTP | Explicitly disabled in every configuration. |
| LoRA / DP attention | Explicitly disabled unless the run explicitly targets them (they are not part of this plan). |
| Radix / prefix cache | Same setting across matched pairs; disable for the primary FAIL / PASS verdicts, enable only for a supplementary run. |
| P_IMG image producing all-zero DeepStack tiles | Layer 1 check aborts to `INFRA_FAILURE` — the test is meaningless without nonzero DeepStack. |
| Runner authoring bugs (missed launch checks, etc.) | Each runner has a CPU-only `--dry-run` (Part 5). Post-run, `verdict.py` re-parses the launch-context JSON and refuses to score across mismatched launches (a direct reaction to the historical R6.5 stale-artifact bug). |

## 8. Deliverables per attempt

For each attempt (indexed under `results/attempt_<id>/`):

- `metadata.json` — provenance preflight output, server flags, PGIDs,
  request-set identity, sampler seed.
- `raw/` — server stderr / stdout, bench-client JSON, per-request
  `cuda graph:` log lines, per-request hidden-state / logit dumps
  (small; a few hundred KB at most). **Not committed** unless
  explicitly approved.
- `verdict.json` — machine-readable {verdict, per-hypothesis-support,
  per-layer-evidence-checks, noise-envelope-bounds, envelope-crossed
  y/n, notes}.
- `verdict.md` — human-readable narrative that explains why the
  machine verdict was chosen, cites the raw evidence, and preserves
  any known caveats.
- `summary.md` — one-paragraph plain-English summary suitable for
  quoting in `plan.md` §7 without further redaction.

Only `metadata.json`, `verdict.*`, and `summary.md` are committed by
default.

## 9. Sequencing (dry-run then GPU)

1. **CPU dry-runs** — every runner and every verdict script must pass
   `--dry-run` (no CUDA import, argument parsing, JSON schema, PGID
   scaffolding). Landed in Part 5.
2. **User authorises a specific GPU ID.**
3. **Infra check pass.** A minimal `INFRA_CHECK` run brings up the
   server on the authorised GPU with no requests, records the log,
   records the `cuda graph:` capture-session banner, tears down
   cleanly. Any anomaly aborts to `INFRA_FAILURE`.
4. **Configuration order** as randomised per §5.
5. **Verdict scoring** by `verdict.py` on the committed
   `verdict.json`.
6. **Report** back to plan.md §7 with the machine verdict verbatim
   plus a human interpretation.

Filing an upstream SGLang issue (or PR) is **not** part of this plan;
it is a separate follow-up gated on the verdict.

## 10. Amendment discipline

If any protocol change is required at any point, it must:

- Be recorded as a numbered "Amendment X" section in this file.
- Preserve the original protocol text verbatim.
- Explain the observed protocol gap.
- Say which future attempts it applies to and which prior ones stay
  under the original protocol.

No silent rewrites, no "corrections" without an amendment block.
