# Validation Plan — Qwen3.5-4B BCG DeepStack

> **Purpose.** Design a small correctness/path experiment that, run on
> the frozen upstream SGLang `main` checkout at
> `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` with `Qwen/Qwen3.5-4B`,
> produces objective per-request evidence sufficient to pick exactly
> one of the five outcomes in `hypothesis.md` §5.
>
> **Scope.** This is a correctness/path experiment, not a performance
> benchmark. Latency percentiles, CV gates, and 30-rep sweeps are
> **not** in scope. The plan uses a small matched initial test and
> a small confirmation only if a signal appears.

## 1. Predeclared verdict shape

Verdict labels are defined in `hypothesis.md` §5 and are the source
of truth. The runner and `verdict.py` must emit exactly one of:

| Verdict | Trigger |
|---|---|
| **`PASS_BCG_CORRECT`** | Image request demonstrably replays BCG (instrumentation confirms `_execute_body_capture` + `replay_layer_forward` served it) AND greedy tokens / logits match `eager_normal` within the eager-vs-eager envelope AND `input_deepstack_embeds` was non-trivially nonzero at request time. |
| **`FEATURE_GAP_EAGER_FALLBACK`** | Image request demonstrably ran on the eager runner despite BCG being enabled. Correctness is trivially preserved; the BCG performance premise is not demonstrated. Documented as a feature gap; **not** "PASS" in the strong sense. |
| **`FAIL_BCG_DEEPSTACK`** | Image request demonstrably replays BCG AND one of: (a) matched greedy-token or hidden-state divergence beyond the eager-vs-eager envelope with a zero-DeepStack signature (i.e. `bcg_normal` ≈ `eager_zero_deepstack`), or (b) the BCG replay raises an assertion / illegal memory access at inference time. |
| **`AMBIGUOUS`** | Divergence exists but attribution fails (envelope swallows the signal, `bcg_normal` matches neither `eager_normal` nor `eager_zero_deepstack`, fixture produced trivially-zero DeepStack, instrumentation self-inconsistent). |
| **`INFRA_FAILURE`** | Provenance preflight fails on a hard pin; GPU 0 cannot be safely acquired; foreign compute PID appears on GPU 0 during the run; server fails to bring up; runner aborts on its own ownership guard. Recorded, does not count for or against any hypothesis. |

The verdict is **declared before observing any result**. Post-hoc
softening requires an explicit written amendment recorded in
`hypothesis.md` §5 and re-committed.

## 2. Prerequisites the runner records for every attempt

Every attempt records, before executing any request:

1. **Frozen local SGLang checkout SHA** (hard pin;
   `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`). Preflight aborts if
   drifted.
2. **Upstream `main` HEAD SHA at query time** — informational only,
   WARN on drift.
3. **Imported `sglang.__file__`** — must resolve inside the frozen
   checkout after the runner's `PYTHONPATH` override.
4. **HF model revision** actually loaded (from the server's startup
   log or an equivalent snapshot probe). Must equal `provenance.md`
   §2.
5. **Framework versions** (torch, flashinfer, sgl_kernel) and
   `libcuda.so` path via `ldconfig`. WARN on drift; hard failure only
   under `--strict-env`.
6. **Server flags emitted verbatim**, including whether BCG is on
   and whether `--enforce-piecewise-cuda-graph` was passed (which it
   must not be for the BCG legs — see §4).
7. **`nvidia-smi --query-gpu=... --id=0`** snapshot at start,
   periodically during execution, and at end. Queries are filtered
   to GPU 0 by UUID; never wildcarded.
8. **`nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory
   --format=csv`** filtered to GPU 0's UUID at start / periodically /
   end. Must show no foreign PID on GPU 0. Foreign PID mid-run →
   attempt classified `INFRA_FAILURE`; the runner does not signal
   the foreign PID.
9. **Launch identity** (PID, PGID, SID, executable path, launch UUID,
   `prelaunch_utc`) for every process the runner launches. Cleanup
   signals only recorded PGIDs whose current members were all
   recorded by this runner. Never `pkill` / `killall` / `fuser -k`
   / `nvidia-smi --gpu-reset`.

## 3. Instrumentation the runner needs

Owned by this branch only. Applied as a lightweight monkey-patch (or
in-tree patch to the frozen checkout, reverted at teardown) that
records per-request:

- Whether the request entered `_execute_body_capture` (BCG replay
  execute path), or the eager runner.
- Whether `replay_layer_forward` was invoked.
- Whether the incoming `layer_kwargs` contained `input_deepstack_embeds`
  and, if so:
  - shape, dtype, numel;
  - `finite = torch.isfinite(x).all()`;
  - `nonzero_frac = (x != 0).float().mean()`;
  - a **compact checksum** (e.g. `float64(x.abs().sum())` +
    `float64((x*x).sum())` + a 16-byte SHA-256 prefix of the raw
    bytes converted to CPU) — enough to detect drift, small enough
    to persist.
  - `.data_ptr()` as diagnostic **only** — see the pointer wording
    in `source_audit.md` §4.1; the runner must not draw
    correctness conclusions from pointer equality alone.
- Whether the served request took BCG replay or eager (the same
  signal as `can_run_graph`'s return value at the model runner
  boundary).
- Startup vs inference-time recompile events (from
  `TORCH_LOGS=recompiles_verbose` on the confirmation run only;
  omitted for the primary run to avoid perturbing behaviour).
- Greedy token IDs and per-token logprobs where supported. If
  logprobs are not supported under BCG replay for this
  configuration, record `logprobs_available=false` and rely on
  hidden-state RMS instead.
- Per-configuration hidden-state RMS at layers 0, 1, 2, 3, and the
  final layer (via a small `LogitsProcessorOutput.hidden_states` hook
  or the profiler-side patch), compared across `eager_normal`,
  `eager_zero_deepstack`, and `bcg_normal` on the *same* request.

**No full DeepStack tensor is dumped.** Compact checksum plus per-layer
RMS is sufficient for attribution; anything larger risks disk-space
regressions and complicates reproducibility.

## 4. Configurations under test

Every configuration serves `Qwen/Qwen3.5-4B` at the pinned revision
under the frozen SGLang checkout. **No configuration uses
`--enforce-piecewise-cuda-graph`** — Qwen3.5 is not on the PCG
allowlist (source_audit.md §2), so that flag is not a valid BCG
control. It is removed from the plan entirely.

| # | Label | Server flags (key) | Purpose |
|---|---|---|---|
| C0 | `eager_ref` | prefill CUDA graph disabled (e.g. by pinning a graph-disabling flag or by running an eager-only variant); radix cache off | Correctness reference. |
| C1 | `bcg_default` | default breakable prefill backend (BCG on); no PCG flag; radix cache off | The path under test. |
| C_ABL | `eager_zero_deepstack` | Same as C0 with the branch-local instrumentation that zeros `input_deepstack_embeds` immediately before the LM forward | Diagnostic ablation for the zero-DeepStack signature. |
| C_TEXT | `text_only_control` | Any of C0 or C1 with a text-only prompt | Confirms the framework serves text-only correctly on the same server; isolates DeepStack-specific regressions from generic BCG regressions. |

Each configuration runs the same **small** request set (§5) with
matched cache state and greedy sampling.

## 5. Fixtures and request schedule

### 5.1 Fixtures

- **Image prompt (P_IMG).** A short instruction ("Describe the
  colours in this image.") applied to the byte-pinned PNG
  `fixtures/image_bands.png`
  (SHA-256 `8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`).
- **Text prompt (P_TXT).** A single short instruction with no image;
  reused from `datasets/qwen3vl8b/caseA_short.jsonl` (existing
  SHA-256-pinned dataset).

`fixtures/manifest.json` pins byte counts, dimensions, and MIME type.

### 5.2 Request schedule (simplified — correctness/path)

- **Batch of 1**, serial (no concurrency), greedy (`temperature=0`,
  `top_p=1`, fixed seed).
- **Initial matched test** — the minimum set required to distinguish
  the five verdicts:
  1. `C0` (`eager_normal`) × 1 request: `P_IMG`.
  2. `C0` (`eager_normal`) × 1 request: `P_TXT` (text-only control).
  3. `C_ABL` (`eager_zero_deepstack`) × 1 request: `P_IMG`.
  4. `C1` (`bcg_default`) × 1 request: `P_TXT` (text-only control,
     to confirm BCG serves text-only without regression).
  5. `C1` (`bcg_default`) × 1 request: `P_IMG` — the primary scored
     leg.
- **Eager-noise envelope** — one additional `C0 × P_IMG` repeat, on
  the same server with matched cache. The greedy tokens must be
  identical; the hidden-state RMS diff between the two `eager_normal`
  runs is the noise floor.
- **Confirmation-only repeats** — if the initial matched test shows
  an apparent signal (any divergence between `bcg_normal` and
  `eager_normal` above the noise floor), the runner may repeat the
  primary `C1 × P_IMG` and the ablation `C_ABL × P_IMG` up to 3
  times each to confirm stability. Absent a signal, no confirmation
  is needed.
- **Server reuse policy** — the initial matched test uses one server
  per configuration (C0, C1). A fresh server is not required for
  every individual request; correctness is verified by matched
  cache state (radix disabled unless the run explicitly targets a
  cache confound) and a warmup pass on each fresh server.
- **Warmup** — the plan retains warmup per fresh server (the
  standard SGLang warmup that runs on server start; no synthetic
  30-request warmup imposed by the runner).
- **Deterministic sampler seed** — fixed integer for every request.

**What was explicitly removed from the earlier plan.** Latency-CV
gates, per-configuration 30-request measured runs (with 5 discards),
fresh-server-per-request, cross-attempt configuration randomisation,
Layer 4 statistical envelope beyond a single eager-vs-eager pair.
These were performance-benchmark controls that do not help a
correctness/path decision and inflate the run cost without adding
attribution power.

## 6. What each verdict requires as evidence

- **`PASS_BCG_CORRECT`** ← Instrumentation shows every scored `C1 ×
  P_IMG` request took the BCG execute path (`_execute_body_capture`
  invoked, `replay_layer_forward` invoked) AND
  `input_deepstack_embeds.nonzero_frac > 0` (checksum recorded) AND
  greedy tokens match `C0 × P_IMG` exactly AND `bcg_normal` per-layer
  RMS vs `eager_normal` is within the eager-vs-eager envelope.
- **`FEATURE_GAP_EAGER_FALLBACK`** ← Instrumentation shows every
  scored `C1 × P_IMG` request went through the eager runner (no
  `_execute_body_capture`, no `replay_layer_forward`). Correctness
  holds by construction; the verdict is not stronger than "BCG did
  not run for this request set".
- **`FAIL_BCG_DEEPSTACK`** ← Instrumentation shows every scored `C1
  × P_IMG` request took the BCG execute path AND (a) greedy tokens
  differ AND `bcg_normal` per-layer RMS pattern is closer to
  `eager_zero_deepstack` than to `eager_normal` (documented ratio,
  e.g. within 1.5× of the ablation RMS and > 3× the eager-vs-eager
  envelope), OR (b) BCG replay raised an assertion or produced an
  illegal memory access mid-run.
- **`AMBIGUOUS`** ← Any of: instrumentation self-inconsistent,
  DeepStack was trivially zero, ablation arm was corrupted, cache
  state was not matched.
- **`INFRA_FAILURE`** ← Provenance preflight failed on a hard pin,
  foreign PID seen on GPU 0 mid-run, GPU 0 could not be safely
  acquired, or server crash during startup before any request.

## 7. Confounders and how the plan controls them

| Confounder | Control |
|---|---|
| bf16 nondeterminism | Single eager-vs-eager pair gives the noise envelope; never claim `FAIL_BCG_DEEPSTACK` below the envelope. |
| Warm-vs-cold KV cache | Matched fresh server per configuration; radix cache off unless explicitly targeted. |
| Silent BCG-off | The branch-local instrumentation is authoritative for path attribution; the runner rejects any scored request it could not classify. |
| Framework version drift | Provenance preflight (§2). |
| Prompt/image tokenisation drift | Byte-pinned image fixture, SHA-pinned text prompts. |
| Shared-GPU noise | Foreign-PID guard aborts to `INFRA_FAILURE`; the acquisition protocol (`plan.md` §7 Step 3) requires 10 continuous minutes of GPU 0 idle before launch and continuous monitoring during execution. |
| Speculative decoding / MTP | Explicitly disabled in every configuration. |
| LoRA / DP attention | Explicitly disabled unless a run explicitly targets them (they are not part of this plan). |
| Radix / prefix cache | Disabled for the primary verdict legs; enabled only for a supplementary run if the plan later needs it. |
| P_IMG image producing all-zero DeepStack tiles | Instrumentation Layer-1 check aborts to `AMBIGUOUS` — the test is meaningless without nonzero DeepStack. |
| Runner authoring bugs (missed launch checks, stale artifacts) | Every runner has a CPU-only `--dry-run`; `verdict.py` re-parses the launch-context JSON and refuses to score across mismatched launch IDs (reaction to the historical Qwen3-VL R6.5 stale-artifact bug). |

## 8. Deliverables per attempt

For each attempt (indexed under `results/attempt_<id>/`):

- `metadata.json` — provenance preflight output, server flags, PGIDs,
  launch UUID, request-set identity, sampler seed.
- `raw/` — server stderr / stdout, bench-client JSON, per-request
  instrumentation records, per-layer hidden-state RMS values, greedy
  token dumps. **Not committed** unless explicitly approved. Path is
  under a gitignored `results/` subtree.
- `verdict.json` — machine-readable {verdict, per-hypothesis-support,
  per-layer-evidence-checks, envelope, envelope-crossed y/n,
  execution-path counts per configuration, notes}.
- `verdict.md` — human-readable narrative that explains why the
  machine verdict was chosen, cites the raw evidence, and preserves
  any known caveats.
- `summary.md` — one-paragraph plain-English summary suitable for
  quoting in `plan.md` §7 without further redaction.

Only `metadata.json`, `verdict.*`, and `summary.md` are committed by
default.

## 9. Sequencing (dry-run then GPU)

1. **CPU dry-runs** — every runner and every verdict script must
   pass `--dry-run` (no CUDA import, argument parsing, JSON schema,
   PGID scaffolding).
2. **GPU 0 authorisation.** GPU 0 is the only authorised device.
3. **GPU 0 acquisition protocol** — see `plan.md` §7. Read-only
   query first; require 10 continuous minutes of idle (zero compute
   processes, memory ≤ 500 MiB, utilisation ≤ 5 %); recheck
   immediately before the first server launch to leave no idle gap;
   never signal any foreign PID; never switch GPUs.
4. **INFRA_CHECK.** Bring up the smallest real Qwen3.5-4B server
   configuration needed to verify model revision, dependencies, CUDA
   / libcuda, multimodal readiness, BCG capture banner, clean
   request-free teardown, GPU memory release. Any anomaly →
   `INFRA_FAILURE`, stop.
5. **Correctness/path validation.** Run the initial matched test in
   §5.2. If a signal appears, run the small confirmation. Score
   via `verdict.py`.
6. **Report** back to `plan.md` §7 with the machine verdict verbatim
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

## Amendment 1 (2026-08-01) — GPU 7 authorisation and idle-gate waiver

**Applies to:** every §7 attempt run on or after 2026-08-01. Previous
attempts (none exist at time of writing) would remain under the
original protocol.

**Original text preserved verbatim** (from §9 sequencing and §7
confounder-controls table):

> 2. **GPU 0 authorisation.** GPU 0 is the only authorised device.
> 3. **GPU 0 acquisition protocol** — see `plan.md` §7. Read-only
>    query first; require 10 continuous minutes of idle (zero compute
>    processes, memory ≤ 500 MiB, utilisation ≤ 5 %); recheck
>    immediately before the first server launch to leave no idle gap;
>    never signal any foreign PID; never switch GPUs.
>
> | Shared-GPU noise | Foreign-PID guard aborts to `INFRA_FAILURE`; the acquisition protocol (`plan.md` §7 Step 3) requires 10 continuous minutes of GPU 0 idle before launch and continuous monitoring during execution. |

**Observed protocol gap.** At Step-3 acquisition on 2026-08-01,
GPU 0 was heavily occupied by foreign compute (~69 GiB / 25 %
utilisation / 2 foreign PIDs) with no visible path to the qualifying
state in a reasonable window. The operator explicitly authorised
GPU 7 mid-turn ("you can use gpu7") and then (having observed GPU 7
was already fully idle: 0 MiB / 0 % / 0 compute apps) explicitly
waived the 10-continuous-minutes idle requirement for this attempt
("you dont need to wait, if it is idle, you can run directly").

**Amended rules for attempts on or after 2026-08-01:**

1. The authorised-GPU allowlist is now `{0, 7}`. GPU 7 is used when
   GPU 0 is not qualifying and the operator has confirmed the switch.
   No other GPU may be touched.
2. The 10-continuous-minute idle requirement is waived **only** when
   the operator confirms an immediate launch AND the target GPU is
   currently in the qualifying state (0 compute processes, ≤ 500 MiB,
   ≤ 5 % utilisation) at the moment of launch.
3. The foreign-PID guard (Step 3 point 7 / §2 point 8) is unchanged:
   any foreign PID that appears on the authorised GPU **during**
   execution still classifies the attempt as `INFRA_FAILURE`; the
   runner stops only its own process group, records evidence,
   commits, and stops.
4. Runner-side changes: `scripts/runner.sh` accepts `--gpu-id 0` or
   `--gpu-id 7` (exit 64 for any other id) and uses `$GPU_ID`
   consistently for `CUDA_VISIBLE_DEVICES`, the `--id` argument to
   `nvidia-smi` and the foreign-PID / GPU-memory checks.

**Not changed by this amendment:**

- Hard SGLang checkout / model-revision / fixture pins.
- Instrumentation semantics, verdict labels, or evidence-layer
  requirements.
- The prohibition on signalling foreign PIDs, resetting GPUs, or
  modifying `/data/sglang-fork`.
- The rule that a `FEATURE_GAP_EAGER_FALLBACK` is never labelled
  "PASS" in the strong sense.

## Amendment 2 (2026-08-01) — repaired instrumentation, corrected image prompt, 2×2 design, widened GPU allowlist

**Applies to:** every §7 attempt run on or after 2026-08-01 that uses
the repaired instrumentation. Attempt `attempt_gpu7_20260801T013522Z`
stays scored under the original protocol and keeps its `AMBIGUOUS`
verdict; it is preserved as historical evidence of the two flaws this
amendment addresses.

**Observed protocol gaps (from `attempt_gpu7_20260801T013522Z`):**

1. `scripts/instrumentation.py`'s `_patch_general_mm_embed_routine`
   assigned `language_model.__dict__["__call__"] = _lm_call_intercept`.
   For `nn.Module` subclasses, Python resolves `__call__` on the
   class, not the instance `__dict__`, so the interceptor never fired.
   `lm_forward_input_deepstack` events were never recorded and
   `QWEN35_ZERO_DEEPSTACK=1` was a no-op — collapsing
   `eager_zero_deepstack` into a second `eager_normal` repeat.
2. `scripts/client.py` hard-coded `<image>` as the multimodal
   placeholder. The pinned Qwen VL processor
   (`python/sglang/srt/multimodal/processors/qwen_vl.py:338`) expects
   `<|vision_start|><|image_pad|><|vision_end|>`. Every image
   request emitted the SGLang warning "More image data items provided
   than corresponding tokens found in the prompt" — the multimodal
   path was not exercised cleanly.
3. `scripts/verdict.py` returned `PASS_BCG_CORRECT` on greedy-text
   equality alone. The predeclared verdict rules require nonzero
   DeepStack in the normal arm, a verified zero replacement in the
   ablation arm, and BCG replay on scored image requests before any
   `PASS`.
4. Three-arm design (`eager_normal`, `eager_zero_deepstack`,
   `bcg_normal`) cannot distinguish "BCG preserves DeepStack" from
   "BCG silently drops DeepStack but the fixture is text-invariant"
   without a `bcg_zero_deepstack` comparator.

**Amended rules for attempts on or after 2026-08-01:**

1. **Repaired DeepStack interceptor.** Instrumentation installs the
   DeepStack observer via
   `language_model.register_forward_pre_hook(hook, with_kwargs=True)`
   for the duration of one `general_mm_embed_routine` call and
   removes it in `finally`. The hook records shape / dtype / numel /
   finite / nonzero_frac / abs_sum / sq_sum / SHA-256-16 / data_ptr
   before modification, and (in zero mode) records the same summary
   after replacement to prove the substitution really is zero.
   Repeated calls do not accumulate hooks. Proved by
   `scripts/test_instrumentation.py` on CPU.
2. **Corrected multimodal request construction.** `scripts/client.py`
   emits `<|vision_start|><|image_pad|><|vision_end|>` verbatim,
   records the rendered prompt, the placeholder count, and the
   supplied image count on every request. `scripts/verdict.py`
   hard-fails on any placeholder-vs-image mismatch or on the
   presence of the SGLang "More image data items…" warning.
3. **Predeclared 2×2 design.** Every scored §7 attempt runs four
   arms serially on the same qualifying GPU with matched revision,
   fixture, prompts, cache, sampling, and request order:
   - `eager_normal`         — BCG off, DeepStack computed normally.
   - `eager_zero_deepstack` — BCG off, DeepStack zeroed by hook.
   - `bcg_normal`           — BCG on,  DeepStack computed normally.
   - `bcg_zero_deepstack`   — BCG on,  DeepStack zeroed by hook.
4. **Predeclared verdict rules (2×2).** Given valid telemetry (image
   / placeholder aligned, DeepStack observed nonzero in normal arms,
   zero replacement verified in ablation arms, BCG replay confirmed
   in both BCG arms, ablation-sensitivity confirmed by
   `eager_normal ≠ eager_zero_deepstack` beyond the eager-repeat
   noise envelope):
   - `bcg_normal == eager_normal` AND `bcg_normal != bcg_zero_deepstack`
     → `PASS_BCG_CORRECT`.
   - `bcg_normal != eager_normal` AND `bcg_normal == bcg_zero_deepstack`
     AND `bcg_zero_deepstack == eager_zero_deepstack` →
     `FAIL_BCG_DEEPSTACK`.
   - BCG replay never fires on the scored image request →
     `FEATURE_GAP_EAGER_FALLBACK`.
   - Any BCG replay error / illegal memory access → `FAIL_BCG_DEEPSTACK`.
   - Missing telemetry, placeholder mismatch, ablation not
     diagnostic, or attribution unclear → `AMBIGUOUS`.
   - Environment / GPU failure → `INFRA_FAILURE`.
5. **Widened GPU allowlist to `{0, 1, 7}`.** The operator extended the
   standing authorisation on 2026-08-01 to GPU 1 (which was fully
   idle at 0 MiB / 0 %). `scripts/runner.sh` accepts `--gpu-id 0`,
   `--gpu-id 1`, or `--gpu-id 7`; any other id is exit-64. The
   Amendment 1 waiver of the 10-continuous-minute idle requirement
   (immediate launch when the target GPU is already qualifying)
   continues to apply. Foreign-PID guard is unchanged.
6. **Runner-side changes.** `scripts/runner.sh` now accepts
   `--config bcg_zero_deepstack` (BCG on, `QWEN35_ZERO_DEEPSTACK=1`).
   `scripts/verdict.py` requires all four arms to be present; a
   missing arm classifies `AMBIGUOUS`.
7. **Ignore-rule tightening.** `results/.gitignore` covers both
   `attempt_*/raw/` and `infracheck_*/raw/`. Currently-untracked raw
   evidence from attempt `attempt_gpu7_20260801T013522Z` and the two
   INFRA_CHECKs remains on-disk as historical evidence; only the
   ignore rule changes.

**Not changed by this amendment:**

- Hard SGLang checkout / model-revision / fixture pins.
- Verdict-label set (still exactly the five in §1).
- Ban on signalling foreign PIDs, resetting GPUs, or modifying
  `/data/sglang-fork`.
- The rule that a `FEATURE_GAP_EAGER_FALLBACK` is never labelled
  "PASS" in the strong sense.
- Preservation of prior attempt directories and their recorded
  verdicts.
