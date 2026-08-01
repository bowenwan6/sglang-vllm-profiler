# Qwen3.5-4B — BCG DeepStack Investigation

> **Investigation, not a confirmed bug.** This directory tracks a
> source-level suspicion on current upstream SGLang. Nothing here
> asserts a runtime failure until runtime evidence supports it.

- **Branch:** `debug/qwen35-4b-bcg-deepstack` (based on `main` = `a803285`).
- **Tracking issue:** [profiler-repo issue #9](https://github.com/bowenwan6/sglang-vllm-profiler/issues/9)
  (sub-track of #3; no upstream SGLang issue filed until runtime
  evidence is in hand).
- **Model target:** `Qwen/Qwen3.5-4B` (HF, `sha=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
  `config.architectures=["Qwen3_5ForConditionalGeneration"]`,
  `model_type=qwen3_5`, ungated).
- **Upstream anchor (rebaselined 2026-08-01):** SGLang `main` @
  `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`. All line references in
  this subtree resolve to this SHA. This is a 21-commit refresh from
  the earlier `89f4a80c1f…` anchor; the source-audit spot-checks are
  recorded in `source_audit.md` § 1.
- **Executed local checkout:** an isolated `git clone` under
  `<scratchpad>/sglang_checkout/sglang/` at the same SHA. Runners
  source it via `PYTHONPATH`. `/data/sglang-fork` is also refreshed
  to upstream `main` on this date; the historical Qwen3-VL branch
  `fix/pcg-vlm-deepstack-warmup` is preserved untouched.
- **sglang-kernel:** upgraded system-wide 2026-08-01 to `0.4.5` to
  satisfy the frozen SGLang `assert_pkg_version` floor (was `0.4.4`).
- **Plan-level context:** `plan.md` §7.

## What this investigation is about

`Qwen/Qwen3.5-4B` is registered on current SGLang as multimodal and
on the multimodal **breakable-CUDA-graph** (BCG) allowlist —
distinct from the piecewise-CUDA-graph (PCG / `tc_piecewise`)
allowlist, which does not contain Qwen3.5. Its language model
receives per-request `input_deepstack_embeds` that are added to
`hidden_states` in layers 0–2 (the effect can propagate through
later layers via the residual stream).

The BCG replay bridge visibly stabilises `input_embeds` (registered
slot + per-request copy into that slot) but does not stabilise
`input_deepstack_embeds` — no slot is registered, and the replay
closure only forwards `input_embeds`. In addition, the single
DeepStack accommodation upstream
(`run_dummy_multimodal_deepstack_forward`, PR #30868) is a Dynamo
shape-stability warmup scoped to the `tc_piecewise` backend; it is
**not** called on the BCG capture path.

The runtime question is which of the following holds for an image
request served by a Qwen3.5-4B BCG-enabled server: (a) it enters BCG
replay and correctness is preserved by some code path this audit
missed; (b) it enters BCG replay and produces silent output
divergence in the "DeepStack-zeroed" signature; (c) it enters BCG
replay and crashes; or (d) some runtime filter routes it to eager,
so BCG is documented as not running for images (a feature gap, not
a bug). The five possible verdicts are enumerated in
`hypothesis.md` §5.

## Layout

| Path | Purpose | Status |
|---|---|---|
| [`README.md`](README.md) | Index (this file) — entry point and reader map. | landed Part 1 |
| [`source_audit.md`](source_audit.md) | Deep source-level audit of upstream SGLang `main` — files, line numbers, PR provenance; corrected 2026-07-31 to separate BCG vs PCG. | landed Part 2, corrected 2026-07-31 |
| [`provenance.md`](provenance.md) | Frozen SHAs / model / environment pins the validation must verify at run time; hard vs soft pin convention. | landed Part 2, corrected 2026-07-31 |
| [`hypothesis.md`](hypothesis.md) | Established facts vs source-level observations vs unverified runtime hypotheses vs pre-declared acceptance criteria; verdict labels revised 2026-07-31. | landed Part 2, corrected 2026-07-31 |
| [`validation_plan.md`](validation_plan.md) | Correctness/path experiment design (small matched test + diagnostic ablation), verdict shape, evidence layers, configurations, fixtures, confounder controls; revised 2026-07-31 to remove perf-benchmark controls and the incorrect `--enforce-piecewise-cuda-graph` control. | landed Part 3, corrected 2026-07-31 |
| [`fixtures/`](fixtures/) | Byte-pinned deterministic assets (image + `manifest.json`). Regeneration must be bit-identical. | landed Part 5 |
| [`scripts/`](scripts/) | CPU-only scaffolding + live runner (Step 2): fixture generator, provenance preflight, live runner, client, verdict scorer, instrumentation patch. All refuse to touch a GPU without an explicitly authorised ID. | evolving |
| [`results/`](results/) | Validation attempts (see `results/README.md`). Raw per-attempt outputs are gitignored; only summary / metadata / verdict files are committed. Step 4 INFRA_CHECK landed 2026-08-01 as `infracheck_gpu7_20260801T012122Z` (PASS). Step 5 correctness/path validation landed same day as `attempt_gpu7_20260801T013522Z` with verdict `AMBIGUOUS` (preserved as historical evidence: `language_model.__call__` instance-dict interceptor ineffective on `nn.Module`, and `<image>` placeholder mismatched the pinned Qwen VL processor's `<\|vision_start\|><\|image_pad\|><\|vision_end\|>`). Both flaws are repaired under `validation_plan.md` Amendment 2 (2026-08-01). The harness-validation follow-up `harness_gpu1_20260801T062833Z` (2026-08-01, GPU 1) confirms the repair works on GPU (pre-hook fires, placeholder warnings gone) but records `HARNESS_NOT_DIAGNOSTIC`: every publicly released `Qwen/Qwen3.5-*` checkpoint ships `vision_config.deepstack_visual_indexes = []`, so `input_deepstack_embeds` is empty (`numel = 0`) and the DeepStack `add_` branch is trivially skipped on every request. Under `validation_plan.md` Amendment 3, the source-level suspicion is not testable against this model family without a model swap. | populated as attempts land |

## Read order for a fresh reader

1. `plan.md` §7 (top-level context and roadmap position).
2. `hypothesis.md` (what is established vs suspected — sets expectations).
3. `source_audit.md` (why we suspect it — direct source citations).
4. `provenance.md` (the SHAs / env this rests on).
5. `validation_plan.md` — how we plan to prove or disprove the
   hypothesis (predeclared verdicts, evidence layers, configurations).
6. `scripts/` and `fixtures/` — CPU-only scaffolding and the live
   runner.
7. `results/` — populated as attempts land.

## CPU dry-run smoke test

Everything below runs on any CPU-only environment and touches no GPU:

```bash
# regenerate the byte-pinned image fixture (must be bit-identical)
python3 experiments/qwen35_4b/scripts/generate_fixture.py --check --strict

# provenance preflight (dry-run skips the network probes)
python3 experiments/qwen35_4b/scripts/preflight_provenance.py --dry-run

# runner dry-run (writes /tmp/qwen35_launch_ctx_*.json, no server)
bash experiments/qwen35_4b/scripts/runner.sh --dry-run

# runner must refuse without --gpu-id or QWEN35_GPU_ID set (exit 64).
# Authorised allowlist: {0, 1, 7} per validation_plan.md Amendment 2.
bash experiments/qwen35_4b/scripts/runner.sh   # expect FATAL + rc=64

# client + verdict dry runs (no network, no CUDA)
python3 experiments/qwen35_4b/scripts/client.py --launch-ctx \
    "$(ls -t /tmp/qwen35_launch_ctx_*.json | head -1)" --dry-run
python3 experiments/qwen35_4b/scripts/verdict.py --attempt-dir /tmp/some-dir --dry-run
```

## Historical context (read-only)

The Qwen3-VL-8B PCG capture-stream investigation on
`debug/v2-imgA-pcg-capture-stream-fix` (§4 in `plan.md`) touches
adjacent code but is a **different** problem and a different
reproduction target. That branch and its
`experiments/qwen3vl8b/v2/…/root_cause/` tree remain historical
evidence; nothing under §7 rewrites, reorders, or supersedes it.
The historical fork at `/data/sglang-fork` (HEAD `986c89e69`) is
not touched by this investigation.

## Reporting rules

- Preserve exact SHAs, commands, and external PR links.
- Never claim "confirmed upstream bug" until runtime reproduction is
  in hand.
- Never launch GPU work outside the authorised allowlist `{0, 1, 7}`
  (see `validation_plan.md` Amendment 1 and Amendment 2, 2026-08-01)
  and without idle-verification of the chosen GPU.
- Commit convention: `docs(qwen35): …`, `feat(qwen35): …`,
  `test(qwen35): …`, `fix(qwen35): …` per `CLAUDE.md`.
