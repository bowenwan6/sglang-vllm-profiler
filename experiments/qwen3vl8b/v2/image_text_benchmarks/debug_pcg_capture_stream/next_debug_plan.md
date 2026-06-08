# PCG capture-stream debug — next stages (E1–E5)

> **Status: PLAN ONLY — nothing here has been executed.** Builds on
> [`debug_status.md`](debug_status.md) (current state, hypotheses H1–H4),
> [`static_audit.md`](static_audit.md) (source-level analysis), and
> [`experiment_plan.md`](experiment_plan.md) / `results/D1234_summary.md`
> (the prior tiny matrix that failed to reproduce). Approval is required
> before any E-stage runs on GPU.

## Constraints (all stages)

- GPU 7 only. Never auto-switch.
- All runs use the fixed-generator path
  (`PYTHONPATH=/data/sglang-pr/python`, `/data/sglang-pr` HEAD
  `62c505a196`, merged fix `07f326c184` in history). Provenance gate must be
  `FIX_OK` per stage; otherwise abort.
- Do **not** modify `/data/sglang-pr` source. If a code-change experiment is
  ever approved, it goes on a fresh worktree, not in place.
- No KAPI (`SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST` must
  be unset). No profiler.
- `SGLANG_USE_CUDA_IPC_TRANSPORT=1` set only for IPC-on cases; explicitly
  popped for IPC-off cases.
- Fresh server per E-stage variant; killed in `finally`; GPU returns below
  the 2000 MiB idle threshold before next variant.
- Stop immediately on first
  `AssertionError: PCG capture stream is not set` per variant. Do not retry.
- Outputs under
  `experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/results/E*/`;
  server logs under
  `logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/E*/`.
  Do **not** commit raw bench JSONL or server logs unless explicitly
  approved. Commit per-stage summary `.md` + aggregate `.json` only.
- Tiny correctness probes — **no** performance conclusions are produced by
  this debug. The headline IMG-A numbers, when we get back to them, will
  come from the formal `scripts/run_image_text_imgA_fixed.py`, not from
  here.

## Stages

### E1 — Fix text-only PCG control

**Goal.** Determine whether `/data/sglang-pr` upstream `main` (HEAD
`62c505a196`) PCG works on a clean **text-only** workload, with no Hugging
Face Hub access during the bench. Targets hypothesis H3 of `debug_status.md`.

**Why this is different from D4.** D4 used
`--dataset-name random`, which causes the bench client to load the model
tokenizer via `huggingface_hub` and fails under `HF_HUB_OFFLINE=1` with
`LocalEntryNotFoundError`. We replace that with the `autobench` dataset,
which reads prompts straight from a local JSONL with no Hub access. This is
the exact path Issue #2 used (`run_caseAC_rebaseline.py:159-163`).

**Config.**

| field | value |
|---|---|
| dataset | `--dataset-name autobench --dataset-path datasets/qwen3vl8b/caseA_short.jsonl` |
| backend | `sglang-oai-chat` |
| `--max-concurrency` | 1 |
| `--num-prompts` | 8 |
| `--warmup-requests` | 0 |
| `--seed` | 1 |
| server | SGLang upstream main from `/data/sglang-pr`, `--enforce-piecewise-cuda-graph`, `--attention-backend flashinfer` |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | unset (text-only baseline; mirrors #2's clean text-only conditions) |
| extra body | `{"temperature": 0, "top_p": 1}` |

caseA_short.jsonl is text-only (no `image_data`), so the chat backend POSTs
plain text content to `/v1/chat/completions`. Approximates the #2 Case A
shape (128→128 c=1). 8 prompts is enough to walk the bench harness past
the smoke probe without any HF Hub usage.

**Expected outcomes.**

- `OK` → upstream `main` PCG works on text-only. Combined with the Stage
  4.2 image+PCG crash, this localises the fault to the multimodal forward
  path, not to upstream PCG in general. Falsifies the "broader upstream
  regression" branch of H3.
- `PCG_CAPTURE_STREAM_ASSERT` → broader upstream PCG regression on this
  HEAD; H3 confirmed in the worst direction. **Stop**, do not proceed to
  E2/E3. Re-audit and report.
- `SERVER_NO_START` / `OTHER_FAILURE` → environment / config bug, not a
  signal about PCG. Stop and fix before any E2 work.

### E2 — Image + IPC + PCG reproduce ladder

**Goal.** Find the smallest sample size at which the Stage 4.2 assertion
reproduces, so we have a deterministic minimal repro for any upstream
issue. Targets H1 and H4.

**Common config across E2a–E2d.**

| field | value |
|---|---|
| dataset | `--dataset-name image --image-count 1 --image-resolution 720p --image-format png --image-content random` |
| backend | `sglang-oai-chat` |
| `--random-input-len` / `--random-output-len` | 128 / **128** (matches Stage 4.2 IMG-A) |
| `--random-range-ratio` | 1.0 |
| `--max-concurrency` | 1 |
| `--warmup-requests` | 30 (matches Stage 4.2 IMG-A) |
| `--seed` | 1 (matches Stage 4.2 IMG-A) |
| server | SGLang upstream main from `/data/sglang-pr`, `--enforce-piecewise-cuda-graph`, `--attention-backend flashinfer` |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | `1` |

| stage | `--num-prompts` | est. runtime |
|---|---|---|
| E2a | 32 | ~3 min |
| E2b | 64 | ~6 min |
| E2c | 100 | ~9 min |
| E2d | 400 (exact Stage 4.2 IMG-A rep replay) | ~35 min |

**Order:** E2a → E2b → E2c → E2d sequentially. **Stop the ladder as soon as
one stage hits `PCG_CAPTURE_STREAM_ASSERT`.** Smallest reproducing size
becomes the canonical minimal-repro size for the upstream issue.

**Outcomes.**

- All four `OK` → Stage 4.2 was intermittent; pivot to E4.
- First failure at E2a → trigger is shape variance from a small number of
  prompts; minimal repro is very cheap.
- First failure at E2b/E2c → trigger needs moderate sample size.
- First failure at E2d → trigger needs the full Stage 4.2 size.
- Anything other than `OK` or `PCG_CAPTURE_STREAM_ASSERT` → bench harness
  or server-startup issue; do not advance the ladder; investigate and fix
  before retry.

### E3 — Match-size image + no-IPC + PCG (only if E2 reproduces)

**Goal.** Isolate IPC's contribution by re-running E2's smallest reproducing
size with `SGLANG_USE_CUDA_IPC_TRANSPORT` unset. Targets H2.

**Config.** Identical to the first reproducing E2 stage, with
`SGLANG_USE_CUDA_IPC_TRANSPORT` removed from the per-case env. All other
parameters identical: same `num_prompts`, same warmup, same seed, same
dataset recipe, same server flags.

**Decision rule.**

| E2 (IPC on) | E3 (IPC off) | conclusion |
|---|---|---|
| ASSERT | ASSERT | IPC is **not** required to trigger. Fault is image/VLM + PCG. |
| ASSERT | OK | IPC + PCG interaction is the trigger. |
| ASSERT | other | Re-investigate the non-IPC startup path; report and stop. |

### E4 — Optional exact formal S2 replay (only if E2 ladder is inconclusive)

**Goal.** Confirm reproducibility of the original Stage 4.2 crash by running
the exact `IMG_A_S2_ipc_pcg` config end-to-end on the same upstream main
HEAD. Targets H4.

**Config.** Bit-for-bit `IMG_A_S2_ipc_pcg`:

| field | value |
|---|---|
| dataset | image, 720p, png, random, count 1 |
| `--num-prompts` | 400 |
| `--warmup-requests` | 30 |
| `--random-input-len` / `--random-output-len` | 128 / 128 |
| `--random-range-ratio` | 1.0 |
| `--seed` | 1 |
| `--max-concurrency` | 1 |
| server | SGLang upstream main + `--enforce-piecewise-cuda-graph` + `--attention-backend flashinfer` |
| `SGLANG_USE_CUDA_IPC_TRANSPORT` | `1` |

Run **once** (no `reps` loop — this is a reproducibility probe, not a
benchmark). Stop on first PCG assertion. Estimated runtime: up to ~30
minutes if the crash happens partway through; ~35 minutes if not.

**Outcomes.**

- Crashes → Stage 4.2 reproducible with the same recipe; we can include
  E4's traceback as the minimal repro in an upstream issue.
- No crash → Stage 4.2 was a single intermittent hit; revisit whether to
  treat the assertion as a release-blocker for #4 or accept it as a
  rare-event upstream issue.

### E5 — Decision (no GPU)

Synthesizes E1–E3 (and E4 if run) into a routing decision. Output goes to
`conclusion.md`. Possible routes:

| E1 (text-only PCG) | E2 (image+IPC+PCG) | E3 (image+noIPC+PCG) | E4 (exact replay) | conclusion |
|---|---|---|---|---|
| OK | ASSERT (any size) | ASSERT | n/a | VLM image + PCG specifically unsupported on this HEAD, IPC not required. File **upstream SGLang issue**: extend HIP fallback to CUDA OR loud warning on VLM + enforce-pcg. **Continue #4 without PCG.** No PR. |
| OK | ASSERT | OK | n/a | IPC + PCG interaction. File **upstream issue** scoped to IPC transport + PCG. Consider a minimal PR only if a clean fix is identified and tested locally. |
| ASSERT | n/a | n/a | n/a | Broader upstream PCG regression. File **upstream issue** about general PCG regression on this HEAD. Pause #4. **No PR.** |
| OK | all OK (incl. E2d) | n/a | OK | Stage 4.2 was intermittent. Document; either rerun formal IMG-A S2 once (low-confidence headline) or skip PCG from #4 entirely. **No PR**, no issue (intermittent reports waste reviewer time). |
| OK | all OK (incl. E2d) | n/a | ASSERT | Stage 4.2 reproduces under exact replay but not under ladder. Strong sample-distribution dependence. **File upstream issue** with full Stage 4.2 recipe; no PR. Continue #4 without PCG. |

In every clean-PR-less branch, the #4 next step is to resume IMG-A with the
non-PCG variants only (S0_ipc_repeat / V0_vllm / S0_noipc) so we can
recover bracket drift + IPC benefit + vLLM anchor while leaving PCG benefit
as undetermined for #4.

---

## Runner implementation plan

The current `scripts/run_pcg_capture_stream_debug.py` is per-stage hard-coded
to D1–D4 with `num_prompts=2 warmup=0 output_len=32`. We need two runner
changes for E1–E4:

1. Parameterise per-stage `num_prompts`, `warmup`, `output_len`,
   `dataset_kind`, and (for `dataset_kind="text_autobench"`) the local
   JSONL dataset path. The existing
   `OK / PCG_CAPTURE_STREAM_ASSERT / FORBIDDEN_TOKEN_ERROR /
   SERVER_NO_START / OTHER_FAILURE` classifier and provenance plumbing
   stay; only the `run_case()` arguments and the `DEBUG_CASES` list are
   replaced.
2. Add a new `text_autobench` case kind that passes
   `--dataset-name autobench --dataset-path
   /data/sglang-vllm-profiler/datasets/qwen3vl8b/caseA_short.jsonl` to
   `sglang.bench_serving`, instead of the broken `--dataset-name random`
   path used by D4.

Other invariants preserved as-is:

- GPU 7 only, no auto-switch.
- `PYTHONPATH=/data/sglang-pr/python` overlay on the case env; no sanitized
  monkeypatch.
- KAPI guards (strip from base env, strip from per-case env, assert unset
  at preflight); no profiler.
- Fresh server per case, killed in `finally`, GPU release confirmed.
- Stop on first assertion per case (no retry).
- Per-stage server-log excerpt captured into per-stage JSON for failure
  classification.

**Suggested filename:** `scripts/run_pcg_capture_stream_debug_v2.py`. The
current runner stays untouched as the historical D1–D4 driver; the v2
runner is the working tool for E1–E4. Implementation goes on this same
debug branch; a code review and per-stage outputs follow before any GPU
run.

**Output layout for E-stages.**

```text
experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/
  results/
    E1/
      E1_summary.md
      E1_results.json
      raw/E1_bench.jsonl                ← NOT committed
    E2/
      E2a_summary.md / E2a_results.json
      E2b_summary.md / E2b_results.json
      E2c_summary.md / E2c_results.json
      E2d_summary.md / E2d_results.json
      raw/E2<a-d>_bench.jsonl            ← NOT committed
    E3/
      E3_summary.md / E3_results.json
      raw/E3_bench.jsonl                ← NOT committed
    E4/                                 ← only if run
      E4_summary.md / E4_results.json
      raw/E4_bench.jsonl                ← NOT committed
  conclusion.md                         ← Step E5 output

logs/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/
  E1/E1_server.log                      ← NOT committed
  E2/E2<a-d>_server.log                 ← NOT committed
  E3/E3_server.log                      ← NOT committed
  E4/E4_server.log                      ← NOT committed
```

Per-stage commits use `test(v2): run image-text pcg debug stage E<n>` for
the run record, and `docs(v2): conclude image-text pcg debug` for E5's
`conclusion.md`.

---

## Out-of-scope for this plan

- IMG-B / IMG-C remain paused. They do not resume until the PCG question is
  classified (one of the routes in §E5) AND a non-PCG IMG-A bracket has
  yielded headline-grade S0_ipc / S0_ipc_repeat / V0_vllm / S0_noipc data.
- No SGLang PR work. The decision tree above explicitly rules PRs out
  unless we both have a clean minimal repro AND a tested code change in a
  fresh worktree. Neither precondition is met as of now.
- No upstream SGLang issue filing until E1–E3 (and possibly E4) have run.
- No further `/sg-workspace/sglang` (old SGLang) use. The #2 baseline is
  locked.
