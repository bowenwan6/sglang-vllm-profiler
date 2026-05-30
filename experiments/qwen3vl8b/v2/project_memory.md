# v2 project memory — quick-resume checkpoint

> Short, executable memory for resuming v2 work across context windows. Not a report.
> Active source of truth is root [`plan.md`](../../../plan.md); this file just speeds re-entry.
> Last refreshed: 2026-05-29 (for issue #4 image+text benchmarks).

## 1. Repo / commit rules

- [`CLAUDE.md`](../../../CLAUDE.md) is the binding commit convention. Conventional Commits:
  `type(scope): summary` (types: docs/feat/fix/chore/refactor/test/perf/ci).
- **No `Co-Authored-By` trailers; never mention Claude/Anthropic/AI** in subjects, bodies, scopes, or
  trailers. Pre-push check: `git log --max-count=5 --format='%h %s%n%B' | rg -i 'claude|anthropic|co-authored-by'`
  must return nothing.
- **Do not stage `.claude/settings.local.json`.**
- **Raw JSON / traces / logs / generated outputs are NOT committed unless the user explicitly approves.**
- Keep commits focused/separate (docs vs runner vs results).

## 2. Active source of truth

- Active v2 plan: root [`plan.md`](../../../plan.md). Old full v1 plan archived at
  [`experiments/qwen3vl8b/v1_archive_plan.md`](../v1_archive_plan.md).
- Experiment: `qwen3vl8b` · model `Qwen/Qwen3-VL-8B-Instruct` @ snapshot `0c351dd` · single H200 ·
  TP=1 · bf16 · greedy.

## 3. Validated findings (do not re-litigate)

- v1 text-only **Case A gap is real**; **Issue #2 = ✅ COMPLETE / PASS** (clean, GPU 1, 0 failures;
  results under [`caseAC_rebaseline/results/`](caseAC_rebaseline/results/)).

| Case | SGLang default | +PCG | vLLM | no-overlap (abl.) | Note |
|---|---|---|---|---|---|
| A (128→128, c=1) | 21.94 ms | 14.04 ms (−36%) | 13.12 ms | 19.07 ms | PCG → vLLM band; TPOT flat |
| C (512→128, c=16) | 204.8 ms | 230.6 ms | 215.7 ms | — | no batched benefit; CV ~14–15% |

- **PCG (`--enforce-piecewise-cuda-graph`) is a testing lever, NOT a production fix.** It still helps on
  the production default (overlap-ON), so the v1 finding is **not** an overlap-OFF artifact (no-overlap
  19.07 < default 21.94 → v1 *understated* the gap).
- **`--disable-overlap-schedule` is ablation only, never the headline baseline.**
- **Not headline:** KAPI-confounded Phase 1 four-case ratios and Phase 2 Case C W500.

## 4. Current next task — Issue #4 (image+text + CUDA IPC)

- Add **image+text** workloads; test whether the Case-A PCG finding **transfers to the image path**.
- **Separate two levers:** CUDA IPC = image-feature *transport* (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`);
  PCG = prefill *graph coverage*. Report as distinct numbers — never conflated.
- **SGLang image headline baseline MUST set `SGLANG_USE_CUDA_IPC_TRANSPORT=1`** (S0_noipc is the ablation).
- **Clean only** (no KAPI, no profiler); servers serialized.
- **Image+text conclusions reported separately from text-only (#2) findings.**

## 5. Issue #4 protocol summary

Protocol: [`image_text_benchmarks/protocol.md`](image_text_benchmarks/protocol.md) (decision-complete,
gated). Scaffold READMEs exist; **no runner, no runs, no servers yet.**

- **Workloads:** IMG-A (1×720p + short text, c=1; primary Case-A analog), IMG-B (medium text, c=1),
  IMG-C (short text, c=16; Case-C analog), IMG-D (opt, 2 images).
- **Dataset:** synthetic `--dataset-name image` (inline base64, seed-reproducible, no downloads, no large
  checked-in assets). Headline: `--image-resolution 720p --image-format png --image-content random
  --image-count 1 --seed 1`. Identity = harness commit + image params + seed.
- **Variants:** `S0_ipc` (IPC on), `S2_ipc_pcg` (IPC+PCG), `S0_noipc` (IPC ablation), `V0_vllm`. Both
  frameworks use `--backend sglang-oai-chat` (image dataset rejects `--backend vllm`).
- **Run design:** IMG-A/B bracket `S0_ipc → S2_ipc_pcg → S0_ipc_repeat → V0_vllm → S0_noipc` (drift ≤5%);
  IMG-C interleaved `S0_a → S2_a → S0_b → S2_b → S0_c → V0`. Reps: c=1 → 5, c=16 → 3.
- **Open items (UNVERIFIED — gate Phase 4.0):** (1) vLLM image anchor via `sglang-oai-chat`;
  (2) text-length pinning via `--random-range-ratio`; (3) `SGLANG_USE_CUDA_IPC_TRANSPORT=1` actually engages.

## 6. Immediate execution plan (Phase 4.0 smoke first)

- **Phase 4.0 = smoke, NOT a full benchmark.** Implement
  `experiments/qwen3vl8b/v2/image_text_benchmarks/run_image_text_smoke.py`.
- Before running: `python3 -m py_compile` + diff review.
- Smoke covers 3 paths: SGLang+IPC, SGLang no-IPC, vLLM anchor. Tiny `--num-prompts 2`, c=1.
  **No perf conclusions** — purpose is only to resolve the 3 open items.
- **If smoke fails → stop and report** (do NOT run IMG-A).
- **If smoke succeeds → commit smoke summary, then implement the IMG-A formal runner.** IMG-A formal runs
  only after smoke passes (Phase 4.1).

## 7. Artifact rules for #4

- Protocol: `experiments/qwen3vl8b/v2/image_text_benchmarks/protocol.md`.
- Future smoke/results under `experiments/qwen3vl8b/v2/image_text_benchmarks/` (summaries + aggregate JSON);
  raw in `results/raw/`; server logs under `logs/qwen3vl8b/v2/image_text_benchmarks/`.
- Raw + logs NOT committed unless explicitly approved.
- **Never overwrite #2 (`caseAC_rebaseline/`) or v1 Phase 0–5 artifacts.**

## 8. Open cautions

1. **vLLM image anchor is unverified** — `sglang-oai-chat` against vLLM's chat endpoint with data-URI
   images must smoke-pass before any comparison is trusted.
2. **Synthetic images ≠ real-user distribution** — fine for relative IPC/PCG contrasts, not an absolute
   production claim.
3. **TTFT is end-to-end (client-side)** — no preprocessing / vision / prefill split (profiler forbidden).
4. **CUDA IPC and PCG must be analyzed separately** — they are independent levers.
5. **If a residual gap remains after IPC + PCG, open a new profiling issue** — do not expand #4
   indefinitely.
