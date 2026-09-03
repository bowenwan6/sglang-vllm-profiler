# v2 project memory — quick-resume checkpoint

> Short, executable memory for resuming v2 work across context windows. Not a report.
> Active source of truth is root [`plan.md`](../../../plan.md); this file just speeds re-entry.
> Last refreshed: **2026-08-29** (post-BCG-detour; #4 resumption).
>
> **Since the 2026-05-29 refresh:** the #4 smoke passed and IMG-A *partially*
> ran; a correctness detour (#9 → Qwen3-VL BCG DeepStack bug) was fixed and
> upstreamed as [sgl-project/sglang#33726](https://github.com/sgl-project/sglang/pull/33726);
> the Qwen3.5 DeepStack question closed as `NOT_APPLICABLE_QWEN35`. §§5–6 below
> are updated; §§1–3, 7–8 still hold.

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
gated). **Runner exists; the Phase-4.0 smoke passed and IMG-A ran partially** — see §6.

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

## 6. Where #4 actually stands (updated 2026-08-29)

Phase 4.0 smoke **passed**; the benchmark-generator `<|video_pad|>` bug was fixed
and merged upstream (`07f326c184`, SGLang #26864). Stage 4.2 IMG-A is **PARTIAL**:

| Variant | Result |
|---|---|
| `IMG_A_S0_ipc` | ✅ 5/5 reps, 2 000 requests, 0 failures, TTFT p50 **64.8 ms** |
| `IMG_A_S2_ipc_pcg` | ❌ rep 1 server crash — `AssertionError: PCG capture stream is not set` |
| `IMG_A_S0_ipc_repeat` / `V0_vllm` / `S0_noipc` | ⏸ skipped per protocol §9 |

So **no comparison exists yet**: there is one SGLang arm, no vLLM anchor, and no
IPC ablation. The single number above is not a finding.

### Resume order (next GPU work)

1. Freeze a **new environment manifest** — the old stack is stale, so re-pin
   SGLang SHA, vLLM version, model revision, harness revision, CUDA/driver/torch/
   kernel, attention backend, and IPC env before anything runs.
2. Re-run Phase-0 correctness plus a **tiny current-upstream image+PCG smoke**.
   If the capture-stream assertion no longer reproduces, restore the PCG arm; if
   it does, record the exact current-upstream failure and exclude PCG
   transparently.
3. Complete the bracket `S0_ipc_repeat → V0_vllm → S0_noipc`. Keep the existing
   `S0_ipc` result **only** if provenance and stack match — otherwise rerun the
   whole bracket.
4. Gate to close #4: headline-quality IMG-A with a vLLM anchor and an IPC
   ablation; PCG either has clean data or a documented current-upstream
   exclusion. **Do not start IMG-B/C before that.**

Recovery plan: [`fixed_generator_plan.md`](image_text_benchmarks/fixed_generator_plan.md).

⚠️ **BCG ≠ PCG.** The Qwen3-VL BCG DeepStack fix (upstream PR #33726) is a
different graph backend. It does not restore the PCG arm and must not be
silently substituted for it here.

⚠️ **Flag restructure (2026-09-03) — re-read before writing any runner.**
`--enforce-piecewise-cuda-graph` is now a deprecated alias for
`--cuda-graph-backend-prefill=tc_piecewise`; prefill backends are
`full | breakable | tc_piecewise | disabled`; **breakable is the CUDA default**;
and PR #33726 puts Qwen3-VL on the breakable allowlist, so its default prefill
backend flips from *disabled* to *BCG* on merge. #4 therefore needs four SGLang
arms, not two, and each must log its **resolved** backend — an unsupported
request is silently downgraded to `disabled`, so flag acceptance proves nothing.
Details and line references: [`plan.md` §3.5](../../../plan.md).

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
