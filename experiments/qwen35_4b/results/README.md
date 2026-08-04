# `results/` — attempts

This directory holds Qwen3.5-4B BCG DeepStack validation attempts.
Raw per-attempt evidence lives under each attempt's `raw/` subdir and
is gitignored; only summary / metadata / verdict files are committed
by default.

## Attempt index

| Attempt | GPU | Kind | Outcome | Notes |
|---|---|---|---|---|
| `infracheck_gpu7_20260801T004841Z` | 7 | INFRA_CHECK | BLOCKED (not committed) | Prior authorised attempt; blocked in `assert_pkg_version` because `sglang-kernel==0.4.4 < 0.4.5`. Raw evidence kept in place as historical record; no `metadata.json` / `verdict` files. |
| `infracheck_gpu7_20260801T012122Z` | 7 | INFRA_CHECK | PASS | See `metadata.json` / `verdict.md` / `summary.md`. Kernel is now `0.4.5`; server came up in 129 s with breakable prefill CUDA graph (58 shape captures), warmup exercised BCG (`cuda graph: True`), teardown clean, foreign PIDs unaffected. Records the spawn caveat carried into Step 5. |
| `attempt_gpu7_20260801T013522Z` | 7 | Correctness/path validation (Step 5) | AMBIGUOUS | See `metadata.json` / `verdict.md` / `summary.md`. Ran the three predeclared configs (`eager_normal`, `eager_zero_deepstack`, `bcg_normal`) with the new `scripts/bootstrap/sitecustomize.py` propagating instrumentation into SGLang spawn workers. `bcg_normal` served both scored image prefills via BCG (`bcg_execute_body_enter` × 5 with `contains_mm_inputs=true`, `cuda graph: True`, no `bcg_execute_body_error`), greedy text bit-identical to `eager_normal`. `PASS_BCG_CORRECT` blocked by the branch instrumentation's `language_model.__call__` interceptor being ineffective on `nn.Module` (writes to instance `__dict__`, but Python resolves `__call__` on the class) — DeepStack nonzero fraction unverified, `QWEN35_ZERO_DEEPSTACK=1` a no-op, so `eager_zero_deepstack` degenerates to `eager_normal`. Reinforced by an image-placeholder mismatch (`<image>` vs the expected `<\|vision_start\|><\|image_pad\|><\|vision_end\|>`) that warned `More image data items provided than corresponding tokens found in the prompt`. No upstream bug demonstrated; no fix implemented per plan. **Preserved verbatim as historical evidence; both flaws are repaired under `validation_plan.md` Amendment 2 (2026-08-01) and subsequent attempts run the four-arm 2×2 design against the repaired harness.** |
| `attempt_gpu1_20260801T115524Z` | 1 | Attempt 03: Qwen3-VL-8B 2×2 under monkey-patched BCG allowlist | **FAIL_BCG_DEEPSTACK** | See `metadata.json` / `verdict.md` / `summary.md`. Full 4-arm 2×2 ran on GPU 1 under `Qwen/Qwen3-VL-8B-Instruct @ 0c351dd0` with the profiler-owned `scripts/bcg_allowlist_patch.py` monkey-patch enabled (`QWEN35_PATCH_BCG_ALLOWLIST=1` + `--patch-bcg-allowlist`). Pre-state allowlist `[Qwen3_5, Qwen3_5Moe]`; post-state `[…, Qwen3VLForConditionalGeneration, Qwen3VLMoeForConditionalGeneration]`. Frozen SGLang source unchanged (`git diff --stat` empty). All arms served the 893-token scored image prefill; both BCG arms served it with `cuda graph: True` and zero `bcg_execute_body_error`. DeepStack tensor observed at `Qwen3LLMModel` entry with `shape=[896, 12288]` (= [N, hidden_size * 3]) and `nonzero_frac ≈ 0.98` in normal arms; zero-substitution verified in ablation arms. `eager_normal` = "The image displays three vertical stripes of red, green, and blue." (15 tokens); `eager_zero_deepstack` diverges at token 7 (20 tokens). **`bcg_normal` bit-identical to `bcg_zero_deepstack` (20/20 tokens equal, mean logprob diff 0.0)** and both track `eager_zero_deepstack`. Direct live-fire evidence that SGLang's `replay_layer_forward` bridge silently drops the DeepStack contribution under BCG replay — the source-level suspicion in `latent_bug_analysis.md` § 2 is confirmed. Caveat: obtained under the runtime monkey-patch; no shipped upstream configuration reaches this code path. GPU 1 stale-VRAM bookkeeping from prior arms' dead workers noted in-run; `foreign_pid_check` refined to skip PIDs missing from `/proc`. `/data/sglang-fork` HEAD unchanged. |
| `harness_gpu1_20260801T062833Z` | 1 | Harness validation (Step 2 of the harness-repair pass) | HARNESS_NOT_DIAGNOSTIC | See `metadata.json` / `verdict.md` / `summary.md`. Ran `eager_normal` + `eager_zero_deepstack` on GPU 1 under the repaired harness (post `fix(qwen35): repair DeepStack instrumentation and image input`). Repaired `nn.Module.register_forward_pre_hook(..., with_kwargs=True)` interceptor fired 111 times per arm on real `Qwen3_5ForCausalLM` prefills; corrected `<\|vision_start\|><\|image_pad\|><\|vision_end\|>` placeholder produced zero "More image data items provided…" warnings; image was really consumed (greedy output describes the fixture's colours). But `Qwen/Qwen3.5-4B`'s `vision_config.deepstack_visual_indexes = []` (checked against every public `Qwen/Qwen3.5-*` size: 0.8B / 2B / 4B / 9B / 27B / 35B-A3B — all empty), so `num_deepstack_embeddings = 0`, `input_deepstack_embeds` is allocated `shape=(N, 0)` / `numel = 0`, and `Qwen3_5ForCausalLM.forward`'s DeepStack `add_` branch is trivially skipped by the `numel() > 0` guard. Runtime instrumentation confirms `nonzero_frac = 0.0` / `numel = 0` on every image request. The zero-substitution guard correctly leaves the empty tensor alone (0 `lm_forward_input_deepstack_zeroed` events), and greedy text is identical across the two arms. The BCG DeepStack correctness hypothesis (F5, F6, F7, F8) is **not testable against any publicly released `Qwen/Qwen3.5-*` checkpoint at the pinned SGLang SHA** — the model config never populates DeepStack in the first place. Step 3 (scored 2×2 rerun) is skipped by design under the brief's Step 2 fail-path rule; GPU 1 returned clean (0 MiB / 0 % / 0 compute apps); `/data/sglang-fork` HEAD unchanged. |

## Layout convention

```
results/
  <attempt-id>/
    verdict.md         # PASS / FAIL / AMBIGUOUS / INFRA_FAILURE
    verdict.json       # machine-readable verdict
    summary.md         # attempt narrative
    metadata.json      # environment fingerprint, launch context
    raw/               # per-run server logs, bench dumps (NOT committed
                       # unless explicitly approved)
```

`<attempt-id>` follows `attempt_gpu<N>_YYYYMMDDTHHMMSSZ` for hardware
runs, `harness_gpu<N>_YYYYMMDDTHHMMSSZ` for harness-validation runs
that stop before the scored 2×2 (introduced 2026-08-01), or
`attempt_cpu_YYYYMMDDTHHMMSSZ` for CPU-only self-checks.

## What must be committed

- `verdict.md`, `verdict.json`, `summary.md`, `metadata.json` (small,
  human-readable, review-friendly).

## What must not be committed without approval

- Any file inside `raw/` — raw server logs, bench JSONs, trace files.
- Anything larger than a few hundred kilobytes.
- Any file containing a moving `nvidia-smi` snapshot that includes
  other tenants' process names.

## What is forbidden here

- No attempt directories may be added until the user has authorised
  a GPU ID for the corresponding attempt.
