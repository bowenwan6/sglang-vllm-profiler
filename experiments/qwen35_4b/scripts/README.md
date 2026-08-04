# `scripts/`

CPU-only scaffolding + live runner for the Qwen3.5-4B BCG DeepStack
validation. Every script below refuses to touch a GPU unless GPU 0 is
explicitly authorised and idle-verified.

## Files

| File | Purpose |
|---|---|
| `generate_fixture.py` | Bit-identical regeneration of `fixtures/image_bands.png`. |
| `preflight_provenance.py` | Verifies frozen SGLang checkout SHA, imported `sglang.__file__`, HF model revision, torch / sgl_kernel / flashinfer / libcuda. |
| `runner.sh` | Live runner. Requires `--gpu-id` in `{0, 1, 7}` (or `QWEN35_GPU_ID=…`). Launches the SGLang server against the frozen SGLang checkout, drives the client, runs the verdict. Accepts `--config` in `{eager_normal, eager_zero_deepstack, bcg_normal, bcg_zero_deepstack}`. |
| `runner_skeleton.sh` | Kept as the CPU-only skeleton for the launch-context JSON / PGID scaffolding that `runner.sh` exercises. Superseded by `runner.sh` for live work. |
| `client.py` | Live client — issues matched `eager_normal`, `eager_zero_deepstack`, `bcg_normal`, and text-only requests; records per-request instrumentation. |
| `client_skeleton.py` | CPU-only skeleton kept for dry-run testing. |
| `instrumentation.py` | Branch-owned monkey-patch: records execute-path (BCG vs eager), and — via a per-call `nn.Module.register_forward_pre_hook(hook, with_kwargs=True)` scoped to one `general_mm_embed_routine` call and removed in `finally` — the incoming `input_deepstack_embeds` shape / dtype / numel / finite / nonzero_frac / abs_sum / sq_sum / SHA-256-16 / diagnostic pointer. Provides both `eager_zero_deepstack` and `bcg_zero_deepstack` ablation hooks (records a post-substitution summary proving the replacement is really zero). Applied at server-launch time via a `sitecustomize.py`-style loader; reverted at teardown. |
| `test_instrumentation.py` | CPU-only tests for the DeepStack pre-hook: normal mode observes nonzero DeepStack without changing module output, zero mode substitutes and verifiably zeros the tensor, hooks are removed after each call, repeated calls do not accumulate hooks. |
| `verdict.py` | Real verdict inference from `metadata.json` + `raw/*.json`. Emits `verdict.json` + `verdict.md`. Refuses to score across mismatched launch IDs. |

## Rules for any runner in this directory

1. **Authorised allowlist `{0, 1, 7}` only.** The runner requires
   `--gpu-id` in `{0, 1, 7}` (or `QWEN35_GPU_ID=…`); any other value
   exits non-zero before any CUDA import.
2. **No process-wide killers.** Never call `pkill`, `killall`,
   `fuser -k`, `nvidia-smi --gpu-reset`, or anything that signals a
   PID the runner did not itself launch.
3. **Ownership-verified PGID cleanup.** The runner records the PGID
   of every process group it launches and, on cleanup, signals only
   that PGID after verifying every PID in it was recorded by this
   runner.
4. **Foreign process abort.** If a foreign compute process is present
   on the target GPU at launch time (filtered by that GPU's UUID), the
   runner exits non-zero (dedicated code 71) without attempting to
   signal the foreign PID.
5. **Provenance preflight.** Before any GPU allocation, the runner
   verifies the hard pins in `provenance.md` §6.
6. **Frozen SGLang checkout is the source of truth.** The runner
   overrides `PYTHONPATH` to point at
   `<scratchpad>/sglang_checkout/sglang/python` and asserts
   `sglang.__file__` resolves inside it. `/data/sglang-fork` and
   `/sgl-workspace/sglang` are NOT the source of truth.
7. **Raw evidence is preserved but not committed by default.** Raw
   logs live under `results/<attempt-id>/raw/`; commits stage only
   `verdict.*`, `summary.md`, `metadata.json`.
8. **CPU-only preflight self-test.** Every runner supports a
   `--dry-run` that exercises argument parsing, provenance emission,
   and cleanup wiring without calling into CUDA.

## What is forbidden

- Downloading model weights during dry-run.
- Any `nvidia-smi --gpu-reset` invocation.
- Editing `/data/sglang-fork`.
- Using any GPU outside the authorised allowlist `{0, 1, 7}`.
