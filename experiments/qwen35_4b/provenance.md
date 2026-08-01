# Provenance — Qwen3.5-4B BCG DeepStack Investigation

> Frozen provenance the validation plan and future runners must verify
> before spending any GPU time. Anything that drifts from these values
> requires a fresh audit before results become comparable.

## 1. Repositories and SHAs

| Item | Value | Verify with |
|---|---|---|
| Profiler repo | `bowenwan6/sglang-vllm-profiler` (this repo) | `git remote -v` |
| Profiler branch | `debug/qwen35-4b-bcg-deepstack` | `git branch --show-current` |
| Profiler base commit | `a803285` (`main`, PR #8 merge) | `git merge-base HEAD main` |
| **Executed local SGLang checkout (HARD PIN)** | `<scratchpad>/sglang_checkout/sglang` cloned from `https://github.com/sgl-project/sglang.git`, HEAD pinned to `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`. The runner sources this via `PYTHONPATH=<scratchpad>/sglang_checkout/sglang/python` and verifies `sglang.__file__` resolves inside it. | `cd <scratchpad>/sglang_checkout/sglang && git rev-parse HEAD` and `python3 -c 'import sglang; print(sglang.__file__)'` |
| Upstream SGLang `main` HEAD at audit rebaseline | `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` (2026-08-01 refresh, subject `[perf] Assemble flat prompt top logprobs scheduler-side as numpy arrays (#32223)`) | `gh api repos/sgl-project/sglang/commits/main --jq .sha` |
| SGLang PR #30872 (`Enable multimodal prefill BCG for VL and audio models`) | **MERGED** 2026-07-28T22:47:40Z, merge SHA `c9947b087bf9d3d16b5198234ba4c39b68bb79e9`. Added Qwen3.5 to `multimodal_breakable_cuda_graph_supported_model_archs`, registered `input_embeds` slot, added `replay_layer_forward` copy of `input_embeds`. No DeepStack slot / copy on the BCG path. | `gh pr view 30872 --repo sgl-project/sglang` |
| SGLang PR #30868 (`fix: fix vlm cuda graph shape stability`) | **MERGED** 2026-07-19T14:35:51Z, merge SHA `d4801be44773`. Added `run_dummy_multimodal_deepstack_forward` and a defensive eager fallback, **both scoped to `tc_piecewise_cuda_graph_backend`**, not BCG. | `gh pr view 30868 --repo sgl-project/sglang` |
| Historical Qwen3-VL fork `/data/sglang-fork` | `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` | read-only reference; not touched by §7 |
| Local mirror `/sgl-workspace/sglang` HEAD (stale) | `da802dd` — do **not** trust as upstream truth or as the "installed" SGLang the runner uses. Runners must override via `PYTHONPATH` and verify `sglang.__file__`. | `cd /sgl-workspace/sglang && git rev-parse HEAD` |

**Provenance hardness convention (new).**

- The **hard pin** is the executed local SGLang checkout SHA. Runner
  preflight aborts with a non-zero exit code if:
  - the frozen checkout directory does not exist, or
  - its git HEAD does not equal
    `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`, or
  - `python3 -c 'import sglang; print(sglang.__file__)'` does not
    resolve inside that checkout after the runner's `PYTHONPATH`
    override.
- The **upstream `main` HEAD at query time** is informational. If it
  has moved past the frozen SHA, the preflight logs a WARN and
  continues; movement of remote main is not a hard failure because
  the executed code path is fully specified by the frozen checkout.
- The **HF model revision** is a hard pin (`851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`).
- The **fixture SHA-256** is a hard pin
  (`8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`).
- **torch / sgl_kernel / flashinfer / libcuda** are pinned soft: the
  preflight logs observed values and warns on any drift; a hard
  failure requires an explicit `--strict-env` flag.

## 2. Model target

### 2.1 Primary target (Attempts 01-02 — Qwen3.5-4B)

| Item | Value | Verify with |
|---|---|---|
| Model id | `Qwen/Qwen3.5-4B` | `curl -sL https://huggingface.co/api/models/Qwen/Qwen3.5-4B` |
| HF revision `sha` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | HF API `.sha` |
| `config.model_type` | `qwen3_5` | HF API `.config.model_type` |
| `config.architectures` | `["Qwen3_5ForConditionalGeneration"]` | HF API `.config.architectures` |
| `pipeline_tag` | `image-text-to-text` | HF API `.pipeline_tag` |
| `gated` | `false` | HF API `.gated` |
| `library_name` | `transformers` | HF API `.library_name` |
| `vision_config.deepstack_visual_indexes` | `[]` (empty in every shipped release; see `latent_bug_analysis.md` § 2) | HF `raw/main/config.json` `.vision_config.deepstack_visual_indexes` |

### 2.2 Retarget target (Attempt 03 onward — Qwen3-VL-8B under monkey-patched BCG)

Introduced 2026-08-01 per `validation_plan.md` Amendment 4. Testable
under the profiler-owned `scripts/bcg_allowlist_patch.py` monkey-patch
only; the shipped SGLang allowlist does not include Qwen3-VL.

| Item | Value | Verify with |
|---|---|---|
| Model id | `Qwen/Qwen3-VL-8B-Instruct` | `curl -sL https://huggingface.co/api/models/Qwen/Qwen3-VL-8B-Instruct` |
| HF revision `sha` | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` | HF API `.sha` |
| `config.model_type` | `qwen3_vl` | HF API `.config.model_type` |
| `config.architectures` | `["Qwen3VLForConditionalGeneration"]` | HF API `.config.architectures` |
| `pipeline_tag` | `image-text-to-text` | HF API `.pipeline_tag` |
| `gated` | `false` | HF API `.gated` |
| `library_name` | `transformers` | HF API `.library_name` |
| `vision_config.deepstack_visual_indexes` | `[8, 16, 24]` (3 layers) | HF `raw/main/config.json` `.vision_config.deepstack_visual_indexes` |
| `text_config.hidden_size` | `4096` | HF `raw/main/config.json` `.text_config.hidden_size` |
| Expected DeepStack tensor shape (per prefill token, at LM entry) | `[N, hidden_size * num_deepstack] = [N, 4096 * 3] = [N, 12288]` | Runtime `lm_forward_input_deepstack.input_deepstack_embeds.shape` |
| SGLang model source | `python/sglang/srt/models/qwen3_vl.py` (`Qwen3VLForConditionalGeneration`; LM class `Qwen3LLMModel`) | grep frozen checkout at pin |
| Language-model class name recorded by the pre-hook | `Qwen3LLMModel` (subclass of `Qwen3Model`) | Runtime `lm_forward_input_deepstack.module_class` |
| Runtime BCG opt-in | `scripts/bcg_allowlist_patch.py` (env `QWEN35_PATCH_BCG_ALLOWLIST=1` or `--patch-bcg-allowlist`); frozen SGLang source unchanged (`git diff` empty) | `python3 scripts/bcg_allowlist_patch.py --apply` |

**Snapshot pinning.** Any future runner **must** pass the exact
revision to `sglang.launch_server` (or the equivalent HF snapshot
download) so we cannot silently follow a moving `main` tag.

## 3. Environment expectations (from existing profiler baselines)

The Qwen3-VL sub-track froze a working environment at:

| Item | Value | Source |
|---|---|---|
| System python + torch | `python 3.12.3` · torch `2.11.0+cu130` | Qwen3-VL R6 provenance |
| CUDA runtime | `13.0` | environment snapshot |
| Host libcuda | `libcuda.so.595.71.05` (driver 595.71.05) at `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`; the `cuda-compat-13-0` loader precedence at `/usr/local/cuda-13.0/compat/libcuda.so.1` must be overridden with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` | R6 Amendment A3 |
| flashinfer | `0.6.12` | Qwen3-VL R6 provenance |
| sgl_kernel | `0.4.5` | upgraded 2026-08-01 for frozen SGLang `58974ca16` `assert_pkg_version` floor (was `0.4.4` in Qwen3-VL R6 provenance) |
| Profiling client env | `/opt/miniconda3/envs/profiling` (torch `2.11.0+cu130`, vLLM `0.21.0`) | Qwen3-VL R6 provenance |

**Warning.** These are the *starting expectations*. The validation
runner must re-verify each of them at run time; drift on
torch/sgl_kernel/flashinfer/libcuda is a WARN by default, and a hard
failure only under `--strict-env`.

## 4. Datasets and fixtures

- **Text dataset (reference control).** `datasets/qwen3vl8b/caseA_short.jsonl`
  (SHA-256 `fab4917772e087447d7c33d53ada63340b126088c1f195f118b9488d5f5b619e`
  from the Qwen3-VL R6 record). Reusable as a text-only control because
  Qwen3.5 accepts the same tokenizer text; **must be verified** at run
  time via `sha256sum` before use.
- **Image fixture.** `experiments/qwen35_4b/fixtures/image_bands.png`,
  SHA-256 `8fa3ed69d78049835d6631b3b4314be21ea3e797626be6c58fc72adfb30070a2`,
  1280×720 three-band PNG. Byte-pinned so repeated runs are comparable.

## 5. What is expressly not part of this investigation's provenance

- `/data/sglang-fork` — historical Qwen3-VL fork; **read-only**. The
  Step 2 runner must not modify it, and its HEAD is expected to
  remain `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` before and after
  every attempt (checked by the runner as a sanity guard).
- `experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/`
  — historical Qwen3-VL PCG capture-stream evidence; §7 references it
  but does not rewrite it. In particular the preserved uncommitted
  `R5C_correctness_audit/audit_report.md` edit and the orphan
  `R6.3_image_and_sweep/attempt_gpu2_partial_orphaned_…/` directory
  are protected: the runner and any commit must leave them
  unmodified.
- Any prior Qwen3-VL numbers (`21.94 ms`, `14.04 ms`, `64.8 ms`,
  R6.x results) — different model, different SHAs, not baselines for
  Qwen3.5-4B.

## 6. Provenance-verification rules for §7 runners

Every runner introduced under §7 **must** perform, and record in its
output, the following before doing any GPU work:

1. Emit the executed local SGLang checkout SHA (from `git -C
   <checkout> rev-parse HEAD`) and confirm it equals the hard pin.
   Non-match → abort with non-zero exit.
2. Emit the current `sgl-project/sglang` remote HEAD SHA (best-effort,
   via `gh api` or GitHub REST). If different from the pin, log a
   WARN; do not abort.
3. Import `sglang` and emit `sglang.__file__`. It must resolve
   under the frozen checkout after the runner's `PYTHONPATH`
   override. Non-match → abort.
4. Emit the HF model revision the runner is about to load; compare
   against §2. Non-match → abort unless `--waive-model-revision` is
   set.
5. Emit `nvidia-smi --query-gpu=driver_version,name,uuid --format=csv`
   for **GPU 0 only** (`--id=0`, no wildcard). Never query all GPUs.
6. Emit the loaded `libcuda.so` path via `ldconfig -p | grep libcuda`
   and confirm the runtime `LD_PRELOAD` targets
   `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` (WARN if
   different, ABORT if unset entirely).
7. Emit `python -c "import torch, sgl_kernel, flashinfer; print(...)"`
   version tuple. WARN on drift; ABORT under `--strict-env`.
