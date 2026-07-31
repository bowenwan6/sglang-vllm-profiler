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
| Upstream SGLang `main` HEAD at audit start | `5f9b0db18c787cf56ed9bbaf255f083f26c6ebc2` (2026-07-31) | `gh api repos/sgl-project/sglang/commits/main --jq .sha` |
| SGLang PR #30872 (`Enable multimodal prefill BCG for VL and audio models`) | **MERGED** 2026-07-28T22:47:40Z, merge SHA `c9947b087bf9d3d16b5198234ba4c39b68bb79e9` | `gh pr view 30872 --repo sgl-project/sglang` |
| SGLang PR #30868 (`fix: fix vlm cuda graph shape stability`) | **MERGED** 2026-07-19T14:35:51Z, merge SHA `d4801be44773` | `gh pr view 30868 --repo sgl-project/sglang` |
| Historical Qwen3-VL fork `/data/sglang-fork` | `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` | read-only reference; not touched by §7 |
| Local mirror `/sgl-workspace/sglang` HEAD (stale) | `da802dd` — do **not** trust as upstream truth | `cd /sgl-workspace/sglang && git rev-parse HEAD` |

## 2. Model target

| Item | Value | Verify with |
|---|---|---|
| Model id | `Qwen/Qwen3.5-4B` | `curl -sL https://huggingface.co/api/models/Qwen/Qwen3.5-4B` |
| HF revision `sha` | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` | HF API `.sha` |
| `config.model_type` | `qwen3_5` | HF API `.config.model_type` |
| `config.architectures` | `["Qwen3_5ForConditionalGeneration"]` | HF API `.config.architectures` |
| `pipeline_tag` | `image-text-to-text` | HF API `.pipeline_tag` |
| `gated` | `false` | HF API `.gated` |
| `library_name` | `transformers` | HF API `.library_name` |

**Snapshot pinning.** Any future runner **must** pass the exact
revision to `sglang.launch_server` (or the equivalent HF snapshot
download) so we cannot silently follow a moving `main` tag.

## 3. Environment expectations (from existing profiler baselines)

The Qwen3-VL sub-track froze a working environment at:

| Item | Value | Source |
|---|---|---|
| System python + torch | `python 3.12.3` · torch `2.11.0+cu130` | Qwen3-VL R6 provenance |
| CUDA runtime | `13.0` | environment snapshot |
| Host libcuda | `libcuda.so.595.71.05` (driver 595.71.05) | R6 Amendment A3 |
| flashinfer | `0.6.12` | Qwen3-VL R6 provenance |
| sgl_kernel | `0.4.4` | Qwen3-VL R6 provenance |
| Profiling client env | `/opt/miniconda3/envs/profiling` (torch `2.11.0+cu130`, vLLM `0.21.0`) | Qwen3-VL R6 provenance |

**Warning.** These are the *starting expectations*. The validation plan
must re-verify each of them at run time (see `scripts/` preflight in
Part 5) because the upstream SGLang HEAD may have changed pinned deps
since the Qwen3-VL sub-track froze them.

## 4. Datasets and fixtures

- **Text dataset (reference control).** `datasets/qwen3vl8b/caseA_short.jsonl`
  (SHA-256 `fab4917772e087447d7c33d53ada63340b126088c1f195f118b9488d5f5b619e`
  from the Qwen3-VL R6 record). Reusable as a text-only control because
  Qwen3.5 accepts the same tokenizer text; **must be verified** at run
  time via `sha256sum` before use.
- **Image fixture.** Defined in `validation_plan.md`. Must be a
  deterministic, byte-pinned file so repeated runs are comparable.
  Committed under `experiments/qwen35_4b/fixtures/` at scaffolding time
  (Part 5).

## 5. What is expressly not part of this investigation's provenance

- `/data/sglang-fork` — historical Qwen3-VL fork; read-only.
- `experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause/`
  — historical Qwen3-VL PCG capture-stream evidence; §7 references it
  but does not rewrite it.
- Any prior Qwen3-VL numbers (`21.94 ms`, `14.04 ms`, `64.8 ms`,
  R6.x results) — different model, different SHAs, not baselines for
  Qwen3.5-4B.

## 6. Provenance-verification rules for §7 runners

Every runner introduced under §7 **must** perform, and record in its
output, the following before doing any GPU work:

1. Emit the current `sgl-project/sglang` HEAD SHA reachable at run
   time. If different from the SHA in this file, warn loudly and
   record both values; do **not** silently proceed as if unchanged.
2. Emit the HF model revision it is about to load; compare byte-for-byte
   against §2.
3. Emit `nvidia-smi --query-gpu=driver_version,name --format=csv` for the
   authorised GPU only (no GPU allocation), and the loaded `libcuda.so`
   path via `ldconfig -p | grep libcuda`.
4. Emit `python -c "import torch, sgl_kernel, flashinfer; print(...)"`
   version tuple.
5. Abort with a non-zero exit code if any hard-required item disagrees
   with the pinned value and the user has not explicitly waived the
   check.
