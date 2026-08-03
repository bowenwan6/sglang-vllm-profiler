# Validation Plan — Qwen3.5-4B GDN Prefill-BCG

> The 4-arm sweep, correctness gates, Nsight capture protocol, and
> verdict rules. Reads with [`README.md`](README.md),
> [`hypothesis.md`](hypothesis.md), and
> [`source_audit.md`](source_audit.md).

## 1. Design

**Target:** `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`,
BF16, single GPU, TP=1, no MTP, no quantization, no MoE, no custom
model patches.

**Frozen SGLang checkout:** `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`,
verified via `sglang.__file__` inside the checkout.

**Four arms** (from `plan.md` §8):

| Arm id | Prefill | Decode | SGLang flags |
|---|---|---|---|
| `A0` `eager_eager` | eager | eager | `--disable-cuda-graph --disable-breakable-cuda-graph` |
| `A1` `bcg_eager` | BCG | eager decode | `--disable-cuda-graph` (BCG on by default when supported) |
| `A2` `eager_dcg` | eager | full-decode CG | `--disable-breakable-cuda-graph` |
| `A3` `bcg_dcg` | BCG | full-decode CG | (defaults) |

Exact flag names must match the frozen SGLang CLI — resolved by the
runner's `--dry-run` printout before any GPU work.

**Sweep** (per arm): prompt length ∈ `{128, 512, 2048, 8192}` ×
batch size ∈ `{1, 4, 16, 32}` → 16 cells × 4 arms = **64 cells**.

Per-cell request shape: greedy, `--tokens 128` new tokens. Warmup:
2 identical requests discarded before the timed set. Timed: `N=8`
requests per cell.

## 2. Fixtures and inputs

- **Prompt set.** 8 golden prompts spanning short/long, code/prose,
  and multi-turn shapes; SHA-256 pinned in
  `fixtures/gdn_prompts.jsonl`. To be generated CPU-only by
  `scripts/generate_gdn_prompts.py` (bit-identical regeneration
  required).
- **Prompt-length materialisation.** Each of the 4 target lengths
  is a fixed-token-count expansion of the golden prompts (padded /
  truncated deterministically). Recorded per-cell so that the same
  request bytes are used across arms and captures.
- **No new image fixtures.** GDN is a text-only investigation on
  Qwen3.5-4B (which is registered as multimodal in SGLang but has
  no shipped DeepStack — the DeepStack sub-track closure is the
  reference for that).

## 3. Provenance preflight (blocking)

Reuse the DeepStack sub-track's `provenance.md` §6 rules with GDN
extensions. Before any GPU work the runner must record:

1. Frozen SGLang checkout HEAD == `58974ca16…` (hard-fail on mismatch).
2. Upstream `sgl-project/sglang` main HEAD (informational WARN
   only).
3. `python3 -c 'import sglang; print(sglang.__file__)'` resolves
   inside the frozen checkout.
4. HF model revision `Qwen/Qwen3.5-4B` == `851bf6e8…` (hard-fail
   without `--waive-model-revision`).
5. `nvidia-smi --id=<GPU_ID> --query-gpu=driver_version,name,uuid`
   for the single authorised GPU. Never wildcard.
6. `LD_PRELOAD` targets `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05`
   (WARN if unset, ABORT if unset in strict mode).
7. `torch`, `sgl_kernel`, `flashinfer` versions (WARN on drift;
   ABORT under `--strict-env`).
8. **New GDN pins** (fail-fast if any are missing / degenerate):
   - `config.layers_block_type` present and non-empty; count of
     `"attention"` and `"linear_attention"` layers logged. If
     either count is 0 the sweep aborts with `INFRA_FAILURE`
     (the target is not actually hybrid).
   - `config.linear_num_key_heads`, `linear_num_value_heads`,
     `linear_key_head_dim`, `linear_value_head_dim`,
     `linear_conv_kernel_dim` present. Ratio
     `linear_num_value_heads / linear_num_key_heads` recorded to
     confirm whether the fused
     `fused_qkvzba_split_reshape_cat_contiguous` path is hit
     (`{1, 2, 4}`) or the Python fallback path is hit.
   - Env vars affecting GDN capture recorded: `_gdn_use_alt_stream`
     value (from a Python probe on the imported module),
     `SGLANG_GDN_QKVZ_BA_ALT_STREAM`.

## 4. Correctness gates (blocking, before any perf claim)

Every arm must pass all four gates on every cell before its Nsight
metrics are admissible in a verdict. Tolerances are predeclared;
loosening them requires an Amendment.

### Gate 1 — Eager-vs-BCG token/logprob equivalence

- Reference: `A0` `eager_eager` on the golden prompt set.
- Test: for each other arm and each prompt, greedy tokens **must be
  bit-identical** to `A0` (128 new tokens each).
- Logprob tolerance: per-token max absolute logprob delta from `A0`
  ≤ `0.05`, established against the `A0` self-repeat noise floor
  measured on the same server (a first-run pilot records the noise
  floor; the tolerance is `max(0.05, 3 × noise_floor)`).
- Failure verdict: `FAIL_BCG_GDN_CORRECTNESS`.

### Gate 2 — Request-order isolation

- For each batch-size cell `b ∈ {4, 16, 32}` and each target
  request `x`, run `x` alone and run `x` alongside `b-1` sibling
  requests drawn deterministically from the golden set.
- Tokens for `x` must be bit-identical between the two runs;
  logprobs within the Gate 1 tolerance.
- GDN state contamination between batched requests would surface
  here.

### Gate 3 — Chunked-prefill equivalence

- For prompt length ∈ `{2048, 8192}` at batch 1, run each arm at
  two `--chunked-prefill-size` values: one that forces multiple
  chunks and one that fits the whole prompt.
- Tokens must be bit-identical between the two chunking settings
  within an arm; logprobs within Gate 1 tolerance.

### Gate 4 — Graph-bucket equivalence

- For each BCG-enabled arm, run the same request at two prompt
  lengths that fall into two different BCG capture buckets (as
  reported in the server-startup log's `capture_num_tokens` list).
- Same-request tokens must be bit-identical between the buckets.

**Gates 1 and 2 are mandatory for every cell. Gates 3 and 4 are
sampled at representative cells; any observed failure fails the
overall verdict.**

## 5. Nsight capture protocol

Runs only after all correctness gates for the target cell have
passed.

- Tool: `nsys profile` (Nsight Systems, whichever version ships in
  the base image). Trace flags: `-t cuda,nvtx,osrt,cudnn,cublas`.
- One `.nsys-rep` per (arm, prompt_len, batch) cell = 64 raw traces
  total per full sweep. Stored in `results/<attempt>/raw/`
  (gitignored). Only the extracted CSVs are committed.
- Per-capture: 2 warmup requests (discarded) + 8 timed requests.
- **NVTX ranges** inserted by a profiler-owned instrumentation
  patch (compatible with the DeepStack sub-track's `instrumentation.py`
  and its `sitecustomize.py` bootstrap):
  - `gdn.forward.layer_<L>` around each `Qwen3_5GatedDeltaNet.forward`.
  - `gdn.<op>` for each of the 10 ops in `source_audit.md` §3
    (`in_proj_qkvz`, `in_proj_ba`, `alt_stream_sync`,
    `fused_qkvzba_split`, `python_split_cat`, `conv1d_state`,
    `radix_linear_attn`, `dp_padding`, `rmsnorm_gated`, `out_proj`).
  - `layer.attention` around every standard-attention layer for
    cross-comparison.
  - Instrumentation must be a hook (no source-file editing on the
    frozen checkout); the DeepStack sub-track's
    `register_forward_pre_hook(with_kwargs=True)` pattern is the
    template.
- **Extraction (post-run, CPU-only):**
  - `nsys stats` → per-cell CSV with columns:
    `arm, prompt_len, batch, request_id, kernel_count_total,
    kernel_count_gdn, kernel_count_attn, cudagraphlaunch_count,
    cudalaunchkernel_count, ttft_ms, prefill_throughput_toks_per_s,
    p50_launch_gap_us, p95_launch_gap_us, p99_launch_gap_us,
    graph_breaks`.
  - `graph_breaks` counts `cudaLaunchKernel` events that occur
    between the outer prefill's `cudaGraphLaunch` entries (i.e.
    launches that are *not* part of the captured graph).
  - `p*_launch_gap` computed from the CPU-side `cudaLaunchKernel` /
    `cudaGraphLaunch` timestamps.

## 6. Arm-comparison thresholds

Per (prompt_len, batch) cell, compare each of `{A1, A2, A3}`
against `A0`:

- **`H_A` support (kernel-count inflation)**: `kernel_count_gdn`
  strictly greater than the `A0` mean by ≥ `10 %` and by more than
  `2 × A0_stddev`, holding across ≥ 2 captures.
- **`H_B` support (graph break)**: `graph_breaks` strictly greater
  than 0 under a BCG-enabled arm on a request whose prefill fits
  in one BCG bucket. Reproducible across ≥ 2 captures.
- **`H_C` support (launch-overhead bottleneck)**: `p95_launch_gap`
  under a BCG-enabled arm ≥ `2 ×` the `A0` `p95_launch_gap` at the
  same cell, with the excess concentrated in an NVTX range tagged
  to a specific GDN op.
- **`H_D` support (no gap)**: none of `H_A`, `H_B`, `H_C` holds at
  any cell; TTFT and prefill throughput follow the expected
  CUDA-graph capture envelope (BCG-enabled TTFT ≤ eager TTFT by a
  margin consistent with the standard-attention part of the stack).

## 7. Confounder controls

- **KV-cache warmup and isolation.** Each per-arm session brings
  the server up cold, warms with the 2 discard requests, then
  runs the timed set — no shared cache across arms.
- **GPU idleness.** Runner queries the target GPU (only) for idle
  compute processes, memory ≤ 500 MiB, util ≤ 5 % at start and
  end. Never signals a foreign PID.
- **`LD_PRELOAD`.** Same
  `/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` override the
  DeepStack sub-track uses.
- **`spawn` propagation.** The instrumentation and NVTX-tagging
  must reach SGLang scheduler / worker subprocesses via the
  existing `scripts/bootstrap/sitecustomize.py`.
- **Nsight overhead disclosure.** Every reported per-cell metric
  is under Nsight; a small sub-sweep (arm `A0` at 4 cells) is run
  *without* Nsight to confirm Nsight overhead does not swamp the
  A0-vs-A1 delta of interest.

## 8. Verdict runner

`scripts/gdn_verdict.py` (planned) ingests the per-cell CSVs plus
the correctness-gate outputs and emits exactly one of the labels
in `hypothesis.md` §5.

Preconditions (all must hold or the runner emits `AMBIGUOUS` or
`INFRA_FAILURE`):

- Provenance preflight passed.
- All 64 cells produced correctness-gate results and Nsight CSVs.
- Correctness gates all pass, or a gate failure is present and
  the runner emits `FAIL_BCG_GDN_CORRECTNESS`.
- Arm-comparison thresholds are applied uniformly.

## 9. Results directory layout

```
gdn/results/
├── README.md              # attempt index (append-only)
├── <attempt_id>/
│   ├── metadata.json      # provenance, flags, env, timing
│   ├── correctness/
│   │   ├── gate1_tokens.csv
│   │   ├── gate2_isolation.csv
│   │   ├── gate3_chunking.csv
│   │   └── gate4_buckets.csv
│   ├── nsys/
│   │   └── <arm>_p<prompt>_b<batch>.csv
│   ├── raw/               # gitignored
│   │   └── <arm>_p<prompt>_b<batch>.nsys-rep
│   ├── summary.md
│   ├── verdict.json
│   └── verdict.md
```

Attempt naming pattern (matches DeepStack sub-track):
`gdn_attempt_gpu<id>_<YYYYMMDDTHHMMSSZ>/`.

## 10. Rules

Restated from `README.md` § "Rules" for the plan reader:

- No upstream SGLang source modification until the baseline
  profile identifies one specific BCG limitation.
- No checkpoint mutation; use shipped `Qwen/Qwen3.5-4B`.
- No perf claim without a passing correctness gate.
- GPU allowlist `{0, 1, 7}` and idle-verification rules from
  `../validation_plan.md` Amendments 1-2 apply unchanged.
- Preservation invariants unchanged: read-only
  `/data/sglang-fork`; frozen SGLang source untouched; §4
  evidence read-only; DeepStack Attempts 01/02/03 preserved.
- No assumption that recurrent-state handling is faulty until
  the baseline profile points there.
