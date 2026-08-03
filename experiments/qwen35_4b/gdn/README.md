# Qwen3.5-4B — GDN Prefill-BCG Investigation

> **Investigation, not a confirmed bug.** This subtree profiles the
> hybrid GDN (Gated DeltaNet / linear-attention) layers in
> `Qwen/Qwen3.5-4B` under SGLang's prefill BCG (breakable CUDA graph)
> and full-decode CUDA-graph paths, to determine whether GDN's op
> shape is a bottleneck under BCG and where it breaks. Any perf claim
> is gated on correctness controls; no upstream source is modified
> until the baseline profile identifies one specific BCG limitation.

- **Branch:** `debug/qwen35-4b-gdn-prefill-bcg` (cut from
  `debug/qwen35-4b-bcg-deepstack` at commit `d29b4a6`, the
  DeepStack sub-track closure).
- **Model target:** `Qwen/Qwen3.5-4B` (HF revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`),
  BF16, single GPU, TP=1, no MTP, no quantization, no MoE, no custom
  model patches. `Qwen3_5ForConditionalGeneration` with the hybrid
  layer stack (`ALL_DECODER_LAYER_TYPES = {"attention", "linear_attention"}`
  at `models/qwen3_5.py:1225-1228`).
- **Executed local SGLang checkout (HARD PIN):** frozen at
  `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`
  (`<scratchpad>/sglang_checkout/sglang`). Runner sources via
  `PYTHONPATH` and verifies `sglang.__file__` resolves inside it.
- **Frozen sgl-kernel:** `0.4.5` (as of the DeepStack sub-track
  close). Full env in [`../provenance.md`](../provenance.md) §3.
- **Plan-level context:** `plan.md` §8.

## What this investigation is about

`Qwen/Qwen3.5-4B` is a **hybrid** model: some decoder layers are
standard attention (`Qwen3_5AttentionDecoderLayer` with
`self_attn`), others are linear-attention over Gated DeltaNet
(`Qwen3_5LinearDecoderLayer` with `linear_attn =
Qwen3_5GatedDeltaNet`). Per-layer selection comes from
`config.layers_block_type[idx]` at `models/qwen3_5.py:1360`. The
audit in [`source_audit.md`](source_audit.md) lays out the exact op
sequence inside a GDN layer.

GDN prefill under BCG raises two related questions:

1. **Coverage.** Which GDN ops are actually inside the captured BCG
   graph, and which end up outside it (host-side, launch-per-token,
   or eager fallback)? Under-covered ops manifest as CPU launch gaps,
   graph breaks, or per-token kernel-count inflation.
2. **Correctness under capture.** GDN has stateful primitives
   (`RadixLinearAttention` over a mamba-style KV pool, `A_log`,
   `dt_bias`, `conv1d`). Cross-stream sync via `alt_stream` is gated
   on `get_is_capture_mode()` and `seq_len < 1024` at
   `models/qwen3_5.py:562-566` — capture-mode branching that is
   subtly stateful. Any perf number is meaningless until we have
   proof that BCG's tokens/logprobs match the eager reference.

The pivot brief (operator, 2026-08-03) frames the investigation as:
identify *where* BCG limits GDN throughput, don't guess that
recurrent-state handling is faulty until the profile points there.

## 4-arm matrix

| Arm | Prefill backend | Decode CUDA-graph | Purpose |
|---|---|---|---|
| `A0` `eager_eager` | eager | eager | Reference for both correctness and no-capture latency. |
| `A1` `bcg_eager` | breakable prefill BCG | eager decode | Isolates prefill-side BCG effect. |
| `A2` `eager_dcg` | eager prefill | full decode CG | Isolates decode-side CG effect. |
| `A3` `bcg_dcg` | breakable prefill BCG | full decode CG | Production-shape configuration. |

Sweep per arm: prompt length ∈ `{128, 512, 2048, 8192}` × batch size
∈ `{1, 4, 16, 32}` (16 cells). Greedy decode; new tokens fixed at
`128` (short-enough to keep TTFT-dominated at small prompt lengths
and still exercise decode CG at larger batches).

## Predeclared verdict labels

The verdict runner must emit exactly one of the following for the
overall investigation. Per-arm / per-cell scoring is separate and
does not close the investigation on its own.

- **`PASS_BCG_GDN_NOTABLE_GAP`** — Correctness gates pass on every
  cell (see below) AND the profile pins **at least one specific
  BCG-side limitation** with quantitative support: a named op (from
  the GDN op list in `source_audit.md` § 3) that is repeatedly
  outside the captured graph, or a measurable and repeatable
  CPU-launch-gap / graph-break signature under BCG that is absent
  under eager. The finding is documented; no upstream fix is
  implemented on this branch.
- **`PASS_BCG_GDN_NO_GAP`** — Correctness gates pass AND the profile
  shows no material BCG-specific overhead beyond what CUDA-graph
  capture normally imposes. No BCG limitation to file; investigation
  closes as "GDN under BCG is not the bottleneck at this
  configuration".
- **`FAIL_BCG_GDN_CORRECTNESS`** — At least one correctness gate
  fails: eager-vs-BCG greedy token divergence beyond noise;
  chunked-prefill divergence; request-order dependence; graph-bucket
  divergence. No perf claim is admissible; investigation closes on
  the correctness gap.
- **`AMBIGUOUS`** — Nsight cannot separate signal from measurement
  noise; correctness gates pass but conflict with each other; or
  the arms cannot be run cleanly (e.g. one arm consistently OOMs at
  a cell the others survive).
- **`INFRA_FAILURE`** — Environment, GPU, or preflight failure
  (unchanged from the DeepStack sub-track's `INFRA_FAILURE`
  criteria).

**Any perf number reported without a passing correctness gate is a
protocol violation, not a finding.**

## Correctness gates (blocking)

Every arm must pass all four gates before any BCG performance
comparison is admissible. Each gate has a predeclared tolerance so
"pass" is not negotiable after the fact.

1. **Eager-vs-BCG token/logprob equivalence.** On a fixed golden
   set of `≥ 8` prompts (mixed lengths and content), each arm's
   greedy tokens must be bit-identical, and per-token logprobs must
   match `A0` within `|Δ| ≤ 0.05` (per-token max-abs). The
   `eager_normal` self-repeat noise floor is the reference; any arm
   whose divergence exceeds that noise floor **and** the tolerance
   fails.
2. **Request-order isolation.** For any batch-size sweep cell, the
   arm's output for a target request must be independent of the
   siblings in its batch: `serve request X alone` == `serve request
   X together with 3 arbitrary siblings` (tokens bit-identical,
   logprobs within tolerance). GDN state contamination between
   batched requests would show up here.
3. **Chunked-prefill equivalence.** For each prompt length ≥ 2048
   at batch 1, the arm's output under `--chunked-prefill-size N`
   for two distinct `N` values (a small `N` that forces multiple
   chunks and a large `N` that keeps a single chunk) must match.
4. **Graph-bucket equivalence.** For each arm, run the same request
   at prompt lengths that fall into two different BCG capture
   buckets, and confirm the output is stable across bucket
   boundaries. (Any GDN op that changes behaviour with capture-size
   granularity shows up here.)

Gates 1 and 2 are hard gates for **every** cell; gates 3 and 4 are
sampled (representative cells) but any observed failure fails the
overall verdict.

## Nsight Systems capture protocol

- Tool: `nsys profile` (Nsight Systems), CUDA + NVTX + osrt trace.
- One capture per (arm, prompt_len, batch) cell = 64 captures total.
- Per-capture warmup: 2 identical requests thrown away before the
  timed request set.
- Per-capture measured: `N=8` timed requests, `--tokens 128` new
  tokens, greedy.
- Metrics extracted per cell:
  - **Kernel count** per request (total and split GDN-vs-attention).
  - **CPU launch-gap distribution** (p50/p95/p99 of inter-launch
    gaps > 5 μs; identifies host-side stalls).
  - **Graph break count** (number of distinct `cudaGraphLaunch`
    entries per request; a fully captured prefill is one launch).
  - **TTFT** (request submit → first-token event).
  - **Prefill throughput** (prompt tokens / prefill duration).
- Extraction driven by `nsys stats` post-processing, not by manual
  UI export.
- Store raw `.nsys-rep` files in `raw/` (gitignored). Commit only
  the extracted per-cell CSVs, the per-cell summary tables, and the
  comparison plots.

## Layout (planned, this directory)

| Path | Purpose | Status |
|---|---|---|
| `README.md` | This file — investigation entry point. | landed |
| `hypothesis.md` | Established facts / source-level observations / runtime hypotheses / verdict criteria (mirrors DeepStack sub-track's structure). | landed |
| `source_audit.md` | GDN forward path, ops, BCG interaction points, hybrid-layer selection, all with line refs at frozen SGLang SHA `58974ca1`. | landed |
| `validation_plan.md` | 4-arm × 4×4 sweep, correctness gates, Nsight protocol, cell tables, verdict rules. | landed |
| `provenance.md` | GDN-specific pins (`layers_block_type`, `linear_num_key_heads`, `linear_num_value_heads`, `linear_conv_kernel_dim`, `linear_key_head_dim`, `linear_value_head_dim`) — extends `../provenance.md` §2.1 with runtime-verifiable GDN fields. | landed |
| `scripts/` | CPU-only scaffolding: 4-arm runner, sweep client, correctness verifier, Nsight wrapper. All refuse GPU without explicit auth. | evolving |
| `results/` | Per-cell outputs (raw `.nsys-rep` gitignored; extracted CSVs + summaries committed). | populated as cells land |

## Read order

1. `plan.md` §8 (top-level pivot context).
2. `hypothesis.md` (what is established vs suspected).
3. `source_audit.md` (why we suspect it — direct source citations).
4. `provenance.md` (which config fields we pin at run time).
5. `validation_plan.md` (how we prove or disprove — 4-arm matrix,
   correctness gates, Nsight capture protocol, verdict rules).
6. `scripts/` (CPU-only scaffolding).
7. `results/` (populated as cells land).

## Rules (recorded so they are not re-litigated)

- **No upstream SGLang source modification** until the baseline
  profile identifies one specific BCG limitation. Profiler-owned
  monkey-patches are allowed only for measurement — the previous
  sub-track's `scripts/bcg_allowlist_patch.py` pattern (opt-in env
  var, idempotent, `git diff --stat` empty on the frozen checkout)
  is the model.
- **No checkpoint mutation** to force GDN behavior. The target model
  is `Qwen/Qwen3.5-4B` as shipped by HF at
  `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- **No perf claim without a passing correctness gate.** Every cell
  must clear all four correctness gates before its `nsys` metrics
  are admissible.
- **GPU allowlist `{0, 1, 7}`** and idle-verification rules from
  `../validation_plan.md` Amendments 1-2 continue to apply.
- **Preservation invariants unchanged.** Read-only
  `/data/sglang-fork` at `986c89e69`; frozen SGLang source
  untouched; §4 evidence read-only; the DeepStack sub-track's
  Attempts 01/02/03 preserved verbatim.
- **Do not assume recurrent-state handling is faulty.** Follow the
  profile evidence — a repeated graph break or a measurable
  launch-overhead bottleneck is the acceptable trigger for a
  targeted deep-dive on any specific GDN op.
