# Stage-3 threshold ladder — GPU 6

**Purpose.** Test the leading H12.1 hypothesis from source_audit.md
§3.3 — is the GDN alt-stream branch (`_forward_input_proj`, fires
when `seq_len < DUAL_STREAM_TOKEN_THRESHOLD (=1024)` where seq_len
is the padded BCG bucket size) the mechanism behind Stage-2's
steady-state +13.6% A1-vs-A0 kernel inflation?

## Configuration

- 4 prompt-length targets × 2 arms (A0, A1) × 2 reps = **16 Nsight
  captures**, all rc=0 on GPU 6.
- Each cell: batch=1, n_warmup=2, n_timed=8, new_tokens=128, greedy.
- Windowed extractor (`capture-cutoff-seconds = server_ready_seconds`)
  per cell.

## Result — hypothesis REJECTED

| prompt_len | actual tok | padded bucket | A0 mean | A1 mean | Δ | pct | branch |
|---|---|---|---|---|---|---|---|
| **128**  | 72–98 | 80–112 | 679,358 | 772,070 | +92,712 | **+13.6 %** | **ON** |
| **1024** | 571–765 | 576–768 | 682,171 | 773,753 | +91,582 | **+13.4 %** | **ON** |
| **2048** | 1139–1528 | 1280–1536 | 681,954 | 775,624 | +93,670 | **+13.7 %** | **OFF** |
| **4096** | 2277–3052 | 2304–3072 | 681,916 | 774,992 | +93,076 | **+13.6 %** | **OFF** |

**The A1-vs-A0 kernel inflation is essentially constant (13.4–13.7 %)
across all prompt-length targets, regardless of whether the padded
bucket is above or below 1024.**

If the alt-stream branch were the mechanism, the delta would shrink
to near zero at p=2048 and p=4096 (branch disabled). It doesn't.

The alt-stream `<1024` branch is **NOT** the source of the +13.6 %
steady-state kernel inflation observed in Stage 2. Hypothesis
`H_12.1` from source_audit.md §3.8 is **REJECTED**.

## Reproducibility (rep1 vs rep2 within cell, steady-state kernels)

| cell | rep1 | rep2 | \|Δ\| | pct of rep1 |
|---|---|---|---|---|
| A0 p=128 | 679,358 | 679,358 | 0 | 0.000 % |
| A1 p=128 | 772,049 | 772,091 | 42 | 0.005 % |
| A0 p=1024 | 682,171 | 682,171 | 0 | 0.000 % |
| A1 p=1024 | 775,136 | 772,370 | 2,766 | 0.357 % |
| A0 p=2048 | 681,954 | 681,953 | 1 | 0.000 % |
| A1 p=2048 | 775,622 | 775,625 | 3 | 0.000 % |
| A0 p=4096 | 681,918 | 681,914 | 4 | 0.001 % |
| A1 p=4096 | 774,948 | 775,036 | 88 | 0.011 % |

All cells reproducible within ≤0.36 %. Signal-to-noise is very high
for the +13.6 % A1-vs-A0 delta.

## Per-request steady-state kernels

| prompt_len | A0 kern/req | A1 kern/req | Δ | pct | A1 GL/req |
|---|---|---|---|---|---|
| 128 | 67,935.8 | 77,207.0 | +9,271.2 | +13.6 % | **36.3** |
| 1024 | 68,217.1 | 77,375.3 | +9,158.2 | +13.4 % | **36.3** |
| 2048 | 68,195.4 | 77,562.4 | +9,367.0 | +13.7 % | **36.3** |
| 4096 | 68,191.6 | 77,499.2 | +9,307.6 | +13.6 % | **36.3** |

**cudaGraphLaunch per request is CONSTANT at 36.3 across every
prompt length.** BCG dispatches ~36 graph replays per prefill
regardless of the padded bucket. This suggests per-layer /
per-segment bookkeeping (Qwen3.5-4B has 32 layers; 36 graph
launches per prefill maps well to ~one replay per layer plus
prologue/epilogue segments), not any per-token work.

## Additional pattern — A0 grows very slowly with prompt length

A0 steady-state kernels are essentially size-independent:
679,358 → 682,171 → 681,954 → 681,916. This is because A0 uses
eager prefill, so kernel COUNT per prefill is nearly constant per
model layer (per-token work happens inside kernels). Only the
per-kernel launch args vary with size, not the number of launches.

A1 grows equally slowly (772K → 773K → 775K → 775K). The +9,300
extra kernels/request on A1 is essentially independent of prompt
length.

## Interpretation

The Stage-2 +13.6 % delta is **not** the alt-stream branch. It is
some **per-request/per-prefill fixed overhead** in BCG that
scales with per-layer/per-segment count, not with token count or
bucket size.

Candidates that fit the observed shape (constant per-request delta,
constant graph-launches-per-prefill):

- **Per-segment `.replay()` dispatch overhead**: BCG breaks the
  forward into eager-attention-broken segments; each segment
  requires a graph-launch + eager-attention block. With ~32 GDN
  layers + ~8 full-attention layers eager-breaks + prologue/epilogue,
  ~36 graph replays × per-request bookkeeping (metadata setup,
  buffer copies) could account for ~9K kernels.
- **Per-request static-slot copy-in**: `replay_layer_forward`
  copies `input_embeds` (and other static-slot inputs) into
  buffer-registry slots on every prefill. Under BCG this is per-
  request; under eager it's a plain-tensor pass-through with no
  copy.
- **Mamba state pointer bookkeeping**: `MambaPool` slice/reindex
  operations per replay, not per token.
- **`_prepare_forward_metadata_for_replay` per replay**: rebuilds
  `mamba_cache_indices`, `query_start_loc`, `extend_seq_lens` from
  live batch state. If it launches small kernels for reindex ops,
  those pile up over 36 replays.

## Stage-3 signal

`SIGNAL_AMBIGUOUS` **resolved internally**: hypothesis rejected but
the +13.6 % delta is real and reproducible. Continue automatically
to Stage 4 (targeted diagnosis using existing Nsight kernel names
and stream IDs before adding NVTX).

## Preservation invariants (verified post-Stage 3)

- `/data/sglang-fork` HEAD unchanged: `986c89e69c…`.
- Frozen SGLang HEAD unchanged: `58974ca16c…`, empty `git diff --stat`.
- GPU 6 memory 0 MiB post-run.

## Files

- `stage3_summary.txt` — per-run CSV (16 rows).
- `driver.log` — driver console.
- `<arm>_p<len>_<rep>/` — 16 subdirs each with `metadata.json`,
  `gpu_pre.txt`, `gpu_post.txt`, `preflight.json`, `runner_*.log`,
  `client_*.log`, `records_*.jsonl`, `extract.log`, and
  `nsys/<arm>_p<len>_b1.csv` (3-row windowed extract).
  `raw/*.nsys-rep` and `raw/server_*.log` are gitignored.
