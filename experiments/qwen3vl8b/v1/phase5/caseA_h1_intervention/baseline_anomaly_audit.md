# Case A — Baseline Reproducibility Anomaly & Instrumentation Confound (audit)

This audit explains why the Phase 5.2 Case A intervention is **downgraded to an instrumented
exploratory screen** and not a clean causal result.

## The anomaly

The Phase 5.2 SGLang baseline (`S0_baseline`) did **not** reproduce the Phase 2 Case A `no_overlap`
TTFT, despite identical server graph configuration:

| Metric | Phase 2 `no_overlap` | Phase 5.2 `S0_baseline` |
|---|---:|---:|
| TTFT p50 median | **19.6 ms** | **53.28 ms** |

## Server-args consistency (graph config identical)

| Server arg | Phase 2 `no_overlap` | Phase 5.2 `S0_baseline` |
|---|---|---|
| `disable_cuda_graph` | False | False |
| `disable_overlap_schedule` | True | True |
| `enable_torch_compile` | False | False |
| `disable_piecewise_cuda_graph` | True (VLM auto-disable) | True (VLM auto-disable) |
| `enforce_piecewise_cuda_graph` | False | False |
| `piecewise_cuda_graph_max_tokens` | 8192 | 8192 |

→ The graph configuration is the **same**; config differences do **not** explain the 19.6 → 53.28 ms gap.

## The protocol difference: KAPI instrumentation

The most direct protocol difference is **kernel-API (KAPI) logging**, which the Phase 5 runner enabled
for **every SGLang variant** but Phase 2 did **not** use:

- Phase 5 runner set `SGLANG_KERNEL_API_LOGLEVEL=1` + `SGLANG_KERNEL_API_LOGDEST=...` on all SGLang launches.
- Phase 2 Case A benchmark ran **without** KAPI logging.
- The vLLM anchor (V0) was **never** KAPI-instrumented (KAPI is SGLang-only) → V0 is uncontaminated.

### Evidence: KAPI log volume tracks eager-launch count

KAPI logs one record per kernel-API call. Observed Phase 5 KAPI log sizes:

| Variant | KAPI log size | TTFT p50 | note |
|---|---:|---:|---|
| **S1 graph-off** | **14.7 GB** | 53.6 ms (TPOT 47.7 ms!) | every eager decode launch logged → 8.6× TPOT blow-up + LFS push failure |
| S0 baseline | ~126 MB | 53.28 ms | eager prefill launches logged |
| S3 torch.compile | ~124 MB | 54.05 ms | |
| **S2 enforce piecewise** | **~40 MB** | **19.74 ms** | piecewise graph → **far fewer logged launches** |

The ordering S1 ≫ S0 ≈ S3 > **S2** shows KAPI logging volume scales with the number of
direct/eager kernel launches, and S2 (piecewise graph) logs the least.

## Why this pollutes the S0/S1/S2/S3 comparison

KAPI logging adds per-launch host-side work (string formatting + I/O) on the **CPU launch path** — the
very path the eager/direct-launch variants stress most. So the instrumentation **penalizes eager
variants (S0/S1/S3) more than the graph variant (S2)**, because S2 issues fewer logged launches.
Therefore the observed S0→S2 improvement (53.28 → 19.74 ms, −63%) is **confounded**: an unknown
fraction is real prefill-graph benefit, and an unknown fraction is differential instrumentation
overhead removed. The two cannot be separated from this run. (The vLLM anchor is unaffected, so V0 is
still a valid reference, but the SGLang-internal flag comparison is not clean.)

## Downgrade rationale

Because the primary comparison axis (SGLang variant TTFT) is instrumentation-confounded and the
baseline does not reproduce Phase 2, the run is recorded as a **promising instrumented exploratory
screen**, not a clean/causal/root-cause result. The fix is an **uninstrumented confirmation** (no
KAPI, no profiler) with an `S0 → S2 → S0` bracket to check baseline stability and isolate the true S2
effect (`caseA_h1_confirmation_protocol.md`).
