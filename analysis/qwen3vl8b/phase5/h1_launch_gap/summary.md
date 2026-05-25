# Phase 5.1 — H1 Observational Validation (offline launch / graph-coverage metrics)

Read-only analysis of existing traces (main experiment `qwen3vl8b`). No GPU, no server, no
re-collection, no source changes. Script: `experiments/qwen3vl8b/phase5/scripts/h1_launch_gap.py`.
SGLang rows come **only** from graph-on **formal** traces (never graph-off mapping). Per-case tables:
`caseA_short.md`, `caseC_batched.md`; raw numbers: `metrics.json`.

## Confirmed mechanism (config + source, not causality)

In the Case A / Case C graph-on **formal** `server_args.json`:
`disable_cuda_graph=False`, **`disable_piecewise_cuda_graph=True`**, `enable_torch_compile=False`.
SGLang source `python/sglang/srt/server_args.py` `_handle_piecewise_cuda_graph`: condition #8 (line
1308–1310) auto-disables piecewise CUDA graph for **multimodal/VLM** models; `enforce_piecewise_cuda_graph`
(line 1276–1278, "Skip auto-disable … for testing") overrides it. Qwen3-VL-8B is multimodal → its
**prefill/mixed path runs without piecewise-graph and without torch.compile by default**; decode graph
is nominally enabled.

**Mechanism confirmed; H1 (causality) remains to be validated.** This is a configuration fact, not
proof that the missing prefill coverage explains the TTFT residual gap.

## Metric table (key columns; full tables per case)

| window | kernels | graph % | eager % | GEMM % | GPU idle % | launch-CPU µs |
|---|---:|---:|---:|---:|---:|---:|
| **A** SGLang formal EXTEND | 5870 | 0.0 | 99.4 | 83.5 | 96.2 | 44157 |
| **A** SGLang formal DECODE | 580 | 0.0 | 99.7 | 74.6 | 95.3 | 4702 |
| **A** vLLM prefill_like | 4328 | 81.7 | 18.0 | 77.3 | 65.0 | 11848 |
| **A** vLLM decode_like | 640230 | 93.1 | 6.5 | 81.1 | 7.8 | 648496 |
| **C** SGLang formal EXTEND | 8732 | 35.7 | 62.1 | 75.0 | 94.1 | 46557 |
| **C** SGLang formal DECODE | 1216 | 0.0 | 99.9 | 85.6 | 40.6 | 8442 |
| **C** vLLM prefill_like | 4320 | 0.0 | 99.6 | 83.3 | 68.6 | 19008 |
| **C** vLLM decode_like | 485351 | 72.7 | 26.7 | 77.9 | 6.3 | 541644 |

## Answers to the Phase-5.1 questions

**1. Can existing traces quantitatively compare SGLang vs vLLM launch / graph coverage?**
**Only partially, and NOT reliably for the coverage claim.** A decisive confound surfaced: the SGLang
**formal DECODE** trace contains **zero `cudaGraphLaunch` ops** (only `cudaLaunchKernelExC` /
`cuLaunchKernelEx`) even though `disable_cuda_graph=False`. The SGLang profiler (`--profile-by-stage`)
captures forward steps in an **eager / non-graph-replayed mode**, so graph-coverage % measured on SGLang
traces reflects the *capture mode*, not the serving path. vLLM traces, by contrast, do preserve
`cudaGraphLaunch` (decode 93%/73%; prefill A 82%). So the SGLang-vs-vLLM graph-% comparison is
**confounded by a capture-mode asymmetry** and cannot be taken as a serving-path coverage comparison.
What *is* robust from the traces: **GEMM share is 75–86% in both frameworks, both stages** → confirms
GEMM is shared absolute cost, not the cross-framework differentiator (H2 context, not H1).

**2. Do Case A/C show larger direct-dispatch / CPU-gap / GPU-idle signal in SGLang formal?**
**Directionally yes, but not conclusively.** SGLang formal EXTEND shows very high GPU idle (A 96.2%,
C 94.1%) and higher summed launch-CPU than vLLM prefill (A 44k vs 12k µs); vLLM prefill idle is lower
(65–69%). This is *consistent* with a dispatch-bound prefill. **But** three contaminations prevent a
causal read: (a) the EXTEND traces were captured under a synthetic prefill-only `max_new_tokens=1`
load, whose inter-request cadence inflates GPU-idle%; (b) the eager-capture mode above; (c) the
classifier detects `cudaGraphLaunch` but **not** torch.compile/inductor coverage, so vLLM's
inductor-compiled-but-not-graphed kernels count as "eager" (e.g. C vLLM prefill 99.6% "eager" is
largely inductor-compiled per Phase 4) — i.e. "eager%" ≠ "uncovered".

**3. Does the evidence strengthen, weaken, or remain inconclusive for H1?**
- **Mechanism: strengthened / confirmed** at the configuration level (VLM auto-disables piecewise graph;
  compile off) — independent of the trace confounds.
- **Causality: inconclusive from offline traces.** The capture-mode confound + load contamination +
  graph-vs-compile blindness mean the traces cannot prove the missing prefill coverage *causes* the
  TTFT residual gap. **H1 stays Medium-confidence, to-be-validated.**

**4. Which metrics could NOT be reliably computed (trace limits):**
- **Serving-path CUDA-graph coverage for SGLang** — confounded by eager-capture mode (DECODE shows 0%
  graph despite graph enabled). Reported as **unreliable**, not used to support H1.
- **torch.compile / inductor coverage** — not detectable via launch-op correlation (no `cudaGraphLaunch`
  marker); the classifier cannot separate inductor-compiled from truly-eager. **unavailable.**
- **True per-forward-step critical-path CPU launch-gap** — multi-thread + multi-step (`num_steps>1`)
  windows are not cleanly separable from the trace alone. **ambiguous**; only a summed launch-op CPU
  proxy is reported.
- **Latency-matched comparison to benchmark TTFT** — the profiled windows are not the benchmark timing
  windows; idle% is load-cadence-contaminated. **not a TTFT proxy.**

**5. Recommendation on Phase 5.2 (Case A controlled intervention):**
**Recommended — and now clearly necessary.** Because the offline route is confounded (eager-capture +
load cadence + compile-blindness), the **only** way to establish H1 causality is the controlled
intervention: measure end-to-end benchmark TTFT under baseline vs coverage-expansion, on the real
serving path (graph replay active, real benchmark load). The offline analysis has done its job — it
**confirmed the mechanism** (default config leaves the VLM prefill path uncovered) and **ruled out** a
naive trace-only proof — but it cannot substitute for the intervention. *(Recommendation only — Phase
5.2 not executed here.)*

## Suggested Phase 5.2 (for approval — NOT executed)

GPU **`CUDA_VISIBLE_DEVICES=3`** only (not 0/1/7); serial servers; GPU 3 < 2000 MiB between launches;
new traces → `traces/qwen3vl8b/phase5/`; stop on crash/OOM/empty-trace/failed-requests>0/unfreed-GPU/
need-for-source-change. **Case A first, then Case C.** Variants (candidate 1 and 2 are **separate
alternatives**, not combined):

| Variant | Flags (on Phase-2 case config) | Purpose |
|---|---|---|
| baseline | Phase-2 locked (A: `--disable-overlap-schedule`; C: default) — decode-graph ON, piecewise OFF, compile OFF | real serving baseline |
| negative control | `--disable-cuda-graph --disable-piecewise-cuda-graph` | fully eager floor |
| coverage candidate 1 | `--enforce-piecewise-cuda-graph` (testing lever; stop on failure/OOM/incorrect output) | force prefill piecewise-graph coverage |
| coverage candidate 2 (alt) | `--enable-torch-compile` (≤ bs 32) | alternative coverage path |

Measure per variant: TTFT p50/p95/p99, TPOT, CV, error rate, + re-run this script on a fresh Phase-5
trace of that variant for graph-share / idle. Pass/fail per `experiments/qwen3vl8b/phase5/plan.md` §3.
