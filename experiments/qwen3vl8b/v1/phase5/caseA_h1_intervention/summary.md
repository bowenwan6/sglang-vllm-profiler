# Case A — Phase 5.2 H1 Intervention — INSTRUMENTED EXPLORATORY SCREEN (not confirmatory)

> ⚠️ **Status: instrumented exploratory screen completed — NOT a clean/causal result.** All SGLang
> variants ran with `SGLANG_KERNEL_API_LOGLEVEL=1` (KAPI logging); the vLLM anchor did not. KAPI
> per-launch logging plausibly penalizes the eager/direct-launch variants (S0/S1/S3) more than the
> piecewise-graph variant (S2), so the S0→S2 improvement below is **instrumentation-confounded** and is
> likely **amplified**. The S0 baseline (53.28 ms) also does **not** reproduce Phase 2's 19.6 ms under
> identical graph config. **S2 is a promising candidate, not confirmatory. No causal / root-cause claim
> is made from this run.** The baseline anomaly is most plausibly linked to KAPI instrumentation,
> pending a clean (uninstrumented) rerun — see `baseline_anomaly_audit.md` and
> `caseA_h1_confirmation_protocol.md`. Numbers below are retained as the exploratory record.



Main experiment `qwen3vl8b` · Workload Case A (128→128, c=1) · GPU **3** · 2026-05-25.
All variants: fresh server, greedy correctness smoke passed, **0 failed requests**, GPU 3 freed
(<2000 MiB) after each. Locked protocol: dataset `caseA_short.jsonl` (sha `fab4917772e08744`),
`--num-prompts 400 --max-concurrency 1 --warmup-requests 30`, reps 3, greedy.

## Results

| Variant | Flags (+ `--disable-overlap-schedule`) | TTFT p50 median | CV | TTFT p95/p99 (med) | TPOT p50 | vs S0 | vs vLLM anchor |
|---|---|---:|---:|---|---:|---:|---:|
| **V0** vLLM anchor | vLLM default | 12.85 ms | 7.5% | — | 5.32 | — | 1.00× |
| **S0** baseline | (decode graph on, VLM piecewise prefill **off**, compile off) | **53.28 ms** | 1.0% | — | 5.56 | — | 4.15× |
| **S1** graph-off (neg. control) | `--disable-cuda-graph --disable-piecewise-cuda-graph` | 53.63 ms | 0.4% | — | **47.67** | +0.7% | 4.18× |
| **S2** enforce piecewise | `--enforce-piecewise-cuda-graph` | **19.74 ms** | 3.1% | — | 5.55 | **−63.0%** | **1.54×** |
| **S3** torch.compile | `--enable-torch-compile` | 54.05 ms | 1.3% | — | 5.15 | +1.4% | 4.21× |

(Per-rep TTFT and full metrics in `results.json` / `raw/`.)

## Reading (decision rules per protocol §6, baseline = S0)

- **S2 (`--enforce-piecewise-cuda-graph`) cut TTFT p50 by 63.0%** (53.28 → 19.74 ms), 0 errors, correct
  smoke output, CV 3.1% (acceptable), and narrowed the gap to the vLLM anchor from **4.15× → 1.54×**.
  This far exceeds the ">5% + gap-narrowing" threshold → **H1 strengthened.** The lever that helps is
  **prefill piecewise CUDA-graph coverage** — exactly the coverage the VLM auto-disable removes by default.
- **S1 (graph-off negative control):** TTFT essentially unchanged vs S0 (53.6 vs 53.3 ms), but TPOT blew
  up **~8.6×** (5.56 → 47.67 ms/token). Consistent and informative: the **decode** CUDA graph (on in S0)
  governs decode throughput, while **TTFT** is governed by the **prefill** path — which is eager in S0,
  so removing decode graph barely moves TTFT. Supports "graph coverage matters", localized to the right stage.
- **S3 (torch.compile):** no TTFT improvement (54.0 ≈ 53.3 ms). The compile path did not help Case A here
  (and is mutually exclusive with piecewise graph). So the Case-A win is specific to **piecewise CUDA
  graph**, not torch.compile.
- **TPOT parity** holds for V0/S0/S2/S3 (~5.1–5.6 ms); only S1 (graph-off) regresses it — TTFT, not
  TPOT, is the axis the intervention moves.

## H1 verdict from this run: **exploratory / inconclusive** (instrumentation-confounded)

This run does **not** provide causal evidence. The S0→S2 −63% effect is confounded by KAPI
instrumentation (see `baseline_anomaly_audit.md`), and S0 did not reproduce the Phase 2 baseline. The
genuine, uncontaminated finding is only directional: **S2 (`--enforce-piecewise-cuda-graph`) is a
promising candidate worth a clean confirmation.** H1 stays a hypothesis; it is neither strengthened nor
weakened until the uninstrumented `S0→S2→S0` bracket runs.

## Caveats (keep rigorous)

- `--enforce-piecewise-cuda-graph` is a **testing lever** that bypasses the VLM auto-disable (which
  exists for a reason). The result **validates the direction / root-cause locus**; it is **not** yet a
  production-ready fix. Correctness was only smoke-checked (8 greedy tokens), not broadly validated.
- **Single case (A).** Case C not yet run — H1 generality across c=16 batched is unconfirmed.
- **Absolute baseline differs from Phase 2** (S0 53.3 ms here vs 19.6 ms recorded in Phase 2, same
  flags). Cause not established (different GPU/build/session); **only the within-session S0→S2 contrast
  is used** for the causal claim. Notably S2 (19.74 ms) lands near the Phase-2 number — coincidental
  alignment, not asserted as equivalence.
- S1's long wall-time per rep (~2450 s vs ~300 s) is the expected eager-decode penalty (TPOT ↑), not a
  measurement fault.

## Recommended next step (for approval — not executed)

Run the **same intervention on Case C** (512→128, c=16) on GPU 3 to test whether the piecewise-graph
TTFT win generalizes to the batched path. If it does, escalate H1 confidence and (separately) evaluate
whether a non-testing route to prefill graph coverage for VLMs is warranted.
