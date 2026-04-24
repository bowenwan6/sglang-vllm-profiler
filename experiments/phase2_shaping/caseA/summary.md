# Phase 2 — Case A Scheduler-Overhead Sweep

Generated: 2026-04-24T11:45:50.396698+00:00 UTC

Phase-1 reference: SGLang TTFT p50 = 54.6 ms | vLLM = 14.1 ms

## Screening results (1 rep each)

| Candidate | Flags | TTFT p50 (ms) | Δ vs baseline | Δ vs vLLM |
|---|---|---|---|---|
| A0_baseline | `(default)` | 57.1 | +0.0 ms | +43.0 ms |
| A1_no_overlap | `--disable-overlap-schedule` | 55.4 | -1.7 ms | +41.3 ms |
| A2_fcfs | `--schedule-policy fcfs` | 57.5 | +0.4 ms | +43.4 ms |
| A3_stream8 | `--stream-interval 8` | 57.0 | -0.0 ms | +42.9 ms |

## Finalist reconfirm (3 reps)

| Candidate | TTFT p50 median (ms) | CV% | Verdict |
|---|---|---|---|
| A0_baseline | 56.0 | 0.1% | STRUCTURAL — floor unchanged |

## Decision

**STRUCTURAL — Case A promotes to Phase 3.**

No scheduler flag moved TTFT by ≥10 ms. Maximum single-flag delta: −1.7 ms (A1, `--disable-overlap-schedule`). The ~56 ms TTFT floor is intrinsic to SGLang's c=1 request-dispatch path and cannot be closed by configuration.

- Finalist config: A0 baseline (default flags, no shaping needed)
- Residual TTFT gap vs vLLM: **+43 ms** (56.0 ms vs 14.1 ms, ~4× ratio)
- Residual TTFT CV: **0.1%** — low-noise, profilable
- Phase-3 phenomenon label: "SGLang ~56 ms structural scheduler/dispatch floor at c=1, unresponsive to overlap/policy/stream flags"
- Fairness dependence: Framework-intrinsic (scheduler architecture difference)
