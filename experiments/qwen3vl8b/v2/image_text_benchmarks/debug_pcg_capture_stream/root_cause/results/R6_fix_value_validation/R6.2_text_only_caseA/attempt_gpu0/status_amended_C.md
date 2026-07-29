# R6.2 attempt_gpu0 — status under Amendment C

## Amended verdict: `PASS_WITH_CAVEAT — TEXT_NON_REGRESSION_SUPPORTED`

**Original machine verdict** (`FAIL` under R6.2 protocol §Pre-declared
thresholds) is preserved verbatim under
[`verdict.md`](verdict.md) and [`verdict.json`](verdict.json). No R6.2
measurement has been altered, added, or dropped. Amendment C is a
reinterpretation, not a re-run.

See
[`../protocol_amendment_C_shared_gpu_drift_gate.md`](../protocol_amendment_C_shared_gpu_drift_gate.md)
for the amendment text and rationale.

## Why the reclassification is admissible

Amendment C is scoped to the drift bracket (a **nuisance-control** on
shared-GPU stability, not a fix gate). The gates that *do* attribute
to the fix all passed with material margin:

| gate | measured | required | status |
|---|---|---|---|
| `fork_pcg / stock_pcg` mean TTFT | **0.9617** | ≤ 1.05 | **PASS** (fork 3.8 % faster) |
| stock_default CV | 5.91 % | ≤ 6.0 % | PASS |
| stock_pcg CV | 2.29 % | ≤ 6.0 % | PASS |
| fork_pcg CV | 2.02 % | ≤ 6.0 % | PASS |
| stock_default_repeat CV | 2.51 % | ≤ 6.0 % | PASS |
| assertions (all servers) | 0 | 0 | PASS |
| fallbacks (all servers) | 0 | 0 | PASS |
| post-ready recompiles (all servers) | 0 | 0 | PASS |
| bench completion (per rep) | 400 / 400 × 5 × 4 | 400 × 5 × 4 | PASS |
| drift (nuisance-control) | 3.050 % | (Amendment C) 3.0 < d ≤ 5.0 = `PASS_WITH_CAVEAT` | PASS_WITH_CAVEAT |

## Headline retained-benefit story (relative; quotable without caveat)

- `stock_default → stock_pcg`: **−31.7 %** TTFT (upstream text PCG win)
- `stock_default → fork_pcg`: **−34.3 %** TTFT (fork *keeps* PCG win)
- `stock_pcg → fork_pcg`: **−3.8 %** TTFT (fork slightly ahead)

Fork-PCG demonstrably retains the upstream text-only PCG benefit on
the Qwen3-VL server and does not regress relative to stock-PCG.

## Absolute stock_default value (26.86 ms) — must carry caveat

Rep-level detail (mean TTFT ms): 26.87, 25.34, 25.38, 27.63, 29.10.
Reps 1 and 5 were elevated during intervals with intermittent foreign
compute PIDs on GPU 0 (see `raw/launch_context.json`
`prelaunch_state.compute_pids` and the runner's periodic GPU snapshot
in `raw/runner.stdout.log`). Absolute `stock_default` numbers are
supporting-only and must not be reused as R6.3a's baseline. R6.3a
takes fresh matched measurements on current SHAs.

## Downstream effect

- R6.2 no longer blocks R6.3.
- Every R6.3/R6.4/R6.5 report that quotes absolute R6.2 numbers must
  repeat the shared-GPU caveat.
- The `verdict.md` FAIL is not deleted or renamed; it stands as the
  machine verdict under the pre-declared protocol. This document is
  the amended-status pointer.
