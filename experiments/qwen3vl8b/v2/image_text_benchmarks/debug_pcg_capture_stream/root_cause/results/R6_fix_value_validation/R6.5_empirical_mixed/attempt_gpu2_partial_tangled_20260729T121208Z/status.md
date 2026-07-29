# R6.5 attempt (tangled) — INFRA_INCOMPLETE

**Executed**: 2026-07-29T~12:10 UTC on GPU 2 (auto-relaunched from
earlier failed attempts).
**Machine verdict**: AMBIGUOUS (only ratio_0p8 stock recorded).

## What happened

Third GPU 2 attempt to run the R6.5 mix-ratio sweep (ratios 0.2,
0.5, 0.8). The runner's stderr / stdout was tangled with a
concurrent runner writing to the same log path in
`/tmp/…/r6.5_run/runner.log`. Individual bench runs for
stock_default at ratio_0p8 completed (mean latency 0.659 s),
but fork_pcg either never launched or was torn down before
serving.

Foreign PIDs continued landing on GPU 2 across the whole window
(three failed attempts in ~15 min). GPU 2 was free at each
pre-launch check but tenants arrived during model load in every
attempt.

## Foreign-PID discipline

Never signalled. Runner's `pre_launch_idle` correctly refused when
tenants were present; `sigquit from a child process` reported by
sglang is the scheduler's self-teardown after
`ValueError: Loaded weights leave no GPU memory for the KV cache`
(the OOM was resolved via sglang's own scheduler exit, not by us).

## Data preserved

- `verdict.md`, `verdict.json` — machine verdict AMBIGUOUS.
- `raw/` — partial server + client logs (gitignored).
