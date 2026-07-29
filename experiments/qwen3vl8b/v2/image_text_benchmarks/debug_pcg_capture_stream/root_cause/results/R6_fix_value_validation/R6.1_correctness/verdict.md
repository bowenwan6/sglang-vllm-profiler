# R6.1 verdict — **INFRA_FAILURE / R7_REQUIRED**

> Verdict rules were pre-declared in [`protocol.md`](protocol.md) BEFORE
> any leg was run. This run did not reach any leg — the first server
> (stock-default) died before HTTP readiness on an environment error
> unrelated to the R6.1 correctness protocol itself. Recorded honestly
> without retry, per user directive: "If a failure occurs, document it
> and stop; do not redesign and rerun the protocol without approval."

## GPU selection (from `raw/monitor_selection.json`)

- **Selected GPU ID:** 1
- Idle streak start (UTC): `2026-07-28T10:35:35+00:00`
- Qualified (UTC): `2026-07-28T10:46:05+00:00`
- Idle hold requirement: 600 s (mem ≤ 500 MiB, util ≤ 5 %, 0 compute PIDs, polled every 30 s)
- Final pre-launch check (UTC): `2026-07-28T10:46:09+00:00`
  → `{compute_pids: [], idle: True, mem_mib: 0, ok: True, util_pct: 0}`
- Monitor started polling: `2026-07-28T07:23:20+00:00`
- Total wait for a GPU to satisfy the 600 s continuous-idle rule:
  approximately 3 h 22 min (all 8 GPUs were occupied by unrelated
  tenants until 10:35:35, at which point GPUs 0/1/2/3 released
  simultaneously; GPU 0's streak was reset once by transient
  activity, so GPU 1 qualified first per the deterministic
  lowest-ID-first rule).

## Root cause

**NVIDIA driver silently upgraded during the monitor wait window.**

- R6.0 provenance (2026-07-28, morning): driver `570.172.08`.
- At R6.1b runner launch (2026-07-28, 10:46:09 UTC): driver
  `595.71.05`.
- Torch 2.11.0+cu130 in this container was built against the R6.0
  driver ABI; on the new driver it emits
  `Error 803: system has unsupported display driver / cuda driver
  combination` from `cudaGetDeviceCount()`, then sglang's
  `ServerArgs.__post_init__` → `get_device()` raises
  `RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS)
  or platform plugin is available.` and the launched python
  process exits before opening the HTTP port.

Server-log excerpt (from
`raw/stock-default_server.log`, not committed):

```
CUDA initialization: Unexpected error from cudaGetDeviceCount().
Error 803: system has unsupported display driver / cuda driver
combination.
...
RuntimeError: No accelerator (CUDA, XPU, HPU, NPU, MUSA, MPS) or
platform plugin is available.
```

## Attribution

- **Not an R6.1 correctness signal.** No leg (a, b, c, d, d', e, or
  f) executed. The R6.1 verdict rules in
  [`protocol.md`](protocol.md) §7 are not evaluable from this run.
- **Not a runner logic bug.** The runner detected the server death
  via `kill -0 "$SRV_PID"` inside the readiness wait loop and
  returned exit code 2 as designed. No fallback runner action was
  taken; no PID was signalled that wasn't ours.
- **Not the fix's fault.** The failure occurs at CUDA init, which
  is before any sglang code that our fix touches runs. It reproduces
  identically on stock SGLang.

## Compliance with safety invariants (verified after runner exit)

- Runner exit code: `2` (server died during startup readiness wait).
- `pkill` / `killall` / `fuser -k` / `nvidia-smi --gpu-reset`
  invocations: **none** (checked runner + monitor sources; only
  documentation strings).
- Foreign processes signalled: **none**. The dead server PID belonged
  to our own launched python (PGID == PID == 111050) and needed no
  signal.
- Tracked PGIDs remaining live at exit: **none** (server died on its
  own before the trap fired; trap's ownership check found no live
  process to signal).
- GPU 1 post-failure state:
  `memory.used=0 MiB, memory.free=143158 MiB, utilization.gpu=0 %,
  compute_apps=[]`. No collateral damage.
- Foreign compute PIDs on GPU 1 during our run: **none observed**
  (no `foreign_pid_detected.txt` written).

## What was NOT done

- Retry of R6.1b was **not** attempted. Per user directive.
- R6.0 provenance was **not** amended in this commit. Amending it
  requires an explicit `docs(v2): update R6 provenance` commit and
  should follow a decision on how to resolve the driver drift
  (options include: rebuild torch against the new driver, roll the
  container back, or accept the new driver and update the frozen
  tuple). Any of those paths must be authorized separately.
- No process outside our own launch was signalled.
- `raw/monitor.lock`, `raw/monitor.jsonl`, `raw/monitor.log`,
  `raw/stock-default_server.log`, `raw/stock-default_server.pid`
  remain locally under `raw/` for post-mortem inspection but are
  **not committed** (all gitignored).
- The R5.C uncommitted `audit_report.md` was not touched.

## Verdict category

**R7_REQUIRED (INFRA_FAILURE — provenance drift).**

Per plan.md §5b, R7_REQUIRED is the correct category when
"correctness, stability, non-regression, or evidence quality fails."
Evidence quality here has failed on the environment axis:
correctness cannot be evaluated because the servers cannot even
initialize CUDA under the current driver.

## Required unblocking work (before any R6.1b retry)

1. **Decide how to resolve the driver / torch ABI mismatch** — either
   rebuild / reinstall the torch + sgl_kernel + flashinfer stack
   against driver 595.71.05, or roll the driver back to 570.172.08,
   or replace the container.
2. **Amend R6.0** — `docs(v2): update R6 provenance` recording the
   final driver / stack tuple that R6.1b will actually run under,
   plus any resulting SHA / version changes.
3. **Re-run R6.1b** — same protocol, same fixture, same prompts,
   same verdict rules. Do not modify any of them.

Nothing else in the R6 protocol changes; R6.1a preparation
artifacts, the monitor, and the runner are all reusable as-is
once the environment is fixed.
