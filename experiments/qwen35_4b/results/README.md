# `results/` — reserved

This directory is reserved for future validation attempts. It is empty
during the CPU-only phase.

## Layout convention (once populated)

```
results/
  <attempt-id>/
    verdict.md         # PASS / FAIL / AMBIGUOUS / INFRA_FAILURE
    verdict.json       # machine-readable verdict
    summary.md         # attempt narrative
    metadata.json      # environment fingerprint, launch context
    raw/               # per-run server logs, bench dumps (NOT committed
                       # unless explicitly approved)
```

`<attempt-id>` follows `attempt_gpu<N>_YYYYMMDDTHHMMSSZ` for hardware
runs, or `attempt_cpu_YYYYMMDDTHHMMSSZ` for CPU-only self-checks.

## What must be committed

- `verdict.md`, `verdict.json`, `summary.md`, `metadata.json` (small,
  human-readable, review-friendly).

## What must not be committed without approval

- Any file inside `raw/` — raw server logs, bench JSONs, trace files.
- Anything larger than a few hundred kilobytes.
- Any file containing a moving `nvidia-smi` snapshot that includes
  other tenants' process names.

## What is forbidden here

- No attempt directories may be added until the user has authorised
  a GPU ID for the corresponding attempt.
