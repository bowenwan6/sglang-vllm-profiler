# `results/` — attempts

This directory holds Qwen3.5-4B BCG DeepStack validation attempts.
Raw per-attempt evidence lives under each attempt's `raw/` subdir and
is gitignored; only summary / metadata / verdict files are committed
by default.

## Attempt index

| Attempt | GPU | Kind | Outcome | Notes |
|---|---|---|---|---|
| `infracheck_gpu7_20260801T004841Z` | 7 | INFRA_CHECK | BLOCKED (not committed) | Prior authorised attempt; blocked in `assert_pkg_version` because `sglang-kernel==0.4.4 < 0.4.5`. Raw evidence kept in place as historical record; no `metadata.json` / `verdict` files. |
| `infracheck_gpu7_20260801T012122Z` | 7 | INFRA_CHECK | PASS | See `metadata.json` / `verdict.md` / `summary.md`. Kernel is now `0.4.5`; server came up in 129 s with breakable prefill CUDA graph (58 shape captures), warmup exercised BCG (`cuda graph: True`), teardown clean, foreign PIDs unaffected. Records the spawn caveat carried into Step 5. |

## Layout convention

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
