# logs/

Raw logs — never edited by hand. One subdirectory per phase.

- `phase1/sglang_%i.log` — kernel-API boundary trail (`SGLANG_KERNEL_API_LOGLEVEL=1`) from Phase 1 baseline runs.
- `phase2/sglang_%i.log` — same, for shaping sweeps.
- `phase3/sglang_%i.log` — same, for mapping + formal trace collection.
- `phase5/sglang_%i.log` — same, for validation sweeps.

If a run crashes, escalate the specific case only (L3 for shape inspection, L5 for NaN/Inf, L10 + `DUMP_DIR` for offline reproducer). Do **not** run at L≥3 during a measured benchmark or profile — it perturbs timing.
