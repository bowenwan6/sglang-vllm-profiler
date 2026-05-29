# v2 / Round 2 — Case A/C rebaseline (issue #2)

Production-default **overlap-ON** rebaseline for Qwen3-VL. Demotes v1's
`--disable-overlap-schedule` headline to an ablation and re-tests whether the PCG lever still
helps against the production baseline.

- **Issue:** #2 (parent #1) on `bowenwan6/sglang-vllm-profiler`.
- **Status:** scaffold only — protocol not yet locked, no runs executed.
- **Protocol:** see `protocol.md` (placeholder).
- **Model:** `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd` (same as v1; verify in env snapshot before runs).

This directory is intentionally empty of results until the protocol is approved and the clean
(no-KAPI, no-profiler) runs are executed.
