# `scripts/` — reserved

CPU-only scaffolding lands here in the Part 5 commit
`feat(qwen35): prepare BCG DeepStack validation`. Later runners land
here too, but only under the strict rules below.

## Rules for any future runner in this directory

1. **GPU authorisation is explicit.** The runner must require a
   command-line `--gpu-id N` or an environment variable
   `QWEN35_GPU_ID=N` (never both, never a default). Absence must exit
   with a non-zero code before any CUDA import.
2. **No process-wide killers.** Never call `pkill`, `killall`,
   `fuser -k`, `nvidia-smi --gpu-reset`, or anything that signals a
   PID the runner did not itself launch.
3. **Ownership-verified PGID cleanup.** The runner must record the
   PGID of every process group it launches and, on cleanup, signal
   only that PGID after verifying every PID in it was recorded by
   this runner. Foreign PIDs in the group abort cleanup instead of
   signalling them.
4. **Foreign process abort.** If a foreign compute process is present
   on the authorised GPU at launch time (via
   `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv`),
   the runner exits non-zero (dedicated code, e.g., 71) without
   attempting to signal the foreign PID.
5. **Provenance preflight.** Before any GPU allocation, the runner
   emits and verifies the pins in `provenance.md` §6.
6. **Raw evidence is preserved but not committed by default.** Raw
   logs live under `results/<attempt-id>/raw/`; commits stage only
   `verdict.*`, `summary.md`, `metadata.json`.
7. **CPU-only preflight self-test is available.** Every runner must
   support a `--dry-run` (or equivalent) that exercises argument
   parsing, provenance emission, and cleanup wiring without calling
   into CUDA, so its skeleton is testable in this scaffolding phase.

## What is forbidden here in the CPU-only phase

- Downloading model weights.
- Initialising CUDA (`torch.cuda.init`, `torch.zeros(device='cuda')`,
  etc.).
- Any `nvidia-smi` invocation that reserves memory (query-only is
  allowed if strictly necessary for a preflight, though even that is
  deferred to the GPU phase).
