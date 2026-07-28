# R6.1 Protocol Amendment B — repeated-shape safety control (2026-07-28)

> **This amendment supersedes** only the negative-control definition in
> Amendment A §2.3 (which relied on a 3-prompt fixture that produces
> distinct prefill shapes and therefore does not exercise the
> historical trigger). All other parts of Amendment A, and all parts
> of the original `protocol.md`, remain in force.
>
> Historical attempts 01–03 stand as recorded. Attempt 04 runs under
> this amendment. Any further change to §B.1 – §B.4 below requires
> a new amendment file (`protocol_amendment_C_*.md`).

## Scope and objective

Attempt 03 exposed a design gap in Amendment A's negative control:
the 3 different image prompts in `fixtures/prompts.json` produce 3
distinct prefill runtime shapes, so the historical bug's *second-
same-shape-after-multimodal-recompile* trigger was never reached.
The neg control classified as `STOCK_NOW_SURVIVES`, which was
misread as evidence that upstream fixed the bug — a claim withdrawn
in the [attempt 03 interpretation addendum](attempt_03_amended_A_gpu0/interpretation_addendum.md).

Amendment B replaces the negative control with an exact reproduction
of the historical E2a / R1 / R2 sustained-workload recipe, which is
already known to reproduce the assertion on the current stock SHA
`da802ddca`. Only the sustained-workload trigger is a valid safety-
superiority test.

## §B.1 Frozen recipe (recovered from `scripts/run_R1_dynamo_recompile.sh`)

Server (both variants share these flags):

```
python3 -m sglang.launch_server \
  --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
  --dtype bfloat16 \
  --port 30003 \
  --tp 1 \
  --attention-backend flashinfer \
  --enforce-piecewise-cuda-graph
```

Env (both variants):

```
CUDA_VISIBLE_DEVICES=0
SGLANG_USE_CUDA_IPC_TRANSPORT=1
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
TORCH_LOGS=recompiles_verbose,dynamic,guards,graph_breaks
TORCHDYNAMO_VERBOSE=1
```

`SGLANG_KERNEL_API_LOGLEVEL` and `SGLANG_KERNEL_API_LOGDEST` must
be unset (KAPI must not run).

Fork variant additionally prepends `PYTHONPATH=/data/sglang-fork/python`.

Bench client (identical for both variants):

```
python3 -m sglang.benchmark.serving \
  --backend sglang-oai-chat \
  --base-url http://127.0.0.1:30003 \
  --model <SNAP> \
  --dataset-name image \
  --image-count 1 \
  --image-resolution 720p \
  --image-format png \
  --image-content random \
  --random-input-len 128 \
  --random-output-len 128 \
  --random-range-ratio 1.0 \
  --max-concurrency 1 \
  --num-prompts 32 \
  --warmup-requests 30 \
  --seed 1 \
  --extra-request-body '{"temperature": 0, "top_p": 1}' \
  --output-details \
  --output-file <BENCH_OUT>
```

The runner records the same env + command verbatim in
`raw/<variant>_recipe.txt` for provenance.

Note on shape stability: `--image-resolution 720p` fixes the vision-
token count; `--random-input-len 128 --random-range-ratio 1.0` pins
text tokens to exactly 128. Even with `--image-content random`, the
total prefill token count is identical across all 32 requests
(only the pixel values change, not the tensor shapes). This is the
exact condition that produced the historical R1 crash.

## §B.2 Predeclared safety verdict

**`SAFETY_SUPERIORITY_PASS`** requires **BOTH**:

1. **stock-PCG reproduces the exact historical assertion.** The
   stock-PCG server's log must contain a line matching
   `AssertionError: PCG capture stream is not set, please check if
   runtime recompilation happened`, AND the preceding Dynamo
   recompile cascade must include a frame whose fail reason contains
   `input_deepstack_embeds is None` (matching R1 [0/3] / [0/4]).
2. **fork-PCG completes the identical workload.** The fork-PCG server
   must:
   - complete all warmup (30 requests) and all measured (32 requests)
     without crashing;
   - `AssertionError: PCG capture stream is not set` count == 0 in
     the fork-PCG server log;
   - `Falling back to eager execution` count == 0;
   - client-side request failures (non-200 / exceptions) == 0;
   - post-server-ready recompiles inside a measured-leg interval
     == 0 (warmup / startup recompiles are reported and do not
     fail the safety gate — see Amendment A §2.1).

**`STOCK_TRIGGER_NOT_REPRODUCED`** (`AMBIGUOUS`) — stock-PCG completes
the exact historical workload without the assertion. This does **not**
prove an upstream fix; it means the environment or trigger is different
from the historical R1/R2 conditions. Record all runtime shapes from
the stock server log; investigate environment differences (torch,
flashinfer, sgl_kernel, driver, sample seed order, radix cache) before
drawing any conclusion. R6.2 remains blocked.

**`FORK_FAIL`** — fork crashes, falls back, has post-ready inflight
recompile events, or has any request failure. R6.2 blocked; the fix
is regressed.

**`INFRA_FAILURE`** — either server fails for an unrelated CUDA / driver /
model-loading / harness reason (e.g. Error 803, port conflict,
disk-full, OOM at startup, foreign process on GPU 0). Not a product
result; environment fix required.

**Overall R6.1 gate**: Attempt 03's Tier 2 CORRECTNESS PASS may be
combined with Attempt 04's `SAFETY_SUPERIORITY_PASS` to declare
**R6.1 = PASS** only if Attempt 04 achieves `SAFETY_SUPERIORITY_PASS`.
Otherwise R6.1 remains blocked (`AMBIGUOUS` if stock trigger not
reproduced; `FAIL` if fork fails; `INFRA` if infra).

## §B.3 Serial execution + operational rules (unchanged from Amendment A2)

- Serial: stock-PCG first (its likely crash + PGID-scoped cleanup),
  wait for GPU memory to drain, then fork-PCG. Both on GPU 0.
- Servers launched via `scripts/R6_setsid_exec.py` so PID == PGID.
- Teardown only signals PGIDs the runner recorded, after
  re-verifying `kill -0`, `ps -o pgid`, and `comm =~ ^python`.
- **Prohibited**: `pkill`, `killall`, `fuser -k`, kill-by-name,
  kill-by-port without verified PID ownership, `nvidia-smi --gpu-
  reset`. **Never** signal foreign processes.
- Foreign compute PID appearing on GPU 0 during our run → runner
  aborts current variant, tears down our own server, exits 71.
- Between servers, GPU memory drain wait (best-effort read-only) up
  to 60 s; if not drained, log and continue (do not signal).
- Pre-launch idle check on GPU 0 is `mem ≤ 500 MiB, util ≤ 5%,
  0 compute PIDs`. Fail → abort variant, do not switch GPUs.
- Host libcuda pin per R6.0 Amendment A3.

## §B.4 Reporting

Attempt 04 writes:

- `raw/{stock,fork}_server.log` — full sglang server stdout/stderr.
  Gitignored (large).
- `raw/{stock,fork}_bench.log` — bench client stdout/stderr.
  Gitignored.
- `raw/{stock,fork}_bench.jsonl` — bench client per-request details.
  Gitignored.
- `raw/{stock,fork}_recipe.txt` — verbatim env + command. Gitignored
  under `raw/` .gitignore, but a canonical copy of the recipe is in
  the runner script (`scripts/run_R6_1_repeated_shape.sh`) which
  IS committed.
- `raw/{stock,fork}_phase_markers.txt` — SERVER_READY / BENCH_START /
  BENCH_END markers. Gitignored.
- `raw/attempt_04_shape_trace.json` — extracted per-request prefill
  shapes with first-occurrence / repeated-occurrence flags (for
  stock only, per §B.2's stock-log parsing). Gitignored.
- `raw/attempt_04_verdict.json` — machine-readable verdict.
  Committed only if it fits the raw/.gitignore committed-verdicts
  exception, or explicitly moved to the attempt directory root as
  `verdict_amended_B.json`.
- `verdict_amended_B.md` + `verdict_amended_B.json` — committed
  verdict artefacts at the attempt-04 directory root (not under
  `raw/`).

Human-readable `verdict_amended_B.md` records:

- stock and fork SHAs at run time
- host libcuda path + driver version
- GPU 0 pre/post state per variant
- stock: recipe used, request counts (warmup / measured / actual
  completed), whether the assertion was reached, if so which
  request index and which prefill shape, the recompile cascade
  seen ([0/1] – [0/n])
- fork: same set of counts, safety counters (assertion, fallback,
  request failures, per-leg post-ready recompiles), warmup vs
  inference recompile split
- pre-declared Amendment B verdict category
- overall R6.1 gate decision including Attempt 03 correctness
  combination.
