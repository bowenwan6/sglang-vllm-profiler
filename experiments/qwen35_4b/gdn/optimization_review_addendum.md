# Addendum to `optimization_review.md` — Stage-2 execution status

**Status.** Stage 2 started 2026-08-04. Milestones M1, M2 completed.
M3 blocked on environment change; further work paused pending user
decision.

---

## Findings that amend the Stage-1 review

### 1. Two SGLang copies exist; the reviews cited the wrong one

The design report (`optimization_design.md`) and Stage-1 review
(`optimization_review.md`) cite paths of the form
`python/sglang/kernels/ops/attention/fla/…`. Those citations were
against a scratchpad reference checkout at HEAD `58974ca16c…`
(`/tmp/claude-0/.../scratchpad/sglang_checkout/sglang`).

The SGLang **actually loaded by the profiler runners** is
`/data/sglang-fork` at HEAD `986c89e69c…`. In that tree the equivalent
files live at `python/sglang/srt/layers/attention/fla/…`.

### 2. Semantic fusion candidate is identical between the two copies

Verified directly by reading `/data/sglang-fork/…/fla/chunk.py`:

```
17:from sglang.srt.layers.attention.fla.l2norm import l2norm_fwd
108:        if use_qk_l2norm_in_kernel:
109:            q = l2norm_fwd(q)
110:            k = l2norm_fwd(k)
```

Bytewise identical F1 target: two consecutive `l2norm_fwd` calls on
independent Q and K tensors, gated by `use_qk_l2norm_in_kernel=True`.
The fusion approach applies unchanged.

### 3. Kernel signature drift — small, adaptable

The live fork's `l2norm_fwd_kernel` (at
`/data/sglang-fork/python/sglang/srt/layers/attention/fla/l2norm.py:54-71`):

* `@triton.jit` (no `do_not_specialize`)
* Adds `NB: tl.constexpr` (unused in the kernel body — appears to be a
  future-hook for a per-NB branch)
* `T: tl.constexpr` (so kernel recompiles per T; under CUDA graph
  capture T is fixed per bucket, so recompilation is one-time)

The prototype (`scratchpad/f1_prototype/l2norm_fwd_pair.py`) was
adapted to match the live fork's signature. Bit-exact equivalence to
the split-kernel path is preserved.

### 4. Empirical hot-path confirmed against live fork traces

Re-ran `nsys stats --report cuda_gpu_kern_sum` on existing Stage-3
captures (which were taken against the live fork, HEAD
`986c89e69c…`):

| capture | l2norm_fwd_kernel count | total time | avg per launch |
|---|---|---|---|
| A0_p128_rep1 | 576 | 635 μs | 1.10 μs |
| A1_p128_rep1 | 8,928 | 35.4 ms | 3.97 μs |
| Δ | +8,352 | +34.8 ms/trace | — |

Fusion would eliminate 4,176 launches × 3.97 μs = **16.6 ms per trace**
= **1.66 ms per prefill** (10 prefills/trace). This is the top end of
the Stage-1 estimate — the benefit is real.

---

## What is blocked

**M3 (parity test) requires a working CUDA runtime in this Python
process.** The NVIDIA kernel driver was upgraded from a
CUDA-13.0-compatible version to 595.71.05 at 2026-08-04 12:53 UTC.
Torch 2.11.0+cu130 now fails `cuInit(0)` with `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH`
(error 803). Verified via ctypes on `libcuda.so.1`.

The Stage-3 GPU captures were taken on 2026-08-03 before the driver
upgrade; those pre-upgrade CUDA runtimes were compatible. Post-upgrade,
no Python process on this host can init CUDA under the current
torch/CUDA build.

External change; not resolvable by a small reversible experiment.
Requires one of:

1. **Driver rollback** to the CUDA-13.0-compatible range
   (system-level — affects other users).
2. **Torch upgrade** to a build linking a CUDA runtime that supports
   driver 595 (e.g., cu131 or nightly).
3. **Alternate host** with a compatible driver.
4. **Wait for repair** if the driver upgrade was inadvertent.

---

## What has been produced

Locally, all inside the scratchpad (nothing committed to the frozen
tree, nothing touching `/data/sglang-fork`):

* `scratchpad/f1_prototype/l2norm_fwd_pair.py` — the fused-pair kernel
  and launcher, Option-b signature, live-fork-compatible.
* `scratchpad/f1_prototype/test_parity.py` — bit-exact parity test
  against `l2norm_fwd`; ready to run when CUDA returns.

## Reversibility

* No frozen source modified.
* No GPU experiments run (only re-read existing captures).
* No git worktree created.
* All work is in the scratchpad, disposable.
* All prior evidence preserved.

## Next actions once environment is restored

1. Run `python3 test_parity.py` → expect element-wise equality on all
   shapes; abort if any mismatch.
2. Write and run a small microbenchmark comparing fused-pair latency
   vs (single + single) on H100 at production shapes.
3. If parity + microbench pass, create a git worktree of
   `/data/sglang-fork` off a fresh `f1-l2norm-fusion` branch and apply
   F1 to `srt/layers/attention/fla/l2norm.py` + `srt/layers/attention/fla/chunk.py`.
4. Continue with M5–M8 per plan.

---

## Signal to user

**SIGNAL: EXEC_UNCLEAR** — blocked by system-level change (driver
upgrade). Please advise: retry-later, rebuild-torch, alternate-host,
or abandon Stage-2 for now.
