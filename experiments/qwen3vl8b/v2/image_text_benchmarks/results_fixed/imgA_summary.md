# IMG-A Benchmark Summary — image+text (c=1) — fixed-generator path

> Run: 2026-06-08 16:20 UTC  GPU=7  seed=1  n=400  warmup=30  reps=5  resolution=720p  range_ratio=1.0

> SGLang image headline baseline: `SGLANG_USE_CUDA_IPC_TRANSPORT=1` (IPC on).

> IPC benefit and PCG benefit reported separately.

> vLLM is anchor only — no causal inference.

> **Image+text conclusions are separate from text-only Issue #2 findings.**

## Fixed-SGLang provenance

- `/data/sglang-pr` HEAD SHA: `62c505a196fd5bc997f478b0a7c6403ce655a838`
- `/data/sglang-pr` branch: `main`
- Merged fix `07f326c184` in history: **True**
- `sglang.__file__`: `/data/sglang-pr/python/sglang/__init__.py`
- `sglang.benchmark.datasets.common.__file__`: `/data/sglang-pr/python/sglang/benchmark/datasets/common.py`
- Fix marker (`get_available_multimodal_text_tokens` in `gen_mm_prompt`): `FIX_OK`
- Fixed-path import gate: **True**

## Headline numbers (TTFT p50, median of reps)

| variant | ipc | pcg | ttft_p50 median (ms) | CV% | tpot_p50 (ms) | out_tok/s | status |
|---|---|---|---|---|---|---|---|
| IMG_A_S0_ipc | on | off | 64.817 | 13.3% | 5.233 | 175.101 | OK |
| IMG_A_S2_ipc_pcg | on | on | FAIL | ?% | ? | ? | INVALID_FAILURES |

## Bracket drift (S0_ipc vs S0_ipc_repeat)

⚠ One or both bracket variants failed — drift cannot be assessed.

## PCG benefit (Q2): S0_ipc vs S2_ipc_pcg

⚠ S0_ipc or S2_ipc_pcg failed — PCG benefit cannot be assessed.

## IPC benefit (Q3): S0_noipc vs S0_ipc

⚠ S0_ipc or S0_noipc failed — IPC benefit cannot be assessed.

## SGLang IPC baseline vs vLLM anchor (Q1)

⚠ S0_ipc or V0_vllm failed.

## Token composition (per request)

Vision tokens: 882/req  |  Text tokens: 142/req  |  Resolution: 720p, image-count=1, range_ratio=1.0, seed=1

## Failure summary

❌ 1 variant with issues; remaining 3 variants skipped per protocol §9 stop condition (`S2_ipc_pcg correctness check fails`):
  - **IMG_A_S2_ipc_pcg**: status=`INVALID_FAILURES` — server-side **AssertionError** in PCG backend on first prefill (rep1 failed in <2 min after server-up)
  - IMG_A_S0_ipc_repeat: NOT RUN
  - IMG_A_V0_vllm: NOT RUN
  - IMG_A_S0_noipc: NOT RUN

### Root cause — upstream SGLang PCG bug (not our generator fix)

Server traceback (`logs/.../results_fixed/IMG_A_S2_ipc_pcg_server.log`):

```
File "/data/sglang-pr/python/sglang/srt/compilation/cuda_piecewise_backend.py", line 171, in __call__
    stream is not None
AssertionError: PCG capture stream is not set, please check if runtime recompilation happened
```

The assertion fires inside `cuda_piecewise_backend.__call__` for a compiled
graph-module call of the Qwen3-VL forward path (`qwen3_vl.py:1277 →
mm_utils.general_mm_embed_routine → language_model.forward → torch.compile
→ <eval_with_key>.669`). On the upstream `main` HEAD `62c505a196`, this
assertion appears in the PCG hot path when `--enforce-piecewise-cuda-graph`
is enabled together with `SGLANG_USE_CUDA_IPC_TRANSPORT=1` for the Qwen3-VL
image+text workload. Prior #2 text-only Case A on the **older** SGLang at
`/sgl-workspace/sglang` (commit `0c8049d9b`) did not exhibit this — so this is
upstream drift in the PCG implementation, not a regression introduced by the
generator fix at `07f326c184`. Our fix gate stayed green throughout (FIX_OK,
sglang.__file__ under `/data/sglang-pr/python`).

This is the second non-generator dependency surprise of this session, after the
flashinfer 0.6.11→0.6.12 and sglang-kernel 0.4.2→0.4.3 upgrades. Both are part
of "rebasing #4 to a newer upstream main than #2 ran against."

### What is still usable from this partial run

- **`IMG_A_S0_ipc`** completed cleanly: 5/5 reps, 0 failures, 5×400=2000 requests
  served, no forbidden-token errors. p50 TTFT 87.2 / 65.1 / 63.7 / 63.6 / 64.8 ms
  across reps (warmup → settled tail within ~1ms). Median = **64.8 ms**, CV 13.3%
  driven by rep1 warmup. TPOT flat at 5.23 ms across all reps. Throughput
  175 tok/s. This is the **only** clean datapoint and is **not yet a headline**
  number — it has no bracket counterpart (`S0_ipc_repeat`) to bound drift.

Cannot be reported as #4 headline yet:
- PCG benefit (Q2): S2_ipc_pcg failed → undetermined.
- IPC benefit (Q3): S0_noipc not run → undetermined.
- vLLM anchor (Q1): V0_vllm not run → undetermined.
- Bracket drift: S0_ipc_repeat not run → undetermined.

## Recommendation

Decide between (a) **rerun the non-PCG variants** on the fixed-generator path
(skip S2_ipc_pcg, keep S0_ipc_repeat / V0_vllm / S0_noipc → recovers bracket
drift + IPC benefit + vLLM anchor; PCG benefit remains undetermined pending
upstream PCG fix), or (b) **file an upstream SGLang PCG issue first** with the
`cuda_piecewise_backend.py:171` traceback and pause #4 until that is resolved,
or (c) **temporarily switch S2_ipc_pcg to the older `/sgl-workspace/sglang`
just for the PCG variant** (loses provenance uniformity but recovers all four
non-fix-related comparisons). Do **not** proceed to IMG-B / IMG-C until IMG-A
yields headline-quality data.