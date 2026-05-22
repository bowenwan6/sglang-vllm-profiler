# run2 Phase 3 — SGLang EXTEND/PREFILL Trace Supplement

Generated: 2026-05-22 · GPU 1 · SGLang only · collection only (no Phase 4 interpretation).

## Why this supplement

The original Phase 3 SGLang traces were **DECODE-stage only**: the profiler armed during steady
in-flight decode, so `--profile-by-stage` emitted only DECODE files. This supplement drives a
**continuous prefill-only load** (`/generate` with `max_new_tokens=1` at the case concurrency, prompts
from the case dataset) so the profile window lands on EXTEND (prefill) forward steps. New traces were
written to `sglang_extend_{mapping,formal}/`; **existing DECODE traces in `sglang_{mapping,formal}/`
were not touched** and remain valid for decode-stage analysis.

Mechanism per trace: launch SGLang (mapping = `--disable-cuda-graph --disable-piecewise-cuda-graph`;
formal = without) → start continuous `max_new_tokens=1` load → `sglang.profiler --profile-by-stage
--num-steps 10`. `SGLANG_KERNEL_API_LOGLEVEL=1` on every launch. GPU 1 freed after every server.

## Result — EXTEND capture status (7 of 8 trace groups)

| Case | extend_mapping (graph-off) | extend_formal (graph-on) |
|---|---|---|
| A `caseA_short` (c=1, `--disable-overlap-schedule`) | ✅ EXTEND · 35 MB | ✅ EXTEND · 35 MB |
| B `caseB_longprefill` (c=1, default) | ✅ EXTEND · 64 MB | ❌ not captured (see below) |
| C `caseC_batched` (c=16, default) | ✅ EXTEND · 47 MB | ✅ EXTEND · 31 MB |
| D `caseD_decode` (c=16, default) | ✅ EXTEND · 51 MB | ✅ EXTEND · 34 MB |

Paths: `traces/run2_qwen3vl8b/{case}/sglang_extend_mapping/<ts>/*-TP-0-EXTEND.trace.json.gz` (+ `server_args.json`),
and the same under `sglang_extend_formal/` (except Case B formal).

## Case B graph-on EXTEND — failed attempt (documented deviation)

Case B `extend_formal` (graph-on, 2048→prefill, c=1) returned **DECODE-only on all 3 attempts**
(1 initial + 2 retries). With CUDA graph on and a single large 2048-token prefill at c=1, the 10-step
profile window consistently lands on fast graph-replayed decode steps. The 3 redundant DECODE-only
attempt dirs were **removed** (DECODE data already exists in `sglang_formal/`), so no EXTEND-formal
dir exists for Case B.

**Impact: low.** The graph-off **mapping** trace is the one that carries the
`kernel → cpu_op → python_scope` mapping required for prefill-stage attribution, and Case B mapping
EXTEND **was** captured (64 MB). The graph-on EXTEND would only have added real-perf timing for the
prefill step; Case B already carries confidence ceiling M (both frameworks bimodal), so a single
missing graph-on prefill window does not change the analysis ceiling.

## Do old DECODE-only SGLang traces remain valid?

Yes. The original `sglang_mapping/` and `sglang_formal/` DECODE traces (all 4 cases) are untouched and
remain the authoritative **decode-stage** SGLang traces. This supplement only **adds** prefill-stage
coverage; it supersedes nothing.

## Phase 4 readiness

- SGLang **EXTEND (prefill) stage**: mapping graph-off captured for **all 4 cases**; formal graph-on
  captured for A/C/D (B graph-on absent — low impact, see above).
- SGLang **DECODE stage**: mapping + formal captured for all 4 cases (original collection).
- vLLM prefill_like + decode_like: captured for all 4 cases (original collection; not re-run).

**Phase 4 can proceed for all 4 cases on both stages.** Only caveat: Case B prefill-stage perf timing
relies on the graph-off mapping trace rather than a graph-on formal trace.
