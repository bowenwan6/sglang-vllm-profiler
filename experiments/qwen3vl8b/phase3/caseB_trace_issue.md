# Case B — SGLang EXTEND trace issue & provenance

This file is the permanent record of why **Case B (`caseB_longprefill`, 2048→128, c=1) has no usable
SGLang EXTEND/prefill-stage trace**. The large failed/corrupt trace files described below were
**deleted** during the repo restructure (they were untracked, ~258 MB total, no LFS/Git value); this
markdown preserves their provenance. Decode-stage analysis is unaffected (canonical
`sglang_mapping/` + `sglang_formal/` DECODE traces are intact and valid).

## Summary

| Stage | SGLang EXTEND availability |
|---|---|
| graph-on **formal** | ❌ never captured — 8 attempts during Phase 3 (3 in the EXTEND supplement + a dedicated 5-strategy retry) |
| graph-off **mapping** | ❌ original gz corrupt; 3 Phase-4 re-collection attempts all failed |

Net: **no usable SGLang EXTEND trace for Case B.** All Case B prefill-stage conclusions are
vLLM-referenced only and carry **confidence ceiling M** (Case B is also bimodal in both frameworks).

## Deleted failed-attempt files (provenance)

| Quarantine dir (deleted) | File | Size | Problem |
|---|---|---|---|
| `CORRUPT_1779467572.6199067_*` | `…-TP-0-EXTEND.trace.json.gz` | 64 MB | original supplement mapping trace — gz truncated (`gzip -t` / python gzip → `EOFError: Compressed file ended before the end-of-stream marker`) |
| `TRUNC_1779496946.9542902_*` | `…-TP-0-EXTEND.trace.json.gz` | 64 MB | Phase-4 re-collect attempt 1 — EXTEND-named but truncated (unreadable) |
| `DECODEONLY_1779497165.450408_*` | `…-TP-0-DECODE.trace.json.gz` | 68 MB | Phase-4 re-collect attempt 2 (flush-aware) — valid but **DECODE-stage**, not EXTEND |
| `DECODEONLY_1779497567_*` | `…-TP-0-DECODE.trace.json.gz` | 62 MB | Phase-4 re-collect attempt 3 (`--disable-radix-cache`) — valid but **DECODE-stage**, not EXTEND |

Also deleted: ~863 MB of untracked `*_kapi_*.log` (re-collect runs) and `*recollect*_server.log`
debug logs (no reproduction value; key facts captured here).

## Re-collection attempts (Phase 4, GPU 1, SGLang only)

Scripts retained for reproducibility under `experiments/qwen3vl8b/phase3/scripts/`:
`run_phase3_caseB_extend_mapping_recollect.py` (attempt 1),
`…_recollect2.py` (attempt 2, flush-aware),
`…_recollect3.py` (attempt 3, `--disable-radix-cache` + post-kill flush poll).

| Attempt | Mechanism | Result |
|---|---|---|
| 1 | original `collect_extend()` (graph-off, prefill-only `max_new_tokens=1`, `num_steps=10`) | truncated EXTEND gz |
| 2 | keep server alive + poll gz readability before kill | valid **DECODE** trace, no EXTEND |
| 3 | `--disable-radix-cache` to force real long prefill + post-kill flush poll | valid **DECODE** trace, no EXTEND (`ok=false`, metadata `extend_mapping_recollect` attempt 3) |

## Root cause

Case B's repeated 64-prompt pool gets **prefix-cached** (server log: `#new-token: 1,
#cached-token: ~2170`), so a `max_new_tokens=1` prefill-only load produces no genuine 2048-token
prefill forward steps. Under `sglang.profiler --profile-by-stage --num-steps 10` the window then lands
on DECODE-labeled steps; the EXTEND-named attempts truncate at shutdown flush (the large long-prefill
trace is written at server SIGTERM and gets cut off). Even `--disable-radix-cache` (forcing real
prefills) still yielded a DECODE-stage trace. This is the **same profiler-mechanism limit** as the
prior 8-attempt graph-on formal miss — capturing a clean Case B long-prefill EXTEND stage would require
SGLang profiler/source changes, which are out of scope.

## Impact (low for attribution, real for timing)

- **Low** for kernel→source attribution: the dominant `nvjet_sm90_*` FP8 GEMM family and its
  `srt/layers/quantization/unquant.py:138 apply` source site are already established from Case B DECODE
  and from Cases A/C/D EXTEND.
- **Real** for prefill-stage *timing*: there is no SGLang prefill timing for Case B; the prefill-side
  cross-check uses only the vLLM `prefill_like` window (which did capture the real 2048-prefill).

See `experiments/qwen3vl8b/phase3/extend_supplement_summary.md` (graph-on 8-attempt detail) and
`analysis/qwen3vl8b/caseB_longprefill/extend_triage.md` (Phase-4 unavailable writeup).
