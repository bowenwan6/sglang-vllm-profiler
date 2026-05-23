# Case B — SGLang EXTEND (prefill) Triage — **UNAVAILABLE**

Run: `run2_qwen3vl8b` · Case B `caseB_longprefill` (2048→128, c=1, default) · stage **EXTEND/prefill**.

> **No usable SGLang EXTEND trace exists for Case B.** All Case B EXTEND conclusions therefore rely on
> the vLLM prefill cross-check (`vllm_crosscheck.md`) for prefill-stage context, and carry **confidence
> ceiling M** (Case B is also bimodal in both frameworks). This is a documented deviation, not an oversight.

## Why unavailable (full provenance)

| Source | Outcome |
|---|---|
| Phase-3 graph-on EXTEND formal | Missing — 8-attempt deviation (num_steps=10 → DECODE-only; ≥50 → empty). |
| Phase-3 graph-off EXTEND mapping (original) | File on disk but **corrupt**: `gzip -t` / python gzip → `EOFError: Compressed file ended before end-of-stream`. |
| Phase-4 re-collect attempt 1 (orig mechanism) | Truncated EXTEND-named gz (66 MB, unreadable). |
| Phase-4 re-collect attempt 2 (flush-aware, server kept alive) | Produced a **valid DECODE** trace (864 MB), not EXTEND. |
| Phase-4 re-collect attempt 3 (`--disable-radix-cache`, post-kill flush poll) | Again produced a **valid DECODE** trace, no EXTEND. |

**Root cause (diagnosed):** the `max_new_tokens=1` prefill-only load on Case B's **2048-token** prompts,
under `sglang.profiler --profile-by-stage --num-steps 10`, does not land a clean EXTEND window:
(a) the repeated 64-prompt pool gets **prefix-cached** (server log: `#new-token: 1, #cached-token: ~2170`),
so there are no real long-prefill forward steps; (b) even with `--disable-radix-cache` forcing genuine
prefills, the 10-step profiler window is labeled **DECODE**; (c) the EXTEND-named attempts truncate at
shutdown flush. This mirrors the original graph-on 8-attempt failure — a profiler-mechanism limit for the
long-prefill Case B path, fixable only with deeper profiler/source changes (out of Phase-4 scope).

All quarantined attempt dirs are preserved in `traces/.../sglang_extend_mapping/{CORRUPT_,TRUNC_,DECODEONLY_}*`
(not deleted). Canonical Case B DECODE mapping/formal traces are intact and used in `decode_triage.md`.

## Prefill-stage substitute evidence

- **vLLM prefill_like** (real 2048-prefill, captured): GEMM-bound — nvjet `128x272` 51.3% + `256x144`
  26.8% via **`aten::mm` (eager, NOT cudaGraphLaunch)**, FA3 attention 6.9%. See `vllm_crosscheck.md`.
- **SGLang DECODE** (captured, valid): see `decode_triage.md` — same nvjet FP8 family at `unquant.py:138`,
  78.3% (PR #22392 catalog hit), confirming the kernel→source mapping that an EXTEND mapping would have
  provided is the *same* GEMM family.

**Net:** the missing SGLang EXTEND trace is low-impact for kernel attribution (the dominant nvjet FP8
GEMM family + its `unquant.py:138` source site are already established from DECODE and from the other
cases), but it means **no SGLang prefill-stage *timing*** for Case B — so prefill-side gap claims for
Case B are vLLM-referenced and ≤ M only.
