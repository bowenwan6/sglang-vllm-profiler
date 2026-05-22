# run2 Phase 3 — Trace Collection Summary

Generated: 2026-05-22 · GPU 1 · model Qwen3-VL-8B-Instruct @ 0c351dd · TP=1 · bf16.
Collection only — **no interpretation** (that is Phase 4). Servers ran strictly serial; GPU 1 freed
(<2000 MiB) and verified after every server.

## Outcome

**All 4 cases completed; all 16 trace groups non-empty.** Case A pilot gate **PASSED**, so C/B/D ran
automatically. 0 fatal errors in any server log. GPU 1 freed after every server. Total traces: ~517 MB.

> Process note: the orchestrator finished cleanly ("Phase 3 collection complete"). The shell pipeline
> reported exit 1 only because the `tee` log-redirect target dir did not exist at redirect time — a
> cosmetic wrapper issue, not a collection failure. Full run log is in the task output / metadata.

## Per-case trace inventory

| Case | SGLang mapping (graph-off) | SGLang formal (graph-on) | vLLM prefill_like | vLLM decode_like |
|---|---|---|---|---|
| A 128→128 c1 (`--disable-overlap-schedule`) | 33 MB ✅ DECODE | 3.4 MB ✅ DECODE | 2.8 MB ✅ | 64 MB ✅ |
| C 512→128 c16 (default) | 34 MB ✅ DECODE | 6.7 MB ✅ DECODE | 4.8 MB ✅ | 108 MB ✅ |
| B 2048→128 c1 (default, **ceiling M**) | 33 MB ✅ DECODE | 3.4 MB ✅ DECODE | 5.0 MB ✅ | 63 MB ✅ |
| D 512→512 c16 (default) | 33 MB ✅ DECODE | 6.7 MB ✅ DECODE | 4.7 MB ✅ | 114 MB ✅ |

Trace file types: SGLang `*-TP-0-DECODE.trace.json.gz` + `server_args.json`; vLLM `async_llm.*.pt.trace.json.gz`
+ `rank0.*.pt.trace.json.gz` + `profiler_out_0.txt`.

## Paths

```
traces/run2_qwen3vl8b/{caseA_short,caseC_batched,caseB_longprefill,caseD_decode}/
  sglang_mapping/<ts>/   *-TP-0-DECODE.trace.json.gz + server_args.json
  sglang_formal/<ts>/    *-TP-0-DECODE.trace.json.gz + server_args.json
  vllm/prefill_like/     async_llm + rank0 *.pt.trace.json.gz
  vllm/decode_like/      async_llm + rank0 *.pt.trace.json.gz
experiments/run2_qwen3vl8b/phase3/metadata/{case}_meta.json   # versions/flags/warmup/reps/SHA/GPU/sizes
experiments/run2_qwen3vl8b/phase3/metadata/phase3_run_summary.json
logs/run2_qwen3vl8b/phase3/                                   # server + kapi L1 logs
```

## Collection mechanism (recorded for reproducibility)

- **SGLang**: server launched per trace (mapping = `--disable-cuda-graph --disable-piecewise-cuda-graph`;
  formal = without). Concurrent `bench_serving` load (case dataset, case concurrency, case warmup) ran
  while `python3 -m sglang.profiler --profile-by-stage --num-steps 10 --output-dir <abs>` captured.
  `SGLANG_KERNEL_API_LOGLEVEL=1` on every launch.
- **vLLM**: server launched per window with
  `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<abs>"}'` (vLLM 0.21.0 dropped the
  `VLLM_TORCH_PROFILER_DIR` env). prefill_like: `/start_profile` → 8×`/v1/completions max_tokens=1` at
  c=1 → `/stop_profile`. decode_like: steady c-concurrency load → `/start_profile` → ~6 s → `/stop_profile`.

## Deviations / caveats (for Phase 4 awareness — not interpretation)

1. **SGLang stage in the ORIGINAL collection = DECODE only.** The initial `--profile-by-stage` runs
   captured 10 decode forward-steps each; no EXTEND/PREFILL-stage SGLang trace was produced (profiler
   armed during in-flight decode). **RESOLVED by the EXTEND supplement** (see
   `extend_supplement_summary.md`): a prefill-only load (`max_new_tokens=1`) was used to capture
   `sglang_extend_{mapping,formal}/` EXTEND traces — captured for A/C/D (both graph modes) and Case B
   graph-off mapping; Case B graph-on EXTEND could not be captured **after 8 attempts** (a dedicated
   5-strategy retry — larger num_steps, higher conc — also failed; num_steps≥50 yields empty stage
   traces on the graph-on path). Low impact — the graph-off mapping trace carries the kernel→source
   mapping. Original DECODE traces remain valid and untouched.
2. **Case B carries confidence ceiling M** (both SGLang and vLLM bimodal at 2048→128 c=1) — recorded
   in `caseB_longprefill_meta.json`; all Case B cross-framework conclusions in Phase 4 must carry M.
3. **Single representative trace per (framework, stage, case)** — not 3/5 repeated trace reps. The
   case `warmup`/`reps` values describe the benchmark protocol and are recorded in metadata; profiling
   captured one steady-state window per group (standard practice; `num_steps=10`).

## Phase 3 exit criteria — status

- ✅ SGLang mapping trace exists + non-empty (all 4 cases).
- ✅ SGLang formal trace exists + non-empty (all 4 cases).
- ✅ vLLM prefill_like trace exists + non-empty (all 4 cases).
- ✅ vLLM decode_like trace exists + non-empty (all 4 cases).
- ✅ Metadata records framework versions/flags/warmup/reps/dataset SHA/GPU/timestamps/sizes; Case B ceiling M.
- ✅ Stage separation: DECODE captured for SGLang (all 4 cases); EXTEND captured via supplement for
  all 4 cases (graph-off mapping) + A/C/D graph-on (Case B graph-on EXTEND absent, low impact).
- ✅ No final hypotheses produced (Phase 4 does that).

**Phase 4 can start** on both stages: SGLang DECODE (original) + EXTEND (supplement) + vLLM
prefill/decode windows. See `extend_supplement_summary.md` for the EXTEND details.
