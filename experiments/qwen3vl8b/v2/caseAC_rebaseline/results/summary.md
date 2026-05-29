# v2 / Issue #2 — Default-overlap Qwen3-VL rebaseline (Case A + C)

**Goal (Issue #2):** re-establish the Qwen3-VL clean headline on the **production-default overlap schedule
(ON)** — demoting v1's `--disable-overlap-schedule` Case-A baseline to an ablation — and re-test whether
the piecewise-CUDA-graph (PCG) lever still helps against that production baseline.

**Run conditions.** GPU **1**, single H200, TP=1, bf16, greedy. Servers serialized (never co-resident).
**Clean:** no KAPI logging, no profiler, `SGLANG_USE_CUDA_IPC_TRANSPORT` unset (text-only). Model
`Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd`. SGLang `0.0.0.dev1+g0c8049d9b` (FlashInfer), vLLM `0.21.0` (FA3).
**Totals:** Case A 5 variants × 5 reps + Case C 6 variants × 3 reps = **43 benchmark runs, 0 failures**.

## Headline numbers (TTFT p50, median of reps)

| | SGLang default (overlap-ON) | SGLang + PCG | vLLM | PCG effect |
|---|---:|---:|---:|---|
| **Case A** (128→128, c=1) | 21.94 ms (CV 1.1%) | **14.04 ms** (CV 4.9%) | 13.12 ms (CV 3.7%) | **−36%, into vLLM band** |
| **Case C** (512→128, c=16) | 204.8 ms (CV 14.5%) | 230.6 ms (CV 14.7%) | 215.7 ms (CV 4.7%) | **none / not below default** |

(Case A also ran the `--disable-overlap-schedule` ablation: 19.07 ms TTFT — lower TTFT than overlap-ON but
worse TPOT 5.87 vs 5.47 ms and throughput 167 vs 179 tok/s. TPOT unchanged by PCG in both cases.)

## Conclusions

1. **The v1 PCG finding holds on the production default.** Under overlap-ON, forcing the piecewise CUDA
   graph drops Case-A TTFT 21.94 → 14.04 ms (TPOT flat, 0 failures), reaching the vLLM band. The default
   overlap-ON gap to vLLM (21.9 vs 13.1) is in fact **larger** than v1's overlap-OFF gap (19.1 vs 13.1), so
   the original baseline choice **understated** the production TTFT gap. **Justin's #2 concern is resolved:
   PCG is not an artifact of the overlap-OFF baseline.**

2. **Case C boundary confirmed on the production default.** At c=16 batched, PCG gives **no Case-A-like
   benefit** (S2 230.6 ms ≥ S0 204.8 ms, within ~14–15% session noise), and SGLang default ≈ vLLM within
   the batched noise band. The lever is **workload-shape-dependent** → favors **selective enablement**, not
   a global VLM force-on (Issue #5).

3. `--enforce-piecewise-cuda-graph` remains a **testing lever, not production behavior**; attention
   backends unaligned (FlashInfer vs FA3) → confidence ceiling M on attention-kernel claims.

## Issue #2 acceptance: **PASS**

Production-default overlap-ON baseline established for both cases, PCG re-tested against it, 0 failures,
Case-A CVs single-digit with negligible bracket drift (S0 21.94 → repeat 21.99 ms). The PCG headline is now
anchored to the production default.

## Artifacts

- `results/caseA_results.json`, `results/caseA_summary.md`
- `results/caseC_results.json`, `results/caseC_summary.md`
- raw per-rep: `results/raw/<variant>_rep<N>.json` (+ `_meta.json`)
- server logs: `logs/qwen3vl8b/v2/caseAC_rebaseline/<variant>_server.log`
- console: `caseA_console.log`, `caseC_console.log`

## Next (per v2 roadmap)

`#2 → {#4 image+text + CUDA-IPC, #3 Qwen3.5} → #5 selective/default-on PCG PR`. With #2 now PASS, the next
foundational step is **#4 (image+text + `SGLANG_USE_CUDA_IPC_TRANSPORT=1`)** and/or **#3 (Qwen3.5 transfer)**.
