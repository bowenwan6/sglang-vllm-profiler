# v2 Issue #2 — Case A results (production-default overlap-ON rebaseline)

**Workload:** `caseA_short` 128→128, c=1 · 400 prompts · warmup 30 · reps 5 · greedy.
**Run:** GPU 1, serialized servers, **clean** (no KAPI, no profiler, no CUDA-IPC). Dataset sha256 `fab4917772e08744…`.
**Model:** `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd`. SGLang `0.0.0.dev1+g0c8049d9b` (FlashInfer), vLLM `0.21.0` (FA3).
**Outcome:** all 5 variants `OK`, **0 failures / 2000 completed requests**, all TTFT CV ≤ 4.9%.

## TTFT / TPOT medians-of-reps

| Variant | Config | TTFT p50 (ms) | CV% | TTFT p95 | TTFT p99 | TPOT p50 (ms) | out tok/s |
|---|---|---:|---:|---:|---:|---:|---:|
| `A_S0_default` | SGLang **default (overlap-ON)** | **21.94** | 1.1 | 26.6 | 27.4 | 5.47 | 178.7 |
| `A_S0_default_repeat` | bracket repeat of S0 | 21.99 | 1.3 | 26.7 | 26.8 | 5.47 | 178.8 |
| `A_S2_pcg` | overlap-ON **+ `--enforce-piecewise-cuda-graph`** | **14.04** | 4.9 | 18.8 | 21.2 | 5.47 | 180.8 |
| `A_V0_vllm` | vLLM anchor | **13.12** | 3.7 | 18.7 | 19.0 | 5.64 | 175.6 |
| `A_S0_abl_no_overlap` | ablation `--disable-overlap-schedule` (v1 baseline) | 19.07 | 1.6 | 24.7 | 24.9 | 5.87 | 167.3 |

## Findings (answers to the three Issue #2 questions)

1. **Does PCG still help against the production-default overlap-ON baseline? YES — and by more.**
   Forcing the piecewise CUDA graph on drops SGLang TTFT **21.94 → 14.04 ms (−36%)** with TPOT unchanged
   (5.47 ms) and 0 failures, landing in the vLLM TTFT band (13.1 ms). The v1 PCG finding is **confirmed
   on the production default**, not an artifact of the overlap-OFF baseline. The default-overlap TTFT gap
   to vLLM (21.9 vs 13.1) is in fact *larger* than the v1 overlap-OFF gap (19.1 vs 13.1).

2. **How does production-default overlap-ON compare to the v1 no-overlap baseline?**
   Overlap-ON has a **higher TTFT** (21.94 vs 19.07 ms) but **better decode**: TPOT 5.47 vs 5.87 ms and
   throughput 178.7 vs 167.3 tok/s. So v1's `--disable-overlap-schedule` choice happened to *lower TTFT*
   while costing decode — choosing it as the Case-A baseline understated the true production TTFT gap. The
   PCG lever closes the gap under either baseline.

3. **SGLang vs vLLM anchor.** Default SGLang TTFT (21.9 ms) sits ~8.8 ms above vLLM (13.1 ms); with PCG
   SGLang (14.0 ms) is within ~0.9 ms of vLLM and overlapping in p95/p99. TPOT is comparable across all
   (SGLang slightly better, 5.47 vs 5.64 ms).

## Stability / caveats

- **Bracket drift negligible:** S0 21.94 → S0_repeat 21.99 ms (~0.2%) → no warm-up/thermal drift across
  the ~70-min Case-A run; the v1 S2-CV weak spot is tightened (S2 CV here 4.9%, single-digit).
- `--enforce-piecewise-cuda-graph` remains a **testing lever, not production behavior** (it overrides the
  VLM auto-disable). The selective/default-on design is Issue #5, not this run.
- Attention backends unaligned (FlashInfer vs FA3) → attention-kernel-level claims carry confidence
  ceiling M; this run makes only a framework-level TTFT-gap claim.

**Acceptance (Issue #2, Case A):** PASS — clean production-default baseline established, PCG re-tested and
confirmed against it, 0 failures, all CV single-digit, bracket drift negligible.
