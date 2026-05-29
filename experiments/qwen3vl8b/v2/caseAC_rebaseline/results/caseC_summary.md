# v2 Issue #2 — Case C results (production-default overlap-ON, batched boundary)

**Workload:** `caseC_batched` 512→128, c=16 · 2000 prompts · warmup 500 · reps 3/block · greedy.
**Design:** **interleaved** S0/S2/S0/S2/S0/vLLM (`C_S0_a, C_S2_a, C_S0_b, C_S2_b, C_S0_c, C_V0_vllm`) to
average out the ~17% batched-session variance seen in v1. **No `C_S0_abl_no_overlap`** (not approved).
**Run:** GPU 1, serialized servers, **clean** (no KAPI, no profiler, no CUDA-IPC). Dataset sha256
`265bde3e48793077…`. SGLang `0.0.0.dev1+g0c8049d9b` (FlashInfer), vLLM `0.21.0` (FA3).
**Outcome:** all 6 variants `OK`, **0 failures / 12000 completed requests**.

## TTFT / TPOT (per block, then pooled)

| Variant | Config | TTFT p50 (ms) | CV% | reps (ms) | TPOT p50 | e2e p50 | out tok/s |
|---|---|---:|---:|---|---:|---:|---:|
| `C_S0_a` | default (overlap-ON) | 207.9 | 12.7 | 207.9 / 161.9 / 219.7 | 6.38 | 1004 | 2028 |
| `C_S0_b` | default | 163.1 | 11.3 | 161.9 / 204.8 / 163.1 | 6.60 | 1004 | 2027 |
| `C_S0_c` | default | 204.8 | 16.8 | 204.8 / 210.4 / 141.5 | 6.41 | 1004 | 2022 |
| `C_S2_a` | + `--enforce-piecewise-cuda-graph` | 231.4 | 0.2 | 230.3 / 231.4 / 231.4 | 6.16 | 1014 | 2006 |
| `C_S2_b` | + PCG | 229.0 | 19.9 | 231.0 / 229.0 / 145.1 | 6.17 | 1014 | 2001 |
| `C_V0_vllm` | vLLM anchor | 215.7 | 4.7 | 223.3 / 199.2 / 215.7 | 6.23 | 998 | 2036 |

**Pooled across blocks:**

| Group | TTFT p50 pooled (ms) | CV% | n reps |
|---|---:|---:|---:|
| SGLang **default (S0)** | **204.8** | 14.5 | 9 |
| SGLang **PCG (S2)** | **230.6** | 14.7 | 6 |
| vLLM | 215.7 | 4.7 | 3 |

## Findings

1. **No Case-A-like PCG benefit at c=16 batched — boundary confirmed on the production default.** Pooled
   PCG TTFT (230.6 ms) is **not below** default (204.8 ms); the −36% Case-A improvement does **not** appear
   here. If anything PCG trends slightly higher, but both groups carry ~14–15% session CV (individual S0
   reps span 141–220 ms, S2 145–231 ms), so the right statement is **no material TTFT improvement from PCG
   in the batched regime**, not that PCG hurts.

2. **SGLang default ≈ vLLM within the batched noise band.** Default SGLang (204.8 ms) and vLLM (215.7 ms)
   sit ~11 ms apart while the SGLang batched CV is ~14.5% (≈ ±30 ms) — i.e. **no material cross-framework
   TTFT gap** at c=16. TPOT (6.2–6.6 ms), e2e (~1.00–1.01 s), and throughput (~2000–2036 tok/s) are
   comparable across all variants.

3. **Confirms the v1 Case C boundary on the production-default baseline.** The PCG lever's value is
   **workload-shape-dependent**: it closes the low-concurrency Case-A first-token gap but yields no benefit
   once requests batch (c=16). This supports **selective enablement** (low-concurrency / text-only /
   shape-stable), not a global VLM force-on — the design question owned by Issue #5.

## Caveats

- High batched session variance (S0/S2 block CV up to ~20%) — the interleaved design averages it but
  per-block medians remain noisy; conclusions are stated at the "no material gap / no Case-A-like benefit"
  level, not as precise parity.
- Attention backends unaligned (FlashInfer vs FA3) → confidence ceiling M on any attention-kernel claim.
- `--enforce-piecewise-cuda-graph` is a testing lever, not production behavior.

**Acceptance (Issue #2, Case C):** PASS — clean production-default batched baseline established, PCG
re-tested, 0 failures; result = no material gap and no Case-A-like PCG benefit at c=16 (boundary confirmed).
