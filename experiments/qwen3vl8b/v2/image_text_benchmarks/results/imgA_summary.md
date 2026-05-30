# IMG-A Benchmark Summary — image+text (c=1)

> Run: 2026-05-30 21:21 UTC  GPU=7  seed=1  n=400  warmup=30  reps=5  resolution=720p  range_ratio=1.0

> SGLang image headline baseline: `SGLANG_USE_CUDA_IPC_TRANSPORT=1` (IPC on).

> IPC benefit and PCG benefit reported separately.

> vLLM is anchor only — no causal inference.

> **Image+text conclusions are separate from text-only Issue #2 findings.**

## Headline numbers (TTFT p50, median of reps)

| variant | ipc | pcg | ttft_p50 median (ms) | CV% | tpot_p50 (ms) | out_tok/s | status |
|---|---|---|---|---|---|---|---|
| IMG_A_S0_ipc | on | off | FAIL | ?% | ? | ? | INVALID_FAILURES |

## Bracket drift (S0_ipc vs S0_ipc_repeat)

⚠ One or both bracket variants failed — drift cannot be assessed.

## PCG benefit (Q2): S0_ipc vs S2_ipc_pcg

⚠ S0_ipc or S2_ipc_pcg failed — PCG benefit cannot be assessed.

## IPC benefit (Q3): S0_noipc vs S0_ipc

⚠ S0_ipc or S0_noipc failed — IPC benefit cannot be assessed.

## SGLang IPC baseline vs vLLM anchor (Q1)

⚠ S0_ipc or V0_vllm failed.

## Token composition (per request)

Vision tokens: ?/req  |  Text tokens: ?/req  |  Resolution: 720p, image-count=1, range_ratio=1.0, seed=1

## Failure summary

❌ 1 variants with issues:
  - IMG_A_S0_ipc: status=INVALID_FAILURES

## Failure diagnosis

**Error:** `"No data iterator found for token: <|video_pad|>"` (SGLang HTTP 400, `BadRequestError`).

**Pattern:** 2 failures at measured-window indices 44 and 185 in rep3 only. Rep1 and rep2 completed 400/400 with 0 failures. All reps use the same 400 seeded requests (deterministic dataset). The same server instance served all 3 reps (~860 total requests before rep3's failure).

**Root cause hypothesis:** Server-side cache state accumulation. After ~800 requests (reps 1–2 + warmup), SGLang's RadixCache/image cache reaches a state where certain requests trigger the `<|video_pad|>` code path. The Qwen3-VL model has both `<|image_pad|>` and `<|video_pad|>` multimodal tokens; if cache contamination causes the server to interpret an image-only request as requiring a video iterator, the 400 error follows. Rep1 and rep2 processed the identical prompts without failure, confirming this is server-state-triggered, not prompt-content-triggered.

**Not investigated:** Whether `/flush_cache` between reps, or a fresh server restart between reps, would prevent the issue.

**Corrective options (requires user decision):**
1. **Restart server between reps** (safest): each rep starts with clean server state. Adds ~30–60s per inter-rep restart overhead.
2. **Call `/flush_cache` between reps**: resets RadixCache without full restart; faster but less thorough.
3. **Try a different seed** (`--seed 2`): different request order might avoid the problematic cache state. Cache contamination root cause unresolved.
4. **Accept 0.5% failure rate** and run with `--allowed-failure-rate` flag if supported: not recommended for headline data.

**Partial data (reps 1–2 of S0_ipc, informational only — not headline):**
| rep | completed | failures | ttft_p50 (ms) | tpot_p50 (ms) |
|---|---|---|---|---|
| 1 | 400 | 0 | 87.06 | 5.20 |
| 2 | 400 | 0 | 61.81 | 5.20 |
| 3 (stopped) | 398 | **2** | 60.69 | 5.20 |

Note: rep1 TTFT (87ms) is higher than rep2 (62ms), suggesting 30 warmup requests are insufficient to fully settle the image pipeline for 720p inputs. This is a secondary finding relevant to warmup tuning.

**Do NOT use partial rep1/rep2 numbers as headline results.** Headline requires 5 clean reps.

## Recommendation

IMG-A incomplete due to server-state-related failures in rep3. **Do not proceed to IMG-B / IMG-C.** Await user decision on corrective option (see above) before re-running S0_ipc.