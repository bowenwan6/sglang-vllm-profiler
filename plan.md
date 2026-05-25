# SGLang vs vLLM Profiling — Execution Plan

---

## 0. Active Run Status — `qwen3vl8b`

> **The active run is `qwen3vl8b`.** It reuses the methodology, fairness model, and artifact
> spec below, but is a *fresh measurement round* on a rebuilt machine. The original run ("run1") is
> historical reference only.

| Item | Value |
|---|---|
| Run id | `qwen3vl8b` |
| Model | `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (weights identical to run1, sha256-verified) |
| GPU | single H200, **serialized** throughout. Phase 0/1: index **0**; Phase 2: index **7**; Case C W500 probe + Phase 3: index **1**. (One GPU at a time; servers never co-resident.) |
| SGLang | `0.0.0.dev1+g0c8049d9b` (system python3, editable `/sgl-workspace/sglang`) |
| vLLM | `0.21.0` (conda env `/opt/miniconda3/envs/profiling`) |
| torch / CUDA | `2.11.0+cu130` / `13.0` (both frameworks aligned) |

**Phase status:**

| Phase | status |
|---|---|
| 0 — Equivalence | ✅ **complete + PASS** (canonical: `experiments/qwen3vl8b/phase0/`) |
| 1 — Baseline | ✅ **complete** (24 runs, 0 failures; `experiments/qwen3vl8b/phase1/summary.md`) |
| 2 — Shaping / Variance gate | ✅ **complete** (0 failures; `experiments/qwen3vl8b/phase2/{summary.md,selected_cases.md}`) |
| 3 — Profiling / Trace collection | ✅ **complete** (GPU 1; `experiments/qwen3vl8b/phase3/{summary.md,extend_supplement_summary.md}`) |
| 4 — Triage | ✅ **complete** (all 4 cases; `analysis/qwen3vl8b/` + `reports/qwen3vl8b/03_profiling_analysis.md`) |
| 5 — Validation | 🔄 **in progress** — Case A clean confirmation completed (H1 strengthened); Case C pending validation |

**Phase 5 status (in progress — NOT complete).**

*Phase 5.1 (offline, done).* **Mechanism confirmed:** for the current Qwen3-VL config SGLang decode
CUDA graph is nominally ON, but VLM auto-disable turns the **prefill piecewise CUDA graph OFF**
(`server_args.py:1308`), and `enable_torch_compile=False`. **But** existing profiler traces **cannot**
reliably establish serving-path graph coverage or CPU-launch-gap causality (captured-window / profiling
confound — formal DECODE trace shows 0 graph-replay despite `disable_cuda_graph=False`). **H1 remains a
hypothesis, not a root cause.** See `analysis/qwen3vl8b/phase5/h1_launch_gap/`.

*Phase 5.2 initial intervention (Case A, GPU 3) — **EXPLORATORY ONLY, instrumentation-confounded.***
All SGLang variants were run with `SGLANG_KERNEL_API_LOGLEVEL=1` (KAPI logging) while the vLLM anchor
was not; S1 graph-off emitted a **14.7 GB** KAPI log. KAPI per-launch logging plausibly slows the
eager/direct-launch path more than the graph path, so the S0→S2 improvement is likely **amplified by
instrumentation**.

| Variant | TTFT p50 median | CV | TPOT p50 | Interpretation |
|---|---:|---:|---:|---|
| V0 vLLM anchor | 12.85 ms | 7.5% | 5.32 ms | contemporaneous reference; **not** KAPI-instrumented |
| S0 SGLang baseline | 53.28 ms | 1.0% | 5.56 ms | instrumented; does **not** reproduce Phase 2's 19.6 ms |
| S1 graph-off | 53.63 ms | 0.4% | 47.67 ms | instrumented negative control; produced 14.7 GB KAPI log |
| S2 enforce piecewise graph | 19.74 ms | 3.1% | 5.55 ms | **promising candidate**, instrumentation-confounded |
| S3 torch.compile | 54.05 ms | 1.3% | 5.15 ms | no observed TTFT benefit in the instrumented screen |

Strongest finding from the *instrumented screen*: "S2 is a promising candidate requiring an
uninstrumented confirmation" (the 14.7 GB S1 KAPI log evidenced the confound). Audit:
`experiments/qwen3vl8b/phase5/caseA_h1_intervention/baseline_anomaly_audit.md`.

*Phase 5.2 CLEAN confirmation (Case A, GPU 6 — GPU 3 was externally occupied; no KAPI, no profiler).*
S0→S2→S0 bracket + clean vLLM anchor, 0 failures:

| Variant | TTFT p50 median | CV | note |
|---|---:|---:|---|
| S0_before | 19.17 ms | 1.5% | **reproduces Phase 2's 19.6 ms** (instrumented 53 ms was a KAPI artifact) |
| S2 enforce piecewise | **11.68 ms** | 10.1% | **−39% vs S0**; 0 errors, smoke OK |
| S0_after | 19.23 ms | 3.8% | bracket drift 0.34% → stable |
| V0 vLLM clean anchor | 13.11 ms | 3.2% | reference |

**H1 strengthened (clean Case A) — but does NOT generalize to batched Case C.**

- *Case A (c=1):* forcing prefill piecewise CUDA-graph coverage cut TTFT ~39% (19.2 → 11.7 ms), TPOT
  unchanged, 0 errors → reached the vLLM TTFT range. **Stability supplement (reps=5):** S2 median
  13.36 ms (CV **12.3%**) vs vLLM 14.49 ms — S2 substantially beats the ~19 ms S0 baseline, but CV>5%
  so **stable parity/superiority over vLLM is NOT claimed** (S2 only "reaches the vLLM TTFT range").
- *Case C (c=16):* **no S2 benefit (confirmed by an interleaved rerun).** First S0→S2→S0 bracket drifted
  17.7% (inconclusive). An **interleaved rerun** (`S0_a→S2_a→S0_b→S2_b→S0_c` + vLLM, GPU 0) better-samples
  the noise: three S0 medians still span **17.3%** (Case-C c=16 TTFT has intrinsic ~17% session variance
  even at w500 — unresolved, a workload property), **but the comparison is now robust**: pooled S2
  (193.6 ms) ≈ pooled S0 (192.2 ms) = **+0.7%**, both ≈ vLLM (189.8 ms). → **H1 does NOT generalize to
  batched c=16**; validated effect limited to Case-A short-latency (c=1) dispatch-bound behavior.
  Consistent with Phase 4 (batched prefill is compute/GEMM-bound). Secondary: enforce-piecewise *reduces*
  batched run-to-run variance (S2 CV 3–4% vs S0 5–9%) without changing median TTFT — a stability note,
  not a latency win.

Caveats: `--enforce-piecewise-cuda-graph` is a **testing lever** (validates locus/direction for Case A,
**not** a production fix; smoke-checked only); Case A S2 CV 10–12%; Case C baseline drifted (warm-up/
environment sensitivity). Detail: `experiments/qwen3vl8b/phase5/{caseA_h1_confirmation,caseA_s2_stability,caseC_h1_confirmation}/summary.md`.
Instrumented GPU-3 screen remains exploratory-only (KAPI confound). Next (needs approval): production-safe
design discussion for the Case-A short-latency path; H2 (PR #22392) as a separate absolute-speed track.

**Phase 4 results (completed 2026-05-23, offline triage, no GPU).** All 4 cases triaged.
A/C/D EXTEND+DECODE two-trace triage successful; B DECODE two-trace successful but **EXTEND
unavailable** (original gz corrupt + 3 re-collect attempts failed — prefix-cache + `--profile-by-stage`
long-prefill limit; caveat, all Case B conclusions ≤ M). vLLM prefill/decode cross-check successful for
all 4 cases. Both stages dominated by the same `nvjet_sm90_*` FP8 GEMM family (72–86%) in **both**
frameworks → GEMM is shared cost, not the differentiator. Global hypotheses + ranked recommendations
available. Commits `871565f` (A) · `440fe0e` (C) · `051e812` (B) · `947fd35` (D) · `55232b3` (synthesis).

- **H1** (impact H / conf M, fairness-independent): SGLang eager `aten::mm` dispatch vs vLLM
  torch.compile/CUDA-graph is the primary TTFT-gap candidate (A, C; not Case B prefill).
- **H2** (abs M / gap L): nvjet FP8 GEMM dominant; PR #22392 CUTLASS-FP8 is an absolute-speed lead, not a gap-closer.
- **H3** (L, ceiling M, fairness-dependent): FlashInfer vs FA3 — not the driver.
- **H4** (L, ceiling M): Case B gap = bimodality + c=1 fixed overhead, not graph coverage.
- **Phase 5 next:** validate H1 first (CPU launch-gap + graph/compile coverage); H2 as a parallel
  absolute-speed track; H3/H4 deprioritized. See §15 + `reports/qwen3vl8b/03_profiling_analysis.md`.

**Phase 3 results (completed 2026-05-22/23, GPU 1).** All 4 cases traced; commits `c6ec1df`
(main collection), `8f41bd3` (EXTEND supplement), `d822bf3` (Case B EXTEND-formal retry). ~517 MB
main + EXTEND traces in Git LFS.

- **DECODE stage (SGLang):** mapping (graph-off) + formal (graph-on) complete for **all 4 cases**.
- **vLLM:** `prefill_like` + `decode_like` windows complete for **all 4 cases** (not re-run since).
- **EXTEND/prefill stage (SGLang):** supplement captured 7/8 groups — EXTEND **mapping** complete for
  **all 4 cases**; EXTEND **formal** complete for **A/C/D**. Only **Case B graph-on EXTEND formal is
  missing** (failed 8 attempts; caveat accepted — see below).
- **Phase 4 can start.** Profile Case A and Case C first; carry the Case B caveats (graph-on EXTEND
  formal missing + confidence ceiling M).

**Phase 1 baseline (active).** 24 runs (4 cases × 2
frameworks × 3 reps), **error rate 0% on all**. Greedy (`temperature=0, top_p=1`, `ignore_eos`
default), GPU 0, both frameworks on torch 2.11.0+cu130 / CUDA 13.0.

| Case | SGLang TTFT p50 | vLLM TTFT p50 | Ratio | TPOT | Note |
|---|---|---|---|---|---|
| A — 128→128, c=1 | 61.8 ms | 12.6 ms | **4.89×** | parity (0.97×) | low CV; cleanest |
| B — 2048→128, c=1 | 66.7 ms | 20.8 ms | **3.20×** | parity (0.97×) | **vLLM bimodal (cv 114%) → ceiling M** |
| C — 512→128, c=16 | 247.5 ms | 187.9 ms | **1.32×** | parity (0.99×) | SGLang p50 cv 9.4% (variance gate needed) |
| D — 512→512, c=16 | 253.0 ms | 189.7 ms | **1.33×** | parity (1.00×) | **SGLang p99 390.6 ms, p99 cv 47% (bimodal tail)** |

Key findings: **TTFT is the only gap**; TPOT and throughput are at parity (0.91–1.01×).
The c=1 dispatch floor persists and **prefill is cheap** — prompt 16× longer (A→B) adds only
**+4.9 ms** to SGLang TTFT (61.8→66.7). Direction is highly consistent with run1; magnitudes differ
(e.g. Case A ratio 4.89× vs run1 3.89×) so **numbers are not interchangeable**. torch/CUDA are now
aligned across frameworks (removes run1's torch-version confound), but the **attention backend still
differs (SGLang FlashInfer vs vLLM FlashAttention v3) → any attention-kernel conclusion carries
confidence ceiling M**. Full table + p95/p99 + CV + error rate: `experiments/qwen3vl8b/phase1/summary.md`.

**Phase 2 results (completed 2026-05-22, GPU 7, 0 failures).** Full detail:
`experiments/qwen3vl8b/phase2/{summary.md,selected_cases.md}`.

| Case | SGLang config (Phase 3) | SGLang TTFT p50 (CV) | vLLM TTFT p50 (CV) | Residual gap | Phase 3 protocol |
|---|---|---|---|---|---|
| A — 128→128 c1 | **`--disable-overlap-schedule`** | 19.6 ms (3.2%) | 12.6 ms (Phase 1) | **1.56×** | warmup 30, 3 reps |
| B — 2048→128 c1 | default | 30.3 ms (**68.4%** ⚠ bimodal) | 21.5 ms (**85.9%** ⚠ bimodal, ceiling M) | 1.41× | warmup 300, 5 reps |
| C — 512→128 c16 | default | **249.1 ms (2.9%, W500)** | 189.0 ms (1.9%, stable) | **1.32×** (SGLang slower) | warmup 500, 5 reps — **clean** |
| D — 512→512 c16 | default | 206.2 ms (3.3%) | 189.7 ms (Phase 1) | 1.09× | warmup 30, 3 reps |

Key Phase 2 findings:
- **Case A — overlap scheduler was a real cost.** `--disable-overlap-schedule` cut TTFT ~10% (21.8→19.6 ms) at clean CV. The Phase-1 4.89× gap is now **1.56×** — a much smaller residual scheduler/dispatch overhead is the Phase-3 target. Other flags (`stream8`, `chunk_off`) were within 5% of default; `chunk_64` was 2.4× worse (eliminated).
- **Case B — both frameworks bimodal.** `chunk_off` beat default by only 3.2% (< 5% threshold) → default wins. SGLang finalist reps 64.3 / 30.3 / 26.9 ms (CV 68.4%); vLLM recheck at warmup=300 still bimodal (first rep high, rest ~21.5 ms, CV 85.9%). **All Case B cross-framework conclusions carry confidence ceiling M.**
- **Case C — clean at W500; SGLang ~1.32× slower (no reversal).** SGLang failed the 5% CV gate at W30/W100/W300 (12.5/15.2/14.9%), but a follow-up **W500 probe (5 reps, GPU 1) passed cleanly at CV 2.9%**, stabilizing at **249.1 ms**. The noisy W100/W300 medians (124–149 ms) that suggested "SGLang faster (0.79×)" were an **under-warmup artifact**: at stable warmup SGLang is ~1.32× *slower* than vLLM (189.0 ms), matching the Phase-1 W30 ratio (247.5 vs 187.9). vLLM is warmup-insensitive (187.9→189.0 ms across W30→W300), so the comparison is a sound stable reference (a strict same-warmup vLLM W500 would only be needed for a "SGLang faster" claim, which no longer holds). **Case C is cleanly profilable.**
- **Case D — clean at warmup=30.** CV 3.3% with no extra warmup; residual gap 1.09× (Phase-1 p99 bimodal tail did not reappear). Lowest Phase-3 priority.

**Phase 3 shortlist (all 4 promote; profiling order A → C → B → D).** Full collection protocol:
`experiments/qwen3vl8b/phase3/plan.md`.

| Case | Decision | SGLang flags | warmup / reps | Phenomenon to profile |
|---|---|---|---|---|
| **A** 128→128 c1 | promote — **highest priority** | `--disable-overlap-schedule` | 30 / 3 | Remaining scheduler/dispatch overhead (19.6 ms vs vLLM 12.6 ms, ~1.56×); cleanest + most actionable |
| **C** 512→128 c16 | promote — **clean** | default | 500 / 5 | Stable batched-c=16 TTFT gap: SGLang ~1.32× slower (249.1 ms vs vLLM 189.0 ms) after W500 |
| **B** 2048→128 c1 | promote — **ceiling M** | default | 300 / 5 | Long-prefill c=1; both SGLang and vLLM bimodal → all cross-framework Case B conclusions carry ceiling M |
| **D** 512→512 c16 | promote — **lower priority** | default | 30 / 3 | Small residual gap (~1.09×, 206.2 ms vs vLLM 189.7 ms); lowest expected payoff |

Order rationale: **A** is the cleanest, most actionable remaining overhead; **C** is now clean post-W500 and represents batched c=16; **B** is important but noisy + ceiling M; **D** has the smallest gap and lowest payoff.

**Historical note.** An earlier exploratory round (informally "run1", different environment — GPU 6,
CUDA 12.9, SGLang `ga4cf2ea12`, vLLM 0.19.0, torch 2.9.1/2.10.0) was **removed** during the repo
restructure; its numbers were not comparable (per §6.4) and are not reused here. The single retained
experiment is `qwen3vl8b`. Environment detail: `experiments/qwen3vl8b/env_snapshot.md`. Working tree:
`experiments/qwen3vl8b/` (see its `README.md`).

---

## 1. Objective

Profile and compare SGLang against vLLM under fair, controlled conditions to extract actionable optimization insights **for SGLang**. The goal is not a generic benchmark summary — it is to explain *why* gaps exist and *what SGLang can do about them*. vLLM is used as a strong reference system whose behavior can falsify or corroborate SGLang-side hypotheses.

## 2. Core Questions

1. Where is the performance gap? (TTFT, TPOT, throughput)
2. Which stage is responsible: prefill/extend or decode?
3. Which subsystem is responsible: kernels, communication, scheduling, memory?
4. Which vLLM behaviors are real design wins vs noise or configuration artifacts?
5. Which of those wins are actionable for SGLang (not already shipped, not already in-flight)?

## 3. Executive Summary

We compare SGLang and vLLM on `Qwen/Qwen3-VL-8B-Instruct` (text-only path first) at TP=1 on a single H200. The work is staged to maximize signal per GPU-hour: establish one small, clean baseline table; find the 1–2 cases whose gap is structural rather than configurational; profile those cases deeply; cross-check findings against vLLM traces; publish ranked recommendations with evidence pointers.

The workflow follows a strict pipeline:

```
 benchmark  →  gap identification  →  config shaping  →  profiling  →  interpretation  →  hypothesis  →  validation
  (Phase 1)       (Phase 1/2)         (Phase 2)       (Phase 3)      (Phase 4)        (Phase 4)     (Phase 5)
```

Three skills each own exactly one layer of this pipeline:

- **`sglang-auto-benchmark`** — controlled-experimentation engine (inputs: flags, datasets, QPS; outputs: metrics).
- **`sglang-torch-profiler-analysis`** — trace interpreter (inputs: traces; outputs: kernel/overlap/fuse tables with catalog cross-reference).
- **`debug-cuda-crash`** — evidence-preservation safety net (passive at `LOGLEVEL=1`; escalated only on specific failure).

Decision rule at every phase boundary: if the gap on a case is <5% we *reshape the workload* before profiling; if it is large and stable we shape SGLang-side configs before concluding it is structural; we only profile cases that survived shaping.

## 4. Models

- **Phase 0–5 (primary)**: `Qwen/Qwen3-VL-8B-Instruct` (≈17 GB, cached at `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/`).
- **Text-only fallback** if Qwen3-VL is not supported on either framework at pinned versions: `Qwen/Qwen3-8B`.
- **Later (Phase 6+, TBD)**: a larger Qwen3-VL variant.

## 5. Environment (active)

- GPUs: 8× H200 144 GB; **active GPU index 0** (`CUDA_VISIBLE_DEVICES=0`); NVIDIA driver 580.159.03, CUDA 13.0.
- **SGLang**: `0.0.0.dev1+g0c8049d9b`, system python3 (3.12), editable at `/sgl-workspace/sglang`, torch 2.11.0+cu130, FlashInfer 0.6.11.post1.
- **vLLM**: `0.21.0` (V1 engine) in conda env `profiling` at `/opt/miniconda3/envs/profiling`, torch 2.11.0+cu130, FlashInfer 0.6.8.post1, FlashAttention v3.
- HF cache: `/root/.cache/huggingface`. Model snapshot `0c351dd…` fully cached, `HF_HUB_OFFLINE=1`.
- Servers run strictly serially (one at a time on GPU 0). Full detail: `experiments/qwen3vl8b/env_snapshot.md`.
- *(An earlier exploratory round used a different stack — GPU 6, CUDA 12.9, SGLang `ga4cf2ea12`, vLLM 0.19.0, torch 2.9.1/2.10.0 — and was removed; see Historical note in §0.)*

---

## 6. Fairness Control Model

Every dimension falls into exactly one tier. Conclusions are only admissible when they are consistent with the tier of the variable they depend on.

### 6.1 Controlled variables — pinned before Phase 1 runs, or the run is invalid

| Dimension | Pinned value | Enforcement |
|---|---|---|
| GPU | Single H200, GPU 6 | `CUDA_VISIBLE_DEVICES=6` on every server + every client |
| Model weights | `Qwen/Qwen3-VL-8B-Instruct` at a fixed HF revision | Record commit-sha of HF snapshot in `env_snapshot.md`; re-verify hash before Phase 3 |
| Tokenizer | Same HF snapshot as weights | Byte-equality check in Phase 0 |
| Dtype | BF16 | Pass `--dtype bfloat16` explicitly on both servers |
| TP / DP / PP | 1 / 1 / 1 | No NCCL on single-GPU baseline |
| Max seq len | 8192 | Pin on both sides |
| Prompt set | One `autobench` JSONL per case, byte-identical | SHA-256 of dataset logged in every run |
| Sampler | temperature=0, top_p=1, greedy | Pass explicitly in client; do not rely on server defaults |
| Output length | Fixed `max_tokens` + `ignore_eos=true` | Prevents early-termination drift between tokenizers |
| Concurrency | {1, 16} per case | Same `--max-concurrency` on both |
| Warmup | ≥30 requests at target concurrency, discarded | Identical warmup script on both |
| Prefix cache | OFF in benchmark runs | Randomized prefixes in the autobench generator |
| Seed | Fixed per case | Both client and dataset generator |

If any of these cannot be pinned on one side, the run is aborted and the mismatch is fixed before Phase 1 begins.

### 6.2 Measured-and-reported variables — cannot be pinned identically, must be logged on every run

| Dimension | Action | Where it goes |
|---|---|---|
| torch / CUDA / FlashInfer / FA version | Record at server startup | `env_snapshot.md`, each run's `meta.json` |
| Attention backend chosen at runtime | Pin where possible (`--attention-backend flashinfer` on SGLang); log actual choice on vLLM | per-run `meta.json` |
| Kernel dispatch (e.g. Triton vs CUDA path) | Observe from trace | Phase 4 triage footer |
| Chunked-prefill chunk size | Log server-reported default | per-run `meta.json` |
| Idle GPU memory after model load | Record once per server startup | `env_snapshot.md` |

Conclusions that depend on these variables must be re-confirmed if the variable changes (e.g. on an SGLang upgrade).

### 6.3 Framework-intrinsic variables — *not* aligned; differences are the observation

| Dimension | Why we don't align it |
|---|---|
| Scheduler policy (SGLang radix vs vLLM cache manager) | This *is* one of the things we are comparing |
| CUDA graph shape selection heuristics | Design difference |
| Chunked-prefill scheduling policy | Design difference |
| Sampler kernel implementation | Implementation difference — relevant if it shows in the trace |

Findings that rest on framework-intrinsic variables are valid *design observations* and go directly into hypotheses. They do not need re-confirmation across runs.

### 6.4 Interpretation rule

> A hypothesis is **admissible** only if every variable it depends on is either *controlled* (pinned) or *framework-intrinsic* (labeled as such in the hypothesis). A hypothesis that depends on a *measured-and-reported* variable carries a confidence ceiling of M until a version-matched re-run confirms it.

---

## 7. Skill Usage Model

Authoritative role definitions. The Quick-Reference in §13 is a summary; this section wins on any conflict.

### 7.0 Skill availability & invocation (verified 2026-05-22)

All three skills this project relies on are **present in the SGLang checkout** at
`/sgl-workspace/sglang/.claude/skills/` and have been registered for Claude Code by symlinking them
into `~/.claude/skills/` (no source modification, no overwrite). Two were upstreamed and **renamed**
to framework-independent names — the names in §7.1/§7.2/§13 below are the original plan names; use
the right-hand column when invoking:

| Plan name (§7/§13) | Installed skill name (current) | Source / status | Availability |
|---|---|---|---|
| `sglang-auto-benchmark` | **`llm-serving-auto-benchmark`** | PR #21736 **merged** into current repo | ✅ `python3 -m sglang.auto_benchmark {convert,validate,run}` works; skill has scripts/configs/references |
| `sglang-torch-profiler-analysis` | **`llm-torch-profiler-analysis`** | upstreamed (unified sglang/vllm/trtllm); BBuf standalone is the older fork | ✅ `analyze_llm_torch_profile.py --help` OK (shim `analyze_sglang_torch_profile.py` kept); catalogs present |
| `debug-cuda-crash` | `debug-cuda-crash` (unchanged) | in current repo; `SGLANG_KERNEL_API_LOGLEVEL/LOGDEST` supported in source | ✅ L1 already in use on every server launch |

Notes: skills load at **session start**, so newly symlinked skills become Skill-tool-invokable only
after a session reload — but their scripts are runnable now via direct path. The optional `b200` /
`h100` / `h200` skills referenced historically no longer exist under those names (BBuf repo
reorganized); no extra install needed.

**When/how to use:**
- **Phase 2 (shaping):** `llm-serving-auto-benchmark` `run` for pure-SGLang YAML flag sweeps, *or*
  the custom scripts (`experiments/qwen3vl8b/phase1/scripts/`-style) when a sweep must also
  drive vLLM or needs full flag control. `debug-cuda-crash` at L1 on every server launch.
- **Phase 3 (collection):** `llm-torch-profiler-analysis` collection scripts
  (`run_sglang_torch_profile_host.sh` + `--profile-by-stage` for SGLang mapping+formal;
  `run_vllm_torch_profile_host.sh` for vLLM windows). No triage yet. `debug-cuda-crash` L1; escalate
  to L3/L5/L10 only on an actual crash.
- **Phase 4 (triage):** `llm-torch-profiler-analysis` `triage` (two-trace SGLang + single-trace vLLM)
  with mandatory catalog lookup against `references/{fuse-overlap-catalog,overlap-catalog,source-map}.md`.
- **Phase 5 (validation):** `llm-serving-auto-benchmark` `run` tier-2 on the hypothesis-named flag;
  optional re-`triage` to confirm the mechanism moved.

### 7.1 `sglang-auto-benchmark` — controlled experimentation on SGLang

**Solves.** Manual flag sweeps are combinatorial, error-prone, non-resumable. This skill takes a YAML spec of candidate flags × QPS × concurrency × SLA, runs each candidate with a fresh server, tracks SLA pass/fail, and writes resumable results. It also owns the canonical autobench JSONL format (`convert` / `validate`).

**Where used.**

| Phase | Subcommand | Purpose |
|---|---|---|
| Phase 1 prep | `convert` + `validate` | Produce the shared autobench JSONL consumed by both `bench_serving --backend sglang-oai` and `bench_serving --backend vllm`. Byte-identity is non-negotiable. |
| Phase 2 | Custom orchestration scripts (not `run`) — direct bench_serving for vLLM compatibility | Rule out "SGLang loses because a flag was wrong". Gate into Phase 3. |
| Phase 5 | `run` tier 2, ≤10 candidates, resumable | Validate specific Phase-4 hypotheses on the exact flag the hypothesis names. |

**Not used for.** Cross-framework comparison (cannot drive vLLM). Broad tier-3 discovery (we do not sweep a space we have not justified). Any interpretation of kernels.

### 7.2 `sglang-torch-profiler-analysis` — trace interpretation

**Solves.** Raw traces are illegible. The skill produces three tables (kernel / overlap / fuse), catalog-checked against `fuse-overlap-catalog.md` and `overlap-catalog.md` so findings are correctly classified as *existing path (disabled/regressed)*, *in-flight PR*, or *truly new* with similarity label.

**Single-trace vs two-trace.** Single-trace is enough for kernel-share and fuse candidates. Two-trace (mapping graph-off + formal graph-on) is required before any overlap claim — graph-off carries the `kernel → cpu_op → python_scope` mapping that graph-on has collapsed.

**Where used.**

| Phase | Flow | Purpose |
|---|---|---|
| Phase 3 | Collection driver only — `sglang.profiler --profile-by-stage` via the skill's script. No triage yet. | Ensures stage separation is shaped the way Phase-4 triage expects. |
| Phase 4 | `triage` two-trace on SGLang per (case × stage); `triage` single-trace on vLLM per (case × window) | Primary interpretive artifact of the project. |
| Phase 4 | Catalog lookup inside each triage (mandatory gate before any hypothesis) | Prevents recommending things SGLang already ships. |
| Phase 5 (optional) | `triage` on winning Phase-5 candidate | Confirms the hypothesized mechanism is the one that moved. |

**Not used for.** Phase 1–2 (no locked case yet). Merged-rank traces (skill prefers rank-local TP-0). Any hypothesis without going through the catalog step.

### 7.3 `debug-cuda-crash` — evidence preservation

**Solves.** CUDA crashes destroy evidence. The `@debug_kernel_api` decorator logs boundary-level input metadata *before* each call so the evidence survives the crash.

**Cost ladder.** L1: names only, near-zero cost. L3: shapes/dtypes, small I/O cost. L5: tensor stats, requires host sync (perturbs timing). L10: full input dumps, disk + real perturbation. Benchmark and profile runs tolerate only L1.

**Where used.**

| Situation | Setting |
|---|---|
| All Phase 1, Phase 2, Phase 3, Phase 5 SGLang runs | `LOGLEVEL=1`, `LOGDEST=logs/{phase}/sglang_%i.log` |
| Crash occurs | Re-run failing case at `LOGLEVEL=3` |
| NaN/Inf suspected in a trace or output divergence appeared in Phase 0 | Targeted `LOGLEVEL=5` reproducer |
| Need offline reproducer | `LOGLEVEL=10` + `DUMP_DIR` + `DUMP_INCLUDE='sglang.custom_op.*'` + `--disable-cuda-graph` |

**Not used for.** Performance analysis. vLLM diagnosis (decorator only instruments SGLang). Running at L≥3 inside any measured run.

### 7.4 Complementarity (one line)

Auto-benchmark controls *inputs*; profiler-analysis interprets *outputs*; debug-cuda-crash preserves *evidence when either fails*. Reaching for the wrong one is the anti-pattern.

---

## 8. Artifact Framework

### 8.1 Filesystem layout

> **Layout note:** the single retained experiment is `qwen3vl8b`; every top-level data directory has a
> matching `qwen3vl8b/` subtree. The earlier exploratory round was removed (see Historical note in §0).

**Directory purpose rule:** `logs/` = infrastructure side-effects (server stderr, kernel-API boundary trails) — consult on failure, never cited in analysis. `experiments/` = research artifacts deliberately produced by the experiment protocol — cited in analysis and reports.

```
/data/sglang-vllm-profiler/
├── plan.md                          this document — single source of truth
├── README.md                        GitHub-facing project overview
│
├── datasets/qwen3vl8b/              canonical autobench JSONL (never regenerate mid-project)
│   ├── caseA_short.jsonl            128→128
│   ├── caseB_longprefill.jsonl      2048→128
│   ├── caseC_batched.jsonl          512→128
│   └── caseD_decode.jsonl           512→512
│
├── logs/qwen3vl8b/                  infrastructure side-effects (consult on failure only)
│   ├── phase1/  phase2/  phase3/    server startup logs + kernel-API (LFS) boundary trails
│
├── experiments/qwen3vl8b/           research artifacts (cited in analysis)
│   ├── README.md                    run overview + phase status
│   ├── env_snapshot.md              versions, backends, GPU memory — all phases
│   ├── phase0/                      equivalence.md + Tier-A/B outputs + scripts (canonical Phase 0)
│   ├── phase1/                      raw/ (bench JSON+meta) + summary.md + scripts/
│   ├── phase2/                      summary.md + selected_cases.md + raw/ + scripts/
│   ├── phase3/                      summary.md + extend_supplement_summary.md + caseB_trace_issue.md
│   │                               + metadata/ + scripts/
│   ├── phase4/                      plan.md
│   └── phase5/                      (pending)
│
├── traces/qwen3vl8b/                raw torch profiler artifacts (Git LFS)
│   └── {case}/
│       ├── sglang_mapping/          graph-off DECODE      sglang_extend_mapping/  graph-off EXTEND
│       ├── sglang_formal/           graph-on DECODE       sglang_extend_formal/   graph-on EXTEND
│       └── vllm/{prefill_like,decode_like}/
│
├── analysis/qwen3vl8b/              interpretation layer (processed from traces)
│   ├── category_regex.md            shared regex applied symmetrically to both frameworks
│   ├── {case}/                      extend_triage.md, decode_triage.md, breakdown.md,
│   │                               vllm_crosscheck.md, preliminary_observations.md (+ *_raw.txt)
│   ├── hypotheses.md                structured hypotheses, de-duplicated
│   └── ranked_recommendations.md    sorted by confidence × impact × feasibility
│
├── reports/qwen3vl8b/               final deliverables (human-facing)
│   ├── 01_current_status_report.md
│   └── 03_profiling_analysis.md
│
└── configs/qwen3vl8b/               (reserved for Phase 5 sweep configs)
```

### 8.2 Artifact layers

| Layer | Contents | Mutability | Purged on rerun? |
|---|---|---|---|
| **Raw** | `datasets/`, `logs/`, `traces/`, `experiments/qwen3vl8b/*/raw/` JSON files | Append-only, never edited by hand | Never — raw evidence is the ground truth |
| **Processed** | `experiments/*/summary.md`, `analysis/**`, `experiments/phase2/selected_cases.md` | Regenerated from raw | Yes, on rerun of the source phase |
| **Deliverable** | `reports/**`, `plan.md`, `README.md`, `experiments/qwen3vl8b/env_snapshot.md`, `experiments/qwen3vl8b/phase0/equivalence.md` | Hand-edited, reviewed | No — edited in place |

### 8.3 Reviewer reading order

A human reviewer validating the project should inspect artifacts in this sequence. Stopping at any layer where confidence is lost is the expected behavior.

1. `reports/05_recommendations.md` — the claims
2. `reports/02_benchmark_table.md` — are the numbers plausible?
3. `analysis/ranked_recommendations.md` — do rankings track evidence?
4. `analysis/{case}/decode_triage.md`, `extend_triage.md` — do the top rows justify the hypothesis?
5. `analysis/{case}/vllm_crosscheck.md` — does vLLM evidence agree or falsify?
6. `traces/{case}/…` — only when challenging a specific row of a triage table

### 8.4 Inter-phase flow

| Produced in | Consumed by | As what |
|---|---|---|
| `datasets/qwen3vl8b/case*.jsonl` | Phase 1, Phase 2, Phase 5 | Byte-identical workload |
| `experiments/phase1/summary.md` | Phase 2 | Input to the Decision Rule |
| `experiments/phase2/selected_cases.md` | Phase 3 | Sole gate into profiling |
| `traces/{case}/sglang_{mapping,formal}` | Phase 4 | Two-trace triage input |
| `traces/{case}/vllm/{prefill_like,decode_like}` | Phase 4 | Single-trace falsification |
| `analysis/hypotheses.md` | Phase 5 | Source of hypotheses to validate |
| `experiments/phase5/{h}/summary.md` | Phase 5 close-out | Updates confidence in `hypotheses.md` |

---

## 9. Phases

### Phase 0 — Environment & Functional Equivalence (≤1 day)

> ✅ **Status: complete + PASS.** Executed on GPU 0 with the current stack; canonical artifacts in
> `experiments/qwen3vl8b/phase0/` (equivalence.md, sglang/vllm outputs, model_files_sha256.txt, tier_a_results.txt).
> The protocol below is the method; the constants reflect the actual execution
> except where noted (Phase 0/1 used GPU **0** and the framework versions in §5).

**Goal.** Establish that both servers are comparable — weights, tokenizer, vocab identical; decoding behavior equivalent under a realistic equivalence standard.

**Operational constants.**
- GPU: **`CUDA_VISIBLE_DEVICES=0`** (Phase 0/1; later phases used GPU 7 then GPU 1 — see §0)
- Model snapshot: `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`
- `HF_HUB_OFFLINE=1` — model is fully cached; no network call or token needed
- Servers run **sequentially** (SGLang first, then vLLM after shutdown) so each gets full GPU memory and there is no cross-process interference during the equivalence test. Phase 1 benchmarks follow the same pattern.

**Actions.**

1. Launch SGLang (background, log to `logs/phase0/sglang_server.log`):
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   SGLANG_KERNEL_API_LOGLEVEL=1 \
   SGLANG_KERNEL_API_LOGDEST=logs/phase0/sglang_%i.log \
   python3 -m sglang.launch_server \
     --model-path <snapshot_path> \
     --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer
   ```
   Wait for `server is fired up` in log. Record FlashInfer version, chunked-prefill default, idle GPU memory.
2. Run equivalence tiers (Tier A tokenizer check + Tier B greedy outputs). Save outputs to `experiments/qwen3vl8b/phase0/sglang_outputs.json`.
3. Kill SGLang. Launch vLLM:
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model <snapshot_path> --dtype bfloat16 \
     --port 30001 --tensor-parallel-size 1
   ```
   Wait for `Application startup complete`. Record attention backend, idle GPU memory.
4. Run same Tier B prompts against vLLM. Save to `experiments/qwen3vl8b/phase0/vllm_outputs.json`. Compare.
5. Record all findings in `experiments/qwen3vl8b/env_snapshot.md` and `experiments/qwen3vl8b/phase0/equivalence.md`.

**Equivalence framework.** Byte-identical greedy decoding across frameworks is not a realistic target — attention kernel, matmul tiling, and reduction order all legitimately differ, and bf16 accumulation order compounds the divergence. The correct standard is tiered:

| Tier | Check | Threshold | Disposition if fail |
|---|---|---|---|
| **A — Blocker** | Tokenizer byte-equality on 5 probe strings (ASCII, CJK, emoji, code, long) | Exact match | **Stop.** The runs are incomparable. Fix before continuing. |
| **A — Blocker** | Model weight hash (SHA-256 of each `*.safetensors` file loaded by each server) | Identical | **Stop.** Wrong snapshot. |
| **A — Blocker** | Vocab size, EOS/BOS/PAD ids, `max_position_embeddings` | Identical | **Stop.** Config drift. |
| **A — Blocker** | Chat-template rendered bytes on a fixed system+user pair | Identical | **Stop.** Template mismatch makes all latency numbers misleading. |
| **B — Correctness** | Top-1 next-token on ≥3 greedy prompts (short / medium / long) | Match on all 3 | **Stop.** Different top-1 at token 0 ⇒ weights loaded differently, wrong dtype, or RoPE misconfig. |
| **B — Correctness** | Top-5 logprob overlap on first token, averaged over the 3 prompts | Jaccard ≥ 0.8 | **Investigate** before Phase 1 — likely sampler or normalization divergence. |
| **B — Correctness** | Coherent continuation at 256 output tokens under greedy sampling | Human-readable, on-topic, no degenerate loops | **Investigate.** Coherent but byte-divergent is acceptable. |
| **C — Informational** | Token-level edit distance of full 256-token continuations | Logged, not gated | Report in `phase0_equivalence.md` |
| **C — Informational** | Output length under `ignore_eos=false` | Logged | Report |

**Rule.** A Tier-A failure halts the plan. A Tier-B failure at the *first token* halts the plan. A Tier-B failure only at token ≥ 2 is expected bf16 drift and proceeds with a note in every downstream conclusion ("greedy outputs diverge after first token — cross-framework output comparisons below are semantic, not token-level").

**Downstream effect on profiling.** Because token-level output equivalence is not required, we do not gate Phase 3 on it. What *does* matter for profiling validity: Tier-A identity (so both frameworks execute the same underlying model) and workload byte-identity (§6.1). Profiling under these conditions is methodologically sound even if produced tokens differ.

**Outputs.** `experiments/qwen3vl8b/env_snapshot.md`, `experiments/qwen3vl8b/phase0/equivalence.md`.

**Risks.** Qwen3-VL may not be fully supported at pinned versions; fall back to `Qwen3-8B` and record the substitution. Vision tower may load even for text-only; record idle memory.

---

### Phase 1 — Minimal Fair Baseline (1 day)

> ✅ **Status: complete.** Executed on GPU 0 with the scripts at
> `experiments/qwen3vl8b/phase1/scripts/` (greedy via `--extra-request-body`, datasets under
> `datasets/qwen3vl8b/`). Results in §0 and §15. The protocol below is the method; the commands
> show the actual form (note the dataset-generator caveat)
> and the custom special-token-safe generator.

**Goal.** Produce one head-to-head table on a small deliberate matrix — clean enough to believe.

**Case matrix.**

| Case | Prompt len | Output len | Concurrency |
|---|---|---|---|
| A. Latency-bound short | 128 | 128 | 1 |
| B. Latency-bound long-prefill | 2048 | 128 | 1 |
| C. Batched throughput | 512 | 128 | 16 |
| D. Decode-heavy | 512 | 512 | 16 |

4 cases × 2 frameworks = 8 runs. Each ≥120 s steady-state after warmup, repeated 3× with independent warmups; take median, reject if stdev/median > 5 %.

**Actions.**

1. Generate byte-identical datasets using the custom generator (do **not** use `sglang.auto_benchmark convert --kind random` — it samples multimodal special tokens that trigger Qwen3-VL OOM):
   ```bash
   HF_HUB_OFFLINE=1 python3 experiments/phase1/scripts/gen_datasets.py
   # Samples token IDs 0–151642 only; outputs datasets/qwen3vl8b/case{A,B,C,D}.jsonl
   # SHA-256 logged to experiments/phase1/raw/dataset_sha256.txt
   ```

2. Launch servers sequentially (one at a time; do not co-run).

   SGLang:
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   SGLANG_KERNEL_API_LOGLEVEL=1 \
   SGLANG_KERNEL_API_LOGDEST=logs/phase1/sglang_%i.log \
   python3 -m sglang.launch_server \
     --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
     --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer \
     2>&1 | tee logs/phase1/sglang_server.log
   ```

   vLLM (after shutting down SGLang):
   ```
   CUDA_VISIBLE_DEVICES=6 HF_HUB_OFFLINE=1 \
   /opt/miniconda3/envs/vllm/bin/python -m vllm.entrypoints.openai.api_server \
     --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b \
     --dtype bfloat16 --port 30001 --tensor-parallel-size 1 \
     2>&1 | tee logs/phase1/vllm_server.log
   ```

3. Run the two frameworks against the same JSONL (`ignore_eos` is the **default** in bench_serving — do not pass `--disable-ignore-eos` unless intentionally testing variable-length output):
   ```
   python -m sglang.bench_serving --backend sglang-oai \
     --base-url http://127.0.0.1:30000 \
     --dataset-name autobench --dataset-path datasets/qwen3vl8b/caseC_batched.jsonl \
     --max-concurrency 16 --seed 1 --warmup-requests 30
   python -m sglang.bench_serving --backend vllm \
     --base-url http://127.0.0.1:30001 \
     --dataset-name autobench --dataset-path datasets/qwen3vl8b/caseC_batched.jsonl \
     --max-concurrency 16 --seed 1 --warmup-requests 30
   ```
4. Write `experiments/phase1/raw/{case}_{framework}_{rep}.json` per run with `meta.json` (versions, attn backend, dataset sha, seed).

**Skill usage.**

- `sglang-auto-benchmark` → `convert`/`validate` only (not `run`; it cannot drive vLLM).
- `debug-cuda-crash` → L1 passive, SGLang side only.
- `sglang-torch-profiler-analysis` → **not used.**

**Outputs.** `datasets/case{A,B,C,D}.jsonl`, `experiments/phase1/raw/*.json`, `experiments/phase1/summary.md` (4×2 table: TTFT p50/p95, TPOT, output tok/s, request throughput, fairness notes), `logs/phase1/*.log`.

**Success criteria.** 8 runs complete; stdev/median ≤ 5 % across repeats; per-cell fairness notes written.

**Risks.** Graph-capture overhead in first run (fix with longer warmup); OOM on one side (reduce equally and document); overhead-dominated ratios in Case A (flag as overhead, not as framework gap).

---

### Phase 2 — Identify Informative Cases 

> ✅ **Status: complete** (2026-05-22, GPU 7, 0 failures). Executed with the scripts at
> `experiments/qwen3vl8b/phase2/scripts/` (5-variant Case A sweep, 4-variant Case B sweep,
> C/D warmup variance gate, vLLM B/C recheck). Results summarized in §0 and detailed in §15
> . The decision rule and methodology below are the **method**; the earlier execution narrative
> has been removed (earlier round). **Outcome:** all 4 cases promote
> to Phase 3 — see the §0 shortlist.

**Goal.** Given Phase-1 evidence (TTFT gap is universal, TPOT at parity), decide which cases enter
Phase-3 profiling and with what shaping. This was answered as follows:

1. **Case A:** the TTFT floor is **partly configurational** — `--disable-overlap-schedule` removed ~10% (gap 4.89×→1.56×). The residual is the Phase-3 target. *(This differs from run1, where the floor looked fully structural; the current SGLang build responds to the overlap flag.)*
2. **Case B:** `chunk_off` did not beat default beyond the 5% threshold → default base. Both SGLang and vLLM are bimodal here → all cross-framework Case B conclusions carry ceiling M.
3. **Case C:** vLLM stabilized at warmup=300 (CV 1.9%). SGLang failed the 5% CV gate at W30/W100/W300 (12.5/15.2/14.9%) but a follow-up **W500 probe passed cleanly (CV 2.9%, 249.1 ms)**. The noisy W100/W300 "SGLang faster / 0.79× reversal" was an under-warmup artifact; clean W500 data shows SGLang **~1.32× slower** (matches Phase-1 W30). **Cleanly profilable at warmup=500.**
4. **Case D:** clean at warmup=30 (CV 3.3%); residual gap 1.09×. No bimodal tail this round.

#### Decision rule (reusable methodology — applied per case)

| Gap size (median) | TTFT CV (both frameworks) | Action |
|---|---|---|
| < 5% | any | Drop, or reshape workload (more prompt / more concurrency). |
| 5–15% | ≤10% | Run tier-1 shaping sweep. Promote to Phase 3 only if sweep cannot close the gap. |
| > 15%, both CVs ≤10% | ≤10% | Run shaping sweep *before* promoting. If gap survives any flag combo, promote as **structural**. |
| > 15%, either CV > 10% | > 10% | Run **variance-reduction** sweep first. Re-evaluate gap after. Do not profile on noisy data (§14 anti-pattern). |

#### Shaping design (as executed)

Custom Python orchestration scripts under `experiments/qwen3vl8b/phase2/scripts/`
(`run_phase2_caseA.py` + `resume_phase2_caseA.py`, `run_phase2_caseB.py`, `run_phase2_variance.py`,
`run_phase2_BCD.sh` chainer, `summarize_phase2.py`). Direct `bench_serving` against
`datasets/qwen3vl8b/case*.jsonl`. Greedy `{"temperature":0,"top_p":1}`, `--output-details`,
GPU 7, serial servers, GPU freed (<2000 MiB) between every server.

- **Case A — scheduler/dispatch floor (5 variants, screen 1 rep → finalist 3 reps):** `default`,
  `no_overlap` (`--disable-overlap-schedule`), `stream8` (`--stream-interval 8`),
  `chunk_off` (`--chunked-prefill-size -1`), `chunk_64` (`--chunked-prefill-size 64`).
  → **`no_overlap` won** (19.6 ms vs default 21.8 ms, ~10%); `stream8`/`chunk_off` within 5% of
  default (dropped); `chunk_64` 2.4× worse (eliminated).
- **Case B — chunked-prefill sweep (4 variants):** `default`, `chunk_off` (-1),
  `chunk_1024`, `chunk_512`. → smaller chunks were strictly worse (80.6 / 91.6 ms); `chunk_off`
  beat default by only 3.2% (< 5%) → **default base.** Both frameworks bimodal (ceiling M).
- **Cases C & D — client-side warmup variance gate:** warmup ∈ {30, 100, 300} × 3 reps (escalate to
  5 reps if CV in 5–10%; report W500 need if W300 still fails — **do not** extend indefinitely).
  → **D passed at W30** (CV 3.3%); **C never passed** (12.5 / 15.2 / 14.9%).
- **vLLM recheck (B + C, warmup=300, 5 reps):** Case B still bimodal (CV 85.9% → ceiling M);
  Case C stable (CV 1.9%, 189 ms — Phase-1 Case C CV 5.8% was warmup-starvation).

#### Artifacts (Phase 2)

| Path | Role |
|---|---|
| `experiments/qwen3vl8b/phase2/scripts/` | All orchestration + summarizer scripts |
| `experiments/qwen3vl8b/phase2/raw/` | Per-(case × variant/warmup × rep) bench JSON + meta + `*_result.json` rollups |
| `experiments/qwen3vl8b/phase2/summary.md` | Full shaping + variance-gate + vLLM-recheck tables |
| `experiments/qwen3vl8b/phase2/selected_cases.md` | **Phase-3 entry gate** — per-case config, warmup, reps, residual gap, CV, ceiling |
| `logs/qwen3vl8b/phase2/` | Server logs (one per server lifetime) + orchestrator logs + L1 kernel trails |

#### Skill usage

- Custom orchestration scripts (not `llm-serving-auto-benchmark run`) — direct `bench_serving` calls for full control over server flags and vLLM compatibility.
- `debug-cuda-crash` → L1 passive (`SGLANG_KERNEL_API_LOGLEVEL=1`) on all SGLang server launches.
- `llm-torch-profiler-analysis` → **not used.** No interpretation in Phase 2.

#### Exit criteria

1. **Met.** Case A has a shaped SGLang config (`--disable-overlap-schedule`) that holds a stable, low-CV residual gap (1.56×); Case B sweep found no winner beyond default and is promoted on default.
2. **Met.** Case D reached CV ≤5% (promote, W30). Case C failed the gate at W30/W100/W300 but the **W500 probe passed cleanly (CV 2.9%)** — promoted **clean** at warmup=500/5 reps. The earlier high-CV/marginal status is resolved.
3. **Met.** vLLM baselines documented: Case C clean (CV 1.9%), Case B ceiling M (CV 85.9%); Case A/D compared to stable Phase-1 vLLM p50.
4. **Met.** `experiments/qwen3vl8b/phase2/selected_cases.md` locks the Phase-3 protocol for all 4 promoted cases.



---

### Phase 3 — Profiling & Trace Collection (✅ complete)

> ✅ **Status: complete** (2026-05-22/23, GPU 1, serial servers, GPU freed after each).
> Collection only — no interpretation. Detail: `experiments/qwen3vl8b/phase3/summary.md` +
> `extend_supplement_summary.md`. **Actual outcome:**
> - The **initial** collection (`run_phase3_collect.py`, commit `c6ec1df`) captured SGLang **DECODE
>   stage only** — the `--profile-by-stage` window armed during in-flight decode, so no EXTEND landed.
>   vLLM `prefill_like` + `decode_like` complete for all 4 cases.
> - The **EXTEND supplement** (`run_phase3_extend.py`, commit `8f41bd3`) fixed this by driving a
>   prefill-only load (`/generate max_new_tokens=1`): EXTEND **mapping** captured for **all 4 cases**,
>   EXTEND **formal** for **A/C/D**.
> - **Case B graph-on EXTEND formal could not be captured** after 8 attempts (incl. a 5-strategy retry
>   `d822bf3`): graph-on `--profile-by-stage` prefill capture does not trigger for Case B (small window
>   → DECODE, large window → empty). **Caveat accepted** — Case B's graph-off mapping EXTEND carries the
>   kernel→source mapping and Case B already has ceiling M.
> - mechanism note: vLLM 0.21.0 dropped `VLLM_TORCH_PROFILER_DIR`; the profiler was enabled via
>   `--profiler-config '{"profiler":"torch","torch_profiler_dir":"<abs>"}'`. SGLang `sglang.profiler`
>   needs concurrent load (it profiles whatever forward steps run), so a `bench_serving` load ran during
>   each capture. The method below is the canonical protocol; these are the adaptations.

**Goal.** For each selected case, produce a clean SGLang mapping+formal trace pair *and* a vLLM trace pair shaped to permit stage-level comparison. No interpretation here.

#### 3.1 SGLang traces

Two-trace protocol:

- **Mapping (graph-off)**: launch with `--disable-cuda-graph --disable-piecewise-cuda-graph --attention-backend flashinfer`; drive via `sglang.profiler --url ... --num-steps 8 --profile-by-stage --output-dir traces/{case}/sglang_mapping`.
- **Formal (graph-on)**: re-launch with graph capture enabled; warm up to stable batch shape; profile `traces/{case}/sglang_formal`.

`--profile-by-stage` is non-optional — it is what lets Phase 4 triage separate EXTEND from DECODE.

#### 3.2 vLLM traces — strengthened protocol

vLLM's profiling does not emit a clean mapping/formal pair, but it does not need to be left as "best effort". The protocol below produces a falsifiable vLLM artifact per case.

1. **Enable the profiler at server start.** Launch with `VLLM_TORCH_PROFILER_DIR=traces/{case}/vllm`. vLLM exposes `/start_profile` and `/stop_profile` HTTP endpoints that open/close one profile window each.
2. **Stage separation by workload shaping.** vLLM has no `--profile-by-stage`; we get the same separation naturally by driving two distinct profile windows per case:
   - **`prefill_like/`** — window opened immediately before sending `N=8` requests of the case's prompt length at concurrency 1, closed immediately after the first token of the last request. The window is then dominated by prefill kernels because decoding contribution is one token per request.
   - **`decode_like/`** — warm the server to a stable steady-state batch at the case's target concurrency, open the window after ≥30 s of steady decoding, capture ~5 s, close.
3. **Category alignment.** Both frameworks' traces are classified by the same regex rules defined in `analysis/category_regex.md` — attention, gemm, communication, norm, quantization, memory, scheduler. Rules are authored once, applied symmetrically; any kernel name not covered is accumulated in an `uncategorized` bucket which must shrink to < 2 % of GPU time before a breakdown is published.
4. **Hotspot-to-source mapping.** vLLM does not yield Python source backing for kernels the way SGLang's mapping trace does. We compensate with a curated static map `analysis/vllm_source_map.md` populated incrementally: every time a vLLM kernel crosses the 1 % GPU-time share threshold in any triage, its name is added with a manually-verified path into `/opt/miniconda3/envs/vllm/lib/python3.12/site-packages/vllm/…`. This trades completeness for correctness — the map covers exactly the kernels we cite, nothing more.
5. **Role in reasoning — falsification, not symmetry.** vLLM traces are used to *test* SGLang-side claims, not to mirror them:
   - Claim of the form *"vLLM overlaps X with Y"* requires X and Y to appear on distinct CUDA streams in the vLLM trace. If they do not, the claim is downgraded from H to at most M.
   - Claim of the form *"vLLM omits kernel Z"* requires Z to be absent from the vLLM kernel table at ≥ 0.5 % share. Otherwise the claim is rejected.
   - Claim of the form *"SGLang kernel K is slower per call than vLLM's equivalent K'"* requires matching K and K' by category (via `category_regex.md`) and comparable invocation count; divergence in invocation count is itself a finding.

#### 3.3 Crash safety

All SGLang runs in Phase 3: `SGLANG_KERNEL_API_LOGLEVEL=1`, `LOGDEST=logs/phase3/sglang_%i.log`. On any crash, re-run only the affected step at L3 (or L10 with `--disable-cuda-graph` if offline repro needed). Do not abandon the case — isolate the trigger.

**Skill usage.**

- `sglang-torch-profiler-analysis` → collection driver only (script calls `sglang.profiler`). No triage.
- `debug-cuda-crash` → L1 passive, escalated only on actual crash.
- `sglang-auto-benchmark` → **not used.**

**Outputs (completed).**
- `traces/qwen3vl8b/{case}/sglang_mapping/`, `sglang_formal/` — DECODE-stage (graph-off / graph-on).
- `traces/qwen3vl8b/{case}/sglang_extend_mapping/`, `sglang_extend_formal/` — EXTEND-stage supplement.
- `traces/qwen3vl8b/{case}/vllm/{prefill_like,decode_like}/`.
- `experiments/qwen3vl8b/phase3/summary.md`, `extend_supplement_summary.md`, `metadata/`.
- `logs/qwen3vl8b/phase3/*.log`.

**Success criteria.**
- ✅ SGLang **DECODE** (mapping + formal) complete for all 4 cases.
- ✅ vLLM **prefill_like + decode_like** complete for all 4 cases.
- ✅ SGLang **EXTEND mapping** complete for all 4 cases.
- ⚠️ SGLang **EXTEND formal** complete for **A/C/D**; **Case B missing** — accepted with documented
  caveat (graph-off mapping EXTEND suffices for kernel→source; ceiling M already applies to B).
- ✅ Phase 4 can start.

---

### Phase 4 — Trace Interpretation & Synthesis (✅ complete)

> **Outcome (completed 2026-05-23, offline triage — no GPU).** All 4 cases triaged via
> `llm-torch-profiler-analysis` `triage` (two-trace SGLang per stage + single-trace vLLM per window),
> each with mandatory catalog lookup. **A/C/D:** EXTEND + DECODE + vLLM cross-check all successful.
> **B:** DECODE + vLLM cross-check successful; **EXTEND unavailable** — the original graph-off mapping
> gz was corrupt (`EOFError`) and 3 GPU-1 re-collect attempts failed (truncated EXTEND / valid-DECODE ×2,
> incl. `--disable-radix-cache`); root cause = prefix-cache + `--profile-by-stage` long-prefill window
> limit (same family as the prior 8-attempt graph-on miss). Quarantined attempt dirs preserved
> (`CORRUPT_/TRUNC_/DECODEONLY_`), trace deletions not staged. All Case B conclusions ≤ ceiling M.
>
> **Headline:** both frameworks are GEMM-bound by the same `nvjet_sm90_*` FP8 family (72–86% per stage)
> → GEMM is shared cost, not the differentiator. The differentiator is dispatch/compile (SGLang eager
> `aten::mm` vs vLLM torch.compile/CUDA-graph). The gap is a first-token fixed-overhead effect (Phase-1
> TPOT parity + Case D 1.09× both confirm). Hypotheses H1–H4 in §15 + `reports/qwen3vl8b/03_profiling_analysis.md`.
>
> **Outputs.** Per case `analysis/qwen3vl8b/{case}/{extend_triage,decode_triage,breakdown,vllm_crosscheck,preliminary_observations}.md`
> (+ `*_raw.txt`); global `analysis/qwen3vl8b/{hypotheses,ranked_recommendations}.md`; shared
> `analysis/category_regex.md`; narrative `reports/qwen3vl8b/03_profiling_analysis.md`.

**Goal.** Convert traces into ranked evidence-backed hypotheses.

**Skill usage.**

- `sglang-torch-profiler-analysis` → two-trace `triage` on SGLang per (case × {EXTEND, DECODE}); single-trace `triage` on vLLM per (case × {prefill_like, decode_like}); mandatory catalog lookup.
- `debug-cuda-crash` → consulted only if a trace reveals NaN/Inf — then L5 on the targeted reproducer.
- `sglang-auto-benchmark` → not used.

**Step 1 — SGLang triage.**
```
python analyze_sglang_torch_profile.py triage \
  --mapping-input traces/{case}/sglang_mapping \
  --formal-input traces/{case}/sglang_formal \
  > analysis/{case}/decode_triage.md
```
(repeat per stage).

**Step 2 — Category breakdown.** Apply `analysis/category_regex.md` to the formal trace; emit `analysis/{case}/breakdown.md` with the compute / memory / comm / scheduler split. The same regex is then applied to the vLLM traces.

**Step 3 — vLLM single-trace triage and falsification.** Single-trace triage on each vLLM window produces a kernel table with catalog-backed pattern matches where applicable (most patterns will not match — that is fine; we are using vLLM triage to probe SGLang claims, not to extract vLLM recommendations). Results written to `analysis/{case}/vllm_crosscheck.md`, organized by the SGLang hypothesis each row tests.

**Step 4 — Hypothesis construction.** Every hypothesis uses this schema:
```
**Hypothesis**: <short title>
- Observation: <kernel/stage, time share, Python source pointer>
- vLLM evidence: <corroborates | falsifies | silent>, pointer to vllm_crosscheck row
- Catalog status: <existing disabled path | in-flight PR | truly new, similarity H/M/L>
- Impact: <estimated latency or throughput delta if closed>
- Evidence: <triage row refs, breakdown refs>
- Confidence: <H | M | L>   (H requires vLLM corroboration AND catalog-backed classification OR a disabled-path finding)
- Fairness dependence: <Controlled | Measured | Framework-intrinsic>
- Next step: <validation sweep, code pointer, or PR draft>
```

A hypothesis missing any field is inadmissible. The `Fairness dependence` field determines the confidence ceiling per §6.4.

**Outputs.** Per case: `extend_triage.md`, `decode_triage.md`, `breakdown.md`, `vllm_crosscheck.md`. Global: `analysis/hypotheses.md`, `analysis/ranked_recommendations.md` (top 5–10, sorted by `confidence × impact × feasibility`).

**Success criteria.** Every hypothesis has specific kernel name + source pointer + vLLM evidence + catalog classification. ≥1 H-confidence hypothesis per selected case. Ranking logic explicit.

---

### Phase 5 — Hypothesis Validation Sweeps (optional, 1 day per hypothesis)

**Goal.** For the top 2–3 hypotheses that can be tested with flag-level changes, confirm or refute the mechanism before any PR.

**Skill usage.**

- `sglang-auto-benchmark` → `run` tier 2, ≤10 candidates, resumable. Dataset is the Phase-1 `datasets/{case}.jsonl` so results are directly comparable to the baseline.
- `sglang-torch-profiler-analysis` → optional re-triage on the winning candidate to confirm the mechanism, not just the metric, moved.
- `debug-cuda-crash` → L1 passive.

**Outputs.** Per hypothesis: `experiments/phase5/{hypothesis}/{live_results.jsonl, results.jsonl, summary.md}` and optional `traces/{case}/sglang_phase5_{hypothesis}/`. Updates `analysis/hypotheses.md` confidence column.

---

## 10. Crash / Debug Workflow (transverse)

| Situation | Setting | Rationale |
|---|---|---|
| Normal runs (Phase 1 / 2 / 3 / 5) | `LOGLEVEL=1`, `LOGDEST=logs/{phase}/sglang_%i.log` | Negligible cost; free crash trail |
| Crash observed | Re-run crashing case with `LOGLEVEL=3` | Shapes/dtypes/device at crash boundary |
| NaN/Inf in trace or Phase-0 divergence | `LOGLEVEL=5` on targeted reproducer | Tensor stats at boundary |
| Offline reproducer needed | `LOGLEVEL=10` + `DUMP_DIR` + `DUMP_INCLUDE='sglang.custom_op.*'` + `--disable-cuda-graph` | Crash-safe input dump |

**Rule.** A crash is a finding, not an abort. Capture, isolate the trigger batch shape, route around, continue.

---

## 11. Decision Gates

| Condition | Action |
|---|---|
| Phase 0 Tier-A fail | Halt. Fix weights / tokenizer / template before continuing. |
| Phase 0 Tier-B fail at first token | Halt. Likely weight-load or RoPE config issue. |
| Phase 0 Tier-B fail only at token ≥ 2 | Proceed; annotate downstream comparisons as semantic-level. |
| Phase 1 all 4 gaps < 5 % | Reshape workload; longer prompts / higher concurrency before concluding "no gap". |
| Phase 1 gap real but stdev/median > 5 % | Increase warmup, pin GPU governor, re-run. Never profile on noisy data. |
| Phase 2 shaping closes the gap | Gap was configurational. Document and pick a harder case. |
| Phase 3 SGLang crash on selected case | Debug-crash flow. Data-specific → drop the prompt; structural → treat as a separate finding. |
| Phase 4 top hotspot is comm-bound at TP=1 | Suspicious (no NCCL on 1 GPU). Check attention backend and dispatch paths — may be mislabeled CPU overhead. |
| Phase 4 top hotspot is scheduler-bound | GPU-only profiling is insufficient; add `py-spy` + CPU-side torch profiler on scheduler process. |
| Phase 4 hotspot maps to a fuse path SGLang already has | Likely gated off. Low-cost, high-confidence recommendation to flip the gate. |
| Phase 4 yields no H-confidence hypothesis | Do not synthesize speculative recommendations. Expand Phase 3 (more iterations, more cases). |
| vLLM evidence contradicts an SGLang claim | Downgrade hypothesis confidence; keep the raw observation as a Phase-5 candidate only if the mechanism can be isolated without the vLLM comparison. |

---

## 12. Deliverables

Ordered by reviewer reading priority.

1. `reports/05_recommendations.md` — top 5–10 actionable directions for SGLang, ordered by `confidence × impact × feasibility`.
2. `reports/04_hypotheses.md` — structured hypotheses with vLLM evidence + catalog status.
3. `reports/03_profiling_analysis.md` — per-case triage + breakdown synthesis.
4. `reports/02_benchmark_table.md` — Phase-1 4×2 table with fairness notes.
5. `reports/01_experiment_summary.md` — environment, versions, fairness tier assignments, equivalence result.
6. All backing `analysis/**` and `experiments/**` artifacts.
7. `traces/**` preserved for independent re-analysis.

**End condition.** `reports/05_recommendations.md` exists and each of its top 3 entries is concrete enough that an SGLang engineer can open a PR without further investigation.

---

## 13. Skill Usage Quick-Reference

Authoritative definitions in §7. If this table disagrees, §7 wins. Current installed skill names
(see §7.0): auto-benchmark = `llm-serving-auto-benchmark`, profiler-analysis = `llm-torch-profiler-analysis`,
debug-cuda-crash = `debug-cuda-crash`.

| Phase | auto-benchmark | profiler-analysis | debug-cuda-crash |
|---|---|---|---|
| 0 | — | — | L1 during server smoke |
| 1 | `convert` + `validate` | — | L1 passive |
| 2 | `run` tier 1, ≤4 candidates, 1 case | — | L1 passive |
| 3 | — | collection driver (`--profile-by-stage`); no triage | L1 passive |
| 4 | — | `triage` 2-trace (SGLang) + 1-trace (vLLM) + **catalog lookup** | L5 only if NaN/Inf suspected |
| 5 | `run` tier 2, resumable, hypothesis-scoped | optional re-triage on winner | L1 passive |

Never invert a row. Auto-benchmark does not read kernels; profiler-analysis does not choose flags; debug-cuda-crash does not explain slowdowns.

---

## 14. Anti-Patterns

- ❌ "vLLM is faster overall" with no mechanistic explanation.
- ❌ Attributing a slow kernel to a design flaw before checking the implementation (and the catalog).
- ❌ Recommending something SGLang already ships or has an in-flight PR for.
- ❌ Confusing benchmark noise with a real difference (stdev/median > 5 %).
- ❌ Proposing a refactor without a validation path.
- ❌ Publishing an H-confidence hypothesis whose fairness dependence is `Measured` and unvalidated.
- ❌ Using vLLM traces only to mirror SGLang findings rather than to falsify them.
- ❌ Token-level equivalence as a gate for cross-framework profiling.

---

## 15. Results

### Phase 0 — Functional Equivalence (completed 2026-05-21)

- GPU 0; SGLang `0c8049d9b` (flashinfer text) vs vLLM `0.21.0` (FlashAttention v3); torch 2.11.0+cu130 both.
- Tier A **PASS** — same snapshot `0c351dd`, safetensors sha256-verified; tokenizer/config/ChatML identical.
- Tier B **EXACT** byte-identical greedy outputs on all 3 prompts.
- **Verdict: PASS** → cleared for Phase 1. Canonical artifacts: `experiments/qwen3vl8b/phase0/`.

### Phase 1 — Minimal Fair Baseline (completed 2026-05-22)

- 24 runs (4 cases × 2 frameworks × 3 reps), **error rate 0% on every run**. GPU 0, serial servers;
  greedy (`temperature=0, top_p=1`, `ignore_eos` default); both frameworks torch 2.11.0+cu130 / CUDA 13.0.
- Datasets: `datasets/qwen3vl8b/case{A,B,C,D}.jsonl` (text-only, special-token-safe, SEED=1; SHA-256 logged). Old root `datasets/case*.jsonl` were removed in the restructure.

| Case | SGLang TTFT p50/p95/p99 (ms) | vLLM TTFT p50/p95/p99 (ms) | p50 ratio | TPOT | Throughput | CV / variance |
|---|---|---|---|---|---|---|
| A 128→128 c1 | 61.8 / 66.1 / 66.4 | 12.6 / 17.9 / 18.0 | **4.89×** | 0.97× ≈ | 0.96× | low (1.5–4.4%) |
| B 2048→128 c1 | 66.7 / 70.6 / 71.6 | 20.8 / 25.4 / 26.0 | **3.20×** | 0.97× ≈ | 0.97× | SGLang low; **vLLM bimodal cv 114% → ceiling M** |
| C 512→128 c16 | 247.5 / 255.4 / 257.4 | 187.9 / 196.7 / 209.1 | **1.32×** | 0.99× ≈ | 0.91× ↓ | SGLang p50 cv 9.4% (gate) |
| D 512→512 c16 | 253.0 / 257.3 / **390.6** | 189.7 / 196.2 / 222.4 | **1.33×** | 1.00× ≈ | 0.98× | **SGLang p99 cv 47% (bimodal tail)** |

- **TTFT is the only gap; TPOT/throughput at parity** (0.91–1.01×).
- **Fixed overhead / prefill cheap holds:** A→B prompt 16× longer adds only **+4.9 ms** SGLang TTFT.
- **Direction matches run1**, magnitudes differ (Case A 4.89× vs run1 3.89×; vLLM Case B bimodal again; Case D bimodal tail again) — **numbers not interchangeable**.
- **Confidence ceilings:** vLLM Case B comparisons → M (bimodal); any attention-kernel finding → M (FlashInfer vs FA3 backend mismatch).
- Phase 2 entry: A primary, B primary (vLLM ceiling M), C secondary (variance gate), D likely drop (bimodal). See §0.
- Artifacts: `experiments/qwen3vl8b/phase1/summary.md`, `phase1/raw/`, `phase1/scripts/`; logs `logs/qwen3vl8b/phase1/`.

### Phase 2 — Shaping / Variance Gate (completed 2026-05-22)

- GPU **7**, serial servers (GPU freed <2000 MiB between every server), greedy
  (`temperature=0, top_p=1`), `--output-details`. **0 failed requests across the entire phase.**
- Artifacts: `experiments/qwen3vl8b/phase2/summary.md`, `phase2/selected_cases.md`,
  `phase2/raw/`, `phase2/scripts/`; logs `logs/qwen3vl8b/phase2/`.

| Case | Winner config | SGLang TTFT p50 (CV) | vLLM TTFT p50 (CV) | Residual gap | Phase-3 protocol |
|---|---|---|---|---|---|
| A 128→128 c1 | **`--disable-overlap-schedule`** | 19.6 ms (3.2%) | 12.6 ms (Phase 1) | **1.56×** | warmup 30, 3 reps |
| B 2048→128 c1 | default | 30.3 ms (**68.4%** ⚠) | 21.5 ms (**85.9%** ⚠, ceiling M) | 1.41× | warmup 300, 5 reps |
| C 512→128 c16 | default | **249.1 ms (2.9%, W500)** | 189.0 ms (1.9%) | **1.32×** | warmup 500, 5 reps — clean |
| D 512→512 c16 | default | 206.2 ms (3.3%) | 189.7 ms (Phase 1) | 1.09× | warmup 30, 3 reps |

- **Case A shaping:** screen p50 — `no_overlap` 19.5 / `default` 22.2 / `stream8` 22.2 / `chunk_off` 22.5 / `chunk_64` 53.9 ms. Finalist (3 reps): `no_overlap` median 19.6 ms (CV 3.2%) vs `default` 21.8 ms (CV 1.7%). **The overlap scheduler costs ~10% at c=1; the Phase-1 4.89× gap collapses to 1.56×.** (run1 had called this floor fully structural — the current build responds to the flag.)
- **Case B shaping:** screen p50 — `chunk_off` 62.8 / `default` 64.9 / `chunk_1024` 80.6 / `chunk_512` 91.6 ms. `chunk_off` < 5% better → default. Finalist reps 64.3 / 30.3 / 26.9 ms (CV 68.4%) — **SGLang itself is bimodal here.** vLLM recheck (w=300, 5 reps): first rep high, rest ~21.5 ms, CV 85.9% — **bimodality not a warmup artifact → ceiling M on all Case B cross-framework claims.**
- **Case C variance gate + W500 probe:** SGLang W30/W100/W300 CV = 12.5 / 15.2 / 14.9% — failed the 5% gate. A follow-up **W500 probe (5 reps, GPU 1, 0 failures) passed at CV 2.9%**, stable median **249.1 ms**. vLLM recheck stable (CV 1.9%, 189 ms). The W100/W300 "SGLang faster / 0.79× reversal" was an **under-warmup artifact**; at stable warmup SGLang is **~1.32× slower** (matches Phase-1 W30). vLLM is warmup-insensitive (187.9→189.0 across W30→W300), so this is a sound stable reference; a strict same-warmup vLLM W500 is only needed for a "SGLang faster" claim (no longer applicable). **Case C cleanly profilable** at warmup=500. Probe artifacts: `phase2/raw/caseC_sglang_w500_rep*.json`, `phase2/raw/caseC_w500_result.json`.
- **Case D variance gate:** passed at W30 (CV 3.3%, 206.2 ms); residual 1.09×. The Phase-1 p99 bimodal tail did not reappear.
- **Phase-3 shortlist:** A (high priority) · B (ceiling M + extra reps) · C (clean at W500, warmup 500/5 reps) · D (lower priority). Locked protocol: `experiments/qwen3vl8b/phase2/selected_cases.md`.

### Phase 3 — Trace Collection (completed 2026-05-22/23)

- GPU **1**, serial servers (freed <2000 MiB after each), 0 failed requests. Collection only — no
  interpretation. Commits: `c6ec1df` (main), `8f41bd3` (EXTEND supplement), `d822bf3` (Case B retry).
  Artifacts: `experiments/qwen3vl8b/phase3/{summary.md,extend_supplement_summary.md,metadata/}`,
  `traces/qwen3vl8b/{case}/...`, `logs/qwen3vl8b/phase3/`. ~517 MB traces in Git LFS.

**Main collection** (`run_phase3_collect.py`). Per case: SGLang mapping (graph-off) + formal (graph-on)
via `sglang.profiler --profile-by-stage` with concurrent `bench_serving` load; vLLM `prefill_like`
(c=1, `max_tokens=1`) + `decode_like` (steady-state) via `--profiler-config` torch + `/start_profile`,
`/stop_profile`. Case A ran as a **pilot** (gate: 4 groups non-empty + metadata + no fatal + GPU freed);
gate passed → C/B/D ran automatically. **All 4 cases × 4 groups non-empty.** But all SGLang traces were
**DECODE-stage only** (profiler armed during in-flight decode).

**EXTEND supplement** (`run_phase3_extend.py`). Drove a prefill-only load (`/generate max_new_tokens=1`)
to force EXTEND into the window. Result (7/8 groups):

| Case | EXTEND mapping (graph-off) | EXTEND formal (graph-on) |
|---|---|---|
| A 128→128 c1 | ✅ 35 MB | ✅ 35 MB |
| B 2048→128 c1 | ✅ 64 MB | ❌ not captured (see caveat) |
| C 512→128 c16 | ✅ 47 MB | ✅ 31 MB |
| D 512→512 c16 | ✅ 51 MB | ✅ 34 MB |

**Case B graph-on EXTEND-formal caveat.** Failed across **8 attempts** (3 in supplement + a dedicated
5-strategy retry `d822bf3`: c=1 num_steps 50/100, c=2/c=4 num_steps 100/200). graph-on
`--profile-by-stage` prefill capture does not trigger for Case B's 2048-prefill path — small windows
land on fast graph-replayed DECODE, larger windows (≥50) produce **empty** stage traces. Resolving it
would need SGLang source changes (out of scope). **Accepted** because: (1) Case B's graph-off **mapping**
EXTEND (64 MB) carries the `kernel→cpu_op→python_scope` mapping needed for prefill-stage attribution;
(2) Case B already carries confidence ceiling M (both frameworks bimodal). All failed/empty attempt
dirs were removed (no LFS bloat). Documented in `extend_supplement_summary.md` +
`metadata/caseB_extend_formal_retry_meta.json`.

**Original DECODE traces remain valid and untouched.** vLLM traces were not re-run.

**Phase 4 readiness:** start with Case A and Case C; carry Case B caveats (graph-on EXTEND formal
missing + ceiling M). SGLang EXTEND mapping (all 4) + DECODE (all 4) + vLLM prefill/decode (all 4) cover
both stages for triage.

### Phase 4 — Trace Interpretation & Synthesis (completed 2026-05-23)

Offline triage (no GPU) of Phase 3 traces; all 4 cases. Commits `871565f` (A) · `440fe0e` (C) ·
`051e812` (B) · `947fd35` (D) · `55232b3` (global synthesis). Outputs under `analysis/qwen3vl8b/`
+ narrative `reports/qwen3vl8b/03_profiling_analysis.md`.

**Per-case summary:**

| Case | EXTEND | DECODE | vLLM crosscheck | Largest category | Key finding |
|---|---|---|---|---|---|
| A 128→128 c1 | ✅ | ✅ | ✅ | GEMM (84%/75%) | eager `aten::mm` vs vLLM graph → dispatch-overhead lead (OBS-A1); cleanest, highest priority |
| C 512→128 c16 | ✅ | ✅ | ✅ | GEMM (73%/85%) | vLLM prefill in torch.compile/inductor regions; batched radix-cache copies hidden; supports H1 |
| B 2048→128 c1 | ❌ unavailable | ✅ | ✅ | GEMM (—/78%) | vLLM long-prefill **also eager** → graph coverage NOT the Case B driver; bimodal → all ≤ M |
| D 512→512 c16 | ✅ | ✅ | ✅ | GEMM (72%/85%) | smallest gap (1.09×); long decode amortizes fixed overhead → corroborates first-token thesis |

**Global hypotheses:**

| ID | Hypothesis | Closes gap? | Impact | Confidence | Fairness dep. |
|---|---|---|---|---|---|
| H1 | SGLang eager `aten::mm` vs vLLM torch.compile/CUDA-graph | **Yes (primary)** | H | M | no |
| H2 | nvjet FP8 GEMM dominant; PR #22392 CUTLASS-FP8 | No (absolute only) | M/L | H/L | no |
| H3 | FlashInfer vs FA3 attention backend | No | L | M (capped) | **yes** |
| H4 | Case B gap = bimodality + c=1 fixed overhead | n/a | L | M | no |

**Phase 5 recommendation:**

| Priority | Hypothesis | Phase 5 action |
|---|---|---|
| 1 | H1 | Measure SGLang prefill CPU launch-gap (A, C); test CUDA-graph / piecewise-graph / torch.compile coverage |
| 2 | H2 | A/B PR #22392 CUTLASS-FP8 as a parallel **absolute-speed** track (not a gap-closer) |
| 3 (low) | H3 / H4 | No kernel-level effort unless later evidence changes — H3 fairness-ceilinged M; H4 bimodal + no SGLang EXTEND trace |

**Caveats:** Case B SGLang EXTEND unavailable → all Case B conclusions ceiling M; attention findings
ceiling M (FlashInfer vs FA3). **Phase 5 update:** H1 is now **strengthened by a clean (uninstrumented)
Case A confirmation** (forcing prefill piecewise CUDA-graph coverage cut Case A TTFT ~39%, TPOT
unchanged) — see §0 Phase 5 status block. H1 generalization to Case C and H2 remain **not yet
validated**; the `--enforce-piecewise-cuda-graph` lever is a testing flag, not a production fix.

---

### Historical note — earlier exploratory round (removed)

An earlier exploratory measurement round (informally "run1", 2026-04-17/04-24) was run on a
different machine/stack (GPU 6, CUDA 12.9, older SGLang/vLLM/torch/FlashInfer). Its artifacts
(`experiments/phase1|phase2|phase2_shaping`, root `datasets/case*.jsonl`, `logs/phase*`) were
**removed** during the repo restructure — they used a different environment and were **not
comparable** to the current `qwen3vl8b` results above (e.g. Case A ratio 4.0× then vs 4.89× now).
Only the current experiment is retained. Direction was consistent across both rounds (TTFT is the
gap; TPOT parity; vLLM Case B bimodal); magnitudes are not interchangeable.

---

## 16. Prioritized Next-Step Checklist

> **Progress:** ✅ env recovery (conda `profiling`, vLLM 0.21.0, model re-downloaded) ·
> ✅ Phase 0 PASS · ✅ Phase 1 (24 runs, 0 failures) · ✅ Phase 2 (GPU 7, 0 failures) ·
> ✅ Case C W500 probe (GPU 1, CV 2.9% — gate clean) · ✅ **Phase 3 trace collection (GPU 1)** —
> all 4 cases; SGLang DECODE + EXTEND-mapping (all 4) + EXTEND-formal (A/C/D); vLLM prefill/decode
> (all 4); Case B graph-on EXTEND-formal missing (caveat accepted). Commits `c6ec1df` / `8f41bd3` / `d822bf3`. ·
> ✅ **Phase 4 triage (offline, all 4 cases)** — A/C/D EXTEND+DECODE + vLLM; B DECODE+vLLM (EXTEND
> unavailable, ≤ M). Hypotheses H1–H4 + ranked recommendations. Commits `871565f` / `440fe0e` /
> `051e812` / `947fd35` / `55232b3`.
> **Next: Phase 5 validation.** Validate H1 first (SGLang prefill CPU launch-gap +
> graph/compile coverage on Case A/C); H2 as a parallel absolute-speed track (PR #22392); H3/H4 low
> priority. Items 1–5 are project setup (done).

1. ✅ Create the filesystem layout from §8.1 (placeholder READMEs in each directory).
2. ✅ Phase 0 — servers up, equivalence matrix run. All Tier-A/B pass; outputs EXACT match. *(Phase 0 also ✅ PASS — see §15.)*
3. ✅ Generate `datasets/qwen3vl8b/case{A..D}.jsonl` — text-only random prompts (special tokens excluded), SHA-256 logged.
4. ✅ Phase 1 — 24 runs (4 cases × 2 frameworks × 3 reps); `experiments/phase1/summary.md` complete.
5. ✅ Phase 2 (fully complete — all 5 exit criteria verified):
   - ✅ Step 2.1 — Case A: STRUCTURAL floor at 56 ms, CV=0.1%. No flag closes it.
   - ✅ Step 2.2 — Case B: STRUCTURAL (same floor); secondary finding: per-chunk dispatch overhead when chunking active.
   - ✅ Step 2.3 — Case C: PROMOTE (CV 4.2% at warmup=100). Case D: DROP (bimodal, V2 CV=14.8%).
   - ✅ Step 2.4 — vLLM recheck: Case B → CEILING M (CV=76%, bimodal); Case C → CLEAN (CV=5.5%, revised baseline 164→181 ms, ratio 1.49→1.33×).
   - ✅ Step 2.5 — selected_cases.md updated; Phase-3 protocol locked; all 5 exit criteria verified.
6. ✅ Phase 3  — SGLang mapping+formal (DECODE) + EXTEND supplement (mapping all 4, formal A/C/D) + vLLM prefill_like+decode_like, all 4 cases. Case B graph-on EXTEND-formal missing (caveat). See §15  + `experiments/qwen3vl8b/phase3/`.
7. ✅ Phase 4  — triage + breakdown + vLLM cross-check for all 4 cases; `analysis/qwen3vl8b/hypotheses.md` + `ranked_recommendations.md` + `reports/qwen3vl8b/03_profiling_analysis.md`. Case B EXTEND unavailable (≤ M). See §15 (Phase 4).
8. **Phase 5 (in progress)** — ✅ **Case A clean confirmation done: H1 strengthened** (`--enforce-piecewise-cuda-graph` cut Case A TTFT ~39%, TPOT unchanged, reaches vLLM TTFT range; testing-lever, not production fix). Next: Case A S2 stability supplement (reps=5) + **Case C** clean bracket validation (generalization to c=16). H2 (PR #22392) parallel absolute-speed track; H3/H4 low priority. Production-design discussion only after A/C validation.
9. Promote `analysis/**` into `reports/**` deliverables. *(03_profiling_analysis.md done; PR-ready writeup pending Phase 5.)*
