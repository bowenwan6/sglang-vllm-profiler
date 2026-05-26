# Phase 5 — Hypothesis Validation Plan (main experiment `qwen3vl8b`)

> ⚠️ **Methodology correction (2026-05-26):** SGLang TTFT figures from Phase 1 / Phase 2 Case C were collected with **SGLang-only KAPI logging** and are **instrumentation-confounded (provenance only)**. The Case C **“1.32× SGLang-slower” gap is SUPERSEDED** by the clean rerun (no material median gap; SGLang ≈ vLLM ≈ 190 ms). Data retained unchanged. See `experiments/qwen3vl8b/methodology_correction.md`.


Status: **PLAN ONLY — not executed.** Authored 2026-05-25. No benchmark / profiling / server / GPU /
source changes were run for this document — research was read-only. `<main>` = `experiments/qwen3vl8b`
(restructure complete; canonical paths already use `qwen3vl8b`).

> Phase 5 validates the Phase-4 ranked hypotheses — it does **not** expand benchmarking. The main line
> is **H1**. Each hypothesis stays a *hypothesis* until Phase 5 evidence moves it; nothing here is a
> pre-decided conclusion.

---

## 1. Objective and Hypotheses

| ID | Hypothesis | Phase 5 role |
|---|---|---|
| **H1** | SGLang's graph/compile **coverage of the prefill/dispatch path** is weaker than vLLM's → extra CPU launch/dispatch overhead → TTFT residual gap | **Main line** |
| H2 | `nvjet → CUTLASS FP8` (SGLang PR #22392) → **absolute** SGLang speedup only | Optional, **independent** track |
| H3 | Attention backend mismatch (FlashInfer vs FA3) | Documented **ceiling M** only — not validated |
| H4 | Case B bimodality + unavailable SGLang EXTEND | **Deprioritized** — not in first-round H1 validation |

**Phase 5 validates H1.** H2 is an optional, separately-reported absolute-speed track. H3/H4 receive no
kernel-level effort this round.

### What the research already established (read-only, this task)

The Case A and Case C **graph-on formal** `server_args.json` show the real serving config:

- `disable_cuda_graph = False` → **decode** CUDA graph is **ON**.
- `disable_piecewise_cuda_graph = True` → **piecewise** CUDA graph (the prefill/mixed-path graph) is **OFF**.
- `enable_torch_compile = False` → torch.compile **OFF**.

Root cause confirmed in SGLang source (`python/sglang/srt/server_args.py:1308-1310`,
`_handle_piecewise_cuda_graph`): **condition #8 auto-disables piecewise CUDA graph for multimodal/VLM
models.** Qwen3-VL-8B is multimodal → its prefill path is **eager by default**, even in the formal
serving config. This is the concrete mechanism behind H1: vLLM compiles/graphs the prefill path
(inductor AOT + CUDA graph), SGLang dispatches it eagerly (`aten::mm` / `cudaLaunchKernelExC`).

**Flag-level levers exist (no source change required):**
- `--enforce-piecewise-cuda-graph` → `server_args.py:1276` "Skip auto-disable when enforce flag is set
  (for testing)" → forces `disable_piecewise_cuda_graph = False` even for the VLM. **Coverage-expansion candidate #1.**
- `--enable-torch-compile` → torch.compile path; applies up to `torch_compile_max_bs=32` (covers Case A
  c=1 and Case C c=16). Mutually exclusive with piecewise (line 1288). **Coverage-expansion candidate #2.**

> Clarification (per the task): the **graph-off mapping** trace is only a kernel→source mapping tool and
> does **not** prove the serving path lacks graph coverage. The evidence above comes from the
> **graph-on formal** `server_args.json` + SGLang source, plus the formal-trace kernel CPU-op column
> (prefill kernels dispatch via `aten::mm`, not `cudaGraphLaunch`). That is what Phase 5.1 quantifies
> and Phase 5.2 intervenes on.

---

## 2. Validation Strategy for H1

Two gated layers: **5.1 observational** (mostly offline, existing traces) → **5.2 controlled intervention** (GPU).

### Phase 5.1 — Observational validation (offline-first)

Inputs (existing, valid; **do not re-collect, do not modify**):
- Case A: `<main>/../../traces/qwen3vl8b/caseA_short/sglang_{formal,extend_formal,mapping,extend_mapping}/`,
  `vllm/{prefill_like,decode_like}/`
- Case C: same under `caseC_batched/`

For Case A and Case C, compute (SGLang formal vs vLLM, EXTEND stage as the gap locus; DECODE for contrast):

| Metric | Definition | Source | Offline now? |
|---|---|---|---|
| **graph-covered kernel time share** | Σ GPU time of kernels whose CPU launch op is `cudaGraphLaunch` ÷ total GPU time | kernel CPU-op column (already in triage kernel table) | ✅ yes |
| **eager-dispatched kernel time share** | Σ GPU time of kernels launched via `aten::mm` / `cudaLaunchKernelExC` ÷ total | same | ✅ yes |
| **kernel launch count** | # GPU-kernel launch events in the profiled window | trace timeline events | ✅ yes (new script) |
| **CPU launch-gap** | Σ CPU time spent in launch ops + inter-launch CPU gaps on the critical path | `triage_overlap_helpers.extract_cpu_launch_contexts` / `build_launch_contexts` | ✅ computable (new script reusing helpers) |
| **inter-kernel GPU idle gap** | Σ wall-time on the GPU stream where no kernel is running, within the window | trace GPU-stream timeline | ✅ yes (new script) |
| **EXTEND-stage critical-path duration** | wall time of one representative prefill forward (first → last kernel of the step) | trace timeline | ✅ yes (new script) |
| **exclusive / hidden GPU time** | per-kernel `exclusive_us` / `hidden_us` / `hidden_by_compute_us` | `triage` overlap table (already emitted) | ✅ already available |

vLLM cross-check: same metrics on `vllm/prefill_like` (and `decode_like` for contrast). Expected
contrast if H1 holds: vLLM prefill ≈ high graph/compile-covered share + low CPU launch-gap; SGLang
prefill ≈ ~0% graph-covered + higher launch-gap / more eager launches.

**Tooling gap.** The public `triage` already gives the kernel CPU-op column + the overlap table
(exclusive/hidden). It does **not** emit launch-count, summed CPU launch-gap, GPU-idle-gap, or
critical-path duration as first-class numbers. Phase 5.1 therefore needs **one new read-only analysis
script** `<main>/phase5/scripts/h1_launch_gap.py` that parses the existing trace JSON (reusing
`profile_common` + `triage_overlap_helpers` launch-context helpers) and emits the table above. It reads
traces only; writes markdown/JSON to `analysis/qwen3vl8b/phase5/h1_launch_gap/`. **No GPU, no re-collection.**

Output: `analysis/qwen3vl8b/phase5/h1_launch_gap/{caseA_short,caseC_batched}.md` + a combined
`summary.md` with the metric table (SGLang formal vs vLLM, EXTEND + DECODE), plus the raw per-kernel
CSV. One table per case; columns = the metrics above; rows = {SGLang EXTEND, SGLang DECODE, vLLM
prefill_like, vLLM decode_like}.

### Phase 5.2 — Controlled intervention (GPU; only if 5.1 supports H1)

Run **only after** 5.1 shows SGLang prefill is eager-dominated with a measurable launch-gap that
plausibly accounts for a meaningful fraction of the residual. Priority: **Case A first, then Case C.**

Variants per case (all on the Phase-2 locked dataset/warmup/reps so results are baseline-comparable):

| Variant | SGLang flags (on top of Phase-2 case config) | Graph state | Purpose |
|---|---|---|---|
| **baseline** | Phase-2 locked config (A: `--disable-overlap-schedule`; C: default) | decode-graph ON, **piecewise OFF** (VLM auto-disable), compile OFF | the real serving baseline (graph-on formal) |
| **negative control** | + `--disable-cuda-graph --disable-piecewise-cuda-graph` | all graph OFF (fully eager) | confirms the metric responds to graph state (should be *worse* or unchanged TTFT, more launch-gap) |
| **coverage-expansion #1** | + `--enforce-piecewise-cuda-graph` | piecewise graph **forced ON** for the VLM | the H1 test: does covering prefill reduce launch-gap & TTFT? |
| **coverage-expansion #2 (if #1 infeasible/unstable)** | + `--enable-torch-compile` | torch.compile (≤ bs 32) | alternative coverage expansion |

> The baseline is **already graph-on for decode**; "turning graph on" is **not** a new variant. The new
> variant is **extending coverage to the prefill path** (piecewise / compile), which is currently
> auto-disabled for this VLM. `--enforce-piecewise-cuda-graph` is explicitly a *testing* flag — its
> stability on a VLM text path is unverified; if the server fails to start or produces empty/invalid
> traces, fall back to coverage-expansion #2, and if both are infeasible without source changes, **stop
> and report** (do not modify source).

Per variant, record:
- exact server flags; warmup / reps / bench_n / concurrency (= Phase-2 locked values per case)
- TTFT p50/p95/p99, TPOT, CV, error rate
- graph-covered kernel share + CPU launch-gap (re-run the 5.1 script on a fresh Phase-5 trace of that variant)
- functional sanity (greedy output non-empty; no NaN/Inf in kapi log)

Phase-2 locked values: **Case A** warmup 30, reps 3, bench_n 400, c=1, flags `--disable-overlap-schedule`.
**Case C** warmup 500, reps 5, bench_n 2000, c=16, default flags. vLLM is the fixed reference (already
measured; not re-run unless a same-warmup control is explicitly needed).

---

## 3. H1 Pass / Fail Criteria

**H1 strengthened / supported** if, for **both** Case A and Case C:
- coverage-expansion measurably **increases** graph-covered prefill kernel share (≈0% → substantial), **and**
- CPU launch-gap / inter-kernel idle on the prefill critical path **drops materially**, **and**
- TTFT residual gap vs vLLM **narrows materially** (Case A; the Case C 1.32× was later SUPERSEDED — KAPI-confounded — so only Case A applies) (move toward 1×
  by a margin well outside the measured CV ~3%), **and**
- TPOT, error rate, and functional correctness show **no material regression**.

**H1 weakened / rejected** if any of:
- coverage-expansion changes graph share / launch-gap but TTFT **does not improve**;
- TTFT change is explained mainly by **kernel compute time** change (not dispatch/launch-gap);
- Case A and Case C **disagree** with no coherent explanation;
- the intervention is infeasible without source changes (then H1 stays Medium — *unvalidated*, documented).

Do **not** presuppose H1 is correct. A clean rejection (graph coverage rises, gap doesn't move) is a
valid, valuable Phase 5 outcome.

---

## 4. Optional H2 Track (independent)

PR #22392 (`nvjet` FP8 GEMM → CUTLASS scaled-MM). **Goal: absolute SGLang latency, NOT closing the vLLM
gap** (vLLM pays the same nvjet cost). Steps (read-only first):
1. **Read-only** check PR #22392 status, change scope, and whether it applies to the current SGLang
   commit `0c8049d9b` (via `gh`/git on `/sgl-workspace/sglang`; do not modify source). If it needs a
   source patch, that is a **source-change decision requiring approval** (default: not allowed).
2. Only if cleanly applicable: A/B baseline vs CUTLASS-FP8 on Case A/C, measuring **absolute** TTFT/TPOT.
3. Report H2 in a **separate table/section** from H1; never phrase it as a vLLM-gap closer.

---

## 5. Case Scope

- **Case A (primary):** cleanest residual (1.56×), c=1, lowest variance (CV 3.2%), fairness-independent
  H1 signal. First intervention target.
- **Case C (primary):** ~~stable batched gap (1.32×)~~ — this pre-run assumption was **falsified**: clean rerun shows no material gap and no Case-A-like benefit (KAPI-confounded 1.32× SUPERSEDED).
- **Case D (corroboration only):** run **only if** Case A/C support H1; its small gap (1.09×) +
  decode-heavy shape should show the smallest prefill-coverage effect (consistency check), not a primary test.
- **Case B (excluded this round):** bimodal in both frameworks + **no usable SGLang EXTEND trace**
  (`<main>/phase3/caseB_trace_issue.md`). Enter H1 validation only after bimodality is resolved and a
  real long-prefill EXTEND trace can be captured — out of scope for the first round.

---

## 6. Artifact Layout (to be created during execution)

```text
experiments/qwen3vl8b/phase5/
  plan.md                  # this file
  protocol.md              # exact commands/flags per variant (written before 5.2)
  scripts/
    h1_launch_gap.py       # read-only offline metric extractor (5.1)
    run_phase5_caseA.py     run_phase5_caseC.py   # 5.2 GPU orchestration (GPU 3)
  raw/                     # Phase-5 bench JSON + meta per variant
  summary.md               # final per-case variant tables + H1 verdict

analysis/qwen3vl8b/phase5/
  h1_launch_gap/           # caseA_short.md, caseC_batched.md, summary.md (+ raw CSV)
  h2_cutlass_fp8_optional/ # only if H2 track runs

reports/qwen3vl8b/
  04_phase5_validation.md  # narrative report (H1 verdict; H2 separate section)

logs/qwen3vl8b/phase5/     # server + kapi L1 logs (GPU 3)
traces/qwen3vl8b/phase5/   # NEW Phase-5 variant traces only (never overwrite Phase-3 traces)
```

Existing Phase 0–4 artifacts, raw JSON, trace metadata, and `*_raw.txt` are **not** modified.

---

## 7. Execution Order (gated)

1. **Preflight** — verify tools (`analyze_llm_torch_profile.py`, helpers), confirm Case A/C formal +
   vLLM traces readable, confirm `<main>/phase5/` paths. *(offline)*
2. **Offline feasibility check** — confirm the existing Case A/C graph-on formal + vLLM traces contain
   the needed CPU-launch / GPU-timeline events for the 5.1 metrics. *(offline)*
3. **Compute observational H1 metrics** — run `h1_launch_gap.py`, emit `analysis/qwen3vl8b/phase5/h1_launch_gap/`. *(offline)*
4. **Review 5.1 results** — decision point: does the offline evidence support H1 enough to justify GPU work?
5. **Case A controlled-intervention pilot** — baseline + negative control + coverage-expansion (GPU 3). *(GPU)*
6. **Review Case A pass/fail.** *(decision point)*
7. **Case C controlled intervention** — same variants (GPU 3). *(GPU)*
8. **Optional H2 / PR #22392 track** — read-only applicability first; A/B only if clean. *(decision point + maybe GPU)*
9. **Summary + report** — `phase5/summary.md` + `reports/qwen3vl8b/04_phase5_validation.md`; update
   `analysis/qwen3vl8b/hypotheses.md` confidence column.

---

## 8. GPU / Safety Rules (for later execution — NOT this task)

- **All** Phase 5 server / client / profiler / any re-collection commands use **`CUDA_VISIBLE_DEVICES=3`**.
- **Do not** use GPU 0, GPU 1, or GPU 7.
- Before any launch: confirm **GPU 3 is idle** (memory < 2000 MiB) and no residual SGLang/vLLM server processes.
- **Serial server lifecycle** — only one server on GPU 3 at a time; never co-resident.
- After every server shutdown: confirm **GPU 3 memory < 2000 MiB** before the next launch.
- Do **not** modify raw JSON / existing Phase-0–4 traces. New traces → `traces/qwen3vl8b/phase5/` only.
- **Stop immediately and report** on: server crash, OOM, empty/invalid trace, `failed requests > 0`,
  GPU 3 not freed, or any need for an unapproved source change.

---

## 9. Estimated Time

| Step | Estimate |
|---|---|
| 5.1 offline metric extraction (script + Case A/C run) | ~0.5–1 day (incl. writing `h1_launch_gap.py`) |
| Case A GPU pilot (3 variants, GPU 3) | ~0.5 day |
| Case C GPU validation (3 variants, GPU 3) | ~0.5–1 day |
| Optional H2 / PR #22392 evaluation | ~0.5 day (read-only) + ~0.5 day if A/B run |
| Summary + report | ~0.5 day |
| **Total** | **~2–3.5 days** (H1 only ~1.5–2.5; +H2 optional) |

---

## 10. Decision Points Requiring Approval

1. Execute **5.1 offline metric extraction** (write + run `h1_launch_gap.py` on existing traces). *(no GPU)*
2. Start **Case A GPU pilot** — confirm **GPU 3**.
3. Allow testing the **graph/compile coverage variants** (`--enforce-piecewise-cuda-graph` / `--enable-torch-compile`).
4. Proceed to **Case C** — confirm **GPU 3**.
5. Open the **H2 / PR #22392** optional track.
6. **Modify source code** — **default: not allowed**; explicit approval required if any variant needs it.

---

### Research summary feeding this plan (read-only, 2026-05-25)
- Canonical paths already `qwen3vl8b/` (restructure complete); working tree clean except `.claude/settings.local.json`.
- Case A/C formal `server_args.json`: decode CUDA graph ON, **piecewise OFF**, torch.compile OFF.
- SGLang `server_args.py` `_handle_piecewise_cuda_graph`: VLM/multimodal models auto-disable piecewise
  (cond. #8); `--enforce-piecewise-cuda-graph` overrides it ("for testing"); `--enable-torch-compile`
  is an alternative (≤ bs 32).
- Trace tooling parses CPU launch events + exclusive/hidden GPU time; explicit launch-count / launch-gap
  / idle-gap / critical-path metrics need one new read-only script (no re-collection for Case A/C).
