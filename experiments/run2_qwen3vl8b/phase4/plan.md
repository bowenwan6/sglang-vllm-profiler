# run2 Phase 4 — Trace Triage / Interpretation Plan

Status: **PLAN ONLY — not executed.** Authored 2026-05-22 for active run `run2_qwen3vl8b`.
Phase 0/1/2/3 complete; this plan converts Phase 3 traces into ranked, evidence-backed hypotheses.

> Scope guard: Phase 4 is **interpretation only**. No benchmark runs, no server launches, no trace
> re-collection, no SGLang source changes. The only execution is running the read-only analysis
> script over the existing Phase 3 trace files. Phase 5 (validation) is out of scope.

---

## A. Phase 4 Goals

1. Convert Phase 3 traces into **ranked, evidence-backed hypotheses** about where SGLang's TTFT gap
   vs vLLM originates (kernel / scheduler-dispatch / overlap / fuse / memory).
2. Per case, produce stage-separated triage (EXTEND + DECODE), a category breakdown, and a vLLM
   cross-check used **only** for falsification/corroboration.
3. Aggregate into `hypotheses.md` and `ranked_recommendations.md`.

**Non-goals (explicit):**
- Do **not** write final/definitive optimization recommendations as if validated — these are ranked
  *hypotheses* with confidence H/M/L, each pointing at a Phase 5 validation.
- Do **not** run Phase 5 validation.
- Do **not** modify SGLang source.
- Do **not** make vLLM optimization recommendations (vLLM is the reference baseline only).

---

## B. Case Priority

| Order | Case | Workload | Why this priority | Phase 4 focus |
|---|---|---|---|---|
| 1 (pilot) | **A** `caseA_short` | 128→128 c1, `--disable-overlap-schedule` | Highest priority; cleanest residual TTFT gap (1.56×) after the overlap-schedule flag | Remaining scheduler/dispatch fixed overhead after overlap disable |
| 2 | **C** `caseC_batched` | 512→128 c16, default | Stable batched gap (1.32×) after W500; CV 2.9% | Batched prefill/dispatch, CUDA-graph shape, batch formation, prefill→decode transition |
| 3 | **B** `caseB_longprefill` | 2048→128 c1, default | Long prefill but noisy; **ceiling M**; graph-on EXTEND-formal missing | Long-prefill EXTEND path; use graph-off mapping only; all hypotheses ≤ M |
| 4 | **D** `caseD_decode` | 512→512 c16, default | Lowest priority; small gap (1.09×); decode-heavy | Decode-path sanity check; confirm TPOT parity story |

---

## C. Per-case Analysis Outputs

For each case under `analysis/run2_qwen3vl8b/{case}/` (where `{case}` ∈
`caseA_short, caseC_batched, caseB_longprefill, caseD_decode`):

- `extend_triage.md` — SGLang EXTEND (prefill) two-trace triage (mapping+formal); three tables.
- `decode_triage.md` — SGLang DECODE two-trace triage (mapping+formal); three tables.
- `breakdown.md` — category breakdown (per §F) folding EXTEND + DECODE kernel tables into shared
  categories; SGLang GPU-time share by category per stage.
- `vllm_crosscheck.md` — vLLM prefill_like + decode_like single-trace triage; used to corroborate or
  falsify SGLang hypotheses, **not** to recommend vLLM changes.

Raw script stdout is captured alongside each `.md` as `*_raw.txt` (e.g. `extend_triage_raw.txt`) so
the rendered markdown is auditable against the tool output.

**Global outputs:**
- `analysis/run2_qwen3vl8b/hypotheses.md` — all hypotheses, schema per §G, grouped by case.
- `analysis/run2_qwen3vl8b/ranked_recommendations.md` — ranked by (impact × confidence), each linking
  to its hypothesis + a proposed Phase 5 validation.
- `analysis/category_regex.md` — shared category definitions (created in pilot; see §F).
- Optional: `reports/run2_qwen3vl8b/03_profiling_analysis.md` — narrative report folding the above
  for the status-report series (written last, only after all per-case + global outputs land).

---

## D. SGLang Triage Protocol

Tooling: `llm-torch-profiler-analysis` skill, entrypoint
`/root/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py`
(run with system `python3`, matching the SGLang collection toolchain). Only public workflow is
`triage` → three tables (kernel, overlap-opportunity, fuse-pattern). **Two-trace** triage
(`--mapping-input` graph-off + `--formal-input` graph-on) is the chosen shape for SGLang because it
recovers `kernel → cpu_op → python scope` and gives trustworthy overlap attribution.

Per case, two stages:

**EXTEND (prefill):**
- mapping input: `traces/run2_qwen3vl8b/{case}/sglang_extend_mapping/`
- formal input: `traces/run2_qwen3vl8b/{case}/sglang_extend_formal/`
- **Case B exception:** no graph-on EXTEND formal. Run **single-trace** triage on the graph-off
  mapping EXTEND only (`--input .../sglang_extend_mapping/`). Mark "EXTEND formal unavailable
  (8-attempt deviation)" in the output. All Case B EXTEND hypotheses **confidence ≤ M**, and the
  overlap table is read with the single-trace caveat the tool itself emits.

**DECODE:**
- mapping input: `traces/run2_qwen3vl8b/{case}/sglang_mapping/`
- formal input: `traces/run2_qwen3vl8b/{case}/sglang_formal/`

Catalog discipline (skill requirement): before calling any top row a "new" finding, look it up in
`references/fuse-overlap-catalog.md` and `references/overlap-catalog.md` (mainline rows first, then
PR-backed/in-flight). Prefer reporting an existing path that should-apply / appears-disabled /
is-missing-locally over inventing a new one. Record catalog status per hypothesis (§G).

---

## E. vLLM Cross-check Protocol

vLLM has no `--profile-by-stage`; use **single-trace** triage per window with `--framework vllm`:
- prefill_like: `--input traces/run2_qwen3vl8b/{case}/vllm/prefill_like/`
- decode_like: `--input traces/run2_qwen3vl8b/{case}/vllm/decode_like/`

Prefer the single-rank `rank0.*.pt.trace.json.gz` over the merged `async_llm.*` when the tool needs a
specific file (per skill: prefer one-rank traces). Point `--input` at the directory and let
auto-detection pick; if it merges undesirably, fall back to the explicit `rank0.*` file.

Use vLLM evidence **only** to:
- corroborate ("vLLM spends X on the same op → SGLang's extra time is elsewhere"), or
- falsify ("vLLM also pays this cost → not the source of the gap").
Do **not** produce vLLM optimization recommendations. Carry **ceiling M** on any conclusion that
hinges on the attention-backend difference (SGLang FlashInfer vs vLLM FlashAttention v3) or the
FlashInfer-version sampling-kernel difference.

---

## F. Category Breakdown

Shared categories (create `analysis/category_regex.md` in the pilot if absent — confirmed absent
today):

| Category | Intent (regex authored against actual kernel names in pilot trace) |
|---|---|
| attention | FlashInfer / FA3 / paged-attn / `*attention*`, `*mha*`, prefill/decode attn kernels |
| gemm | cutlass/cublas GEMM, `*gemm*`, `*matmul*`, linear/proj |
| memory | copy/cat/`*elementwise*` memcpy, `*memset*`, kv-cache write/gather, reshape |
| scheduler / CPU gap | CPU-side launch gaps, `cudaLaunchKernel` idle, host scheduling between forwards |
| norm | RMSNorm / LayerNorm / `*norm*` |
| quantization | `*quant*`, `*dequant*`, fp8/int8 scale kernels (likely empty at bf16) |
| communication | NCCL / all-reduce / all-gather (likely empty at TP=1) |
| sampling | `*sample*`, argmax/topk/`*softmax*` sampling-side kernels |
| uncategorized | everything else; must stay a small residual or regex is refined |

`breakdown.md` aggregates the kernel-table rows (GPU-time share) into these buckets, per stage
(EXTEND vs DECODE), for SGLang. The regex is finalized against the Case A pilot kernel names, then
reused verbatim for C/B/D so categories are comparable across cases.

---

## G. Hypothesis Schema

Every hypothesis (in per-case notes and aggregated in `hypotheses.md`) must carry:

- **title** — short, specific
- **case / stage** — e.g. `A / EXTEND`, `C / DECODE`
- **observation** — what the tables show (numbers, share %)
- **kernel name / operator / source location** — concrete; from kernel table + mapping trace
- **SGLang evidence** — table rows / shares / overlap or fuse findings
- **vLLM evidence** — corroborating or falsifying cross-check result (or "n/a — vLLM window differs")
- **catalog status** — one of: existing-should-apply / existing-disabled-or-regressed /
  mainline-elsewhere-missing-locally / open-upstream / no-catalog-fit (new); cite catalog row
- **impact estimate** — qualitative H/M/L tied to the case's residual gap (e.g. "explains ~Xms of the
  1.56× Case A gap")
- **confidence** — H/M/L (capped at M where ceiling M applies: any attention-backend-dependent claim;
  all Case B cross-framework claims; Case B EXTEND-formal-missing)
- **fairness dependence** — does the claim depend on the FlashInfer-vs-FA3 backend difference or other
  uncontrolled var? (yes/no + which)
- **caveats** — bimodality, single-trace overlap limit, missing formal trace, warmup mismatch, etc.
- **recommended Phase 5 validation / next action** — concrete experiment to confirm (no execution here)

---

## H. Stop Conditions

Stop and ask the user **before** (or immediately upon hitting) any of:

1. Analysis script missing, errors out, or cannot parse a trace.
2. A required trace file is missing or empty (beyond the known Case B EXTEND-formal gap).
3. The tool would require modifying SGLang source to proceed.
4. Case B missing formal EXTEND turns out to **block the tool entirely** (expected: it does not — fall
   back to single-trace mapping; only stop if single-trace also fails).
5. Disk usage would balloon (analysis outputs are markdown/txt — should be tiny; stop if any step
   tries to write large artifacts).
6. An analysis output would **overwrite an existing reviewed file** (today `analysis/run2_qwen3vl8b/`
   holds only `.gitkeep` — clean; stop if that changes).

---

## I. Execution Strategy (phased, gated)

1. **Pilot — Case A only.** Run EXTEND two-trace triage + DECODE two-trace triage + vLLM
   prefill/decode cross-check. Create `analysis/category_regex.md` from Case A kernel names. Write
   Case A's 4 files. **Gate: stop and let the user review** that the output is readable/useful before
   proceeding (per the user's pilot-first instruction).
2. If Case A output is good → **Case C** (same protocol, reuse category_regex).
3. Then **Case B** with caveats (single-trace EXTEND mapping; all hypotheses ≤ M).
4. Then **Case D** (sanity check; expect small/decode-parity findings).
5. **Only after** all per-case triage is done → write `hypotheses.md` then `ranked_recommendations.md`,
   then optionally `reports/run2_qwen3vl8b/03_profiling_analysis.md`.

Git checkpoint after each case (and after the global aggregation) per §K.

---

## J. Command Templates (do NOT execute yet)

Let `SKILL=/root/.claude/skills/llm-torch-profiler-analysis/scripts/analyze_llm_torch_profile.py` and
`TR=/data/sglang-vllm-profiler/traces/run2_qwen3vl8b` and
`OUT=/data/sglang-vllm-profiler/analysis/run2_qwen3vl8b`. Run with system `python3`.

**SGLang EXTEND triage (two-trace, A/C/D):**
```bash
python3 "$SKILL" --framework sglang \
  --mapping-input "$TR/{case}/sglang_extend_mapping/" \
  --formal-input  "$TR/{case}/sglang_extend_formal/" \
  --profile-by-stage \
  > "$OUT/{case}/extend_triage_raw.txt" 2>&1
```

**SGLang EXTEND triage (Case B — single-trace, mapping only):**
```bash
python3 "$SKILL" --framework sglang \
  --input "$TR/caseB_longprefill/sglang_extend_mapping/" \
  --profile-by-stage \
  > "$OUT/caseB_longprefill/extend_triage_raw.txt" 2>&1
```

**SGLang DECODE triage (two-trace, all 4):**
```bash
python3 "$SKILL" --framework sglang \
  --mapping-input "$TR/{case}/sglang_mapping/" \
  --formal-input  "$TR/{case}/sglang_formal/" \
  --profile-by-stage \
  > "$OUT/{case}/decode_triage_raw.txt" 2>&1
```

**vLLM prefill_like cross-check (single-trace, all 4):**
```bash
python3 "$SKILL" --framework vllm --no-profile-by-stage \
  --input "$TR/{case}/vllm/prefill_like/" \
  > "$OUT/{case}/vllm_prefill_raw.txt" 2>&1
```

**vLLM decode_like cross-check (single-trace, all 4):**
```bash
python3 "$SKILL" --framework vllm --no-profile-by-stage \
  --input "$TR/{case}/vllm/decode_like/" \
  > "$OUT/{case}/vllm_decode_raw.txt" 2>&1
```

**Category breakdown:** no separate subcommand — the kernel table *is* the breakdown. `breakdown.md`
is produced by bucketing the kernel-table rows (from the `*_raw.txt` outputs) into the §F categories
using `analysis/category_regex.md`. Done as a manual/markdown step over captured stdout, not a script.

**Hypothesis aggregation:** manual synthesis step reading all per-case `.md` + catalog lookups into
`hypotheses.md` and `ranked_recommendations.md` (schema §G). No script.

Notes:
- If `--profile-by-stage` on existing-trace input is rejected by the script, drop it (the stage is
  already fixed by which trace dir is passed); fall back to plain two-trace/single-trace.
- If a directory input merges ranks undesirably for vLLM, point `--input` at the explicit
  `rank0.*.pt.trace.json.gz` file instead.

---

## K. Git Workflow

- Start each work session with `git status`.
- This plan file: write to `experiments/run2_qwen3vl8b/phase4/plan.md`, then:
  ```bash
  git add experiments/run2_qwen3vl8b/phase4/plan.md
  git commit -m "feat(phase4): plan run2 trace triage"
  git push
  ```
- During execution, checkpoint after each case and after global aggregation with English
  `feat(phase4): ...` messages describing case/stage/result/caveat.
- **Never** stage `.claude/settings.local.json`. **Never** force push. No `git reset --hard` /
  `git checkout -- <file>`. No empty commits.

---

## Phase 4 readiness summary (verified 2026-05-22, read-only)

- All 16 vLLM trace groups + SGLang DECODE (mapping+formal ×4) + SGLang EXTEND mapping ×4 + EXTEND
  formal (A/C/D) present and non-empty on disk.
- Only gap: `caseB_longprefill/sglang_extend_formal/` absent (8-attempt deviation, accepted) → Case B
  EXTEND uses single-trace mapping, hypotheses ≤ M.
- Analysis script present and `--help` works; catalogs present (`fuse-overlap-catalog.md`,
  `overlap-catalog.md`, etc.). `analysis/category_regex.md` absent → create in pilot.
- `analysis/run2_qwen3vl8b/` holds only `.gitkeep` → no overwrite risk.
