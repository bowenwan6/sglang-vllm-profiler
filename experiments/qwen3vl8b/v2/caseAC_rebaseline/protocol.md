# Protocol — Issue #2: Rebaseline Qwen3-VL on default overlap schedule

> **Status: APPROVED & EXECUTING.** Decision-complete experiment plan. Foundational v2 experiment —
> #3/#4/#5 rest on it. Clean only: never set `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST`;
> no profiler.

## 0. Execution instance (this run)

- **GPU: 1** (`CUDA_VISIBLE_DEVICES=1`). Verified idle at preflight (util 0%, <2000 MiB; ~1.1 GB foreign
  baseline held by an out-of-namespace orphan process, 0% util — not heavy compute). Do **not** auto-
  switch GPUs; if GPU 1 is not idle at run time, stop and report.
- **Case A** runs the optional `A_S0_abl_no_overlap` ablation (for v1 comparability).
- **Case C** does **NOT** run `C_S0_abl_no_overlap` unless explicitly approved later (avoid long runtime).
- **Text-only** runs — `SGLANG_USE_CUDA_IPC_TRANSPORT` is **unset** (image+text/IPC is issue #4, not here).
- All headline runs forbid KAPI/profiler; servers serialized; env stripped of KAPI + IPC vars before launch.

## 1. Goal

Re-establish the Qwen3-VL clean headline baseline on the **production-default overlap schedule (ON)** and
re-test the validated v1 Case-A finding against it.

v1's clean Case-A finding (PCG lever → ~39% TTFT cut, TPOT unchanged) is sound, **but its S0 baseline ran
with `--disable-overlap-schedule`** — an overlap-**OFF** config that Phase 2 *selected as the fastest*,
not the production default. So the −39% was only ever measured in the overlap-OFF regime, and v1's Case A
(overlap-off) vs Case C (overlap-on, default) were asymmetric. This experiment removes that limitation by
making **SGLang default (overlap-ON)** the headline baseline and demoting `--disable-overlap-schedule` to
an optional ablation. (Note: v1's clean **Case C** rerun was already run on default/overlap-ON, so for
Case C this round is primarily a **confirmation refresh**; Case A is the genuinely new baseline.)

## 2. Experimental Questions

- **Q1 (gap):** On SGLang **default (overlap-ON)** vs vLLM default, how large is the Case A TTFT gap?
- **Q2 (intervention):** Does SGLang default **+ `--enforce-piecewise-cuda-graph`** reduce Case A TTFT
  relative to SGLang default (overlap-ON), with TPOT not worsening and 0 failures?
- **Q3 (shape boundary):** Is the PCG effect specific to Case A (c=1), or does it also appear at
  Case C (c=16)?
- **Q4 (ablation delta):** How much does `--disable-overlap-schedule` differ from the production-default
  baseline (i.e., how much of v1's number was the overlap-off choice vs the PCG lever)?

## 3. Workloads

| Case | Shape | Concurrency | Why included |
|---|---|---|---|
| **A** `caseA_short` | 128 → 128 | 1 | validated short-latency locus; the new production-default baseline |
| **C** `caseC_batched` | 512 → 128 | 16 | batched boundary; confirm "no material gap / no Case-A-like benefit" under default |

Only A and C: A is where the validated effect lives and where the overlap-OFF→ON correction matters most;
C is the boundary check. B/D are out of scope this round (B has no clean EXTEND trace; D is lowest payoff).

## 4. Variants

Common server flags (both cases): SGLang `--dtype bfloat16 --tp 1 --attention-backend flashinfer`;
vLLM `--dtype bfloat16 --tensor-parallel-size 1`. All clean (no KAPI, no profiler). Greedy
(`temperature=0, top_p=1`).

**Case A**

| id | framework | distinguishing flags | role |
|---|---|---|---|
| `A_S0_default` | SGLang | *(none — overlap ON)* | **headline production baseline** |
| `A_S2_pcg` | SGLang | `--enforce-piecewise-cuda-graph` | PCG testing lever (on default) |
| `A_V0_vllm` | vLLM | *(none)* | clean cross-framework anchor |
| `A_S0_abl_no_overlap` | SGLang | `--disable-overlap-schedule` | **optional** ablation — v1 comparability only |

**Case C**

| id | framework | distinguishing flags | role |
|---|---|---|---|
| `C_S0_default` | SGLang | *(none — overlap ON)* | headline production baseline |
| `C_S2_pcg` | SGLang | `--enforce-piecewise-cuda-graph` | PCG testing lever (on default) |
| `C_V0_vllm` | vLLM | *(none)* | clean cross-framework anchor |
| `C_S0_abl_no_overlap` | SGLang | `--disable-overlap-schedule` | optional ablation — only if runtime allows |

## 5. Run Design (bracketing / interleaving to bound drift)

- **Case A (low variance, v1 CV ~1.5%):** simple bracket —
  `A_S0_default → A_S2_pcg → A_S0_default(repeat) → A_V0_vllm → [optional A_S0_abl_no_overlap]`.
  The S0 repeat brackets S2 to measure baseline drift.
- **Case C (high session variance):** **interleaved** —
  `C_S0_a → C_S2_a → C_S0_b → C_S2_b → C_S0_c → C_V0_vllm → [optional C_S0_abl_no_overlap]`.
  **Why interleave:** v1's clean Case C rerun showed ~17% run-to-run *session* variance even at w500; a
  single S0→S2→S0 bracket is inconclusive. Interleaving samples S2 twice between three S0 blocks so the
  S0-vs-S2 *comparison* is robust to that drift.

Each variant: fresh server, smoke check, then reps; server killed and GPU freed (<2000 MiB) before the
next variant. Servers never co-resident.

## 6. Warmup / reps / request counts

Matches v1 clean validation so numbers are directly comparable.

| Case | dataset (sha256 prefix) | num-prompts | max-concurrency | warmup-requests | reps | seed |
|---|---|---|---|---|---|---|
| A | `caseA_short.jsonl` (`fab4917772e08744`) | 400 | 1 | 30 | **5** | 1 |
| C | `caseC_batched.jsonl` (`265bde3e48793077`) | 2000 | 16 | 500 | **3 per block** | 1 |

- Case A uses **reps 5** (not v1's 3) specifically to tighten the S2 CV — v1's S2 CV was ~10–12%, the
  one weak spot of the validated finding; more reps lets us state CV honestly.
- Case C uses **reps 3 per block** (5 SGLang blocks + vLLM = matches v1 interleaved sampling).

**Reduced plan if runtime is too long:** Case A reps 3 (still brackets S2); Case C 2 reps per block and
drop the optional `*_abl_no_overlap` ablation. **Do not sacrifice:** num-prompts (sample size), warmup,
the 0-failure requirement, the clean (no-KAPI/profiler) condition, or the S0 bracket/interleave (drift
control). I.e. reduce *repetitions*, never *sample size* or *drift control*.

## 7. Metrics (record per rep + per-variant median/CV)

TTFT p50 / p95 / p99 · TPOT p50 · E2E latency · output throughput (tok/s) + request throughput (req/s) ·
CV across reps · failures / error rate · GPU id · exact flags · model snapshot sha · framework versions
(SGLang + vLLM) · dataset sha256 · num-prompts / concurrency / warmup / reps · explicit
`kapi_logging=false, profiler=false` confirmation · smoke-check output.

## 8. Acceptance Criteria

- **Clean-run gate:** 0 failures on every counted rep; no KAPI env var present; no profiler; servers
  serialized; smoke check passes (non-empty greedy output) before reps.
- **Bracket-drift gate:** S0 repeat (Case A) or the three S0 blocks (Case C) drift ideally **≤ 5%**. If
  drift > 5%, **downgrade** the affected absolute number to "indicative" and lean on the within-design
  comparison (interleaved S2-vs-S0) rather than the absolute baseline.
- **PCG-win rule (Q2):** declare a PCG benefit only if `S2_pcg` median TTFT improves **≥ 5%** vs the
  bracketed `S0_default`, **TPOT does not worsen**, and 0 failures. Otherwise: "no benefit on the
  production-default baseline."
- **vLLM role:** vLLM is an **anchor only**; do not treat vLLM's graph/compile strategy as direct causal
  proof for SGLang. Report SGLang-vs-SGLang (S0 vs S2) as the causal comparison; SGLang-vs-vLLM as
  context.
- **Case C verdict:** if `S0`, `S2`, and `vLLM` all sit inside the noise band, the conclusion is
  **"no material gap / no Case-A-like benefit under the production-default baseline,"** not parity and
  not a proven gap.

## 9. Stop Conditions (abort the run, do not work around)

- Target GPU not idle (≥ 2000 MiB used) or any other heavy process present.
- Any counted variant has failures > 0.
- A server fails to release GPU memory after kill (still occupied before next variant).
- Any `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST` is set, or a profiler is active.
- The plan would require editing SGLang source.
- `S2_pcg` smoke / correctness check fails.
- A server or KAPI log file grows abnormally large (instrumentation leak signal).

## 10. Artifact Plan

- Protocol: `experiments/qwen3vl8b/v2/caseAC_rebaseline/protocol.md` (this file).
- Results (future): `experiments/qwen3vl8b/v2/caseAC_rebaseline/results/` — per-variant `results.json`,
  per-rep `raw/*.json` + `*_meta.json`, and `summary.md`.
- Server logs (future): `logs/qwen3vl8b/v2/caseAC_rebaseline/`.
- **Do not** overwrite or touch any v1 Phase 5 results (`experiments/qwen3vl8b/phase5/...`).

## 11. Execution Checklist (for the approved run)

1. Preflight `git status` — clean tree, on `main`.
2. `nvidia-smi` — choose an idle GPU; record its id.
3. Verify no KAPI env vars are set (`env | grep SGLANG_KERNEL_API` must be empty).
4. Launch SGLang default server → **smoke check** (greedy 8-token) before any rep.
5. Run **Case A** variants in the §5 bracket order; free GPU between variants.
6. **Summarize Case A** (medians, CV, drift, PCG verdict) *before* starting Case C.
7. Run **Case C** interleaved variants; free GPU between variants.
8. Update `summary.md` (both cases, with the §8 verdicts).
9. Commit **only** protocol/result docs (no raw artifact rewrites of v1; no source).

## 12. Command Templates (DO NOT EXECUTE — reference only)

Reusable runner template: `experiments/qwen3vl8b/phase5/scripts/run_caseA_h1_confirmation.py` (clean
bracket harness) and `run_caseC_clean_rerun.py` (interleaved harness). **They hardcode
`--disable-overlap-schedule` and a fixed GPU**, so #2 needs a new runner
`experiments/qwen3vl8b/v2/caseAC_rebaseline/run_caseAC_rebaseline.py` whose Case-A `S0`/`S2` variants
**omit** `--disable-overlap-schedule`. *Plan only — not implemented in this task.*

```bash
# Clean env (every run): pin GPU, force offline, strip any KAPI instrumentation.
export CUDA_VISIBLE_DEVICES=<GPU_ID>          # from nvidia-smi at preflight
export HF_HUB_OFFLINE=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

# --- SGLang default baseline (overlap ON): A_S0_default / C_S0_default ---
python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 \
  --port 30000 --tp 1 --attention-backend flashinfer

# --- SGLang + PCG lever: A_S2_pcg / C_S2_pcg (add the one flag) ---
python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 \
  --port 30000 --tp 1 --attention-backend flashinfer --enforce-piecewise-cuda-graph

# --- Optional ablation only: *_S0_abl_no_overlap ---
python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 \
  --port 30000 --tp 1 --attention-backend flashinfer --disable-overlap-schedule

# --- vLLM clean anchor: A_V0_vllm / C_V0_vllm ---
/opt/miniconda3/envs/profiling/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$SNAP" --dtype bfloat16 --port 30001 --tensor-parallel-size 1

# --- Bench client (per variant; backend sglang-oai | vllm) ---
# Case A: --max-concurrency 1  --num-prompts 400  --warmup-requests 30   (dataset caseA_short.jsonl)
# Case C: --max-concurrency 16 --num-prompts 2000 --warmup-requests 500  (dataset caseC_batched.jsonl)
python3 -m sglang.bench_serving --backend <sglang-oai|vllm> \
  --base-url http://127.0.0.1:<PORT> --dataset-name autobench \
  --dataset-path datasets/qwen3vl8b/<caseA_short|caseC_batched>.jsonl \
  --max-concurrency <C> --num-prompts <N> --seed 1 --warmup-requests <W> \
  --extra-request-body '{"temperature": 0, "top_p": 1}' \
  --output-details --output-file experiments/qwen3vl8b/v2/caseAC_rebaseline/results/<variant>_rep<k>.json
```
