# run2_qwen3vl8b — Active Profiling Run

Started 2026-05-21 on a rebuilt machine (run1's `/opt/miniconda3/envs/vllm` and the Qwen3-VL-8B
cache were lost in the reinstall). run2 reuses the methodology in `../../plan.md` but is a **fresh
measurement round** — run1's Phase 1/2 numbers are historical reference only and cannot be reused
(see `../env_snapshot.md` §"Why run2 ≠ run1").

## Environment (summary)
- Model: `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` (weights identical to run1, sha256-verified)
- GPU: index **0** (`CUDA_VISIBLE_DEVICES=0`)
- SGLang `0.0.0.dev1+g0c8049d9b` (system python3) · vLLM `0.21.0` (conda env `/opt/miniconda3/envs/profiling`)
- torch `2.11.0+cu130`, CUDA `13.0` (aligned across both frameworks)
- Full detail: `env_snapshot.md` (run-local) and `../env_snapshot.md` (canonical active).

## Phase status
| Phase | Status | Artifacts |
|---|---|---|
| 0 — Equivalence | ✅ **PASS** | `phase0/` (run-local) + promoted to canonical `../phase0/` |
| 1 — Baseline | ✅ **complete** (24 runs, 0 failures) | `phase1/summary.md`, `phase1/raw/`, `phase1/scripts/` |
| 2 — Shaping | ⬜ not started | `phase2/`, `phase2_shaping/` |
| 3 — Profiling | ⬜ not started | `../../traces/run2_qwen3vl8b/` |
| 4 — Triage | ⬜ not started | `../../analysis/run2_qwen3vl8b/` |
| 5 — Validation | ⬜ not started | `phase5/` |

## Artifact index
- `env_snapshot.md` — run-local environment record (versions, backends, memory)
- `phase0/` — run2 Phase 0 (run-local original):
  - `equivalence.md` — Tier A/B/C matrix + verdict
  - `model_files_sha256.txt` — per-shard safetensors sha256 (integrity)
  - `tier_a_results.txt` — tokenizer/vocab/template probe output
  - `sglang_outputs.json`, `vllm_outputs.json` — greedy outputs
  - `scripts/` — tier_a / tier_b collection + compare scripts
- Sibling run2 trees: `../../datasets/run2_qwen3vl8b/`, `../../logs/run2_qwen3vl8b/`,
  `../../traces/run2_qwen3vl8b/`, `../../analysis/run2_qwen3vl8b/`, `../../reports/run2_qwen3vl8b/`,
  `../../configs/run2_qwen3vl8b/`

## Conventions
- **Canonical Phase 0** lives at `experiments/phase0/` (replaced stale run1 Phase 0). The copy here
  is the run-local original; the two are kept in sync as completed deliverables (rarely edited).
- run1's `experiments/phase1/`, `experiments/phase2/`, `experiments/phase2_shaping/` are preserved
  unchanged as historical reference. run2 Phase 1+ working artifacts live under this run2 tree (not
  overwriting run1).
- Isolation: never overwrite run1 artifacts, `datasets/case*.jsonl`, `plan.md`, or `README.md`.

## Next step
Generate run2 datasets for the Qwen3-VL-8B text-only path (special-token-safe, as in run1's
`gen_datasets.py`), then run Phase 1 baseline (4-case matrix A/B/C/D) on GPU 0. Plan first, no
execution until confirmed.
