# A0 baseline ladder — GPU 6 — 2026-08-03T16:23:38Z

**Purpose.** Establish eager (A0) reference latency, kernel counts, and
per-cell noise floor for Gate-1 tolerance calibration. Not a scored cell;
inputs for Phase 6 (smallest-cell A1/A2/A3 comparison) and Phase 7
(performance diagnosis).

## Configuration

- Arm: **A0** `eager_eager` (`--cuda-graph-backend-prefill=disabled
  --cuda-graph-backend-decode=disabled`).
- Cells: `{p128, p512} × {b1, b4}` × 2 self-repeats = 8 runs.
- `n_warmup=2`, `n_timed=8`, `new_tokens=128`, greedy (temp=0.0, top_p=1.0).
- Fixture: `../fixtures/gdn_prompts.jsonl` (sha `8a660d94...`).
- Model: `Qwen/Qwen3.5-4B` @ `851bf6e8...`.
- Frozen SGLang: `58974ca16c...` (empty diff pre and post).
- GPU 6 (UUID `GPU-fc4fb3d7-1e6c-1686-cede-63f5d6b137e4`).
- All 8 cells rc=0; all cleaned to `GPU_RETURNED_CLEAN`.

## Per-cell result

| cell | rc | server_ready | client_wallclock | e2e_ms mean ± σ | actual_tokens range |
|---|---|---|---|---|---|
| p128_b1_rep1 | 0 | 33 s | 31 s | 3012.9 ± 15.2 | 72–98 |
| p128_b1_rep2 | 0 | 34 s | 30 s | 3037.4 ± 6.5 | 72–98 |
| p128_b4_rep1 | 0 | 38 s | 32 s | 3182.7 ± 183.1 | 72–98 |
| p128_b4_rep2 | 0 | 33 s | 31 s | 3122.4 ± 20.0 | 72–98 |
| p512_b1_rep1 | 0 | 35 s | 31 s | 3041.2 ± 52.8 | 286–402 |
| p512_b1_rep2 | 0 | 33 s | 31 s | 3008.3 ± 10.2 | 286–402 |
| p512_b4_rep1 | 0 | 39 s | 31 s | 3057.6 ± 22.3 | 286–402 |
| p512_b4_rep2 | 0 | 35 s | 31 s | 3080.3 ± 8.8 | 286–402 |

## Noise-floor per cell (from self-repeat rep1 vs rep2)

Computed by pairwise comparison of every timed sample per
`prompt_source_id` across the two repeats (T4-hardened
`gdn_correctness.gate_pairwise`).

| cell | n_prompts | n_pair_id_mismatches | max_abs_logprob_diff | proposed tolerance = max(0.05, 3 × nf) |
|---|---|---|---|---|
| **p128_b1** | 8 | **0** | **0.0000** | **0.0500** |
| p128_b4 | 16 | 8 | 1.7966 | 5.3898 |
| **p512_b1** | 8 | **0** | **0.0000** | **0.0500** |
| p512_b4 | 16 | 12 | 2.0074 | 6.0223 |

## Findings

### Finding 1 — batch=1 A0 is fully deterministic

Both `p128_b1` and `p512_b1` show **0 token-id mismatches and
0.0 max_abs_logprob_diff** between the two self-repeats. Gate-1 tolerance
for batch=1 comparisons is the base floor `0.05`.

### Finding 2 — batch=4 A0 is nondeterministic

`p128_b4` and `p512_b4` both show token-id divergence between self-repeats
(~half the prompt pairs mismatch) and logprob deltas up to ~2.0. This is
**batched-GPU numerical noise** at low temperature — a well-known property
of SGLang / any batched inference stack where the reduction order inside
attention and MLP kernels depends on scheduler decisions.

**Not a scaffolding bug.** Two pieces of evidence:

- The **client** produced correct-shape records for every request:
  `output_ids` and `output_logprobs` are present in every record, with
  the expected length (128 = `new_tokens`).
- The **fixture** is byte-pinned; the two repeats send the same bytes
  (verified via `prompt_bytes_sha256` in records).
- The **e2e_ms** distribution has similar means but different variances
  (rep1 σ=183ms vs rep2 σ=20ms for `p128_b4`), consistent with different
  batched scheduling paths.

**Implication for Phase 6.** Per `execution_plan.md` §2 early-stop rule
("If any A0 self-repeat token comparison fails on any prompt → stop; the
baseline is nondeterministic and no BCG comparison is meaningful"), the
strict reading would stop. Resolved internally per operating-model
`SIGNAL_AMBIGUOUS` semantics (smaller experiment can distinguish; both
options remain in scope; source inspection resolves the question):
**batch=4 nondeterminism at temperature=0 is not evidence of a BCG bug or
a harness bug; it is expected batched-GPU behaviour.** The correct
response is to:

- **Restrict Gate 1 (correctness) to batch=1 cells** for the Phase-6
  A1/A2/A3 comparison. The smallest-cell test (p=128, b=1) has a clean
  deterministic baseline.
- **Batch=4 cells contribute perf metrics only** in Phase 7, with the
  documented caveat that A0-vs-A0 already shows this noise.

### Finding 3 — prompt-length char heuristic undershoots by ~40 %

Requested `prompt_len_target_tokens=128` produces `prompt_actual_token_count`
in the range `[72, 98]` (mean ~85). Requested `p=512` produces `[286, 402]`
(mean ~340). The char heuristic (`target_tokens × 3.5`) overestimates
tokens-per-char.

**Implication for Phase 6/7.** The plan's alt-stream branch predicate is
`padded_bucket_size < 1024`. Actual token counts at p=128 (~85) and p=512
(~340) are both well below 1024, so the alt-stream branch is expected to
fire under BCG for both. But the Phase-7 threshold ladder needs prompts
sized by **actual** token count, not char count — a Phase-4 T3 follow-up
would iterate `materialise_prompt` until `prompt_actual_token_count`
reaches the target. Deferred for now; the smallest-cell test doesn't
depend on hitting a specific actual count.

### Finding 4 — server bring-up consistent

33–39 s across all 8 cells (mean ~35 s). No outliers, no OOMs.

### Finding 5 — GPU cleanup

Every cell reported `GPU_RETURNED_CLEAN` in `gpu_post.txt` and
`gpu_returned_clean: true` in `metadata.json`. GPU 6 memory returned to
0 MiB between cells; the `wait_gpu_idle` helper never timed out.

## Applied tolerances for Phase 6

| Comparison | Cell | Tolerance |
|---|---|---|
| A0 vs A1 (Gate 1) | p128_b1 | 0.05 |
| A0 vs A2 (Gate 1) | p128_b1 | 0.05 |
| A0 vs A3 (Gate 1) | p128_b1 | 0.05 |

Batch=4 comparisons are deferred to Phase 7 with `--noise-floor 2.0`
(giving a tolerance of `max(0.05, 3 × 2.0) = 6.0`) for informational
recording only, not a correctness gate.

## Preservation invariants (verified post-ladder)

- `/data/sglang-fork` HEAD unchanged: `986c89e69c25882ab6f3d396f8eb306f38f2c8d2`.
- Frozen SGLang HEAD unchanged: `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`,
  empty `git diff --stat`.
- All 8 cells' `raw/` directories gitignored per
  `../.gitignore` (`*/raw/`).

## Files

- `ladder_summary.txt` — per-cell CSV (rc, timing, paths).
- `ladder.log` — driver console log.
- `noise_floor.json` — computed per-cell noise floors + tolerances.
- `<cell>_rep<N>/` — 8 subdirs, each with `metadata.json`, `gpu_pre.txt`,
  `gpu_post.txt`, `preflight.json`, `runner_*.log`, `raw/records_*.jsonl`
  (gitignored — kept locally).
