# R6.5 predeclaration — 2026-07-29

**Predeclared before running any R6.5 measurement.**

## Ratios

Per the R6.4 analytical result (`p* = −3.912`, out of `[0, 1]`; fork
dominates the whole mix-ratio axis on the R6.2/R6.3a operating point),
the "below/near/above p*" recipe from §6 of the revised framework
is not identifiable. Applying the R6.4 `analysis.md` recommendation
of spanning the `[0, 1]` interval:

| ratio_id | text_ratio | rationale |
|---|---|---|
| `ratio_0p2` | 0.20 | image-heavy — most conservative check that fork still wins where image dominates |
| `ratio_0p5` | 0.50 | balanced — mid-range mixture |
| `ratio_0p8` | 0.80 | text-heavy — where fork's text-only PCG gain is dominant |

## Fixed parameters (identical for stock and fork)

- `n_per_ratio`: 100 requests
- `seed`: 42
- text prompts: first N from `datasets/qwen3vl8b/caseA_short.jsonl`
  (`caseA_short` used in R6.2)
- image prompts: fixed pool of 3, cycled deterministically
- image fixture: `R6.1_correctness/fixtures/R6.1_fixture.png`
  (720p PNG, sha256 pinned in R6.1)
- sampling: `temperature=0, top_p=1, seed=42, max_tokens=128`
- request sequence: deterministic Fisher–Yates shuffle of
  `n_text` + `n_image` kind tokens with fixed seed 42;
  **identical order across stock_default and fork_pcg for each
  ratio**

## Acceptance criteria (from `R6_5_verdict.py`, unchanged)

- **hard-FAIL**: any request failure, capture-stream assertion,
  eager fallback, or post-server-ready inference recompile on any
  server across any ratio.
- **agreement requirement**: predicted direction (from analytical
  `p*`) matches empirical direction on at least 2 of 3 ratios.
- **completeness**: at least 3 ratios with rep-level summary
  populated for both variants.

## Predicted direction per ratio

Given `p* = −3.912`, the analytical prediction is `predicted_fork_wins
= (tr >= p*)` = **True for all three ratios** (fork should win at
`p = 0.2`, `0.5`, `0.8`).

R6.5 is a direct empirical falsification test of that prediction.
