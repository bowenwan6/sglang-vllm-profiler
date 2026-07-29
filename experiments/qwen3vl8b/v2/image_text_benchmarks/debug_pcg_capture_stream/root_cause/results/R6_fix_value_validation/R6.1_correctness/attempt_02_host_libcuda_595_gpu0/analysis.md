# R6.1b attempt 02 — forensic analysis

> **Read-only forensic analysis** of the already-committed verdict.md / verdict.json under this directory. Extends the pre-declared machine verdict (FAIL) with token-level comparisons, equivalence classes, and cache-state provenance. Does **not** modify the machine verdict.

## Server launch + leg order (cache-state provenance)

- Radix cache: **enabled (sglang default; no --disable-radix-cache flag)**
- Servers were serialized on GPU 0; each was a fresh process (cold radix cache at startup). Within a server, later legs saw whatever prefix cache the earlier legs left.

| Server | PID/PGID | Legs (in order) |
|---|---|---|
| `stock-default` | 165003 | c → f1 |
| `fork-default` | 166180 | a1 → a2 |
| `stock-pcg` | 167376 | d → f2 |
| `fork-pcg` | 168759 | b → dp → e |

> Each variant ran on a fresh server (radix cache empty at server startup). WITHIN a single server, the SECOND leg saw a warm cache populated by the first leg's prompt tokens if any overlapped. Specifically: a2 ran after a1 on the SAME fork-default server, so a2 saw whatever prefix cache state a1 left. This is the 'cold vs warm' confound the R6 refinement instructions call out.

## Per-leg summary

| Leg | Requests | HTTP 200 | Errors | Mean latency (s) | Char lens | Tok lens | Finish |
|---|---|---|---|---|---|---|---|
| `a1` | 3 | True | False | 0.3706 | [194, 267, 298] | [39, 63, 71] | ['stop', 'stop', 'stop'] |
| `a2` | 3 | True | False | 0.3412 | [194, 267, 295] | [39, 63, 70] | ['stop', 'stop', 'stop'] |
| `b` | 3 | True | False | 0.4842 | [194, 267, 291] | [39, 63, 70] | ['stop', 'stop', 'stop'] |
| `c` | 3 | True | False | 0.6253 | [194, 267, 298] | [39, 63, 71] | ['stop', 'stop', 'stop'] |
| `d` | 3 | True | False | 0.2385 | [107, 387, 2] | [21, 84, 2] | ['stop', 'stop', 'stop'] |
| `dp` | 3 | True | False | 0.1989 | [107, 372, 2] | [21, 83, 2] | ['stop', 'stop', 'stop'] |
| `e` | 6 | True | False | 0.2655 | [107, 194, 387, 267, 2, 298] | [21, 39, 84, 63, 2, 71] | ['stop', 'stop', 'stop', 'stop', 'stop', 'stop'] |
| `f1` | 3 | True | False | 0.2106 | [107, 387, 2] | [21, 84, 2] | ['stop', 'stop', 'stop'] |
| `f2` | 3 | True | False | 1.6021 | [107, 417, 2] | [21, 91, 2] | ['stop', 'stop', 'stop'] |

## Pairwise comparison matrix

Each cell: char equal? token equal? then (common-prefix tokens / first-differing token / normalized token Levenshtein).

### `a1_vs_a2` — image  (a1 vs a2)
- Provenance: same-server SECOND leg (a2 saw warm cache from a1)

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 39 | None | 0 / 0.0 | (194, 194) | (39, 39) |
| 1 | True | True | 63 | None | 0 / 0.0 | (267, 267) | (63, 63) |
| 2 | False | False | 60 | 60 | 4 / 0.0563 | (298, 295) | (71, 70) |

### `a1_vs_c` — image  (a1 vs c)
- Provenance: different SERVERS (a1=fork-default, c=stock-default) — both first-leg-on-fresh-server; matched cold cache

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 39 | None | 0 / 0.0 | (194, 194) | (39, 39) |
| 1 | True | True | 63 | None | 0 / 0.0 | (267, 267) | (63, 63) |
| 2 | True | True | 71 | None | 0 / 0.0 | (298, 298) | (71, 71) |

### `a1_vs_b` — image  (a1 vs b)
- Provenance: different SERVERS (a1=fork-default first leg, b=fork-PCG first leg on fresh server) — matched cold cache but different PCG mode

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 39 | None | 0 / 0.0 | (194, 194) | (39, 39) |
| 1 | True | True | 63 | None | 0 / 0.0 | (267, 267) | (63, 63) |
| 2 | False | False | 60 | 60 | 2 / 0.0282 | (298, 291) | (71, 70) |

### `c_vs_b` — image  (c vs b)
- Provenance: different SERVERS (c=stock-default first leg, b=fork-PCG first leg on fresh server) — matched cold cache

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 39 | None | 0 / 0.0 | (194, 194) | (39, 39) |
| 1 | True | True | 63 | None | 0 / 0.0 | (267, 267) | (63, 63) |
| 2 | False | False | 60 | 60 | 2 / 0.0282 | (298, 291) | (71, 70) |

### `d_vs_dp` — text  (d vs dp)
- Provenance: different SERVERS (d=stock-PCG first-leg-text, dp=fork-PCG SECOND-leg-text after b) — CACHE STATE MISMATCH: d cold, dp saw b's cache

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 21 | None | 0 / 0.0 | (107, 107) | (21, 21) |
| 1 | False | False | 15 | 15 | 42 / 0.5 | (387, 372) | (84, 83) |
| 2 | True | True | 2 | None | 0 / 0.0 | (2, 2) | (2, 2) |

### `f1_vs_f2` — text  (f1 vs f2)
- Provenance: different SERVERS (f1=stock-default SECOND leg after c; f2=stock-PCG SECOND leg after d) — both saw prior-leg cache but from different prompts

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 21 | None | 0 / 0.0 | (107, 107) | (21, 21) |
| 1 | False | False | 15 | 15 | 42 / 0.4615 | (387, 417) | (84, 91) |
| 2 | True | True | 2 | None | 0 / 0.0 | (2, 2) | (2, 2) |

### `d_vs_f2` — text  (d vs f2)
- Provenance: same SERVER stock-pcg (d first leg cold; f2 second leg warm) — natural cold-vs-warm PCG delta

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 21 | None | 0 / 0.0 | (107, 107) | (21, 21) |
| 1 | False | False | 15 | 15 | 42 / 0.4615 | (387, 417) | (84, 91) |
| 2 | True | True | 2 | None | 0 / 0.0 | (2, 2) | (2, 2) |

### `f1_vs_dp` — text  (f1 vs dp)
- Provenance: different SERVERS (f1=stock-default warm; dp=fork-PCG warm) — best available fork-vs-stock text comparison but with different warm-cache contents

| Prompt | char== | tok== | tok_common_prefix | tok_first_diff | tok_lev / norm | char lens (a,b) | tok lens (a,b) |
|---|---|---|---|---|---|---|---|
| 0 | True | True | 21 | None | 0 / 0.0 | (107, 107) | (21, 21) |
| 1 | False | False | 15 | 15 | 42 / 0.5 | (387, 372) | (84, 83) |
| 2 | True | True | 2 | None | 0 / 0.0 | (2, 2) | (2, 2) |

## Equivalence classes per prompt

### Image prompts (across a1, a2, b, c)

- Prompt 0: 1 class(es) → {a1, a2, b, c}
- Prompt 1: 1 class(es) → {a1, a2, b, c}
- Prompt 2: 3 class(es) → {a1, c}; {a2}; {b}

### Text prompts (across d, dp, f1, f2)

- Prompt 0: 1 class(es) → {d, dp, f1, f2}
- Prompt 1: 3 class(es) → {d, f1}; {dp}; {f2}
- Prompt 2: 1 class(es) → {d, dp, f1, f2}

## Confirmed facts

These follow directly from the recorded artifacts and require no additional runs to establish:

1. **All 4 servers launched and served every leg with HTTP 200** across every request. No 5xx / connection errors / request-side timeouts. See per-leg summary table.
2. **stock-default vs fork-default on image is bit-identical on every prompt** (`a1_vs_c` all `char==True, tok==True`, common_prefix equals leg length). The fix is a demonstrable no-op when PCG is off.
3. **No PCG capture-stream assertion, no eager-fallback warning, no request failure** was observed on the fork-PCG server during the mixed-modality interleaved sequence (leg e, six requests, all 200).
4. **Every Dynamo recompile of `qwen3_vl.forward` in the fork-PCG server log occurred before server-ready** — recompile lines at 30, 53, 158, 193; server-ready marker at line 570 in a 613-line log. Post-server-ready recompile count: 0.
5. **Same-server sequential-repeat (a1 → a2) on fork-default diverges on prompt 2** (`a1_vs_a2` prompt 2 differs). The second request saw a warm radix cache populated by the first request. This is a *cache-state confound*, not a same-input equality test.
6. **Stock-PCG survived the R6.1 protocol on the frozen environment**: `d` and `f2` both returned 3 × HTTP 200 with no assertion — but the R6.1 protocol never issued an IMAGE request to stock-PCG, so the historical first-image capture-stream failure was neither reproduced nor ruled out. That is the direct negative control the refinement instructions call out.

## Open hypotheses (require additional evidence)

These are consistent with the recorded artifacts but have **not** been proven and must not be asserted as facts:

1. **The leg `a1_vs_a2` divergence is server-level non-determinism (radix cache order + prefix reuse)** — plausible given radix caching is enabled and a2 ran immediately after a1, but no matched cold-cache repeat is in evidence. Also consistent with the divergence being a different (currently unknown) source.
2. **The `a1_vs_b` (fork-default vs fork-PCG image) divergence is the same non-determinism (not a PCG-vs-eager delta)** — plausible because a1 and b diverge at the SAME prompt (idx=2) and roughly the same first-diff character offset (241 for both), and both a1 and b are the FIRST leg on their respective servers (cold cache). Still, cold-cache-matched leg-a1 differs from cold-cache-matched leg-b by the same magnitude as a1 differs from a2 (warm), which needs an independent same-cold-cache repeat to disentangle.
3. **The `d_vs_dp` (stock-PCG vs fork-PCG text) divergence is server-level non-determinism, not a fork-specific text-PCG effect** — plausible but the cache states are *mismatched*: `d` is stock-PCG's first leg (cold), `dp` is fork-PCG's second leg (warm after `b`). A matched cold-vs-cold text repeat is the missing evidence.
4. **The `f1_vs_f2` (stock-default text vs stock-PCG text) divergence characterizes the natural PCG-vs-eager delta** — plausible; both are the second leg on their servers, so cache-warmth mismatch is smaller. Still needs a matched cold-cache stock-eager-vs-stock-PCG text repeat to be authoritative.
5. **stock-PCG would still crash on the first image request** — historically observed (see Issue #4 debug), but attempt 02 did not run this specific negative control on the current frozen stock SHA `da802dd`. The refinement calls this out explicitly as `EXPECTED_STOCK_FAILURE` vs `STOCK_NOW_SURVIVES` vs `UNRELATED_FAILURE`.

## Summary of what attempt 02 does and does not establish

**Provides strong evidence for**:
- Operational safety: fork-PCG survives interleaved text/image traffic on Qwen3-VL with 0 crashes / 0 assertions / 0 fallbacks / 0 request failures / 0 post-server-ready recompiles.
- Zero side effect when PCG is off: `stock-default` and `fork-default` produce byte-identical outputs on every image prompt.

**Does NOT establish (and cannot be dismissed)**:
- Whether cross-config output divergences (`a1_vs_b`, `d_vs_dp`) are cache-state non-determinism or real fork-vs-stock effects. Requires *matched cache state* and *paired same-config repeats*.
- Whether the previous behavior — stock-PCG crashing on the first image — still reproduces on the current frozen stock SHA. Requires the negative-control leg the refinement adds.

**Corrections to the old attempt-02 verdict framing**:

- The old `inference_recompiles == 4` reason **incorrectly counted warmup recompiles as if they were inference recompiles**. Phase-split evidence (all 4 lines < server-ready line) shows the true inference-recompile count is 0. The metric must be sharpened to only count post-`SERVER_READY` recompile events per the refinement instructions.
- The old `(a) fork-default same-run repeat NOT bit-identical` reason **compared different cache states** (cold prompt 2 vs warm prompt 2 within one server). It is *not* a determinism baseline; it is a cache-state confound. The refinement replaces it with cache-matched repeats.
- The `(d) stock-PCG text != fork-PCG text` divergence **remains unresolved** and must not be attributed to any specific cause (fork effect, PCG-vs-eager noise, or cache-state confound) without additional matched repeats.
