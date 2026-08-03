# Stage-1 T8 — arm self-repeat determinism — GPU 6

**Timestamp:** 2026-08-03T22:55:29Z
**Purpose.** Stage-1 correctness attribution: determine whether each
graph arm (A1/A2/A3) is *internally* deterministic across independent
cold-server bring-ups. If yes, then the Phase-6 A0-vs-A1/A2/A3
divergences are execution-path variation, not runtime nondeterminism.

## Configuration

- Cell: (p=128, b=1), n_warmup=2, n_timed=8, new_tokens=128, greedy.
- 6 sequential runs (A1×2 + A2×2 + A3×2), each a fresh cold-server
  bring-up on GPU 6, rotated ports 30200-30205.
- Client records include `output_ids`, per-token selected logprob,
  and **top-5 alternates + logprobs per T7 (`top_logprobs_num=5`)**.
- A0×A0 not re-run — Phase-5 baseline already established A0 batch=1
  is bit-identical across self-repeats.

## Result

**All arms internally deterministic across cold bring-ups.**

| arm | overall | passed/total | token mismatches | max_lp_diff range |
|---|---|---|---|---|
| **A0** (Phase-5 baseline) | PASS | 8/8 | 0 | [0.0, 0.0] |
| **A1** BCG prefill (T8) | PASS | 8/8 | 0 | [0.0, 0.0] |
| **A2** decode CG (T8) | PASS | 8/8 | 0 | [0.0, 0.0] |
| **A3** both graphs (T8) | PASS | 8/8 | 0 | [0.0, 0.0] |

For every prompt in the golden fixture, both cold bring-ups of the
same arm produced **bit-identical output_ids** and **bit-identical
selected-token logprobs** (max diff = 0.0 to floating-point precision).

Top-5 logprobs per token confirmed present in every record (5
entries per output position — T7 landed correctly).

## Interpretation

Per Stage-1 rubric:

> If an arm is internally stable but differs consistently from A0,
> classify it as deterministic execution-path divergence.

That is the observed condition:
- **Within-arm** determinism: A1 rep1 == A1 rep2 (exact); same for A2, A3.
- **Cross-arm** divergence: from Phase-6, A0 ≠ A1 ≠ A2 ≠ A3 on 1-3 prompts each.

Conclusion: the Phase-6 Gate-1 failures are **deterministic
execution-path divergence** driven by which graphs were captured
and how they were replayed, not runtime nondeterminism. Every graph
arm produces exactly the same output every time; the graph and the
eager path just happen to differ (in expected numerical ways when
non-associative reductions are re-ordered).

This rules out:
- Runtime nondeterminism.
- Request-state contamination between requests within an arm.
- Harness instability across cold-server bring-ups.
- Random port-rotation timing effects on tokens.

**Signal: T8 PASS.** Proceed to T9 (max_new_tokens=1 first-token
comparison) to isolate whether the initial divergence is a
razor-thin greedy-boundary flip.

## Preservation invariants (verified post-T8)

- `/data/sglang-fork` HEAD unchanged: `986c89e69c…`.
- Frozen SGLang HEAD unchanged: `58974ca16c…`, empty `git diff --stat`.
- GPU 6 memory 0 MiB post-run.

## Files

- `selfrepeat_summary.txt` — per-run CSV (6 rows).
- `driver.log` — driver console.
- `<arm>_<rep>/` — 6 subdirs each with `metadata.json`, `gpu_pre.txt`,
  `gpu_post.txt`, `preflight.json`, `runner_*.log`, `client_*.log`,
  `records_<arm>_p128_b1.jsonl` (with top-5 logprobs per token),
  `raw/server_*.log` (gitignored).
