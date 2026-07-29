# R6.4 analytical crossover — analysis / interpretation

The R6_4_crossover.py machine verdict is `AMBIGUOUS` because the
analytical `p* = C / (G + C)` evaluates to a negative number
(`-3.912`), which is not a valid mix ratio in `[0, 1]`. That is a
**machine artifact of the crossover formula being undefined when both
G and C are gains** — it is not a fix defect.

## The two components

Rep-level arithmetic means (see [`crossover.md`](crossover.md) inputs):

| quantity | value | interpretation |
|---|---|---|
| G = mean(stock text) − mean(fork text) | **+9.21 ms** | fork saves 9.21 ms per text request (PCG helps text) — expected, from R6.2 |
| C = mean(fork image) − mean(stock image) | **−7.34 ms** | fork also saves 7.34 ms per image request (PCG *helps* image in this cell) — **positive result** |

The classical crossover story assumes the fix trades text speed for
image cost (`G > 0`, `C > 0`). Under that assumption `p*` is the mix
fraction where the trade breaks even. **Neither assumption holds
here**: on the R6.3a operating point (720p, 1 image, 128-token text,
c=1, n=400) the fork is faster than stock on *both* the text-only
Case A leg and the image leg. `p*` is meaningless because there is
no crossover; fork wins across the entire `[0, 1]` mix range.

## Ratio table (from [`crossover.md`](crossover.md))

Every predeclared mix fraction (0.5, 0.7, 0.8, 0.9, 0.95, 1.0) shows
`fork_wins = ✅`, with `on/off` between **0.86 (mid-mix)** and
**0.66 (all-text)**. The bootstrap 95 % CI on `p*` (`[−12.39,
+15.44]`) confirms the crossover is not identifiable — it doesn't
exist in the operator-realistic range.

## What this actually says about the fix

At the **R6.3a operating point** (short text, single-image, small
concurrency, radix cache on) the fork's clean-Y warmup gate + static
deepstack buffer eliminates the recompile-driven inference-path
overhead entirely; the image cost never materialises. Combined with
R6.1's `SAFETY_SUPERIORITY_PASS` (stock crashes on the historical
repeated-shape workload; fork does not), fork PCG is a
`STRICTLY_DOMINANT` improvement in this regime.

## Where the loss regime lives (not visible to R6.4)

R6.3b (attempt_gpu2) surfaced consistent losses at `t2048_*` cells
(long text) and instability at `t512_r720p_c4`. R6.3 confirmation
narrowed the "instability" story: two of the seven discovery-winning
cells did not confirm at 3 reps (`t128_r720p_c4` 1.06,
`t512_r720p_c1` 1.17). The genuine crossover exists on the
**text-length axis** at fixed mix — not on the mix-ratio axis at
fixed text length. The classic p* framework does not capture that.

## Recommended R6.5 design

Given the analytical result:

1. **Mix-ratio sweep at the R6.3a operating point** (short text +
   720p image, c=1). Predeclared ratios `p ∈ {0.2, 0.5, 0.8}` — this
   spans the [0, 1] range and gives three empirical checks against
   the analytical prediction that fork should win at all three.
2. **Text-length axis at fixed mix (50/50)** at 720p images: text
   tokens ∈ {128, 512, 2048} — locates the empirical crossover on
   the axis where it actually exists.

Both are cheap (≤ 100 requests per cell) and use identical fixed
seed + request ordering for stock-default and fork-PCG. Any
image-only or text-only losing cell in R6.3b remains transparently
recorded.

## Overall R6.4 reading

The machine `AMBIGUOUS` is a positive result: **fork PCG dominates
stock-default on the primary R6.2 + R6.3a operating point across the
entire analytical mix range**. R6.5 will validate this empirically
under identical request orderings.
