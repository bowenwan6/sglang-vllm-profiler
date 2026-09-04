# Issue #4 v3 — IMG-R ratio sweep (300 prompts, 3 reps, c=1)

Generated 2026-09-04 16:21 UTC. Stack and model: [`manifest.md`](manifest.md). Design: [`plan.md` §11.8](../../../plan.md).

Transport is pinned to `cuda_ipc` on every cell — issue #4's stated standard condition for SGLang image runs — so the only variable is the prefill CUDA-graph backend.

## Bracket validity

Reference cell `R3_720p__disabled` 104.51 ms → repeat 100.74 ms, drift **3.60%** (gate ≤ 5%): **PASS**

**Resolution floor: 3.60%.** Any effect smaller than this cannot be given a sign by this bracket, however tight the per-cell CV looks — the repeat of an *identical* configuration moved by that much.

## Graph effect by workload

| workload | N | vision tok | text tok | disabled | breakable | saving | graph effect | verdict |
|---|---|---|---|---|---|---|---|---|
| `R0_text` | 128 | 0 | 128 | 27.48 ms | 15.16 ms | +12.32 ms | **-44.83%** | **graph wins** |
| `R1_tiny` | 208 | 66 | 142 | 55.23 ms | 46.23 ms | +9.00 ms | **-16.30%** | **graph wins** |
| `R2_360p` | 364 | 222 | 142 | 64.00 ms | 55.03 ms | +8.97 ms | **-14.02%** | **graph wins** |
| `R6_640` | 544 | 402 | 142 | 69.99 ms | 66.81 ms | +3.18 ms | **-4.54%** | no material difference |
| `R7_768` | 720 | 578 | 142 | 78.72 ms | 79.70 ms | -0.98 ms | **+1.25%** | **not resolvable** (below drift) |
| `R8_896` | 928 | 786 | 142 | 101.12 ms | 99.74 ms | +1.38 ms | **-1.37%** | **not resolvable** (below drift) |
| `R3_720p` | 1024 | 882 | 142 | 104.51 ms | 105.34 ms | -0.84 ms | **+0.80%** | **not resolvable** (below drift) |
| `R4_720p_longtext` | 1969 | 882 | 1087 | 135.71 ms | 135.99 ms | -0.28 ms | **+0.21%** | **not resolvable** (below drift) |
| `R5_1080p` | 2184 | 2042 | 142 | 203.92 ms | 206.01 ms | -2.09 ms | **+1.02%** | **not resolvable** (below drift) |

## Answer

**Yes — there are image+text mixes where the prefill CUDA graph pays, and the boundary is measurable.**

- `R0_text` (N=128): **-44.83%**, +12.32 ms
- `R1_tiny` (N=208): **-16.30%**, +9.00 ms
- `R2_360p` (N=364): **-14.02%**, +8.97 ms

The transition sits between **N ≈ 364** (last workload with a material win) and **N ≈ 544** (first workload without one).


---

## Reading of these numbers

Generated output above; interpretation below, kept separate.

### The answer, stated for use

The prefill CUDA graph pays when the LM prefill is short, and images are the
thing that makes it long. Practically, for Qwen3-VL-8B at c=1:

| regime | N | what to do |
|---|---|---|
| text-only, short prompts | ~128 | **turn the prefill graph on** — 45% of TTFT |
| small image (≤ ~360p) + short text | 208–364 | **turn it on** — 14–16% of TTFT |
| ~640×640 + short text | 544 | marginal: −4.5%, resolvable but below the 5% bar |
| 720p and above, any text length | ≥ 720 | **no measurable effect either way** |

Qwen3-VL spends one visual token per 32×32 pixels, so the rule of thumb is the
image's visual-token count rather than its file size: **under ~250 visual tokens
the graph is clearly worth enabling; past ~600 it stops mattering.**

### Why — and what was wrong along the way

The saving in *milliseconds*, not percent, is the quantity that behaves simply:

```
N=128  +12.32    N=208  +9.00    N=364  +8.97    N=544  +3.18
N=720  −0.98     N=928  +1.38    N=1024 −0.84    N=1969 −0.28   N=2184 −2.09
```

It declines monotonically to ~3 ms by N=544 and is inside the noise floor from
N≈720 on. Two explanations were tried and discarded against this data:

1. **`saving = C − k·N`** (a constant benefit minus a per-token DeepStack copy
   cost) — refuted by `R2_360p`: N nearly doubled from 208 to 364 and the saving
   did not move (9.00 → 8.97 ms), where a per-token cost would have removed
   ~6.5 ms.
2. **A ratio story** (benefit tracks the text share of the prompt) — refuted by
   `R4_720p_longtext`: text share went from 13% to 55% at a fixed image and
   nothing changed (+0.80% → +0.21%). The same cell refutes a pure-N story in
   the large-N regime, since N grew from 1024 to 1969 with no deterioration.

The surviving hypothesis is that the graph recovers **only the launch overhead
not already hidden behind GPU execution**. As per-kernel work grows, the CPU-side
launch cost overlaps away and there is progressively less to recover, so the
saving decays toward zero rather than turning sharply negative. It accounts for
the asymptote, for the step when an image first appears (the vision encoder adds
overlappable GPU work: 12.32 → 9.00 ms), and for `R4`'s null result.

**It is a hypothesis, and its quantitative predictions have failed twice.**
`R5_1080p` was predicted at 202–206 ms with saving 0 ± 2 ms and effect within ±1%;
it came in at 206.008 ms, −2.09 ms, +1.02% — three near-misses in the same
direction. `R6_640` was predicted at 4–8 ms of saving; it came in at 3.18 ms,
below the band, because the true decay is convex and the prediction interpolated
linearly. The shape claim (monotone, no cliff) holds; the point predictions do
not, and are not used further.

### Limits on these numbers

1. **The resolution floor is 3.60%**, from the reference cell's repeat
   (104.51 → 100.74 ms). Everything at N ≥ 720 is inside it. The apparent
   non-monotonicity there — `R7` at +1.25%, `R8` at −1.37% — is noise, not
   structure, and none of those cells supports a claim about sign.
2. **The gap-fill sweep carries no drift cell of its own.** `R6`/`R7`/`R8` were
   run as a separate bracket and inherit the main sweep's drift estimate. That is
   a weaker guarantee than the main sweep has.
3. **`R6_640` at −4.54% sits between the two thresholds** — above the 3.60%
   resolution floor, below the 5% materiality bar. The mechanical verdict is "no
   material difference", which is correct, but it is the closest cell to the
   boundary and reads more naturally as the tail of the decline than as a null.
4. **`R0_text` uses the `random` dataset**, not `image`, so its prompt
   construction differs from the other cells. Its role is the within-workload
   contrast, which shares a generator; it is not a point on the same prompt
   family as R1–R8.
5. **All cells are c=1 and `cuda_ipc`.** Batched workloads (#4's c=16 case) and
   CPU transport are untested here; the graph's value under batching is a
   different question and is not answered.
6. **`tc_piecewise` is absent throughout.** #4's hypothesis names PCG
   specifically, and PCG cannot be measured at these run lengths on current
   upstream ([`plan.md` §11.9](../../../plan.md)). Every "graph" result here is
   BCG (`breakable`).

### What this says about issue #4

#4 guessed that "image workloads may behave differently and PCG could help more
there". Both halves need correcting:

- Image workloads **do** behave differently — but the graph helps *less*, not
  more, and the reason is that images add TTFT the graph cannot touch (vision
  encoder and preprocessing: R0→R1 adds 80 tokens but 28 ms).
- The graph does have a clear image+text regime where it pays, just not the one
  #4 assumed: **small images**, where the LM prefill is still short.
