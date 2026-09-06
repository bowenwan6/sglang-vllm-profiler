# Issue #4 follow-on — Q2: the request-stream mix

Generated 2026-09-06 08:19 UTC. Design: [`plan.md` §12.2](../../../plan.md).

Two bench clients against one server at once — one text-only, one image — each with its own Poisson arrival rate, so the image share of arrivals is set by their ratio at a fixed total rate. Launching them together is the point: a sequential run would never co-batch.

Observed in-flight requests: mean **5.55** across cells (target was a mean of ~8; Poisson arrivals make this a distribution, so the measured value is reported rather than the target).

## Text-class requests

| image share of arrivals | graph off | graph on | graph effect | spread | blocks |
|---|---|---|---|---|---|
| 0 | 35.66 ms | 30.13 ms | **-15.14%** | 0.76 pp | 3 |
| 0.2 | 35.78 ms | 30.63 ms | **-13.84%** | 3.10 pp | 3 |

Change in the graph's benefit to text requests as images enter the stream, measured against the image-free case:

- at f=0.2: **+1.30 pp** (-15.14% → -13.84%)

A value near zero means co-batching does not erode the benefit and the aggregate is a simple weighted average of the homogeneous results. A large negative shift is the interference this experiment exists to detect.

## Image-class requests

| image share of arrivals | graph off | graph on | graph effect | spread | blocks |
|---|---|---|---|---|---|
| 0.2 | 108.92 ms | 112.44 ms | **+3.86%** | 3.24 pp | 3 |
| 1 | 112.86 ms | 119.76 ms | **+4.81%** | 2.43 pp | 3 |

## The same result under two metrics

Everything above is time-to-first-token, where #2 and #4 located the gap. End-to-end latency is what a user waits for, and here it is decode-dominated — about 870 ms of ~910 — so a few milliseconds of TTFT dilute to well under a percent.

| image share | class | TTFT | end-to-end | TPOT |
|---|---|---|---|---|
| 0 | text | -15.14% | **-3.72%** | -3.04% |
| 0.2 | text | -13.84% | **-3.45%** | -2.49% |
| 0.2 | image | +3.86% | **-1.45%** | -0.32% |
| 1 | image | +4.81% | **+3.43%** | +2.73% |

## A confound this design could not avoid

| image share | mean in-flight requests |
|---|---|
| 0 | 4.79 |
| 0.2 | 4.67 |
| 1 | 7.20 |

The arrival **rate** was held fixed, not the **load**. Image requests take longer to serve, so a higher image share at the same rate produces a busier server. `f = 0` and `f = 0.2` land within 3% of each other and are comparable; `f = 1` runs about 54% busier and differs from the others in *two* ways at once.

**Consequence:** the image class's end-to-end effect moves from −1.45% at f = 0.2 to +3.43% at f = 1, and that difference **cannot be attributed** — composition and load both changed. Fixing it means tuning the arrival rate per fraction to equalise in-flight requests, which is a follow-up, not a reinterpretation of this data.

**TTFT, at ~4.7 in flight: break-even at f ≈ 0.54** (text saves 4.92 ms, image costs 4.20 ms per request).
  Using instead the image cost measured on the pure-image stream (5.49 ms at ~7.2 in flight, 600 requests rather than 120) gives f ≈ 0.47 — but that mixes two operating points and is quoted only to bound the range.

**end-to-end latency, at ~4.7 in flight: no break-even.** The graph is faster for *both* classes — text by 31.31 ms and image by 14.57 ms per request — so it pays at every image fraction **at this load**.
  Using instead the image cost measured on the pure-image stream (42.44 ms at ~7.2 in flight, 600 requests rather than 120) gives f ≈ 0.42 — but that mixes two operating points and is quoted only to bound the range.


---

## Reading

### What is answered cleanly

**Adding images to a text stream does not take the graph's benefit away from the
text requests.** `f = 0` and `f = 0.2` run at the same load (4.79 vs 4.67 in
flight), so this comparison is clean: the graph's benefit to text moves from
−15.14% to −13.84%, a 1.30 pp shift inside the 3.10 pp block spread.

The mechanism this experiment was built to detect is present but negligible.
Recovering prefill-batch composition from the server logs:

| cell | multi-request batches | cross-class | share of all |
|---|---|---|---|
| `f=0.2 disabled` | 14 (2.3%) | 0 | 0.0% |
| `f=0.2 breakable` | 9–12 (1.5–2.0%) | 1–4 | **0.2–0.7%** |

At this load **97.7% of prefill batches carry a single request**, so a text
request almost never shares a batch with an image one. Co-batching cannot erode
what it barely touches. **The weighted average of homogeneous measurements is
therefore valid here** — which had been an assumption until now, and is now a
measurement.

The precise claim is "too rare to matter **at this load**", not "does not
occur". Establishing whether co-batching is harmful when batches routinely
combine needs a load where they do; that is a follow-up, and this bracket cannot
speak to it.

### The deployment answer, and why it has two numbers

| metric | at ~4.7 in flight |
|---|---|
| **TTFT** | break-even at **f ≈ 0.54** (0.47 if the busier image cost is used) |
| **end-to-end** | **no break-even** — the graph is faster for both classes |

Both are true; they measure different things. End-to-end here is ~910 ms of which
decode is ~870, so a 4 ms TTFT change dilutes below a percent while a small
consistent TPOT difference over 128 output tokens is worth ~30 ms. For the text
class, e2e improves by 31.31 ms of which only 5.14 ms is TTFT.

**Reporting only TTFT would understate the graph's value; reporting only
end-to-end would hide that image requests genuinely wait longer for their first
token.** Both belong in the recommendation.

**For the question that prompted this work** — a deployment where users attach an
image now and then, f ≈ 0.05–0.2 — the graph pays on both metrics by a wide
margin, and the break-even is far away.

### What this bracket cannot say

**Load and image fraction are confounded**, because the arrival *rate* was fixed
rather than the *load*. `f = 1` runs 54% busier than the others, so the image
class's end-to-end effect flipping from −1.45% to +3.43% between f = 0.2 and
f = 1 has two possible causes and this data separates neither. Any statement
about high image fractions must carry that caveat, and the fix — tuning the
arrival rate per fraction to equalise in-flight requests — is a follow-up.

**A decode-side effect is visible and unexplained.** TPOT is consistently better
under `breakable` at ~4.7 in flight (−0.3% to −3.0% across every cell and both
classes) and worse at ~7.2 (+2.7%). A prefill CUDA graph should not touch decode,
and both arms run the identical decode backend. The plausible route is indirect —
prefill issuing far fewer kernel launches leaves CPU headroom for the decode loop,
until higher load changes the balance — but that is a hypothesis, and the
end-to-end conclusions rest on it more than the TTFT ones do.

### One v3 caution now superseded

The v3 report recorded the image-class graph effect as **not resolvable**, since
it fell inside that bracket's 3.60% floor. Three independent brackets now agree:

| bracket | concurrency | image requests | effect |
|---|---|---|---|
| v3 IMG-A | 1 | 2000 | +3.66% |
| Q2 `f = 0.2` | ~4.7 | 120 | +3.86% |
| Q2 `f = 1.0` | ~7.2 | 600 | +4.81% |

A single bracket's resolution floor is not the floor on accumulated evidence.
The sign is established: **the prefill graph costs image requests roughly 4% of
their time to first token.**
