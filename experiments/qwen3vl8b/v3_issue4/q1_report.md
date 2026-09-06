# Issue #4 follow-on — Q1: is a text token equivalent to a visual token?

Generated 2026-09-06 07:07 UTC from `results.json, results.json`. Design: [`plan.md` §12.1](../../../plan.md).

Three text-only workloads whose total prefill token count matches an image workload already measured in v3, so half of each comparison already exists. Paired A/B/A/B blocking, transport `cuda_ipc`, c=1.

## Matched-N pairs

| N | text-only | with an image | text-only effect | image effect | difference | within-pair drift |
|---|---|---|---|---|---|---|
| 208 | 27.30 → 16.44 ms | 55.23 ms (v3) | **-39.77%** | -16.30% | -23.47 pp | 2.98% |
| 544 | 27.60 → 23.31 ms | 69.99 ms (v3) | **-15.94%** | -4.54% | -11.40 pp | 3.00% |
| 1024 | 34.39 → 33.95 ms | 104.51 ms (v3) | **-1.30%** | +0.80% | -2.10 pp | 3.00% ⚠ |

Drift here is the spread of the **paired** effects — the resolution floor after A/B/A/B blocking has removed common-mode drift. (Gating the absolute levels instead would reinstate exactly what the pairing removed: `text-208`'s levels move 19.4% across the bracket on a cold-start ramp while its paired effects agree to 3 pp.) ⚠ marks a workload whose spread is comparable to its own effect, which should therefore not be given a sign.

## Absolute saving vs percentage effect

The two readings of "token type is irrelevant" are incompatible, because the image cells have a much larger denominator — the vision encoder's fixed cost, which the graph cannot touch.

| N | text-only saving | image saving (v3) | text-only effect | image effect |
|---|---|---|---|---|
| 208 | **+10.86 ms** | +9.00 ms | -39.77% | -16.30% |
| 544 | **+4.33 ms** | +3.18 ms | -15.94% | -4.54% |
| 1024 | **+0.45 ms** | -0.84 ms | -1.30% | +0.80% |

If the **saving** column matches across a row, the graph recovers the same absolute work regardless of whether the tokens are visual or textual, and the percentage differs only because the image adds fixed cost below it. If the **effect** column matches instead, token count alone fixes the relative benefit. They cannot both match.


---

## Reading

**The saving column matches; the effect column does not.** Across a 5× range of
token counts the absolute saving differs between compositions by a roughly
constant **1.2–1.9 ms**, while the percentage differs by a factor of 2–3.5.

That settles the question in a specific form:

- **Total prefill token count controls what the graph recovers, in milliseconds** —
  and it does so almost independently of whether those tokens are visual or
  textual.
- **Composition controls how large that recovery looks as a percentage**, because
  an image adds preprocessing and vision-encoder time to the denominator that no
  CUDA graph can touch. At N=208 that fixed cost is **23.6 ms**, which is why the
  same ~10 ms of recovered work reads as −39.8% on text and −16.3% on an image.
- The residual ~1.4 ms gap is the part that is genuinely lost rather than diluted:
  the vision encoder's extra GPU work overlaps away some of the launch overhead
  the graph would otherwise have recovered.

### What this does to the v3 report's claim

[`issue4_v3_report.pdf`](issue4_v3_report.pdf) states that "the controlling
variable is the number of prefill tokens, not the image-to-text ratio". That is
**right about the crossover and wrong about the magnitude**, and is corrected to
the two-part statement above. The crossover behaviour survives intact — both
compositions cross zero between N=544 and N=1024 — which is why the v3 operating
guidance does not change.

### Two corrections earned along the way

**The gate was reading the wrong quantity.** The first pass discarded all three
workloads on a 2% drift gate applied to the absolute levels. But `text-208`'s
levels fall **19.4%** across the bracket (a cold-start ramp on a freshly idle
GPU) while its paired effects agree to **2.98 pp** — the common-mode drift that
A/B/A/B blocking exists to cancel, cancelling exactly as designed. Gating on
levels put it straight back in. The gate now reads paired effects and reports
level drift separately as the quantity pairing removed.

**A 24% padding artifact masqueraded as a result.** The first `text-1024` cell
returned **+10.09%** — the graph appearing to cost 10% on a long prompt. The
client's `--random-input-len` excludes the chat template's ~8 tokens, so the
request arrived as **1032** tokens, overshot the 1024 capture bucket by eight,
and padded to **1280**. The graph arm was doing 24% more prefill compute than it
needed. Re-run at 1016 so the server sees 1024, the cell gives **−1.30%**.

The alarm that this might also confound the v3 IMG-A bracket was raised and is
**withdrawn**: every v3 workload lands within 0–6.7% padding, and the image cells
land on 1024 exactly (882 visual + 142 text). v3 is unaffected.

**The general lesson is worth keeping**: near a sparse region of the capture
ladder — the steps are 16–64 tokens below 1024 and 256 above — a request a
handful of tokens above a boundary pays up to 25% padding. A benchmark that
lands there measures padding, not the graph.
