# Attempt 03 interpretation addendum (2026-07-28)

> The machine outputs in this directory
> (`verdict_amended.md`, `verdict_amended.json`) **stand as recorded**.
> This addendum corrects only the *higher-level interpretation* attached
> to them by the prior test commit `ac7fdfc` — specifically the claim
> that "an upstream commit fixed the bug." That claim is **not
> supported**; withdrawn here.
>
> No verdict rule is amended. Amendment A verdict logic remains as
> declared. What follows is a corrected forensic reading of the same
> raw evidence.

## Correction summary

**Attempt 03 negative control (`STOCK_NOW_SURVIVES`) is
`INCONCLUSIVE_TRIGGER_NOT_REPRODUCED`, not evidence of an upstream
fix.** The historical stock-PCG-crashes-on-first-image bug requires
a *repeated call to the same concrete runtime shape* after the
multimodal recompile; attempt 03 issued only three image requests,
each with a distinct prefill token count, so the runtime never
reached that repeated-shape call.

## Evidence (all sourced from existing committed artefacts)

### 1. Historical R1 crashed on the SAME stock SHA that attempt 03 used

- R1 summary — `results/R1_dynamo_recompile_log/summary.md` §1
  "Run identity": `sglang | 0.0.0.dev1+gda802ddca (/sgl-workspace/sglang)`.
- Attempt 03 launch context —
  `attempt_03_amended_A_gpu0/verdict_amended.json`
  `.launch_context.sglang_stock_head =`
  `da802ddcafe55e25b3e1db86b1e0444afc3e05bc`.

The two are the same stock SHA. So the "upstream commit between old
HEAD and current HEAD fixed it" narrative is impossible — there is
no commit between `da802ddca` and `da802ddca`.

### 2. Current stock still contains the assertion

`/sgl-workspace/sglang/python/sglang/srt/compilation/cuda_piecewise_backend.py`
line 170–172 (verified 2026-07-28):

```python
assert (
    stream is not None
), "PCG capture stream is not set, please check if runtime recompilation happened"
```

The source path that triggers the historical crash is unchanged.

### 3. R1 already characterised the trigger as requiring a repeated shape

R1 summary — `results/R1_dynamo_recompile_log/summary.md` §2 last
paragraph explicitly documents that after the four Dynamo recompiles
`[0/1]` – `[0/4]`, "~9 s of mixed prefill batches succeed
(`cuda graph: True` on lines 43954–43988). Then the assertion
fires…". R2 (`results/R2_pcg_call_trace/summary.md`) confirmed the
asserting instance is at `runtime_shape=1024, entry.compiled=False,
entry.num_finished_warmup=1`. Combined:

- The recompile creates a new `CUDAPiecewiseBackend` object per
  Dynamo frame.
- The *first* call to that backend at a concrete `runtime_shape`
  is treated as a soft warmup pass (`entry.num_finished_warmup` is
  incremented; no capture-stream assertion).
- The assertion fires on the *second* call to that same
  `runtime_shape` when the still-empty capture stream is discovered.

Historical R1 achieved this because its recipe was `n=32` requests
on the same 720p image with `--random-input-len 128 --random-output-len
128 --random-range-ratio 1.0`, which produces effectively the same
prefill shape on every request. With 32 requests, the same shape
was called many times → assertion reached.

### 4. Attempt 03 shapes did NOT repeat

Attempt 03's neg-control `stock-PCG` server served 3 image prompts:

| prompt idx | prompt text (from `fixtures/prompts.json`) | approx text tokens | prefill shape ≈ text + image + template |
|---|---|---|---|
| 0 | "Describe this image in one sentence." | small | one shape |
| 1 | "What colors are visible in the image?" | small (~same as 0) | a second shape |
| 2 | "How many distinct regions of solid color are present, and what colors are they?" | larger | a third shape |

The three prompts have different total prefill token counts and
therefore different piecewise `runtime_shape` values (each in the
approximately 895–905 range once the ~882 image tokens plus the chat
template are counted; the exact per-shape values live in
`raw/neg_stock_pcg_image_server.log` and can be re-extracted in
Amendment B). No prompt was issued twice; no prefill shape was
reached twice. The `entry.num_finished_warmup=1 → 2` step that
triggers the assertion never occurred.

### 5. All four Dynamo recompiles observed in attempt 03 neg log are pre-server-ready warmup

Grep count of `Recompiling function` in
`raw/neg_stock_pcg_image_server.log`: **4** — timestamps
13:47:05 – 13:47:37, all before HTTP-ready. Recompile signature
matches R1's `[0/1] – [0/4]` cascade. So the *recompile* half of the
historical bug happened; the *repeated-shape-after-recompile* half
did not, because attempt 03's workload didn't repeat shapes.

## Restated attempt 03 classification

- **Tier 1 SAFETY_SUPERIORITY**: withdraw the `STOCK_NOW_SURVIVES`
  → "upstream fix" reading. Correct classification is
  `INCONCLUSIVE_TRIGGER_NOT_REPRODUCED`. Attempt 03's negative
  control was under-powered: it exercised the recompile but not the
  repeated-shape post-recompile call. Amendment B (2026-07-28) adds
  a repeated-shape control that will properly test the historical
  trigger.
- **Tier 2 CORRECTNESS**: unchanged — PASS. Attempt 03's matched
  cold-cache correctness comparisons remain valid; they don't
  depend on repeated shapes.
- **Overall R6.1**: still FAIL (the pre-declared Amendment A rule
  requires `EXPECTED_STOCK_FAILURE` for safety PASS, and this
  attempt's neg control does not qualify — but for the corrected
  reason "trigger not reproduced", not "trigger no longer exists").
  R6.2 – R6.5 remain blocked.

## What must NOT be claimed from attempt 03 alone

- "Upstream fixed the bug." → **Not supported.** No specific commit
  or source change has been identified, and the stock source still
  contains the assertion on the same SHA that historically crashed.
- "Fork's clean-Y is functionally redundant." → **Not supported.**
  The redundancy claim depended on the upstream-fix claim.
- "Stock-PCG now serves image requests cleanly." → **Only in the
  narrow, misleading sense that** stock-PCG serves *three
  different-shape* image requests without hitting the second-shape
  assertion. Whether stock-PCG survives a sustained same-shape
  workload is what Amendment B tests.

The Amendment A tier-1 rule is still evaluable; attempt 03 simply
did not run the negative control that would actually exercise it.
Amendment B fixes that.

## What this addendum does NOT change

- The machine-generated `verdict_amended.md` / `verdict_amended.json`
  files in this directory. They record the outputs of the
  pre-declared rules as they applied to attempt 03's specific
  observations, and they are correctly `FAIL / NONE` under those
  rules. This addendum is a *narrative* correction of the
  interpretation attached to that machine verdict in the prior
  test commit — it does not re-label the machine verdict.
- Attempt 03 raw evidence under `raw/` remains untouched.
- Amendment A itself
  (`../protocol_amendment_A_direct_fix_comparison.md`) remains as
  authored; it is Amendment B that adds the repeated-shape control.
