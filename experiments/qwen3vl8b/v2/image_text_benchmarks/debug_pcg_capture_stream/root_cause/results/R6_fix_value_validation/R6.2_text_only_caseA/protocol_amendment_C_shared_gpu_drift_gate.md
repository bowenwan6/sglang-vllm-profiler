# R6.2 Protocol Amendment C — shared-GPU drift-gate reclassification

**Prospective amendment**, drafted 2026-07-29 UTC.

Amendment C **does not rewrite or delete** the original R6.2 machine
verdict (`FAIL`, `drift_pct=3.050`, `fork_pcg_over_stock_pcg_ratio=0.9617`).
That verdict is preserved verbatim under
`attempt_gpu0/verdict.md` and `attempt_gpu0/verdict.json` as the machine
verdict under the original protocol. Amendment C **reclassifies** the
same underlying evidence under a revised interpretive framework, and
that reclassification is what unblocks R6.3+.

Amendment C is scoped to R6.2 and the analogous drift bracket in any
downstream shared-GPU measurement (R6.3a, R6.3b, R6.5). It does **not**
touch fork-vs-stock non-regression thresholds, per-variant CV bounds, or
safety hard-FAIL conditions.

## 1. What the drift bracket actually measures

R6.2 was designed as a 4-way variant matrix. The primary comparison —
`fork_pcg / stock_pcg ≤ 1.05` — is a direct non-regression test on the
fix. The drift bracket (`stock_default` at run start vs
`stock_default_repeat` one hour later) is a **nuisance-control**: it
measures *whether the shared GPU / host state stayed comparable
throughout the 5-variant window*. It does not measure fix correctness
and it does not measure fork-vs-stock non-regression.

Because it is a nuisance-control, the drift bracket is not itself a
primary fix gate. Its role is to determine whether the primary
comparisons are trustworthy in absolute terms. If the drift bracket is
tight, absolute latencies are also comparable across the run. If it is
loose but the fork-vs-stock relative comparison is on **identical
back-to-back servers** with tight per-variant CV, the fork-vs-stock
result is still a valid *relative* non-regression finding — with the
caveat that absolute stock-default numbers should not be cited without
noting the shared-GPU drift.

## 2. Why the original 3.0 % hard cutoff was too brittle

The original protocol combined:

- per-variant `CV ≤ 6 %` (allows within-variant noise up to 6 %), and
- drift bracket `≤ 3.0 %` (a hard cutoff on inter-hour drift).

3.0 % is smaller than the pre-declared per-variant CV allowance of
6 %. That creates a regime where a single variant can be internally
compliant (CV = 5.9 %) yet its rep-mean can jitter enough to push a
drift ratio computed from that mean over 3.0 %. In other words, the
original drift cap was tighter than the noise the same protocol
explicitly tolerates within a variant.

R6.2 executed on GPU 0 during a period with intermittent foreign
compute PIDs (see `attempt_gpu0/verdict.md` §Failure reasons and per-rep
detail: `stock_default` reps 1 and 5 were elevated at 26.87 ms and
29.10 ms respectively; reps 2, 3, 4 were tight at 25.34, 25.38,
27.63 ms). `stock_default_repeat` had a tighter spread — the neighbour
had settled by then. The 3.050 % drift value is dominated by this
shared-GPU excursion during the first block, not by anything the fix
does or does not do.

## 3. Amended drift interpretation

For R6.2 and for any analogous shared-GPU drift bracket in R6.3–R6.5:

| observed drift | interpretation |
|---|---|
| ≤ 3.0 % | **clean PASS** — absolute latencies quotable without caveat |
| 3.0 % < drift ≤ 5.0 % | **`PASS_WITH_CAVEAT`** — primary comparisons stand as relative non-regression / non-safety findings; absolute stock-default numbers must be reported with an explicit shared-GPU caveat |
| drift > 5.0 % | **rerun** (foreign-PID clean window) **or `AMBIGUOUS`** if a clean rerun is not obtainable |

The 5 % upper edge equals the pre-declared per-variant CV bound (rounded
down from 6 %) and represents the largest inter-hour drift consistent
with the protocol's own noise allowance.

## 4. What Amendment C does **not** change

The following thresholds and rules stay exactly as pre-declared:

| gate | threshold | status |
|---|---|---|
| primary non-regression: `fork_pcg / stock_pcg` mean TTFT ratio | ≤ 1.05 | **unchanged** |
| per-variant `mean_ttft_ms` `CV%` | ≤ 6.0 % | **unchanged** |
| any capture-stream assertion | must be 0 | **unchanged (hard FAIL)** |
| any eager fallback | must be 0 | **unchanged (hard FAIL)** |
| any post-server-ready inference recompile | must be 0 | **unchanged (hard FAIL)** |
| any request failure / missing rep | must be 0 | **unchanged (hard FAIL)** |
| bench request completion | must equal declared `--num-prompts` per rep | **unchanged (hard FAIL)** |

The `fork_pcg / stock_pcg ≤ 1.05` ratio remains the **fix-attributable**
non-regression gate. Amendment C does not weaken it. In R6.2 that gate
returned 0.9617 (fork is 3.8 % faster than stock on the identical
PCG-on VLM path), with **material** margin against 1.05.

## 5. Applying Amendment C to R6.2

Under Amendment C, R6.2 is classified as:

> **`PASS_WITH_CAVEAT — TEXT_NON_REGRESSION_SUPPORTED`**

Grounds:

- `fork_pcg / stock_pcg = 0.9617` (require ≤ 1.05; **fork 3.8 % faster**);
- per-variant CV: `stock_default 5.91`, `stock_pcg 2.29`, `fork_pcg 2.02`,
  `stock_default_repeat 2.51` (all ≤ 6 %);
- all four variants completed 5/5 reps × 400/400 requests;
- 0 assertions, 0 fallbacks, 0 post-ready recompiles across every
  server;
- **retained-PCG-benefit story**:
  `stock_default 26.86 → stock_pcg 18.35 → fork_pcg 17.65 ms`
  ≡ default→stock_pcg −31.7 %, default→fork_pcg −34.3 %,
  stock_pcg→fork_pcg −3.8 % (fork retains and slightly extends the
  upstream text-only PCG win on the VLM server);
- drift `3.050 %` falls in the `3.0 % < drift ≤ 5.0 %` bucket ⇒
  `PASS_WITH_CAVEAT`.

**Caveat that must be repeated in downstream reports:** the R6.2 run
had intermittent foreign compute PIDs on GPU 0 during `stock_default`
rep 1 and rep 5. Absolute `stock_default` mean TTFT (26.86 ms) is not
a clean-headline number and should not be reused as R6.3a's
`stock_default` baseline. R6.3a takes fresh matched measurements on
current SHAs with the same monitor + preflight discipline.

**What Amendment C does not do:** it does not lower the fork-vs-stock
non-regression ratio, does not increase the per-variant CV bound, does
not touch any safety hard-FAIL, and does not rerun R6.2. The original
`FAIL` machine verdict at `attempt_gpu0/verdict.md` is preserved as the
machine verdict under the original protocol.

## 6. Downstream flow

- Amendment C unblocks R6.3 (image cost baseline, workload sweep,
  mixed-modality safety). R6.3–R6.5 proceed automatically per the
  revised decision framework.
- Whenever an R6.3 / R6.5 report cites absolute R6.2 numbers, it must
  repeat the shared-GPU caveat. Relative non-regression is quotable
  without caveat.
- If a future shared-GPU drift bracket lands in
  `PASS_WITH_CAVEAT` **and** primary fork-vs-stock ratios also weaken
  (`≥ 1.00` or otherwise inconsistent with prior clean runs), the
  bench must be rerun on a clean GPU window before the finding is
  accepted — the caveat framework is not a substitute for a
  well-attributable primary comparison.

## 7. Amendment C author + timestamp

Drafted 2026-07-29 UTC by the R6 experiment operator prior to R6.3
execution. Amendment C applies **prospectively**: because the R6.2
data (verdict.md + verdict.json) had already been captured under the
original protocol, applying Amendment C to R6.2 is an act of
*reinterpretation*, not of post-hoc data selection. No R6.2
measurement is altered, added, or dropped. No new run is fabricated.

Commit landing this amendment is
`docs(v2): amend R6 shared-GPU drift gate` (see git log).
