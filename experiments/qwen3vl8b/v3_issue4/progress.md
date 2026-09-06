# Issue #4 v3 — execution log

Executes [`plan.md`](../../../plan.md) §11. One row per step, with a status:

- **Accepted** — ran as intended, nothing anomalous.
- **solvable** — hit a problem, resolved it with the engineering-correct choice; the choice and its reasoning are recorded here.
- **Fail** — blocked on a decision that is not mine to make; needs the owner.

GPU: **7 only**. Stack frozen in [`manifest.md`](manifest.md).

---

## Phase 0 — desk work, no GPU

### 0.2 — SGLang SHA policy vs #33726 merge state — **solvable**

Checked live: PR #33726 is **open, not merged** (`merged=false`, head `32dbab0bb1`).

Plan's literal instruction for that branch was "pin a pre-merge SHA". Following it
literally would have made `A3_bcg` unmeasurable: on unpatched upstream,
`Qwen3VLForConditionalGeneration` is not on
`multimodal_breakable_cuda_graph_supported_model_archs`, and forcing the backend
bypasses the auto-disable cascade but runs straight into the DeepStack replay bug
#33726 exists to fix. The arm would have produced a plausible latency number
attached to numerically wrong output.

**Resolved** by pinning one merged-preview stack — `exp/issue4-v3` =
`upstream/main` @ `ff1285cc28` + the PR branch, merged clean — and reaching both
worlds by explicit flags instead of by the default:
`A0_default` → resolves to `breakable` (post-merge default);
`A1_disabled` → today's actual production behaviour for this arch.
The 0.2 intent — never straddle the merge inside one bracket — is preserved.
Full reasoning: [`manifest.md`](manifest.md) §1.

### 0.1 — environment manifest, rebuild vs stale — **solvable**

`upstream/main` pins `torch==2.13.0` / `transformers==5.12.1` /
`sglang-kernel==0.4.6.post1` (the `sgl-kernel` package was renamed). The box has
torch `2.11.0+cu130`.

Honouring the pins means replacing torch, which would also invalidate the vLLM
0.21.0 environment that shares that torch build — destroying the cross-framework
anchor the bracket exists to produce.

**Resolved** as run-from-source via `PYTHONPATH` (pyproject pins are not enforced
on that path); the recipe is the one already proven on this box by the M10 smoke
on 2026-08-29. The staleness confound is accepted, applies identically to every
arm, and is stated in the report: internal contrasts hold, absolute numbers are
not production claims.

Also settled here, all verified rather than assumed:

- Model **Qwen3-VL-8B-Instruct** at revision `0c351dd01e` — the exact revision
  v2's protocol pinned, re-fetched (17 GB). Chosen over the already-cached 4B so
  the #2 → #4 baseline line stays continuous.
- **Feature engagement precondition met**: `deepstack_visual_indexes = [8,16,24]`,
  `out_hidden_size = 4096` ⇒ replay width 12288 > 0. The target genuinely
  populates the path under test. (Being on an allowlist is not the same as using
  the feature; this is the check that distinguishes them.)
- GPU 7 idle at 212 MiB.
- vLLM anchor env present: `0.21.0`, `/opt/miniconda3/envs/profiling/bin/python`.

Manifest: [`manifest.md`](manifest.md).

### 0.3 — port the runner to the v3 flag surface — **Accepted**

[`scripts/run_imgA_v3.py`](scripts/run_imgA_v3.py). An edit of the v2 runner, as
the plan intended: bracket ordering, drift gating, forbidden-token guards, GPU
idle checks and artifact layout carried over; variant matrix, flag surface and
engagement capture replaced.

Every v3 flag was verified to exist on the pinned stack before being written into
the runner, rather than assumed from the audit:

| Surface | Verified at |
|---|---|
| `--mm-feature-transport` (`Optional[Literal["cpu","cuda_ipc","cuda_vmm"]] = None`) | `server_args.py:2954` |
| unset ⇒ `cpu` coercion | `multimodal/processors/base_processor.py:247-252` |
| `--cuda-graph-backend-prefill` | `arg_groups/cuda_graph_hook.py:75-76` |
| `sglang.benchmark.serving` is the real module; `sglang.bench_serving` is a `FutureWarning` shim | `python/sglang/benchmark/serving.py`, `python/sglang/bench_serving.py:1-21` |
| image flags survive (`--image-count/-resolution/-format/-content`, `720p` preset) | `benchmark/serving.py:2342-2375` |
| `--num-prompts`, `--max-concurrency`, `--random-range-ratio`, `--output-file` | `benchmark/serving.py` |

`SGLANG_USE_CUDA_IPC_TRANSPORT` is now actively **stripped** from the runner's
base environment, so a stale export cannot silently re-enter an arm.

### 0.4 — engagement verifier — **Accepted**

[`scripts/engagement_verify.py`](scripts/engagement_verify.py). Emits
`engagement: VERIFIED|UNVERIFIED (<reason>)` per arm; no number is quotable
without `VERIFIED`.

Three independent classes of evidence, each anchored to a line on the pinned
stack rather than to a guess about log wording:

1. **Resolved configuration** — `GET /server_info`, documented upstream
   (`http_server.py:811-820`) as "the resolution result: what the launcher was
   given, with every decision resolution made applied over it". Compared against
   what the arm requested. Unreadable ⇒ `UNVERIFIED`, never assumed-agreeing.
2. **Behavioural graph engagement** — the scheduler's per-prefill line ends with
   `cuda graph: True|False` (`metrics_reporter.py:655`, label from `:186-190`).
   A graph-on arm below 90% True is `UNVERIFIED`; a `disabled` arm above 0% is
   too, because the flag then did not take effect. This is the check that
   catches a config that reads right but did not run.
3. **Degradation signals** in the server log:
   - `PCG capture stream is not set` — `compilation/cuda_piecewise_backend.py:168`
     (change C: warn-once + eager fallback, no longer a crash);
   - `falling back to non-IPC transport` / `MmItemMemoryPool has no free chunk`
     — `multimodal/transport/cuda_ipc.py:167-176` (change B);
   - any deprecation warning naming a flag we set (change A/D detector).

Arms that leave a flag unset (`A0_default`) are only *recorded*, not asserted —
recording what the default resolves to is the arm's purpose.

---

## Phase 1 — cheap gates

### 1.1 GPU idle check — **Accepted**

GPU 7 at 212 MiB, no stale `sglang.launch_server` / `vllm.entrypoints`
processes. (The plan was drafted while all 8 GPUs were busy at 43–124 GB; that
schedule block has cleared.)

### 1.3-pre vLLM Qwen3-VL support — **Accepted**

Checked before spending any GPU time, because the whole cross-framework half of
#4 rests on it:

```
vLLM 0.21.0 ModelRegistry → ['Qwen3VLNemotronEmbedModel',
                             'Qwen3VLForConditionalGeneration',
                             'Qwen3VLMoeForConditionalGeneration']
```

The anchor's premise holds. The live image-anchor run is still 1.3 proper.

### 1.4-pre stack bring-up (A0_default, 20 prompts) — **Accepted**

First GPU contact for the v3 stack. Purpose: prove that latest-upstream SGLang
source runs at all on this box's torch 2.11 before spending anything on a
matrix.

| | |
|---|---|
| Server up | 48 s (04:09:40 → 04:10:28) |
| Completed | 20/20, 0 failures, no forbidden-token error |
| TTFT p50 | 141.2 ms |
| TPOT p50 | 5.65 ms |
| Vision tokens/req | 882 |
| Text tokens/req | 143 |
| Verdict | `engagement: VERIFIED` |

**The manifest §1 prediction is confirmed empirically.** With the prefill flag
left unset, the server resolved to:

```
Capture target prefill CUDA graph begin. backend=breakable, num_tokens=[4 … 8192]
/server_info → cuda_graph_config.prefill.backend = "breakable"
/server_info → mm_feature_transport = "cpu"
```

So on the merged-preview stack `A0_default` **is** the post-merge default, and
`--mm-feature-transport` unset **does** resolve to `cpu` — both assumptions the
matrix rests on, now measured rather than argued.

### 1.4-pre-fix verifier defect found by the bring-up — **solvable**

The bring-up scored `graph=91.3% of 23 prefill batches` against a 90% floor — a
healthy arm nearly failing. Investigating the denominator rather than raising
the floor found a real defect in my own step-0.4 verifier:

```
#new-token:    1  ×2  cuda graph: False   ← server's own readiness probes
#new-token:   78  ×1  cuda graph: True
#new-token: ~1015-1032 ×20 cuda graph: True   ← the 20 benchmark requests
```

The two `False` rows are the server's 1-token internal probes. They are not
benchmark work, they never run under a graph, and on a 20-prompt smoke they are
8.7% of the denominator — enough to fail a perfect arm. On a 400-prompt bracket
they would have been invisible, which is exactly the kind of scale-dependent
threshold that produces an inconsistent verdict between smoke and headline.

Two fixes, both making the check stricter rather than looser:

1. Denominator is now benchmark-sized batches only (`#new-token ≥ 8`; the
   smallest captured bucket is 4). Probes are counted and reported separately.
   Floor raised **90% → 99%**, since real batches should be ~100%.
2. Added an independent capture-time signal:
   `Capture target prefill CUDA graph begin. backend=<x>`. `/server_info`
   reports what the config *resolved to*; this reports what was *actually
   captured*. The verifier now requires them to agree, and treats a captured
   graph on a `disabled` arm as a failure in the other direction.

Re-verdict on the same run, unchanged data:

```
engagement: VERIFIED (backend=breakable, transport=cpu, captured=breakable,
                      graph=100.0% of 21 bench prefill batches, 2 probes excluded)
```

### 1.4 five-arm engagement smoke (pre-instrumentation, stack `48b0365bcc`) — **Accepted** as a gate, with one arm excluded

20 prompts, 1 rep per arm — sized to run the 0.4 verifier, **not** to measure.
No number below is quotable as a result; they are here because the spread is
what motivated the phase-2 design.

| arm | requested | resolved / captured | TTFT p50 | engagement |
|---|---|---|---|---|
| `A0_default` | — / — | breakable / breakable, cpu | 139.6 ms | **VERIFIED** |
| `A1_disabled` | — / `disabled` | disabled / none captured, cpu | 142.4 ms | **VERIFIED** |
| `A2_tcp` | — / `tc_piecewise` | tc_piecewise / tc_piecewise, cpu | 143.7 ms | **UNVERIFIED** |
| `A3_bcg` | — / `breakable` | breakable / breakable, cpu | 138.5 ms | **VERIFIED** |
| `A4_ipc` | `cuda_ipc` / — | breakable / breakable, **cuda_ipc** | **102.3 ms** | **VERIFIED** |

All arms: 20/20 completed, 0 failures, no forbidden-token error.

Plan gate ("1.2 + 1.3 pass, and ≥ A0/A1/A2 or A3/A4 verify"): 4 of 5 verify.

Two things worth recording before phase 2 is designed around them:

- **The four graph arms sit inside 138.5–143.7 ms — a 3.7% spread at n=20, one
  rep.** That is noise, not a result, but it says the prefill-graph lever may
  turn out immaterial for this workload, and phase 2 has to be able to report
  "no material difference" credibly rather than hunt for a trend.
- **`A4_ipc` is 102.3 ms, 26.7% below `A0_default`**, with `transport=cuda_ipc`
  confirmed resolved and no pool-fallback signal. The *transport* lever, not the
  graph lever, is where the image-path headroom appears to be. This is the
  hypothesis phase 2 exists to test properly.

### 1.4a `A2_tcp` — the change-C silent degradation, caught and then made measurable — **solvable**

The verifier failed `A2_tcp` on `PCG capture stream is not set`. This is exactly
the hazard plan §11.1 change C describes, and it is worth being precise about why
it is worse than the crash it replaced:

- **The batch-level indicator cannot see it.** All 21 benchmark prefill batches
  reported `cuda graph: True`, and `/server_info` reported
  `prefill.backend = tc_piecewise`, and the capture line reported
  `backend=tc_piecewise`. Every configuration- and behaviour-level check passed.
  Only the log-signal check caught the degradation, because the scheduler's flag
  is per *batch* while the piecewise fallback is per *subgraph*.
- **The warning fired mid-benchmark**, at 04:16:20, between decode batches —
  not during warmup.
- **The degradation is unbounded from the log.** `print_warning_once` is
  `@functools.lru_cache(None)` on the message string
  (`utils/common.py:2796-2799`), so the count is capped at 1 by construction —
  "1×" means "at least once", never "once".
- **And it is sticky.** The fallback branch `return`s *without capturing*
  (`cuda_piecewise_backend.py:166-173`), so a subgraph that once missed the
  capture stream runs eager for the remainder of the process. The comment's
  reassurance that "subsequent matching shapes still use their captured graphs"
  is true only for shapes that were captured at startup.

Excluding the arm would have satisfied §11.5 criterion 4 but left #4's **Q2**
("does #2's PCG win transfer to images?") permanently unanswerable. Instead the
pinned stack now carries measurement-only instrumentation
(`471e549959`, manifest §7): counters for graph-eligible calls, fallback
occurrences and distinct fallback shapes, emitted as a parseable `PCG_STATS`
line. Cost is one integer increment on the eligible path; nothing is logged
unless a fallback happens, so the patch is inert for every other arm.

The verifier reads those counters and now reports the magnitude instead of an
unbounded suspicion — verified against a synthetic log:

```
engagement: UNVERIFIED (PCG eager fallback on 2.55% of graph-eligible calls
                        (37/1450, 3 distinct shapes) — arm ran partially eager)
```

The verdict stays `UNVERIFIED` — the plan's rule that a partially-eager arm is
not a PCG number is deliberate and I am not softening it — but the report can now
state *how* degraded rather than only *that* it was.

### 1.4b re-run on the pinned stack `471e549959` — **Accepted**, and it produced the strongest result of phase 1

The instrumentation commit landed while `A4_ipc` was starting, so the first smoke
straddles two SHAs (`A0`–`A3` on `48b0365bcc`, `A4` on `471e549959`). The patch is
inert outside the piecewise fallback path, but "every arm shares one stack" is not
a rule to argue around, so the whole smoke was re-run on `471e549959`. The first
run is kept as the pre-instrumentation record.

#### The `tc_piecewise` degradation, now measured

`A2_tcp` on the pinned stack:

```
engagement: UNVERIFIED (PCG eager fallback on 8.53% of graph-eligible calls
                        (600/7038, 2 distinct shapes) — arm ran partially eager)
TTFT p50 = 144.8 ms
```

The `PCG_STATS` trace is more informative than the headline percentage:

| eligible calls | eager fallbacks | distinct shapes |
|---|---|---|
| 1000 | 0 | 0 |
| 2000 | 0 | 0 |
| **6402** | **1** | 1 |
| 6538 | 100 | 1 |
| 6638 | 200 | 2 |
| 7000 | 561 | 2 |
| 7038 | 600 | 2 |

Three facts follow, none of which were available from the log before:

1. **The degradation has a late onset.** Nothing at all for the first 6401
   graph-eligible calls, then the first fallback at 04:24:38 — deep inside the
   benchmark, long past warmup.
2. **After onset it is essentially total for the affected shapes.** 600
   fallbacks in the 637 eligible calls following onset — **94.2%**. This is the
   sticky mechanism confirmed by measurement: the branch returns *without*
   capturing, so a shape that once missed the capture stream runs eager for the
   remainder of the process.
3. **Onset is at a fixed call index, and the observed share is an artefact of
   run length.** *(Corrected in 2.3 — the smoke's 8.53% initially looked like
   run-to-run nondeterminism. The headline run put onset at exactly the same
   call index, 6402, so it is deterministic; the smoke simply ended at 7038
   calls, shortly after onset.)*

And the reason this matters for the whole experiment: **every other check
passed.** `/server_info` reported `prefill.backend = tc_piecewise`; the capture
line reported `backend=tc_piecewise`; all 21 benchmark prefill batches reported
`cuda graph: True`; TTFT p50 came back at a perfectly ordinary 144.8 ms. Without
the log-signal check the arm looks healthy, and without the counters "1×
warning" is indistinguishable from "600 fallbacks on 94% of post-onset calls".
Anyone reading only the batch-level indicator would have published 144.8 ms as
"the tc_piecewise number".

This is a genuine upstream finding — the capture-stream eager fallback is not a
rare transient on Qwen3-VL image serving — but per §11.5 criterion 5 it belongs
in a **new issue**, not in #4's scope. Recorded as a follow-up.

#### Run-to-run noise at smoke size

Same configuration, two runs:

Same configuration, two independent runs (smoke-1 on `48b0365bcc`, pinned on
`471e549959`; the instrumentation is inert for every arm but `A2_tcp`):

| arm | transport | smoke-1 | pinned | Δ | engagement |
|---|---|---|---|---|---|
| `A0_default` | cpu | 139.6 ms | 143.2 ms | +2.6% | VERIFIED |
| `A1_disabled` | cpu | 142.4 ms | 133.8 ms | **−6.1%** | VERIFIED |
| `A2_tcp` | cpu | 143.7 ms | 144.8 ms | +0.8% | UNVERIFIED |
| `A3_bcg` | cpu | 138.5 ms | 149.1 ms | **+7.6%** | VERIFIED |
| `A4_ipc` | **cuda_ipc** | 102.3 ms | 102.0 ms | **−0.3%** | VERIFIED |

**Repeat noise on the CPU-transport arms reaches 7.6% — twice the entire 3.7%
spread between the four graph arms within smoke-1.** The smoke cannot
discriminate between prefill backends and was never meant to. This is the
quantitative justification for the phase-2 sizing (400 prompts × 5 reps) and for
the reporting rule that a delta under 5% is "no material difference" rather than
a trend.

The line that does *not* fit that pattern is the interesting one. **`A4_ipc`
reproduces to within 0.3% across two independent runs on two different SHAs**,
while every CPU-transport arm scatters by 2.6–7.6%. So the `cuda_ipc` arm is not
only 28.8% below `A0_default` on the pinned run — it is also far more stable.

That suggests the run-to-run variance in this workload lives in the **host-side
multimodal feature path**, not in the prefill graph, which would also explain why
the graph lever looks immaterial here: the graph is optimising a part of the
timeline that the CPU feature copy dominates. Stated as a hypothesis, not a
result — two points per arm cannot establish it. Phase 2's five reps per arm can,
and the phase-3 report reports CV per arm precisely so this is checkable rather
than asserted.


The instrumentation commit landed while `A4_ipc` was starting, so the smoke above
straddles two SHAs (`A0`–`A3` on `48b0365bcc`, `A4` on `471e549959`). The patch is
inert outside the piecewise fallback path, but "every arm shares one stack" is
not a rule to argue around. The whole smoke is re-run on `471e549959` as
`phase1_engagement_smoke_pinned`; the run above is kept as the pre-instrumentation
record.


### 1.4c orchestration defect in my own tooling — **solvable**

The phase-1.2 parity run did not start when the smoke finished. Cause was mine,
not the experiment's: the background waiters I used to sequence the runs polled
with `pgrep -f "run_imgA_v3"`, and `pgrep -f` matches against **full command
lines** — including the waiter shells' own, which contain that pattern. Seven
waiters were therefore matching each other and looping forever, and the chain
script waiting on the same predicate never saw the runner exit.

No measurement was affected — GPU 7 sat idle at 212 MiB throughout, and every
result above was already written — but it cost wall-clock time and would have
silently stalled the phase-2 launch. Stuck shells killed; the sequencing now keys
off artifacts (a results file appearing, a marker line in a log) rather than off
process-name polling, which cannot self-match.

### Follow-up drafted, not filed

[`pcg_eager_fallback_finding.md`](pcg_eager_fallback_finding.md) — the
capture-stream eager fallback written up for a **separate** upstream issue, per
§11.5 criterion 5. It is a draft in this repo only; **nothing has been filed
upstream**, since opening an issue is an outward-facing action and is the owner's
call. It carries the mechanism, the measured magnitude, the reason the batch-level
indicator cannot see it, a reproduction, and three suggested directions offered as
options rather than a preferred design.

---

### 1.2 / 1.3 correctness parity — **Accepted**, with one caveat that must travel with Q1

Four fixed fixtures (two text, two image), greedy (`temperature=0`, `top_p=1`,
`seed=0`, 48 tokens). The image fixture is generated in-process — fixed RGB
vertical stripes — so every arm sees byte-identical input.
Reference arm: `A1_disabled` (no prefill CUDA graph, eager).

| comparison | verdict |
|---|---|
| `A1_disabled` vs `A0_default` (default → breakable) | **IDENTICAL**, 4/4 |
| `A1_disabled` vs `A3_bcg` (explicit breakable) | **IDENTICAL**, 4/4 |
| `A1_disabled` vs `V0_vllm` | 2/4 identical — both text fixtures exact, both image fixtures diverge in wording |

#### Cross-backend (the check engagement verification cannot make)

Both graph arms reproduce the eager reference **token for token**, on image
fixtures included. On Qwen3-VL-8B — `deepstack_visual_indexes = [8, 16, 24]`,
replay width 12288 — that is the DeepStack replay path PR #33726 fixes,
exercised and numerically correct.

This is the failure mode engagement verification is structurally blind to: an arm
can resolve to exactly the requested backend, capture the graph, run 100% of its
prefill batches under it, and still compute the wrong thing. `A0`/`A3` pass both
checks independently.

Worth noting for the PR: M10 established this on Qwen3-VL-**4B**
(`[5, 11, 17]`). This is the same property on **8B**, the model whose
`FAIL_BCG_DEEPSTACK` reproduction motivated the fix.

Sanity that the fixture is actually perceived rather than confabulated: every arm
reads the stripe order back correctly as
`Red, Green, Blue, Yellow, Red, Green, Blue, Yellow`.

#### Cross-framework

The divergence is confined to the two image fixtures and is **phrasing only,
with identical content**:

```
image_colors  SGLang: "The colors in the image, from left to right, are:"  + Red, Green, Blue, Yellow, Red, Green, Blue, Yellow
              vLLM  : "From left to right, the colors in the image are:"   + Red, Green, Blue, Yellow, Red, Green, Blue, Yellow
```

Both text fixtures are exact across all four arms, vLLM included. **Text exact +
image divergent localises the difference to the vision path** — image
preprocessing kernels, vision-encoder numerics, feature projection — not to the
language model or to sampling. Small numeric differences there shift the greedy
argmax at some token; neither framework is wrong.

The anchor itself is healthy: no crash signature in the vLLM server log, so it
passes the anchor criteria.

**Gate: PASS, with a caveat that must be carried into the report.** `V0_vllm`
remains a valid latency anchor — same model, same revision, same fixture, same
greedy settings, same token budget — but SGLang and vLLM do **not** produce
token-identical output on images. Q1 is therefore a framework-to-framework
comparison, **not** a strict like-for-like equivalence, and must be reported that
way. #4 does not degrade to an SGLang-internal study; the anchor works.

### Phase 1 gate — **PASSED**

- 1.1 GPU idle ✅
- 1.2 correctness parity ✅ (cross-backend exact; cross-framework anchor valid with the caveat above)
- 1.3 vLLM image anchor ✅ (serves the image workload cleanly)
- 1.4 engagement ✅ 4 of 5 arms VERIFIED; `A2_tcp` excluded with its exact, now-quantified upstream failure

Plan's gate wording — "1.2 + 1.3 pass, and ≥ A0/A1/A2 or A3/A4 verify" — is met
by `A0`, `A1`, `A3`, `A4`. Proceeding to phase 2.

---

## Phase 2 — IMG-A headline

### 2.0 `A5` would have measured a duplicate — **solvable**

Caught 20 minutes into the first bracket launch, and it is the same class of
mistake as everything else in this experiment: a rule that was correct when
written and stopped being correct when the stack moved.

Plan §11.2 defines `A5_ipc_best` as "`cuda_ipc` + winner of {A2, A3}". `A2_tcp`
does not verify, so the winner is `A3_bcg` → `breakable`. But **on this stack the
default prefill backend already *is* `breakable`**, and `A4_ipc` is
`cuda_ipc` + default. `A5` would therefore have re-measured `A4`'s exact resolved
configuration — 40 minutes spent on a duplicate, and the composition question
left unanswered.

The plan's *stated purpose* for the cell is "do the two levers compose or
interfere?". The cell that answers it is the missing corner of the factorial:

| prefill graph | cpu transport | cuda_ipc transport |
|---|---|---|
| `disabled` | `A1_disabled` | **`A5_ipc_nograph`** ← was missing |
| `breakable` | `A0_default` | `A4_ipc` |

`A5` is redefined as `cuda_ipc` + `disabled`, completing a full 2×2 over the two
levers inside one bracket, under one drift gate. That is strictly stronger than
the plan's pairwise rule: it yields the **interaction** (does the graph pay the
same amount under IPC as under CPU transport?), not just a single composed point.
The adaptive-selection code is deleted — the cell is now fixed, so there is no
mid-bracket choice to record.

The bracket was restarted rather than patched in flight (the running process had
already imported the old arm table). Cost: ~20 min of `A0_default`. The aborted
run is kept as `results/phase2_imgA_headline_aborted_20260904T043425Z/`.

Its one useful datum, from the pre-abort `A0_default`: **400/400 completed, 0
failures, TTFT p50 138.3 ms over 441.8 s**, warmup 30 in 45.5 s. That sets the
real cost per arm at ~40 min (5 reps + warmup + start/stop) and the full 8-arm
bracket at **~5.3 GPU-hours**, not the ~1 GPU-day the plan estimated.

The report generator computes the 2×2 and its interaction term when all four
cells verify, and says so explicitly when they do not.

### 2.1 first headline attempt was measuring the prefix cache — **solvable**

Caught on `A0_default`'s completion, from two numbers that did not look right:

```
A0_default: ttft_p50_median=125.485ms cv=8.2%
engagement: VERIFIED (... graph=100.0% of 401 bench prefill batches, 1637 probes excluded)
```

**CV 8.2%** is far above the ≤5% band, and **401 benchmark batches out of 2030
requests** is impossible if every request does a real prefill. The per-rep series
settles it — this is not scatter, it is a monotonic slide:

| rep | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| TTFT p50 | 147.5 ms | 126.8 ms | 125.5 ms | 118.8 ms | 119.6 ms |

**Cause: the runner never passed `--disable-radix-cache`.** With a fixed seed
every rep replays the identical prompt set, so reps 2–5 were served from the
prefix cache. Measured after the fact with the fixed verifier: **80.3% of prefill
tokens came from cache** (1 676 835 cached vs 410 100 new). Reps 2–5 were
measuring cache lookup, and the prefill-graph lever — the entire object of the
experiment — barely executed.

The v2 protocol and the M10 smoke both set the flag. I dropped it while porting
the runner.

Three fixes, and the second one matters more than the first:

1. **`--disable-radix-cache`** on every SGLang arm, and
   **`--no-enable-prefix-caching`** on the vLLM anchor (vLLM V1 enables prefix
   caching by default, so the anchor needs the same condition or the comparison
   is not like-for-like).
2. **The verifier was complicit and is now not.** It filed all 1637 cache-served
   batches under "probes" — the exclusion I added in 1.4-pre-fix for the server's
   1-token readiness probes — and reported `VERIFIED, 100% of 401 batches`. That
   was *technically true and substantively misleading*: 80% of the workload was
   not doing prefill and the verdict said nothing. Two checks added:
   - probe batches greatly outnumbering benchmark batches;
   - **cached tokens as a share of prefill tokens above 20%** — counted in
     *tokens*, not batches, because a few tokens of shared chat-template prefix
     appear on nearly every batch and a batch-count rule would fire on a healthy
     run.
3. Retro-validated both ways. The invalid run now fails loudly
   (`80.3% of prefill tokens came from cache`); a healthy phase-1 log reads
   **5.1%** cached — the legitimate shared prefix — and passes.

**Phase 1's conclusions are unaffected**: at 1 rep × 20 unique prompts there was
no repeated prompt set to cache, and its measured cached share is 5.1%.

The invalid bracket is kept as
`results/phase2_imgA_headline_invalid_prefixcache_20260904T051344Z/`. An orphaned
`A1_disabled` server outlived the killed runner and had to be reaped by PID before
the GPU freed — worth remembering for any future abort.

### 2.2 bracket — *running* (restarted 05:57 UTC, prefix caching off)

Order: `A0_default → A1_disabled → A2_tcp → A3_bcg → A4_ipc → A5_ipc_nograph →
V0_vllm → A0_repeat`, 400 prompts, 30 warmup, 5 reps, c=1.

`A0_default` confirms the fix, side by side with the invalid run:

| | invalid (cache on) | valid (cache off) |
|---|---|---|
| benchmark prefill batches | 401 of 2030 requests | **2037** |
| probe batches excluded | 1637 | **1** |
| cached token share | **80.3%** | **0.0%** |
| TTFT p50 median | 125.5 ms | **146.9 ms** |
| CV over 5 reps | **8.2%** | **1.7%** |
| per-rep series | 147.5 → 126.8 → 125.5 → 118.8 → 119.6 (monotonic) | 148.4, 143.8, 146.9, 144.6, 150.5 (scattered) |

The cache was inflating apparent performance by ~15% *and* destroying rep
independence — the reps were a warming curve, not five samples. 2 086 935 new
tokens over 2037 batches, 100% of them under the prefill CUDA graph, 0 cached.

Measured cost: **40.5 min/arm**, so the 8-arm bracket lands around 11:20 UTC.

### 2.3 `A2_tcp` at full length — the design premise, vindicated

```
engagement: UNVERIFIED (PCG eager fallback on 92.11% of graph-eligible calls
                        (75200/81638, 2 distinct shapes) — arm ran partially eager)
TTFT p50 = 147.02 ms, CV 0.7%
```

The trace corrects what phase 1 could not see:

```
eligible=1000 … 6000   eager_fallback=0
eligible=6402          eager_fallback=1     ← onset
eligible=81638         eager_fallback=75200, 2 shapes
```

**Onset lands on call 6402 — the same index as the pinned smoke.** It is not
nondeterministic, as the smoke's single data point suggested; it is
*deterministic in call count*. The smoke's 8.53% was purely an artefact of that
run ending at 7038 calls, 636 calls after onset. Past onset the degradation is
**99.95%** (75 200 of 75 237 calls), which is the sticky mechanism running to its
conclusion: the branch returns without capturing, so the two affected shapes are
eager forever.

Two things follow that are worth stating plainly.

**A2_tcp's 147.02 ms is an eager number, not a tc_piecewise number.** It sits
between `A1_disabled` (142.3 ms) and `A0_default` (146.9 ms), exactly where a
mostly-eager arm should.

**And it has the lowest CV in the bracket — 0.7%.** The most degraded arm looks
like the cleanest measurement. Low variance is not evidence of health; it is
evidence of consistency, and an arm that is consistently eager is consistently
eager. Anyone ranking these arms by CV would have picked the broken one as the
most trustworthy.

Without the counters this arm reports one suppressed warning and a tight, tidy
147.02 ms. That is the whole case for the v3 design in one number.

---

## Phase 2b — IMG-R ratio sweep (plan.md §11.8)

Started 11:08 UTC. Transport pinned to `cuda_ipc` (issue #4's standard
condition), two prefill-graph arms per workload run back to back, 300 prompts /
20 warmup / 3 reps, c=1.

### Out-of-sample prediction, recorded before `R2_360p__breakable` ran

After the first two workloads the mechanism from §11.8 (`saving ≈ C − k·N`) has
two points to fit:

```
N=128  saving 12.32 ms      N=208  saving 9.00 ms
⇒ k = 3.32 / 80 = 0.0415 ms/token,  C = 17.63 ms,  sign change at N ≈ 425
```

The pre-registered prediction in §11.8 was "sign change somewhere below N ≈ 400",
written before any of this ran. The two-point fit puts it at **425**.

`R2_360p` measured `N = 364` with `disabled` at 64.005 ms, so the fit predicts a
saving of **2.5 ms → breakable ≈ 61.5 ms, effect ≈ −3.9%**. Recorded here before
the arm finished so the test is genuinely out of sample. A materially different
result refutes `C − k·N` and the model must be replaced, not patched.

### The `C − k·N` model is refuted

`R2_360p__breakable` came in at **55.033 ms**, a saving of **8.97 ms**. The fit
predicted 61.5 ms and a saving of 2.5 ms. That is not a near miss; the model is
wrong and is discarded rather than patched.

Savings at fixed `cuda_ipc` transport:

| workload | N | disabled | breakable | saving | effect |
|---|---|---|---|---|---|
| `R0_text` | 128 | 27.48 ms | 15.16 ms | **12.32 ms** | −44.8% |
| `R1_tiny` | 208 | 55.23 ms | 46.23 ms | **9.00 ms** | −16.3% |
| `R2_360p` | 364 | 64.01 ms | 55.03 ms | **8.97 ms** | −14.0% |
| IMG-A (from the phase-2 IPC row) | ~1010 | 102.17 ms | 105.91 ms | **−3.74 ms** | +3.66% |

**N nearly doubled from 208 to 364 and the saving did not move** (9.00 → 8.97 ms).
A cost proportional to N would have removed ~6.5 ms over that span. Whatever
erodes the graph's benefit, it is not a per-token copy cost in this range.

The shape that the data actually shows is a **step, then a plateau, then a
collapse**:

- text-only → any image costs ~3.3 ms of the saving (12.32 → 9.00), a step
  associated with *the presence of an image*, not its size;
- across 208 → 364 tokens the saving is **flat**;
- by ~1010 tokens it has gone negative.

The percentage effect falls from −44.8% to −14.0% mostly because the
**denominator grows** — the vision encoder and preprocessing add a large fixed
cost that the prefill graph cannot touch (measured directly: R0 → R1 adds only
80 tokens but 28 ms). The numerator barely changes.

**A comparison error of mine, corrected here.** The earlier running table put
IMG-A's `+3.24%` (measured at **cpu** transport) on the same curve as this sweep
(**cuda_ipc** throughout). They are not comparable. The IPC-row figure is
`A5_ipc_nograph` 102.17 vs `A4_ipc` 105.91 → **−3.74 ms**, which is what appears
above.

**Design gap this exposes.** The sign change lies between N = 364 and N ≈ 1010,
and the sweep has no sampling point in that interval: `R3_720p` is at 1024 and
`R4`/`R5` are both above it. An intermediate point is needed —
`512x512` (~256 visual tokens) is too low; `896x896` (~784 + 128 ≈ 912) and
`640x640` (~400 + 128 ≈ 528) bracket it usefully. To be added once the sweep
finishes rather than by interrupting it.

**None of this changes the answer to the question that prompted the sweep.** Two
genuine image+text points show a double-digit graph win — `R1_tiny` −16.3% at
N=208 and `R2_360p` −14.0% at N=364, every arm independently verified. What
changed is the explanation for where the win disappears, and the old explanation
is withdrawn rather than reshaped to fit.

### `R4` refutes both candidate explanations; a third hypothesis, recorded before `R5`

`R4_720p_longtext` was built to separate two stories. It separated them by
falsifying both.

| | | |
|---|---|---|
| `R4_720p_longtext__disabled` | 135.709 ms | N = 1969 (882 vision + 1087 text) |
| `R4_720p_longtext__breakable` | 135.994 ms | |
| graph effect | **+0.21%** | saving −0.29 ms |

- A **ratio**-driven story predicted improvement: text share rises from 13% to
  55%. It did not improve.
- An **N**-driven story predicted deterioration: N grows 1024 → 1969. It did not
  deteriorate.

The effect is simply flat near zero: +0.80% at N=1024, +0.21% at N=1969.

Full series at fixed `cuda_ipc` transport:

| N | vision | text | saving |
|---|---|---|---|
| 128 | 0 | 128 | **+12.32 ms** |
| 208 | 80 | 128 | +9.00 ms |
| 364 | 236 | 128 | +8.97 ms |
| 1024 | 896 | 128 | −0.84 ms |
| 1969 | 882 | 1087 | −0.29 ms |

The saving plateaus around +9 ms, transitions, and **settles at ~0 rather than
going increasingly negative**. At large N the prefill graph is *neutral*, not
harmful — which also suggests IMG-A's +3.66% overstated the cost.

**Third hypothesis — explicitly a hypothesis, after one refutation.** The graph
does not save a constant and pay a per-token cost. It saves **the portion of
kernel-launch overhead that is not already hidden behind GPU execution**. When
per-kernel GPU work is small, launch overhead is exposed and lands in TTFT; as
sequence length grows, each kernel does more work and the launch cost overlaps
away, so the recoverable amount falls toward zero on its own.

This accounts for three things the discarded model could not:

- the saving **asymptotes to 0** instead of turning sharply negative;
- **the presence of an image is a step** (12.32 → 9.00 ms) — the vision encoder
  adds GPU work that provides overlap;
- **`R4` changes nothing** — at N=1024 the recoverable overhead is already spent,
  so another 1087 text tokens have nothing left to take.

**Falsifiable prediction, written before `R5_1080p__breakable` ran:** with
`R5_1080p__disabled` at 203.92 ms (N=2184), the model predicts **saving ≈ 0 ± 2 ms,
breakable ≈ 202–206 ms, effect within ±1% — neutral, not negative.** A clearly
negative result (say ≥ 210 ms) refutes this model too, and it will be withdrawn
rather than adjusted.

The gap-filling workloads (N ≈ 528, 704, 912) test the other half: the model
predicts a smooth monotone decline from ~9 ms toward ~0 across them, with no
sharp cliff.

### `R5` outcome and the gap-fill prediction

`R5_1080p__breakable` came in at **206.008 ms** against `disabled` 203.92 ms —
**+1.02%**, saving −2.09 ms.

By the pre-registered refutation criterion (≥210 ms would refute), the model
stands: the result is neutral, not clearly negative. But the tighter bands I also
quoted were each missed by a hair — predicted 202–206 ms (got 206.008), saving
0 ± 2 ms (got −2.09), effect within ±1% (got +1.02%). Three near-misses in the
same direction, and the arm's CV is 1.6%, so **the bands I quoted were tighter
than the data can support**. That is a fault in how I stated the prediction, not
a success of the model.

**And the sweep's own drift makes the large-N cells unresolvable anyway.** The
reference cell repeat moved 104.51 → 100.74 ms, **3.60%**. Every effect at
N ≥ 1024 (+0.21% to +1.02%) is smaller than that. The only defensible statement
there is "not resolvable at this bracket's precision" — not "neutral", and
certainly not "a small cost". My earlier phrasing of "hovering around −1 ms"
overstated what the bracket can see, and the report generator now enforces this
as a rule rather than leaving it to prose.

The N = 128–364 effects (14–45%) are an order of magnitude above both drift and
CV, so those stand.

**Gap-fill prediction, recorded before `R6_640__breakable` finished.** With
`R6_640__disabled` at 69.992 ms (N=544), linear interpolation between (364,
+8.97 ms) and (1024, −0.84 ms) gives a saving of **6.3 ms → breakable ≈ 63.7 ms,
effect ≈ −9%**. Stated as a band this time: **saving 4–8 ms, effect −6% to −11%**.

The discriminating test is the *shape* across all three gap-fill points, not this
one number: the hypothesis predicts N = 544, 704, 912 fall **monotonically**
between ~9 ms and ~0. **A cliff refutes it** — if R6 is still ~9 ms and R7 drops
straight to ~0, the cause is a threshold (graph bucket sizing, a chunked-prefill
switch) rather than launch overhead being progressively overlapped away, and the
model gets withdrawn like the last one.

---

## Phase 3 — issue #4 follow-on (plan.md §12)

Re-organised around three questions rather than bracket IDs. **Q0 (transport)**
is answered and closed at −28.2%. **Q1** and **Q2** are what remain.

### Q1 — is a text token equivalent to a visual token? — *running*

Started 22:10 UTC 2026-09-05 on **GPU 5** (GPU 7 held 122 GB by another tenant).

The v3 sweep moved visual tokens across seven points but moved text tokens
exactly once, at 720p, where both cells sat inside the ±3.60% resolution floor.
So "the controlling variable is total prefill tokens, not the image/text ratio"
was never tested where token *type* could show a difference — and the
load-bearing −44.83% is a measurement at **128 text tokens**, while real prompts
carry system context and retrieved passages.

Three text-only cells, chosen so their token counts match workloads v3 already
measured with images:

| cell | text tok | matches | that cell's composition | its measured effect |
|---|---|---|---|---|
| `text-208` | 208 | `R1_tiny` | 66 visual + 142 text | −16.30% |
| `text-544` | 544 | `R6_640` | 402 visual + 142 text | −4.54% |
| `text-1024` | 1024 | `R3_720p` | 882 visual + 142 text | +0.80% |

`text-1024` doubles as the realistic-prompt-length answer.

**Method change — paired A/B/A/B blocking.** v3's floor was set by drift across a
bracket spanning hours, while per-cell CV was only 0.2–1.8%: drift, not variance,
was binding. The arms now alternate in short blocks (one rep each) instead of
running all of one arm then all of the other, so a comparison spans minutes.
Gate: the three `disabled` blocks of a workload must agree within **2%** or the
workload is discarded rather than averaged. `--chunked-prefill-size` is pinned at
8192 rather than left to resolution.

18 blocks, ~2 GPU-hours.

**Discriminating criterion, recorded before the first `breakable` block finished.**
`text-208__disabled` came in at **31.62 ms**, against `R1_tiny__disabled`
(66 visual + 142 text, same N=208) at **55.23 ms** — a 23.6 ms gap at *identical*
token count, which is the image's fixed preprocessing and vision-encoder cost
confirmed on matched N rather than inferred.

That gap makes the matched-N test sharper, because "token type is irrelevant" has
two incompatible readings:

| if the quantity fixed by N is… | predicted `text-208` graph effect |
|---|---|
| the **absolute saving** (~9 ms, as at `R1_tiny`) | −9 / 31.6 ≈ **−28%** |
| the **percentage effect** | **−16.3%**, matching `R1_tiny` |

They cannot both hold: the denominators differ by 43%. Which one holds matters
for the deployment claim — if the saving is what is conserved, then the graph's
value on text requests is *systematically understated* by percentages measured on
image workloads, and a text-heavy deployment gains more than those percentages
suggest.

### Q2 — the request-stream mix — *running*

Two bench clients against one server at fixed total arrival rate, per-class TTFT
reported separately, staged: `f ∈ {0, 0.2, 1.0}` first, and `{0.05, 0.5}` only if
stage 1 shows text-class degradation under `breakable` that `disabled` does not
show.

### Q1 first pair (N=208) — token type is **not** irrelevant

Block 1 of 3, both cells `VERIFIED` (`captured=None` / `captured=breakable`,
graph 0.0% / 100.0% of 323 benchmark prefill batches).

| N=208, identical token count | disabled | breakable | saving | effect |
|---|---|---|---|---|
| **text-only** (208 text) | 31.62 ms | **18.74 ms** | **+12.88 ms** | **−40.74%** |
| **with an image** (66 visual + 142 text, v3) | 55.23 ms | 46.23 ms | +9.00 ms | −16.30% |

**Both pre-registered predictions fail.** Conserved-effect predicted −16.3% and
conserved-saving predicted −28%; the measurement is −40.74% with a saving of
12.88 ms. The outcome is a third one: at the same token count, **a visual token
erodes the graph's benefit more than a text token does**, in the numerator as
well as the denominator.

**Exact decomposition of the 24.45 pp gap:**

| component | size | share |
|---|---|---|
| **denominator** — the image adds 23.61 ms of fixed cost (preprocessing + vision encoder) that the graph cannot touch | 17.41 pp | **71%** |
| **numerator** — recoverable overhead genuinely shrinks, 12.88 → 9.00 ms | 7.03 pp | **29%** |

(The counterfactual "same saving, image denominator" is −23.33%; the distance
from −40.74% to there is the denominator's work, and from there to −16.30% is
the numerator's.)

The split matters for the deployment claim, because the two components mean
different things. **Roughly seven tenths of "the graph looks useless on images"
is the percentage being diluted by a cost that has nothing to do with CUDA
graphs**, not the graph failing. Its absolute contribution survives; it is
simply buried under vision-encoder time.

Consistent with both prior observations and with the surviving hypothesis: text-
only saving is flat across 128 → 208 tokens (12.32 → 12.88 ms), and the drop
that appears with an image is what the encoder's extra GPU work overlaps away.

**This contradicts a sentence in the published report.**
[`issue4_v3_report.pdf`](issue4_v3_report.pdf) states that "the controlling
variable is the number of prefill tokens, not the image-to-text ratio". At
matched N the two compositions give −40.74% and −16.30%, so as written that is
wrong: token count alone does not fix the effect. The claim narrows to the visual
axis — *along the visual axis, benefit falls as visual tokens grow* — and the
report and PDF are corrected once the remaining blocks land and the within-pair
drift gate is applied.

### Q1 complete — two errors of mine, one real result

The 18 blocks all ran clean: every cell `VERIFIED`, 300/300 completed, no
failures. The runner then declared all three workloads FAILED. Both of those
verdicts were wrong, for two different reasons, and the diagnosis is the useful
part.

#### Error 1 — I gated on the quantity the design exists to cancel

| workload | `disabled` blocks | paired effects | level spread | paired spread |
|---|---|---|---|---|
| `text-208` | 31.62 → 27.30 → 26.32 | −40.74, −39.77, −37.76 | **19.4%** | **2.98 pp** |
| `text-544` | 27.16 → 27.60 → 28.73 | −15.94, −15.54, −18.54 | 5.7% | 3.00 pp |
| `text-1024` | 36.05 → 35.06 → 34.88 | +7.87, +10.09, +13.89 | 3.3% | 6.02 pp |

`text-208`'s levels fall 19.4% across the bracket — a cold-start ramp, since the
GPU had been idle and the per-block warmup is only 20 prompts. But the **paired**
effects agree to 3 pp. That is A/B/A/B blocking working exactly as designed: the
common-mode drift cancels in the ratio.

My gate then measured the level spread and discarded everything. **I built the
right instrument and read the wrong dial.** Fixed: the gate now reads paired
effects, and reports level drift separately as the quantity pairing removed.

#### Error 2 — and an alarm I raised and must withdraw

`text-1024` returned **+10.09%** — the graph appearing to *cost* 10% on a long
text prompt. The mechanism is not long prompts:

| cell | client asks | **server sees** | bucket | padding |
|---|---|---|---|---|
| `text-208` | 208 | 216 | 224 | 3.7% |
| `text-544` | 544 | 552 | 576 | 4.3% |
| `text-1024` | 1024 | **1032** | **1280** | **24.0%** |

`--random-input-len` excludes the chat template, which adds ~8 tokens. The
capture ladder is dense below 1024 and steps by 256 above it, so 1032 tokens
overshoot the 1024 bucket by eight and pad to 1280. The graph arm does **24% more
prefill compute than it needs**; the +10.09% is that, not a property of long
prompts.

**The alarm I raised — that this might also confound IMG-A — was wrong, and I
withdraw it.** Checked across every v3 workload:

```
IMG-A (A0/A4/A5)   mode 1024 -> bucket 1024   padding  0.0%
R3_720p            mode 1024 -> bucket 1024   padding  0.0%
R0 R1 R2 R6 R7 R8 R4 R5      padding 0.0-6.7%
```

The image workloads land on 1024 exactly (882 visual + 142 text), so **v3 is
unaffected**. Only the text-only cell I added last night mis-lands.

Consequence: the N=208 and N=544 pairs are valid (both members land within
~6% padding, and at 544 both are in the same 576 bucket). **The N=1024 pair is
not a matched comparison at all** — it compares 24% padding against 0% — and is
discarded. `text-1016` is running now: 1016 requested makes the server see 1024,
landing on the same bucket as its partner.

#### The result that survives

| N | text-only | with an image (v3) | difference |
|---|---|---|---|
| 208 | **−39.77%** (+10.86 ms) | −16.30% (+9.00 ms) | 23.5 pp |
| 544 | **−15.94%** (+4.33 ms) | −4.54% (+3.18 ms) | 11.4 pp |

Token type is **not** irrelevant: at matched token count the text-only workload
gains far more. But the refinement matters — the *saving in milliseconds* is
close between compositions (10.86 vs 9.00; 4.33 vs 3.18), while the *percentage*
differs two- to three-fold. So the graph recovers a similar absolute amount
either way, and the image simply buries it under vision-encoder time that the
graph cannot touch.

That narrows rather than refutes the v3 claim. **N sets where the benefit
crosses zero; composition sets how large the percentage looks.** The published
sentence "the controlling variable is the number of prefill tokens, not the
image-to-text ratio" is right about the crossover and wrong about the magnitude,
and will be corrected to say so.

---

### Q2 — control cell (`f = 0`, pure text stream) complete

Two bench clients, Poisson arrivals, total rate 6/s. Observed in-flight requests:
mean **4.7–4.9**, max 12–13 (the target was ~8; Poisson makes this a distribution
and the measured value is what is reported).

| block | graph off | graph on | effect |
|---|---|---|---|
| b1 | 35.80 ms | 30.13 ms | −15.83% |
| b2 | 35.66 ms | 30.28 ms | −15.07% |
| b3 | 34.87 ms | 29.59 ms | −15.14% |
| **median** | **35.66 ms** | **30.13 ms** | **−15.14%** (spread 0.76 pp) |

**Concurrency does not erode the graph's benefit on a text stream.** Q1 measured
−15.94% for a 544-token text prompt at c=1; this is −15.14% at ~4.8 in flight.
That control is what makes the mixed cells interpretable: any extra degradation
once images enter the stream is attributable to their presence, not to load.

### A precondition check that may invalidate Q2 as designed

Recovering prefill-batch composition from the `f = 0` server log
([`analyze_batch_mix.py`](scripts/analyze_batch_mix.py), no source patch needed —
the two classes have distinct token counts, so `#new-seq` and `#new-token`
determine the split):

```
prefill batches       614
  single-request      605  (98.5%)
  multi-request         9  ( 1.5%)
  #new-seq             {1: 605, 2: 9}
```

**At this load 98.5% of prefill batches carry exactly one request.** The
co-batching interference Q2 exists to detect requires *multi*-request batches —
a text request sharing a batch with an image request. If batches essentially
never combine, a null result means the mechanism never engaged, which is a much
weaker statement than the mechanism engaging and proving harmless.

The cause is scheduling, not the design: with ~5 requests in flight most are
decoding, so the prefill queue rarely holds two at once. Image requests take
~100 ms to prefill against ~30 ms for text, so the mixed cells may queue more —
the `f = 0.2` composition is measured as soon as its log has batches, and if it
is still low single digits the arrival rate is raised toward saturation and the
mixed cells re-run rather than reporting "no effect" from an experiment that
never applied the stress.

This is the same failure shape as three earlier ones in this study — the PCG arm
reporting 100% graph usage while 92% eager, the prefix cache serving 80% of
prefill tokens while every check passed, and the engagement verifier filing
cache-served batches as probes. **The tool reports success while the thing under
test never runs.** Checking it is not optional diligence here; it is the third
time it would have changed a conclusion.

### Q2 — mixed stream at f = 0.2 complete: the deployment number

| block | text | image | break-even f |
|---|---|---|---|
| b1 | −13.84% (save 4.92 ms) | +3.86% (cost 4.20 ms) | 0.539 |
| b2 | −16.40% (save 5.97 ms) | +1.29% (cost 1.41 ms) | 0.809 |
| b3 | −13.30% (save 4.76 ms) | +4.53% (cost 4.87 ms) | 0.494 |
| **median** | **−13.84%** (spread 3.10 pp) | **+3.86%** (spread 3.24 pp) | **0.539** |

Net saving per request as the image share of arrivals rises:

| f | 0.05 | 0.10 | 0.20 | 0.30 | 0.50 |
|---|---|---|---|---|---|
| net | **+4.46 ms** | **+4.01 ms** | **+3.10 ms** | +2.18 ms | +0.36 ms |

**The net stays positive up to a 43–59% image share on TTFT** (the range is the
load confound, not noise). *(Superseded framing: this originally read "at any
realistic image fraction the graph pays… f ≈ 0.05–0.2 nets 3–4.5 ms" — that
fraction was invented and is withdrawn; see the workload-realism entry below.)*

**Break-even is f ≈ 0.54, but the block range is 0.49–0.81 and that width is
honest, not cosmetic.** It comes from the image-side cost being small and noisy
(1.4–4.9 ms, i.e. +1.3% to +4.5%), which is the same magnitude v3 could not
resolve at all. The `f = 1.0` cells measure that cost on 600 image requests
rather than 120 and should tighten it.

The image-class cost measured here (+3.86% median) matches v3's independent
IMG-A measurement (+3.66%) — two different brackets, different concurrency,
same number.

#### Co-batching: present, but too rare to matter at this load

Corrected from the earlier note, which read the `disabled` arm only:

| cell | multi-request batches | of which cross-class | share of all batches |
|---|---|---|---|
| `f=0.2 disabled` b1 | 14 (2.3%) | 0 | 0.0% |
| `f=0.2 breakable` b1 | 9 (1.5%) | 4 | **0.7%** |
| `f=0.2 breakable` b2 | 12 (2.0%) | 1 | **0.2%** |

So co-batching does happen — the graph arm mixes a little more, plausibly because
faster prefill shifts the scheduler's cadence — but at 0.2–0.7% of batches it
cannot move a median even if those requests were badly hurt.

The precise claim is therefore **"co-batching is too rare to matter at this
load"**, not "co-batching does not occur". The distinction is the whole point:
the first is a statement about this operating point, the second would be a
statement about the mechanism, and only a load where batches routinely combine
can support the second. A rate ladder to find that load is the follow-on.

### The break-even depends on which latency you optimise — and on end-to-end there isn't one

The whole study has reported TTFT, because that is where #2 and #4 located the
gap. Recovering `median_e2e_latency_ms` from the same raw files changes the
recommendation:

| class | TTFT effect | **end-to-end effect** | TPOT effect |
|---|---|---|---|
| text | −13.3 to −16.4% | **−1.9 to −3.7%** | −1.3 to −2.7% |
| image | **+1.3 to +4.5%** | **−0.7 to −1.5%** | −0.3 to −1.4% |

**The image class costs 3.9% on TTFT and gains ~1.2% end-to-end.** So on
end-to-end latency there is **no break-even at all** — enabling the graph is net
positive for both classes at every arrival fraction.

The arithmetic is simple once the decomposition is done. End-to-end here is
~910 ms, of which decode is ~870 (128 output tokens at ~6.8 ms). A 4 ms TTFT
change is diluted to 0.5%, while a small but consistent TPOT improvement
multiplied by 128 tokens is worth ~23 ms. For the text class, e2e falls
907.5 → 874.4 ms: only −4.9 ms of that is TTFT and **−28 ms comes from decode**.

**This needs an explanation and I only have a hypothesis.** A prefill CUDA graph
should not touch decode, and both arms run the identical decode backend (`full`).
The plausible route is indirect: under concurrency, prefill and decode interleave
on the same CPU thread, so a prefill that issues far fewer kernel launches leaves
more CPU headroom for the decode loop. The effect is negative in all six cells
across both classes, which is not the shape of noise — but the mechanism is
untested, and it is recorded as an observation, not a finding.

**Consequence for the recommendation: report both metrics, separately.**

- **On TTFT** — break-even at f ≈ 0.52–0.54; net positive at every realistic
  fraction.
- **On end-to-end** — no break-even; net positive everywhere measured.

They do not conflict; they measure different things. Reporting only the first
understates the graph's value, and reporting only the second would hide that the
first-token experience does get worse for image requests. Both go in the report.

This is the check I flagged as owed two steps earlier — "只报一个会误导" — and
it turned out to matter.

---

## Phase 3 complete — what shipped, and what it opened

| deliverable | |
|---|---|
| [`issue4_v3_report.pdf`](issue4_v3_report.pdf) | 7 pages, every number read from the results JSON by the generator |
| [`q1_report.md`](q1_report.md) | matched-N composition test |
| [`q2_report.md`](q2_report.md) | mixed arrival stream, both metrics, confound stated |
| [`pcg_eager_fallback_finding.md`](pcg_eager_fallback_finding.md) | drafted for a separate upstream issue, **not filed** |

**Answers.** Transport is the large lever (−28.2%). The prefill graph's recovery
in milliseconds tracks prefill token count almost regardless of composition; the
percentage tracks composition, because an image buries the same recovery under
vision-encoder time. On a mixed stream the net stays positive to a 43–59% image
share on TTFT and further on end-to-end.

**Five claims of mine were withdrawn during execution**, each recorded with the
measurement that killed it: the `C − k·N` model, the ratio hypothesis, a drift
gate that read absolute levels instead of paired effects, a 24% bucket-padding
artifact mistaken for a result, and "no break-even on end-to-end" stated from one
operating point and contradicted by the next.

**Open, none started:**

1. **Load-matched rerun of Q2** — tune the arrival rate per fraction so in-flight
   requests match, separating image fraction from concurrency. Without it the
   f = 1 end-to-end sign flip cannot be attributed.
2. **A high-load bracket** where prefill batches routinely combine, to test
   co-batching rather than record its absence.
3. **The decode-side effect** — TPOT consistently better under a prefill graph at
   ~4.7 in flight and worse at ~7.2, which a prefill graph should not cause. The
   end-to-end conclusions lean on it.
4. **`tc_piecewise` inside the sub-onset window** (§11.9) — #4's hypothesis names
   PCG and it remains unmeasured on current upstream.
5. **Filing the PCG fallback finding upstream** — owner's call.

---

## Correction: the assumed workload was never checked

The Q2 analysis, the reports and the PDF were all framed around an image arrival
fraction of **5–20%**. **That range was invented.** The brief was qualitative —
users attach an image now and then — and the numbers were mine, with no source,
carried through every deliverable until they were questioned.

Checking them changed two things.

**The fraction cannot be sourced, and the question has no single answer.** The
best public production data is Microsoft's Azure LMM inference trace (one week,
1M requests, image count per request). Its modality mix is balanced by
construction — exactly 500 000 of each — so it cannot supply the number, and the
accompanying paper describes the cluster as serving *image-heavy* and
*text-heavy* services whose behaviour is opposite. The image share is a property
of a deployment. WildChat, LMSYS-Chat-1M and published provider material carry no
modality breakdown either.

**The trace's size distribution is real, and it undercuts the framing.** This
study's own result is that the graph's benefit is set by request size. Real
medians are **792 prefill tokens text-only and 1422 with an image**, against
benchmark prompts of 512. Only **15.2%** of real requests land where a material
win was measured; **75.3%** sit above it. Re-weighting the measured curve over
the trace gives **+3.96 ms** per text-only request and **+0.07 ms** per image
request, against the +4.0–4.5 ms previously quoted — direction intact, magnitude
roughly halved.

**What replaced it.** Every deliverable is now descriptive rather than
prescriptive, and the output is two figures instead of a threshold:
[`fig_mix.png`](figures/fig_mix.png) — net saving against image share, both
metrics, with the load band — and
[`fig_sizes_traffic.png`](figures/fig_sizes_traffic.png) — saving by request
size with the million-request distribution drawn above it. Full note:
[`workload_realism.md`](workload_realism.md).

The lesson is the same shape as the measurement failures earlier in this study:
**a number that was never checked propagated into every conclusion**. Here it was
not a tool reporting false success but an assumption I supplied myself and then
stopped seeing.
