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
3. **It is nondeterministic across runs.** The pre-instrumentation smoke fired
   the warning; this run showed `eager_fallback=0` at eligible=2000 and only
   began at 6402. Whether an arm degrades at all depends on when Dynamo meets a
   previously-unseen guard.

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
