# Issue #4 v3 — IMG-A headline (400 prompts, 5 reps, c=1)

Generated 2026-09-04 11:08 UTC from `results.json`. Stack and model: [`../manifest.md`](../manifest.md).

## Bracket validity

`A0_default` 146.91 ms → `A0_repeat` 145.63 ms, drift **0.87%** (gate ≤ 5.0%): **PASS**

## Arms

| arm | transport | prefill backend | TTFT p50 | CV | TPOT p50 | engagement |
|---|---|---|---|---|---|---|
| `A0_default` | cpu | breakable | 146.91 ms | 1.7% | 5.65 ms | **VERIFIED** |
| `A1_disabled` | cpu | disabled | 142.30 ms | 1.6% | 5.67 ms | **VERIFIED** |
| `A2_tcp` | cpu | tc_piecewise | 147.02 ms | 0.7% | 5.68 ms | **UNVERIFIED** |
| `A3_bcg` | cpu | breakable | 148.62 ms | 1.7% | 5.65 ms | **VERIFIED** |
| `A4_ipc` | cuda_ipc | breakable | 105.91 ms | 1.8% | 5.71 ms | **VERIFIED** |
| `A5_ipc_nograph` | cuda_ipc | disabled | 102.17 ms | 1.4% | 5.68 ms | **VERIFIED** |
| `V0_vllm` | — | — | 161.99 ms | 0.8% | 5.16 ms | **VERIFIED** |
| `A0_repeat` | cpu | breakable | 145.63 ms | 4.4% | 5.66 ms | **VERIFIED** |

Arms marked UNVERIFIED are excluded from every comparison below. Their numbers are shown only so the exclusion can be audited.

- `A2_tcp` — PCG eager fallback on 92.11% of graph-eligible calls (75200/81638, 2 distinct shapes) — arm ran partially eager

## The four questions, answered separately

### Q1 — Cross-framework gap, #4's standard condition

#4's SGLang image baseline is IPC-on. How far is it from vLLM on image+text? This is #2's Case-A question moved to the image path.

`A5_ipc_nograph` 102.17 ms vs `V0_vllm` 161.99 ms → **-36.93%** — faster.

### Q1b — Cross-framework gap, today's upstream default transport

The same comparison from the transport a user gets without setting anything on current upstream (`cpu`). Reported alongside Q1 because the two answer different questions: what #4 specifies versus what ships.

`A0_default` 146.91 ms vs `V0_vllm` 161.99 ms → **-9.31%** — faster.

### Q2 — Does #2's PCG win transfer to images?

#2 showed tc_piecewise took text-only Case A from 21.94 ms to 14.04 ms. Against the no-prefill-graph floor, does the same lever pay on images?

**Unanswered.** `A2_tcp` did not verify: PCG eager fallback on 92.11% of graph-eligible calls (75200/81638, 2 distinct shapes) — arm ran partially eager.

### Q3 — CUDA IPC feature transport, isolated

The transport lever with the graph held OFF on both sides — #4's "separate IPC benefit from PCG benefit" requirement. The graph-ON counterpart is `A4_ipc` vs `A0_default`; both appear in the 2×2.

`A5_ipc_nograph` 102.17 ms vs `A1_disabled` 142.30 ms → **-28.20%** — faster.

### Q4 — Prefill CUDA graph value under #4's standard condition

What the breakable prefill graph buys with IPC on, which is the configuration #4 specifies. The cpu-row counterpart (`A0_default` vs `A1_disabled`) is the ablation, and both appear in the 2×2.

`A4_ipc` 105.91 ms vs `A5_ipc_nograph` 102.17 ms → **+3.66%** — no material difference.

## The two levers as a 2×2

TTFT p50, median of 5 reps.

| prefill graph | cpu transport | cuda_ipc transport | IPC effect |
|---|---|---|---|
| `disabled` | 142.30 ms | 102.17 ms | **-28.20%** |
| `breakable` | 146.91 ms | 105.91 ms | **-27.91%** |
| **graph effect** | **+3.24%** | **+3.66%** | |

Interaction (graph effect under IPC minus graph effect under CPU): **+0.42 pp**. A value near zero means the levers are independent — each pays the same regardless of the other. A large value means one lever's benefit depends on the other's setting.


---

## Reading of these numbers

Everything above is generated mechanically from the bracket. This section is the
interpretation, and it is kept separate so the two are not confused.

### The headline reverses #2's text-only conclusion

#2 found SGLang **slower** than vLLM on text-only Case A: 21.94 ms against
13.12 ms, a 67% deficit that motivated this whole line of work. On the image
path, at the configuration issue #4 specifies, the sign flips:

| | SGLang | vLLM | gap |
|---|---|---|---|
| #2, text-only Case A | 21.94 ms | 13.12 ms | SGLang **+67%** |
| #4 IMG-A, IPC-on | **102.17 ms** | 161.99 ms | SGLang **−36.9%** |
| #4 IMG-A, upstream default transport | 146.91 ms | 161.99 ms | SGLang **−9.3%** |

So **#4's premise that image workloads "may behave differently" is confirmed,
but not in the direction it expected.** #4 guessed PCG would help more on images;
what actually differs is that SGLang's image path is faster than vLLM's, while
its text path was slower.

The gap is entirely on the first-token side. TPOT is 5.65–5.71 ms for SGLang and
**5.16 ms for vLLM** — vLLM decodes ~9% faster. A workload dominated by long
generations rather than TTFT would rank these differently, and nothing here
speaks to that.

### The transport lever dominates; the graph lever does not exist here

- **CUDA IPC feature transport: −28.2%** (`A5_ipc_nograph` vs `A1_disabled`,
  graph off on both sides, which is #4's "separate IPC benefit from PCG benefit"
  requirement). This is the whole story of the image path at c=1.
- **Prefill CUDA graph: +3.66%** under #4's standard condition — a *cost*, and
  below the 5% materiality rule, so it is reported as no material difference.

The 2×2's interaction is **+0.42 pp**, so the two levers are independent: IPC
pays ~28% whether the graph is on or off, and the graph costs ~3.2–3.7%
whichever transport is used. The consistency matters more than either number:
three independent estimates of the graph effect (+3.24% cpu row, +3.66% IPC row,
+4.4% against the explicit-flag arm `A3_bcg`) all have the same sign at CV ≈ 1.5%.

That the default transport is `cpu` on current upstream therefore costs a
Qwen3-VL image deployment about **28% of TTFT** — which is the practical finding
here, and it has nothing to do with CUDA graphs.

### What is not answered, and why

**Q2 is unanswered, not negative.** `A2_tcp` ran 92.11% of its graph-eligible
calls eagerly (75 200 / 81 638, 2 shapes), so its 147.02 ms is an eager number
wearing a `tc_piecewise` label. #4's hypothesis names PCG specifically, so this
is a real gap. [`plan.md` §11.9](../../../plan.md) gives a protocol that can
measure it honestly — onset is deterministic at graph-eligible call ~6402, so
runs of ≤150 prompts on a fresh server stay entirely pre-onset.

**One operating point is not a claim about images in general.** IMG-A is a single
720p image plus ~128 text tokens at c=1 — ~1010 tokens through the LM prefill,
about 8× the token count of #2's Case A. The follow-on sweep
([`plan.md` §11.8](../../../plan.md), results in `imgR_report.md`) varies that
count from ~128 to ~2100 to find whether the graph pays anywhere.

### Caveats that travel with these numbers

1. **Cross-framework outputs are not token-identical on images.** Phase-1 parity
   found both text fixtures exact across all arms including vLLM, and both image
   fixtures divergent in wording with identical content — which localises the
   difference to the vision path, not the LM or sampling. Q1 and Q1b are
   framework-to-framework comparisons, **not** strict like-for-like equivalence.
2. **vLLM is an anchor at its defaults**, with prefix caching disabled to match
   the SGLang arms. It was not tuned. A tuned vLLM configuration is a different
   experiment.
3. **The library stack is older than what upstream pins** (torch 2.11 against a
   pinned 2.13 — [`manifest.md`](manifest.md) §2). Every arm shares the confound
   identically, so internal contrasts hold; the absolute latencies are not
   production claims.
4. **`A0_repeat` carries CV 4.4%**, against 1.4–1.8% for every other verified
   arm. It still passes the band and the drift gate (0.87%), but it is the one
   arm whose spread is worth watching if this bracket is repeated.
