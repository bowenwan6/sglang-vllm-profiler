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

### 0.3 — port the runner to the v3 flag surface — *pending*

### 0.4 — engagement verifier — *pending*
