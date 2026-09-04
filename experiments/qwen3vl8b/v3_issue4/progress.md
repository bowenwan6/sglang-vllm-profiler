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
