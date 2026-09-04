# Issue #4 v3 — frozen environment manifest

Corresponds to [`plan.md`](../../../plan.md) §11.3 step **0.1** and **0.2**.
Frozen 2026-09-04. **Every arm in the bracket shares this stack, or the
comparison is void.**

## 1. Step 0.2 — SGLang SHA policy and the #33726 merge-state decision

Checked live on 2026-09-04:

```
GET /repos/sgl-project/sglang/pulls/33726
  state=open  merged=false  merged_at=null  head=32dbab0bb1  base=main
```

**PR #33726 is NOT merged.** Plan §11.3 step 0.2 prescribes, for that branch of
the decision, "pin a pre-merge SHA; `A0_default` will resolve to `disabled`;
`A3_bcg` is the preview of the incoming default."

**We deviate from the literal wording, and pin a merged-preview stack instead.**
Reason — a pre-merge pin cannot answer Q4 at all:

- On unpatched upstream, `Qwen3VLForConditionalGeneration` is absent from
  `multimodal_breakable_cuda_graph_supported_model_archs`
  (verified at `ff1285cc28`, `model_config.py:1985-1994` — list contains
  Cohere2Vision, InternS2Mobius, PaddleOCRVL, Qwen3_5, Qwen3_5Moe, MuseGlimmer,
  KimiK3, KimiK25; **no Qwen3VL**).
- Forcing `--cuda-graph-backend-prefill breakable` there bypasses the
  auto-disable cascade (`cuda_graph_hook.py:115`) and *runs* — but it runs the
  exact DeepStack replay bug that PR #33726 fixes. The arm would produce a
  clean-looking latency number attached to **numerically wrong output**.
- Measuring the speed of a known-incorrect path is not a result.

The decision preserves the *intent* of 0.2 — never straddle the merge inside
one bracket — by pinning **one** stack for all arms:

| | |
|---|---|
| Stack | `/data/sglang-fork` branch `exp/issue4-v3` @ `48b0365bcc` |
| = | `upstream/main` @ `ff1285cc28` (2026-09-03 20:24 -0700) merged into `fix/bcg-deepstack-replay-slot` @ `d1cd8c583b` |
| Merge result | clean, no conflicts |
| PR under test | #33726 `fix(bcg): preserve Qwen3-VL DeepStack inputs during replay` |

**Consequence for arm semantics** — this is the post-merge world, so:

- `A0_default` resolves to **`breakable`** (Qwen3-VL is on the allowlist on this
  stack). It is therefore the *post-merge* production default, not today's.
- `A1_disabled` is **today's effective production behaviour** for Qwen3-VL:
  upstream auto-disables the prefill graph for this arch, so `disabled` is what
  a user on `ff1285cc28` actually gets. It is reached by an explicit flag rather
  than by the default, but the resolved configuration is identical.
- Both worlds are therefore measured, on one stack, with every arm numerically
  correct. `A0_default` must still **record what it resolves to** and the
  resolution must be checked, not assumed.

## 2. Step 0.1 — rebuild vs stale-but-controlled

**Decision: run from source via `PYTHONPATH`, do not rebuild the container.**

`upstream/main`'s `python/pyproject.toml` pins `torch==2.13.0`,
`transformers==5.12.1`, `sglang-kernel==0.4.6.post1` (note: `sgl-kernel` was
renamed). The container ships torch `2.11.0+cu130`, transformers `5.8.1`,
`sgl_kernel 0.4.5`. Honouring those pins means replacing torch, which would also
invalidate the vLLM anchor environment (same torch build) — i.e. it would
destroy the very cross-framework comparison the bracket exists to make.

Running from source via `PYTHONPATH` does not enforce pyproject pins. This is
the recipe already proven on this box by the M10 post-merge smoke
(2026-08-29, `qwen3vl_bcg_deepstack_fix/results/m10_postmerge_dense_smoke_gpu5_20260829T163500Z`).

**Confound accepted and stated**: absolute latencies are not production claims,
because the library stack is older than what upstream pins. Every arm shares the
confound identically, so *internal contrasts hold*. Repeated in the report per
§11.6 risk 5.

## 3. Frozen stack

| Component | Value |
|---|---|
| SGLang source | `/data/sglang-fork` `exp/issue4-v3` @ `48b0365bcc` (via `PYTHONPATH=/data/sglang-fork/python`) |
| SGLang upstream base | `ff1285cc28d6b3e0ad19c45e8b14a5966bc95c78` |
| Profiler repo | `/data/sglang-vllm-profiler` @ `156c4b7131` |
| Installed sglang (**must never be imported**) | `0.0.0.dev1+gda802ddca` at `/sgl-workspace/sglang` — 3123 commits behind `upstream/main` |
| vLLM | `0.21.0` in `/opt/miniconda3/envs/profiling` (`/opt/miniconda3/envs/profiling/bin/python`) |
| torch | `2.11.0+cu130` (identical in both envs) |
| transformers | `5.8.1` (system) / `5.12.1` (vLLM env) |
| sgl_kernel | `0.4.5` — requires `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` |
| flashinfer-python | `0.6.12` (system) / `0.6.8.post1` (vLLM env) |
| xgrammar | `0.2.1` |
| CUDA toolkit | `13.0` V13.0.88 |
| Driver | `595.71.05` |
| Runtime libcuda | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` (overrides `cuda-compat-13-0` loader precedence) |
| GPU | **7 only** — NVIDIA H200, 143771 MiB (user-specified, never auto-switch) |
| Python | 3.12.3 |
| Attention backend | `flashinfer` (unchanged from v2) |

## 4. Model

| | |
|---|---|
| Model | `Qwen/Qwen3-VL-8B-Instruct` |
| Revision | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` — **byte-identical to the revision v2's protocol pinned** |
| Snapshot | `/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| Size on disk | 17 GB, 4 safetensors shards |
| Arch | `Qwen3VLForConditionalGeneration` |
| dtype | `bfloat16` (explicit `--dtype bfloat16`) |

**Feature-engagement precondition (checked, not assumed):**

```
text_config.hidden_size            = 4096
text_config.num_hidden_layers      = 36
vision_config.deepstack_visual_indexes = [8, 16, 24]   → 3 entries
vision_config.out_hidden_size      = 4096
⇒ deepstack replay width = 4096 × 3 = 12288  (> 0)
```

`deepstack_visual_indexes` is **non-empty**, so this target genuinely populates
the DeepStack replay path that #33726 fixes and that `A3_bcg` / `A0_default`
exercise. (Contrast: Qwen3.5-4B ships `deepstack_visual_indexes = []`, which is
exactly why issue #9 closed as `NOT_APPLICABLE_QWEN35`.)

Rejected alternative: `Qwen3-VL-4B-Instruct` (already cached, `[5, 11, 17]`,
used by the M10 smoke). It would also exercise the path, but it is not the model
v2's protocol pinned, so it would break continuity with the #2 / #4 baseline
line for no gain.

## 5. Launch invariants

Every SGLang arm:

```
CUDA_VISIBLE_DEVICES=7
PYTHONPATH=/data/sglang-fork/python
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
HF_HUB_OFFLINE=1
```

Never set: `SGLANG_KERNEL_API_LOGLEVEL`, `SGLANG_KERNEL_API_LOGDEST`, any
profiler flag. `SGLANG_USE_CUDA_IPC_TRANSPORT` is **never** set — it is the
deprecated surface (§11.1 change A); transport is selected with
`--mm-feature-transport`.

## 6. Package readiness check

| Requirement | State |
|---|---|
| SGLang source at latest upstream + PR | ✅ `48b0365bcc`, clean merge |
| DeepStack fix present in source | ✅ `cuda_graph_buffer_registry.py:809,898` (`deepstack_replay_width`), `qwen3_vl.py` replay path |
| Qwen3VL on breakable allowlist | ✅ `model_config.py:1992-1993` on the exp branch; ❌ absent upstream (as expected) |
| CUDA usable | ✅ `torch.cuda.is_available()=True`, 8 devices, with `LD_PRELOAD` |
| sgl_kernel importable | ✅ `0.4.5` (version check skipped by env var) |
| vLLM importable | ✅ `0.21.0` in the profiling env |
| Model weights local | ✅ 17 GB, pinned revision |
| GPU 7 free | ✅ 212 MiB at Phase-0 time |
