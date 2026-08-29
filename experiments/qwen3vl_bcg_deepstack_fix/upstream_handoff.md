# Upstream handoff — Qwen3-VL DeepStack BCG replay-slot fix

> **Current source of truth for this track.** Supersedes
> [`submission_package.md`](submission_package.md) and
> [`pr_submission_manual.md`](pr_submission_manual.md), both of which were
> written before the PR existed and describe an obsolete branch snapshot.
> Last verified against the live GitHub API: **2026-08-29 UTC**.

## Live PR state

| | |
|---|---|
| PR | [sgl-project/sglang#33726](https://github.com/sgl-project/sglang/pull/33726) |
| Title | `fix(bcg): preserve Qwen3-VL DeepStack inputs during replay` |
| Opened | 2026-08-05 |
| State | open, not merged |
| Head | `c31e6fe315` on `bowenwan6:fix/bcg-deepstack-replay-slot` |
| Base | `sgl-project:main` @ `6afb5e1771` |
| Mergeable | **true** (`mergeable_state: unstable` — CI in flight) |
| Diff | **+294 / −12 across 7 files**, 20 commits |

### Review status

| Reviewer | State | Date | Disposition |
|---|---|---|---|
| [@JustinTong0323](https://github.com/JustinTong0323) (Xinyuan Tong) | **APPROVED** | 2026-08-06 | Also co-maintains the branch — has pushed test-pruning, comment-cleanup, allowlist, and several `main` merge commits directly. |
| [@charliechenye](https://github.com/charliechenye) | COMMENTED | 2026-08-14 | Fail-closed review; addressed in `f2596d5e99`. No re-review requested since. |

Note these are two different people; earlier docs in this track conflate them.

## Files touched

| File | Role |
|---|---|
| `srt/model_executor/runner/prefill_cuda_graph_runner.py` | the fix: `_refresh_deepstack_replay_slot` helper + replay call site |
| `srt/model_executor/runner_utils/buffers.py` | `input_deepstack_embeds` buffer allocation |
| `srt/model_executor/cuda_graph_buffer_registry.py` | slot registration |
| `srt/models/qwen3_vl.py` | `supports_bcg_deepstack_replay` capability opt-in |
| `srt/configs/model_config.py` | BCG allowlist entries (added by the maintainer) |
| `test/registered/unit/model_executor/test_deepstack_replay_slot.py` | new: 8 tests / 11 subtests |
| `test/registered/unit/server_args/test_server_args.py` | allowlist coverage |

## Chronology of the branch

| Date | Event |
|---|---|
| 2026-08-05 | PR opened from the validated fix branch |
| 2026-08-06 | Maintainer approval |
| 2026-08-14 | Charlie's fail-closed review: a malformed non-`None` DeepStack tensor was warned-and-zeroed, the same silent-drift class the PR exists to kill |
| 2026-08-15 | `f2596d5e99` — fail closed: validate before any write, raise on a non-empty tensor that does not fit the slot; `None`/empty still clear it as genuine absence |
| 2026-08-15/16 | Maintainer pruned tests, trimmed comments, added the Qwen3-VL + Qwen3-VL-MoE allowlist entries |
| 2026-08-16 | M9 MoE smoke (`Qwen3-VL-30B-A3B-Instruct`) |
| 2026-08-17 → 25 | Maintainer merged `main` five times to keep the branch current |
| **2026-08-29** | **`c31e6fe315`** — merged `upstream/main` (263 commits) resolving 5 conflicts from #35451 and #35758; M10 post-merge smoke |

## The 2026-08-29 merge

`git merge upstream/main` conflicted in two files, five hunks. Every hunk was an
*additive* collision — both sides inserted a new field, parameter, or kwarg at
the same anchor — so all five resolved as a union, keeping both sides.

| File | Ours | Theirs (upstream) |
|---|---|---|
| `buffers.py` ×3 | `input_deepstack_embeds` field; `deepstack_replay_width` param; its `cls(...)` kwarg | `pp_proxy_tensors` field; `pp_size` / `hc_hidden_size` / `pp_proxy_topk_size` / `pp_proxy_residual_num_blocks`; its kwarg |
| `prefill_cuda_graph_runner.py` ×2 | `deepstack_replay_width=…`; `**extra_kwargs` | the four PP kwargs; `**pp_kwargs` |

Union resolution is provably safe for the kwarg splat: `ModelRunner._pp_kwargs()`
returns only `{"pp_proxy_tensors": …}` or `{}`, which is disjoint from
`input_deepstack_embeds`, so `**extra_kwargs, **pp_kwargs` cannot raise a
duplicate-keyword `TypeError`.

Evidence the merge changed nothing semantically: the net PR delta against the
*new* base is still **+294 / −12 across 7 files**, byte-identical to the
pre-merge figure.

## Validation ledger

| Milestone | Scope | Verdict |
|---|---|---|
| M2 / M4 / M4b / M4c | dense Qwen3-VL-8B, four arms, pre/post fix | `FAIL_BCG_DEEPSTACK` → `PASS_BCG_CORRECT` |
| M7 | isolation | pass |
| M8 | Qwen3.5 gate | `NOT_APPLICABLE_QWEN35` |
| [M9](results/m9_moe_smoke_gpu1_20260816T092836Z/report.md) | MoE `Qwen3-VL-30B-A3B-Instruct`, pre-merge tree | PASS |
| [M10](results/m10_postmerge_dense_smoke_gpu5_20260829T163500Z/report.md) | dense `Qwen3-VL-4B-Instruct`, **post-merge** tree | PASS |
| CPU unit suite (post-merge) | `test_deepstack_replay_slot` + `test_server_args` | 193 passed, 30 subtests |

## Known open items

1. **CI on `c31e6fe315`** was still running at the time of writing. On the
   previous head the only reds were AMD ROCm / NPU / XPU lanes plus the
   aggregate `finish` gate; every NVIDIA and CPU lane was green, including
   `base-a-test-cpu`, which had been red earlier and has since cleared. Those
   AMD reds sit outside this change's blast radius (CUDA-only DeepStack path)
   but were never root-caused from their logs.
2. **Production-stack confirmation.** Every smoke in this track ran in a
   container now ~771 commits behind, requiring `LD_PRELOAD` for a CUDA 803
   stub and `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1`. Both arms share the
   confound so each comparison is internally controlled, but no run here is
   production-representative. The maintainer asked for a confirming run on an
   up-to-date devbox; that remains unrun.
3. **Post-merge MoE repeat.** M10 covers the dense arch on the post-merge tree;
   M9 covers MoE on the pre-merge tree. The MoE × post-merge cell is empty.
4. **Reviewer item 6 (optional).** Tighten `assertRaises(RuntimeError)` to
   `assertRaisesRegex` so the guard's message is pinned. One-line change, not
   yet made.

## Scope boundary — do not conflate

This PR fixes the **breakable CUDA graph (BCG)** prefill path. It is *not* the
**TC-piecewise PCG** path that Issue #4's capture-stream sub-track and Issue #5's
selective-enablement question concern. They are different graph backends. A
green BCG result here does not close either of those, and BCG must not be
silently substituted for the PCG arm in any benchmark protocol.
