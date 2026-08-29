# M10 — post-merge dense Qwen3-VL smoke under breakable prefill CUDA graph

Date: 2026-08-29 UTC · GPU 5 (H200, idle) · PR branch `fix/bcg-deepstack-replay-slot`

## Why this run exists

Upstream moved 263 commits between the PR's last sync (`bf1e03f712`, 2026-08-25)
and `6afb5e1771`. Two of those commits collide with this PR:

- **#35451** `[Feature] Support PP in full prefill CUDA graphs`
- **#35758** `qwen 3.8 rebase`

#35451 matters beyond the textual conflict: it rewires the exact closure the
DeepStack fix lives in. The capture path now threads `pp_proxy_tensors` through
`layer_model.forward`, and `PrefillInputBuffers` gained PP proxy buffers
alongside our replay slot. Unit tests cover the helper in isolation; they do not
prove the capture/replay path still works end to end after that rewire.

This run closes that gap on the merged tree before the merge was pushed.

## Verdict

PASS. BCG replay and eager produce byte-identical greedy completions on both an
image and a text prompt, with BCG demonstrably engaged on every substantive
prefill.

## Target exercises the feature

Being on the allowlist does not by itself mean the DeepStack path runs. Checked
before spending GPU time:

| Property | Value |
|---|---|
| Repo | `Qwen/Qwen3-VL-4B-Instruct` |
| Snapshot | `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| `architectures` | `["Qwen3VLForConditionalGeneration"]` |
| `vision_config.deepstack_visual_indexes` | `[5, 11, 17]` (non-empty) |
| On `multimodal_breakable_cuda_graph_supported_model_archs` | yes |

The width chain is closed in source, not assumed:
`qwen3_vl.py:1369` sets `num_deepstack_embeddings = len(deepstack_visual_indexes)`
= 3, the class carries `supports_bcg_deepstack_replay = True`, and the runner
computes `deepstack_replay_width = hidden_size * 3 > 0`. A non-zero width is what
allocates the slot, so the copy genuinely happens.

The 4B was chosen over the 8B/30B used in M4/M9 purely for turnaround: the
Qwen3-VL weights had been purged from the container's HF cache, and the 4B is
the smallest checkpoint carrying the same architecture and a non-empty index
list.

## Method

Two arms, same server code (merged working tree via `PYTHONPATH`), same GPU,
same inputs:

| Arm | Flag |
|---|---|
| `bcg` | `--cuda-graph-backend-prefill breakable` |
| `eager` | `--disable-prefill-cuda-graph` |

Both arms: TP=1 on one H200, `--disable-radix-cache` so prefix reuse cannot mask
a replay difference, greedy (`temperature=0`, `seed=0`), `max_tokens=48`,
`--mem-fraction-static 0.80`. The image is generated in-process as fixed RGB
vertical stripes, so both arms receive byte-identical input with no download or
disk dependency.

## Result

Completions were byte-identical across arms (`diff` empty).

Image prompt ("Describe the colors in this image in order."), both arms:

```text
The image consists of vertical stripes in the following order from left to right:

1. Red
2. Green
3. Blue
4. Yellow
5. Red
6. Green
7. Blue
8. Yellow
```

The answer is also *correct* for the generated stripe pattern, which matters:
identical-but-wrong would not have distinguished a working visual path from a
broken one shared by both arms.

Text prompt ("Name the first four prime numbers."), both arms:

```text
The first four prime numbers are:

**2, 3, 5, 7**
```

## BCG was actually engaged

Identical output only means something if the `bcg` arm did not silently fall
back to eager. Per-prefill graph status, from the two server logs:

| `#new-token` | `bcg` arm | `eager` arm |
|---:|---|---|
| 78 | `cuda graph: True` | `cuda graph: False` |
| 1 | `cuda graph: False` | `cuda graph: False` |
| 1 | `cuda graph: False` | `cuda graph: False` |
| 119 | `cuda graph: True` | `cuda graph: False` |
| 15 | `cuda graph: True` | `cuda graph: False` |

Identical request sequence in both arms; the only difference is graph
engagement. All three substantive prefills replayed under BCG. Capture banner
from the `bcg` arm:

```text
Capture target prefill CUDA graph begin. backend=breakable, num_tokens=[4, 8, ... 8192], avail mem=25.80 GB
Capture target prefill CUDA graph end. elapsed=30.90 s, mem usage=1.40 GB, avail mem=24.39 GB.
```

58 token buckets captured. The `eager` arm shows **zero** `Capture target
prefill` lines, confirming the control arm really is uncaptured.

## Relationship to earlier milestones

Same semantic control as the dense-8B four-arm experiment (M2/M4): before the
fix, a BCG-replayed image prefill dropped its DeepStack contribution and
diverged from eager. Here the BCG-replayed image prefill matches eager exactly
on the post-merge tree, so the contribution survives replay after #35451's
rewire.

M9 established the same property on the MoE arch (`Qwen3-VL-30B-A3B-Instruct`)
but against the pre-merge tree. M10 covers the post-merge tree on the dense
arch. Neither supersedes the other; a post-merge MoE repeat remains unrun.

## Accompanying static checks (merged tree, CPU)

| Check | Result |
|---|---|
| Conflict markers remaining | none |
| `py_compile` on both conflicted files | pass |
| `black` / `isort` | clean |
| `test_deepstack_replay_slot.py` + `test_server_args.py` | 193 passed, 30 subtests |
| Net PR delta vs new upstream main | +294 / −12 across 7 files — unchanged by the merge |

## Environment caveat

Unchanged from M9, and now worse: this container is roughly 771 commits behind
the PR branch. Two workarounds were required:

1. **CUDA error 803.** Zero-byte `libcuda.so` stubs mislead the loader. Fixed
   with `LD_PRELOAD` of the real `libcuda.so.595.71.05`.
2. **`sglang-kernel` 0.4.5 below the required version.** Ran with
   `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` on the container's existing kernel.
   A global upgrade was not attempted: the newer wheel needs a newer torch than
   this container ships, and other users' jobs share this host.

Both arms ran on that same stack, so the eager-vs-BCG comparison stays a
controlled test — the kernel version is a shared confound that cancels between
arms. What this run does **not** establish is behaviour on a
production-representative stack. A confirming run on an up-to-date devbox
remains worth having, as the reviewer requested.

## Artefacts

`raw/` holds the two server logs and the two client responses. `scripts/` holds
the runner (`vl_smoke.sh`) and client (`vl_client.py`).
