# M9 — Qwen3-VL MoE smoke under breakable prefill CUDA graph

Requested by the PR reviewer: the allowlist commit adds
`Qwen3VLMoeForConditionalGeneration` alongside the dense arch, but hardware
validation to that point covered only the dense 8B. The MoE variant inherits
both the allowlist entry and `supports_bcg_deepstack_replay`, so it needed to be
served at least once before merge.

## Verdict

PASS. BCG replay and eager produce byte-identical greedy completions for both an
image and a text prompt, with BCG demonstrably engaged on both substantive
prefills.

## Target exercises the feature

Being on the allowlist does not by itself mean the DeepStack path runs. Checked
before spending GPU time:

| Property | Value |
|---|---|
| Repo | `Qwen/Qwen3-VL-30B-A3B-Instruct` |
| Snapshot | `9c4b90e1e4ba969fd3b5378b57d966d725f1b86c` |
| `architectures` | `["Qwen3VLMoeForConditionalGeneration"]` |
| `vision_config.deepstack_visual_indexes` | `[8, 16, 24]` (non-empty) |

Non-empty indexes mean `num_deepstack_embeddings = 3`, so the replay slot is
allocated with a non-zero width and the copy actually happens. This is not the
vacuous case seen with the released Intern-S2-Mobius checkpoint, whose indexes
are `[]`.

## Method

Two arms, same server code (PR branch via `PYTHONPATH`), same GPU, same inputs:

| Arm | Flag |
|---|---|
| `bcg` | `--cuda-graph-backend-prefill breakable` |
| `eager` | `--disable-prefill-cuda-graph` |

Both arms: TP=1 on one H200, `--disable-radix-cache` so prefix reuse cannot mask
a replay difference, greedy (`temperature=0`, `seed=0`), `max_tokens=48`.
The image is generated in-process as fixed RGB vertical stripes, so both arms
receive byte-identical input with no download or disk dependency.

## Result

Completions were byte-identical across arms.

Image prompt ("Describe the colors in this image in order."), both arms:

```text
Of course. Here is a description of the colors in the image, listed from left to
right.

The image displays a pattern of seven vertical stripes of equal width. The
colors are arranged in a repeating sequence.

-   **Red**
-
```

Text prompt ("Name the first four prime numbers."), both arms:

```text
The first four prime numbers are:

1. **2**
2. **3**
3. **5**
4. **7**

These are the first four numbers greater than 1 that have no positive divisors other than
```

## BCG was actually engaged

Identical output only means something if the `bcg` arm did not silently fall back
to eager. Per-prefill graph status, from the two server logs:

| `#new-token` | `bcg` arm | `eager` arm |
|---:|---|---|
| 78 | `cuda graph: True` | `cuda graph: False` |
| 1 | `cuda graph: False` | `cuda graph: False` |
| 1 | `cuda graph: False` | `cuda graph: False` |
| 119 | `cuda graph: True` | `cuda graph: False` |
| 15 | `cuda graph: True` | `cuda graph: False` |

Identical request sequence in both arms; the only difference is graph engagement.
Both substantive prefills replayed under BCG. Capture banner from the `bcg` arm:

```text
Capture target prefill CUDA graph begin. backend=breakable, num_tokens=[4, 8, ...]
Capture target prefill CUDA graph end. elapsed=47.06 s, mem usage=2.74 GB
```

58 token buckets captured. The `eager` arm shows no prefill capture.

This is the same semantic control as the dense-8B four-arm experiment: before the
fix, a BCG-replayed image prefill dropped its DeepStack contribution and diverged
from eager. Here the BCG-replayed image prefill matches eager exactly, so the
contribution survived replay on the MoE arch.

## Environment caveat

This container is materially behind the PR branch, which now sits on upstream
main several hundred commits ahead of what the image was provisioned for. Three
blockers were worked around:

1. **CUDA error 803.** Zero-byte `libcuda.so.570.172.08` / `libcuda.so.595.84`
   stubs mislead the loader. Fixed with `LD_PRELOAD` of the real
   `libcuda.so.595.71.05`, the same workaround the Qwen3.5-4B harness carries.
2. **`sglang-kernel` 0.4.5 < required 0.4.6.post1.** An isolated `--target`
   install of 0.4.6.post1 failed to import: its compiled extension needs a newer
   torch than this container ships (`undefined symbol: torch::Library::_def`).
   A global upgrade was therefore not attempted — it would have broken other
   users' jobs running on this host and still not worked.
3. Ran with `SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1` on the container's existing
   0.4.5 kernel.

Both arms ran on that same stack, so the eager-vs-BCG comparison stays a
controlled test: the kernel version is a shared confound that cancels between
arms. What this run does **not** establish is behaviour on a
production-representative stack. A confirming run on an up-to-date devbox is
still worth having before merge.

## Artefacts

`raw/` holds the two server logs and the two client responses (untracked, per the
repo convention for generated outputs). `scripts/` holds the runner and client.
