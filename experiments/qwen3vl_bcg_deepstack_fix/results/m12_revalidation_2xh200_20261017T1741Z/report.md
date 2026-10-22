# M12 — revalidation of PR #33726 at head `ccfda8dfd9`

Date: 2026-10-17 UTC · one RADIX node, 2× H200, assigned 17:41 UTC · plan:
[`../../m12_revalidation_plan.md`](../../m12_revalidation_plan.md) (committed before any GPU run, the Q5
amendment before any Q5 data, on a branch later squashed into `main`).

## Verdict

- **The fix is still needed on today's `main`.** The control tree `allow` (`main` plus the two allowlist
  lines, no slot) under BCG changes 29 of 31 image and video answers against eager (largest log-probability
  difference 9.61). C3 detected the bug class.
- **With the PR, BCG and FULL follow eager to bf16 rounding.** Dense 8B, TP=1: 24 of 31 visual requests
  token-identical to eager, the other 7 split at a near tie (both top-2 gaps at most one bf16 step, 0.25);
  largest visual difference 0.50, against 0.59 on text-only requests, which never touch DeepStack. The same at
  chunk size 2048, where the 16.5 s video spans five chunks. TP=2: 28 of 31 identical, 3 near ties, nothing
  else. MoE 30B-A3B is noisier under a graph backend, on text-only requests as much as on visual ones, so its
  two non-tie visual splits carry no DeepStack signal (details below).
- **The pre-registered verdict is FAIL as written**, on three rules, none of them about DeepStack: C1 for every
  graph arm (its 0.1 tolerance is below the 0.25 step of the returned log probabilities), the third C2 pair (a
  bf16 tie at position 17 of the long video, explained below), and P-1 on the text class (+1.1 ms against a
  1.0 ms limit). Each is reported as measured; no rule was changed and nothing was re-run.
- **Finding: `main`'s FULL prefill backend drops Qwen3-VL's DeepStack inputs.**
  `--cuda-graph-backend-prefill full` on `main` behaves like the no-slot control: 29 of 31 visual answers
  change, and they read as plausible descriptions of a different picture. FULL has no multimodal
  allowlist. The PR fixes this for the Qwen3-VL family (`pr/full`: 23 identical, 8 near ties, 0 other).
- **CI at `ccfda8dfd9`:** `base-a-test-cpu` fails on the PR's own test (#44131 made replay read
  `is_dsv41_model`, which the `__new__`-built test runner lacks). Local commit `8560a72b8c` sets it; on the
  node the error reproduces without it and the file passes 12/12 with it. Not pushed. The `extra-*` root cause
  (Qwen2.5-VL-3B TTFT ceiling) is a model this PR does not reach.
- **Q5, both of Bowen's PRs together (#33726 + #44129):** no interaction. One request at a time, adding #44129
  changes no log probability, eager or BCG, at chunk 8192 or 2048; with both PRs BCG follows eager as with
  #33726 alone.

## Setup

| Tree | Commit | Root tree | What |
|---|---|---|---|
| `main` (node `base`) | `80f40522cd` | `4303668ed3c9` | the `main` merged into the PR head |
| `pr` (`bcgfix`) | `8560a72b8c` | `73918a02a69c` | PR head `ccfda8dfd9` + the one-line test fix |
| `pr` without the fix (`bcghead`) | `ccfda8dfd9` | | for the CI reproduction only |
| `allow` (`bcgallow`) | `b0c8105c75` | | `main` + `Qwen3VLForConditionalGeneration` / `Qwen3VLMoeForConditionalGeneration` on the breakable allowlist |
| `both` (`bcgboth`) | `9bd9e4b5b9` | `64912f662f0a` | local merge of `8560a72b8c` and #44129's head `ae05ff8cd1`; clean, the PRs touch disjoint files |

All local commits reached the node as patches against `80f40522cd`; each tree's root hash was checked after
applying. Models: `Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed8` (dense, DeepStack layers from vision blocks 8,
16, 24) and `Qwen/Qwen3-VL-30B-A3B-Instruct` @ `9c4b90e1e4ba` (MoE). Default server flags (radix cache on,
chunked prefill 8192 on H200) plus the arm's flags; greedy; W1 16 requests one at a time, W2 three waves of 8
released together, 32 output tokens with top-5 log probabilities; W4 speed in three blocks (order reversed in
the second). Workloads, rules and arms as in the plan.

Each arm's backend was read from `/get_server_info`, and the per-batch `cuda graph: True/False` from its
server log: every BCG and FULL arm ran 14 of 17 W1 prefill batches in the graph (21 of 24 at chunk 2048);
the three eager ones are 1-token prefills of radix hits; every eager arm ran 0.

## Pre-registered checks

### C1 — FAIL as written, for every graph arm

| Group | Arm | Backend | Graph batches W1 | W1 pass | W2 pass |
|---|---|---|---|---:|---:|
| dense, TP=1 | `pr/eager2` (noise) | disabled | 0/17 | 16/16 | 19/24 |
| dense, TP=1 | `pr/bcg` | breakable | 14/17 | 0/16 | 0/24 |
| dense, TP=1 | `pr/full` | full | 14/17 | 0/16 | 0/24 |
| dense, TP=1 | `main/default` | disabled | 0/17 | 16/16 | 18/24 |
| dense, TP=1 | `main/full` (finding, not judged) | full | 14/17 | 0/16 | 0/24 |
| dense, TP=1 | `allow/bcg` (control) | breakable | 14/17 | 0/16 | 0/24 |
| dense, chunk 2048 | `pr/bcg-c2048` | breakable | 21/24 | 0/16 | 0/24 |
| MoE, TP=1 | `pr/bcg` | breakable | 14/17 | 0/16 | 0/24 |
| MoE, TP=1 | `pr/full` | full | 14/17 | 0/16 | 0/24 |
| dense, TP=2 | `pr/bcg-tp2` | breakable | 14/17 | 0/16 | 0/24 |

The rule cannot pass for any graph backend: the server returns log probabilities in steps of 0.25 (bf16), the
W1 noise floor of two eager launches is exactly 0, so the tolerance is the 0.1 minimum, and the graph
backends differ from eager by 0.59 already on the text-only requests, where DeepStack is not involved. The
rule was miscalibrated when written. The table below is the reading used instead; it was defined after the
first arms had run and is descriptive, not a pass rule.

### Post-hoc reading

A divergence is a near tie when both sides' top-2 gap at the first differing position is at most 0.25. Max
diff is the largest absolute difference of the log probabilities both sides list, over the first 8 positions
(up to the divergence). W1 and W2 together; 31 visual requests per arm.

| Group | Arm | Text max diff | Visual identical | Visual near-tie | Visual other | Visual max diff |
|---|---|---:|---:|---:|---|---:|
| dense, TP=1 | `pr/eager2` (noise) | 0.375 | 30 | 1 | 0 | 0.253 |
| dense, TP=1 | `pr/bcg` | 0.592 | 24 | 7 | 0 | 0.500 |
| dense, TP=1 | `pr/full` | 0.592 | 23 | 8 | 0 | 0.500 |
| dense, TP=1 | `main/default` | 0.375 | 30 | 1 | 0 | 0.375 |
| dense, TP=1 | **`main/full`** | 0.592 | 2 | 0 | **29** | **9.610** |
| dense, TP=1 | **`allow/bcg`** | 0.592 | 2 | 0 | **29** | **9.610** |
| dense, chunk 2048 | `pr/bcg-c2048` | 0.592 | 24 | 7 | 0 | 0.500 |
| MoE, TP=1 | `pr/bcg` | 1.247 | 14 | 15 | 2 (`w2_a5`@0, `w2_a7`@23) | 1.125 |
| MoE, TP=1 | `pr/full` | 1.000 | 14 | 15 | 2 (`w2_a5`@0, `w2_a7`@23) | 1.375 |
| dense, TP=2 | `pr/bcg-tp2` | 0.499 | 28 | 3 | 0 | 0.500 |

The MoE model is noisier under a graph backend, and not because of DeepStack: its text-only requests diverge
from eager too, some away from a tie (`w2_a6` at position 28, eager's top-2 gap 0.5; `w2_c8` at 25, BCG's gap
0.5), and their largest difference is 1.25. The two visual divergences sit at positions where eager's own gap
was 0.125 (`w2_a5`: "Of" −0.64 / "Based" −0.76 in eager, the order swapped under BCG; `w2_a7`: " graphic" /
" simple"), both in concurrent waves. A second eager launch of the MoE model, added after these arms had
run, is bit-identical to the first one request at a time and token-identical on all 31 visual requests, but in
the concurrent waves its log probabilities already move by up to 1.25 on a visual request, more than BCG's
1.125: this model's numbers depend on what else shares the batch.

### C2 — isolation: the two pairs of the plan pass; the script's third pair does not

| Model | Arms | text 3 = 1 | image 11 = 2 | long video 15 = 9 |
|---|---|---|---|---|
| 8B | all 15 arms | same | same | same, except `pr/bcg`, `pr/full`, `pr/bcg-c2048`, `both/bcg`, `both/bcg-c2048`: different |
| 30B-A3B | `pr/eager`, `pr/eager2` | same | same | different |
| 30B-A3B | `pr/bcg`, `pr/full` | same | different | same |

The plan names two pairs (3 = 1, 11 = 2); `m12_compare.py`, committed with the plan, also checks 15 = 9. The
differences come from one mechanism, and it is not leakage between requests. Requests 11 and 15 repeat
requests 2 and 9 exactly, so the radix cache serves all but the last prompt token, and that one token runs as
a 1-token prefill outside the graph (`#new-token: 1, #cached-token: 9366, cuda graph: False`). Its hidden
state is computed in a different batch shape than in the full prefill, which can flip a tie:

- 8B, `pr/bcg` (likewise `pr/full` and `pr/bcg-c2048`): request 9 departs from eager at position 17, where the
  graph backend has an exact tie (`' —'` and `'—'` both −1.153) and eager a one-step gap (−0.995 / −1.245).
  Request 15 matches eager 32/32. So the KV the graph prefill wrote for the whole video, DeepStack layers
  included, is the KV eager would write; only the last token's tie broke the other way.
- 30B-A3B: the same flip happens in both eager launches (15 vs 9 at position 23, gaps 0.125 / 0.000) and, under
  BCG and FULL, on the image pair (11 vs 2 at position 24, gaps 0.25 / 0.000).

### C3 — positive control: detected

`allow/bcg` diverges from eager on 29 of 31 visual requests (11 of them at the first output token, the latest
at position 27), with differences up to 9.61. It gives the same tokens as `main/full` on 14 of the 16 W1
requests (`pr/bcg` and `pr/full` agree on 15 of 16): two backends missing the same input.

### C4 — pass

`main/default` against `pr/eager`, exact tokens: W1 16/16. W2 23/24 (report only: batch membership depends on
timing).

### P-1 — FAIL as written on the text class

Median of the three per-block medians, TTFT in ms (8 measured requests per class and block):

| Arm | img 1280×720 | img 1920×1080 | two images | video 16.5 s | text | concurrent req/s | concurrent median TTFT |
|---|---:|---:|---:|---:|---:|---:|---:|
| `main/default` (eager) | 77.0 | 139.7 | 65.6 | 1067.2 | 31.8 | 16.25 | 395.6 |
| `allow/bcg` (BCG, no slot) | 77.5 | 146.2 | 55.5 | 1063.5 | 14.3 | 16.06 | 401.3 |
| `pr/bcg` (BCG, slot) | 75.2 | 145.4 | 56.2 | 1065.4 | 15.4 | 16.00 | 402.3 |

P-1 compares `pr/bcg` with `allow/bcg`: every class within the limit except text, 15.4 against 14.3 ms, +1.1 ms
against max(5 %, 1 ms) = 1.0 ms. Per block the text gap is +0.8, +1.6 and +0.3 ms. For a text request the slot
costs one memset of 192 × 12 288 bf16 values (4.7 MB) and three in-graph adds, which is microseconds; the
block with +1.6 ms ran while the other GPU of the node ran a speed block of the same arm. I read it as noise,
but the rule fails as written and was not re-run.

### P-2 — pass

`nvidia-smi` after the server came up: 124 054 MiB (`pr/bcg`) against 123 862 MiB (`allow/bcg`), +192 MiB
(+0.15 %); the same +192 MiB after the speed window (128 002 / 127 810). The server's capture log agrees:
prefill graph 1.54 GB against 1.35 GB (8192 × 4096 × 3 bf16 = 0.19 GiB), and FULL 1.41 GB (`pr/full`) against
1.22 GB (`main/full`). The KV pool is the same (708 335 tokens), because SGLang sizes it before capture;
`m12_compare.py`'s P-2 line uses that pool and so cannot see the slot.

### Speed of the new default (reported, no rule)

`pr/bcg` against `main/default` (eager is `main`'s prefill backend for Qwen3-VL): text −52 % (31.8 → 15.4 ms),
two images −14 %, 1280×720 −2 %, the 16.5 s video unchanged, the 1920×1080 image **+4 % (139.7 → 145.4 ms,
slower)**, the concurrent run −1.5 % requests/s and +1.7 % median TTFT. The 1080p prompt is 2054 tokens and
replays in the 2304-token bucket (12 % padding); 892 tokens (720p) and 509 (two images) fall on or just under a
bucket. Turning BCG on also reserves memory for the prefill graphs: the KV pool shrinks from 712 041 to 708 335
tokens (−0.5 %). These costs are BCG's, not the slot's, and are the trade-off of making BCG the default for
Qwen3-VL.

## Q5 — both PRs together

Tree `both`: the PR head with the test fix, merged with #44129's head `ae05ff8cd1`. At chunk size 2048 the long
video is five chunks, so #44129 hands the rows it kept from the first chunk's encode to four later replays.

| Pair | W1 tokens identical | W1 largest difference, all 32 positions | Visual identical / near tie / other (W1 + W2) | Visual max diff |
|---|---:|---:|---|---:|
| `both/eager` against `pr/eager` | 16/16 | 0.0 | 31 / 0 / 0 | 0.375 |
| `both/bcg` against `pr/bcg` | 16/16 | 0.0 | 31 / 0 / 0 | 0.000 |
| `both/eager-c2048` against `pr/eager-c2048` | 16/16 | 0.0 | 30 / 1 / 0 | 0.375 |
| `both/bcg-c2048` against `pr/bcg-c2048` | 16/16 | 0.0 | 30 / 1 / 0 | 0.426 |
| `both/bcg` against `both/eager` | 14/16 | | 24 / 7 / 0 | 0.500 |
| `both/bcg-c2048` against `both/eager-c2048` | 14/16 | | 26 / 5 / 0 | 0.500 |

One request at a time, adding #44129 changes no log probability at any position, eager or BCG, at either chunk
size. The W2 differences are the batch-membership noise two eager launches also show (0.375). BCG against
eager with both PRs reads as with #33726 alone, including the same tie at position 17 of the long video
(request 15 = 9 differs in `both/bcg` and `both/bcg-c2048` too). No interaction between the two PRs.

## TP=2

`pr/bcg-tp2` against `pr/eager-tp2`, both across the two GPUs; the backend resolved to breakable and ran 14 of
17 W1 prefill batches in the graph. 28 of 31 visual requests identical, 3 near ties (in W1 the 448×448 image at
position 22, gaps 0.25 / 0.00, and the two-image request at 9, gaps 0.00 / 0.25), no other divergence; largest
difference 0.50 on visual and on text requests. All three C2 pairs hold. TP=2 eager itself differs from TP=1
eager on 4 of the 16 W1 requests, as a different reduction order should.

## Findings

1. **FULL prefill drops DeepStack on `main` (confirmed on GPU).** See `main/full` above. The flag is opt-in and
   marked experimental, and FULL's incompatibility list is empty, so every multimodal model may select it.
   For a DeepStack model the request's DeepStack input never reaches the replay, the answer is served
   anyway, and nothing fails or logs. The PR binds its slot for FULL as well, which fixes the Qwen3-VL
   family on both backends.
2. **CI at `ccfda8dfd9`.** `base-a-test-cpu`: the PR's test, fixed by `8560a72b8c` (node: `bcghead` stops with
   `AttributeError: 'PrefillCudaGraphRunner' object has no attribute 'is_dsv41_model'`, `bcgfix` 12/12 OK).
   `extra-a-test-1-gpu-small`: `test_vlm_serving_flashinfer`, Qwen2.5-VL-3B median TTFT 85.1 / 87.1 ms against
   83 ms; that model is off the allowlist and has no DeepStack, so the PR's resolver returns 0 for it.
3. **A test-order failure in `test_server_args.py`.**
   `TestMultimodalFeatureTransport::test_default_transport_is_cpu_for_unsupported_multinode_model` fails when
   `unit/model_executor` and `unit/server_args` run in one pytest process, on `main` (800 passed, 1 failed)
   and on the PR (812 passed, 1 failed) alike; the class passes alone (16/16). CI runs one file per process,
   so CI does not see it. Not related to the PR.

## Node CPU checks

| Check | Tree | Result |
|---|---|---|
| `test_deepstack_replay_slot.py`, one process, `-f` | `bcghead` (`ccfda8dfd9`) | `AttributeError … is_dsv41_model`, as on CI |
| same | `bcgfix` (`8560a72b8c`) | 12 tests OK |
| the PR's three test files, then `pre-commit run --files` on its 8 files | `bcgfix` | ok |
| `unit/model_executor` + `unit/server_args`, one pytest process | `base` / `bcgfix` | 800 / 812 passed, the same 1 failure (finding 3) |

## GPU use

Assigned 17:41 UTC, released 18:31 UTC: 50 minutes on 2 GPUs, about 1.7 GPU-hours of the 10 allowed (the
assignment was booked for 3 hours and given back early). `release_check` found nothing left on the node; the
lab unit run's evidence archive is a release asset of the private repository.

## Files

- Scripts: `context/node-artifacts/2026-10-17_pr33726-m12/scripts/` in the SGL workspace (`m12_arm.py`,
  `m12_run_arm.sh`, `m12_compare.py`, `m12_q5.py`, `m12_fixcheck.sh`).
- Client JSON, scrubbed server logs and the node logs: `context/node-artifacts/2026-10-17_pr33726-m12/m12_out.tar.gz`.
- `compare.md` and `q5.md` here: the two scripts' full output on the final data.
