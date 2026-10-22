# M12 — revalidation of PR #33726 at head `ccfda8dfd9` (plan, 2026-10-17)

Written before any GPU run. The pass rules below do not change after the data arrive.

## Why now

The last GPU validation (10-13, at `79163e9582`) predates three merges of `main` into the branch
(`6fcaecd32e`, `b58855ba98`, and JustinTong0323's `ccfda8dfd9` on 10-17). The merged-in `main`
(`80f40522cd`) carries seven upstream commits that edit the same breakable replay path the slot
lives in: #44131 (DSv4.1 BCG image handling), #43736 / #44326 / #44850 (decoder SWA bounded replay
inside BCG), #44022 (CP turns BCG on by default), #43772 (Full prefill capture batches) and #43642
(DSA prefill under BCG on HIP). Nothing of the PR's own diff changed (+447/−14 in 8 files).

CI at `ccfda8dfd9` (10-17):

- `base-a-test-cpu (0)`: **the PR's own test file fails**. `TestDeepStackReplaySlotWiring` builds
  the runner with `__new__`; `replay_layer_forward` now reads `self.is_dsv41_model` (#44131), so the
  wiring tests raise `AttributeError` before they reach the slot. Fix: local commit `8560a72b8c`
  (`runner.is_dsv41_model = False`, as upstream's own runner helper tests do). Not pushed.
- `extra-a-test-1-gpu-small (0)` (root cause of every red `extra-*` job by fail-fast):
  `test_vlm_serving_flashinfer`, online median TTFT 85.1 and 87.1 ms against a ceiling of 83 ms, on
  `Qwen/Qwen2.5-VL-3B-Instruct`. That model is not on the breakable allowlist and has no DeepStack,
  so the PR's code is not on its path (the width resolver returns 0, no slot is registered, its
  prefill graph stays disabled). Not attributed to the PR.
- AMD, NPU and Xeon lanes: not read (the change is CUDA-only).

Gaps of every earlier GPU check (M2–M11): radix cache always off; one request at a time; TP=1
only; the FULL backend (the other backend that binds the slot) never run; images no larger than
448 px; no multi-image request; no video; no chunk boundary inside a visual item; Gates P-1 and P-2
of `plan.md` §5 never measured; and BCG against eager speed never measured, although the PR now
makes BCG the default prefill backend for Qwen3-VL.

## Static audit (done on the Mac, 10-17)

- Breakable allowlist on `main`: 12 archs. The DeepStack-capable ones (`Qwen3_5`, `Qwen3_5Moe`,
  `Qwen4Exp`, `InternS2Mobius`, all subclasses of `Qwen3VLForConditionalGeneration`) have zero
  DeepStack levels on their public checkpoints: `Qwen/Qwen3.5-27B` `deepstack_visual_indexes: []`,
  `internlm/Intern-S2-Mobius` and `-Preview` `[]`, Qwen4-Exp "ships an empty list" (code comment).
  So `main` serves no DeepStack model under BCG today; Qwen3-VL / Qwen3-VL-MoE (the PR's two new
  entries) are the first.
- `tc_piecewise`: Qwen3-VL is not on the multimodal piecewise allowlist (3 archs); the resolver
  returns 0 there by design. Not tested.
- **FULL** (`--cuda-graph-backend-prefill full`): "experimental", and its incompatibility rule list
  is empty, so any multimodal model may use it on `main`. On `main` no DeepStack slot exists, so a
  Qwen3-VL FULL replay may reuse the DeepStack tensor of the capture — the bug class this PR fixes.
  The PR binds the slot for FULL too (it is an eager-tail backend). Checked on the GPU below.

## Questions

- **Q1** Does the PR's default (BCG) give the same outputs as eager prefill at default server flags
  (radix cache on, default chunk size) for images of several sizes, multi-image requests, a video
  that spans two chunks, text, radix prefix hits, one request at a time and in mixed concurrent
  batches; dense and MoE; TP=1 and TP=2?
- **Q2** Does the PR's FULL backend give eager's outputs? Does `main`'s FULL backend (the upstream
  risk above)?
- **Q3** Is the fix still needed on today's `main`? Control tree `allow` = `main` plus the two
  allowlist lines, no slot: under BCG it must corrupt image outputs, or the harness cannot see the
  bug class.
- **Q4** Cost: Gate P-1 (TTFT of BCG with the slot vs BCG without it), Gate P-2 (peak GPU memory),
  and the speed of the new default (PR BCG vs `main`'s default, which is eager for Qwen3-VL).
- **Q5** (optional, if time is left) The tree with both of Bowen's open PRs (#33726 and #44129):
  long video under BCG vs eager.

## Trees (node base = upstream `80f40522cd`, tree `4303668ed3c9`)

| Tree | Commit | What |
|---|---|---|
| `main` | `80f40522cd` | the `main` merged into the PR head |
| `pr` | `8560a72b8c` | PR head `ccfda8dfd9` + the test fix (the source code is the PR head's) |
| `allow` | `b0c8105c75` | `main` + `Qwen3VLForConditionalGeneration` / `Qwen3VLMoeForConditionalGeneration` on the breakable allowlist, nothing else |

Local commits reach the node as patches (`ensure_fork_tree`); nothing is pushed.

## Models

`Qwen/Qwen3-VL-8B-Instruct` @ `0c351dd01ed8` (DeepStack layers 8, 16, 24) and
`Qwen/Qwen3-VL-30B-A3B-Instruct` @ `9c4b90e1e4ba` (MoE).

## Arms

Default server flags plus the listed ones; greedy decoding. Each arm's resolved prefill backend is
read from `/get_server_info`, and the server log's per-batch `cuda graph: True/False` shows whether
the graph ran.

| Group | Arms |
|---|---|
| dense, TP=1 | `pr/eager` (`--disable-prefill-cuda-graph`), `pr/eager2` (second launch: noise floor), `pr/bcg` (default), `pr/full` (`--cuda-graph-backend-prefill full`), `main/default`, `main/full`, `allow/bcg` (default), `pr/eager-c2048` and `pr/bcg-c2048` (`--chunked-prefill-size 2048`) |
| MoE, TP=1 | `pr/eager`, `pr/bcg`, `pr/full` |
| dense, TP=2 | `pr/eager-tp2`, `pr/bcg-tp2` |

## Workloads

- **W1, one request at a time** (16 requests, 32 output tokens, top-5 log probabilities of every
  position): text; 1280×720 image; the same text again; 448×448 image; 1920×1080 image; two images;
  three images; 8.5 s video; 16.5 s video (two chunks at the default size); the 1280×720 image with
  another question (radix hit on the image); the 1280×720 request again (radix hit to the last
  token); another text; a small image followed by a long text; a long text followed by an image;
  the 16.5 s video again; a 2560×1440 image. Images are structured (colour stripes, bands, shapes),
  generated in-process, byte-identical across arms.
- **W2, concurrent** (3 waves of 8 requests released together): the W1 kinds mixed in each wave,
  so prefill batches hold image, video and text rows together.
- **W4, speed** (dense TP=1 only; arms `main/default`, `allow/bcg`, `pr/bcg`; 3 blocks, order
  reversed in the second): single requests, 2 warm-ups then 8 distinct inputs per class (1280×720
  image, 1920×1080 image, two images, 16.5 s video, text), TTFT from the stream, 16 output tokens;
  then 48 requests 8 at a time (requests per second, median TTFT); peak allocated memory from the
  server's own report and one `nvidia-smi` reading after the window.

## Checks and pass rules

- **C1 (Q1, Q2)**: every W1 request of a graph arm against `pr/eager` of the same model, TP and chunk
  size: the 32 generated tokens identical, and the largest absolute difference of aligned top-5 log
  probabilities over the first 8 positions at most max(0.1, 2 × the `pr/eager2` vs `pr/eager`
  noise floor). One request failing fails the arm. W2: the same rule, except that a divergence is
  accepted at a position where the reference's top-2 log-probability gap is below 0.1 (a near tie
  that batch composition can flip); any other divergence fails the arm.
- **C2 (isolation, inside every arm)**: W1 request 3 equals request 1 (text after an image), and
  request 11 equals request 2 (the same image again after other inputs).
- **C3 (positive control)**: `allow/bcg` must fail C1 on at least one image or video request. If it
  passes, the harness does not see the bug class, every C1 pass above carries no weight, and the
  run is reported as inconclusive (GPU work stops there).
- **C4**: `main/default` equals `pr/eager` token for token on every W1 request (the PR does not
  touch the eager path). W2 is reported only: which requests share a prefill batch depends on
  timing, so two eager launches need not agree there.
- `main/full`: reported as a finding, pass or fail; no rule on the PR.
- **P-1**: `pr/bcg` against `allow/bcg`, median TTFT per W4 class: worse by at most 5 % or 1 ms,
  whichever is larger (`plan.md` §5 Gate P-1).
- **P-2**: peak allocated GPU memory of `pr/bcg` within 5 % of `allow/bcg` (Gate P-2).
- **Speed of the new default**: `pr/bcg` against `main/default` per W4 class and for the
  concurrent run; reported, no pass rule (the default is the maintainers' call), but a slower
  class is called out in the PR text.
- **Node CPU checks**: the PR's four unit files, one per process, on `pr` (all pass) and the
  wiring tests on `ccfda8dfd9` without the fix (the CI error reproduces); the
  `test/registered/unit/model_executor` sweep on `pr`; pinned lint (ruff 0.15.1, isort 7.0.0,
  codespell 2.4.1) on the PR's files.

## Budget and order

Bowen's cap (10-17): 10 GPU-hours, at most 2 GPUs. One assignment of 2 GPUs for 3 hours (6 GPU-hours
booked), released as soon as the results are pulled. Slot 0: CPU checks, then the dense TP=1 arms,
then W4. Slot 1: the MoE arms, then idle until TP=2 takes both slots at the end. Q5 only if more
than 30 minutes remain. Estimated use: about 4.5 GPU-hours.

Stop rules: C3 fails (no positive control) → stop and report; a server of an arm fails to start
twice → that arm is reported as not run, the queue goes on.

## Outputs

Scripts, client JSON, scrubbed server logs and a summary under
`context/node-artifacts/2026-10-17_<node>/` of the SGL workspace; a report in
`results/m12_revalidation_<stamp>/report.md` here; the state in `context/pr-33726.md`.

## Not in scope

Pushing anything, editing the PR description or replying on the PR (Bowen does that);
`tc_piecewise`; models other than the two above.

## Amendment, 10-17 18:20 UTC, written before any Q5 data: Q5

Tree `both` = `9bd9e4b5b9` (node tree name `bcgboth`), a local merge of `8560a72b8c` with
#44129's head `ae05ff8cd1` (JustinTong0323's last push), root tree `64912f662f0a`. The merge is
clean: the two PRs touch disjoint files. Arms on the 8B model, TP=1, W1 and W2: `both/eager`,
`both/bcg`, `both/eager-c2048`, `both/bcg-c2048`. At chunk size 2048 the 16.5 s video spans five
chunks, so the rows #44129 keeps feed the DeepStack slot in four later replays of each such request.

Reading (descriptive; the C1 tolerance is below one bf16 step of the returned log probabilities):

- `both/bcg` against `both/eager`, at each chunk size, in the columns of the post-hoc table. A
  divergence on a visual request that is not a near tie (either top-2 gap above 0.25), or a visual
  max diff above 1.0 (twice the 0.5 of `pr/bcg` against `pr/eager`), is a finding.
- Each `both/X` against `pr/X` (does #44129 change the outputs?): the same columns, read against
  the launch-to-launch noise of `pr/eager2` against `pr/eager`.
