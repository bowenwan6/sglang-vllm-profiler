# Debug Plan — `No data iterator found for token: <|video_pad|>`

> **Status:** PENDING — no experiments run yet. All stages require explicit approval
> before execution.
>
> **Prerequisite reading:** `audit_notes.md` (same directory).
>
> **Goal:** Confirm the root cause, verify the workaround, and produce evidence for
> an upstream SGLang issue/PR.
>
> **Constraints:**
> - Do **not** modify SGLang source.
> - Do **not** run formal IMG-A/B/C benchmarks until this blocker is resolved.
> - All debug artifacts under `experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/`
>   and logs under `logs/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/`.
> - Do not overwrite existing smoke or IMG-A results.
> - No KAPI logging, no profiler.
> - Raw/debug logs not committed unless explicitly approved.

---

## Hypothesis (from audit)

`gen_mm_prompt` in `sglang/benchmark/datasets/common.py` includes `video_pad_id`
(151656) in the random token pool. Approximately 0.084% of 128-token prompts will
contain `<|video_pad|>`. Because the dataset is **non-deterministic** between
subprocess invocations (same seed ≠ same prompts), each rep independently draws
from this distribution. With 430 prompts/rep, E[failures/rep] ≈ 0.36, and over 5
reps P(at least one failure) ≈ 83%.

---

## Stage D0 — Payload audit (confirm root cause)

**Goal:** Verify that the failing requests at indices 44 and 185 contain
`<|video_pad|>` in the text sent to the server. Confirm non-determinism is the
dataset-level cause.

**What to build:** A small `debug_payload_audit.py` that:
1. Imports `sglang.benchmark.datasets.image` and calls `sample_image_requests`
   with `--seed 1` and `num_requests=430`
2. Inspects `DatasetRow.prompt` (the raw text sent by `sglang-oai-chat`) at indices
   30+44=74 and 30+185=215 (warmup=30)
3. Prints: does the prompt contain `<|video_pad|>`? Which token IDs are in the
   decoded text?
4. Runs the same generation 20 times (new seed each time: 1–20) and reports
   how many prompts across all runs contain `<|video_pad|>` and at which indices

**No server needed.** Pure Python; runs offline.

**Expected outcome if hypothesis is correct:**
- Some (not all) runs produce 1–2 prompts containing `<|video_pad|>`
- ~83% of 20 runs will have at least 1 hit (expected ~7 clean runs)
- The hit indices will vary (since dataset is non-deterministic)
- A direct confirm: `dataset[74].prompt` contains the literal `<|video_pad|>` string

**Decision gate:**
- If confirmed → proceed to D1 (workaround search)
- If NOT confirmed → audit_notes root cause is wrong; escalate with server-log trace

**Sub-question D0.1 (low priority):** Why is the dataset non-deterministic despite
fixed seed? Compare Python `random.getstate()` at the start vs inside
`sample_image_requests`. If the processor or tokenizer modifies Python random state,
that explains it. This doesn't change the fix but informs whether "seed 2 will work"
thinking is valid.

---

## Stage D1 — Same-run reproducibility

**Goal:** Confirm the failure recurs in the same run configuration (S0_ipc,
seed=1) when executed again from scratch.

**What to run:** A single-variant mini-runner that:
- Starts SGLang with `SGLANG_USE_CUDA_IPC_TRANSPORT=1`
- Runs **3 reps** of `num_prompts=400 warmup=30 seed=1`
- Stops on first failure (existing runner behavior)
- Records which rep and which measured-window indices fail

**Expected outcome:** At least 1 rep in 3 will have ≥1 failure (P ≈ 66%). The
failure will be a video_pad 400 error.

**Decision gate:**
- If rep3 fails again at ~same indices → direct confirm of stochastic reproducibility
- If no failure in 3 reps → sampling variance; run 5 reps to gather statistics
- If failure occurs in a different kind of error → new issue unrelated to video_pad

---

## Stage D2 — Fresh server per rep

**Goal:** Determine whether restarting SGLang between every rep eliminates the
failures.

**Rationale:** If failures are server-state-dependent (RadixCache, IPC pool
exhaustion, etc.), a fresh server each rep would prevent them. If failures are
prompt-content-dependent (video_pad in text), a fresh server will NOT help.

**What to run:** Modified runner that:
- Starts SGLang, runs **1 rep**, kills server, repeats for 3 reps
- Same seed, same config as D1

**Expected outcome (if audit is correct):** Failures still occur (just in different
reps now, since each rep gets a fresh dataset draw from non-deterministic generation).
Approximately 0.36 expected failures per rep regardless of server freshness.

**Decision gate:**
- If fresh-server-per-rep eliminates ALL failures → server state was the cause,
  not the prompt content → audit hypothesis was wrong; re-investigate
- If failures persist (different reps) → confirms prompt-content root cause

---

## Stage D3 — `/flush_cache` between reps

**Goal:** Control for RadixCache specifically, without full server restart overhead.

**What to run:** Runner that:
- Keeps same SGLang server alive for all 3 reps
- Between each rep, calls `POST http://127.0.0.1:30000/flush_cache`
- Verifies 200 OK response before each rep

**Expected outcome:** Failures persist (if audit is correct). RadixCache flush does
not change the prompt content.

**Decision gate:** Same as D2. If failures disappear: RadixCache was involved
(unlikely per audit). If failures persist: confirms the prompt root cause.

**Note:** D2 and D3 can be run in sequence in the same mini-run session.

---

## Stage D4 — No-IPC control

**Goal:** Determine whether `SGLANG_USE_CUDA_IPC_TRANSPORT=1` is involved.

**What to run:** Repeat D1 (3 reps, same seed) with `SGLANG_USE_CUDA_IPC_TRANSPORT`
**unset** (IPC off, `S0_noipc` config).

**Expected outcome (if audit is correct):** Failures persist at the same rate
(~0.36/rep). The IPC flag changes image tensor transport but not the text prompt
content. Failures may appear in different reps (due to non-determinism) but at
the same average rate.

**Decision gate:**
- If no-IPC has the same failure rate → IPC is not causally involved; issue is in
  `gen_mm_prompt` alone
- If no-IPC has zero failures → IPC transport changes the code path through
  `load_mm_data` in a way that either avoids or triggers the error differently;
  re-investigate with server-side logging

---

## Stage D5 — Radix cache disable

**Goal:** Isolate RadixCache as a contributing factor.

**What to run:** Repeat D1 with `--disable-radix-cache` added to the SGLang
server command.

**Expected outcome:** Same failure rate as D1. Video_pad errors are not related
to the cache.

**Decision gate:**
- Zero failures with `--disable-radix-cache` → surprising; RadixCache was
  somehow introducing the video_pad token (would require deep investigation)
- Same failure rate → RadixCache not involved

---

## Stage D6 — vLLM control

**Goal:** Confirm the failure is SGLang-specific (not dataset-level), and verify
that the vLLM anchor survives the same prompt set.

**What to run:** Run the same image workload (seed=1, 400 prompts, 30 warmup, 3
reps) against **vLLM** via `sglang-oai-chat` at port 30001.

**Expected outcome:** vLLM should NOT produce the same error, because vLLM's
chat endpoint (`/v1/chat/completions`) handles the user message text differently:
it does not run SGLang's multimodal preprocessor. If the text contains `<|video_pad|>`,
vLLM would treat it as ordinary text (a non-special token string) and complete the
request normally.

This confirms that the error is in SGLang's multimodal preprocessing, not in the
dataset.

**Decision gate:**
- vLLM completes 3 reps with 0 failures → SGLang-specific issue; confirms
  the upstream fix target is `gen_mm_prompt` (benchmark) or SGLang's multimodal
  preprocessor (server)
- vLLM also fails → the request itself is malformed; investigate request payload

---

## Stage D7 — Minimal repro package

**Goal:** Produce a self-contained repro that can accompany a SGLang GitHub issue
and/or PR.

**Components:**

### D7a. Direct server repro (no bench_serving)
Send a single crafted request that reliably triggers the error:
```bash
# server: start SGLang with Qwen3-VL-8B-Instruct (any GPU)
# Then send:
curl -s -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_path>",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<tiny_base64_image>"}},
        {"type": "text", "text": "<|video_pad|>describe this image"}
      ]
    }],
    "max_tokens": 16,
    "temperature": 0
  }'
# Expected: HTTP 400 {"message": "No data iterator found for token: <|video_pad|>"}
```

If this always fails: it's a **server-side fix target** — the server should gracefully
handle user-supplied `<|video_pad|>` text (return error with clearer message, or
skip the video token if no video data is provided).

If this does NOT fail: the issue is specifically in how the tokenized prompt_ids
are decoded and split, and only appears after many requests (RepN > 1).

### D7b. Benchmark repro
A minimal bench_serving invocation that reproduces the failure within 3 runs:
```bash
# Run bench_serving 10 times with different seeds; ~83% should fail
for seed in 1 2 3 4 5 6 7 8 9 10; do
  python3 -m sglang.bench_serving \
    --backend sglang-oai-chat \
    --base-url http://127.0.0.1:30000 \
    --dataset-name image \
    --image-count 1 --image-resolution 720p \
    --image-format png --image-content random \
    --random-input-len 128 --random-output-len 32 \
    --random-range-ratio 1.0 \
    --max-concurrency 1 --num-prompts 200 \
    --seed $seed --warmup-requests 5 \
    --extra-request-body '{"temperature": 0, "top_p": 1}' \
    --output-details --output-file /tmp/repro_seed${seed}.jsonl
  python3 -c "
import json
d = json.loads(open('/tmp/repro_seed${seed}.jsonl').read().strip().split('\n')[-1])
errors = [e for e in d.get('errors',[]) if e]
print(f'seed={seed}: completed={d[\"completed\"]}, fail={len(errors)}')
"
done
```

### D7c. Evidence threshold for SGLang issue

Sufficient to open an issue:
- [ ] D7a direct request confirms HTTP 400 with `<|video_pad|>` text in user message
- [ ] D6 confirms vLLM does NOT fail on the same workload
- [ ] `gen_mm_prompt` in `common.py` confirmed not excluding `video_pad_id`

Sufficient to propose a PR:
- [ ] Above evidence collected
- [ ] PR text: "Add `video_pad_id` (and other multimodal special tokens) to the
  exclusion list in `gen_mm_prompt` so benchmark prompts never contain tokens that
  trigger multimodal iterator lookup"
- [ ] Optional: also fix the server to gracefully handle `<|video_pad|>` in user
  text for image-only requests (return HTTP 400 with a clear explanatory message
  rather than a crash-style `ValueError`)

---

## Execution order and approval gates

```
D0 (payload audit, no server)
  ↓ if confirmed
D1 (same-run S0_ipc 3×reps)
  ↓ simultaneously or in sequence
D2 (fresh-server-per-rep) + D3 (flush_cache) + D4 (no-IPC) + D5 (disable-radix)
  ↓
D6 (vLLM control)
  ↓
D7 (minimal repro + issue draft)
```

D0 can run any time (no server, no GPU). D1-D6 each require GPU 7 to be idle and
explicit approval. D7 synthesis requires no GPU if D7a is treated as a bonus.

**Minimum path to resume formal IMG-A:**
- D0 confirms root cause → no further experiments needed
- Implement workaround (pre-flight batch check; see below)
- Get explicit approval from user to resume IMG-A with the workaround in place

---

## Workaround for formal benchmarks (without SGLang source modification)

Once D0 confirms the hypothesis, the `run_image_text_imgA.py` runner can be augmented
with a **pre-flight dataset check**:

```python
def check_dataset_for_video_pad(seed, n_requests=430, input_len=128, resolution='720p'):
    """Return True if no prompt in the generated dataset contains <|video_pad|>."""
    import sys
    sys.path.insert(0, '/sgl-workspace/sglang/python')
    import random, numpy as np
    from sglang.benchmark.utils import get_processor
    from sglang.benchmark.datasets.image import sample_image_requests
    random.seed(seed); np.random.seed(seed)
    proc = get_processor(SNAPSHOT)
    dataset = sample_image_requests(
        num_requests=n_requests, image_count=1, input_len=input_len, output_len=32,
        range_ratio=1.0, processor=proc, image_content='random', image_format='png',
        image_resolution=resolution, backend='sglang-oai-chat',
    )
    return all('<|video_pad|>' not in row.prompt for row in dataset)
```

Before launching the server, the runner iterates `seed in [1, 2, 3, ...]` until
`check_dataset_for_video_pad(seed)` returns True. The first clean seed is then used
for all reps of that variant. This does NOT require modifying SGLang source.

**Limitation:** Since dataset generation is non-deterministic between subprocess
calls, a seed that passes the pre-flight check may still produce a video_pad-
containing prompt in the actual bench_serving subprocess. The pre-flight check
reduces (but cannot eliminate) the risk. It's a best-effort workaround until the
upstream fix lands.

**Stronger workaround:** Run all bench_serving subprocess calls with the same
`PYTHONHASHSEED` environment variable set, which may reduce (but not eliminate)
non-determinism in tokenizer vocab ordering.

---

## Artifact locations

| artifact | path |
|---|---|
| Debug scripts | `debug_video_pad/debug_payload_audit.py` (D0) |
| Mini-runner | `debug_video_pad/debug_runner.py` (D1–D6) |
| Results JSON | `debug_video_pad/results/D<N>_results.json` |
| Logs | `logs/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/` |
| Issue draft | `debug_video_pad/upstream_issue_draft.md` (D7) |

Raw logs and results JSON are **not committed unless explicitly approved**.
Only committed: this plan, `audit_notes.md`, summary MDs after each stage.
