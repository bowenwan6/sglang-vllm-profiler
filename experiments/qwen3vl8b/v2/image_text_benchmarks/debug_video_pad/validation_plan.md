# Validation plan — `<|video_pad|>` benchmark-generator blocker

> **Status: V1/V2 PASS; V3 pending. Do NOT modify `/sgl-workspace/sglang` yet.**
> The generator bug has been validated inside the profiler repo at the payload and
> serving-symptom levels. Branch SGLang and prepare a PR only after V1–V3 pass.
>
> Companion docs: [`audit_notes.md`](audit_notes.md) (root-cause audit),
> [`debug_plan.md`](debug_plan.md) (staged D0–D7), [`workaround_design.md`](workaround_design.md)
> (sanitized monkeypatch design).

---

## 1. Goal

Validate, **inside `/data/sglang-vllm-profiler` only**, the following chain before
touching upstream SGLang:

1. The stock SGLang image benchmark generator (`gen_mm_prompt`) **can emit**
   `<|video_pad|>` (and other multimodal/control special tokens) in random prompt text.
2. Such text **triggers the observed HTTP 400** `No data iterator found for token:
   <|video_pad|>` for an **image-only** request to Qwen3-VL.
3. A **sanitized** generator / prompt path (exclude all special ids) **eliminates** the
   issue (0 forbidden tokens, 0 serving failures).
4. **Only then** branch SGLang, patch `gen_mm_prompt`, add tests, and open a PR.

Non-goal: performance conclusions. This is a correctness-bug validation, kept strictly
separate from the IPC/PCG benchmark questions.

---

## 2. Validation stages

### V0 — Static source confirmation (NO GPU, read-only)

Confirm three facts by reading SGLang source (read-only; no edits):

- `gen_mm_prompt` in `/sgl-workspace/sglang/python/sglang/benchmark/datasets/common.py`
  removes **only** `image_pad_id` from the random token pool
  (`if image_pad_id: all_available_tokens.remove(image_pad_id)`), nothing else.
- The image dataset
  (`/sgl-workspace/sglang/python/sglang/benchmark/datasets/image.py`) passes
  **only** `processor.image_token_id` into `gen_mm_prompt` (no video/vision exclusion).
- The error originates at
  `/sgl-workspace/sglang/python/sglang/srt/multimodal/processors/base_processor.py:610`
  (`raise ValueError(f"No data iterator found for token: {text_part}")`), reached
  when a prompt contains a video special token but the request carries no video data.

**Output:** record findings as a short section appended to `audit_notes.md` (or note
"already documented" if the audit already covers all three — it does as of this writing).
No new artifact strictly required; V0 is a confirmation gate, not a run.

### V1 — CPU-only generator audit (NO GPU)

Run the existing `debug_payload_audit.py` (re-verify it `py_compile`s first). It
replicates `gen_mm_prompt` over many seeds/prompts and compares against a sanitized
generator that excludes **all** `tokenizer.all_special_ids` + `get_added_vocab()` ids.

- Expected: the **stock** replica produces a nonzero count of prompts containing
  `<|video_pad|>` and/or other forbidden multimodal tokens over a large sample
  (rate ≈ 0.08%/prompt → use ≥ ~20k prompts so hits are statistically visible); the
  **sanitized** generator produces **zero** forbidden tokens over the same sample.
- **Required outputs (committed):**
  - `debug_video_pad/results/D0_payload_audit.md`
  - `debug_video_pad/results/D0_payload_audit.json`

> Note on sample size: at ~0.08%/prompt, a 12–50 seed × 430-prompt sweep
> (≈ 5k–21k prompts) is sufficient to observe hits. If a small sweep shows zero hits,
> **increase `--seeds`** rather than concluding "no bug" — the forbidden-token inventory
> in the JSON (which lists `video_pad_id` as in-pool) is the deterministic backstop.

#### V1 result — ✅ PASS (2026-05-31, 30 seeds × 430 = 12,900 prompts, CPU only)

| metric | value | gate | verdict |
|---|---|---|---|
| `image_pad_id` | 151655 (excluded by stock gen) | — | — |
| `video_pad_id` | 151656 (**in** stock pool) | — | — |
| stock `<\|video_pad\|>` hits | **8 / 12,900** (0.062%) | ≥ 1 | ✅ |
| stock any-forbidden hits | **42 / 12,900** (0.326%) | ≥ 1 | ✅ |
| sanitized any-forbidden hits | **0 / 12,900** | = 0 | ✅ |

Root cause confirmed at the **generator** level: `gen_mm_prompt` leaves `video_pad_id`
(and other multimodal/control ids) in the random pool; excluding all special ids
eliminates every forbidden token. Artifacts: `results/D0_payload_audit.{md,json}`
(run config `--seeds 30 --prompts-per-seed 430 --input-len 128`).

### V2 — Tiny serving repro (GPU 7 needed)

Prove the failure path is **real at serving time**, not only theoretical, with a
deliberate single-request probe. No benchmark, no perf numbers.

1. Preflight: confirm GPU 7 idle (< 2000 MiB) and no KAPI env; else stop.
2. Launch one SGLang server on GPU 7 (clean: no KAPI, no profiler).
3. **Failing probe:** POST one image-only request to `/v1/chat/completions` whose text
   content deliberately includes the literal `<|video_pad|>`.
   - Expected: HTTP 400 with `No data iterator found for token: <|video_pad|>`.
4. **Control probe:** POST the same request shape with **safe** text (no special token).
   - Expected: HTTP 200, non-empty greedy output.
5. Kill server; confirm GPU 7 returns below idle threshold.

**Output (committed):** `debug_video_pad/results/V2_serving_repro.md` (the two probes,
exact status codes, error string, sanitized request shape). Raw server log stays under
`logs/.../debug_video_pad/` and is **not** committed unless approved.

#### V2 result — PASS (2026-05-31, GPU 7, clean serving probe)

| probe | text | status | expectation | verdict |
|---|---|---:|---|---|
| failing | `<\|video_pad\|>describe the image` | 400 | 400 + `No data iterator found for token: <\|video_pad\|>` | PASS |
| control | `describe the image` | 200 | 200 + non-empty output | PASS |

V2 confirms the serving symptom is real: an image-only request whose text contains
`<|video_pad|>` returns HTTP 400, while the same image request with safe text succeeds.
This does **not** make SGLang serving the primary bug; the server is rejecting a video
placeholder without video payload. Together with V1, the fix target remains the benchmark
generator (`gen_mm_prompt`). Artifact: `results/V2_serving_repro.{md,json}`.

### V3 — Sanitized smoke (GPU 7 needed)

Run `run_image_text_smoke_sanitized.py` (drives `bench_serving` through the
`bench_serving_sanitized.py` monkeypatch wrapper; writes to `smoke_sanitized/`, does
**not** touch the original `smoke/` artifacts).

Validate all three smoke paths:

- SGLang **IPC on** (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`): 0 failures
- SGLang **no-IPC**: 0 failures
- vLLM anchor (via `sglang-oai-chat`): 0 failures
- Pre-flight **special-token audit** in the runner reports 0 forbidden tokens in the
  generated prompt set.

**Output (committed):** `smoke_sanitized/smoke_summary.md` +
`smoke_sanitized/smoke_results.json`. Raw `*_bench.jsonl` not committed unless approved.

### V4 — Formal IMG-A rerun gate

Only if **V1, V2, V3 all pass**. Resume the formal sanitized IMG-A via
`run_image_text_imgA_sanitized.py` (writes `results/imgA_sanitized_*`, separate from the
invalidated original `imgA_*`). **Do not run IMG-B / IMG-C** until sanitized IMG-A passes
with 0 failures and acceptable bracket drift. (Formal IMG-A is governed by the existing
protocol §6/§8; this plan only gates *entry* to it.)

---

## 3. Acceptance criteria (exact pass/fail)

| Stage | PASS if … | FAIL → action |
|---|---|---|
| **V0** | all three source facts confirmed (only `image_pad_id` excluded; image dataset passes only `image_token_id`; error at `base_processor.py:610`) | revise hypothesis; do **not** branch SGLang |
| **V1** | stock generator shows **≥ 1** forbidden-token hit over the sample **and** sanitized generator shows **0** hits | if stock shows 0: enlarge `--seeds` and retry; if sanitized shows > 0: the fix rule is wrong → stop, fix rule |
| **V2** | deliberate `<\|video_pad\|>` request reproduces the **exact** 400 error **and** the safe-text control returns 200 / non-empty | if no repro: serving path differs from hypothesis → stop, re-investigate, do **not** branch |
| **V3** | **all** smoke variants 0 failures **and** runner special-token audit = 0 forbidden | any failure → stop, record, do not enter V4 |
| **V4 gate** | V1 ∧ V2 ∧ V3 all PASS | otherwise IMG-A stays blocked |

If **any** stage fails: stop, write the stage's result markdown with the failure, commit
it, and **do not** proceed or modify SGLang.

---

## 4. Artifact layout

All validation artifacts under:

```
experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/
  validation_plan.md            (this file)
  results/
    D0_payload_audit.md / .json (V1)
    V2_serving_repro.md         (V2)
  ../smoke_sanitized/           (V3: smoke_summary.md + smoke_results.json)
  ../results/imgA_sanitized_*   (V4, gated)
```

Logs under:

```
logs/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/   (V2 server log)
logs/qwen3vl8b/v2/image_text_benchmarks/smoke_sanitized/   (V3 server logs)
```

**Do not overwrite** the original/invalidated IMG-A artifacts (`results/imgA_results.json`,
`results/imgA_summary.md`, `results/raw/IMG_A_S0_ipc_rep*.jsonl`) or the original
`smoke/` artifacts. Sanitized runs use the `*_sanitized` paths exclusively.

---

## 5. Safety / execution rules

- **No modification to `/sgl-workspace/sglang`** during validation (read-only).
- No profiler. No KAPI logging (`SGLANG_KERNEL_API_LOGLEVEL` / `_LOGDEST` must be unset).
- **GPU 7 only** for GPU stages (V2, V3, V4); never auto-switch.
- Check GPU 7 idle (< 2000 MiB, no foreign heavy process) **before** V2 and V3; stop if busy.
- Kill the SGLang/vLLM server and confirm GPU returns below idle after **each** GPU stage.
- One server at a time; never co-resident.
- Commit after each completed stage (summary markdown + aggregate JSON only).
- Do **not** stage raw JSONL or server logs unless explicitly approved.
- Commit convention `type(scope): summary`; **no** Claude/Anthropic/AI attribution, no
  `Co-Authored-By`. Do not stage `.claude/settings.local.json`.
- Execute stages **serially**, verifying each result from a file before the next (the
  tool-output channel has been unreliable this session; serial + file-read confirmation
  is mandatory).

---

## 6. Command / probe templates — **DO NOT EXECUTE YET**

```bash
# Common clean env for any GPU stage
export CUDA_VISIBLE_DEVICES=7
export HF_HUB_OFFLINE=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST
```

### V1 — CPU audit (DO NOT EXECUTE YET)

```bash
python3 -m py_compile experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/debug_payload_audit.py
python3 experiments/qwen3vl8b/v2/image_text_benchmarks/debug_video_pad/debug_payload_audit.py \
  --seeds 50 --prompts-per-seed 430 --input-len 128
# writes debug_video_pad/results/D0_payload_audit.{md,json}
```

### V2 — serving repro (DO NOT EXECUTE YET)

Server (one only):

```bash
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
SGLANG_USE_CUDA_IPC_TRANSPORT=1 python3 -m sglang.launch_server \
  --model-path "$SNAP" --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer
```

Failing probe vs. control (pseudocode — endpoint `/v1/chat/completions`, one data-URI
image, text containing the literal `<|video_pad|>`):

```python
# tiny 1x1 PNG data URI as the single image
img = "data:image/png;base64,<tiny_png_base64>"
def probe(text):
    payload = {
      "model": SNAP,
      "messages": [{"role": "user", "content": [
          {"type": "image_url", "image_url": {"url": img}},
          {"type": "text", "text": text},
      ]}],
      "max_tokens": 8, "temperature": 0,
    }
    # POST http://127.0.0.1:30000/v1/chat/completions ; record status + body

probe("<|video_pad|>describe the image")   # EXPECT: 400, "No data iterator found for token: <|video_pad|>"
probe("describe the image")                # EXPECT: 200, non-empty greedy output
```

### V3 — sanitized smoke (DO NOT EXECUTE YET)

```bash
python3 experiments/qwen3vl8b/v2/image_text_benchmarks/run_image_text_smoke_sanitized.py
# writes smoke_sanitized/smoke_{summary.md,results.json}; logs under logs/.../smoke_sanitized/
```

---

## 7. Next step after validation

**If V1 ∧ V2 ∧ V3 pass:**

1. Branch SGLang (separate working copy / fork; not the in-place `/sgl-workspace/sglang`
   used for serving) — coordinate before any edit.
2. Patch `gen_mm_prompt` to exclude **all** multimodal/special token ids (e.g.
   `tokenizer.all_special_ids`, plus added-vocab control tokens) from the random pool,
   not just `image_pad_id`.
3. Add a unit test asserting generated prompts contain no special tokens for a VLM
   tokenizer (Qwen3-VL), over a fixed seed sweep.
4. Run minimal validation of the patched generator (re-run V1-style audit → 0 hits).
5. Open a PR referencing this validation (link the D0 audit + V2 repro evidence). The
   profiler-side sanitized wrapper remains as the local unblock; the PR is the durable fix.

**If validation fails at any stage:** update `audit_notes.md` / `debug_plan.md` with the
corrected diagnosis and **do not** modify SGLang.
