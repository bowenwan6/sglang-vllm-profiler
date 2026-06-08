# PCG capture-stream debug — conclusion

> Closes the E-stage matrix planned in `next_debug_plan.md`. Evidence base:
> `static_audit.md`, `results/E1_text_autobench_PCG_control_summary.md`,
> `results/E2a_image_IPC_PCG_n32_summary.md`,
> `results/E3_image_noIPC_PCG_n32_summary.md`. No performance numbers are
> claimed below — this debug exists solely to classify the Stage 4.2
> `IMG_A_S2_ipc_pcg` crash and decide what to do about #4.

## 1. What was tested

All three E-stages ran on
`/data/sglang-pr` upstream `main` at HEAD `62c505a196`, merged generator fix
`07f326c184` confirmed in history, runtime `sglang.__file__` resolved to
`/data/sglang-pr/python/sglang/__init__.py`, fix marker `FIX_OK`, no KAPI,
no profiler, GPU 7 only, snapshot
`Qwen/Qwen3-VL-8B-Instruct @ 0c351dd…`. Each stage launched a fresh server
and killed it before the next.

| stage | config | result | runtime |
|---|---|---|---|
| E1 | text-only via `--dataset-name autobench --dataset-path datasets/qwen3vl8b/caseA_short.jsonl`, c=1, n=8, warmup=0, output_len=128, IPC **off**, PCG **on** | `OK` (8 prompts served, 0 failures, no PCG assertion) | bench ~14 s |
| E2a | image, 720p, c=1, n=32, warmup=30, output_len=128, IPC **on**, PCG **on** | **`PCG_CAPTURE_STREAM_ASSERT`** mid-bench (multiple `200 OK` before the assert) | bench ~30 s |
| E3 | image, 720p, c=1, n=32, warmup=30, output_len=128, IPC **off**, PCG **on** | **`PCG_CAPTURE_STREAM_ASSERT`** mid-bench (identical signature) | bench ~30 s |

E1 displaces D4: D4's `OTHER_FAILURE` was a bench-client HF Hub offline
issue, not a server crash. The `autobench` dataset reads prompts straight
from a local JSONL and never touches the Hub, so E1 cleanly answers the
"is upstream main PCG broadly broken on text-only?" question.

E2a / E3 share every parameter except `SGLANG_USE_CUDA_IPC_TRANSPORT`. Both
hit the **same** assertion (`PCG capture stream is not set, please check if
runtime recompilation happened`) in
`srt/compilation/cuda_piecewise_backend.py:170-172` in roughly the same
wall-clock (~30 s of bench, after ~60 s server startup). The crash fires
**mid-sequence** in both — multiple successful `POST /v1/chat/completions
HTTP/1.1" 200 OK` lines precede the trace in both E2a and E3 logs.

## 2. Findings

- **Upstream `main` HEAD `62c505a196` PCG is not broadly broken.** E1 served
  8 text-only chat-completion requests through PCG with 0 failures and no
  assertion. **Confirmed: the assertion is not a general PCG regression.**
- **Image (multimodal) path + PCG triggers the assertion deterministically.**
  E2a and E3 both reproduce at the same recipe size (n=32, warmup=30,
  output_len=128, c=1) within ~30 s of bench traffic. The smallest known
  reproducing config is **n=32 with warmup=30**; n=2 did not reproduce in
  D1/D2, so the exact threshold lies somewhere in `[3, 32]`. We do not
  claim a tighter bound from the present evidence.
- **IPC is not a required trigger.** Toggling
  `SGLANG_USE_CUDA_IPC_TRANSPORT` between `1` (E2a) and unset (E3) leaves
  the failure mode unchanged. The fault is **VLM image path + PCG alone**.
  CUDA IPC transport, image-feature memory pool, and the
  `MmItemMemoryPool` path are not implicated.
- **Crash is mid-sequence, not first-prefill.** Server logs in both E2a and
  E3 show several requests handled cleanly (`Prefill batch ... cuda graph:
  True`, `Decode batch ... cuda graph: True`, `POST ... 200 OK`) before the
  assertion fires. This refines the `static_audit.md` "fires on first
  prefill" framing: the trigger is a **runtime Dynamo recompile after
  multiple multimodal forward calls**, consistent with the inline comment
  at `cuda_piecewise_backend.py:156-161` ("Dynamo may silently recompile …
  whose token count exceeds the captured range") — but our token count
  (≈1024) is *inside* the captured range; the recompile is being driven by
  some other guard inside the mm embed path.
- **Generator fix gate stayed green throughout.** Every E-stage ran with
  `FIX_OK` and `sglang.__file__` under `/data/sglang-pr/python`. No
  forbidden-token errors. **The Stage 4.2 crash is not a generator bug and
  is not influenced by the generator fix.**

## 3. Interpretation against `next_debug_plan.md` decision matrix

This matches the first row of the §E5 decision matrix:

| E1 (text-only PCG) | E2 (image+IPC+PCG) | E3 (image+noIPC+PCG) | E4 (exact replay) | route |
|---|---|---|---|---|
| **OK** | **ASSERT** | **ASSERT** | n/a | VLM image + PCG specifically unsupported on this upstream main HEAD, **IPC not required**. File upstream SGLang issue. **Continue #4 without PCG.** **No PR**. |

Static-audit §5/§9 read of `server_args.py:1342-1346` and the
`is_multimodal` auto-disable rule lines up: upstream deliberately
auto-disables PCG for multimodal models, `--enforce-piecewise-cuda-graph`
is a documented "for testing" override that bypasses that safety, and the
defensive assertion at `cuda_piecewise_backend.py:170-172` is the price of
using the override on a VLM. Forced PCG on Qwen3-VL is **expected
unsupported behavior** in the current upstream main, not a regression of
the generator fix.

We deliberately did **not** run E4 (exact Stage 4.2 replay at n=400) or
E2b/E2c/E2d. The E2a + E3 minimal repro at n=32 is faster, cheaper, and
sufficient for an upstream issue; one extra exact-replay confirmation is
not worth the ~35-minute GPU window.

## 4. Recommendation

| question | answer |
|---|---|
| Is this our benchmark generator issue? | No. Generator fix `07f326c184` works correctly; `FIX_OK` throughout. |
| Is this our runner/config issue? | No. Same recipe with PCG off (`IMG_A_S0_ipc`) runs 5/5 reps clean at n=400; same recipe with PCG on hits the assertion at n=32 regardless of IPC. |
| Is IPC required to trigger? | **No** (E3 matches E2a). |
| Is image / VLM required to trigger? | **Yes** (E1 text-only PCG is `OK` on the same HEAD). |
| Is it an upstream SGLang PCG bug or unsupported case? | **Unsupported case under the documented `--enforce-piecewise-cuda-graph` override** for multimodal models. The auto-disable for VLMs (`server_args.py:1374-1376`) exists precisely because PCG cannot safely handle the multimodal forward path; the override bypasses that safety. The assertion is a defensive guard, not a regression. |
| Should we file an upstream SGLang issue? | **Yes — informational, scoped.** Ask upstream to either (a) extend the existing AMD/HIP fallback at `cuda_piecewise_backend.py:163-169` to CUDA so that `--enforce-piecewise-cuda-graph` degrades gracefully on multimodal models instead of crashing, or (b) print a loud warning when the override is set on a multimodal model. Include the E2a minimal repro (the recipe is a deterministic ~30-second GPU run) as supporting evidence. The existing in-source error message already tells the user "add `--disable-piecewise-cuda-graph`", but we still surface as a hard crash. |
| Should we prepare a SGLang PR? | **No, not at this stage.** A PR would require choosing between (a) graceful fallback to eager execution on a recompile (matches the HIP behavior) and (b) refusing the `--enforce` override outright on VLMs. Both are policy choices upstream owns. We do not have evidence to prefer one over the other, and "selective/default-on PCG" is the scope of Issue #5 — not this debug. |

## 5. Implication for Issue #4

- The **PCG benefit on image+text (Q2)** cannot be measured on this upstream
  main HEAD without an upstream change. The Case-A text-only finding that
  PCG drops TTFT from 21.94 ms to 14.04 ms (Issue #2) **does not transfer
  to the image path** as measurable evidence within #4 under the current
  upstream codebase.
- The other three #4 questions remain answerable on the fixed-generator
  path **without PCG**:
  - Q1 (vLLM anchor): `IMG_A_V0_vllm` vs `IMG_A_S0_ipc`.
  - Q3 (IPC benefit): `IMG_A_S0_noipc` vs `IMG_A_S0_ipc`.
  - Bracket drift: `IMG_A_S0_ipc_repeat` vs `IMG_A_S0_ipc`.
- The first Stage 4.2 partial run got `IMG_A_S0_ipc` clean (5/5 reps, 0
  failures, TTFT p50 64.8 ms). The next #4 step, **outside this debug**, is
  to resume IMG-A with the non-PCG variants only
  (`S0_ipc_repeat → V0_vllm → S0_noipc`) so we recover bracket drift, IPC
  benefit, and vLLM anchor. **`S2_ipc_pcg` stays excluded** with a documented
  rationale: PCG is upstream-auto-disabled for VLMs and the override
  required to force it crashes deterministically on this HEAD.

## 6. Do **not**

- Do not claim PCG performance numbers (the only `IMG_A_S2_ipc_pcg` data we
  have is the partial-run crash record).
- Do not claim a tighter minimal-reproducing size than n=32 with warmup=30
  (we did not test the range 3 ≤ n < 32).
- Do not claim the generator bug is still open. It is not.
- Do not propose an upstream PR before this debug evidence is reviewed by
  the user; the recommendation is informational issue first, PR only if
  the issue review identifies an agreed minimal fix.
