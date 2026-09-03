# Protocol — Issue #4: Qwen3-VL image+text benchmarks with CUDA IPC transport

> ⚠️ **Amendment pending (2026-09-03) — the arm matrix below is out of date.**
> Upstream restructured the CUDA-graph flags: `--enforce-piecewise-cuda-graph`
> is now a deprecated alias for `--cuda-graph-backend-prefill=tc_piecewise`,
> **breakable (BCG) is the default prefill backend on CUDA**, and PR #33726 adds
> Qwen3-VL to the breakable allowlist — so this protocol's `S0` "default" arm
> flips meaning when that PR merges. The two-arm `S0` / `S2_ipc_pcg` design must
> be widened to four (`default` / `disabled` / `tc_piecewise` / `breakable`) and
> every arm must log its **resolved** backend, because an unsupported request is
> silently downgraded to `disabled`. See [`plan.md` §3.5](../../../../plan.md).
> **Superseded 2026-09-03:** the variant matrix in §4–§6 below is replaced by
> [`plan.md` §11](../../../../plan.md) (Issue #4 execution plan v3). The goal,
> dataset, workload shapes and artifact rules here still stand. The CUDA-IPC
> half is **also** affected: `SGLANG_USE_CUDA_IPC_TRANSPORT` is deprecated in
> favour of `--mm-feature-transport={cpu,cuda_ipc,cuda_vmm}`, and unset now
> resolves to `cpu` — so this protocol's "headline must set IPC on" rule
> describes a non-default configuration.

> **Status: protocol drafting / pending approval. No benchmark runs, no servers started, no SGLang source
> changes.** Builds on #2 (text-only, complete). Clean only: never set `SGLANG_KERNEL_API_LOGLEVEL` /
> `SGLANG_KERNEL_API_LOGDEST`; no profiler. SGLang image headline runs **must** set
> `SGLANG_USE_CUDA_IPC_TRANSPORT=1`.

## 0. Context carried from #2 (text-only)

- Production-default text-only Case A (overlap-ON): SGLang default **21.94 ms**, +PCG **14.04 ms**, vLLM
  **13.12 ms**; Case C batched: **no material gap / no Case-A-like PCG benefit**.
- **These are text-only results. They do NOT transfer to the image path automatically** — that is the
  reason #4 exists.
- Two distinct, separately-explained levers:
  - **CUDA IPC transport** (`SGLANG_USE_CUDA_IPC_TRANSPORT=1`): an *image-feature transport* optimization
    (how multimodal tensors move between processes). Only meaningful for SGLang image runs.
  - **PCG** (`--enforce-piecewise-cuda-graph`): a *prefill graph-coverage* lever (testing lever, **not** a
    production fix). Same flag as #2.
  These must never be conflated in any conclusion.

## 1. Goal

1. Add clean Qwen3-VL **image+text** benchmarks (single H200, TP=1, bf16, greedy), serialized servers.
2. Determine whether the text-only Case-A **PCG** finding **transfers to the image path**.
3. **Separate** the CUDA-IPC benefit (transport) from the PCG benefit (prefill graph coverage), each
   measured against its own baseline.
4. Provide real VLM (image) evidence to scope **#5** (selective/default-on PCG PR) — so #5 is grounded in
   the realistic multimodal path, not just text-only.

## 2. Experimental Questions

- **Q1 (gap):** image+text **c=1**, SGLang **default + IPC** vs vLLM — how large is the TTFT gap?
- **Q2 (PCG transfer):** on the SGLang **default + IPC** baseline, does adding `--enforce-piecewise-cuda-graph`
  reduce TTFT (≥5%, TPOT not worse, 0 failures)? I.e. does the Case-A PCG win appear in the image path?
- **Q3 (IPC benefit):** how much does CUDA IPC itself buy? `S0_noipc` vs `S0_ipc` delta on the **same**
  workload (≥5% to count).
- **Q4 (shape boundary):** at **c=16** batched image+text, is there still no Case-A-like PCG benefit (does
  the #2 boundary hold for images)?
- **Q5 (composition):** how is the TTFT composed — preprocessing / vision encoder / prefill / decode? What
  can the current harness record, and what is deferred to a profiler track? (See §7.)

## 3. Workloads

The SGLang `image` dataset (`--dataset-name image`) **synthesizes images inline** (random or blank pixels →
base64 data URIs); no external assets, no runtime downloads. Text length is sampled via `--random-input-len`
with `--random-range-ratio` (pin to a fixed length — verify the exact range-ratio semantics in Phase 4.0).
`prompt_len` counts **text + vision** tokens; `text_prompt_len` / `vision_prompt_len` are recorded
separately by the harness.

Primary workloads (single image per request unless noted):

| id | images | resolution | text in (target) | output | concurrency | num-prompts | warmup | reps |
|---|---|---|---|---|---|---|---|---|
| **IMG-A** single image + short text | 1 | 720p | ~128 tok | 128 | 1 | 400 | 30 | 5 |
| **IMG-B** single image + medium text | 1 | 720p | ~512 tok | 128 | 1 | 400 | 30 | 5 |
| **IMG-C** single image + text, batched | 1 | 720p | ~128 tok | 128 | 16 | 2000 | 500 | 3 per interleaved block |

Optional (only if harness multi-image is confirmed working in 4.0 **and** explicitly approved):

| id | images | resolution | text in | output | concurrency | num-prompts | warmup | reps |
|---|---|---|---|---|---|---|---|---|
| IMG-D multi-image | 2–3 (`--image-count`) | 720p | ~128 tok | 128 | 1 | 400 | 30 | 5 |

- **Resolution policy:** fix at **720p (1280×720)** for the headline (moderate vision-token load,
  reproducible). Resolution is a controlled variable; do not mix resolutions within a workload. Higher res
  (1080p/4k) only as a separate, explicitly-approved sensitivity check.
- **Image content:** `random` pixels (realistic vision-token cost; `blank` compresses to near-zero and
  under-loads the vision tower — use `blank` only for a transport-only micro-check if ever needed).
- **Reduced plan (if runtime too long):** IMG-A/B reps 3 (still bracket S2); IMG-C 2 reps per block; drop
  IMG-D. **Never sacrifice:** num-prompts (sample size), warmup, the 0-failure gate, the clean
  (no-KAPI/profiler) condition, the IPC-on requirement for SGLang headline, or the bracket/interleave drift
  control. Reduce *repetitions*, never *sample size* or *drift control*.

## 4. Assets and Dataset Policy

- **Synthetic, in-harness generation (chosen).** Why: the `image` dataset generates images at runtime from
  `random.seed`/`np.random.seed` (both seeded from `--seed`), so a run is reproducible from
  **(harness git commit + image params + seed)** with **no external URL download** and no large binaries to
  check in. This satisfies "no runtime external download" cleanly.
- **Dataset identity recorded per run** (in place of a file sha256): harness module path + **SGLang git
  commit** (`bench_serving` provenance), `--seed`, `--image-count`, `--image-resolution`, `--image-format`,
  `--image-content`, `--random-input-len`, `--random-output-len`, `--random-range-ratio`, `--num-prompts`.
- **Optional byte-level provenance:** if a checked-in artifact is later required, dump the generated
  request payloads to `datasets/qwen3vl8b/image_text/<workload>.generated.jsonl` and record its sha256.
  Not required for the headline (the seed+params recipe is the canonical identity).
- **If real images are ever substituted** (not the plan): they must be checked into
  `datasets/qwen3vl8b/image_text/assets/` with a per-file sha256 in `assets_manifest.md`, never fetched at
  runtime. JSONL prompt files would live in `datasets/qwen3vl8b/image_text/` with recorded sha256.
- Image format: **png** for the headline (lossless, deterministic given seed); jpeg only if a size/latency
  sensitivity check is explicitly requested.

## 5. Variants

Common server flags: SGLang `--dtype bfloat16 --tp 1 --attention-backend flashinfer`; vLLM
`--dtype bfloat16 --tensor-parallel-size 1`. All clean (no KAPI, no profiler). Greedy
(`temperature=0, top_p=1`). **Benchmark backend = `sglang-oai-chat` for both frameworks** (image dataset
rejects `--backend vllm`; the chat request function POSTs `image_url` data URIs to any
`/v1/chat/completions`). SGLang on port 30000, vLLM on 30001. **This dual-use of `sglang-oai-chat` MUST be
smoke-verified against vLLM in Phase 4.0 before it is trusted** (see §11, and the Open Items).

Per primary workload:

| id | framework | env | distinguishing flags | role |
|---|---|---|---|---|
| `S0_ipc` | SGLang | `SGLANG_USE_CUDA_IPC_TRANSPORT=1` | *(none — overlap ON)* | **headline production baseline (image)** |
| `S2_ipc_pcg` | SGLang | `SGLANG_USE_CUDA_IPC_TRANSPORT=1` | `--enforce-piecewise-cuda-graph` | PCG testing lever on the IPC baseline |
| `V0_vllm` | vLLM | *(IPC n/a)* | *(none)* | clean cross-framework anchor |
| `S0_noipc` | SGLang | `SGLANG_USE_CUDA_IPC_TRANSPORT` **unset** | *(none — overlap ON)* | **IPC ablation** |

- `S0_noipc` (IPC ablation) is **required on IMG-A only**. On IMG-B / IMG-C it is **optional, run only if
  explicitly approved** (controls runtime).
- PCG ablation `--disable-overlap-schedule` is **not** part of #4 (overlap question was settled in #2).

## 6. Run Design (bracket / interleave to bound drift)

- **IMG-A / IMG-B (low-variance, c=1) — bracket:**
  `S0_ipc → S2_ipc_pcg → S0_ipc_repeat → V0_vllm → S0_noipc`
  The `S0_ipc_repeat` brackets `S2_ipc_pcg` to measure baseline drift; `S0_noipc` last gives the IPC delta.
- **IMG-C (high session variance, c=16) — interleaved:**
  `S0_a → S2_a → S0_b → S2_b → S0_c → V0`
  (no-IPC batched ablation only if explicitly approved — append `S0_noipc` block then).
- Each variant: fresh server, **smoke check** (a real image+text request returning non-empty greedy
  output), then reps; server killed and GPU freed below the idle threshold before the next variant. Servers
  never co-resident.

## 7. Metrics (per rep + per-variant median / CV)

- **Timing:** TTFT p50 / p95 / p99 · TPOT p50 · E2E latency · output throughput (tok/s) + request
  throughput (req/s) · CV across reps · failures / error rate.
- **Token composition (recorded, not timed):** `prompt_len` (text+vision), `text_prompt_len`,
  `vision_prompt_len`, total vision tokens — all emitted by the image dataset / harness.
- **Provenance per run:** GPU id · exact server flags · **`SGLANG_USE_CUDA_IPC_TRANSPORT` on/off** ·
  **PCG on/off** · framework versions (SGLang + vLLM) · model snapshot sha · benchmark backend
  (`sglang-oai-chat`) · the §4 dataset-recipe fields (incl. seed) · explicit
  `kapi_logging=false, profiler=false` confirmation · smoke-check output.
- **TTFT composition boundary (Q5):** the benchmark client measures **end-to-end serving TTFT** — it
  **includes** image preprocessing + vision-encoder + prefill, and does **not** split them. Therefore:
  - **Headline metric = end-to-end serving TTFT as observed by the benchmark client.**
  - A preprocessing / vision-encoder / prefill / decode breakdown is an **optional future profiler track**
    (separate trace collection), **not required** for the #4 headline. Record token-count splits
    (text vs vision) as the available proxy.

## 8. Acceptance Criteria

- **Clean-run gate:** 0 failures on every counted rep; no KAPI env var; no profiler; servers serialized;
  smoke check passes (non-empty greedy output on a real image+text request) before reps.
- **IPC-on gate (headline):** every SGLang **headline** image run has `SGLANG_USE_CUDA_IPC_TRANSPORT=1`
  recorded; `S0_noipc` is the only SGLang variant with it unset.
- **No external download gate:** images synthesized in-harness (or read from checked-in hashed assets); no
  network fetch during the benchmark.
- **PCG-benefit rule (Q2):** declare a PCG benefit only if `S2_ipc_pcg` median TTFT improves **≥ 5%** vs the
  bracketed `S0_ipc`, **TPOT not worse**, 0 failures. Otherwise: "no PCG benefit on the image
  default+IPC baseline."
- **IPC-benefit rule (Q3):** declare an IPC benefit only if `S0_noipc` vs `S0_ipc` differ by **≥ 5%** on the
  same workload (state direction; same 0-failure/clean gates).
- **Bracket-drift gate:** `S0_ipc` vs `S0_ipc_repeat` (or the three IMG-C S0 blocks) drift ideally **≤ 5%**;
  if > 5%, downgrade the affected absolute number to "indicative" and lean on the within-design comparison.
- **vLLM is anchor only** — context for the SGLang gap, not direct causal proof of SGLang's mechanism.
  Report SGLang-vs-SGLang (S0 vs S2, S0_ipc vs S0_noipc) as the causal comparisons.
- **Separation requirement:** image+text conclusions are reported **separately** from text-only (#2);
  IPC-benefit and PCG-benefit are reported as **two distinct** results, never merged.
- **IMG-C verdict wording:** if `S0`, `S2`, `vLLM` all sit inside the noise band, conclude **"no material
  gap / no Case-A-like benefit on the image batched path,"** not parity and not a proven gap.

## 9. Stop Conditions (abort, do not work around)

- Target GPU not idle (≥ 2000 MiB used / other heavy process) at run time.
- OOM (vision tower / KV cache) — record resolution/image-count that triggered it; do not silently retry.
- Any counted variant has failures > 0.
- Image preprocessing errors (chat-template fallback to `<image>` tag, processor exceptions).
- Asset missing / sha256 mismatch (if checked-in assets are used).
- Any `SGLANG_KERNEL_API_LOGLEVEL` / `SGLANG_KERNEL_API_LOGDEST` set, or a profiler active.
- A server fails to release GPU memory after kill (still occupied before next variant).
- The plan would require editing SGLang source.
- `S2_ipc_pcg` smoke / correctness check fails.
- Server log / raw output grows abnormally large (instrumentation-leak signal).
- **Phase-4.0 blocker:** if `--backend sglang-oai-chat` cannot drive vLLM's chat endpoint with images,
  stop and report — do **not** improvise an unverified vLLM image path.

## 10. Artifact Plan

- Protocol: `experiments/qwen3vl8b/v2/image_text_benchmarks/protocol.md` (this file).
- README: `experiments/qwen3vl8b/v2/image_text_benchmarks/README.md` (scaffold).
- Future runner: `experiments/qwen3vl8b/v2/image_text_benchmarks/run_image_text_benchmarks.py`
  (**not implemented until Phase 4.0 confirms the schema**).
- Future results: `experiments/qwen3vl8b/v2/image_text_benchmarks/results/` — per-variant `results.json`,
  per-rep `raw/*.json` + `*_meta.json`, `summary.md`.
- Datasets / recipe: `datasets/qwen3vl8b/image_text/` (recipe README + optional generated/hashed JSONL);
  real assets (if ever) under `datasets/qwen3vl8b/image_text/assets/` with `assets_manifest.md`.
- Server logs: `logs/qwen3vl8b/v2/image_text_benchmarks/`.
- **Raw per-rep dumps and server logs are NOT committed unless explicitly approved** (committed
  deliverables = summaries + aggregate `results.json`). **Never** touch v1 Phase 0–5 artifacts or #2
  artifacts.

## 11. Execution Phases (gated — approval required between gates)

- **Phase 4.0 — Schema discovery / smoke (no perf run).** Verify with a tiny `--num-prompts 2`:
  1. `--dataset-name image --backend sglang-oai-chat` against **SGLang** `/v1/chat/completions` (30000)
     returns non-empty greedy output, 0 errors, and reports vision token counts.
  2. The **same** invocation against **vLLM** `/v1/chat/completions` (30001) works (resolves the vLLM-anchor
     Open Item). If not, stop and report alternatives.
  3. Confirm `--random-range-ratio` value that pins text length; confirm `--seed` reproducibility (two runs
     identical token counts).
  4. Confirm `SGLANG_USE_CUDA_IPC_TRANSPORT=1` is accepted by the SGLang server and observable (log line /
     no error); confirm unset path also works.
- **Phase 4.1 — Assets + dataset recipe.** Finalize image params; if byte-level provenance is wanted, add a
  small generator/dump that writes + hashes `image_text/<workload>.generated.jsonl`. No perf run.
- **Phase 4.2 — Smoke SGLang + vLLM image request** end-to-end through the runner skeleton (still tiny n).
- **Phase 4.3 — Run IMG-A only** (bracket, GPU reconfirmed idle, clean). Summarize before proceeding.
- **Phase 4.4 — Review IMG-A**, then decide IMG-B / IMG-C (and whether any optional no-IPC batched ablation
  or IMG-D runs).
- **Phase 4.5 — Summarize** (IPC benefit, PCG transfer verdict, batched boundary) and update the v2 roadmap;
  feed evidence to #5.

## Open Items (must be resolved before any perf run)

1. **vLLM image anchor backend.** Plan uses `--backend sglang-oai-chat` for vLLM too (the image dataset
   rejects `--backend vllm`, but the chat request function is HTTP-generic). **Unverified** — Phase 4.0
   smoke #2 is the gate. If vLLM's chat endpoint rejects the data-URI `image_url`, fall back to a
   documented alternative (e.g. vLLM-native chat client) or report the anchor as unavailable.
2. **Range-ratio / length pinning semantics** for the image dataset (`compute_random_lens`) — confirm the
   value that fixes text length (Phase 4.0 #3) so IMG-A/B text length is controlled.
3. **CUDA IPC observability** — confirm the env var actually engages the IPC transport path (not silently
   ignored) so `S0_ipc` vs `S0_noipc` is a real contrast (Phase 4.0 #4).

## 12. Command Templates (DO NOT EXECUTE — reference only)

```bash
# Clean env (every run): pin GPU, offline, strip KAPI. IPC set ONLY for SGLang headline variants.
export CUDA_VISIBLE_DEVICES=<GPU_ID>            # from nvidia-smi at preflight
export HF_HUB_OFFLINE=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b

# --- SGLang headline (image, overlap ON, IPC ON): S0_ipc / S2_ipc_pcg ---
SGLANG_USE_CUDA_IPC_TRANSPORT=1 python3 -m sglang.launch_server --model-path "$SNAP" \
  --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer            # + --enforce-piecewise-cuda-graph for S2

# --- SGLang IPC ablation: S0_noipc (env var UNSET) ---
python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 \
  --port 30000 --tp 1 --attention-backend flashinfer

# --- vLLM clean anchor (served via OpenAI chat endpoint) ---
/opt/miniconda3/envs/profiling/bin/python -m vllm.entrypoints.openai.api_server \
  --model "$SNAP" --dtype bfloat16 --port 30001 --tensor-parallel-size 1

# --- Bench client (image dataset; backend sglang-oai-chat for BOTH frameworks) ---
# IMG-A: --image-count 1 --image-resolution 720p --random-input-len 128 --random-output-len 128 \
#        --max-concurrency 1  --num-prompts 400  --warmup-requests 30
# IMG-C: ... --max-concurrency 16 --num-prompts 2000 --warmup-requests 500
python3 -m sglang.bench_serving --backend sglang-oai-chat \
  --base-url http://127.0.0.1:<30000|30001> --dataset-name image \
  --image-count 1 --image-resolution 720p --image-format png --image-content random \
  --random-input-len <T> --random-output-len 128 --random-range-ratio <pin> \
  --max-concurrency <C> --num-prompts <N> --seed 1 --warmup-requests <W> \
  --extra-request-body '{"temperature": 0, "top_p": 1}' \
  --output-details --output-file experiments/qwen3vl8b/v2/image_text_benchmarks/results/raw/<variant>_rep<k>.json
```
