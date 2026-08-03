# GDN Prefill-BCG Investigation — Consolidated Audit

**Date:** 2026-08-03
**Branch:** `debug/qwen35-4b-gdn-prefill-bcg`
**HEAD:** `2490057` — smoke test recorded `SCAFFOLDING_PASS`.
**Auditors:** Lead + three parallel agents (harness / source / validation-methodology).
**Purpose:** ground the execution plan in verified state, not assumptions.

## Executive summary

The scaffolding *appears* to be a scored-sweep harness but is really a shell:
the plumbing between runner → server → client → correctness verifier → verdict
runner exists and demonstrably passes 4 requests end-to-end, but at least six
load-bearing pieces (logprob collection, `output_ids` capture, Nsight
extraction, Gate-2/3/4 data-producing runners, noise-floor pilot, post-teardown
GPU idleness assertion) are absent. A scored run today would produce a
`Gate 1 = PASS` verdict that validated *nothing*, because the client does not
request or record the fields the gate needs to compare, and the verifier
silently treats absence as within-tolerance.

The source-level story is now well-understood: on the frozen SGLang SHA
`58974ca16c` the Qwen3.5-4B execution path under BCG is fully reachable on
H200, the GDN alt-stream branch fires unconditionally on CUDA whenever the
padded bucket size is `< 1024`, and `get_is_capture_mode()` returns True for
**both** BCG capture *and* replay. That gives us a concrete, testable
hypothesis for the leading performance concern with a specific source
citation. The audit surfaces one additional correctness risk worth watching
(alt-stream capture join-wait may be discarded by BCG's stream-tracker hook
under some driver conditions — R13.4 below).

Preservation invariants are intact. No major blocker.

---

## 1. Repository, preservation, environment (verified)

- Branch: `debug/qwen35-4b-gdn-prefill-bcg`; HEAD `2490057d01c4c2b43333c648832ef4d8782078fa`.
- Upstream tracking: `origin/debug/qwen35-4b-gdn-prefill-bcg` @ `2490057` (in sync).
- All 10 expected commits (`d29b4a6` → `2490057`) present on origin.
- Working tree: only the two protected DeepStack items (`M …/R5C_correctness_audit/audit_report.md`, `?? …/R6.3_image_and_sweep/attempt_gpu2_partial_orphaned_20260729T094128Z/`); nothing in the gdn subtree is dirty.
- Frozen SGLang: `/tmp/claude-0/…/scratchpad/sglang_checkout/sglang`, HEAD `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`, `git diff --stat` empty.
- `/data/sglang-fork`: HEAD `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` — unchanged.
- No stray worktrees; single tree at `/data/sglang-vllm-profiler`.
- Environment: Python `3.12.3`, Nsight Systems `2026.3.1.157-263138048394v0`.
- Model checkpoint: `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` cached at `/root/.cache/huggingface/hub/…` (confirmed loaded during the smoke).
- **GPU availability changed since the smoke.** GPUs 0-5 currently occupied by foreign compute processes (PIDs 3222330, 3222334, 3222335, 3222337, 3351862, 3351866); GPUs 6 and 7 are free (0 MiB, 0% util, no compute apps). Working pool for this investigation is `{6, 7}`.
- No forbidden-string commits in recent 12.

```
SIGNAL: SIGNAL_GOOD
PHASE: Repository & preservation audit
SUMMARY: Branch state clean, all 10 commits on origin, preservation invariants held (fork + frozen SGLang unchanged, protected DeepStack items untouched). GPU pool is {6, 7} — narrower than allowlist but sufficient for a single-cell smoke ladder.
EVIDENCE: git status, git log --oneline, git rev-parse in both checkouts, nvidia-smi census.
DECISION: Proceed to harness and source audits; no blocker.
NEXT_ACTION: Consolidate harness + source + validation findings, produce plan.
COMMIT: 2490057
```

---

## 2. Harness audit — verified gaps

Every gap below is a defect with a clear fix; none is a preservation or
correctness-of-history concern.

### 2.1 Client cannot produce Gate-1 data (blocking for scoring)
- `gdn_client.py:145` sends `"return_logprob": False`.
- `gdn_client.py:138-148` builds the request body without `top_logprobs_num`.
- `gdn_client.py:244-257` records neither `output_ids` nor `output_logprobs` nor `finish_reason` at the record top level.
- Smoke's `records_A0_p128_b1.jsonl` confirms: top-level keys are `arm, batch_size, e2e_ms, new_tokens, output_len_tokens, output_text, prompt_bytes_sha256, prompt_len_target_chars, prompt_len_target_tokens, prompt_source_id, request_id, server, server_error, status_code, timestamp, ttft_ms`. No `output_ids`, no `output_logprobs`.
- No `/tokenize` query. `approx_chars_for_tokens` (line 95-103) always uses `target_tokens × 3.5`.
- `contextlib.suppress(OSError)` on record write (`gdn_client.py:259-260`) → I/O failure would drop the entire JSONL while still printing "N records written".
- Batched HTTP response: `zip(batch_seeds, prompts, request_ids, results)` at line 242 silently truncates if server returns fewer entries than sent.
- TTFT unmeasurable in non-stream mode (`stream: False` at line 146); always `null` in records.

### 2.2 Correctness verifier silently degrades (blocking for scoring)
- `gdn_correctness.py:75-86` — `_token_stream` falls back to `record["output_text"].split()` (whitespace tokenisation) when `output_ids` is missing. Two arms producing identical whitespace-segmented text pass Gate 1 even with wildly different actual tokens.
- `gdn_correctness.py:111-113` — `within_logprob_tolerance = (max_abs_logprob_diff is None or ...)`; missing logprobs are treated as within tolerance.
- Tolerance formula `max(0.05, 3 × noise_floor)` documented at `gdn_correctness.py:54-55` and `validation_plan.md:103` but **not implemented**; only fixed `--tolerance 0.05` exists.
- No `--noise-floor` CLI argument.
- Only the *first* record per `prompt_source_id` is compared (`gdn_correctness.py:71`); other 7 timed records silently ignored.
- Gate-2/3/4 labels are free text — a mis-labelled invocation still returns a verdict.
- No test exercises the whitespace-fallback path or Gate 4.

### 2.3 Verdict runner conflates missing with failed (blocking for interpretation)
- `gdn_verdict.py:73-87` — a missing gate is `{"overall": "MISSING"}` and flips `all_pass=False`; `decide()` then returns `FAIL_BCG_GDN_CORRECTNESS` (line 197-198). No way to distinguish "gate missing because scaffolding gap" from "gate failed because real divergence".
- **No `PARTIAL_SWEEP` verdict label.** Grep across `gdn/` returns zero matches.
- `H_A` iterates `("A1","A2","A3")` at `gdn_verdict.py:145` — should be `("A1","A3")` per `hypothesis.md:78-83`. A2 (eager prefill + decode CG) could falsely trigger `PASS_BCG_GDN_NOTABLE_GAP` from a decode-CG artefact.
- All 4 gates PASS + 0 nsys rows → `AMBIGUOUS` at `decide()` line 202-203 (correct behaviour, but exactly what a scored sweep would currently emit because no extractor exists).

### 2.4 Nsight capture wrapper has no extractor (blocking for perf verdict)
- `nsys_capture.sh:9-12` explicitly defers `extract_nsys_metrics.sh` to "a follow-up commit" that doesn't exist.
- Uses `--stats=false` (line 131); `nsys stats` is not invoked as part of capture.
- Consequence: `gdn_verdict._read_nsys_csv` (verdict.py:90-98) always sees 0 rows → any all-passing gate summary collapses to `AMBIGUOUS`.
- NVTX-tagged per-op ranges (12 predeclared in `validation_plan.md:147-159`) don't exist because `gdn_instrumentation.py` is a documented no-op. `H_C` ("excess concentrated in NVTX range tagged to specific GDN op", `validation_plan.md:186-187`) is unverifiable in the current scaffold.

### 2.5 Runner ordering / gap defects (blocking)
- **Preflight ordering bug**: `gdn_runner.sh:227` writes `preflight.json` before `export LD_PRELOAD` (line 285) and `export CUDA_VISIBLE_DEVICES` (line 286). Confirmed in smoke's `preflight.json:74`: both `LD_PRELOAD: null` and `CUDA_VISIBLE_DEVICES: null` despite the runner setting them. Silently defeats those checks.
- **Missing `mkdir` before preflight write**: `preflight.json` writes to `$RESULTS_DIR/preflight.json` at line 227; `mkdir -p "$RAW_DIR"` is not until line 262. Fresh attempts would fail; smoke worked because the operator hand-created the results dir.
- **No post-teardown GPU idleness assertion**: `gdn_runner.sh:305` trap only signals PGID. Smoke's `gpu_post_state` in `metadata.json:16-17` was hand-composed, not runner-written.
- **`$SERVER_PORT=30001` default hard-coded**: concurrent captures on the same host would collide.
- **`--mem-fraction-static 0.7` and `--tp 1` hard-coded** at `gdn_runner.sh:276-277`.
- **No `--chunked-prefill-size` plumbing** anywhere in runner or client → Gate 3 cannot be exercised.
- **No 64-cell sweep orchestrator** — runner is single-cell.

### 2.6 Preflight coverage gaps (blocking for provenance)
- `REQUIRED_GDN_CONFIG_FIELDS` (line 48-58) omits `full_attention_interval` (pinned in `provenance.md:58` but not asserted).
- Does not shell out to `git -C <frozen> rev-parse HEAD` to verify frozen SHA.
- Does not import `sglang` to check `sglang.__file__` resolves inside frozen checkout.
- Does not verify model revision hard-fail (only records it).
- Does not verify `LD_PRELOAD` value.
- Does not record `torch`, `sgl-kernel`, `flashinfer` versions.
- Does not check `nsys --version`.
- Steps 1-2, 4, 6, 7 of `validation_plan.md §3` are missing; only steps 3 (config) and 8 (env) implemented — and step 8 is broken by the ordering bug (§2.5).

### 2.7 Test coverage gaps
- No live-process lifecycle test — no test that actually spawns the launcher (only argparse dry-runs).
- No fake-HTTP-server test for the client.
- No test for the runner's setsid PGID capture (would require actually spawning setsid).
- No test for Gate 4.
- No test for the client → correctness → verdict data-flow with `output_ids` missing (the exact scenario the smoke exercised).

### 2.8 Cosmetic / non-blocking
- Stale comments in `gdn_runner.sh:6` and `nsys_capture.sh:6` still say `{0, 1, 7}` allowlist (the actual default in both is `{0..7}` per Amendment 1).
- Fixture SHA `8a660d94…` not stamped in charter docs; only in `manifest.json`. A fixture change would not show as a docs diff.
- HF preflight hits internet on every run (no offline mode). Not a bug, but a run-time dependency.

```
SIGNAL: SIGNAL_GOOD
PHASE: Harness audit
SUMMARY: Nine categories of harness defects, all with clear low-risk fixes. Six are blocking for a meaningful scored sweep; three are non-blocking. No preservation, correctness-of-history, or scope concern. The scaffolding requires hardening but the design is right.
EVIDENCE: Direct read of client, runner, correctness verifier, verdict runner, nsys wrapper, preflight, tests; cross-checked against smoke result at commit 2490057.
DECISION: Fold every §2.1–§2.7 defect into Phase 4 tooling hardening. Cosmetic §2.8 items get tidied opportunistically.
NEXT_ACTION: Complete source and validation audits, then write Phase 2 execution plan.
COMMIT: 2490057
```

---

## 3. Source & BCG audit — established facts

All line numbers against frozen SGLang at `58974ca16c`.

### 3.1 Qwen3.5-4B execution path

- Entry: `Qwen3_5ForConditionalGeneration` (`python/sglang/srt/models/qwen3_5.py:1771`), inherits `Qwen3VLForConditionalGeneration`. LM class: `Qwen3_5ForCausalLM` (`qwen3_5.py:1242`).
- Layer selection at `qwen3_5.py:1359-1373`: `layer_type = config.layers_block_type[idx]`; internal renaming from HF's `text_config.layer_types` (`"linear_attention" | "full_attention"`) to SGLang's `("linear_attention" | "attention")`.
- Qwen3.5-4B topology (verified 2026-08-03 via preflight): 32 layers, `full_attention_interval=4` → **24 GDN layers + 8 full-attention layers** (3:1 ratio).
- `heads_ratio = linear_num_value_heads / linear_num_key_heads = 32/16 = 2` → **fused split kernel path always taken** (`qwen3_5.py:635`).
- Under BCG, `_run_forward` (`prefill_cuda_graph_runner.py:645-653`) calls `self.layer_model.forward(input_ids, positions, forward_batch, forward_batch.input_embeds)` — the inner transformer only, bypassing `Qwen3VLForConditionalGeneration.forward` / `general_mm_embed_routine` / `logits_processor` (they run eagerly on replay).

### 3.2 GDN forward path (ordered ops in `Qwen3_5GatedDeltaNet.forward`, `qwen3_5.py:620-685`)

1. `_forward_input_proj` — parallel `in_proj_qkvz` + `in_proj_ba` (optionally alt-stream).
2. `fused_qkvzba_split_reshape_cat_contiguous` (fused kernel; hit for Qwen3.5-4B).
3. `RadixLinearAttention.forward` (**graph break under BCG** — see §3.4).
4. Reshape / DP-attn padding guard.
5. `RMSNormGated`.
6. Output projection `out_proj` (RowParallelLinear).

### 3.3 Alt-stream branch (`qwen3_5.py:551-585`) — leading perf hypothesis

- `_gdn_use_alt_stream = True` **unconditionally on CUDA** (`qwen3_5.py:128`); env var only consulted on HIP.
- Predicate to enter the alt-stream branch (line 561-566):
  ```
  self.alt_stream is not None
  AND get_is_capture_mode()          # True during BCG capture AND replay
  AND seq_len < DUAL_STREAM_TOKEN_THRESHOLD
  AND _gdn_use_alt_stream            # always True on CUDA
  ```
- `DUAL_STREAM_TOKEN_THRESHOLD = 0` on CPU/NPU/TC piecewise; **`1024` everywhere else — including BCG**. **No BCG-specific short-circuit exists** (only TC piecewise is zeroed).
- `seq_len = hidden_states.shape[0]` is the **padded bucket size** at BCG replay (`_pad_to_bucket` at `prefill_cuda_graph_runner.py:1363-1374`), not the raw request token count.
- `get_is_capture_mode()` (`runner_utils/capture_mode.py:39-40`) returns `is_capture_mode OR is_in_breakable_cuda_graph()`. The BCG scope `enable_breakable_cuda_graph()` wraps **both capture and replay** (`breakable_cuda_graph_backend.py:90-105, 237-240`). So the branch fires on every BCG replay when the padded bucket is `< 1024`.
- Fork/join primitives inside the branch (`qwen3_5.py:567-572`):
  ```python
  current_stream = torch.cuda.current_stream()
  self.alt_stream.wait_stream(current_stream)
  projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)  # main stream
  with torch.cuda.stream(self.alt_stream):
      projected_states_ba, _ = self.in_proj_ba(hidden_states)  # alt stream
  current_stream.wait_stream(self.alt_stream)                  # join
  ```
- BCG installs `_hooked_wait_stream` (`breakable_cuda_graph.py:112-136`) that intercepts `Stream.wait_stream` during capture and tracks side-stream membership. `_end_current_segment` auto-joins any forked-but-not-rejoined streams (line 390-406).

### 3.4 BCG capture / replay for GDN

- `BreakableCudaGraphBackend` (`runner_backend/breakable_cuda_graph_backend.py:56-257`). One `torch.cuda.CUDAGraph` per capture bucket (`self._graphs: Dict[ShapeKey, BreakableCUDAGraph]`).
- Capture buckets: `sorted(prefill_config.bs)` (`prefill_cuda_graph_runner.py:262`). Small buckets like `[4, 8, 12, 16, …]` (up to 8192) are the norm.
- **GDN core is an eager break under BCG**: `bcg_unified_linear_attention_with_output = eager_on_graph(True)(unified_linear_attention_with_output)` (`radix_linear_attention.py:159-161`); dispatched when `forward_mode.is_extend() AND TC-piecewise-ctx is set AND is_in_breakable_cuda_graph()` (line 96-103). Reachable on every GDN layer in a BCG-captured prefill.
- Segment count per prefill bucket on Qwen3.5-4B: 24 GDN eager breaks + segments around each full-attention layer + prologue/epilogue segments. **Several dozen segments per bucket**. Each segment `.replay()` has non-trivial launch overhead.
- Replay entry: `_execute_body_capture` (line 1540-1593) **monkey-patches `self.layer_model.forward = replay_layer_forward`** (line 1574-1575). `replay_layer_forward` copies live `input_embeds` into the static slot then triggers `self._graphs[shape_key].replay()`.
- Padding: `_pad_to_bucket` with `_MAX_PREFILL_CUDA_GRAPH_PADDING_FACTOR = 2` (line 126). GDN attention is eager, so it runs on `mixed_qkv[:real_num_tokens]` (padding does not blow up eager kernel work). But captured projections + RMSNormGated + out_proj do the full padded matmul.

### 3.5 TC piecewise vs BCG for GDN

- TC piecewise uses `torch.compile` with per-shape traces (`tc_piecewise_cuda_graph_backend.py:143-149, 151-218`).
- Alt-stream disabled under TC piecewise because dynamo cannot trace raw side-stream fork/join (would graph-break the compiled callable).
- BCG has no such tracer constraint (it captures raw CUDA API calls via `torch.cuda.CUDAGraph`) — **so the alt-stream branch is left enabled under BCG**. This is the load-bearing asymmetry.
- BCG has **no DeepStack warmup** — but Qwen3.5 ships `deepstack_visual_indexes = []` per the closed sub-track (NOT_APPLICABLE_QWEN35), so this is not a new concern for the GDN investigation.
- **No chunked-prefill CG capture path exists for BCG** (`_capture_chunked_prefix` is Full-only, `prefill_cuda_graph_runner.py:405-408`). Chunked prefill under BCG runs eagerly for state accumulation.

### 3.6 Recurrent state (mamba pool)

- Pool: `MambaPool` (`mem_cache/memory_pool.py:329-…`); per-layer `conv[i]` + `temporal[i]` tensors are pinned Parameters/Tensors — pointer-stable across BCG replays.
- `cache_indices` come from `forward_metadata.mamba_cache_indices`, recomputed per replay via `_prepare_forward_metadata_for_replay` (`prefill_cuda_graph_runner.py:956-997`) because `GDNAttnBackend` does **not** declare `use_captured_forward_metadata_for_breakable_cuda_graph=True`.
- Because GDN core is a graph break, state reads happen eagerly between segments — the eager `forward_extend` sees fresh `mamba_cache_indices`. **Cross-request state contamination risk is low** (R13.3).

### 3.7 Multimodal BCG allowlist

- `multimodal_breakable_cuda_graph_supported_model_archs` (`configs/model_config.py:1845-1848`) contains **only** `Qwen3_5ForConditionalGeneration` and `Qwen3_5MoeForConditionalGeneration`.
- Qwen3.5-4B in allowlist → BCG stays enabled by default (`server_args.py:4467-4471`).
- CUDA prefill default is `Backend.BREAKABLE` (`cuda_graph_config.py:109-118`).

### 3.8 Hypotheses to test (source-supported but require runtime evidence)

- **H12.1 (leading perf hypothesis)**: Alt-stream branch is active under BCG for every prefill bucket `< 1024` padded tokens; captured fork/join may or may not yield real overlap during CUDAGraph replay on H200 driver 595.71.05. If not overlapping, added launch + join overhead is pure cost.
- **H12.4 (padding waste)**: Small requests (< 512 tokens hitting a 1024 bucket) do ~2× wasted GEMM work in the captured projections + RMSNormGated + out_proj (attention is eager, so not affected).
- **H12.5 (segment count)**: 24 GDN eager-break points + attention break points = dozens of `.replay()` calls per prefill bucket. Compare launch overhead per replay vs TC piecewise's compiled-callable overhead.
- **H12.6 (per-replay buffer copies)**: `input_embeds` slot copy + several small metadata slot copies per replay.

### 3.9 Correctness risks to watch (source-supported but not confirmed)

- **R13.4 (alt-stream capture join integrity)**: `_hooked_wait_stream` (`breakable_cuda_graph.py:112-136`) discards `wait_stream(other)` when the "other" stream is not itself capturing (line 128-131). The alt-stream join `current_stream.wait_stream(self.alt_stream)` has `is_self_cap=True, is_other_cap=?` — depends on driver behavior for `_is_stream_capturing(self.alt_stream)`. If the check returns False when a with-block registered a capture on alt_stream, the join wait may be silently dropped, leaving an outstanding kernel on the alt stream without a captured join edge. **Concrete correctness risk to test on the smallest cell.**
- **R13.5 (chunked prefill state boundaries)**: No BCG chunked-prefill capture. If Qwen3.5-4B chunks a prefill across `execute()` calls, GDN state produced by chunk N reaches chunk N+1's eager `forward_extend` through the pool. Reachability under BCG needs confirmation.

```
SIGNAL: SIGNAL_GOOD
PHASE: Source & BCG audit
SUMMARY: Qwen3.5-4B execution path under BCG is fully reachable and mapped down to per-op source citations. Leading perf hypothesis (alt-stream branch active under BCG for padded bucket < 1024) has a concrete, testable source basis. One additional correctness risk (R13.4, alt-stream capture join integrity) identified. No central-premise concerns.
EVIDENCE: Direct source reads at frozen SHA 58974ca16c; citations in §3.1–§3.9 with path:line for every claim.
DECISION: Adopt H12.1 as the leading hypothesis for the smallest-cell test; add R13.4 to Gate-1 attention (compare token sequences carefully at padded-bucket boundary).
NEXT_ACTION: Complete validation audit consolidation; produce Phase 2 execution plan.
COMMIT: 2490057
```

---

## 4. Validation & methodology audit (design vs implementation)

### 4.1 Design coherence (docs agree with each other)
- 4-arm × 4×4 matrix consistent across README.md, hypothesis.md, validation_plan.md.
- Verdict labels identical across all docs and `gdn_verdict.py`.
- Arm-comparison thresholds identical between `validation_plan.md §6` and `gdn_verdict.py`.
- Charter reads like a rigorous investigation.

### 4.2 Design vs implementation gap
- **Gate 1** has both a comparison utility (`gdn_correctness.py`) AND a data-producing runner (`gdn_client.py`), but the runner doesn't ask for the data the comparison needs, and the comparison silently passes when data is absent. Two-layer false-pass hole (§2.1, §2.2).
- **Gates 2, 3, 4** have comparison utilities but no data-producing runners.
- **Perf side** has an Nsight wrapper but no extractor. Every scored sweep collapses to `AMBIGUOUS`.
- **Noise-floor pilot** predeclared (`validation_plan.md:103`) but has no code.
- **Sweep driver** absent. Runner is single-cell.

### 4.3 Distinguishability of failure modes

| Failure mode | Distinguishable today? |
|---|---|
| BCG replay dropping a kernel | No — needs Nsight extractor + `graph_breaks` column. |
| Alt-stream capture serializing | No — needs NVTX-tagged per-op ranges. |
| Graph-bucket contamination | No — no bucket discovery, no Gate-4 runner. |
| Chunked-prefill divergence | No — no `--chunked-prefill-size` wiring, no Gate-3 runner. |
| Ordinary numerical noise | Would be distinguishable if noise floor were measured; today it's fixed `0.05`. |

### 4.4 Measurement risk register (highest-severity items)

1. **Silent-tokenisation false pass** (Gate 1 whitespace fallback + missing logprobs) — a scored sweep would report Gate 1 PASS on essentially any output.
2. **Nsight overhead itself changes the code path** — no A0-without-nsys sub-sweep is scaffolded despite being called out in `validation_plan.md §7`.
3. **Client `e2e_ms` conflates prefill + decode** with `new_tokens=128`; comparing A0 vs A1 on e2e folds decode-side effects into a "prefill-BCG" claim.
4. **Server "cuda graph: True" reflects capture, not per-request replay** — only Nsight can distinguish captured-and-replayed from captured-but-fell-through-to-eager.
5. **`H_A` scoping bug in `gdn_verdict.py:145`** — iterates A2; false-positive risk.
6. **`n_timed=8` cannot compute a meaningful p95 per request**; between-request mean of p95s at n=8 has poor CI.
7. **Fixture cycling for Gate 2 isolation** — no code to pin a target `prompt_source_id` to both alone and batched runs.

### 4.5 What must exist before a scored sweep

Ordered by blocking severity:

1. Client `return_logprob=True` + `output_ids` capture.
2. `.nsys-rep → CSV` extractor.
3. Noise-floor pilot + tolerance formula wired.
4. Gate-1 hard-fail on whitespace fallback.
5. Gate-2 isolation runner (pin prompt_source_id).
6. Gate-3 `--chunked-prefill-size` plumbing + runner.
7. Gate-4 bucket-boundary discovery + runner.
8. 64-cell sweep driver with resume.
9. NVTX-tagged instrumentation (deferred until baseline motivates).
10. A0-without-nsys sub-sweep (Nsight overhead disclosure).
11. Post-teardown GPU idleness assertion in the runner.
12. `H_A` scoping fix (A2 → out).
13. `PARTIAL_SWEEP` label + amendment.
14. Fixture SHA verified by the client.
15. HF preflight offline mode.

---

## 5. Blocking gaps summary (Phase-4 backlog)

| # | Blocker | Fix locus |
|---|---|---|
| B1 | Gate 1 cannot fail — client no logprob/output_ids, verifier silent-fallback | `gdn_client.py` + `gdn_correctness.py` |
| B2 | No `.nsys-rep` extractor → perf verdict always `AMBIGUOUS` | new `scripts/extract_nsys_metrics.sh` |
| B3 | Gates 2/3/4 have no data-producing runner | new `scripts/run_isolation_pair.sh`, runner `--chunked-prefill-size` knob, log parser for bucket discovery |
| B4 | Preflight writes before env exports; `LD_PRELOAD`/`CUDA_VISIBLE_DEVICES` fields always null | `gdn_runner.sh` — reorder |
| B5 | Preflight skips 5 of 8 validation_plan.md §3 steps | `gdn_preflight.py` — add frozen-HEAD, sglang.__file__, model-revision hard-fail, LD_PRELOAD, lib versions |
| B6 | No post-teardown GPU idleness assertion | `gdn_runner.sh` — add post-check |
| B7 | Noise-floor pilot + tolerance formula not implemented | `gdn_correctness.py` — `--noise-floor` arg + `max(0.05, 3·nf)` |
| B8 | `H_A` iterates A2 (should be A1, A3 only) | `gdn_verdict.py:145` one-line fix |
| B9 | No `PARTIAL_SWEEP` label | `gdn_verdict.py` + `hypothesis.md` Amendment 2 |
| B10 | No sweep orchestrator | new `scripts/gdn_sweep.sh` or Python driver |
| B11 | No real-process lifecycle test | new test in `test_gdn_scaffolding.py` |
| B12 | Client silent I/O suppression + batched-response silent-truncation | `gdn_client.py` — hard-fail on both |
| B13 | Prompt-length materialisation char-heuristic (labels may lie) | `gdn_client.py` — query `/tokenize`, record actual token count |

Non-blocking cosmetic: `gdn_runner.sh:6` / `nsys_capture.sh:6` stale allowlist comments, fixture SHA not stamped in charter, HF preflight requires internet.

---

## 6. Preservation status

| Invariant | State |
|---|---|
| Fork `/data/sglang-fork` at `986c89e69c…` | Unchanged |
| Frozen SGLang at `58974ca16c…` | Unchanged, `git diff --stat` empty |
| DeepStack Attempts 01/02/03 verbatim | Unchanged |
| Protected R5C `audit_report.md` (M) | Untouched |
| Protected R6.3 orphan dir (??) | Untouched |
| No stray worktrees | Verified |
| No forbidden-string commits | Verified |
| `.gitignore` excludes `*/raw/` under gdn results | Verified |
| No `.nsys-rep`, `__pycache__`, or transient JSON tracked | Verified |

---

## 7. Major-blocker check

Cross-checked against operating-model criteria for "major blocker":

- Branch/repo does not match documented start? **No** — matches `2490057`.
- Required commits missing from origin? **No** — all 10 on origin.
- Frozen SGLang has unexplained diff? **No** — empty.
- GPU environment unsafe? **No** — GPUs 6 and 7 idle and available.
- Experiment risks another user's process? **No** — foreign PIDs on GPUs 0-5 respected via allowlist + foreign-PID guard.
- Model/dep/execution path structurally unavailable? **No** — smoke proved server bring-up.
- Audit invalidates central premise? **No** — source audit confirms all necessary code paths reachable on target hardware.
- Broad refactor required? **No** — 13 focused fixes.
- Destructive operation needed? **No**.
- Agent disagreement unresolvable from evidence? **No** — the three agents converge.

**No major blocker. Autonomous execution continues.**

---

## 8. Recommended immediate fixes (feed into Phase 2 plan)

Ordering respects dependency chain and minimises wasted work:

1. **Fix runner env-export ordering (B4)** — trivial but currently invalidates every preflight check.
2. **Strengthen preflight (B5, B6)** — add missing §3 steps + post-teardown GPU assertion.
3. **Client logprob + output_ids + `/tokenize` (B1a, B13, B12)** — one commit, unlocks Gate 1 and correct prompt-length labels.
4. **Correctness verifier hardening (B1b, B7)** — hard-fail on whitespace fallback; add `--noise-floor`.
5. **Verdict runner fixes (B8, B9)** — `H_A` scoping, `PARTIAL_SWEEP` label.
6. **Nsight extractor (B2)** — new script, test against smoke's captured `.nsys-rep`? No, we don't have one yet. Test against a first live capture on the smallest cell.
7. **Sweep orchestrator (B10)** — minimal driver that iterates (arm, prompt, batch, chunk_size). Optional in the ladder; not needed for the smallest-cell diagnosis.
8. **Gate-2/3/4 runners (B3)** — deferred until we have a green A0 baseline + smallest-cell A1/A2/A3 comparison; then targeted per-gate.
9. **Real-process lifecycle test (B11)** — deferred; can be added after tooling settles.

---

## 9. Consolidated signals

```
SIGNAL: SIGNAL_GOOD
PHASE: Repository & preservation audit
SUMMARY: Branch state clean, 10/10 commits on origin, preservation invariants held. GPU pool narrower than allowlist (6, 7 available) but adequate for a smallest-cell diagnosis ladder.
EVIDENCE: git status/log/rev-parse in both checkouts; nvidia-smi census.
DECISION: Proceed autonomously.
NEXT_ACTION: Consume harness + source + validation audits.
COMMIT: 2490057
```

```
SIGNAL: SIGNAL_GOOD
PHASE: Harness audit
SUMMARY: 13 blocking + several non-blocking defects, all with clear low-risk fixes. Design is coherent; implementation is a shell.
EVIDENCE: Direct read of all 8 scripts + tests + smoke evidence; cross-checked against charter docs.
DECISION: Fold into Phase 4 tooling hardening.
NEXT_ACTION: Produce Phase 2 execution plan grounded in these gaps.
COMMIT: 2490057
```

```
SIGNAL: SIGNAL_GOOD
PHASE: Source & BCG audit
SUMMARY: Full execution path mapped from launch_server down to per-op source citations. Leading perf hypothesis (alt-stream branch active under BCG for padded bucket < 1024) has a concrete testable basis. One correctness risk (R13.4 alt-stream join integrity) added to Gate-1 attention list.
EVIDENCE: Direct source reads at frozen SHA 58974ca16c with path:line for every claim.
DECISION: Adopt H12.1 as the leading hypothesis for smallest-cell diagnosis.
NEXT_ACTION: Produce plan; land tooling; run baseline; run smallest-cell A1/A2/A3.
COMMIT: 2490057
```

---

## Investigation log (append-only)

- 2026-08-03 phase 1 audit landed — this document.
- 2026-08-03 GPU pool at audit time: {6, 7} available; {0..5} foreign-occupied.
