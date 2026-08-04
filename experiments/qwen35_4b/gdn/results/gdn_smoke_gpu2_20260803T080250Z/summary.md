# GDN smoke — GPU 2 — 2026-08-03T08:02:50Z

**Purpose.** End-to-end smoke test of the GDN sub-track scaffolding
(runner → server_launcher → SGLang server → /health → client → 4
timed requests → PGID-scoped teardown) on a real GPU under the
frozen SGLang checkout. **This is not a scored cell**; it is a
plumbing validation before the actual A0 baseline sweep begins.

## Configuration

- Arm: **A0** (`eager_eager` — `--cuda-graph-backend-prefill=disabled --cuda-graph-backend-decode=disabled`).
- Prompt length: 128 target tokens (materialised to 448 chars per
  the golden fixture's deterministic stretch).
- Batch size: 1.
- Warmup: 2 requests (discarded).
- Timed: 4 requests.
- Max new tokens per request: 128.
- Fixture: `experiments/qwen35_4b/gdn/fixtures/gdn_prompts.jsonl`
  (sha256 `8a660d94982a3965f1f97a8a0ad3998f22bdbc486c80329d20814f0ad8ba035f`).
- GPU: 2 (UUID `GPU-e19275bf-adc5-9fc3-42d7-9a3d4b666b81`).
- Frozen SGLang: `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` under
  scratchpad `sglang_checkout/sglang`.
- Model: `Qwen/Qwen3.5-4B` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`.
- Instrumentation: `gdn_instrumentation.py` (baseline no-op).

## Result

**PASS** for scaffolding. Every plumbing check the smoke was
designed to exercise held:

| Check | Result |
|---|---|
| Preflight (GDN config fields under `text_config`) | OK (24 linear + 8 full attention layers, heads_ratio=2 → fused split path) |
| Server bring-up on GPU 2 | Ready after **37 s** |
| SGLang initialised hybrid GDN backend | Confirmed in server log: `Using hybrid linear attention backend for hybrid GDN models` / `GDN kernel dispatcher: decode=TritonGDNKernel, extend=TritonGDNKernel` |
| Prefill CUDA graph really disabled for A0 | Confirmed in server log: `Disable prefill CUDA graph because cuda_graph_config resolved prefill.backend='disabled'` |
| Client requests | 4/4 status 200, output_len_tokens=128 each |
| e2e latency consistency | All 4 within **[2994, 3010] ms** at prompt=128 batch=1 (0.5% spread) |
| PGID-scoped teardown | GPU 2 memory 0 MiB post-run |
| Foreign-PID guard | 0 foreign PIDs pre and post |
| `/data/sglang-fork` HEAD unchanged | `986c89e69c25882ab6f3d396f8eb306f38f2c8d2` |
| Frozen SGLang HEAD unchanged | `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8` (empty `git diff --stat`) |
| All 8 GPUs clean post-run | 0 MiB across GPUs 0-7 |

## Per-request timings (A0 eager, prompt=128 tokens, 128 new tokens)

| request_id | seed | e2e_ms |
|---|---|---|
| `A0_p128_b1_r2_i0` | `g1_short_qa_c256` | 2998.1 |
| `A0_p128_b1_r3_i0` | `g1_short_qa_c4096` | 2994.4 |
| `A0_p128_b1_r4_i0` | `g2_short_code_c256` | 3009.6 |
| `A0_p128_b1_r5_i0` | `g2_short_code_c4096` | 2995.3 |

Mean: ~2999 ms · SD: ~7 ms · this is the eager baseline latency
floor at this cell shape.

## Fixes landed during the smoke (all pushed)

The smoke surfaced three real bugs in the scaffolding before the
runner would launch cleanly. Each was fixed and committed:

1. **`fix(qwen35): read GDN config fields from text_config`
   (`66d91cd`).** `gdn_preflight.py` had been reading top-level
   config fields, but Qwen3.5 nests the language-model config under
   `text_config`; also the HF-side layer-type field is `layer_types`
   (not `layers_block_type`). Preflight failed with `MISSING_FIELDS`
   on every GDN field.
2. **`fix(qwen35): use canonical --cuda-graph-backend-{prefill,decode}
   flags` (`271a666`).** The runner had used flag names that don't
   exist on the frozen SHA (`--disable-breakable-cuda-graph`,
   `--disable-cuda-graph-padding`). SGLang exposes
   `--cuda-graph-backend-prefill=<mode>` /
   `--cuda-graph-backend-decode=<mode>` as canonical; the arm-flag
   mapping now uses those explicitly per arm.
3. **`fix(qwen35): capture setsid'd launcher PID directly, not
   subshell PID` (`5736f96`).** The runner wrapped the launcher in
   a subshell (`( setsid python ... ) &`), so `$!` captured the
   subshell PID rather than the setsid'd python. The readiness
   check `kill -0 -"$SERVER_PGID"` misfired once the subshell
   exited, the runner declared "server died," and its EXIT trap
   killed the wrong PGID — leaving the entire SGLang subtree
   orphaned on the GPU (~131 GiB). Fix removes the subshell so
   `$!` is the setsid'd python PID (== PGID); health check uses
   the positive PID.

## What this smoke does *not* establish

- Not a scored cell. `n_timed=4` is below the plan's `n_timed=8`.
- No baseline noise floor calibration (that needs the
  `eager_normal`-vs-`eager_normal` self-repeat pair from the real
  A0 baseline, not four different prompts).
- No correctness gates evaluated. `gdn_client.py` does not currently
  request `output_ids` / `output_logprobs` from the server — the
  server returned `output_ids` in the smoke's ad-hoc `curl` probe
  (see the direct-launch log below), but `gdn_client.py` needs a
  small update to opt in on the timed path. Follow-up.
- No Nsight capture (baseline `nsys` sweep is the next step).
- No arm comparison (only A0 was run; A1/A2/A3 pending).

## Files in this attempt

- `metadata.json` — attempt provenance.
- `preflight.json` — GDN preflight snapshot (all `OK`).
- `gpu_pre.txt` — GPU 2 pre-state (0 MiB / 0 %).
- `records_A0_p128_b1.jsonl` — 4 timed per-request records.
- `client_A0_p128_b1.log` — client stdout.
- `runner_stdout.log` / `runner_stderr.log` — runner control-flow.
- `raw/server_A0_p128_b1.log` — full SGLang server log (gitignored
  per `../.gitignore`; kept locally for post-hoc inspection).
