# INFRA_CHECK verdict — PASS

Human-readable verdict for `infracheck_gpu7_20260801T012122Z`. Companion
machine-readable data lives in `metadata.json`; raw evidence lives in
`raw/`.

## Scope

Step 4 per `experiments/qwen35_4b/validation_plan.md` §9.4: bring up
the smallest real Qwen3.5-4B server configuration under the frozen
SGLang checkout on the authorised GPU 7, verify BCG capture, tear down
cleanly, confirm no foreign PID was affected and GPU 7 memory returns
to the qualifying state. No client requests are fired in INFRA_CHECK
mode.

## Verdict

`PASS` on every INFRA_CHECK evidence layer:

| Layer | Evidence |
|---|---|
| Provenance hard pins | `frozen_sglang_checkout=PIN_MATCH` (SHA `58974ca16…`), `imported_sglang_path=INSIDE_FROZEN`, `hf_model=PIN_MATCH` (`Qwen3_5ForConditionalGeneration` @ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`), `image_fixture=PIN_MATCH`, `sglang_fork_unchanged=PIN_MATCH` (`986c89e69…`). |
| Kernel gate | `sglang-kernel==0.4.5` satisfies the frozen SGLang `assert_pkg_version(sglang-kernel, ≥0.4.5)`. The prior attempt `infracheck_gpu7_20260801T004841Z` blocked here with `0.4.4`. |
| CUDA / libcuda init | `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05` applied; `Init torch distributed ends. elapsed=0.08 s`. |
| Model + weights load | `Load weight end. elapsed=3.90 s, type=Qwen3_5ForConditionalGeneration, avail mem=130.51 GB, mem usage=8.62 GB.` |
| Multimodal readiness | `Multimodal data loading enabled with 16 worker threads (auto)`, `Multimodal processor concurrency enabled with 2 isolated worker threads (auto)`, `Using fa3 as multimodal attention backend`. |
| KV / mamba cache alloc | `KV Cache is allocated. dtype: torch.bfloat16, #tokens: 1651079, K size: 25.19 GB, V size: 25.19 GB`; `Memory pool end. avail mem=33.84 GB`. |
| BCG capture banner | Server args show `cuda_graph_config.prefill=PhaseConfig(backend='breakable', max_bs=8192, bs=[4..8192] × 58 buckets, tc_compiler='eager')`. 58 `Capturing num tokens (num_tokens=…)` progress lines observed for buckets from 8192 down to 4. |
| Warmup exercised BCG | Server warmup logged `Prefill batch, #new-seq: 1, #new-token: 80, …, cuda graph: True` — a real BCG replay on the eager `tc_compiler` breakable backend. |
| Server ready | `The server is fired up and ready to roll!` at 01:23:50 UTC; `/health` returned 200 at 129s after launch. |
| Clean teardown | The runner signalled only its own PGID (770564); the recorded pid list contains only descendant workers. GPU 7 memory dropped from 111,760 MiB peak back to 4 MiB post-teardown. |
| No foreign-PID impact | Pre-run and post-run `nvidia-smi --query-compute-apps` snapshots contain the same 11 foreign compute processes on GPUs 0/1/2/3/4/5; none appear on GPU 7 at any point. |

## Instrumentation propagation caveat (recorded for Step 5 planning)

The branch-owned instrumentation ran in the launcher parent process
(pid 770564) and successfully patched
`prefill_cuda_graph_runner.PrefillCudaGraphRunner._execute_body_capture`,
`model_runner.ModelRunner.forward`, and
`mm_utils.general_mm_embed_routine` — five `install_start` /
`patch_ok` / `install_done` events landed in
`raw/instrumentation_bcg_normal.jsonl`.

However, SGLang bootstraps its scheduler and model-worker as `spawn`
subprocesses (`engine.py:1621 mp.set_start_method("spawn", force=True)`).
`spawn` re-imports all modules in the child, which means the parent's
monkey-patches do not carry over. Request-level events
(`bcg_execute_body_enter`, `bcg_replay_layer_forward_enter`,
`lm_forward_input_deepstack`, `model_runner_forward_enter`) are
therefore **not expected to appear** in the JSONL when real requests
are issued. This does not affect INFRA_CHECK (which fires no requests)
but limits the tensor-level attribution planned for Step 5.

Step 5 mitigations:

1. Per-batch path attribution is still authoritative: every prefill
   batch line in the server stderr contains `cuda graph: True/False`.
   `True` means BCG replay was used for that batch; `False` means the
   eager runner was used.
2. Greedy-text divergence across the eager-normal and bcg-normal legs
   is captured client-side and needs no in-process instrumentation.
3. The `eager_zero_deepstack` ablation depends on the interceptor
   running in the worker; without the propagation fix it will behave
   identically to `eager_normal`. This weakens attribution power for
   the `FAIL_BCG_DEEPSTACK` verdict via the "zero-DeepStack signature"
   path, but does not affect the direct-divergence path or the other
   three verdict outcomes.

Step 5 will proceed and record the verdict based on server-log path
attribution plus client-side greedy-text comparison. If the primary
signal is ambiguous, the ablation-propagation gap will be reflected
in the verdict rationale.

## GPU acquisition (Step 3 recap)

Under Amendment 1 of the validation plan, GPU 7 became an authorised
alternate on 2026-08-01 and the 10-continuous-minute idle requirement
was waived when the target GPU is already qualifying. At Step-3
recheck, GPU 7 held 4 MiB / 0 % utilisation / 0 compute apps for its
UUID `GPU-da0d9c21-…`; the runner proceeded immediately. Post-run,
GPU 7 returned to the same state.

## What comes next

Step 5 (correctness/path validation) — the predeclared controls
`eager_normal`, `eager_zero_deepstack`, and `bcg_normal`, plus the
text-only eager/BCG controls, scored via
`experiments/qwen35_4b/scripts/verdict.py` and interpreted against
the criteria in `validation_plan.md` §6. No fix, no upstream issue.
