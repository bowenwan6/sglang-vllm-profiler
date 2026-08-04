Qwen3.5-4B BCG DeepStack INFRA_CHECK attempt
`infracheck_gpu7_20260801T012122Z` on the authorised alternate GPU 7:
PASS. Every provenance hard pin matched (frozen SGLang
`58974ca16...`, imported sglang inside frozen, `Qwen3_5ForConditionalGeneration`
@ `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, image fixture,
`/data/sglang-fork` still `986c89e69...`). The prior kernel blocker
(`sglang-kernel 0.4.4`) is cleared at `0.4.5`. The server came up in
129 s with the breakable prefill CUDA graph (BCG) backend, captured 58
shape buckets, exercised BCG in warmup (`cuda graph: True`), reported
`The server is fired up and ready to roll!`, and tore down cleanly --
the runner signalled only its own PGID and GPU 7 memory returned to 4
MiB. The 11 foreign compute processes on other GPUs were unchanged
pre-vs-post. One caveat is recorded for Step 5: SGLang spawns its
scheduler / model-worker subprocesses with `mp.set_start_method('spawn',
force=True)`, so the branch instrumentation installed in the launcher
parent does not propagate to workers; per-batch BCG-vs-eager
attribution will therefore rely on SGLang's own `cuda graph: True/False`
server-log line rather than on the request-level JSONL events, and the
`eager_zero_deepstack` ablation will behave identically to
`eager_normal`, weakening attribution power for the `FAIL_BCG_DEEPSTACK`
zero-DeepStack signature path only. INFRA_CHECK itself is unaffected.
