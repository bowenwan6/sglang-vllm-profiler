Qwen3.5-4B BCG no-regression test (single arm: bcg_normal) on patched
upstream 198a3bc29b: verdict PASS. All 4 requests served; instrumentation
shows input_deepstack_embeds present=False on every LM entry, confirming
the fix's code path is entirely skipped for Qwen3.5's empty-DeepStack
config (num_deepstack_embeddings=0 → deepstack_replay_width=0 → no slot).
