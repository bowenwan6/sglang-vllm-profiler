# Qwen3.5-4B GDN — Source Audit

> Direct-source review of `Qwen/Qwen3.5-4B`'s hybrid GDN layers and
> their interaction with SGLang's prefill BCG (breakable CUDA graph)
> and full-decode CUDA-graph paths. All line numbers resolve to the
> frozen SGLang checkout at
> `58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8`. Nothing here is a bug
> claim; runtime evidence lives in `validation_plan.md`.

## 1. Hybrid layer selection

`ALL_DECODER_LAYER_TYPES` at `python/sglang/srt/models/qwen3_5.py:1225-1228`:

```python
ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}
```

Per-layer selection at `qwen3_5.py:1359-1373` (inside `get_layer`):

```python
def get_layer(idx: int, prefix: str):
    layer_type = config.layers_block_type[idx]
    layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
    if layer_type == "attention":
        prefix = add_prefix("self_attn", prefix)
    else:
        prefix = add_prefix("linear_attn", prefix)
    ...
```

The per-layer type list is `config.layers_block_type` on the HF
config. On `Qwen/Qwen3.5-4B @ 851bf6e8`, this list determines the
attention/GDN interleaving pattern; the actual list is verified at
run time (see `provenance.md` §2).

## 2. GDN forward path

`Qwen3_5GatedDeltaNet.forward` at `qwen3_5.py:620-685`, in order:

1. **Input projection** — `_forward_input_proj(hidden_states)` at
   `qwen3_5.py:542-585`. Two parallel `ColumnParallelLinear`s
   (`in_proj_qkvz` output = `key_dim + key_dim + value_dim +
   value_dim`; `in_proj_ba` output = `num_v_heads + num_v_heads`),
   optionally on a cross-stream via `alt_stream` when
   `get_is_capture_mode()` is true, `seq_len < DUAL_STREAM_TOKEN_THRESHOLD`,
   and `_gdn_use_alt_stream` env var is on. `DUAL_STREAM_TOKEN_THRESHOLD`
   is `1024` on GPU except under `check_cuda_graph_backend(Phase.PREFILL,
   Backend.TC_PIECEWISE)` where it drops to `0` (`qwen3_5.py:551-558`).
   **BCG is not called out in that branch** — under BCG the
   threshold stays `1024`, so the alt-stream path is *active* for
   small prefills captured under BCG.
2. **Fused split/reshape** — `fused_qkvzba_split_reshape_cat_contiguous`
   at `qwen3_5.py:642-649` when `num_v_heads/num_k_heads ∈ {1, 2, 4}
   and not _is_npu`. Otherwise the Python fallback
   `fix_query_key_value_ordering` + `torch.cat` at `qwen3_5.py:520-540`
   (host-side reshape+cat, more launches).
3. **Linear-attention call** — `self.attn(forward_batch, mixed_qkv,
   a, b)` at `qwen3_5.py:662-667`. The `self.attn` is
   `RadixLinearAttention` from
   `python/sglang/srt/layers/radix_linear_attention.py`, constructed
   at `qwen3_5.py:333-346` with `conv_weights`, `A_log`, `dt_bias` —
   a stateful primitive over the mamba-style KV pool
   (`python/sglang/srt/mem_cache/mamba_checkpoint_pool.py`).
4. **Gated norm** — `RMSNormGated(core_attn_out, z)` at
   `qwen3_5.py:680`. `RMSNormGated` is in
   `python/sglang/srt/layers/layernorm.py`; runs over the value dim,
   gated by `z` (the last split from `in_proj_qkvz`).
5. **Output projection** — `self.out_proj(core_attn_out)` at
   `qwen3_5.py:684`, a `RowParallelLinear`.

Weight loaders on `in_proj_qkvz` / `in_proj_ba` are packed via
`_make_packed_weight_loader` (`qwen3_5.py:427-476`); the
`in_proj_qkvz` merges `[in_proj_qkv, in_proj_z]` and `in_proj_ba`
merges `[in_proj_b, in_proj_a]` per `packed_modules_mapping`
(`qwen3_5.py:1245-1250`). These fusions matter only at load time,
not per-forward, but they shape the kernel call sites.

## 3. GDN op focus list (for Nsight per-op accounting)

Named against `qwen3_5.py` symbols so the extractor can group NVTX
ranges cleanly.

| # | Op | Symbol / location | Notes |
|---|---|---|---|
| 3.1 | `in_proj_qkvz` GEMM | `ColumnParallelLinear` @ `qwen3_5.py:266-274` | Column-parallel; alt-stream branch. |
| 3.2 | `in_proj_ba` GEMM | `ColumnParallelLinear` @ `qwen3_5.py:276-283` | Column-parallel; alt-stream branch. |
| 3.3 | Cross-stream sync (alt-stream) | `alt_stream.wait_stream` / `torch.cuda.stream` @ `qwen3_5.py:567-572` | Active on BCG at `seq_len < 1024`. |
| 3.4 | Fused split/reshape (kernel) | `fused_qkvzba_split_reshape_cat_contiguous` @ `qwen3_5.py:642` | sgl-kernel fused op. |
| 3.5 | Split/reshape/cat (Python fallback) | `fix_query_key_value_ordering` @ `qwen3_5.py:520-540` + `torch.cat` @ `qwen3_5.py:660` | Multiple small launches. |
| 3.6 | `conv1d` (state-carrying) | `self.conv1d` @ `qwen3_5.py:254-263` (weights); consumed inside `RadixLinearAttention`. | 1D causal conv over KV state. |
| 3.7 | `RadixLinearAttention` | `self.attn` @ `qwen3_5.py:333-346`; call site @ `qwen3_5.py:662-667`. | Stateful; touches `A_log`, `dt_bias`, mamba pool. |
| 3.8 | Padding branch | `qwen3_5.py:674-678` (only when `core_attn_out.shape != z.shape`) | DP-Attn padding; conditional. |
| 3.9 | `RMSNormGated` | `qwen3_5.py:348-360` (init) + `qwen3_5.py:680` (call). | Uses `z` gate. |
| 3.10 | `out_proj` GEMM | `RowParallelLinear` @ `qwen3_5.py:362-372`. | Row-parallel; all-reduce at end when TP > 1 (here TP=1). |

The `alt_stream` sync (3.3) is the most likely BCG interaction
point at short prefills: it is guarded by `get_is_capture_mode()`
so the branch taken under `torch.cuda.CUDAGraph.capture_begin` may
differ from eager, and the wait/sync introduces cross-stream
dependencies inside the captured graph.

## 4. BCG capture path — what actually gets captured

`prefill_cuda_graph_runner.py` `_run_forward` at
`prefill_cuda_graph_runner.py:606-649` drives:

```python
layer_model.forward(input_ids, positions, forward_batch,
                    forward_batch.input_embeds)
```

`layer_model` here is the LM head module iterated one layer at a
time via `_execute_body_capture` (`prefill_cuda_graph_runner.py`
around line `1498-1519`) which owns the `replay_layer_forward`
closure. That closure copies `input_embeds` into a stable buffer
slot (registered at `cuda_graph_buffer_registry.py:867-877`) and
calls `self.backend.replay(shape_key, static_forward_batch,
**kwargs)` — the same replay bridge that the DeepStack sub-track
audited.

For GDN layers, `layer_model.forward(...)` eventually invokes the
per-layer `Qwen3_5LinearDecoderLayer.forward`, which calls
`self.linear_attn(...)`, which runs the ops in §2 and §3 above.
None of those ops are on the BCG-registry allowlist — only
`input_embeds` is stabilised. The **RadixLinearAttention state**
lives in the mamba pool, addressed by `forward_batch` metadata;
it is not a buffer-registry slot and its pointer stability across
BCG replay depends on the mamba pool's own contract, not on the
BCG buffer registry.

Open questions that the Nsight profile has to answer (do not
guess — measure):

- Is the entire `Qwen3_5GatedDeltaNet.forward` inside the captured
  graph, or does one of the ops (fused kernel dispatch, alt-stream
  sync, mamba-pool touch) force a break?
- Does the alt-stream branch produce cross-stream syncs that get
  captured but replayed serially, negating any overlap benefit
  under BCG?
- Does the mamba-pool state read/write get baked into the graph
  (pointer captured), and does replay across requests reuse the
  captured pointer safely?

## 5. Full-decode CUDA-graph interaction

The four-arm matrix separates prefill BCG from full-decode CG so
either can be attributed independently. Decode CG is the standard
`cuda_graph_runner.py` path (single-token replay per step, batch
along the token dim). GDN inside decode CG is a different capture
than GDN inside prefill BCG (different `seq_len`, different code
paths inside `RadixLinearAttention`). The two arms `A1` and `A2`
exist so a bottleneck can be attributed to one side and not the
other.

## 6. Env vars and gating flags relevant to GDN

- `SGLANG_GDN_QKVZ_BA_ALT_STREAM` (`qwen3_5.py:129`) — enables the
  alt-stream QKVZ/BA path (ROCm gated). CUDA is on by default via
  `_gdn_use_alt_stream`.
- `_gdn_use_alt_stream` — module-level bool derived from CUDA
  availability + env var; controls whether the alt-stream branch
  in `_forward_input_proj` is active.
- `check_cuda_graph_backend(Phase.PREFILL, Backend.TC_PIECEWISE)` —
  runtime check whether the piecewise backend is active; when true,
  the alt-stream threshold collapses to 0 (alt-stream disabled).
  **No equivalent short-circuit for BCG** is present, so BCG runs
  the alt-stream branch on short prefills.
- `get_is_capture_mode()` — CUDA graph capture context detection;
  gates the alt-stream branch inside `_forward_input_proj`.

The 4-arm sweep will record which env-var / gate values were
active per arm; discrepancies show up in the Nsight kernel-count
table.

## 7. What this audit is not

- Not a bug claim. Everything above is neutral source review.
- Not a fix. `plan.md` §8 rules out upstream source modification
  until the baseline profile pins a specific limitation.
- Not a claim that GDN is the bottleneck. Standard attention layers
  in the same hybrid stack may be the actual bottleneck; the 4-arm
  sweep and NVTX per-layer tagging will show whichever it is.
- Not an assertion that recurrent-state handling is faulty; that
  claim would need direct evidence from the mamba-pool state and
  replay-pointer stability, which is out of scope for the source
  audit and belongs to a follow-up investigation triggered only if
  the baseline profile points there.
