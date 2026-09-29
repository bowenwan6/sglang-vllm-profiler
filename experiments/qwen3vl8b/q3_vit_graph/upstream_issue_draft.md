# Upstream issue drafts — ViT CUDA graph for Qwen3-VL (not submitted)

Two drafts for `sgl-project/sglang`, written from the Q3 measurements (`results/q3_report.md`,
`PLAN.md` "Outcome"). Line numbers refer to `upstream/main` @ `89e1316eae` (2026-09-28).
Draft 1 is a correctness bug and stands on its own; draft 2 is a performance/design report that
can be filed separately or folded into a discussion. Neither has been posted.

---

## Draft 1 — [Bug] `SGLANG_VIT_ENABLE_CUDA_GRAPH=1` changes Qwen3-VL outputs unless `--enable-precise-embedding-interpolation` is set

### Checklist

- [x] Searched existing issues for "ViT CUDA graph", "vit_cuda_graph_runner", "precise embedding interpolation".
- [x] Reproduced on current main (`89e1316eae`).
- [x] Minimal reproduction below.

### Describe the bug

With the ViT CUDA graph enabled, Qwen3-VL's vision encoder interpolates the learned position
embeddings differently from the eager path, so the model's outputs change:

- Eager `forward` (`python/sglang/srt/models/qwen3_vl.py:947`) calls
  `fast_pos_embed_interpolate_from_list` (L589), which always samples with `torch.linspace`
  (align-corners semantics, L596–599).
- The graph path `forward_with_cuda_graph` → `_prepare_graph_inputs` (L1115) calls the legacy
  `fast_pos_embed_interpolate` (L1140) unless `align_corners and ≥ 6 images` (L1133). The legacy
  path builds its coordinates in `_get_interpolation_indices` (L503), which follows
  `self.align_corners = enable_precise_embedding_interpolation` (L368). That flag defaults to
  `False`, i.e. half-pixel sampling (`(i + 0.5) * grid / size − 0.5`).

So with default flags the two execution modes use two different interpolation grids for the same
image. Only with `--enable-precise-embedding-interpolation` do both become linspace.

Measured on `Qwen/Qwen3-VL-8B-Instruct` (rev `0c351dd`), one H200, bf16, `--mm-attention-backend fa3`,
`--mm-feature-transport cuda_ipc`, `--skip-server-warmup`, a 336×336 and a 512×512 vertical-stripe PNG
with the prompt "Describe the colors in this image in order.", greedy, 48 tokens:

| comparison of the encoder output (merger + DeepStack, `[tokens, 16384]`) | 336² | 512² |
|---|---|---|
| graph vs eager, **default flags** | rel. Frobenius error **0.469**, max-elem 0.62 | **0.311**, 0.83 |
| graph vs eager, both with `--enable-precise-embedding-interpolation` | 0.058 | 0.069 |
| eager vs eager with the unfused rotary (the encoder's own implementation noise floor) | 0.021 | 0.072 |

Greedy text, default flags, 336² image: the graph run diverges from eager at **token 1**, where the
eager top-1/top-2 log-prob margin is 1.0 nat (14/45 tokens shared; second fixture diverges at token 9,
margin 0.25 nat). With the flag on both sides: 45/45 and 48/48 tokens identical, mean |Δlogprob|
0.006–0.008 nat. Text-only prompts are identical in every configuration, as expected.

### Reproduction

```bash
# eager
python -m sglang.launch_server --model-path Qwen/Qwen3-VL-8B-Instruct --dtype bfloat16 --port 30000 \
  --mm-attention-backend fa3 --mm-feature-transport cuda_ipc --skip-server-warmup --disable-radix-cache
# graph (default flags)
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 python -m sglang.launch_server ...same flags...
# graph + precise interpolation: outputs match eager again
SGLANG_VIT_ENABLE_CUDA_GRAPH=1 python -m sglang.launch_server ...same flags... --enable-precise-embedding-interpolation
```

```python
# stripe image + greedy request; compare the text (and, with a debug hook, the encoder output)
import base64, io, json, urllib.request
from PIL import Image
img = Image.new("RGB", (336, 336)); px = img.load()
pal = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
for x in range(336):
    for y in range(336):
        px[x, y] = pal[(x // 42) % 4]
buf = io.BytesIO(); img.save(buf, format="PNG"); b64 = base64.b64encode(buf.getvalue()).decode()
req = {"model": "Qwen/Qwen3-VL-8B-Instruct", "temperature": 0, "max_tokens": 48, "logprobs": True, "top_logprobs": 2,
       "messages": [{"role": "user", "content": [
           {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}},
           {"type": "text", "text": "Describe the colors in this image in order."}]}]}
r = urllib.request.Request("http://127.0.0.1:30000/v1/chat/completions", data=json.dumps(req).encode(),
                           headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(r))["choices"][0]["message"]["content"])
```

Run the script against each server and diff the outputs.

### Expected behavior

Graph mode is a performance switch; it should produce the same result as eager up to bf16 noise,
independent of `--enable-precise-embedding-interpolation`.

### Suggested fix

In `_prepare_graph_inputs`, use the same interpolation as the eager path (the linspace list/vectorized
implementation) regardless of the flag, or make the legacy path default to the eager semantics when the
flag is unset. A unit test that compares the encoder output of `forward` and `forward_with_cuda_graph`
on one image (relative error against the eager-vs-eager floor) would catch this class of drift.

### Environment

sglang main `89e1316eae` (2026-09-28), from source; torch 2.13.0+cu130, CUDA 13.0 toolkit, flashinfer
0.6.18, sgl-kernel 0.4.7; NVIDIA H200 143 GB, driver 595.71.05; Python 3.12; Ubuntu 24.04.

---

## Draft 2 — [Perf] Qwen3-VL ViT CUDA graph: net loss on large images (unfused rotary inside the graph), and exact-shape keys without eviction

### Summary

Measured on Qwen3-VL-8B, one H200, c=1, TTFT p50 over 200 prompts per cell, 3–4 A/B/B/A blocks per size,
one server per cell (eager vs `SGLANG_VIT_ENABLE_CUDA_GRAPH=1`, both with
`--enable-precise-embedding-interpolation` so the outputs match, see draft 1):

| image | ViT patches | TTFT eager → graph (ms) | effect |
|---|---|---|---|
| 256×256 | 256 | 46.0 → 32.4 | **−29 %** |
| 640×360 | 880 | 53.0 → 39.9 | −24 % |
| 512×512 | 1024 | 53.9 → 41.0 | −24 % |
| 640×640 | 1600 | 59.6 → 47.7 | −20 % |
| 1280×720 | 3520 | 87.0 → 87.6 | +1 % (no gain) |
| 1920×1080 | 8160 | 183.7 → 191.2 | **+4 % (slower)** |

Text-only requests are unaffected (−1 %, inside noise).

### Why the graph loses on large images

`VisionAttention` switches the rotary implementation when the graph is enabled
(`python/sglang/srt/layers/attention/vision.py:1484–1488`): `apply_rotary_pos_emb_native_eager`
instead of the `torch.compile`d `apply_rotary_pos_emb`, because the compiled kernel cannot be
specialised inside capture. The unfused version costs extra GPU time that grows with the patch count.
Measured by running the eager path with the unfused rotary forced on (GPU busy time inside the encoder
call, torch profiler):

| patches | 256 | 880 | 1024 | 1600 | 3520 | 8160 |
|---|---|---|---|---|---|---|
| extra GPU time of the unfused rotary (ms) | 0.75 | 1.2 | 1.3 | 1.8 | 3.7 | 8.3 |

Below ~1600 patches the encoder is launch-bound (eager: ≈ 600 kernel launches, 13–15 ms of un-overlapped
CPU time per call) and the graph recovers that, so the extra GPU work is hidden. At 3520 patches the two
cancel; at 8160 patches the encoder is already GPU-bound and the graph only adds the rotary cost.

### Exact-shape keys, never evicted

The graph key is `(patch count, cu_seqlens)` (`python/sglang/srt/multimodal/vit_cuda_graph_runner.py:146–159`);
each capture keeps a private memory pool (`torch.cuda.graph(graph)` at L223) and nothing is ever evicted
(only the Kimi-K3 runner has a capacity / min-hits policy). With resolutions drawn uniformly from
256–720 × 256–1280 px (300 requests, same sequence on both arms):

| | eager | graph |
|---|---|---|
| captures | — | 167 (hit rate 44 %) |
| mean TTFT | 64.8 ms | 94.7 ms (**+46 %**) |
| p99 TTFT | 94 ms | 189 ms |
| first-seen shapes, p50 penalty | — | +22 ms (capture p50 27 ms; first capture 4.6 s) |
| repeated shapes, p50 gain | — | −8.2 ms |
| GPU memory held by graphs | — | ≈ 15 GB |

### Suggestions

1. Run the compiled rotary's first specialisation before capture (at model load or in a warm-up call
   outside the graph), so graph mode does not have to fall back to the unfused kernel; that alone would
   turn the 720p/1080p loss into a small gain.
2. Pad the patch count to a bucket ladder, as the LM prefill graph does, so varied resolutions hit a
   bounded set of graphs; or add the Kimi-K3 runner's capacity / min-hits policy to the generic runner.
3. Document the regime: the feature pays for recurring shapes up to ~640×640 and costs time and memory on
   varied or large images (the Kimi-K3 cookbook already says "keep off for general serving").

### Environment

Same as draft 1. Method and per-cell data: `experiments/qwen3vl8b/q3_vit_graph/` in
`bowenwan6/sglang-vllm-profiler` (`results/q3_report.md`, `results/cells/`).
