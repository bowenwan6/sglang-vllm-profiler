# PCG capture-stream debug — D1–D4 results

> Run: 2026-06-08 17:25 UTC  GPU=7  seed=1  num_prompts=2  warmup=0

> Tiny correctness probe — NO performance conclusions.

## Fixed-SGLang provenance

- `/data/sglang-pr` HEAD SHA: `62c505a196fd5bc997f478b0a7c6403ce655a838`
- `/data/sglang-pr` branch: `main`
- Merged fix `07f326c184` in history: **True**
- `sglang.__file__`: `/data/sglang-pr/python/sglang/__init__.py`
- `sglang.benchmark.datasets.common.__file__`: `/data/sglang-pr/python/sglang/benchmark/datasets/common.py`
- Fix marker (`get_available_multimodal_text_tokens` in `gen_mm_prompt`): `FIX_OK`
- Fixed-path import gate: **True**

## Per-case results

| case | label | dataset | IPC | PCG | classification | expected | match |
|---|---|---|---|---|---|---|---|
| D1 | image + IPC + PCG (reproduce 4.2 crash) | image | on | on | `OK` | `PCG_CAPTURE_STREAM_ASSERT` | ⚠️ |
| D2 | image + no IPC + PCG (factor IPC out) | image | off | on | `OK` | `PCG_CAPTURE_STREAM_ASSERT` | ⚠️ |
| D3 | image + IPC + no PCG (positive control, mirrors IMG_A_S0_ipc) | image | on | off | `OK` | `OK` | ✅ |
| D4 | text-only + PCG (does upstream PCG regress generally?) | text | off | on | `OTHER_FAILURE` | `OK` | ⚠️ |

## Decision-matrix interpretation

**D1 did not reproduce.** Stage 4.2 crash may be intermittent. Retry D1 with larger sample (e.g. 8 prompts) before drawing conclusions.

## Per-case detail

### D1 — image + IPC + PCG (reproduce 4.2 crash)

- Classification: **OK** (expected `PCG_CAPTURE_STREAM_ASSERT`)
- bench_rc: 0 | elapsed: 15.9 s | wait_s: 300
- completed=2 fails=0 non_empty=True vision_tok=1764 text_tok=281
- env: SGLANG_USE_CUDA_IPC_TRANSPORT='1'  KAPI_LOGLEVEL=None  KAPI_LOGDEST=None  PYTHONPATH_prefix='/data/sglang-pr/python'
- sample output[0]: `The image you've provided appears to be **completely corrupted or garbled**, likely due to a transmission error, file da`

### D2 — image + no IPC + PCG (factor IPC out)

- Classification: **OK** (expected `PCG_CAPTURE_STREAM_ASSERT`)
- bench_rc: 0 | elapsed: 16.4 s | wait_s: 300
- completed=2 fails=0 non_empty=True vision_tok=1764 text_tok=287
- env: SGLANG_USE_CUDA_IPC_TRANSPORT=None  KAPI_LOGLEVEL=None  KAPI_LOGDEST=None  PYTHONPATH_prefix='/data/sglang-pr/python'
- sample output[0]: `The image you've provided appears to be completely corrupted or filled with noise — it looks like a static-filled screen`

### D3 — image + IPC + no PCG (positive control, mirrors IMG_A_S0_ipc)

- Classification: **OK** (expected `OK`)
- bench_rc: 0 | elapsed: 11.0 s | wait_s: 240
- completed=2 fails=0 non_empty=True vision_tok=1764 text_tok=287
- env: SGLANG_USE_CUDA_IPC_TRANSPORT='1'  KAPI_LOGLEVEL=None  KAPI_LOGDEST=None  PYTHONPATH_prefix='/data/sglang-pr/python'
- sample output[0]: `The text you've provided appears to be a random, nonsensical string of characters, symbols, and words — likely generated`

### D4 — text-only + PCG (does upstream PCG regress generally?)

- Classification: **OTHER_FAILURE** (expected `OK`)
- bench_rc: 1 | elapsed: 8.3 s | wait_s: 300
- env: SGLANG_USE_CUDA_IPC_TRANSPORT=None  KAPI_LOGLEVEL=None  KAPI_LOGDEST=None  PYTHONPATH_prefix='/data/sglang-pr/python'
- server log excerpt (head of failure region):
```text
[2026-06-08 17:25:21] Capture piecewise CUDA graph end. Time elapsed: 22.90 s. mem usage=1.29 GB. avail mem=17.97 GB.
[2026-06-08 17:25:21] max_total_num_tokens=725101, chunked_prefill_size=8192, max_prefill_tokens=16384, max_running_requests=2048, context_len=262144, available_gpu_mem=17.97 GB
[2026-06-08 17:25:21] Tree cache initialized: source=default impl=RadixCache hybrid_swa=False hybrid_ssm=False hierarchical=False streaming_wrapped=False
[2026-06-08 17:25:21] INFO:     Started server process [2436472]
[2026-06-08 17:25:21] INFO:     Waiting for application startup.
[2026-06-08 17:25:21] Using default chat sampling params from model generation config: {'repetition_penalty': 1.0, 'temperature': 0.7, 'top_k': 20, 'top_p': 0.8}
[2026-06-08 17:25:21] INFO:     Application startup complete.
[2026-06-08 17:25:21] INFO:     Uvicorn running on http://127.0.0.1:30000 (Press CTRL+C to quit)
[2026-06-08 17:25:22] INFO:     127.0.0.1:33970 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-06-08 17:25:22] INFO:     127.0.0.1:33978 - "GET /model_info HTTP/1.1" 200 OK
[2026-06-08 17:25:25] INFO:     127.0.0.1:33800 - "GET /health HTTP/1.1" 503 Service Unavailable
[2026-06-08 17:25:26] Compiling a graph for dynamic shape takes 0.23 s
[2026-06-08 17:25:27] Prefill batch, #new-seq: 1, #new-token: 78, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 13.98
[2026-06-08 17:25:27] INFO:     127.0.0.1:33986 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-06-08 17:25:27] The server is fired up and ready to roll!
[2026-06-08 17:25:28] Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 0, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 0.95
[2026-06-08 17:25:29] INFO:     127.0.0.1:33816 - "GET /health HTTP/1.1" 200 OK
/usr/local/lib/python3.12/dist-packages/fastapi/routing.py:120: FastAPIDeprecationWarning: ORJSONResponse is deprecated, FastAPI now serializes data directly to JSON bytes via Pydantic when a return type or response model is set, which is faster and doesn't need a custom response class. Read more in the FastAPI docs: https://fastapi.tiangolo.com/advanced/custom-response/#orjson-or-response-model and https://fastapi.tiangolo.com/tutorial/response-model/
  response = await f(request)
[2026-06-08 17:25:35] INFO:     127.0.0.1:38924 - "GET /v1/models HTTP/1.1" 200 OK
```
