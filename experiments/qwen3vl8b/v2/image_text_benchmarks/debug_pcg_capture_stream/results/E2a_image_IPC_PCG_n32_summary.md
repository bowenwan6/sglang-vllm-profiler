# PCG debug — E2a (image_IPC_PCG_n32)

> Run: 2026-06-08 19:59 UTC  GPU=7  seed=1

> Description: Image+IPC+PCG ladder step 1 (n=32). Probes whether the Stage 4.2 PCG capture-stream assertion reproduces at a small sample size.

## Fixed-SGLang provenance

- `/data/sglang-pr` HEAD SHA: `62c505a196fd5bc997f478b0a7c6403ce655a838`
- `/data/sglang-pr` branch: `main`
- Merged fix `07f326c184` in history: **True**
- `sglang.__file__`: `/data/sglang-pr/python/sglang/__init__.py`
- `sglang.benchmark.datasets.common.__file__`: `/data/sglang-pr/python/sglang/benchmark/datasets/common.py`
- Fix marker: `FIX_OK`
- Fixed-path import gate: **True**

## Verdict

- Classification: **`PCG_CAPTURE_STREAM_ASSERT`** (expected `OK`) ⚠️
- bench_rc: 1 | elapsed: 29.7 s | server_wait_s: 480
- env signals: SGLANG_USE_CUDA_IPC_TRANSPORT='1'  KAPI_LOGLEVEL=None  KAPI_LOGDEST=None  HF_HUB_OFFLINE='1'  PYTHONPATH_prefix='/data/sglang-pr/python'  CUDA_VISIBLE_DEVICES='7'

## Case spec

```json
{
  "stage_id": "E2a",
  "label": "image_IPC_PCG_n32",
  "description": "Image+IPC+PCG ladder step 1 (n=32). Probes whether the Stage 4.2 PCG capture-stream assertion reproduces at a small sample size.",
  "dataset_kind": "image",
  "ipc_on": true,
  "pcg_on": true,
  "num_prompts": 32,
  "warmup": 30,
  "output_len": 128,
  "input_len": 128,
  "image_resolution": "720p",
  "image_content": "random",
  "image_format": "png",
  "image_count": 1,
  "range_ratio": 1.0,
  "server_wait_s": 480,
  "expected_classification": "OK"
}
```

## Server command

```bash
python3 -m sglang.launch_server --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer --enforce-piecewise-cuda-graph
```

## Bench command

```bash
python3 -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --dataset-name image --image-count 1 --image-resolution 720p --image-format png --image-content random --random-input-len 128 --random-output-len 128 --random-range-ratio 1.0 --max-concurrency 1 --num-prompts 32 --warmup-requests 30 --seed 1 --extra-request-body {"temperature": 0, "top_p": 1} --output-details --output-file /data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/results/raw/E2a_image_IPC_PCG_n32_bench.jsonl
```

## Server log excerpt (failure region or tail)

```text
[2026-06-08 19:59:36] INFO:     127.0.0.1:42104 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-06-08 19:59:36] INFO:     127.0.0.1:42118 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-06-08 19:59:36] INFO:     127.0.0.1:42120 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-06-08 19:59:36] INFO:     127.0.0.1:42126 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-06-08 19:59:36] Decode batch, #running-req: 30, #token: 0, token usage: 0.00, cuda graph: True, gen throughput (token/s): 32.56, #queue-req: 0
[2026-06-08 19:59:37] Prefill batch, #new-seq: 1, #new-token: 1, #cached-token: 1020, token usage: 0.00, #running-req: 0, #queue-req: 0, #pending-token: 0, cuda graph: True, input throughput (token/s): 0.79
[2026-06-08 19:59:37] INFO:     127.0.0.1:32868 - "POST /v1/chat/completions HTTP/1.1" 200 OK
[2026-06-08 19:59:37] Decode batch, #running-req: 1, #token: 1061, token usage: 0.00, cuda graph: True, gen throughput (token/s): 53.92, #queue-req: 0
[2026-06-08 19:59:37] Decode batch, #running-req: 1, #token: 1101, token usage: 0.00, cuda graph: True, gen throughput (token/s): 190.71, #queue-req: 0
[2026-06-08 19:59:38] Decode batch, #running-req: 1, #token: 1141, token usage: 0.00, cuda graph: True, gen throughput (token/s): 190.53, #queue-req: 0
[2026-06-08 19:59:38] Piecewise CUDA Graph failed with error: PCG capture stream is not set, please check if runtime recompilation happened
Piecewise CUDA Graph is enabled by default as an experimental feature.
To work around this error, add --disable-piecewise-cuda-graph to your launch command.
Please report this issue at https://github.com/sgl-project/sglang/issues/new/choose
[2026-06-08 19:59:38] Scheduler hit an exception: Traceback (most recent call last):
  File "/data/sglang-pr/python/sglang/srt/managers/scheduler.py", line 4038, in run_scheduler_process
    scheduler.run_event_loop()
  File "/data/sglang-pr/python/sglang/srt/managers/scheduler.py", line 1420, in run_event_loop
    dispatch_event_loop(self)
  File "/data/sglang-pr/python/sglang/srt/managers/scheduler.py", line 3903, in dispatch_event_loop
```
