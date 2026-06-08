# PCG debug — E1 (text_autobench_PCG_control)

> Run: 2026-06-08 19:47 UTC  GPU=7  seed=1

> Description: Text-only Case A-like + PCG on /data/sglang-pr upstream main. Confirms whether upstream PCG itself regresses on text-only Qwen3-VL. Uses --dataset-name autobench with a local JSONL so the bench client never touches the HF Hub (replaces the D4 --dataset-name random path that failed in offline mode).

## Fixed-SGLang provenance

- `/data/sglang-pr` HEAD SHA: `62c505a196fd5bc997f478b0a7c6403ce655a838`
- `/data/sglang-pr` branch: `main`
- Merged fix `07f326c184` in history: **True**
- `sglang.__file__`: `/data/sglang-pr/python/sglang/__init__.py`
- `sglang.benchmark.datasets.common.__file__`: `/data/sglang-pr/python/sglang/benchmark/datasets/common.py`
- Fix marker: `FIX_OK`
- Fixed-path import gate: **True**

## Verdict

- Classification: **`OK`** (expected `OK`) ✅
- bench_rc: 0 | elapsed: 14.2 s | server_wait_s: 360
- completed=8 fails=0 non_empty=True median_ttft_ms=15.130484942346811
- env signals: SGLANG_USE_CUDA_IPC_TRANSPORT=None  KAPI_LOGLEVEL=None  KAPI_LOGDEST=None  HF_HUB_OFFLINE='1'  PYTHONPATH_prefix='/data/sglang-pr/python'  CUDA_VISIBLE_DEVICES='7'

## Case spec

```json
{
  "stage_id": "E1",
  "label": "text_autobench_PCG_control",
  "description": "Text-only Case A-like + PCG on /data/sglang-pr upstream main. Confirms whether upstream PCG itself regresses on text-only Qwen3-VL. Uses --dataset-name autobench with a local JSONL so the bench client never touches the HF Hub (replaces the D4 --dataset-name random path that failed in offline mode).",
  "dataset_kind": "text_autobench",
  "dataset_path": "/data/sglang-vllm-profiler/datasets/qwen3vl8b/caseA_short.jsonl",
  "ipc_on": false,
  "pcg_on": true,
  "num_prompts": 8,
  "warmup": 0,
  "output_len": 128,
  "server_wait_s": 360,
  "expected_classification": "OK"
}
```

## Server command

```bash
python3 -m sglang.launch_server --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --dtype bfloat16 --port 30000 --tp 1 --attention-backend flashinfer --enforce-piecewise-cuda-graph
```

## Bench command

```bash
python3 -m sglang.bench_serving --backend sglang-oai-chat --base-url http://127.0.0.1:30000 --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b --dataset-name autobench --dataset-path /data/sglang-vllm-profiler/datasets/qwen3vl8b/caseA_short.jsonl --max-concurrency 1 --num-prompts 8 --warmup-requests 0 --seed 1 --extra-request-body {"temperature": 0, "top_p": 1} --output-details --output-file /data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/results/raw/E1_text_autobench_PCG_control_bench.jsonl
```

## Sample bench output

- req1: `Dường như bạn đã dán một đoạn văn bản hỗn hợp, bao gồm nhiều từ, ký hiệu, tên riêng, mã nguồn, và thậm chí cả ký tự Unic`
- req2: `It seems like your message is a mix of random characters, code fragments, non-English words, symbols, and possibly corru`
