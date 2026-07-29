#!/usr/bin/env bash
# R5.B — n=400 stretch for clean Y (steady-state TTFT)
set -u
ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R5_clean_Y/R5B_n400_stretch"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003
FORK_PY=/data/sglang-fork/python

SERVER_LOG="$RAW/server.log"; BENCH_LOG="$RAW/bench.log"; BENCH_OUT="$RAW/bench.jsonl"
SUMMARY="$RESULTS/bench_summary.txt"; CLASSIFY="$RESULTS/classification.txt"

mkdir -p "$RAW"
[ -f "$RAW/.gitignore" ] || printf '*.log\n*.jsonl\n*.txt\n' > "$RAW/.gitignore"
rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT" "$SUMMARY" "$CLASSIFY"

export CUDA_VISIBLE_DEVICES=0 SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE TORCH_LOGS
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"

python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
  --attention-backend flashinfer --enforce-piecewise-cuda-graph > "$SERVER_LOG" 2>&1 &
SRV_PID=$!
for i in $(seq 1 600); do
  curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/get_model_info 2>/dev/null | grep -q 200 && { echo "ready ${i}s"; break; }
  kill -0 $SRV_PID 2>/dev/null || { echo "DIED"; break; }
  sleep 2
done

python3 -m sglang.benchmark.serving --backend sglang-oai-chat --base-url http://127.0.0.1:$PORT --model "$SNAP" \
  --dataset-name image --image-count 1 --image-resolution 720p --image-format png --image-content random \
  --random-input-len 128 --random-output-len 128 --random-range-ratio 1.0 \
  --max-concurrency 1 --num-prompts 400 --warmup-requests 30 --seed 1 \
  --extra-request-body '{"temperature": 0, "top_p": 1}' --output-details --output-file "$BENCH_OUT" \
  > "$BENCH_LOG" 2>&1 || true

pkill -TERM -P $SRV_PID 2>/dev/null; kill -TERM $SRV_PID 2>/dev/null; sleep 5
pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 3
GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

if grep -q "PCG capture stream is not set, please check if runtime recompilation" "$SERVER_LOG"; then CLASS="PCG_CAPTURE_STREAM_ASSERT_STILL_FIRES"
elif grep -q "AssertionError" "$SERVER_LOG"; then CLASS="SERVER_ASSERTION_OTHER"
elif grep -q "Traceback (most recent call last)" "$SERVER_LOG"; then CLASS="SERVER_TRACEBACK_OTHER"
elif grep -q "Falling back to eager execution for this" "$SERVER_LOG"; then CLASS="OK_FALLBACK_TAKEN"
else CLASS="OK_NO_FALLBACK_NEEDED"; fi

grep -A 50 "Serving Benchmark Result" "$BENCH_LOG" | head -45 > "$SUMMARY"
{
  echo "stage: R5.B (n=400 stretch with clean Y)"
  echo "classification: $CLASS"
  echo "gpu_used_mib_after_teardown: $GPU_MIB"
  echo "fork_head: $(cd /data/sglang-fork && git rev-parse HEAD)"
  echo "pcg_assert_count:      $(grep -c "please check if runtime recompilation" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "fallback_warning_count: $(grep -c "Falling back to eager execution for this" "$SERVER_LOG" 2>/dev/null || echo 0)"
} > "$CLASSIFY"

echo "=== $CLASS ==="; cat "$CLASSIFY"; echo ""; cat "$SUMMARY"
