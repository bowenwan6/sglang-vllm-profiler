#!/usr/bin/env bash
# Root-cause R4.A — production-shape sanity for the (X) fix.
#
# Same recipe as R3.B (E2a: image 720p c=1 n=32 warmup=30 output_len=128
# PCG on IPC on GPU 0), but with SGLANG_DEBUG_PCG_CALL_TRACE UNSET so
# we exercise the production path. Confirms (X) works without the
# diagnostic gate.
#
# Outputs:
#   raw/server.log, raw/bench.log, raw/bench.jsonl   NOT committed
#   bench_summary.txt                                committed
#   classification.txt                               committed
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R4_fix_X_validation/R4A_production_sanity"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003
FORK_PY=/data/sglang-fork/python

SERVER_LOG="$RAW/server.log"
BENCH_LOG="$RAW/bench.log"
BENCH_OUT="$RAW/bench.jsonl"
SUMMARY="$RESULTS/bench_summary.txt"
CLASSIFY="$RESULTS/classification.txt"

mkdir -p "$RAW"
rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT" "$SUMMARY" "$CLASSIFY"

export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST
# Production-shape: NO diagnostic gate
unset SGLANG_DEBUG_PCG_CALL_TRACE
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"

echo "=== launching sglang server (production sanity: no diagnostic gate) ==="
python3 -m sglang.launch_server \
  --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
  --attention-backend flashinfer --enforce-piecewise-cuda-graph \
  > "$SERVER_LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 300); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/get_model_info 2>/dev/null | grep -q 200; then
    echo "server READY after $((i*2)) s"; break; fi
  if ! kill -0 $SRV_PID 2>/dev/null; then echo "server DIED after ~$((i*2)) s"; break; fi
  sleep 2
done

if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/get_model_info 2>/dev/null | grep -q 200; then
  echo "=== bench (n=32, warmup=30, image 720p, c=1) ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat --base-url http://127.0.0.1:$PORT --model "$SNAP" \
    --dataset-name image --image-count 1 --image-resolution 720p \
    --image-format png --image-content random \
    --random-input-len 128 --random-output-len 128 --random-range-ratio 1.0 \
    --max-concurrency 1 --num-prompts 32 --warmup-requests 30 --seed 1 \
    --extra-request-body '{"temperature": 0, "top_p": 1}' \
    --output-details --output-file "$BENCH_OUT" \
    > "$BENCH_LOG" 2>&1 || true
fi

pkill -TERM -P $SRV_PID 2>/dev/null
kill -TERM $SRV_PID 2>/dev/null
sleep 5
pkill -9 -f "sglang.launch_server" 2>/dev/null
sleep 3
GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

if grep -q "PCG capture stream is not set, please check if runtime recompilation" "$SERVER_LOG"; then
  CLASS="PCG_CAPTURE_STREAM_ASSERT_STILL_FIRES"
elif grep -q "AssertionError" "$SERVER_LOG"; then
  CLASS="SERVER_ASSERTION_OTHER"
elif grep -q "Traceback (most recent call last)" "$SERVER_LOG"; then
  CLASS="SERVER_TRACEBACK_OTHER"
elif grep -q "Falling back to eager execution for this" "$SERVER_LOG"; then
  CLASS="OK_FALLBACK_TAKEN"
else
  CLASS="OK_NO_FALLBACK_NEEDED"
fi

[ -f "$BENCH_LOG" ] && awk '/Serving Benchmark Result/,/^==/{ print }' "$BENCH_LOG" > "$SUMMARY"

{
  echo "stage: R4.A"
  echo "purpose: production-shape sanity, no diagnostic gate"
  echo "snapshot: $(basename $SNAP)"
  echo "gpu: 0"
  echo "classification: $CLASS"
  echo "gpu_used_mib_after_teardown: $GPU_MIB"
  echo "fork_head: $(cd /data/sglang-fork && git rev-parse HEAD)"
  echo "pcg_assert_count:           $(grep -c "please check if runtime recompilation" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "fallback_warning_count:     $(grep -c "Falling back to eager execution for this" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "pcg_debug_trace_count:      $(grep -c "\[PCG_DEBUG\]" "$SERVER_LOG" 2>/dev/null || echo 0)  # expect 0 with gate unset"
} > "$CLASSIFY"

echo ""
echo "=== classification: $CLASS ==="
cat "$CLASSIFY"
