#!/usr/bin/env bash
# Root-cause R4.C — validate the (Y) prototype on E2a.
#
# Fork branch fix/pcg-vlm-deepstack-warmup HEAD 31cc8752f carries both:
#   (X) extend HIP eager fallback to CUDA           (8a2dcb33a)
#   (Y) prototype: pcg_warmup_multimodal_branch hook
#
# Recipe is the same E2a (image 720p c=1 n=32 warmup=30 PCG on IPC on
# GPU 0). With (X)+(Y) both active, expect:
#   - server startup runs the regular text-only Compile-num-tokens
#     loop AND a second 'Compiling MM num tokens' loop
#   - bench warmup -> bench: no Dynamo recompile of qwen3_vl.forward
#     for the input_deepstack_embeds branch (frame already compiled)
#   - fallback warning count = 0 (no missing-stream cases at inference)
#   - TTFT trends DOWN from R3.B's 103 ms toward something close to
#     the PCG-OFF baseline of 65 ms (best case) or at minimum stays
#     stable
#
# Outputs:
#   raw/server.log, raw/bench.log, raw/bench.jsonl   NOT committed
#   bench_summary.txt                                committed
#   classification.txt                               committed
#   compile_log_tail.log                             committed (~last 80 startup lines to confirm the MM warmup loop ran)
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R4_fix_X_validation/R4C_Y_prototype"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003
FORK_PY=/data/sglang-fork/python

SERVER_LOG="$RAW/server.log"
BENCH_LOG="$RAW/bench.log"
BENCH_OUT="$RAW/bench.jsonl"
SUMMARY="$RESULTS/bench_summary.txt"
CLASSIFY="$RESULTS/classification.txt"
COMPILE_TAIL="$RESULTS/compile_log_tail.log"

mkdir -p "$RAW"
rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT" "$SUMMARY" "$CLASSIFY" "$COMPILE_TAIL"

export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST
unset SGLANG_DEBUG_PCG_CALL_TRACE
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"

echo "=== launching sglang server (fix branch HEAD with X + Y prototype) ==="
python3 -m sglang.launch_server \
  --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
  --attention-backend flashinfer --enforce-piecewise-cuda-graph \
  > "$SERVER_LOG" 2>&1 &
SRV_PID=$!

for i in $(seq 1 600); do
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

[ -f "$BENCH_LOG" ] && grep -A 50 "Serving Benchmark Result" "$BENCH_LOG" | head -45 > "$SUMMARY"
# Capture the MM warmup loop announcement + last few startup lines
grep -E "Compiling num tokens|Compiling MM num tokens|Capturing num tokens|Engine .* started|Application startup complete" "$SERVER_LOG" \
  | tail -40 > "$COMPILE_TAIL" || true

{
  echo "stage: R4.C"
  echo "purpose: validate (Y) prototype eliminates the multimodal recompile + fallback"
  echo "snapshot: $(basename $SNAP)"
  echo "gpu: 0"
  echo "classification: $CLASS"
  echo "gpu_used_mib_after_teardown: $GPU_MIB"
  echo "fork_head: $(cd /data/sglang-fork && git rev-parse HEAD)  (X + Y prototype)"
  echo "pcg_assert_count:                          $(grep -c "please check if runtime recompilation" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "fallback_warning_count:                    $(grep -c "Falling back to eager execution for this" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "compiling_num_tokens_text_lines:           $(grep -c "Compiling num tokens (num_tokens=" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "compiling_MM_num_tokens_lines:             $(grep -c "Compiling MM num tokens (num_tokens=" "$SERVER_LOG" 2>/dev/null || echo 0)"
} > "$CLASSIFY"

echo ""
echo "=== classification: $CLASS ==="
cat "$CLASSIFY"
echo ""
echo "=== compile log tail ==="
[ -f "$COMPILE_TAIL" ] && cat "$COMPILE_TAIL"
echo ""
echo "=== bench summary ==="
[ -f "$SUMMARY" ] && cat "$SUMMARY"
