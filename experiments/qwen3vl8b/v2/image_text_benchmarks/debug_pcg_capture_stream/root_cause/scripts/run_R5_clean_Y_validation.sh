#!/usr/bin/env bash
# Root-cause R5 — validate the clean (Y) implementation on the E2a recipe.
#
# Fork branch fix/pcg-vlm-deepstack-warmup HEAD 1f19ecd1a carries:
#   - reverted R4.C naive (Y) prototype                  (ca11cf2aa)
#   - (X) extend HIP eager fallback to CUDA              (8a2dcb33a)
#   - clean (Y): thread-local force_warmup_deepstack_embeds
#     contextmanager + synthesis branch in mm_utils.py +
#     warmup loop in tc_piecewise_cuda_graph_backend.py    (1f19ecd1a)
#
# Same E2a recipe as R3.B / R4.A (image 720p c=1 n=32 warmup=30 PCG on
# IPC on GPU 0). Re-enables TORCH_LOGS=recompiles_verbose for ONE run
# so we can prove the runtime Dynamo recompile of qwen3_vl.forward is
# gone.
#
# Expected:
#   - server startup shows BOTH 'Compiling num tokens' and 'Compiling
#     MM num tokens' loops complete (the latter only with clean Y)
#   - classification = OK_NO_FALLBACK_NEEDED  (no recompile -> no
#     missing-stream cases -> no eager fallback)
#   - fallback_warning_count = 0
#   - dynamo_recompile_count for qwen3_vl.forward = 0 in the
#     bench-traffic portion of the log (startup recompiles for
#     sgl_kernel cache fill / shape sweep are still expected)
#   - bench TTFT clearly below R3.B / R4.A's ~104 ms and ideally
#     below the IMG_A_S0_ipc PCG-off baseline of 64.8 ms
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R5_clean_Y"
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
# Re-enable verbose recompile logging for this validation run only
export TORCH_LOGS="recompiles_verbose"
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"

echo "=== launching sglang server (clean Y; TORCH_LOGS=recompiles_verbose) ==="
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
  # Mark where the bench begins so we can split server-log events into
  # 'startup' vs 'inference traffic' for recompile counting.
  echo "[R5_MARKER] BENCH_START $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SERVER_LOG"
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
  echo "[R5_MARKER] BENCH_END $(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$SERVER_LOG"
fi

pkill -TERM -P $SRV_PID 2>/dev/null
kill -TERM $SRV_PID 2>/dev/null
sleep 5
pkill -9 -f "sglang.launch_server" 2>/dev/null
sleep 3
GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

# Outcome classification.
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

# Bench summary.
[ -f "$BENCH_LOG" ] && grep -A 50 "Serving Benchmark Result" "$BENCH_LOG" | head -45 > "$SUMMARY"

# Compile-log trim: keep both warmup loops + the bench marker + first inference recompiles.
grep -E "Compiling num tokens|Compiling MM num tokens|Capturing num tokens|R5_MARKER|Recompiling function" "$SERVER_LOG" \
  | tail -60 > "$COMPILE_TAIL" || true

# Split recompile counts by phase (startup vs after BENCH_START marker).
BENCH_START_LINE=$(grep -n "R5_MARKER. BENCH_START" "$SERVER_LOG" | head -1 | cut -d: -f1)
TOTAL_RECOMPILE=$(grep -c "Recompiling function" "$SERVER_LOG" 2>/dev/null || echo 0)
if [ -n "$BENCH_START_LINE" ]; then
  STARTUP_RECOMPILE=$(awk -v end=$BENCH_START_LINE 'NR<end' "$SERVER_LOG" | grep -c "Recompiling function" 2>/dev/null || echo 0)
  INFERENCE_RECOMPILE=$(awk -v start=$BENCH_START_LINE 'NR>=start' "$SERVER_LOG" | grep -c "Recompiling function" 2>/dev/null || echo 0)
else
  STARTUP_RECOMPILE="?"
  INFERENCE_RECOMPILE="?"
fi

{
  echo "stage: R5"
  echo "purpose: validate clean (Y) eliminates inference-time Dynamo recompile"
  echo "snapshot: $(basename $SNAP)"
  echo "gpu: 0"
  echo "classification: $CLASS"
  echo "gpu_used_mib_after_teardown: $GPU_MIB"
  echo "fork_head: $(cd /data/sglang-fork && git rev-parse HEAD)  (clean Y)"
  echo "pcg_assert_count:                              $(grep -c "please check if runtime recompilation" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "fallback_warning_count:                        $(grep -c "Falling back to eager execution for this" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "compiling_num_tokens_lines:                    $(grep -c "Compiling num tokens (num_tokens=" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "compiling_MM_num_tokens_lines:                 $(grep -c "Compiling MM num tokens (num_tokens=" "$SERVER_LOG" 2>/dev/null || echo 0)"
  echo "total_dynamo_recompile_count:                  $TOTAL_RECOMPILE"
  echo "startup_dynamo_recompile_count:                $STARTUP_RECOMPILE"
  echo "INFERENCE_dynamo_recompile_count_AFTER_BENCH:  $INFERENCE_RECOMPILE  # must be 0 for clean Y PASS"
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
