#!/usr/bin/env bash
# Root-cause R3.B — validate fix shape (X) on the E2a recipe.
#
# Uses the patched sglang from /data/sglang-fork at branch
# fix/pcg-vlm-deepstack-warmup (HEAD 8a2dcb33a). The (X) patch extends
# the existing HIP eager fallback in CUDAPiecewiseBackend.__call__ to
# also cover CUDA, so the missing-capture-stream case falls back to
# entry.runnable (the inductor-compiled general-shape graph) instead
# of asserting.
#
# Keeps SGLANG_DEBUG_PCG_CALL_TRACE=1 so we can observe the fallback
# being taken (look for "Falling back to eager execution" + the
# [PCG_DEBUG] 'about to capture; stream=None' lines from R2's gate).
#
# Outputs:
#   raw/server.log               full server stdout/stderr (NOT committed)
#   raw/bench.log, raw/bench.jsonl                       (NOT committed)
#   pcg_fallback_excerpt.log     trimmed [PCG_DEBUG] + fallback lines (committed)
#   bench_summary.txt            client-side bench summary (committed)
#   classification.txt           terminal classification (committed)
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R3_fix_feasibility"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003
FORK_PY=/data/sglang-fork/python

SERVER_LOG="$RAW/server.log"
BENCH_LOG="$RAW/bench.log"
BENCH_OUT="$RAW/bench.jsonl"
EXCERPT="$RESULTS/pcg_fallback_excerpt.log"
SUMMARY="$RESULTS/bench_summary.txt"
CLASSIFY="$RESULTS/classification.txt"

mkdir -p "$RAW"
rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT" "$EXCERPT" "$SUMMARY" "$CLASSIFY"

export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST
export SGLANG_DEBUG_PCG_CALL_TRACE=1
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"

echo "=== fork verification ==="
python3 -c "import sglang; print('sglang.__file__=', sglang.__file__); print('expect: $FORK_PY/sglang/__init__.py')"
echo ""

echo "=== launching sglang server (IPC=on, PCG=on, GPU=0, fix branch fix/pcg-vlm-deepstack-warmup) ==="
python3 -m sglang.launch_server \
  --model-path "$SNAP" \
  --dtype bfloat16 \
  --port "$PORT" \
  --tp 1 \
  --attention-backend flashinfer \
  --enforce-piecewise-cuda-graph \
  > "$SERVER_LOG" 2>&1 &
SRV_PID=$!
echo "server pid=$SRV_PID  log=$SERVER_LOG"

for i in $(seq 1 300); do
  if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/get_model_info 2>/dev/null | grep -q 200; then
    echo "server READY after $((i*2)) s"
    break
  fi
  if ! kill -0 $SRV_PID 2>/dev/null; then
    echo "server DIED before ready after ~$((i*2)) s"
    break
  fi
  sleep 2
done

if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/get_model_info 2>/dev/null | grep -q 200; then
  echo "=== running bench (n=32, warmup=30, image 720p, c=1) ==="
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url http://127.0.0.1:$PORT \
    --model "$SNAP" \
    --dataset-name image \
    --image-count 1 \
    --image-resolution 720p \
    --image-format png \
    --image-content random \
    --random-input-len 128 \
    --random-output-len 128 \
    --random-range-ratio 1.0 \
    --max-concurrency 1 \
    --num-prompts 32 \
    --warmup-requests 30 \
    --seed 1 \
    --extra-request-body '{"temperature": 0, "top_p": 1}' \
    --output-details \
    --output-file "$BENCH_OUT" \
    > "$BENCH_LOG" 2>&1 || true
  echo "bench exit=$?"
fi

pkill -TERM -P $SRV_PID 2>/dev/null
kill -TERM $SRV_PID 2>/dev/null
sleep 5
pkill -9 -f "sglang.launch_server" 2>/dev/null
sleep 3
GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

# Outcome classification
if grep -q "PCG capture stream is not set" "$SERVER_LOG"; then
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

# Trim excerpt
grep -E "\[PCG_DEBUG\]|Falling back to eager execution|PCG capture stream|AssertionError|qwen3_vl|cuda_piecewise|Recompiling" "$SERVER_LOG" \
  | head -400 > "$EXCERPT" || true

# Bench summary: extract the final summary block from sglang.benchmark.serving
if [ -f "$BENCH_LOG" ]; then
  # The CLI prints a table at the end; grep the summary fields
  awk 'BEGIN{p=0} /Backend:|Successful requests:|Benchmark duration|Request throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Output token throughput|Total token throughput|Mean E2E/{p=1} p' "$BENCH_LOG" | head -50 > "$SUMMARY"
fi

{
  echo "stage: R3.B"
  echo "snapshot: $(basename $(dirname $SNAP))/$(basename $SNAP)"
  echo "gpu: 0"
  echo "classification: $CLASS"
  echo "gpu_used_mib_after_teardown: $GPU_MIB"
  echo "fork_branch: fix/pcg-vlm-deepstack-warmup"
  echo "fork_head: $(cd /data/sglang-fork && git rev-parse HEAD)"
  echo "pythonpath_prefix: $FORK_PY"
  echo "sglang_debug_pcg_call_trace: $SGLANG_DEBUG_PCG_CALL_TRACE"
  echo "server_log: $SERVER_LOG (NOT committed)"
  echo "bench_log:  $BENCH_LOG (NOT committed)"
  echo "bench_jsonl: $BENCH_OUT (NOT committed)"
  echo "pcg_fallback_excerpt: $EXCERPT (committed)"
  echo "bench_summary:        $SUMMARY (committed)"
  FALLBACK_HITS=$(grep -c "Falling back to eager execution for this" "$SERVER_LOG" 2>/dev/null || echo 0)
  ASSERT_HITS=$(grep -c "PCG capture stream is not set" "$SERVER_LOG" 2>/dev/null || echo 0)
  echo "fallback_warning_lines_in_server_log: $FALLBACK_HITS"
  echo "pcg_assert_lines_in_server_log:       $ASSERT_HITS"
} > "$CLASSIFY"

echo ""
echo "=== classification: $CLASS ==="
echo "=== GPU after teardown: $GPU_MIB MiB ==="
cat "$CLASSIFY"
