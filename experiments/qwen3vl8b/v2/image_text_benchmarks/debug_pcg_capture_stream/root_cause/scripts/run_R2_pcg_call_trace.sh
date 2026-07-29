#!/usr/bin/env bash
# Root-cause R2 — source-level instrumentation of
# CUDAPiecewiseBackend.__call__ to identify which CUDAPiecewiseBackend
# instance asserts on the missing capture stream and confirm it was
# never warmed up by PiecewiseCudaGraphRunner.capture().
#
# Sources sglang from the user's fork at /data/sglang-fork branch
# fix/pcg-vlm-deepstack-warmup (commit 2167b5f4d adds the debug gate)
# via PYTHONPATH so the container's installed sglang stays untouched.
#
# Same E2a recipe as R1; switches off the Dynamo verbose tracing (we
# already have it from R1) and switches on SGLANG_DEBUG_PCG_CALL_TRACE.
#
# Outputs:
#   raw/server.log                    — full server stdout/stderr (NOT committed)
#   raw/bench.log, raw/bench.jsonl    — (NOT committed)
#   pcg_call_trace_excerpt.log         — trimmed [PCG_DEBUG] lines (committed)
#   classification.txt                 — terminal classification (committed)
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R2_pcg_call_trace"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003
FORK_PY=/data/sglang-fork/python

SERVER_LOG="$RAW/server.log"
BENCH_LOG="$RAW/bench.log"
BENCH_OUT="$RAW/bench.jsonl"
EXCERPT="$RESULTS/pcg_call_trace_excerpt.log"
CLASSIFY="$RESULTS/classification.txt"

mkdir -p "$RAW"
[ -f "$RAW/.gitignore" ] || cat > "$RAW/.gitignore" <<'IGN'
*.log
*.jsonl
*.txt
IGN

rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT" "$EXCERPT" "$CLASSIFY"

# Pin GPU 0, IPC on, no KAPI / profiler
export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST

# R2 instrumentation: enable per-call PCG debug + source sglang from fork
export SGLANG_DEBUG_PCG_CALL_TRACE=1
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"
# No TORCH_LOGS — we already have that from R1

echo "=== fork verification ==="
python3 -c "import sglang; print('sglang.__file__=', sglang.__file__); print('expect: $FORK_PY/sglang/__init__.py')"
echo ""

echo "=== launching sglang server (IPC=on, PCG=on, GPU=0, SGLANG_DEBUG_PCG_CALL_TRACE=1) ==="
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

# Wait up to 600 s for /get_model_info or PCG assertion
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

# Tear down
pkill -TERM -P $SRV_PID 2>/dev/null
kill -TERM $SRV_PID 2>/dev/null
sleep 5
pkill -9 -f "sglang.launch_server" 2>/dev/null
sleep 3
GPU_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)

# Classify + extract PCG_DEBUG lines + assertion stack
if grep -q "PCG capture stream is not set" "$SERVER_LOG"; then
  CLASS="PCG_CAPTURE_STREAM_ASSERT"
elif grep -q "AssertionError" "$SERVER_LOG"; then
  CLASS="SERVER_ASSERTION_OTHER"
elif grep -q "Traceback" "$SERVER_LOG"; then
  CLASS="SERVER_TRACEBACK_OTHER"
else
  CLASS="NO_SERVER_FAILURE_SEEN"
fi

# Trim: keep all [PCG_DEBUG] lines + the assertion stack
grep -E "\[PCG_DEBUG\]|PCG capture stream|AssertionError|qwen3_vl\.py|cuda_piecewise|submod_|Recompiling" "$SERVER_LOG" \
  > "$EXCERPT" || true

{
  echo "stage: R2"
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
  echo "pcg_call_trace_excerpt: $EXCERPT (committed)"
  PCG_DEBUG_LINES=$(grep -c "\[PCG_DEBUG\]" "$SERVER_LOG" 2>/dev/null || echo 0)
  echo "pcg_debug_lines_total: $PCG_DEBUG_LINES"
  echo "pcg_debug_lines_kept_in_excerpt: $(wc -l < "$EXCERPT")"
} > "$CLASSIFY"

echo ""
echo "=== classification: $CLASS ==="
echo "=== GPU after teardown: $GPU_MIB MiB ==="
cat "$CLASSIFY"
