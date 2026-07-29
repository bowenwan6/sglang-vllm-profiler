#!/usr/bin/env bash
# Root-cause R1 — capture Dynamo recompile reason for the PCG capture-stream
# assertion on Qwen3-VL image+text path. Env-vars only, no source patches.
#
# Recipe mirrors E2a (the minimal reproducer from ../../conclusion.md):
#   image 720p, 1 image, c=1, n=32, warmup=30, output_len=128,
#   IPC on, --enforce-piecewise-cuda-graph, GPU 0, snapshot 0c351dd.
#
# Adds:
#   TORCH_LOGS=recompiles_verbose,dynamic,guards,graph_breaks
#   TORCHDYNAMO_VERBOSE=1
# so the server log contains the exact guard expression that triggers the
# recompile landing the piecewise submodule without a capture stream.
#
# Outputs:
#   raw/server.log        — full server stdout/stderr (NOT committed; per .gitignore)
#   raw/bench.log         — bench client stdout/stderr (NOT committed)
#   raw/bench.jsonl       — bench output details (NOT committed)
#   recompile_excerpt.log — trimmed recompile-reason lines (committed)
#   classification.txt    — terminal classification + GPU memory at end (committed)
#
# Reverse order — what stays out of git is raw/, what stays in is the two trimmed files.
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R1_dynamo_recompile_log"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003

SERVER_LOG="$RAW/server.log"
BENCH_LOG="$RAW/bench.log"
BENCH_OUT="$RAW/bench.jsonl"
EXCERPT="$RESULTS/recompile_excerpt.log"
CLASSIFY="$RESULTS/classification.txt"

mkdir -p "$RAW"
rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT" "$EXCERPT" "$CLASSIFY"

# Pin GPU 0, IPC on, no KAPI / profiler
export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST

# Dynamo verbose logging — the actual R1 instrumentation
export TORCH_LOGS="recompiles_verbose,dynamic,guards,graph_breaks"
export TORCHDYNAMO_VERBOSE=1

echo "=== launching sglang server (IPC=on, PCG=on, GPU=0, TORCH_LOGS=$TORCH_LOGS) ==="
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

# Wait up to 600 s for /get_model_info to respond OR PCG assertion to fire
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

# Run bench if server is up; otherwise skip (assertion may have fired during startup)
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

# Classify + extract recompile reasons
if grep -q "PCG capture stream is not set" "$SERVER_LOG"; then
  CLASS="PCG_CAPTURE_STREAM_ASSERT"
elif grep -q "AssertionError" "$SERVER_LOG"; then
  CLASS="SERVER_ASSERTION_OTHER"
elif grep -q "Traceback" "$SERVER_LOG"; then
  CLASS="SERVER_TRACEBACK_OTHER"
else
  CLASS="NO_SERVER_FAILURE_SEEN"
fi

# Trim recompile-related lines into the committed excerpt. Pattern set is
# defensive — recompile_verbose tends to vary by torch version.
grep -E "Recompiling|recompile_reason|GuardOnDataDependentSymNode|GUARD|TENSOR_MATCH|SHAPE_ENV|graph_break|fail_reason|PCG capture stream|AssertionError|qwen3_vl|cuda_piecewise|submod_" "$SERVER_LOG" \
  | head -400 > "$EXCERPT" || true

# Classification summary file
{
  echo "stage: R1"
  echo "snapshot: $(basename $(dirname $SNAP))/$(basename $SNAP)"
  echo "gpu: 0"
  echo "classification: $CLASS"
  echo "gpu_used_mib_after_teardown: $GPU_MIB"
  echo "server_log: $SERVER_LOG (NOT committed)"
  echo "bench_log:  $BENCH_LOG (NOT committed)"
  echo "bench_jsonl: $BENCH_OUT (NOT committed)"
  echo "recompile_excerpt: $EXCERPT (committed)"
  echo "torch_logs: $TORCH_LOGS"
  echo "torchdynamo_verbose: $TORCHDYNAMO_VERBOSE"
} > "$CLASSIFY"

echo ""
echo "=== classification: $CLASS ==="
echo "=== GPU after teardown: $GPU_MIB MiB ==="
cat "$CLASSIFY"
