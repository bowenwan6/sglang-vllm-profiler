#!/usr/bin/env bash
# Qwen3-VL MoE smoke for the DeepStack BCG replay slot.
#
# Serves Qwen3VLMoeForConditionalGeneration (deepstack_visual_indexes=[8,16,24],
# so the replay path is genuinely exercised) once under breakable prefill CUDA
# graph and once eager, sends the same deterministic image prompt greedily, and
# compares the two completions.
#
# Runs the PR branch via PYTHONPATH, not the installed sglang.
set -uo pipefail

GPU_ID="${GPU_ID:-4}"
PORT="${PORT:-31337}"
FORK=/data/sglang-fork
MODEL=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-A3B-Instruct/snapshots/9c4b90e1e4ba969fd3b5378b57d966d725f1b86c
OUT="${OUT:-/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/moe_smoke}"
mkdir -p "$OUT"

run_arm () {
  local arm="$1"; shift
  echo "=== arm=$arm ==="
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$FORK/python" \
  LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05${LD_PRELOAD:+:$LD_PRELOAD}" \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --port "$PORT" --host 127.0.0.1 \
      --mem-fraction-static 0.85 \
      --disable-radix-cache \
      "$@" > "$OUT/server_$arm.log" 2>&1 &
  local pid=$!

  for _ in $(seq 1 180); do
    sleep 5
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED (arm=$arm)"; tail -30 "$OUT/server_$arm.log"; return 1; }
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "HEALTH TIMEOUT (arm=$arm)"; return 1; }

  PORT="$PORT" OUT="$OUT" ARM="$arm" python3 "$OUT/../moe_client.py" > "$OUT/client_$arm.json" 2>"$OUT/client_$arm.err"
  local rc=$?

  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 5
  return $rc
}

run_arm bcg   --cuda-graph-backend-prefill breakable
run_arm eager --disable-prefill-cuda-graph

echo "=== capture / replay evidence (bcg arm) ==="
grep -iE "breakable|prefill cuda graph|capture" "$OUT/server_bcg.log" | tail -15
