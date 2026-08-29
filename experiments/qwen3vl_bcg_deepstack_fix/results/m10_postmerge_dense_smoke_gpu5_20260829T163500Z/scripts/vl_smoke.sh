#!/usr/bin/env bash
# Post-merge smoke for the DeepStack BCG replay slot.
#
# Serves Qwen3VLForConditionalGeneration (deepstack_visual_indexes=[5,11,17],
# so the replay path is genuinely exercised) once under breakable prefill CUDA
# graph and once eager, sends the same deterministic image + text prompts
# greedily, and compares the completions.
#
# Runs the merged PR branch via PYTHONPATH, not the installed sglang.
set -uo pipefail

GPU_ID="${GPU_ID:-5}"
PORT="${PORT:-31441}"
FORK=/data/sglang-fork
MODEL=$(python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3-VL-4B-Instruct', local_files_only=True))
")
OUT="${OUT:-/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/vl_smoke}"
mkdir -p "$OUT"
echo "MODEL=$MODEL"
echo "GPU=$GPU_ID PORT=$PORT OUT=$OUT"

run_arm () {
  local arm="$1"; shift
  echo "=== arm=$arm ==="
  CUDA_VISIBLE_DEVICES="$GPU_ID" PYTHONPATH="$FORK/python" \
  LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05${LD_PRELOAD:+:$LD_PRELOAD}" \
  SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1 \
    python3 -m sglang.launch_server \
      --model-path "$MODEL" \
      --port "$PORT" --host 127.0.0.1 \
      --mem-fraction-static 0.80 \
      --disable-radix-cache \
      "$@" > "$OUT/server_$arm.log" 2>&1 &
  local pid=$!

  for _ in $(seq 1 180); do
    sleep 5
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    kill -0 "$pid" 2>/dev/null || { echo "SERVER DIED (arm=$arm)"; tail -40 "$OUT/server_$arm.log"; return 1; }
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || { echo "HEALTH TIMEOUT (arm=$arm)"; tail -40 "$OUT/server_$arm.log"; kill -9 "$pid" 2>/dev/null; return 1; }

  PORT="$PORT" ARM="$arm" python3 "$OUT/../vl_client.py" > "$OUT/client_$arm.json" 2>"$OUT/client_$arm.err"
  local rc=$?

  kill -TERM "$pid" 2>/dev/null
  for _ in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
  kill -9 "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
  sleep 5
  return $rc
}

run_arm bcg   --cuda-graph-backend-prefill breakable || exit 1
run_arm eager --disable-prefill-cuda-graph || exit 1

echo "=== capture evidence (bcg arm) ==="
grep -iE "Capture target prefill|backend=breakable" "$OUT/server_bcg.log" | tail -5
echo "=== per-prefill graph status: bcg ==="
grep -oE "#new-token: [0-9]+.*cuda graph: (True|False)" "$OUT/server_bcg.log" | grep -oE "#new-token: [0-9]+|cuda graph: (True|False)" | paste - - | head -10
echo "=== per-prefill graph status: eager ==="
grep -oE "#new-token: [0-9]+.*cuda graph: (True|False)" "$OUT/server_eager.log" | grep -oE "#new-token: [0-9]+|cuda graph: (True|False)" | paste - - | head -10
echo "=== completion diff (empty == identical) ==="
diff <(python3 -c "import json;d=json.load(open('$OUT/client_bcg.json'));print(d['image']);print('---');print(d['text'])") \
     <(python3 -c "import json;d=json.load(open('$OUT/client_eager.json'));print(d['image']);print('---');print(d['text'])") \
  && echo "IDENTICAL"
