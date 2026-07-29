#!/usr/bin/env bash
# R6.3 — image cost rebaseline + workload sweep + mixed safety.
#
# R6.3a: 720p, 128->128, c=1, n=400, 3 reps × {stock_default, fork_pcg}
# R6.3b: sweep matrix
#         text ∈ {128, 512, 2048}
#         image ∈ {360p, 720p}  (sglang bench serving accepts {4k,1080p,720p,360p};
#                                 360p is the smallest offered and stands in for the
#                                 intended "small image" cell that was originally
#                                 written as 224p — 224p is not a supported enum)
#         concurrency ∈ {1, 4}
#         n=100 per cell, 1 rep, identical seed
# R6.3c: fork_pcg interleaved text+image (50+50) mixed safety
#
# Sweep: for each variant we reuse a single server across all cells to
# amortize model-loading time. Server variants: stock_default (image
# path) and fork_pcg (image path). 12 cells × 2 servers = 24 bench runs.

set -uo pipefail

GPU_ID="${R6_GPU_ID:-${1:-}}"
[[ -z "$GPU_ID" || ! "$GPU_ID" =~ ^[0-9]+$ ]] && { echo "[R6.3] REFUSING: bad GPU_ID" >&2; exit 64; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"

R6_HOST_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
[[ -s "$R6_HOST_LIBCUDA" ]] || { echo "missing libcuda" >&2; exit 65; }
export LD_PRELOAD="${R6_HOST_LIBCUDA}${LD_PRELOAD:+ ${LD_PRELOAD}}"
export R6_HOST_LIBCUDA

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
ATTEMPT="${R6_ATTEMPT_DIR:-attempt_gpu${GPU_ID}}"
BASE="$ROOT/results/R6_fix_value_validation/R6.3_image_and_sweep/$ATTEMPT"
RAW="$BASE/raw"
SETSID=$ROOT/scripts/R6_setsid_exec.py
PREFLIGHT=$ROOT/scripts/R6_preflight_libcuda.py
VERDICT=$ROOT/scripts/R6_3_verdict.py
MIXED_CLIENT=$ROOT/scripts/R6_3_mixed_client.py
FIXTURE=$ROOT/results/R6_fix_value_validation/R6.1_correctness/fixtures/R6.1_fixture.png

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
STOCK_HEAD_EXPECTED=da802ddcafe55e25b3e1db86b1e0444afc3e05bc
FORK_HEAD_EXPECTED=986c89e69c25882ab6f3d396f8eb306f38f2c8d2
PORT=30003

mkdir -p "$RAW" "$RAW/a_rebaseline/stock_default" "$RAW/a_rebaseline/fork_pcg" \
         "$RAW/b_sweep" "$RAW/c_mixed_safety"

echo "[R6.3] GPU=$GPU_ID attempt=$ATTEMPT"
STOCK_HEAD=$(cd /sgl-workspace/sglang && git rev-parse HEAD)
FORK_HEAD=$(cd /data/sglang-fork && git rev-parse HEAD)
[[ "$STOCK_HEAD" == "$STOCK_HEAD_EXPECTED" ]] || { echo "stock SHA drift" >&2; exit 65; }
[[ "$FORK_HEAD"  == "$FORK_HEAD_EXPECTED"  ]] || { echo "fork SHA drift" >&2; exit 65; }
python3 "$PREFLIGHT" --gpu "$GPU_ID" 2>&1 | tee "$RAW/preflight.log"
sleep 3

GPU_ID_ARG="$GPU_ID" ATTEMPT_ARG="$ATTEMPT" python3 - > "$RAW/launch_context.json" <<'PYEOF'
import json, os, socket, subprocess
from datetime import datetime, timezone
GPU_ID=os.environ["GPU_ID_ARG"]; ATTEMPT=os.environ["ATTEMPT_ARG"]
def cmd(a): return subprocess.run(a,capture_output=True,text=True,timeout=10).stdout.strip()
print(json.dumps({
    "launched_by":"run_R6_3_image_and_sweep","attempt_dir":ATTEMPT,
    "selected_gpu_id":int(GPU_ID),
    "host_libcuda":os.environ.get("R6_HOST_LIBCUDA"),
    "ld_preload":os.environ.get("LD_PRELOAD"),
    "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
    "prelaunch_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "nvidia_driver":cmd(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader","-i",GPU_ID]),
    "sglang_stock_head":cmd(["git","-C","/sgl-workspace/sglang","rev-parse","HEAD"]),
    "sglang_fork_head":cmd(["git","-C","/data/sglang-fork","rev-parse","HEAD"]),
    "hostname":socket.gethostname(),
},indent=2,sort_keys=True))
PYEOF

# ---- PGID discipline ----
declare -a TRACKED_PGIDS=()
SRV_PID=""; SRV_PGID=""; SRV_LABEL=""
record_pgid(){ local pg="$1"; [[ -z "$pg" ]] && return 0
  for e in "${TRACKED_PGIDS[@]:-}"; do [[ "$e" == "$pg" ]] && return 0; done
  TRACKED_PGIDS+=("$pg"); }
verify_ownership(){ local pid="$1" rec="$2"; [[ -z "$pid" || -z "$rec" ]] && return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cur_pg cur_comm
  cur_pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  cur_comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
  [[ "$cur_pg" == "$rec" ]] || return 1
  [[ "$cur_comm" =~ ^python ]] || return 1
  return 0; }
signal_owned_pgid(){ local pg="$1" sig="$2"; [[ -z "$pg" ]] && return 0
  local live=0
  while IFS= read -r line; do local pp p2
    pp=$(echo "$line"|awk '{print $1}'); p2=$(echo "$line"|awk '{print $2}')
    if [[ "$p2" == "$pg" ]] && kill -0 "$pp" 2>/dev/null; then
      local cm; cm=$(ps -o comm= -p "$pp" 2>/dev/null | tr -d ' ')
      [[ "$cm" =~ ^python ]] && { live=1; break; }
    fi
  done < <(ps -eo pid,pgid --no-headers 2>/dev/null)
  [[ "$live" -eq 0 ]] && return 0
  kill "-${sig}" "-${pg}" 2>/dev/null || true; }
pre_launch_idle(){ local mem util pids
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
  echo "[R6.3] GPU $GPU_ID pre-launch: mem=${mem}MiB util=${util}% pids=[${pids}]"
  [[ "$mem" -le 500 && "$util" -le 5 && -z "$pids" ]] || return 1; return 0; }
check_no_foreign(){ local pids pid pgid_of foreign=()
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$')
  for pid in $pids; do
    pgid_of=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -z "$pgid_of" ]] && continue
    local ours=0
    for our in "${TRACKED_PGIDS[@]:-}"; do [[ -z "$our" ]] && continue; [[ "$pgid_of" == "$our" ]] && { ours=1; break; }; done
    [[ "$ours" -eq 0 ]] && foreign+=("pid=$pid pgid=$pgid_of")
  done
  [[ ${#foreign[@]} -gt 0 ]] && { echo "FOREIGN: ${foreign[*]}" >&2; echo "$(date -u -Iseconds) ${foreign[*]}" > "$RAW/foreign_pid_detected.txt"; return 71; }
  return 0; }

launch_server(){ local LABEL="$1" USE_FORK="$2" EXTRA="$3" LOG_DIR="$4"
  pre_launch_idle || return 1
  check_no_foreign || return 71
  local LOG="$LOG_DIR/server.log" PIDFILE="$LOG_DIR/server.pid"
  mkdir -p "$LOG_DIR"; rm -f "$LOG" "$PIDFILE"
  local ENV_PREFIX=""; [[ "$USE_FORK" == "yes" ]] && ENV_PREFIX="PYTHONPATH=${FORK_PY}"
  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST TORCH_LOGS
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1
  SRV_LABEL="$LABEL"
  echo "[R6.3] launching $LABEL fork=$USE_FORK extra='$EXTRA' -> $LOG"
  # shellcheck disable=SC2086
  env $ENV_PREFIX python3 "$SETSID" "$PIDFILE" \
    python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 --attention-backend flashinfer $EXTRA \
    > "$LOG" 2>&1 &
  local w=0; while [[ ! -s "$PIDFILE" && $w -lt 100 ]]; do sleep 0.1; w=$((w+1)); done
  [[ -s "$PIDFILE" ]] || return 3
  SRV_PID=$(cat "$PIDFILE"); SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  [[ "$SRV_PGID" == "$SRV_PID" ]] || return 3
  record_pgid "$SRV_PGID"
  echo "[R6.3]   $LABEL PID=$SRV_PID PGID=$SRV_PGID"
  local i R=0
  for i in $(seq 1 900); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then R=1; break; fi
    kill -0 "$SRV_PID" 2>/dev/null || { echo "$LABEL DIED" >&2; return 2; }
    sleep 2
  done
  [[ "$R" -eq 1 ]] || return 2
  return 0; }
teardown_server(){ local pid="$SRV_PID" pgid="$SRV_PGID" label="$SRV_LABEL"
  [[ -z "$pid" ]] && { SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; return 0; }
  echo "[R6.3] teardown $label"
  if verify_ownership "$pid" "$pgid"; then
    signal_owned_pgid "$pgid" TERM
    local i; for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null && verify_ownership "$pid" "$pgid"; then signal_owned_pgid "$pgid" KILL; sleep 2; fi
  fi
  SRV_PID=""; SRV_PGID=""; SRV_LABEL=""
  local m; for _ in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' '); [[ "$m" -lt 500 ]] && break; sleep 2; done; }
cleanup(){ local rc=$?
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" TERM; done
  sleep 5
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" KILL; done
  return $rc; }
trap cleanup EXIT INT TERM

# ---- R6.3a rebaseline ----
run_rebaseline_variant(){ local LABEL="$1" USE_FORK="$2"
  local vdir="$RAW/a_rebaseline/$LABEL"
  launch_server "rebaseline_${LABEL}" "$USE_FORK" "$([[ $USE_FORK == yes ]] && echo --enforce-piecewise-cuda-graph)" "$vdir" || return $?
  local r; for r in 1 2 3; do
    check_no_foreign || return 71
    python3 -m sglang.benchmark.serving \
      --backend sglang-oai-chat --base-url "http://127.0.0.1:$PORT" --model "$SNAP" \
      --dataset-name image --image-count 1 --image-resolution 720p --image-format png --image-content random \
      --random-input-len 128 --random-output-len 128 --random-range-ratio 1.0 \
      --max-concurrency 1 --num-prompts 400 --warmup-requests 30 --seed 1 \
      --extra-request-body '{"temperature": 0, "top_p": 1}' \
      --output-file "$vdir/rep${r}.jsonl" > "$vdir/rep${r}.log" 2>&1 || true
    echo "[R6.3a] $LABEL rep$r done"
  done
  teardown_server; }

echo "[R6.3] ===== R6.3a rebaseline (stock_default) ====="
run_rebaseline_variant stock_default no
echo "[R6.3] ===== R6.3a rebaseline (fork_pcg) ====="
run_rebaseline_variant fork_pcg yes

# ---- R6.3b sweep matrix ----
run_sweep_variant(){ local LABEL="$1" USE_FORK="$2"
  local server_log_dir="$RAW/b_sweep/_server_${LABEL}"
  launch_server "sweep_${LABEL}" "$USE_FORK" "$([[ $USE_FORK == yes ]] && echo --enforce-piecewise-cuda-graph)" "$server_log_dir" || return $?
  local txt res conc
  for txt in 128 512 2048; do
    for res in 360p 720p; do
      for conc in 1 4; do
        local cell="cell_t${txt}_r${res}_c${conc}"
        local cdir="$RAW/b_sweep/$cell/$LABEL"
        mkdir -p "$cdir"
        # Link server log to cell dir for verdict compatibility
        ln -sf "$server_log_dir/server.log" "$cdir/server.log" 2>/dev/null || true
        check_no_foreign || return 71
        python3 -m sglang.benchmark.serving \
          --backend sglang-oai-chat --base-url "http://127.0.0.1:$PORT" --model "$SNAP" \
          --dataset-name image --image-count 1 --image-resolution "$res" --image-format png --image-content random \
          --random-input-len "$txt" --random-output-len 128 --random-range-ratio 1.0 \
          --max-concurrency "$conc" --num-prompts 100 --warmup-requests 10 --seed 1 \
          --extra-request-body '{"temperature": 0, "top_p": 1}' \
          --output-file "$cdir/bench.jsonl" > "$cdir/bench.log" 2>&1 || true
        echo "[R6.3b] $LABEL $cell done"
      done
    done
  done
  teardown_server; }

echo "[R6.3] ===== R6.3b sweep (stock_default) ====="
run_sweep_variant stock_default no
echo "[R6.3] ===== R6.3b sweep (fork_pcg) ====="
run_sweep_variant fork_pcg yes

# ---- R6.3c mixed safety ----
echo "[R6.3] ===== R6.3c mixed safety (fork_pcg interleaved 50 text + 50 image) ====="
# Wait for GPU 6 to be idle again (foreign tenants may land between sweep
# teardown and R6.3c). Retry up to 20 x 15s.
C_SAFETY_READY=0
for wait_i in $(seq 1 20); do
  if launch_server "mixed_fork_pcg" yes "--enforce-piecewise-cuda-graph" "$RAW/c_mixed_safety"; then
    C_SAFETY_READY=1; break
  fi
  echo "[R6.3c] launch attempt $wait_i failed (foreign PID or not idle); waiting 15s and retrying" >&2
  sleep 15
done
if [[ "$C_SAFETY_READY" -ne 1 ]]; then
  echo "[R6.3c] ABORT: could not bring mixed-safety server up on GPU $GPU_ID after 20 retries; refusing to run client against nonexistent server" >&2
  echo "$(date -u -Iseconds) c_mixed_safety_launch_failed" > "$RAW/c_mixed_safety/launch_failed.txt"
else
  python3 "$MIXED_CLIENT" \
    --base-url "http://127.0.0.1:$PORT" --model "$SNAP" \
    --fixture "$FIXTURE" \
    --n-text 50 --n-image 50 \
    --out-jsonl "$RAW/c_mixed_safety/fork_pcg_interleaved.jsonl" \
    --out-summary "$RAW/c_mixed_safety/client_summary.json" \
    > "$RAW/c_mixed_safety/client.log" 2>&1 || true
  teardown_server
fi

# ---- Verdict ----
python3 "$VERDICT" --in-dir "$RAW" --out-md "$BASE/verdict.md" --out-json "$BASE/verdict.json"
echo "[R6.3] done: $BASE/verdict.md"
