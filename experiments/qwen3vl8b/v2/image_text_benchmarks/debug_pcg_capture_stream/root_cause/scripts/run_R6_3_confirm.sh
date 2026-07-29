#!/usr/bin/env bash
# R6.3 confirmation reps for winning cells identified by run_R6_3_image_and_sweep.sh.
#
# Per the R6.3 revised decision framework: "For every apparent winning cell,
# run at least three clean confirmation repetitions for both variants.
# A headline winning cell must survive confirmation and cannot rely on one
# noisy run."
#
# Inputs:
#   R6_GPU_ID (env or $1)  - physical GPU
#   R6_ATTEMPT_DIR         - defaults to "attempt_gpuN_confirm"
#   R6_CELLS               - space-separated list of cell IDs. Defaults to
#                            the 7 winners from attempt_gpu2 (2026-07-29).
#                            Each cell must match the format
#                            "cell_t<TXT>_r<RES>_c<CONC>".
#   R6_REPS                - reps per variant per cell (default 3)
#
# Runs one server per variant (stock_default, fork_pcg) and issues R6_REPS
# n=100 benches per cell against the same server (matches discovery run
# server-reuse pattern). Each rep starts with a fresh cache-flush hop
# omitted; timing pattern matches discovery so ratios are apples-to-apples.

set -uo pipefail

GPU_ID="${R6_GPU_ID:-${1:-}}"
[[ -z "$GPU_ID" || ! "$GPU_ID" =~ ^[0-9]+$ ]] && { echo "[R6.3-confirm] REFUSING: bad GPU_ID" >&2; exit 64; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"

R6_HOST_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
[[ -s "$R6_HOST_LIBCUDA" ]] || { echo "missing libcuda" >&2; exit 65; }
export LD_PRELOAD="${R6_HOST_LIBCUDA}${LD_PRELOAD:+ ${LD_PRELOAD}}"
export R6_HOST_LIBCUDA

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
ATTEMPT="${R6_ATTEMPT_DIR:-attempt_gpu${GPU_ID}_confirm}"
BASE="$ROOT/results/R6_fix_value_validation/R6.3_image_and_sweep/$ATTEMPT"
RAW="$BASE/raw"
SETSID=$ROOT/scripts/R6_setsid_exec.py
PREFLIGHT=$ROOT/scripts/R6_preflight_libcuda.py

DEFAULT_CELLS="cell_t128_r360p_c1 cell_t128_r360p_c4 cell_t128_r720p_c1 cell_t128_r720p_c4 cell_t512_r360p_c1 cell_t512_r360p_c4 cell_t512_r720p_c1"
CELLS="${R6_CELLS:-$DEFAULT_CELLS}"
REPS="${R6_REPS:-3}"

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
STOCK_HEAD_EXPECTED=da802ddcafe55e25b3e1db86b1e0444afc3e05bc
FORK_HEAD_EXPECTED=986c89e69c25882ab6f3d396f8eb306f38f2c8d2
PORT=30003

mkdir -p "$RAW"

echo "[R6.3-confirm] GPU=$GPU_ID attempt=$ATTEMPT reps=$REPS cells=[$CELLS]"
STOCK_HEAD=$(cd /sgl-workspace/sglang && git rev-parse HEAD)
FORK_HEAD=$(cd /data/sglang-fork && git rev-parse HEAD)
[[ "$STOCK_HEAD" == "$STOCK_HEAD_EXPECTED" ]] || { echo "stock SHA drift" >&2; exit 65; }
[[ "$FORK_HEAD"  == "$FORK_HEAD_EXPECTED"  ]] || { echo "fork SHA drift" >&2; exit 65; }
python3 "$PREFLIGHT" --gpu "$GPU_ID" 2>&1 | tee "$RAW/preflight.log"
sleep 3

GPU_ID_ARG="$GPU_ID" ATTEMPT_ARG="$ATTEMPT" CELLS_ARG="$CELLS" REPS_ARG="$REPS" python3 - > "$RAW/launch_context.json" <<'PYEOF'
import json, os, socket, subprocess
from datetime import datetime, timezone
GPU_ID=os.environ["GPU_ID_ARG"]; ATTEMPT=os.environ["ATTEMPT_ARG"]
def cmd(a): return subprocess.run(a,capture_output=True,text=True,timeout=10).stdout.strip()
print(json.dumps({
    "launched_by":"run_R6_3_confirm","attempt_dir":ATTEMPT,
    "selected_gpu_id":int(GPU_ID),
    "cells":os.environ["CELLS_ARG"].split(),
    "reps_per_cell_per_variant":int(os.environ["REPS_ARG"]),
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
  echo "[R6.3-confirm] GPU $GPU_ID pre-launch: mem=${mem}MiB util=${util}% pids=[${pids}]"
  [[ "$mem" -le 500 && "$util" -le 5 && -z "$pids" ]] || return 1; return 0; }
check_foreign_pids(){ local pids pid pgid_of foreign=()
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$')
  for pid in $pids; do
    pgid_of=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -z "$pgid_of" ]] && continue
    local ours=0
    for our in "${TRACKED_PGIDS[@]:-}"; do [[ -z "$our" ]] && continue; [[ "$pgid_of" == "$our" ]] && { ours=1; break; }; done
    [[ "$ours" -eq 0 ]] && foreign+=("pid=$pid pgid=$pgid_of")
  done
  [[ ${#foreign[@]} -gt 0 ]] && { echo "FOREIGN: ${foreign[*]}" >&2; return 71; }
  return 0; }

launch_server(){ local LABEL="$1" USE_FORK="$2" EXTRA="$3" LOG_DIR="$4"
  pre_launch_idle || return 1
  check_foreign_pids || return 71
  local LOG="$LOG_DIR/server.log" PIDFILE="$LOG_DIR/server.pid"
  mkdir -p "$LOG_DIR"; rm -f "$LOG" "$PIDFILE"
  local ENV_PREFIX=""; [[ "$USE_FORK" == "yes" ]] && ENV_PREFIX="PYTHONPATH=${FORK_PY}"
  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST TORCH_LOGS
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1
  SRV_LABEL="$LABEL"
  echo "[R6.3-confirm] launching $LABEL fork=$USE_FORK extra='$EXTRA'"
  # shellcheck disable=SC2086
  env $ENV_PREFIX python3 "$SETSID" "$PIDFILE" \
    python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 --attention-backend flashinfer $EXTRA \
    > "$LOG" 2>&1 &
  local w=0; while [[ ! -s "$PIDFILE" && $w -lt 100 ]]; do sleep 0.1; w=$((w+1)); done
  [[ -s "$PIDFILE" ]] || return 3
  SRV_PID=$(cat "$PIDFILE"); SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  [[ "$SRV_PGID" == "$SRV_PID" ]] || return 3
  record_pgid "$SRV_PGID"
  echo "[R6.3-confirm]   $LABEL PID=$SRV_PID PGID=$SRV_PGID"
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
  echo "[R6.3-confirm] teardown $label"
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

# Parse cell string "cell_t<TXT>_r<RES>_c<CONC>" into TXT / RES / CONC.
cell_parse(){ local cell="$1"
  echo "$cell" | sed -E 's|^cell_t([0-9]+)_r([0-9]+p)_c([0-9]+)$|\1 \2 \3|'
}

run_variant(){ local LABEL="$1" USE_FORK="$2"
  local server_log_dir="$RAW/_server_${LABEL}"
  local extra=""; [[ "$USE_FORK" == "yes" ]] && extra="--enforce-piecewise-cuda-graph"
  launch_server "confirm_${LABEL}" "$USE_FORK" "$extra" "$server_log_dir" || return $?
  local cell parsed txt res conc rep cdir
  for cell in $CELLS; do
    parsed=$(cell_parse "$cell")
    txt=$(echo "$parsed" | awk '{print $1}')
    res=$(echo "$parsed" | awk '{print $2}')
    conc=$(echo "$parsed" | awk '{print $3}')
    [[ -z "$txt" || -z "$res" || -z "$conc" ]] && { echo "[R6.3-confirm] REFUSING cell '$cell' -- unparseable" >&2; return 66; }
    cdir="$RAW/$cell/$LABEL"
    mkdir -p "$cdir"
    # link server log for verdict compatibility
    ln -sf "$server_log_dir/server.log" "$cdir/server.log" 2>/dev/null || true
    for rep in $(seq 1 $REPS); do
      check_foreign_pids || { echo "[R6.3-confirm] foreign PID mid-run on $cell rep $rep; marking rep MISSING and continuing" >&2
        echo "$(date -u -Iseconds) foreign_pid_seen" > "$cdir/rep${rep}.INVALIDATED"
        continue; }
      python3 -m sglang.benchmark.serving \
        --backend sglang-oai-chat --base-url "http://127.0.0.1:$PORT" --model "$SNAP" \
        --dataset-name image --image-count 1 --image-resolution "$res" --image-format png --image-content random \
        --random-input-len "$txt" --random-output-len 128 --random-range-ratio 1.0 \
        --max-concurrency "$conc" --num-prompts 100 --warmup-requests 10 --seed 1 \
        --extra-request-body '{"temperature": 0, "top_p": 1}' \
        --output-file "$cdir/rep${rep}.jsonl" > "$cdir/rep${rep}.log" 2>&1 || true
      echo "[R6.3-confirm] $LABEL $cell rep$rep done"
    done
  done
  teardown_server; }

echo "[R6.3-confirm] ===== confirm stock_default ====="
run_variant stock_default no
echo "[R6.3-confirm] ===== confirm fork_pcg ====="
run_variant fork_pcg yes

# Verdict
python3 "$ROOT/scripts/R6_3_confirm_verdict.py" --in-dir "$RAW" --out-md "$BASE/verdict.md" --out-json "$BASE/verdict.json"
echo "[R6.3-confirm] done: $BASE/verdict.md"
