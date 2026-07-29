#!/usr/bin/env bash
# R6.5 — empirical mixed-workload validation.
#
# Usage: R6_GPU_ID=0 R6_RATIOS="0.5 0.83 0.95" run_R6_5_empirical_mixed.sh [gpu]
# Or: run_R6_5_empirical_mixed.sh <gpu> --ratios 0.5 0.83 0.95
#
# For each ratio in R6_RATIOS:
#   1. Launch stock_default server, run mixed client with the ratio,
#      identical seed, teardown.
#   2. Launch fork_pcg server, run mixed client with SAME ratio +
#      SAME seed (same request sequence), teardown.
# Then compute verdict.

set -uo pipefail

GPU_ID="${R6_GPU_ID:-${1:-}}"
[[ -z "$GPU_ID" || ! "$GPU_ID" =~ ^[0-9]+$ ]] && { echo "REFUSING: bad GPU_ID" >&2; exit 64; }
export CUDA_VISIBLE_DEVICES="$GPU_ID"

RATIOS="${R6_RATIOS:-}"
if [[ -z "$RATIOS" ]]; then
  echo "REFUSING: R6_RATIOS env var required (space-separated)" >&2; exit 64
fi

N_PER_RATIO="${R6_N_PER_RATIO:-100}"
SEED="${R6_SEED:-42}"

R6_HOST_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
[[ -s "$R6_HOST_LIBCUDA" ]] || { echo "missing libcuda" >&2; exit 65; }
export LD_PRELOAD="${R6_HOST_LIBCUDA}${LD_PRELOAD:+ ${LD_PRELOAD}}"
export R6_HOST_LIBCUDA

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
ATTEMPT="${R6_ATTEMPT_DIR:-attempt_gpu${GPU_ID}}"
BASE="$ROOT/results/R6_fix_value_validation/R6.5_empirical_mixed/$ATTEMPT"
RAW="$BASE/raw"
SETSID=$ROOT/scripts/R6_setsid_exec.py
PREFLIGHT=$ROOT/scripts/R6_preflight_libcuda.py
CLIENT=$ROOT/scripts/R6_5_mixed_client.py
VERDICT=$ROOT/scripts/R6_5_verdict.py
FIXTURE=$ROOT/results/R6_fix_value_validation/R6.1_correctness/fixtures/R6.1_fixture.png
CASEA=/data/sglang-vllm-profiler/datasets/qwen3vl8b/caseA_short.jsonl
R64_JSON=$ROOT/results/R6_fix_value_validation/R6.4_analytical_crossover/crossover.json

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
PORT=30003

mkdir -p "$RAW"

python3 "$PREFLIGHT" --gpu "$GPU_ID" 2>&1 | tee "$RAW/preflight.log" || exit 65
sleep 3

GPU_ID_ARG="$GPU_ID" ATTEMPT_ARG="$ATTEMPT" python3 - > "$RAW/launch_context.json" <<'PYEOF'
import json, os, socket, subprocess
from datetime import datetime, timezone
GPU_ID=os.environ["GPU_ID_ARG"]; ATTEMPT=os.environ["ATTEMPT_ARG"]
def cmd(a): return subprocess.run(a,capture_output=True,text=True,timeout=10).stdout.strip()
print(json.dumps({"launched_by":"run_R6_5_empirical_mixed","attempt_dir":ATTEMPT,
    "selected_gpu_id":int(GPU_ID),
    "host_libcuda":os.environ.get("R6_HOST_LIBCUDA"),"ld_preload":os.environ.get("LD_PRELOAD"),
    "prelaunch_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "nvidia_driver":cmd(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader","-i",GPU_ID]),
    "sglang_stock_head":cmd(["git","-C","/sgl-workspace/sglang","rev-parse","HEAD"]),
    "sglang_fork_head":cmd(["git","-C","/data/sglang-fork","rev-parse","HEAD"]),
    "hostname":socket.gethostname()}, indent=2, sort_keys=True))
PYEOF

declare -a TRACKED_PGIDS=()
SRV_PID=""; SRV_PGID=""; SRV_LABEL=""

record_pgid(){ local pg="$1"; [[ -z "$pg" ]] && return 0
  for e in "${TRACKED_PGIDS[@]:-}"; do [[ "$e" == "$pg" ]] && return 0; done
  TRACKED_PGIDS+=("$pg"); }
verify_ownership(){ local pid="$1" rec="$2"; [[ -z "$pid" || -z "$rec" ]] && return 1
  kill -0 "$pid" 2>/dev/null || return 1
  [[ "$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')" == "$rec" ]] || return 1
  [[ "$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')" =~ ^python ]] || return 1
  return 0; }
signal_owned_pgid(){ local pg="$1" sig="$2"; [[ -z "$pg" ]] && return 0
  local live=0
  while IFS= read -r line; do local pp p2; pp=$(echo "$line"|awk '{print $1}'); p2=$(echo "$line"|awk '{print $2}')
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
  [[ "$mem" -le 500 && "$util" -le 5 && -z "$pids" ]] || return 1; return 0; }
check_no_foreign(){ local pids pid pgid_of foreign=()
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$')
  for pid in $pids; do pgid_of=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -z "$pgid_of" ]] && continue
    local ours=0; for our in "${TRACKED_PGIDS[@]:-}"; do [[ -z "$our" ]] && continue; [[ "$pgid_of" == "$our" ]] && { ours=1; break; }; done
    [[ "$ours" -eq 0 ]] && foreign+=("pid=$pid pgid=$pgid_of")
  done
  [[ ${#foreign[@]} -gt 0 ]] && { echo "FOREIGN: ${foreign[*]}" >&2; return 71; }; return 0; }
launch_server(){ local LABEL="$1" USE_FORK="$2" EXTRA="$3" LOG_DIR="$4"
  pre_launch_idle || return 1
  check_no_foreign || return 71
  local LOG="$LOG_DIR/server.log" PIDFILE="$LOG_DIR/server.pid"
  mkdir -p "$LOG_DIR"; rm -f "$LOG" "$PIDFILE"
  local ENV_PREFIX=""; [[ "$USE_FORK" == "yes" ]] && ENV_PREFIX="PYTHONPATH=${FORK_PY}"
  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST TORCH_LOGS
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1
  SRV_LABEL="$LABEL"
  # shellcheck disable=SC2086
  env $ENV_PREFIX python3 "$SETSID" "$PIDFILE" \
    python3 -m sglang.launch_server --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 --attention-backend flashinfer --mem-fraction-static 0.88 $EXTRA \
    > "$LOG" 2>&1 &
  local w=0; while [[ ! -s "$PIDFILE" && $w -lt 100 ]]; do sleep 0.1; w=$((w+1)); done
  [[ -s "$PIDFILE" ]] || return 3
  SRV_PID=$(cat "$PIDFILE"); SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  [[ "$SRV_PGID" == "$SRV_PID" ]] || return 3
  record_pgid "$SRV_PGID"
  echo "[R6.5]   $LABEL PID=$SRV_PID PGID=$SRV_PGID"
  local i R=0
  for i in $(seq 1 900); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then R=1; break; fi
    kill -0 "$SRV_PID" 2>/dev/null || return 2
    sleep 2
  done
  [[ "$R" -eq 1 ]] || return 2; return 0; }
teardown_server(){ local pid="$SRV_PID" pgid="$SRV_PGID"; [[ -z "$pid" ]] && return 0
  if verify_ownership "$pid" "$pgid"; then
    signal_owned_pgid "$pgid" TERM
    local i; for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null && verify_ownership "$pid" "$pgid"; then signal_owned_pgid "$pgid" KILL; sleep 2; fi
  fi
  SRV_PID=""; SRV_PGID=""; SRV_LABEL=""
  local m; for _ in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' '); [[ "$m" -lt 500 ]] && break; sleep 2; done; }
cleanup(){ local rc=$?
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" TERM; done; sleep 5
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" KILL; done; return $rc; }
trap cleanup EXIT INT TERM

run_ratio(){ local RATIO="$1"; local RID="ratio_$(echo $RATIO | tr '.' 'p')"
  local rdir="$RAW/$RID"
  echo "[R6.5] ===== $RID (ratio=$RATIO) ====="
  for VARIANT_INFO in "stock_default:no:" "fork_pcg:yes:--enforce-piecewise-cuda-graph"; do
    local V=$(echo "$VARIANT_INFO" | cut -d: -f1)
    local USE_FORK=$(echo "$VARIANT_INFO" | cut -d: -f2)
    local EXTRA=$(echo "$VARIANT_INFO" | cut -d: -f3)
    local READY=0
    for LA in $(seq 1 20); do
      if launch_server "${RID}_${V}" "$USE_FORK" "$EXTRA" "$rdir/$V"; then
        READY=1; break
      fi
      echo "[R6.5] launch attempt $LA failed for $V (foreign PID / not idle); waiting 15s and retrying" >&2
      sleep 15
    done
    if [[ "$READY" -ne 1 ]]; then
      echo "[R6.5] ABORT: could not bring $V up for $RID after 20 retries; skipping this variant" >&2
      echo "$(date -u -Iseconds) launch_failed_after_retries" > "$rdir/$V.LAUNCH_FAILED"
      continue
    fi
    python3 "$CLIENT" \
      --base-url "http://127.0.0.1:$PORT" --model "$SNAP" \
      --fixture "$FIXTURE" --caseA "$CASEA" \
      --text-ratio "$RATIO" --n "$N_PER_RATIO" --seed "$SEED" \
      --out-jsonl "$rdir/$V/requests.jsonl" \
      --out-summary "$rdir/$V/summary.json" \
      > "$rdir/$V/client.log" 2>&1 || true
    teardown_server
  done; }

for R in $RATIOS; do run_ratio "$R" || echo "[R6.5] ratio $R had issues (continuing)"; done

python3 "$VERDICT" --in-dir "$RAW" --r64-json "$R64_JSON" \
  --out-md "$BASE/verdict.md" --out-json "$BASE/verdict.json"
echo "[R6.5] done: $BASE/verdict.md"
