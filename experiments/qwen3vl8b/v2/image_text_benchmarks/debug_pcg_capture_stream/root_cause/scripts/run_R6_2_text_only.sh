#!/usr/bin/env bash
# R6.2 — text-only Case A non-regression + retained-benefit on Qwen3-VL.
# Uses caseA_short.jsonl via --dataset-name autobench for exact request
# replay across variants. 4 variants × 5 reps × 400 measured + 30 warmup.
# GPU 0 only (user-authorized).

set -uo pipefail

GPU_ID="${R6_GPU_ID:-${1:-}}"
if [[ -z "${GPU_ID}" ]]; then
  echo "[R6.2] REFUSING: no GPU_ID" >&2; exit 64
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[R6.2] GPU_ID must be integer; got: ${GPU_ID}" >&2; exit 64
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

R6_HOST_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
if [[ ! -s "$R6_HOST_LIBCUDA" ]]; then echo "missing host libcuda" >&2; exit 65; fi
export LD_PRELOAD="${R6_HOST_LIBCUDA}${LD_PRELOAD:+ ${LD_PRELOAD}}"
export R6_HOST_LIBCUDA

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
ATTEMPT="${R6_ATTEMPT_DIR:-attempt_gpu${GPU_ID}}"
BASE="$ROOT/results/R6_fix_value_validation/R6.2_text_only_caseA/$ATTEMPT"
RAW="$BASE/raw"
SETSID_HELPER="$ROOT/scripts/R6_setsid_exec.py"
PREFLIGHT_PY="$ROOT/scripts/R6_preflight_libcuda.py"
VERDICT_PY="$ROOT/scripts/R6_2_verdict.py"

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
STOCK_HEAD_EXPECTED=da802ddcafe55e25b3e1db86b1e0444afc3e05bc
FORK_HEAD_EXPECTED=986c89e69c25882ab6f3d396f8eb306f38f2c8d2
CASEA=/data/sglang-vllm-profiler/datasets/qwen3vl8b/caseA_short.jsonl
CASEA_SHA_EXPECTED=fab4917772e087447d7c33d53ada63340b126088c1f195f118b9488d5f5b619e
PORT=30003

mkdir -p "$RAW"

echo "[R6.2] GPU=$GPU_ID  attempt=$ATTEMPT"
STOCK_HEAD=$(cd /sgl-workspace/sglang && git rev-parse HEAD)
FORK_HEAD=$(cd /data/sglang-fork && git rev-parse HEAD)
[[ "$STOCK_HEAD" == "$STOCK_HEAD_EXPECTED" ]] || { echo "stock SHA drift" >&2; exit 65; }
[[ "$FORK_HEAD"  == "$FORK_HEAD_EXPECTED"  ]] || { echo "fork SHA drift" >&2; exit 65; }
CASEA_SHA=$(sha256sum "$CASEA" | awk '{print $1}')
[[ "$CASEA_SHA" == "$CASEA_SHA_EXPECTED" ]] || { echo "caseA SHA drift" >&2; exit 65; }
echo "[R6.2] provenance OK"

if ! python3 "$PREFLIGHT_PY" --gpu "$GPU_ID" 2>&1 | tee "$RAW/preflight.log"; then
  echo "[R6.2] preflight FAILED" >&2; exit 65
fi
sleep 3
mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
[[ "$mem" -le 500 ]] || { echo "post-preflight mem>500" >&2; exit 65; }

GPU_ID_ARG="$GPU_ID" ATTEMPT_ARG="$ATTEMPT" \
  python3 - > "$RAW/launch_context.json" <<'PYEOF'
import json, os, socket, subprocess
from datetime import datetime, timezone
GPU_ID = os.environ["GPU_ID_ARG"]; ATTEMPT = os.environ["ATTEMPT_ARG"]
def cmd(a): return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
pids = cmd(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits","-i",GPU_ID])
print(json.dumps({
    "launched_by":"run_R6_2_text_only","attempt_dir":ATTEMPT,
    "selected_gpu_id":int(GPU_ID),
    "host_libcuda":os.environ.get("R6_HOST_LIBCUDA"),
    "ld_preload":os.environ.get("LD_PRELOAD"),
    "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),
    "prelaunch_utc":datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "prelaunch_state":{
        "mem_mib":int(cmd(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits","-i",GPU_ID])),
        "util_pct":int(cmd(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits","-i",GPU_ID])),
        "compute_pids":[x for x in pids.splitlines() if x],
    },
    "nvidia_driver":cmd(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader","-i",GPU_ID]),
    "sglang_stock_head":cmd(["git","-C","/sgl-workspace/sglang","rev-parse","HEAD"]),
    "sglang_fork_head": cmd(["git","-C","/data/sglang-fork","rev-parse","HEAD"]),
    "hostname":socket.gethostname(),
}, indent=2, sort_keys=True))
PYEOF

declare -a TRACKED_PGIDS=()
SRV_PID=""; SRV_PGID=""; SRV_LABEL=""

record_pgid () { local pg="$1"; [[ -z "$pg" ]] && return 0
  for e in "${TRACKED_PGIDS[@]:-}"; do [[ "$e" == "$pg" ]] && return 0; done
  TRACKED_PGIDS+=("$pg"); }

verify_ownership () { local pid="$1" rec="$2"
  [[ -z "$pid" || -z "$rec" ]] && return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cur_pg cur_comm
  cur_pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  cur_comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
  [[ "$cur_pg" == "$rec" ]] || return 1
  [[ "$cur_comm" =~ ^python ]] || return 1
  return 0; }

signal_owned_pgid () { local pg="$1" sig="$2"; [[ -z "$pg" ]] && return 0
  local live=0
  while IFS= read -r line; do
    local pp p2; pp=$(echo "$line"|awk '{print $1}'); p2=$(echo "$line"|awk '{print $2}')
    if [[ "$p2" == "$pg" ]] && kill -0 "$pp" 2>/dev/null; then
      local cm; cm=$(ps -o comm= -p "$pp" 2>/dev/null | tr -d ' ')
      [[ "$cm" =~ ^python ]] && { live=1; break; }
    fi
  done < <(ps -eo pid,pgid --no-headers 2>/dev/null)
  [[ "$live" -eq 0 ]] && return 0
  kill "-${sig}" "-${pg}" 2>/dev/null || true; }

pre_launch_idle () { local mem util pids
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
  echo "[R6.2] GPU $GPU_ID pre-launch: mem=${mem}MiB util=${util}% pids=[${pids}]"
  [[ "$mem" -le 500 && "$util" -le 5 && -z "$pids" ]] || return 1
  return 0; }

check_no_foreign () { local pids pid pgid_of foreign=()
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$')
  for pid in $pids; do
    pgid_of=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    [[ -z "$pgid_of" ]] && continue
    local ours=0
    for our in "${TRACKED_PGIDS[@]:-}"; do
      [[ -z "$our" ]] && continue
      [[ "$pgid_of" == "$our" ]] && { ours=1; break; }
    done
    [[ "$ours" -eq 0 ]] && foreign+=("pid=$pid pgid=$pgid_of")
  done
  if [[ ${#foreign[@]} -gt 0 ]]; then
    echo "[R6.2] FOREIGN COMPUTE PIDS: ${foreign[*]}" >&2
    echo "gpu=$GPU_ID at=$(date -u -Iseconds) ${foreign[*]}" > "$RAW/foreign_pid_detected.txt"
    return 71
  fi
  return 0; }

launch_server () { local LABEL="$1" USE_FORK="$2" EXTRA="$3"
  pre_launch_idle || return 1
  check_no_foreign || return 71
  local LOG="$RAW/$LABEL/server.log" PIDFILE="$RAW/$LABEL/server.pid"
  mkdir -p "$RAW/$LABEL"
  rm -f "$LOG" "$PIDFILE"
  local ENV_PREFIX=""
  [[ "$USE_FORK" == "yes" ]] && ENV_PREFIX="PYTHONPATH=${FORK_PY}"
  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE TORCH_LOGS
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1
  SRV_LABEL="$LABEL"
  echo "[R6.2] launching $LABEL (fork=$USE_FORK extra='$EXTRA')"
  # shellcheck disable=SC2086
  env $ENV_PREFIX \
    python3 "$SETSID_HELPER" "$PIDFILE" \
      python3 -m sglang.launch_server \
        --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
        --attention-backend flashinfer $EXTRA \
    > "$LOG" 2>&1 &
  local waited=0
  while [[ ! -s "$PIDFILE" && $waited -lt 100 ]]; do sleep 0.1; waited=$((waited+1)); done
  [[ -s "$PIDFILE" ]] || { echo "$LABEL: no pidfile" >&2; return 3; }
  SRV_PID=$(cat "$PIDFILE"); SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  [[ -n "$SRV_PGID" && "$SRV_PGID" == "$SRV_PID" ]] || { echo "PGID mismatch $SRV_PGID != $SRV_PID" >&2; return 3; }
  record_pgid "$SRV_PGID"
  echo "[R6.2]   $LABEL PID=$SRV_PID PGID=$SRV_PGID"
  local READY=0 i
  for i in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then
      READY=1; break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then echo "$LABEL DIED" >&2; return 2; fi
    sleep 2
  done
  [[ "$READY" -eq 1 ]] || { echo "$LABEL not ready" >&2; return 2; }
  return 0; }

teardown_server () {
  local pid="$SRV_PID" pgid="$SRV_PGID" label="$SRV_LABEL"
  if [[ -z "$pid" || -z "$pgid" ]]; then SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; return 0; fi
  echo "[R6.2] teardown $label"
  if verify_ownership "$pid" "$pgid"; then
    signal_owned_pgid "$pgid" TERM
    local i; for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
      if verify_ownership "$pid" "$pgid"; then signal_owned_pgid "$pgid" KILL; sleep 2; fi
    fi
  fi
  SRV_PID=""; SRV_PGID=""; SRV_LABEL=""
  local m; for _ in $(seq 1 30); do m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' '); [[ "$m" -lt 500 ]] && break; sleep 2; done; }

cleanup_on_exit () { local rc=$?
  echo "[R6.2] cleanup_on_exit rc=$rc"
  local pg; for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" TERM; done
  sleep 5
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" KILL; done
  return $rc; }
trap cleanup_on_exit EXIT INT TERM

run_bench_rep () { local LABEL="$1" REP="$2"
  local OUT="$RAW/$LABEL/rep${REP}.jsonl" LOG="$RAW/$LABEL/rep${REP}.log"
  check_no_foreign || return 71
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:$PORT" \
    --model "$SNAP" \
    --dataset-name autobench --dataset-path "$CASEA" \
    --max-concurrency 1 --num-prompts 400 --warmup-requests 30 \
    --seed 1 --extra-request-body '{"temperature": 0, "top_p": 1}' \
    --output-file "$OUT" \
    > "$LOG" 2>&1 || true
  echo "[R6.2] bench $LABEL rep$REP done"; }

run_variant () { local LABEL="$1" USE_FORK="$2" EXTRA="$3"
  launch_server "$LABEL" "$USE_FORK" "$EXTRA" || return $?
  local r; for r in 1 2 3 4 5; do run_bench_rep "$LABEL" "$r" || return $?; done
  teardown_server; }

echo "[R6.2] ====== stock-default ======"
run_variant stock_default no  ""
echo "[R6.2] ====== stock-PCG ======"
run_variant stock_pcg     no  "--enforce-piecewise-cuda-graph"
echo "[R6.2] ====== fork-PCG ======"
run_variant fork_pcg      yes "--enforce-piecewise-cuda-graph"
echo "[R6.2] ====== stock-default-repeat (drift bracket) ======"
run_variant stock_default_repeat no ""

python3 "$VERDICT_PY" --in-dir "$RAW" --out-md "$BASE/verdict.md" --out-json "$BASE/verdict.json"
echo "[R6.2] done: $BASE/verdict.md"
