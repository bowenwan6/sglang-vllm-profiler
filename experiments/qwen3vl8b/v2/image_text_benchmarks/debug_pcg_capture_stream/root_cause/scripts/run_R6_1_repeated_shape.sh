#!/usr/bin/env bash
# R6.1 Attempt 04 — repeated-shape safety control (Amendment B, 2026-07-28).
#
# Recovers the exact historical R1 / E2a sustained-workload recipe and
# runs it under --enforce-piecewise-cuda-graph on stock-PCG and fork-PCG
# serially, on GPU 0. Predeclared safety verdict in
#   R6.1_correctness/protocol_amendment_B_repeated_shape_safety.md §B.2
# enforced by scripts/R6_1_verdict_amendment_B.py.
#
# Reuses R6_setsid_exec.py, R6_preflight_libcuda.py. Does not use
# the R6_1_client.py (which is 3-prompt-image only); instead invokes
# sglang.benchmark.serving directly with the R1 arg vector.
#
# stock-PCG's server is expected to crash mid-bench with the historical
# capture-stream assertion. That crash is scoped to the launched PGID:
# we let the server process die naturally, then the runner's teardown
# is a no-op (no live PID to signal). The bench client returns with
# HTTP failures on the requests after the crash. No pkill/killall.
#
# Refuses to run without an explicit GPU_ID (user must authorize).

set -uo pipefail

# --------------------------------------------------------------------------
# 0. Argument / env parsing + safety refusals
# --------------------------------------------------------------------------
GPU_ID="${R6_GPU_ID:-${1:-}}"
if [[ -z "${GPU_ID}" ]]; then
  cat >&2 <<EOF
[R6.1B] REFUSING TO RUN — no GPU ID provided.
Pass explicitly:
  R6_GPU_ID=<id> ./run_R6_1_repeated_shape.sh
EOF
  exit 64
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[R6.1B] GPU_ID must be a non-negative integer; got: ${GPU_ID}" >&2
  exit 64
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# --------------------------------------------------------------------------
# 0b. Host-libcuda LD_PRELOAD
# --------------------------------------------------------------------------
R6_HOST_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
if [[ ! -s "$R6_HOST_LIBCUDA" ]]; then
  echo "[R6.1B] Missing host libcuda: $R6_HOST_LIBCUDA" >&2; exit 65
fi
export LD_PRELOAD="${R6_HOST_LIBCUDA}${LD_PRELOAD:+ ${LD_PRELOAD}}"
export R6_HOST_LIBCUDA

# --------------------------------------------------------------------------
# 1. Paths + provenance
# --------------------------------------------------------------------------
ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
ATTEMPT="${R6_ATTEMPT_DIR:-attempt_04_repeated_shape_gpu${GPU_ID}}"
BASE="$ROOT/results/R6_fix_value_validation/R6.1_correctness/$ATTEMPT"
RAW="$BASE/raw"
SETSID_HELPER="$ROOT/scripts/R6_setsid_exec.py"
PREFLIGHT_PY="$ROOT/scripts/R6_preflight_libcuda.py"
VERDICT_PY="$ROOT/scripts/R6_1_verdict_amendment_B.py"

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
STOCK_HEAD_EXPECTED=da802ddcafe55e25b3e1db86b1e0444afc3e05bc
FORK_HEAD_EXPECTED=986c89e69c25882ab6f3d396f8eb306f38f2c8d2
PORT=30003

mkdir -p "$RAW" "$RAW/stock" "$RAW/fork"

echo "[R6.1B] approved GPU: $GPU_ID  attempt: $ATTEMPT"
STOCK_HEAD=$(cd /sgl-workspace/sglang && git rev-parse HEAD)
FORK_HEAD=$(cd /data/sglang-fork && git rev-parse HEAD)
[[ "$STOCK_HEAD" == "$STOCK_HEAD_EXPECTED" ]] || { echo "stock SHA drift $STOCK_HEAD" >&2; exit 65; }
[[ "$FORK_HEAD"  == "$FORK_HEAD_EXPECTED"  ]] || { echo "fork SHA drift $FORK_HEAD" >&2; exit 65; }
echo "[R6.1B] provenance OK (stock=$STOCK_HEAD fork=$FORK_HEAD)"

# --------------------------------------------------------------------------
# 1b. Libcuda preflight + CUDA smoke
# --------------------------------------------------------------------------
if ! python3 "$PREFLIGHT_PY" --gpu "$GPU_ID" 2>&1 | tee "$RAW/preflight.log"; then
  echo "[R6.1B] preflight FAILED" >&2; exit 65
fi
sleep 3
mem_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
[[ "$mem_after" -le 500 ]] || { echo "post-preflight mem>500" >&2; exit 65; }

# Launch context
GPU_ID_ARG="$GPU_ID" ATTEMPT_ARG="$ATTEMPT" \
  python3 - > "$RAW/launch_context.json" <<'PYEOF'
import json, os, socket, subprocess
from datetime import datetime, timezone
GPU_ID = os.environ["GPU_ID_ARG"]; ATTEMPT = os.environ["ATTEMPT_ARG"]
def cmd(a): return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
pids = cmd(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits","-i",GPU_ID])
print(json.dumps({
    "launched_by": "run_R6_1_repeated_shape",
    "attempt_dir": ATTEMPT,
    "selected_gpu_id": int(GPU_ID),
    "host_libcuda": os.environ.get("R6_HOST_LIBCUDA"),
    "ld_preload": os.environ.get("LD_PRELOAD"),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "prelaunch_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    "prelaunch_state": {
        "mem_mib": int(cmd(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits","-i",GPU_ID])),
        "util_pct": int(cmd(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits","-i",GPU_ID])),
        "compute_pids": [x for x in pids.splitlines() if x],
    },
    "nvidia_driver": cmd(["nvidia-smi","--query-gpu=driver_version","--format=csv,noheader","-i",GPU_ID]),
    "sglang_stock_head": cmd(["git","-C","/sgl-workspace/sglang","rev-parse","HEAD"]),
    "sglang_fork_head":  cmd(["git","-C","/data/sglang-fork","rev-parse","HEAD"]),
    "hostname": socket.gethostname(),
}, indent=2, sort_keys=True))
PYEOF

# --------------------------------------------------------------------------
# 2. PGID tracking + ownership-verified signalling
# --------------------------------------------------------------------------
declare -a TRACKED_PGIDS=()
SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; SRV_LOG=""

record_pgid () {
  local pg="$1"; [[ -z "$pg" ]] && return 0
  for e in "${TRACKED_PGIDS[@]:-}"; do [[ "$e" == "$pg" ]] && return 0; done
  TRACKED_PGIDS+=("$pg")
}

verify_ownership () {
  local pid="$1" recorded="$2"
  [[ -z "$pid" || -z "$recorded" ]] && return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cur_pg cur_comm
  cur_pg=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  cur_comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
  [[ "$cur_pg" == "$recorded" ]] || return 1
  [[ "$cur_comm" =~ ^python ]] || return 1
  return 0
}

signal_owned_pgid () {
  local pg="$1" sig="$2"; [[ -z "$pg" ]] && return 0
  local live=0
  while IFS= read -r line; do
    local pp p2
    pp=$(echo "$line" | awk '{print $1}'); p2=$(echo "$line" | awk '{print $2}')
    if [[ "$p2" == "$pg" ]] && kill -0 "$pp" 2>/dev/null; then
      local cm; cm=$(ps -o comm= -p "$pp" 2>/dev/null | tr -d ' ')
      [[ "$cm" =~ ^python ]] && { live=1; break; }
    fi
  done < <(ps -eo pid,pgid --no-headers 2>/dev/null)
  [[ "$live" -eq 0 ]] && return 0
  kill "-${sig}" "-${pg}" 2>/dev/null || true
}

pre_launch_idle_check () {
  local mem util pids
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
  echo "[R6.1B] GPU $GPU_ID pre-launch: mem=${mem}MiB util=${util}% pids=[${pids}]"
  [[ "$mem" -le 500 && "$util" -le 5 && -z "$pids" ]] || return 1
  return 0
}

check_no_foreign_gpu_procs () {
  local pids pid pgid_of foreign=()
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
    echo "[R6.1B] FOREIGN COMPUTE PIDS on GPU $GPU_ID: ${foreign[*]}" >&2
    {
      echo "gpu=$GPU_ID at=$(date -u -Iseconds)"
      for f in "${foreign[@]}"; do echo "  $f"; done
    } > "$RAW/foreign_pid_detected.txt"
    return 71
  fi
  return 0
}

write_marker () {
  # $1=kind $2=label $3=side-channel path
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  echo "R6_MARK $1 $2 $ts" >> "$3"
  echo "[R6.1B] MARK $1 $2 $ts"
}

# --------------------------------------------------------------------------
# 3. Server launcher (fresh setsid-owned server, PGID scope)
# --------------------------------------------------------------------------
launch_server () {
  local LABEL="$1" USE_FORK="$2" LOG="$3" PIDFILE="$4" MARKERS="$5"

  pre_launch_idle_check || return 1
  check_no_foreign_gpu_procs || return 71

  rm -f "$LOG" "$PIDFILE" "$MARKERS"
  local ENV_PREFIX=""
  [[ "$USE_FORK" == "yes" ]] && ENV_PREFIX="PYTHONPATH=${FORK_PY}"

  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE
  export TORCH_LOGS="recompiles_verbose,dynamic,guards,graph_breaks"
  export TORCHDYNAMO_VERBOSE=1
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1

  SRV_LABEL="$LABEL"; SRV_LOG="$LOG"

  echo "[R6.1B] launching $LABEL (fork=$USE_FORK)"
  # shellcheck disable=SC2086
  env $ENV_PREFIX \
    python3 "$SETSID_HELPER" "$PIDFILE" \
      python3 -m sglang.launch_server \
        --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
        --attention-backend flashinfer --enforce-piecewise-cuda-graph \
    > "$LOG" 2>&1 &

  local waited=0
  while [[ ! -s "$PIDFILE" && $waited -lt 100 ]]; do sleep 0.1; waited=$((waited+1)); done
  if [[ ! -s "$PIDFILE" ]]; then
    echo "[R6.1B] $LABEL: pidfile never populated" >&2
    SRV_PID=""; SRV_PGID=""; return 3
  fi
  SRV_PID=$(cat "$PIDFILE")
  SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  if [[ -z "$SRV_PGID" || "$SRV_PGID" != "$SRV_PID" ]]; then
    echo "[R6.1B] $LABEL: PGID mismatch $SRV_PGID != $SRV_PID; refusing" >&2
    SRV_PID=""; SRV_PGID=""; return 3
  fi
  record_pgid "$SRV_PGID"
  echo "[R6.1B]   $LABEL PID=$SRV_PID PGID=$SRV_PGID"

  local READY=0
  for i in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then
      READY=1; break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[R6.1B]   $LABEL DIED during startup"
      SRV_PID=""; SRV_PGID=""; return 2
    fi
    sleep 2
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[R6.1B]   $LABEL did not become ready in 20 min" >&2
    return 2
  fi
  write_marker SERVER_READY "$LABEL" "$MARKERS"
  return 0
}

teardown_server () {
  local pid="$SRV_PID" pgid="$SRV_PGID" label="$SRV_LABEL"
  if [[ -z "$pid" || -z "$pgid" ]]; then
    SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; return 0
  fi
  echo "[R6.1B] teardown $label PID=$pid PGID=$pgid"
  if verify_ownership "$pid" "$pgid"; then
    signal_owned_pgid "$pgid" TERM
    local i
    for i in $(seq 1 30); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then
      if verify_ownership "$pid" "$pgid"; then
        signal_owned_pgid "$pgid" KILL; sleep 2
      fi
    fi
  fi
  SRV_PID=""; SRV_PGID=""; SRV_LABEL=""
  local mem
  for _ in $(seq 1 30); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
    [[ "$mem" -lt 500 ]] && break; sleep 2
  done
  echo "[R6.1B]   teardown $label done (mem=${mem}MiB)"
}

cleanup_on_exit () {
  local rc=$?
  echo "[R6.1B] cleanup_on_exit rc=$rc tracked pgids=[${TRACKED_PGIDS[*]:-}]"
  local pg
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" TERM; done
  sleep 5
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" KILL; done
  return $rc
}
trap cleanup_on_exit EXIT INT TERM

# --------------------------------------------------------------------------
# 4. Bench runner (sglang.benchmark.serving directly, R1 recipe)
# --------------------------------------------------------------------------
run_bench () {
  local SIDE="$1"   # "stock" or "fork"
  local BENCH_LOG="$RAW/$SIDE/bench.log"
  local BENCH_OUT="$RAW/$SIDE/bench.jsonl"
  local MARKERS="$RAW/$SIDE/phase_markers.txt"
  rm -f "$BENCH_LOG" "$BENCH_OUT"
  check_no_foreign_gpu_procs || return 71
  write_marker BENCH_START "$SIDE" "$MARKERS"
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat \
    --base-url "http://127.0.0.1:$PORT" \
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
  write_marker BENCH_END "$SIDE" "$MARKERS"
  echo "[R6.1B] bench.$SIDE done (client rc=? intentionally not checked)"
}

# --------------------------------------------------------------------------
# 5. Stock-PCG run (expect assertion)
# --------------------------------------------------------------------------
echo "[R6.1B] ============ STOCK-PCG (repeated-shape, expected assertion) ============"
launch_server "stock-pcg" "no" \
  "$RAW/stock/server.log" "$RAW/stock/server.pid" "$RAW/stock/phase_markers.txt" \
  || echo "[R6.1B] stock server did not become ready (recording anyway)"
if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
  run_bench "stock"
fi
teardown_server

# --------------------------------------------------------------------------
# 6. Fork-PCG run (expect clean completion)
# --------------------------------------------------------------------------
echo "[R6.1B] ============ FORK-PCG (identical workload, expected clean) ============"
launch_server "fork-pcg" "yes" \
  "$RAW/fork/server.log" "$RAW/fork/server.pid" "$RAW/fork/phase_markers.txt" \
  || { echo "[R6.1B] fork server did not become ready" >&2; }
if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
  run_bench "fork"
fi
teardown_server

# --------------------------------------------------------------------------
# 7. Verdict
# --------------------------------------------------------------------------
python3 "$VERDICT_PY" \
  --in-dir "$RAW" \
  --out-md "$BASE/verdict_amended_B.md" \
  --out-json "$BASE/verdict_amended_B.json"
echo "[R6.1B] done."
