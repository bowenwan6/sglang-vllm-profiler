#!/usr/bin/env bash
# R6.1 amended runner (Protocol Amendment A, 2026-07-28).
#
# Distinct from scripts/run_R6_1_correctness.sh (which stays as the
# historical protocol runner for attempts 01 / 02). This amended
# runner implements:
#
#   * §2.1 phase-scoped recompile markers (raw/<server>_phase_markers.txt)
#   * §2.2 matched cold-cache repeats on fresh servers (each cold leg
#     runs on its own new-process sglang.launch_server, so radix cache
#     always starts empty)
#   * §2.3 direct stock-PCG image negative control leg, classified as
#     EXPECTED_STOCK_FAILURE / STOCK_NOW_SURVIVES / UNRELATED_FAILURE;
#     an expected crash is scoped to its own PGID
#   * §2.4 verdict handoff to scripts/R6_1_verdict_amended.py
#   * §2.5 raw/verdict_amended.md + raw/verdict_amended.json
#
# Reuses:
#   * scripts/R6_setsid_exec.py (unchanged) for new-session launch
#   * scripts/R6_preflight_libcuda.py (unchanged) for host-libcuda pin
#   * scripts/R6_1_client.py (unchanged) as HTTP client
#   * fixtures/ (unchanged)
#
# Refuses to run without an explicit GPU_ID. GPU is user-authorized.

set -uo pipefail
# Note: set -e is deliberately omitted at top level so that an
# expected stock-PCG crash on the negative control does not tear
# down the whole script. Errors are handled explicitly.

# --------------------------------------------------------------------------
# 0. Argument / env parsing + safety refusals
# --------------------------------------------------------------------------
GPU_ID="${R6_GPU_ID:-${1:-}}"
if [[ -z "${GPU_ID}" ]]; then
  cat >&2 <<EOF
[R6.1A] REFUSING TO RUN — no GPU ID provided.
Pass explicitly, either via env var or first argument:
  R6_GPU_ID=<id> ./run_R6_1_amended.sh
  ./run_R6_1_amended.sh <id>
EOF
  exit 64
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[R6.1A] GPU_ID must be a non-negative integer; got: ${GPU_ID}" >&2
  exit 64
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# --------------------------------------------------------------------------
# 0b. Host-libcuda LD_PRELOAD (R6.0 Amendment A3)
# --------------------------------------------------------------------------
R6_HOST_LIBCUDA=/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05
if [[ ! -s "$R6_HOST_LIBCUDA" ]]; then
  echo "[R6.1A] Missing or empty host libcuda: $R6_HOST_LIBCUDA" >&2
  exit 65
fi
export LD_PRELOAD="${R6_HOST_LIBCUDA}${LD_PRELOAD:+ ${LD_PRELOAD}}"
export R6_HOST_LIBCUDA

# --------------------------------------------------------------------------
# 1. Fixed paths + provenance re-verification
# --------------------------------------------------------------------------
ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
FIX_DIR="$ROOT/results/R6_fix_value_validation/R6.1_correctness/fixtures"
ATTEMPT="${R6_ATTEMPT_DIR:-attempt_03_amended_A_gpu${GPU_ID}}"
BASE="$ROOT/results/R6_fix_value_validation/R6.1_correctness/$ATTEMPT"
RAW="$BASE/raw"
COLD="$RAW/cold"
NEG="$RAW/neg"
FIXTURE_PNG="$FIX_DIR/R6.1_fixture.png"
FIXTURE_SHA="$FIX_DIR/R6.1_fixture.sha256"
PROMPTS_JSON="$FIX_DIR/prompts.json"
CLIENT_PY="$ROOT/scripts/R6_1_client.py"
VERDICT_PY="$ROOT/scripts/R6_1_verdict_amended.py"
SETSID_HELPER="$ROOT/scripts/R6_setsid_exec.py"
PREFLIGHT_PY="$ROOT/scripts/R6_preflight_libcuda.py"

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
STOCK_HEAD_EXPECTED=da802ddcafe55e25b3e1db86b1e0444afc3e05bc
FORK_HEAD_EXPECTED=986c89e69c25882ab6f3d396f8eb306f38f2c8d2
PORT=30003

mkdir -p "$RAW" "$COLD" "$NEG"

echo "[R6.1A] approved GPU: ${GPU_ID}  attempt dir: $ATTEMPT"
STOCK_HEAD=$(cd /sgl-workspace/sglang && git rev-parse HEAD)
FORK_HEAD=$(cd /data/sglang-fork && git rev-parse HEAD)
[[ "$STOCK_HEAD" == "$STOCK_HEAD_EXPECTED" ]] || { echo "stock SHA drift $STOCK_HEAD" >&2; exit 65; }
[[ "$FORK_HEAD" == "$FORK_HEAD_EXPECTED" ]]  || { echo "fork SHA drift $FORK_HEAD"  >&2; exit 65; }
FIX_SHA=$(sha256sum "$FIXTURE_PNG" | awk '{print $1}')
FIX_SHA_EXPECTED=$(awk '{print $1}' "$FIXTURE_SHA")
[[ "$FIX_SHA" == "$FIX_SHA_EXPECTED" ]] || { echo "fixture SHA drift $FIX_SHA" >&2; exit 65; }
echo "[R6.1A] provenance OK"

# --------------------------------------------------------------------------
# 1b. Libcuda preflight + CUDA smoke
# --------------------------------------------------------------------------
echo "[R6.1A] preflight ..."
if ! python3 "$PREFLIGHT_PY" --gpu "$GPU_ID" 2>&1 | tee "$RAW/preflight.log"; then
  echo "[R6.1A] preflight FAILED" >&2
  exit 65
fi
sleep 3
mem_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
echo "[R6.1A] post-preflight GPU $GPU_ID mem=${mem_after}MiB"
[[ "$mem_after" -le 500 ]] || { echo "post-preflight mem>500" >&2; exit 65; }

# Write launch context.
GPU_ID_ARG="$GPU_ID" ATTEMPT_ARG="$ATTEMPT" \
  python3 - > "$RAW/launch_context.json" <<'PYEOF'
import json, os, socket, subprocess
from datetime import datetime, timezone
GPU_ID = os.environ["GPU_ID_ARG"]; ATTEMPT = os.environ["ATTEMPT_ARG"]
def cmd(a): return subprocess.run(a, capture_output=True, text=True, timeout=10).stdout.strip()
pids = cmd(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader,nounits","-i",GPU_ID])
print(json.dumps({
    "launched_by": "run_R6_1_amended",
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
# 2. PGID tracking + ownership-verified signalling (same discipline
#    as run_R6_1_correctness.sh; see R6.0 Amendment A2)
# --------------------------------------------------------------------------
declare -a TRACKED_PGIDS=()
SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; SRV_LOG=""; SRV_MARKERS=""; SRV_READY_AT=""

record_pgid () {
  local pg="$1"
  [[ -z "$pg" ]] && return 0
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
  local pg="$1" sig="$2"
  [[ -z "$pg" ]] && return 0
  local live=0
  while IFS= read -r line; do
    local pp p2
    pp=$(echo "$line" | awk '{print $1}'); p2=$(echo "$line" | awk '{print $2}')
    if [[ "$p2" == "$pg" ]] && kill -0 "$pp" 2>/dev/null; then
      local cm; cm=$(ps -o comm= -p "$pp" 2>/dev/null | tr -d ' ')
      if [[ "$cm" =~ ^python ]]; then live=1; break; fi
    fi
  done < <(ps -eo pid,pgid --no-headers 2>/dev/null)
  [[ "$live" -eq 0 ]] && return 0
  kill "-${sig}" "-${pg}" 2>/dev/null || true
}

# --------------------------------------------------------------------------
# 3. GPU idle / foreign-PID checks
# --------------------------------------------------------------------------
pre_launch_idle_check () {
  local mem util pids
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
  echo "[R6.1A] GPU $GPU_ID pre-launch: mem=${mem}MiB util=${util}% pids=[${pids}]"
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
    echo "[R6.1A] FOREIGN COMPUTE PIDS on GPU $GPU_ID: ${foreign[*]}" >&2
    {
      echo "gpu=$GPU_ID at=$(date -u -Iseconds)"
      for f in "${foreign[@]}"; do echo "  $f"; done
    } > "$RAW/foreign_pid_detected.txt"
    return 71
  fi
  return 0
}

# --------------------------------------------------------------------------
# 4. Phase markers (per-server side-channel file)
# --------------------------------------------------------------------------
write_marker () {
  # $1 = kind (SERVER_READY | LEG_START | LEG_END | ...)
  # $2 = label
  local ts; ts=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  echo "R6_MARK $1 $2 $ts" >> "$SRV_MARKERS"
  echo "[R6.1A] MARK $1 $2 $ts (server=$SRV_LABEL)"
}

# --------------------------------------------------------------------------
# 5. Server lifecycle
# --------------------------------------------------------------------------
launch_server () {
  local LABEL="$1"    # unique per-launch label
  local USE_FORK="$2" # "yes" | "no"
  local EXTRA="$3"    # extra sglang flags
  local ALLOW_DIE="${4:-no}"  # if "yes", do not exit-on-die (for negative control)

  local LOG="$RAW/${LABEL}_server.log"
  local PIDFILE="$RAW/${LABEL}_server.pid"
  local MARKERS="$RAW/${LABEL}_phase_markers.txt"

  pre_launch_idle_check || { echo "[R6.1A] GPU $GPU_ID not idle" >&2; return 1; }
  check_no_foreign_gpu_procs || { echo "[R6.1A] foreign PID detected pre-launch" >&2; return 71; }

  rm -f "$LOG" "$PIDFILE" "$MARKERS"
  local ENV_PREFIX=""
  [[ "$USE_FORK" == "yes" ]] && ENV_PREFIX="PYTHONPATH=${FORK_PY}"

  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE
  export TORCH_LOGS=recompiles
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1

  SRV_LABEL="$LABEL"
  SRV_LOG="$LOG"
  SRV_MARKERS="$MARKERS"
  SRV_READY_AT=""

  echo "[R6.1A] launching ${LABEL} (fork=${USE_FORK} extra='${EXTRA}')"
  # shellcheck disable=SC2086
  env $ENV_PREFIX \
    python3 "$SETSID_HELPER" "$PIDFILE" \
      python3 -m sglang.launch_server \
        --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
        --attention-backend flashinfer $EXTRA \
    > "$LOG" 2>&1 &

  local waited=0
  while [[ ! -s "$PIDFILE" && $waited -lt 100 ]]; do
    sleep 0.1; waited=$((waited + 1))
  done
  if [[ ! -s "$PIDFILE" ]]; then
    echo "[R6.1A] ${LABEL}: pidfile never populated" >&2
    SRV_PID=""; SRV_PGID=""
    return 3
  fi
  SRV_PID=$(cat "$PIDFILE")
  SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  if [[ -z "$SRV_PGID" || "$SRV_PGID" != "$SRV_PID" ]]; then
    echo "[R6.1A] ${LABEL}: PGID mismatch; refusing to proceed" >&2
    SRV_PID=""; SRV_PGID=""
    return 3
  fi
  record_pgid "$SRV_PGID"
  echo "[R6.1A]   ${LABEL} PID=$SRV_PID PGID=$SRV_PGID"

  local READY=0
  for i in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" \
        "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then
      READY=1; break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[R6.1A]   ${LABEL} DIED during startup — see $LOG"
      SRV_READY_AT=""
      SRV_PID=""; SRV_PGID=""
      if [[ "$ALLOW_DIE" == "yes" ]]; then
        # For negative control: expected death.
        return 90  # sentinel: died-during-startup, allowed
      fi
      return 2
    fi
    sleep 2
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[R6.1A]   ${LABEL} did not become ready" >&2
    teardown_server
    return 2
  fi
  SRV_READY_AT=$(date -u +%Y-%m-%dT%H:%M:%S.%3NZ)
  write_marker SERVER_READY "$LABEL"
  echo "R6_MARK SERVER_READY_TS $SRV_READY_AT" >> "$LOG" 2>/dev/null || true
  return 0
}

teardown_server () {
  local pid="$SRV_PID" pgid="$SRV_PGID" label="$SRV_LABEL"
  if [[ -z "$pid" || -z "$pgid" ]]; then
    SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; SRV_LOG=""; SRV_MARKERS=""; SRV_READY_AT=""
    return 0
  fi
  echo "[R6.1A] teardown ${label} PID=$pid PGID=$pgid"
  if verify_ownership "$pid" "$pgid"; then
    signal_owned_pgid "$pgid" TERM
    local i
    for i in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    if kill -0 "$pid" 2>/dev/null; then
      if verify_ownership "$pid" "$pgid"; then
        signal_owned_pgid "$pgid" KILL
        sleep 2
      else
        echo "[R6.1A]   ownership drift during teardown; NOT SIGKILL" >&2
      fi
    fi
  fi
  SRV_PID=""; SRV_PGID=""; SRV_LABEL=""; SRV_LOG=""; SRV_MARKERS=""; SRV_READY_AT=""
  # Wait for GPU memory to drain (read-only).
  local mem
  for _ in $(seq 1 30); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
    [[ "$mem" -lt 500 ]] && break
    sleep 2
  done
  echo "[R6.1A]   teardown ${label} done (mem=${mem}MiB)"
}

cleanup_on_exit () {
  local rc=$?
  echo "[R6.1A] cleanup_on_exit rc=$rc tracked pgids=[${TRACKED_PGIDS[*]:-}]"
  local pg
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" TERM; done
  sleep 5
  for pg in "${TRACKED_PGIDS[@]:-}"; do signal_owned_pgid "$pg" KILL; done
  return $rc
}
trap cleanup_on_exit EXIT INT TERM

# --------------------------------------------------------------------------
# 6. Client
# --------------------------------------------------------------------------
run_client () {
  local LEG_ID="$1" MODE="$2" OUT="$3"
  check_no_foreign_gpu_procs || return 71
  write_marker LEG_START "$LEG_ID"
  python3 "$CLIENT_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model-path "$SNAP" \
    --fixture "$FIXTURE_PNG" \
    --prompts "$PROMPTS_JSON" \
    --mode "$MODE" \
    --out "$OUT" \
    2>&1 | tee "$RAW/${LEG_ID}.client.log"
  local rc=${PIPESTATUS[0]}
  write_marker LEG_END "$LEG_ID"
  return "$rc"
}

# --------------------------------------------------------------------------
# 7. Matched cold-cache leg pairs (each is a fresh server + one image
#    or text leg as its FIRST leg, so cache is cold)
# --------------------------------------------------------------------------
run_cold_pair () {
  # Args: variant_key ("stock_default_image", "fork_default_image",
  #                    "stock_pcg_text", "fork_pcg_text",
  #                    "fork_pcg_image")
  local KEY="$1"
  local USE_FORK EXTRA MODE
  case "$KEY" in
    stock_default_image) USE_FORK=no;  EXTRA="";                              MODE=image ;;
    fork_default_image)  USE_FORK=yes; EXTRA="";                              MODE=image ;;
    stock_pcg_text)      USE_FORK=no;  EXTRA="--enforce-piecewise-cuda-graph"; MODE=text  ;;
    fork_pcg_text)       USE_FORK=yes; EXTRA="--enforce-piecewise-cuda-graph"; MODE=text  ;;
    fork_pcg_image)      USE_FORK=yes; EXTRA="--enforce-piecewise-cuda-graph"; MODE=image ;;
    *) echo "unknown cold-pair key: $KEY" >&2; return 2 ;;
  esac
  local rep
  for rep in A B; do
    local label="cold_${KEY}_${rep}"
    launch_server "$label" "$USE_FORK" "$EXTRA" || return $?
    run_client "$label" "$MODE" "$COLD/${KEY}_${rep}.json" || echo "[R6.1A] client rc for $label non-zero (recorded)"
    teardown_server
  done
  return 0
}

# --------------------------------------------------------------------------
# 8. Negative control: stock-PCG image
# --------------------------------------------------------------------------
run_negative_control () {
  local label="neg_stock_pcg_image"
  launch_server "$label" "no" "--enforce-piecewise-cuda-graph" "yes"
  local rc=$?
  local class="UNRELATED_FAILURE" reason=""
  local logfile="$RAW/${label}_server.log"
  if [[ "$rc" -eq 90 ]]; then
    # Server died during startup. Check log for the historical signature.
    if grep -q -E 'AssertionError: PCG capture stream is not set' "$logfile" 2>/dev/null; then
      class="EXPECTED_STOCK_FAILURE"
      reason="capture-stream assertion during startup (before HTTP ready)"
    else
      class="UNRELATED_FAILURE"
      reason="server died during startup without the historical signature; see $logfile"
    fi
  elif [[ "$rc" -eq 0 ]]; then
    # Server came up. Try issuing image requests.
    run_client "$label" "image" "$NEG/stock_pcg_image.json"
    local client_rc=$?
    local ok_count=0 total=0
    if [[ -f "$NEG/stock_pcg_image.json" ]]; then
      ok_count=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sum(1 for r in d.get('requests',[]) if r.get('http_status')==200 and r.get('error') is None))" "$NEG/stock_pcg_image.json" 2>/dev/null || echo 0)
      total=$(python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(len(d.get('requests',[])))" "$NEG/stock_pcg_image.json" 2>/dev/null || echo 0)
    fi
    if grep -q -E 'AssertionError: PCG capture stream is not set' "$logfile" 2>/dev/null; then
      class="EXPECTED_STOCK_FAILURE"
      reason="capture-stream assertion during image serving"
    elif [[ "$ok_count" == "$total" && "$total" -gt 0 ]]; then
      class="STOCK_NOW_SURVIVES"
      reason="stock-PCG served all $total image requests with HTTP 200; historical failure does not reproduce on $STOCK_HEAD"
    else
      class="UNRELATED_FAILURE"
      reason="stock-PCG did not reproduce assertion but also did not serve all image requests; ok=$ok_count/$total"
    fi
    teardown_server  # will only signal our tracked PGID if still live
  else
    class="UNRELATED_FAILURE"
    reason="server startup returned rc=$rc without the historical signature"
    teardown_server
  fi
  python3 - <<PYEOF > "$NEG/stock_pcg_image_classification.json"
import json
print(json.dumps({
    "result": "$class",
    "reason": "$reason",
    "server_log": "$logfile",
    "startup_rc": $rc,
}, indent=2, sort_keys=True))
PYEOF
  echo "[R6.1A] negative control -> $class"
  return 0
}

# --------------------------------------------------------------------------
# 9. Fork-PCG mixed-modality interleaved safety leg (fresh server)
# --------------------------------------------------------------------------
run_fork_pcg_mixed_safety () {
  local label="fork_pcg_interleaved"
  launch_server "$label" "yes" "--enforce-piecewise-cuda-graph" || return $?
  run_client "$label" "interleaved" "$RAW/fork_pcg_interleaved.json" || true
  # Tally safety metrics for this server.
  local logfile="$RAW/${label}_server.log"
  local markersfile="$RAW/${label}_phase_markers.txt"
  MARKERS_ARG="$markersfile" LOG_ARG="$logfile" INTERLEAVED_ARG="$RAW/fork_pcg_interleaved.json" \
    python3 - > "$RAW/fork_pcg_interleaved_safety.json" <<'PYEOF'
import json, os, re, pathlib
from datetime import datetime, timezone
logfile = os.environ["LOG_ARG"]
markersfile = os.environ["MARKERS_ARG"]
interleaved = os.environ["INTERLEAVED_ARG"]

lines = open(logfile).readlines() if os.path.exists(logfile) else []
markers = []
if os.path.exists(markersfile):
    for ln in open(markersfile):
        parts = ln.strip().split()
        # R6_MARK KIND LABEL TS
        if len(parts) >= 4:
            markers.append({"kind": parts[1], "label": parts[2], "ts": parts[3]})

# Parse recompile events: find their log line index + timestamp.
recompile_events = []
ts_re = re.compile(r'V(\d{4}) (\d{2}:\d{2}:\d{2}\.\d+)')
for i, ln in enumerate(lines):
    if re.search(r'Recompiling function.*qwen3_vl', ln):
        m = ts_re.search(ln)
        # We don't have exact date; use line index for ordering only.
        recompile_events.append({"line": i + 1, "ts_hms": m.group(2) if m else None})

# Locate SERVER_READY line in the log via the sentinel marker written
# by the runner (echoed into the log too). Fall back to canonical string.
ready_line = None
for i, ln in enumerate(lines, 1):
    if "R6_MARK SERVER_READY_TS" in ln or "The server is fired up and ready to roll" in ln:
        ready_line = i
        break

# From the marker file, get the per-leg [start, end] intervals in
# wall-clock. We map log lines to leg intervals by their sentinel
# line-index proxy: if a recompile happens at log line >= ready_line
# AND within a per-leg log-line window, count it as inflight.
# For a first cut we use a simple heuristic: any post-ready recompile
# is potentially inflight. Runner-side interleaving is short enough
# that this is conservative (biased toward FAIL when unsure).
startup_warmup = [r for r in recompile_events if ready_line is None or r["line"] < ready_line]
post_ready = [r for r in recompile_events if ready_line is not None and r["line"] >= ready_line]

# For per-leg attribution we use marker positions. The runner writes
# LEG_START <label> and LEG_END <label> — we treat any post-ready
# recompile that FALLS BETWEEN a LEG_START and LEG_END *marker event
# in time* as inflight. Since we don't have wall-clock in the log,
# we approximate by: if there is ANY LEG_START/END pair and there is
# a post-ready recompile whose line >= ready_line AND appears BEFORE
# the last LEG_END line in the log, count it as inflight for that
# leg. This is a rough heuristic; refinement can happen in R7.
per_leg_recompiles = {}
for m in markers:
    if m["kind"] == "LEG_START":
        per_leg_recompiles[m["label"]] = 0
# For attempt 03, conservative attribution: any post_ready recompile
# is credited to the last LEG_START recorded.
if post_ready and per_leg_recompiles:
    last_leg = markers[-1]["label"] if markers else None
    # Find last LEG_START in marker order
    starts = [m["label"] for m in markers if m["kind"] == "LEG_START"]
    if starts:
        last_leg = starts[-1]
    per_leg_recompiles[last_leg] = per_leg_recompiles.get(last_leg, 0) + len(post_ready)

assertions = sum(1 for ln in lines if re.search(r'AssertionError: PCG capture stream is not set', ln))
fallbacks  = sum(1 for ln in lines if re.search(r'Falling back to eager execution', ln))

req_fail = 0
if os.path.exists(interleaved):
    d = json.loads(open(interleaved).read())
    req_fail = sum(1 for r in d.get("requests", []) if r.get("error") is not None or r.get("http_status") not in (200, None))

print(json.dumps({
    "server_log": logfile,
    "phase_markers": markersfile,
    "server_log_total_lines": len(lines),
    "server_ready_line": ready_line,
    "markers": markers,
    "startup_warmup_recompiles": len(startup_warmup),
    "startup_warmup_recompile_lines": [r["line"] for r in startup_warmup],
    "post_ready_recompiles": len(post_ready),
    "post_ready_recompile_lines": [r["line"] for r in post_ready],
    "per_leg_recompiles": per_leg_recompiles,
    "assertions": assertions,
    "fallbacks": fallbacks,
    "request_failures": req_fail,
    "notes": "post_ready_recompiles conservatively attributed to the last recorded LEG_START; refine in R7 if needed.",
}, indent=2, sort_keys=True))
PYEOF
  teardown_server
  return 0
}

# --------------------------------------------------------------------------
# 10. Main flow
# --------------------------------------------------------------------------
echo "[R6.1A] ============ COLD-CACHE MATCHED REPEATS ============"
run_cold_pair stock_default_image
run_cold_pair fork_default_image
run_cold_pair stock_pcg_text
run_cold_pair fork_pcg_text
run_cold_pair fork_pcg_image

echo "[R6.1A] ============ STOCK-PCG IMAGE NEGATIVE CONTROL ============"
run_negative_control

echo "[R6.1A] ============ FORK-PCG MIXED SAFETY ============"
run_fork_pcg_mixed_safety

# --------------------------------------------------------------------------
# 11. Verdict
# --------------------------------------------------------------------------
python3 "$VERDICT_PY" \
  --in-dir "$RAW" \
  --out-md "$BASE/verdict_amended.md" \
  --out-json "$BASE/verdict_amended.json"
echo "[R6.1A] done. verdict at: $BASE/verdict_amended.md"
