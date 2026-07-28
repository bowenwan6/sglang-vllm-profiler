#!/usr/bin/env bash
# R6.1 — Correctness gate + mixed-modality safety subtest.
#
# Predefined protocol at
#   results/R6_fix_value_validation/R6.1_correctness/protocol.md
# Verdict rules pre-declared in protocol.md and enforced by
#   scripts/R6_1_verdict.py
# Safe-cleanup + PGID-scoped signalling design documented in
#   scripts/R6_setsid_exec.py and the amendment section of
#   results/R6_fix_value_validation/R6.0_provenance.md
#
# Safety invariants (never violated):
#   * Never invoke `pkill`, `pkill -f`, `killall`, `fuser -k`, or any
#     kill-by-name / kill-by-port command.
#   * Never issue `nvidia-smi --gpu-reset`.
#   * Never signal a PID unless (i) we launched it, (ii) its current
#     PGID still equals the PGID we recorded at launch, and
#     (iii) its `comm` starts with "python" (defense-in-depth).
#   * `kill -TERM -<PGID>` is scoped to a single process group we
#     created (with new-session helper) and can only reach processes
#     that inherited that PGID from our own launch.
#   * If ownership cannot be re-proven at teardown, we log and STOP,
#     never signal.
#
# Refuses to run without an explicit approved GPU_ID (safety: never
# infer from R6.0, current availability, or any default).

set -euo pipefail

# --------------------------------------------------------------------------
# 0. Argument / env parsing + safety refusals
# --------------------------------------------------------------------------
GPU_ID="${R6_GPU_ID:-${1:-}}"
if [[ -z "${GPU_ID}" ]]; then
  cat >&2 <<EOF
[R6.1] REFUSING TO RUN — no GPU ID provided.

Pass explicitly, either via env var or first argument:
  R6_GPU_ID=<id> ./run_R6_1_correctness.sh
  ./run_R6_1_correctness.sh <id>

R6 protocol requires the caller to provide the approved GPU; the
runner never assumes a default. See the R6.0 provenance and the R6.1
protocol for the safety rationale. In autonomous execution, the
monitor script (scripts/monitor_idle_gpu.py) selects the GPU after
observing 600 s of continuous idle and passes it in.
EOF
  exit 64
fi
if ! [[ "${GPU_ID}" =~ ^[0-9]+$ ]]; then
  echo "[R6.1] GPU_ID must be a non-negative integer; got: ${GPU_ID}" >&2
  exit 64
fi
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

# --------------------------------------------------------------------------
# 1. Fixed paths + provenance re-verification
# --------------------------------------------------------------------------
ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
FIX_DIR="$ROOT/results/R6_fix_value_validation/R6.1_correctness/fixtures"
RAW_DIR="$ROOT/results/R6_fix_value_validation/R6.1_correctness/raw"
SUMMARY_DIR="$ROOT/results/R6_fix_value_validation/R6.1_correctness"
FIXTURE_PNG="$FIX_DIR/R6.1_fixture.png"
FIXTURE_SHA="$FIX_DIR/R6.1_fixture.sha256"
PROMPTS_JSON="$FIX_DIR/prompts.json"
CLIENT_PY="$ROOT/scripts/R6_1_client.py"
VERDICT_PY="$ROOT/scripts/R6_1_verdict.py"
SETSID_HELPER="$ROOT/scripts/R6_setsid_exec.py"

SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
FORK_PY=/data/sglang-fork/python
STOCK_HEAD_EXPECTED=da802ddcafe55e25b3e1db86b1e0444afc3e05bc
FORK_HEAD_EXPECTED=986c89e69c25882ab6f3d396f8eb306f38f2c8d2
PORT=30003

mkdir -p "$RAW_DIR"

echo "[R6.1] approved GPU: ${GPU_ID}"
echo "[R6.1] verifying R6.0 provenance ..."
STOCK_HEAD=$(cd /sgl-workspace/sglang && git rev-parse HEAD)
FORK_HEAD=$(cd /data/sglang-fork && git rev-parse HEAD)
if [[ "$STOCK_HEAD" != "$STOCK_HEAD_EXPECTED" ]]; then
  echo "[R6.1] stock SHA drift: expected $STOCK_HEAD_EXPECTED got $STOCK_HEAD" >&2
  exit 65
fi
if [[ "$FORK_HEAD" != "$FORK_HEAD_EXPECTED" ]]; then
  echo "[R6.1] fork SHA drift: expected $FORK_HEAD_EXPECTED got $FORK_HEAD" >&2
  exit 65
fi
FIX_SHA=$(sha256sum "$FIXTURE_PNG" | awk '{print $1}')
FIX_SHA_EXPECTED=$(awk '{print $1}' "$FIXTURE_SHA")
if [[ "$FIX_SHA" != "$FIX_SHA_EXPECTED" ]]; then
  echo "[R6.1] fixture SHA drift: expected $FIX_SHA_EXPECTED got $FIX_SHA" >&2
  exit 65
fi
if [[ ! -x "$SETSID_HELPER" ]]; then
  echo "[R6.1] setsid helper not executable: $SETSID_HELPER" >&2
  exit 65
fi
echo "[R6.1] provenance OK (stock=$STOCK_HEAD fork=$FORK_HEAD fixture=$FIX_SHA)"

# --------------------------------------------------------------------------
# 2. Per-process tracking (only signal what we launched)
# --------------------------------------------------------------------------
declare -a TRACKED_PGIDS=()
SRV_PID=""
SRV_PGID=""

record_pgid () {
  local pgid="$1"
  # Idempotent record; ignore empty.
  [[ -z "$pgid" ]] && return 0
  for existing in "${TRACKED_PGIDS[@]}"; do
    [[ "$existing" == "$pgid" ]] && return 0
  done
  TRACKED_PGIDS+=("$pgid")
}

verify_ownership () {
  # Args: PID PGID. Returns 0 iff:
  #   * PID exists
  #   * ps -o pgid= matches the recorded PGID
  #   * comm starts with "python"
  local pid="$1" recorded_pgid="$2"
  [[ -z "$pid" || -z "$recorded_pgid" ]] && return 1
  kill -0 "$pid" 2>/dev/null || return 1
  local cur_pgid cur_comm
  cur_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
  cur_comm=$(ps -o comm= -p "$pid" 2>/dev/null | tr -d ' ')
  [[ "$cur_pgid" == "$recorded_pgid" ]] || return 1
  [[ "$cur_comm" =~ ^python ]] || return 1
  return 0
}

signal_owned_pgid () {
  # Args: PGID SIGNAL. Verifies at least one live process has the
  # recorded PGID before signalling. Never signals a foreign PGID.
  local pgid="$1" sig="$2"
  [[ -z "$pgid" ]] && return 0
  # Is there any live process whose PGID == our recorded pgid?
  local live_owned=0
  # ps -o pid,pgid = all processes
  while IFS= read -r line; do
    local pp pg
    pp=$(echo "$line" | awk '{print $1}')
    pg=$(echo "$line" | awk '{print $2}')
    if [[ "$pg" == "$pgid" ]] && kill -0 "$pp" 2>/dev/null; then
      local cm
      cm=$(ps -o comm= -p "$pp" 2>/dev/null | tr -d ' ')
      if [[ "$cm" =~ ^python ]]; then
        live_owned=1
        break
      fi
    fi
  done < <(ps -eo pid,pgid --no-headers 2>/dev/null)
  if [[ "$live_owned" -eq 0 ]]; then
    return 0
  fi
  # Signal by negative PGID (targets whole process group).
  kill "-${sig}" "-${pgid}" 2>/dev/null || true
}

# --------------------------------------------------------------------------
# 3. GPU idle / foreign-PID checks (read-only)
# --------------------------------------------------------------------------
gpu_state () {
  # echoes: "mem_mib util_pct pids_csv"
  local mem util pids
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$' | tr '\n' ',' | sed 's/,$//')
  echo "$mem $util $pids"
}

pre_launch_idle_check () {
  local st mem util pids
  st=$(gpu_state)
  mem=$(echo "$st" | awk '{print $1}')
  util=$(echo "$st" | awk '{print $2}')
  pids=$(echo "$st" | awk '{print $3}')
  echo "[R6.1] GPU $GPU_ID pre-launch: mem=${mem}MiB util=${util}% pids=[${pids}]"
  if [[ "$mem" -gt 500 || "$util" -gt 5 || -n "$pids" ]]; then
    echo "[R6.1] GPU $GPU_ID no longer idle at pre-launch; aborting attempt" >&2
    return 1
  fi
  return 0
}

check_no_foreign_gpu_procs () {
  # Any compute PID on GPU_ID that is NOT part of a TRACKED PGID
  # (either == pgid or descendant of it) is foreign. Foreign presence
  # is classified as resource contention; we return 71 and let the
  # main flow tear down our own servers gracefully.
  local pids pid pgid_of foreign=()
  pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits -i "$GPU_ID" | grep -v '^$')
  for pid in $pids; do
    pgid_of=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
    if [[ -z "$pgid_of" ]]; then
      # PID disappeared between nvidia-smi and ps — ignore.
      continue
    fi
    local is_ours=0 our_pgid
    for our_pgid in "${TRACKED_PGIDS[@]:-}"; do
      [[ -z "$our_pgid" ]] && continue
      if [[ "$pgid_of" == "$our_pgid" ]]; then
        is_ours=1
        break
      fi
    done
    if [[ "$is_ours" -eq 0 ]]; then
      foreign+=("pid=$pid pgid=$pgid_of")
    fi
  done
  if [[ ${#foreign[@]} -gt 0 ]]; then
    echo "[R6.1] FOREIGN COMPUTE PIDS on GPU $GPU_ID: ${foreign[*]}" >&2
    {
      echo "gpu=$GPU_ID at=$(date -u -Iseconds)"
      for f in "${foreign[@]}"; do echo "  $f"; done
    } > "$RAW_DIR/foreign_pid_detected.txt"
    return 71
  fi
  return 0
}

# --------------------------------------------------------------------------
# 4. Server lifecycle
# --------------------------------------------------------------------------
launch_server () {
  local LABEL="$1"    # e.g. "stock-default"
  local USE_FORK="$2" # "yes" | "no"
  local EXTRA="$3"    # extra sglang flags string (may be empty)
  local LOG="$RAW_DIR/${LABEL}_server.log"
  local PIDFILE="$RAW_DIR/${LABEL}_server.pid"

  pre_launch_idle_check || return 1

  rm -f "$LOG" "$PIDFILE"
  local ENV_PREFIX=""
  if [[ "$USE_FORK" == "yes" ]]; then
    ENV_PREFIX="PYTHONPATH=${FORK_PY}"
  fi

  # Correctness gate: capture Dynamo recompiles. Unset KAPI / profiler
  # env vars per protocol §2.
  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE
  export TORCH_LOGS=recompiles
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1

  echo "[R6.1] launching ${LABEL} (fork=${USE_FORK} extra='${EXTRA}')"
  # Use the setsid helper so the launched python3 is a new session
  # leader (PID == PGID == SID). The helper writes its own PID to
  # $PIDFILE BEFORE exec, so we know exactly which PID to trust.
  # shellcheck disable=SC2086
  env $ENV_PREFIX \
    python3 "$SETSID_HELPER" "$PIDFILE" \
      python3 -m sglang.launch_server \
        --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
        --attention-backend flashinfer $EXTRA \
    > "$LOG" 2>&1 &

  # Wait for pidfile to appear (up to 10 s).
  local waited=0
  while [[ ! -s "$PIDFILE" && $waited -lt 100 ]]; do
    sleep 0.1
    waited=$((waited + 1))
  done
  if [[ ! -s "$PIDFILE" ]]; then
    echo "[R6.1] ${LABEL}: pidfile never populated" >&2
    SRV_PID=""
    SRV_PGID=""
    return 3
  fi
  SRV_PID=$(cat "$PIDFILE")
  SRV_PGID=$(ps -o pgid= -p "$SRV_PID" 2>/dev/null | tr -d ' ')
  if [[ -z "$SRV_PGID" ]]; then
    echo "[R6.1] ${LABEL}: server PID $SRV_PID has no PGID (died instantly?)" >&2
    SRV_PID=""
    SRV_PGID=""
    return 3
  fi
  if [[ "$SRV_PGID" != "$SRV_PID" ]]; then
    echo "[R6.1] ${LABEL}: PGID $SRV_PGID != PID $SRV_PID; setsid helper failed; refusing to proceed (will not signal)" >&2
    SRV_PID=""
    SRV_PGID=""
    return 3
  fi
  record_pgid "$SRV_PGID"
  echo "[R6.1]   ${LABEL} PID=$SRV_PID PGID=$SRV_PGID"

  # Wait for HTTP readiness.
  local READY=0 i
  for i in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" \
        "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then
      READY=1
      echo "[R6.1]   ${LABEL} ready after ~$((i * 2)) s"
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[R6.1]   ${LABEL} DIED during startup — see $LOG" >&2
      # Server died on its own; nothing to signal.
      SRV_PID=""
      SRV_PGID=""
      return 2
    fi
    sleep 2
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[R6.1]   ${LABEL} did not become ready" >&2
    teardown_server "$LABEL"
    return 2
  fi
  return 0
}

teardown_server () {
  local LABEL="$1"
  if [[ -z "${SRV_PID:-}" || -z "${SRV_PGID:-}" ]]; then
    return 0
  fi
  echo "[R6.1] teardown ${LABEL} PID=$SRV_PID PGID=$SRV_PGID"
  if verify_ownership "$SRV_PID" "$SRV_PGID"; then
    signal_owned_pgid "$SRV_PGID" TERM
    local i
    for i in $(seq 1 30); do
      if ! kill -0 "$SRV_PID" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$SRV_PID" 2>/dev/null; then
      # Re-verify ownership before SIGKILL.
      if verify_ownership "$SRV_PID" "$SRV_PGID"; then
        echo "[R6.1]   SIGKILL group $SRV_PGID after grace"
        signal_owned_pgid "$SRV_PGID" KILL
        sleep 2
      else
        echo "[R6.1]   ownership drift during teardown; NOT SIGKILL; investigation required" >&2
      fi
    fi
  else
    # Cannot prove ownership — do NOT signal.
    if kill -0 "${SRV_PID:-0}" 2>/dev/null; then
      echo "[R6.1]   ownership cannot be proven for PID $SRV_PID PGID $SRV_PGID; NOT signaling; investigation required" >&2
    fi
  fi
  SRV_PID=""
  SRV_PGID=""
  # Read-only wait for GPU memory to drain (best effort; DO NOT signal anything).
  local mem
  for _ in $(seq 1 30); do
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
    [[ "$mem" -lt 500 ]] && break
    sleep 2
  done
  echo "[R6.1]   teardown ${LABEL} done (post-teardown mem=${mem}MiB)"
}

cleanup_on_exit () {
  local rc=$?
  echo "[R6.1] cleanup_on_exit rc=$rc; tracked pgids=[${TRACKED_PGIDS[*]:-}]"
  # First pass: SIGTERM every tracked PGID (only if still live and PGID matches).
  local pg
  for pg in "${TRACKED_PGIDS[@]:-}"; do
    signal_owned_pgid "$pg" TERM
  done
  sleep 5
  # Second pass: SIGKILL any tracked PGID that is still alive.
  for pg in "${TRACKED_PGIDS[@]:-}"; do
    signal_owned_pgid "$pg" KILL
  done
  return $rc
}
trap cleanup_on_exit EXIT INT TERM

# --------------------------------------------------------------------------
# 5. Client invocation helper
# --------------------------------------------------------------------------
run_client () {
  local LABEL="$1"
  local MODE="$2"
  local OUT="$RAW_DIR/${LABEL}.json"
  # Foreign-PID check: any compute PID on GPU_ID that is not a
  # descendant of our tracked PGIDs is contention.
  if ! check_no_foreign_gpu_procs; then
    echo "[R6.1] foreign PID detected before ${LABEL}; aborting" >&2
    return 71
  fi
  echo "[R6.1] client: leg=${LABEL} mode=${MODE}"
  python3 "$CLIENT_PY" \
    --base-url "http://127.0.0.1:$PORT" \
    --model-path "$SNAP" \
    --fixture "$FIXTURE_PNG" \
    --prompts "$PROMPTS_JSON" \
    --mode "$MODE" \
    --out "$OUT" \
    | tee "$RAW_DIR/${LABEL}.client.log"
}

# --------------------------------------------------------------------------
# 6. Log-scan helper for the mixed-safety subtest
# --------------------------------------------------------------------------
tally_safety () {
  local LABEL="$1"
  local LOG="$RAW_DIR/${LABEL}_server.log"
  local OUT="$RAW_DIR/safety_summary.json"
  local INTERLEAVED_JSON="$RAW_DIR/leg_e_fork_pcg_interleaved.json"

  local assertions fallbacks recompiles req_fail
  assertions=$(grep -c -E 'AssertionError: PCG capture stream is not set' "$LOG" 2>/dev/null || echo 0)
  fallbacks=$(grep -c -E 'Falling back to eager execution' "$LOG" 2>/dev/null || echo 0)
  recompiles=$(grep -c -E 'Recompiling function.*qwen3_vl' "$LOG" 2>/dev/null || echo 0)
  req_fail=$(python3 - <<PYEOF
import json, sys, pathlib
p = pathlib.Path("$INTERLEAVED_JSON")
if not p.exists():
    print(-1); sys.exit()
d = json.loads(p.read_text())
print(sum(1 for r in d.get("requests", []) if r.get("error") is not None or r.get("http_status") not in (200, None)))
PYEOF
)

  python3 - <<PYEOF > "$OUT"
import json
print(json.dumps({
    "path": "$OUT",
    "server_log": "$LOG",
    "assertions": int("$assertions"),
    "fallbacks": int("$fallbacks"),
    "inference_recompiles": int("$recompiles"),
    "request_failures": int("$req_fail"),
    "notes": "recompiles counted by 'Recompiling function.*qwen3_vl' grep on the fork-PCG server log; assertion count on 'AssertionError: PCG capture stream is not set'; fallback count on 'Falling back to eager execution'.",
}, indent=2, sort_keys=True))
PYEOF
  echo "[R6.1] safety summary written to $OUT"
  cat "$OUT"
}

# --------------------------------------------------------------------------
# 7. Legs (serialized; one server at a time; PGID-scoped teardown)
# --------------------------------------------------------------------------
if ! check_no_foreign_gpu_procs; then
  echo "[R6.1] foreign compute PID present on GPU $GPU_ID at start; aborting" >&2
  exit 71
fi

# ---- variant 1: stock-default ---------------------------------------------
launch_server "stock-default"  "no"  ""
run_client    "leg_c_stock_default_image" "image"
run_client    "leg_f_stock_default_text"  "text"
teardown_server "stock-default"

# ---- variant 2: fork-default ----------------------------------------------
launch_server "fork-default"   "yes" ""
run_client    "leg_a_fork_default_run1" "image"
run_client    "leg_a_fork_default_run2" "image"
teardown_server "fork-default"

# ---- variant 3: stock-PCG (text only) -------------------------------------
launch_server "stock-pcg"      "no"  "--enforce-piecewise-cuda-graph"
run_client    "leg_d_stock_pcg_text" "text"
run_client    "leg_f_stock_pcg_text" "text"
teardown_server "stock-pcg"

# ---- variant 4: fork-PCG (image + text + interleaved) ---------------------
launch_server "fork-pcg"       "yes" "--enforce-piecewise-cuda-graph"
run_client    "leg_b_fork_pcg_image"          "image"
run_client    "leg_dprime_fork_pcg_text"      "text"
run_client    "leg_e_fork_pcg_interleaved"    "interleaved"
tally_safety  "fork-pcg"
teardown_server "fork-pcg"

# --------------------------------------------------------------------------
# 8. Verdict computation
# --------------------------------------------------------------------------
python3 "$VERDICT_PY" \
  --in-dir  "$RAW_DIR" \
  --out-md  "$SUMMARY_DIR/verdict.md" \
  --out-json "$SUMMARY_DIR/verdict.json"

echo "[R6.1] done. verdict: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$SUMMARY_DIR/verdict.json")"
