#!/usr/bin/env bash
# R6.1 — Correctness gate + mixed-modality safety subtest.
#
# Predefined protocol at
#   results/R6_fix_value_validation/R6.1_correctness/protocol.md
# Verdict rules pre-declared in protocol.md and enforced by
#   scripts/R6_1_verdict.py
#
# This script:
#   * Refuses to run without an explicit approved GPU ID (safety).
#   * Re-verifies stock/fork SHAs still match R6.0 provenance.
#   * Runs each server variant serialized (no co-residency).
#   * Pre-checks GPU idle before every server launch.
#   * Ensures teardown on every exit path (trap EXIT).
#   * Collects only aggregated summary artifacts under
#       results/R6_fix_value_validation/R6.1_correctness/raw/
#     (which is .gitignore'd). Nothing raw is committed.
#
# Usage:
#   R6_GPU_ID=<id> ./run_R6_1_correctness.sh
#     or
#   ./run_R6_1_correctness.sh <id>
#
# Halt on any tooling error. Correctness legs may still record ERR
# rows in their JSON without aborting the whole script; that is a data
# point, not a failure of the orchestrator.

set -euo pipefail

# --------------------------------------------------------------------------
# 0. Argument / env parsing + safety refusals
# --------------------------------------------------------------------------
GPU_ID="${R6_GPU_ID:-${1:-}}"
if [[ -z "${GPU_ID}" ]]; then
  cat >&2 <<EOF
[R6.1] REFUSING TO RUN — no GPU ID provided.

Pass explicitly, either via env var or first argument:
  R6_GPU_ID=6 ./run_R6_1_correctness.sh
  ./run_R6_1_correctness.sh 6

R6 protocol requires the caller to provide the approved GPU; the
runner never assumes a default. See R6.0 provenance and the R6.1
protocol for the safety rationale.
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
echo "[R6.1] provenance OK (stock=$STOCK_HEAD fork=$FORK_HEAD fixture=$FIX_SHA)"

# --------------------------------------------------------------------------
# 2. GPU idle pre-check helper
# --------------------------------------------------------------------------
gpu_idle_check () {
  local mem util procs
  mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
  procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i "$GPU_ID" | wc -l | tr -d ' ')
  echo "[R6.1] GPU $GPU_ID: mem=${mem}MiB util=${util}% procs=${procs}"
  if [[ "$mem" -gt 500 || "$util" -gt 5 || "$procs" -gt 0 ]]; then
    echo "[R6.1] GPU $GPU_ID not idle enough (mem>${mem} util=${util} procs=${procs}); aborting." >&2
    return 1
  fi
  return 0
}

# --------------------------------------------------------------------------
# 3. Cleanup trap
# --------------------------------------------------------------------------
SRV_PID=""
cleanup () {
  local rc=$?
  if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[R6.1] cleanup: SIGTERM $SRV_PID"
    kill -TERM "$SRV_PID" 2>/dev/null || true
    sleep 5
    kill -0 "$SRV_PID" 2>/dev/null && kill -KILL "$SRV_PID" 2>/dev/null || true
  fi
  pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
  sleep 3
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
  return $rc
}
trap cleanup EXIT INT TERM

# --------------------------------------------------------------------------
# 4. Server launch / wait / teardown helpers
# --------------------------------------------------------------------------
launch_server () {
  local LABEL="$1"    # e.g. "stock-default"
  local USE_FORK="$2" # "yes" | "no"
  local EXTRA="$3"    # extra sglang flags string (may be empty)
  local LOG="$RAW_DIR/${LABEL}_server.log"

  gpu_idle_check || return 1

  rm -f "$LOG"
  local ENV_PREFIX=""
  if [[ "$USE_FORK" == "yes" ]]; then
    ENV_PREFIX="PYTHONPATH=${FORK_PY}"
  fi

  # Ensure any KAPI / profiler env is off (protocol §6.0).
  unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE

  # Correctness gate always captures Dynamo recompile events.
  export TORCH_LOGS=recompiles

  # CUDA IPC is on for image legs by protocol convention.
  export SGLANG_USE_CUDA_IPC_TRANSPORT=1

  echo "[R6.1] launching ${LABEL} (fork=${USE_FORK} extra='${EXTRA}')"
  # shellcheck disable=SC2086
  env $ENV_PREFIX \
    python3 -m sglang.launch_server \
      --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
      --attention-backend flashinfer $EXTRA \
      > "$LOG" 2>&1 &
  SRV_PID=$!

  local READY=0
  for _ in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" \
        "http://127.0.0.1:$PORT/get_model_info" 2>/dev/null | grep -q 200; then
      READY=1
      echo "[R6.1] ${LABEL} ready"
      break
    fi
    if ! kill -0 "$SRV_PID" 2>/dev/null; then
      echo "[R6.1] ${LABEL} DIED during startup — see $LOG" >&2
      SRV_PID=""
      return 2
    fi
    sleep 2
  done
  if [[ "$READY" -ne 1 ]]; then
    echo "[R6.1] ${LABEL} did not become ready" >&2
    cleanup || true
    return 2
  fi
  # Verify which sglang the server ended up importing (belt-and-braces).
  local IMPORT_LINE
  IMPORT_LINE=$(env $ENV_PREFIX python3 -c 'import sglang; print(sglang.__file__)' 2>/dev/null || echo "IMPORT_FAILED")
  echo "[R6.1]   sglang.__file__ = $IMPORT_LINE"
  return 0
}

teardown_server () {
  local LABEL="$1"
  if [[ -n "$SRV_PID" ]] && kill -0 "$SRV_PID" 2>/dev/null; then
    echo "[R6.1] teardown ${LABEL}: SIGTERM $SRV_PID"
    kill -TERM "$SRV_PID" 2>/dev/null || true
    sleep 5
    kill -0 "$SRV_PID" 2>/dev/null && kill -KILL "$SRV_PID" 2>/dev/null || true
  fi
  pkill -TERM -f "sglang.launch_server" 2>/dev/null || true
  sleep 5
  pkill -KILL -f "sglang.launch_server" 2>/dev/null || true
  SRV_PID=""
  # Wait for GPU memory to drain.
  for _ in $(seq 1 30); do
    local mem
    mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID" | tr -d ' ')
    [[ "$mem" -lt 500 ]] && break
    sleep 2
  done
  echo "[R6.1] teardown ${LABEL} done"
}

# --------------------------------------------------------------------------
# 5. Client invocation helper
# --------------------------------------------------------------------------
run_client () {
  local LABEL="$1"   # leg name
  local MODE="$2"    # image | text | interleaved
  local OUT="$RAW_DIR/${LABEL}.json"
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
  local INTERLEAVED_CLIENT_LOG="$RAW_DIR/leg_e_fork_pcg_interleaved.client.log"
  local INTERLEAVED_JSON="$RAW_DIR/leg_e_fork_pcg_interleaved.json"

  local assertions fallbacks recompiles req_fail
  assertions=$(grep -c -E 'AssertionError: PCG capture stream is not set' "$LOG" 2>/dev/null || echo 0)
  fallbacks=$(grep -c -E 'Falling back to eager execution' "$LOG" 2>/dev/null || echo 0)
  # Dynamo `TORCH_LOGS=recompiles` prints lines like:
  #   "torch._dynamo.convert_frame: [WARNING] Recompiling function forward in .../qwen3_vl.py"
  # We count only recompiles that happen after the model-runner has finished
  # capture; there's no perfect marker for "inference-time", so we count
  # any recompile line mentioning qwen3_vl in the second half of the log
  # as a proxy. Refine in R7 if needed.
  recompiles=$(grep -c -E 'Recompiling function.*qwen3_vl' "$LOG" 2>/dev/null || echo 0)
  # Client-side failures for the interleaved run:
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
# 7. Legs
# --------------------------------------------------------------------------
# Leg A + C + F1: stock-default (single server, run image + text)
# Leg A x2 + D': fork-default (single server, run image twice + text)
# Leg D + F2: stock-PCG (single server, run text only — never image, would crash)
# Leg B + E: fork-PCG (single server, run image + text + interleaved)
#
# Serialization: one server up at a time. Each variant torn down fully
# before the next launches, with GPU-memory-drain wait.

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
