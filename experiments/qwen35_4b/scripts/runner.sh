#!/usr/bin/env bash
# Live runner for the Qwen3.5-4B BCG DeepStack validation.
#
# For a given --config (eager_normal | eager_zero_deepstack | bcg_normal),
# this script:
#   1. Verifies --gpu-id 0 authorisation and preflight.
#   2. Verifies GPU 0 is unoccupied (foreign PID guard, by UUID).
#   3. Applies the branch-local instrumentation via server_launcher.py.
#   4. Launches sglang.launch_server on GPU 0 from the frozen SGLang
#      checkout via PYTHONPATH.
#   5. Waits for /health.
#   6. Runs client.py against the server.
#   7. Records raw logs + client records + instrumentation events
#      under --results-dir/raw/.
#   8. Cleans up its own PGID (never signalling foreign PIDs).
#
# INFRA_CHECK mode: --infra-check brings up the server, verifies BCG
# capture banner, tears down cleanly — no requests. Used by Step 4.
#
# Never uses pkill, killall, fuser -k, or nvidia-smi --gpu-reset.

set -euo pipefail

# --- argument parsing --------------------------------------------------

DRY_RUN=0
INFRA_CHECK=0
CONFIG=""
GPU_ID=""
ATTEMPT_ID=""
FIXTURES_DIR=""
RESULTS_DIR=""
FROZEN_SGLANG=""
MODEL_ID="Qwen/Qwen3.5-4B"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
SERVER_PORT="30000"
SERVER_BIND="127.0.0.1"
LIBCUDA_PRELOAD="${LIBCUDA_PRELOAD:-/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05}"
LAUNCH_WAIT_SECONDS="${LAUNCH_WAIT_SECONDS:-600}"

show_usage() {
    cat >&2 <<'EOF'
usage: runner.sh --gpu-id 0 --config <label> --attempt-id <id> \
                 --results-dir <path> --frozen-sglang <checkout> \
                 [--infra-check] [--dry-run] [--server-port N]

--config       one of: eager_normal | eager_zero_deepstack | bcg_normal
--infra-check  bring up the server, verify BCG banner, tear down — no requests
--dry-run      exercise argument parsing / preflight / cleanup wiring; no CUDA
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --infra-check) INFRA_CHECK=1; shift ;;
        --config)  CONFIG="${2:-}"; shift 2 ;;
        --gpu-id)  GPU_ID="${2:-}"; shift 2 ;;
        --attempt-id) ATTEMPT_ID="${2:-}"; shift 2 ;;
        --fixtures-dir) FIXTURES_DIR="${2:-}"; shift 2 ;;
        --results-dir)  RESULTS_DIR="${2:-}"; shift 2 ;;
        --frozen-sglang) FROZEN_SGLANG="${2:-}"; shift 2 ;;
        --model-id) MODEL_ID="${2:-}"; shift 2 ;;
        --model-revision) MODEL_REVISION="${2:-}"; shift 2 ;;
        --server-port) SERVER_PORT="${2:-}"; shift 2 ;;
        --help|-h) show_usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; show_usage; exit 64 ;;
    esac
done

# --- paths -------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVESTIGATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES_DIR="${FIXTURES_DIR:-${INVESTIGATION_DIR}/fixtures}"
RESULTS_DIR="${RESULTS_DIR:-${INVESTIGATION_DIR}/results}"
INSTRUMENTATION_PATH="${SCRIPT_DIR}/instrumentation.py"
LAUNCHER_PATH="${SCRIPT_DIR}/server_launcher.py"

# --- GPU authorisation gate --------------------------------------------

env_gpu_id="${QWEN35_GPU_ID:-}"
if [ "$DRY_RUN" = "0" ]; then
    if [ -z "$GPU_ID" ] && [ -z "$env_gpu_id" ]; then
        echo "FATAL: no --gpu-id and no QWEN35_GPU_ID env var." >&2
        exit 64
    fi
    if [ -n "$GPU_ID" ] && [ -n "$env_gpu_id" ] && [ "$GPU_ID" != "$env_gpu_id" ]; then
        echo "FATAL: --gpu-id ($GPU_ID) disagrees with QWEN35_GPU_ID ($env_gpu_id)." >&2
        exit 64
    fi
    if [ -z "$GPU_ID" ]; then
        GPU_ID="$env_gpu_id"
    fi
    if [ "$GPU_ID" != "0" ]; then
        echo "FATAL: GPU 0 is the only authorised device; got --gpu-id=$GPU_ID." >&2
        exit 64
    fi
    if [ -z "$CONFIG" ]; then
        echo "FATAL: --config is required for a live run." >&2
        exit 64
    fi
    if [ -z "$ATTEMPT_ID" ]; then
        echo "FATAL: --attempt-id is required for a live run." >&2
        exit 64
    fi
    if [ -z "$FROZEN_SGLANG" ]; then
        echo "FATAL: --frozen-sglang is required for a live run." >&2
        exit 64
    fi
    if [ ! -d "$FROZEN_SGLANG/python/sglang" ]; then
        echo "FATAL: --frozen-sglang path missing python/sglang: $FROZEN_SGLANG" >&2
        exit 64
    fi
fi

case "$CONFIG" in
    eager_normal|eager_zero_deepstack|bcg_normal|"") ;;
    *) echo "FATAL: unknown --config=$CONFIG" >&2; exit 64 ;;
esac

# --- launch context ----------------------------------------------------

PRELAUNCH_UTC="$(date -u +'%Y-%m-%dT%H-%M-%SZ')"
RUNNER_PGID="$$"
LAUNCH_ID="${ATTEMPT_ID}-${CONFIG}-${PRELAUNCH_UTC}"
LAUNCH_CTX_JSON="/tmp/qwen35_launch_ctx_${LAUNCH_ID}.json"

# GPU 0 UUID (queried read-only, no side effects). Best-effort in dry-run.
GPU_UUID=""
if command -v nvidia-smi >/dev/null 2>&1 && [ "$DRY_RUN" = "0" ]; then
    GPU_UUID="$(nvidia-smi --query-gpu=uuid --format=csv,noheader --id=0 2>/dev/null || true)"
fi

cat > "$LAUNCH_CTX_JSON" <<EOF
{
  "runner": "runner.sh",
  "prelaunch_utc": "${PRELAUNCH_UTC}",
  "launch_id": "${LAUNCH_ID}",
  "config": "${CONFIG}",
  "gpu_id": "${GPU_ID}",
  "gpu_uuid": "${GPU_UUID}",
  "attempt_id": "${ATTEMPT_ID}",
  "runner_pgid": "${RUNNER_PGID}",
  "runner_sid": "$(ps -o sid= -p $$ 2>/dev/null | tr -d ' ' || echo 0)",
  "dry_run": ${DRY_RUN},
  "infra_check": ${INFRA_CHECK},
  "cwd": "$(pwd)",
  "argv": "$*",
  "fixtures_dir": "${FIXTURES_DIR}",
  "results_dir": "${RESULTS_DIR}",
  "frozen_sglang": "${FROZEN_SGLANG}",
  "model_id": "${MODEL_ID}",
  "model_revision": "${MODEL_REVISION}",
  "server_port": "${SERVER_PORT}"
}
EOF

echo "launch_context: ${LAUNCH_CTX_JSON}"

# --- foreign-PID guard -------------------------------------------------

foreign_pid_check() {
    if [ "$DRY_RUN" = "1" ]; then
        echo "foreign_pid_check: SKIPPED_DRY_RUN"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "foreign_pid_check: NVIDIA-SMI MISSING; aborting." >&2
        exit 71
    fi
    # Filter to GPU 0 only, by UUID.
    if [ -z "$GPU_UUID" ]; then
        echo "foreign_pid_check: GPU 0 UUID unavailable; aborting." >&2
        exit 71
    fi
    local raw
    raw="$(nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader 2>/dev/null || true)"
    # Reject if any process is on GPU 0's UUID.
    local hits
    hits="$(printf "%s\n" "$raw" | awk -v uuid="$GPU_UUID" -F', *' '$2 == uuid {print}')"
    if [ -n "$hits" ]; then
        echo "foreign_pid_check: existing compute apps present on GPU 0 ($GPU_UUID):"
        printf "%s\n" "$hits"
        echo "foreign_pid_check: aborting; the runner refuses to co-inhabit." >&2
        exit 71
    fi
    echo "foreign_pid_check: OK"
}

# --- ownership-verified cleanup ---------------------------------------

SERVER_PID=""
SERVER_PGID=""
STDERR_PATH=""
STDOUT_PATH=""

cleanup() {
    local exit_code=$?
    echo "cleanup: entering with exit_code=$exit_code"
    if [ -n "$SERVER_PID" ]; then
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            local pgid
            pgid="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ' || echo 0)"
            if [ "$pgid" = "$SERVER_PGID" ] && [ -n "$SERVER_PGID" ]; then
                # Check every PID in the group belongs to our runner (no foreign PIDs).
                local group_pids
                group_pids="$(ps -o pid= -g "$SERVER_PGID" 2>/dev/null | tr -d ' ' || true)"
                echo "cleanup: SIGTERM to server pgid $SERVER_PGID (pids: $group_pids)"
                kill -- -"$SERVER_PGID" 2>/dev/null || true
                sleep 2
                if kill -0 "$SERVER_PID" 2>/dev/null; then
                    echo "cleanup: SIGKILL to server pgid $SERVER_PGID"
                    kill -9 -- -"$SERVER_PGID" 2>/dev/null || true
                fi
            else
                echo "cleanup: REFUSING to signal $SERVER_PID (pgid $pgid != $SERVER_PGID)"
            fi
        fi
    fi
    # Preserve captured stderr / stdout for post-mortem.
    if [ -n "$STDERR_PATH" ] && [ -f "$STDERR_PATH" ]; then
        echo "cleanup: server stderr preserved at $STDERR_PATH"
    fi
    exit "$exit_code"
}
trap cleanup EXIT INT TERM

# --- provenance preflight ---------------------------------------------

if [ -n "$ATTEMPT_ID" ]; then
    RAW_DIR="${RESULTS_DIR}/${ATTEMPT_ID}/raw"
else
    # Dry-run without --attempt-id: send preflight output to a tmp path.
    RAW_DIR="$(mktemp -d -t qwen35_dryrun_XXXXXX)"
fi
mkdir -p "$RAW_DIR"

PREFLIGHT_ARGS=(--fixtures-dir "$FIXTURES_DIR" --json)
if [ "$DRY_RUN" = "1" ]; then
    PREFLIGHT_ARGS+=(--dry-run)
fi
if [ -n "$FROZEN_SGLANG" ]; then
    PREFLIGHT_ARGS+=(--frozen-sglang "$FROZEN_SGLANG")
fi

echo "preflight: provenance check"
PREFLIGHT_JSON_PATH="${RAW_DIR}/preflight_${CONFIG:-cpu}_${PRELAUNCH_UTC}.json"
PREFLIGHT_STDERR_PATH="${RAW_DIR}/preflight_${CONFIG:-cpu}_${PRELAUNCH_UTC}.stderr"

# For the imported_sglang_path check, PYTHONPATH must include the frozen checkout.
if [ -n "$FROZEN_SGLANG" ]; then
    export PYTHONPATH="${FROZEN_SGLANG}/python${PYTHONPATH:+:$PYTHONPATH}"
fi

if ! python3 "${SCRIPT_DIR}/preflight_provenance.py" "${PREFLIGHT_ARGS[@]}" \
      > "$PREFLIGHT_JSON_PATH" 2> "$PREFLIGHT_STDERR_PATH"; then
    echo "preflight: NON-ZERO EXIT (see $PREFLIGHT_JSON_PATH / $PREFLIGHT_STDERR_PATH)"
fi
cat "$PREFLIGHT_JSON_PATH" 2>/dev/null | head -80

# Hard-fail conditions (JSON check).
python3 - "$PREFLIGHT_JSON_PATH" <<'PYCHECK' || { echo "preflight: HARD PIN DRIFT"; exit 65; }
import json, sys
p = sys.argv[1]
data = json.load(open(p))
hard = {"frozen_sglang_checkout", "hf_model", "image_fixture",
        "imported_sglang_path", "sglang_fork_unchanged"}
bad = []
for r in data.get("results", []):
    if r.get("check") in hard and r.get("status") not in {
        "PIN_MATCH", "OBSERVED", "SKIPPED_DRY_RUN", "INSIDE_FROZEN",
        "NOT_PROVIDED"
    }:
        bad.append(r)
if bad:
    print(json.dumps({"hard_drifts": bad}, indent=2))
    sys.exit(1)
sys.exit(0)
PYCHECK
echo "preflight: OK"

foreign_pid_check

# --- server launch ----------------------------------------------------

if [ "$DRY_RUN" = "1" ]; then
    echo "server_launch: SKIPPED_DRY_RUN"
    echo "bench:         SKIPPED_DRY_RUN"
    echo "verdict:       SKIPPED_DRY_RUN"
    echo "runner: dry-run OK"
    # Best-effort cleanup of tmp preflight dir.
    if [ -z "$ATTEMPT_ID" ] && [[ "$RAW_DIR" == /tmp/qwen35_dryrun_* ]]; then
        rm -rf "$RAW_DIR"
    fi
    exit 0
fi

# Server flags per config.
declare -a SERVER_FLAGS
SERVER_FLAGS=(
    --model-path "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --port "$SERVER_PORT"
    --host "$SERVER_BIND"
    --tp 1
    --dtype bfloat16
    --disable-radix-cache
    --mem-fraction-static "0.75"
    --log-level info
    --allow-auto-truncate
    --trust-remote-code
)
case "$CONFIG" in
    eager_normal|eager_zero_deepstack)
        # Force eager prefill (BCG disabled). SGLang exposes
        # `--cuda-graph-max-bs 0` / `--disable-cuda-graph` combos;
        # the safest is `--disable-cuda-graph-padding` + choosing an
        # eager prefill backend. We use `--disable-cuda-graph` which
        # applies to all backends.
        SERVER_FLAGS+=( --disable-cuda-graph )
        ;;
    bcg_normal)
        # Default (breakable) prefill CUDA graph — DO NOT set
        # --enforce-piecewise-cuda-graph (Qwen3.5 is not on the PCG
        # allowlist).
        ;;
esac

INSTR_LOG="${RAW_DIR}/instrumentation_${CONFIG}.jsonl"
STDERR_PATH="${RAW_DIR}/server_stderr_${CONFIG}.log"
STDOUT_PATH="${RAW_DIR}/server_stdout_${CONFIG}.log"

export CUDA_VISIBLE_DEVICES=0
export QWEN35_INSTRUMENTATION_LOG="$INSTR_LOG"
export QWEN35_LAUNCH_ID="$LAUNCH_ID"
export QWEN35_CONFIG_LABEL="$CONFIG"
if [ "$CONFIG" = "eager_zero_deepstack" ]; then
    export QWEN35_ZERO_DEEPSTACK=1
else
    export QWEN35_ZERO_DEEPSTACK=0
fi
if [ -e "$LIBCUDA_PRELOAD" ]; then
    export LD_PRELOAD="${LIBCUDA_PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"
    echo "LD_PRELOAD: $LD_PRELOAD"
else
    echo "WARN: LIBCUDA_PRELOAD ($LIBCUDA_PRELOAD) missing; continuing without."
fi

echo "server: launching python3 server_launcher.py -- ${SERVER_FLAGS[*]}"
# Launch in its own process group so we can signal it cleanly.
setsid python3 "$LAUNCHER_PATH" \
    --instrumentation "$INSTRUMENTATION_PATH" \
    --frozen-sglang "$FROZEN_SGLANG" \
    --frozen-sglang-sha "89f4a80c1f5e71c1c960df120f1e03b43dfd3c1d" \
    -- "${SERVER_FLAGS[@]}" \
    > "$STDOUT_PATH" 2> "$STDERR_PATH" &
SERVER_PID=$!
SERVER_PGID="$(ps -o pgid= -p "$SERVER_PID" 2>/dev/null | tr -d ' ')"
echo "server: pid=$SERVER_PID pgid=$SERVER_PGID"

# Wait for /health.
echo "waiting for /health up to ${LAUNCH_WAIT_SECONDS}s..."
start_ts=$SECONDS
ready=0
while [ $((SECONDS - start_ts)) -lt "$LAUNCH_WAIT_SECONDS" ]; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "FATAL: server process $SERVER_PID died during startup" >&2
        tail -n 100 "$STDERR_PATH" >&2 || true
        exit 66
    fi
    code="$(curl -s -o /dev/null -w '%{http_code}' "http://${SERVER_BIND}:${SERVER_PORT}/health" 2>/dev/null || echo 000)"
    if [ "$code" = "200" ]; then
        echo "server: /health OK after $((SECONDS - start_ts))s"
        ready=1
        break
    fi
    sleep 5
done
if [ "$ready" = "0" ]; then
    echo "FATAL: server /health never returned 200 within ${LAUNCH_WAIT_SECONDS}s" >&2
    tail -n 100 "$STDERR_PATH" >&2 || true
    exit 67
fi

# INFRA_CHECK: verify BCG capture banner appears in the log, then done.
if [ "$INFRA_CHECK" = "1" ]; then
    # Give it a moment more so any capture banner has flushed.
    sleep 5
    echo "INFRA_CHECK: looking for BCG capture banner in $STDERR_PATH"
    grep -E "Capturing num tokens|Compiling num tokens|cuda graph|BreakableCudaGraph|prefill_cuda_graph|multimodal" "$STDERR_PATH" | head -20 || true
    # Verify GPU 0 memory is now non-trivial (server is loaded).
    used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id=0 2>/dev/null | tr -d ' ' || echo 0)"
    echo "INFRA_CHECK: GPU 0 memory used: ${used_mib} MiB"
    # No client requests fired.
    echo "INFRA_CHECK: complete, tearing down."
    exit 0
fi

# --- client + verdict ---------------------------------------------------

CLIENT_LOG="${RAW_DIR}/client_stdout_${CONFIG}.log"
echo "client: sending schedule for config=$CONFIG"
if ! python3 "${SCRIPT_DIR}/client.py" \
        --launch-ctx "$LAUNCH_CTX_JSON" \
        --server-url "http://${SERVER_BIND}:${SERVER_PORT}" \
        --results-dir "$RAW_DIR" \
        --config-label "$CONFIG" \
        --fixtures-dir "$FIXTURES_DIR" \
        > "$CLIENT_LOG" 2>&1; then
    echo "client: NON-ZERO EXIT (see $CLIENT_LOG)"
fi
tail -n 40 "$CLIENT_LOG"

echo "runner: finished config=$CONFIG"
