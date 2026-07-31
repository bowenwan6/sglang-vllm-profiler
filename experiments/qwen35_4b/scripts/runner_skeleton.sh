#!/usr/bin/env bash
# Runner skeleton for the Qwen3.5-4B BCG DeepStack validation.
#
# CPU-only in --dry-run. Refuses to run without an explicitly authorised
# GPU ID. Records launch context and PGID for ownership-verified cleanup.
# Never invokes pkill / killall / fuser -k / GPU reset.
#
# Usage
# -----
#     bash runner_skeleton.sh --dry-run
#     bash runner_skeleton.sh --config bcg_default --gpu-id 3 --attempt-id attempt_gpu3_20260801T000000Z
#
# The --gpu-id must equal the QWEN35_GPU_ID env var (or be the only
# source); this two-channel check is a deliberate belt-and-braces guard.

set -euo pipefail

# --- argument parsing --------------------------------------------------

DRY_RUN=0
CONFIG=""
GPU_ID=""
ATTEMPT_ID=""
FIXTURES_DIR=""
RESULTS_DIR=""

show_usage() {
    cat >&2 <<'EOF'
usage: runner_skeleton.sh [--dry-run]
                          [--config <label>]
                          [--gpu-id <N>]
                          [--attempt-id <id>]
                          [--fixtures-dir <path>]
                          [--results-dir <path>]

--dry-run runs argument parsing, provenance preflight, PGID scaffolding,
          and cleanup wiring WITHOUT any CUDA import or server launch.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --config)  CONFIG="${2:-}"; shift 2 ;;
        --gpu-id)  GPU_ID="${2:-}"; shift 2 ;;
        --attempt-id) ATTEMPT_ID="${2:-}"; shift 2 ;;
        --fixtures-dir) FIXTURES_DIR="${2:-}"; shift 2 ;;
        --results-dir)  RESULTS_DIR="${2:-}"; shift 2 ;;
        --help|-h) show_usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; show_usage; exit 64 ;;
    esac
done

# --- paths -------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVESTIGATION_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES_DIR="${FIXTURES_DIR:-${INVESTIGATION_DIR}/fixtures}"
RESULTS_DIR="${RESULTS_DIR:-${INVESTIGATION_DIR}/results}"

# --- GPU authorisation gate --------------------------------------------

env_gpu_id="${QWEN35_GPU_ID:-}"
if [ "$DRY_RUN" = "0" ]; then
    if [ -z "$GPU_ID" ] && [ -z "$env_gpu_id" ]; then
        echo "FATAL: no --gpu-id and no QWEN35_GPU_ID env var." >&2
        echo "This runner refuses to touch a GPU without an explicit ID." >&2
        exit 64
    fi
    if [ -n "$GPU_ID" ] && [ -n "$env_gpu_id" ] && [ "$GPU_ID" != "$env_gpu_id" ]; then
        echo "FATAL: --gpu-id ($GPU_ID) disagrees with QWEN35_GPU_ID ($env_gpu_id)." >&2
        exit 64
    fi
    if [ -z "$GPU_ID" ]; then
        GPU_ID="$env_gpu_id"
    fi
    if [ -z "$CONFIG" ]; then
        echo "FATAL: --config is required for a live run." >&2
        exit 64
    fi
    if [ -z "$ATTEMPT_ID" ]; then
        echo "FATAL: --attempt-id is required for a live run." >&2
        exit 64
    fi
fi

# --- launch context ----------------------------------------------------

PRELAUNCH_UTC="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
RUNNER_PGID="$$"
LAUNCH_CTX_JSON="/tmp/qwen35_launch_ctx_${PRELAUNCH_UTC//:/-}.json"

cat > "$LAUNCH_CTX_JSON" <<EOF
{
  "runner": "runner_skeleton.sh",
  "prelaunch_utc": "${PRELAUNCH_UTC}",
  "config": "${CONFIG}",
  "gpu_id": "${GPU_ID}",
  "attempt_id": "${ATTEMPT_ID}",
  "runner_pgid": "${RUNNER_PGID}",
  "dry_run": ${DRY_RUN},
  "cwd": "$(pwd)",
  "argv": "$*",
  "fixtures_dir": "${FIXTURES_DIR}",
  "results_dir": "${RESULTS_DIR}"
}
EOF

echo "launch_context: ${LAUNCH_CTX_JSON}"

# --- foreign-PID guard -------------------------------------------------

foreign_pid_check() {
    # NEVER signals; report-only. Aborts if any foreign compute PID is
    # present on the authorised GPU.
    if [ "$DRY_RUN" = "1" ]; then
        echo "foreign_pid_check: SKIPPED_DRY_RUN"
        return 0
    fi
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "foreign_pid_check: NVIDIA-SMI MISSING; aborting rather than trusting a bare GPU." >&2
        exit 71
    fi
    local raw
    raw="$(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader || true)"
    if [ -n "$raw" ]; then
        echo "foreign_pid_check: existing compute apps present:"
        echo "$raw"
        echo "foreign_pid_check: aborting; the runner refuses to co-inhabit." >&2
        exit 71
    fi
    echo "foreign_pid_check: OK"
}

# --- ownership-verified cleanup ---------------------------------------

OWNED_PIDS=()
register_owned() { OWNED_PIDS+=("$1"); }

cleanup() {
    echo "cleanup: signalling owned PIDs only (${#OWNED_PIDS[@]} recorded)."
    local pid
    for pid in "${OWNED_PIDS[@]:-}"; do
        [ -z "$pid" ] && continue
        # Verify PID exists AND belongs to our PGID before signalling.
        if kill -0 "$pid" 2>/dev/null; then
            local pgid
            pgid="$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')"
            if [ "$pgid" = "$RUNNER_PGID" ]; then
                echo "cleanup: SIGTERM to $pid (pgid $pgid)"
                kill -TERM "$pid" 2>/dev/null || true
            else
                echo "cleanup: REFUSING to signal $pid (pgid $pgid != runner $RUNNER_PGID)"
            fi
        fi
    done
}
trap cleanup EXIT INT TERM

# --- provenance preflight ---------------------------------------------

echo "preflight: provenance check"
if [ "$DRY_RUN" = "1" ]; then
    python3 "${SCRIPT_DIR}/preflight_provenance.py" --dry-run --strict || {
        rc=$?
        echo "preflight: FAILED (rc=$rc)"
        exit "$rc"
    }
else
    python3 "${SCRIPT_DIR}/preflight_provenance.py" --strict || {
        rc=$?
        echo "preflight: FAILED (rc=$rc)"
        exit "$rc"
    }
fi

echo "preflight: OK"
foreign_pid_check

# --- server + bench (STUB) --------------------------------------------
#
# The following block is a stub. It is intentionally EMPTY at this
# stage; a live server launch is only permitted once (a) --gpu-id is
# authorised, (b) preflight passes, (c) foreign_pid_check passes.
#
# When live launch is authorised, this stub is replaced with:
#   - python3 -m sglang.launch_server ... &   # capture PID
#   - register_owned <server_pid>
#   - wait for /health
#   - python3 client_skeleton.py --launch-ctx "$LAUNCH_CTX_JSON" ...
#   - python3 verdict.py --launch-ctx "$LAUNCH_CTX_JSON" ...

if [ "$DRY_RUN" = "1" ]; then
    echo "server_launch: SKIPPED_DRY_RUN"
    echo "bench:         SKIPPED_DRY_RUN"
    echo "verdict:       SKIPPED_DRY_RUN"
    echo "runner_skeleton: dry-run OK"
    exit 0
fi

echo "FATAL: server launch is not yet enabled in this skeleton." >&2
echo "       Live GPU work requires a follow-up change that lands the" >&2
echo "       actual server-launch + bench-client wiring." >&2
exit 66
