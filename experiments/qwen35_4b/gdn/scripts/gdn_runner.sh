#!/usr/bin/env bash
# 4-arm GDN prefill-BCG runner for Qwen3.5-4B.
#
# Given --arm { A0 | A1 | A2 | A3 } and an authorised --gpu-id, this
# script:
#   1. Verifies GPU authorisation (allowlist {0, 1, 7}) and preflight.
#   2. Runs the GDN provenance preflight (fields, hybrid check, env).
#   3. Verifies the target GPU is unoccupied (foreign PID guard, by UUID).
#   4. Applies the GDN baseline instrumentation (no-op module) via
#      server_launcher.py.
#   5. Launches sglang.launch_server on the target GPU from the frozen
#      SGLang checkout via PYTHONPATH.
#   6. Waits for /health.
#   7. Runs gdn_client.py against the server for a single
#      (prompt_len, batch) cell.
#   8. Records raw logs + per-request records under --results-dir/raw/.
#   9. Cleans up its own PGID (never signalling foreign PIDs).
#
# Never uses pkill, killall, fuser -k, or nvidia-smi --gpu-reset.
#
# The per-arm flag deltas that this script threads to SGLang. Uses the
# canonical --cuda-graph-backend-{prefill,decode}=<mode> pair
# (server_args.py:1784-1792 at frozen SHA 58974ca1); backend enum:
# {full, breakable, tc_piecewise, disabled} — see
# model_executor/cuda_graph_config.py:38-45.
#
#   A0 eager_eager: --cuda-graph-backend-prefill=disabled
#                   --cuda-graph-backend-decode=disabled
#   A1 bcg_eager:   --cuda-graph-backend-prefill=breakable
#                   --cuda-graph-backend-decode=disabled
#   A2 eager_dcg:   --cuda-graph-backend-prefill=disabled
#                   --cuda-graph-backend-decode=full
#   A3 bcg_dcg:     --cuda-graph-backend-prefill=breakable
#                   --cuda-graph-backend-decode=full

set -euo pipefail

# --- argument parsing --------------------------------------------------

DRY_RUN=0
ARM=""
GPU_ID=""
ATTEMPT_ID=""
RESULTS_DIR=""
FROZEN_SGLANG=""
FIXTURES_DIR=""
MODEL_ID="Qwen/Qwen3.5-4B"
MODEL_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
FROZEN_SGLANG_SHA="58974ca16ca2a4bb2f02f9ceb9622a0fd2ccf7f8"
SERVER_PORT="30001"
SERVER_BIND="127.0.0.1"
PROMPT_LEN=""
BATCH_SIZE=""
NEW_TOKENS="128"
N_WARMUP="2"
N_TIMED="8"
LIBCUDA_PRELOAD="${LIBCUDA_PRELOAD:-/usr/lib/x86_64-linux-gnu/libcuda.so.595.71.05}"
LAUNCH_WAIT_SECONDS="${LAUNCH_WAIT_SECONDS:-600}"
GPU_ALLOWLIST="${GDN_GPU_ALLOWLIST:-0,1,2,3,4,5,6,7}"

show_usage() {
    cat >&2 <<'EOF'
usage: gdn_runner.sh --gpu-id <id> --arm <A0|A1|A2|A3> \
                     --attempt-id <id> --results-dir <path> \
                     --frozen-sglang <checkout> --fixtures-dir <path> \
                     --prompt-len <N> --batch-size <N> \
                     [--new-tokens 128] [--n-warmup 2] [--n-timed 8] \
                     [--server-port 30001] [--dry-run]

--arm        A0 eager_eager  = eager prefill, eager decode
             A1 bcg_eager    = BCG prefill, eager decode
             A2 eager_dcg    = eager prefill, full-decode CG
             A3 bcg_dcg      = BCG prefill, full-decode CG
--dry-run    exercise argument parsing / preflight / cleanup wiring; no CUDA

GPU allowlist: {0..7} — any GPU 0-7 provided it is unoccupied and idle
              (compute processes = 0, memory <= 500 MiB, util <= 5%).
              Extended from {0,1,7} on 2026-08-03 per operator authorisation
              (see gdn/validation_plan.md Amendment 1). Override with
              GDN_GPU_ALLOWLIST env var.
Preservation invariants: /data/sglang-fork read-only; frozen SGLang
source unchanged.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --arm)          ARM="${2:-}"; shift 2 ;;
        --gpu-id)       GPU_ID="${2:-}"; shift 2 ;;
        --attempt-id)   ATTEMPT_ID="${2:-}"; shift 2 ;;
        --results-dir)  RESULTS_DIR="${2:-}"; shift 2 ;;
        --frozen-sglang) FROZEN_SGLANG="${2:-}"; shift 2 ;;
        --fixtures-dir) FIXTURES_DIR="${2:-}"; shift 2 ;;
        --model-id)     MODEL_ID="${2:-}"; shift 2 ;;
        --model-revision) MODEL_REVISION="${2:-}"; shift 2 ;;
        --frozen-sglang-sha) FROZEN_SGLANG_SHA="${2:-}"; shift 2 ;;
        --server-port)  SERVER_PORT="${2:-}"; shift 2 ;;
        --prompt-len)   PROMPT_LEN="${2:-}"; shift 2 ;;
        --batch-size)   BATCH_SIZE="${2:-}"; shift 2 ;;
        --new-tokens)   NEW_TOKENS="${2:-}"; shift 2 ;;
        --n-warmup)     N_WARMUP="${2:-}"; shift 2 ;;
        --n-timed)      N_TIMED="${2:-}"; shift 2 ;;
        --help|-h)      show_usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; show_usage; exit 64 ;;
    esac
done

# --- paths -------------------------------------------------------------

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GDN_ROOT="$(cd "$HERE/.." && pwd)"
REPO_ROOT="$(cd "$GDN_ROOT/../../.." && pwd)"
DS_SCRIPTS="$REPO_ROOT/experiments/qwen35_4b/scripts"

# --- required arg gate -------------------------------------------------

env_gpu_id="${QWEN35_GPU_ID:-}"
if [ -z "$GPU_ID" ]; then
    if [ -n "$env_gpu_id" ]; then
        GPU_ID="$env_gpu_id"
    else
        echo "FATAL: no --gpu-id and no QWEN35_GPU_ID env var." >&2
        exit 64
    fi
elif [ -n "$env_gpu_id" ] && [ "$env_gpu_id" != "$GPU_ID" ]; then
    echo "FATAL: --gpu-id ($GPU_ID) disagrees with QWEN35_GPU_ID ($env_gpu_id)." >&2
    exit 64
fi

case ",$GPU_ALLOWLIST," in
    *,"$GPU_ID",*) : ;;
    *)
        echo "FATAL: --gpu-id=$GPU_ID not in allowlist {$GPU_ALLOWLIST}." >&2
        exit 64
        ;;
esac

if [ "$DRY_RUN" -eq 0 ]; then
    if [ -z "$ARM" ]; then
        echo "FATAL: --arm is required for a live run." >&2
        exit 64
    fi
    if [ -z "$ATTEMPT_ID" ] || [ -z "$RESULTS_DIR" ] || \
       [ -z "$FROZEN_SGLANG" ] || [ -z "$FIXTURES_DIR" ]; then
        echo "FATAL: --attempt-id, --results-dir, --frozen-sglang and --fixtures-dir required for a live run." >&2
        exit 64
    fi
    if [ -z "$PROMPT_LEN" ] || [ -z "$BATCH_SIZE" ]; then
        echo "FATAL: --prompt-len and --batch-size required for a live run." >&2
        exit 64
    fi
fi

# --- arm-to-flag mapping ----------------------------------------------

# Canonical prefill/decode backend selectors (frozen SGLang 58974ca1).
ARM_FLAGS=()
case "$ARM" in
    A0|eager_eager)
        ARM="A0"
        ARM_FLAGS+=(--cuda-graph-backend-prefill=disabled)
        ARM_FLAGS+=(--cuda-graph-backend-decode=disabled)
        ;;
    A1|bcg_eager)
        ARM="A1"
        ARM_FLAGS+=(--cuda-graph-backend-prefill=breakable)
        ARM_FLAGS+=(--cuda-graph-backend-decode=disabled)
        ;;
    A2|eager_dcg)
        ARM="A2"
        ARM_FLAGS+=(--cuda-graph-backend-prefill=disabled)
        ARM_FLAGS+=(--cuda-graph-backend-decode=full)
        ;;
    A3|bcg_dcg)
        ARM="A3"
        ARM_FLAGS+=(--cuda-graph-backend-prefill=breakable)
        ARM_FLAGS+=(--cuda-graph-backend-decode=full)
        ;;
    "")
        # dry-run only path validated below
        :
        ;;
    *) echo "FATAL: unknown --arm=$ARM" >&2; exit 64 ;;
esac

# --- context blob (dry-run + live) ------------------------------------

CONTEXT_BLOB="/tmp/qwen35_gdn_launch_ctx_$(date -u +%Y%m%dT%H%M%SZ)_$$.json"
python3 - <<PY > "$CONTEXT_BLOB"
import json, os, sys
payload = {
    "runner": "gdn_runner.sh",
    "arm": "$ARM",
    "arm_flags": [$(printf '"%s",' "${ARM_FLAGS[@]}")],
    "gpu_id": "$GPU_ID",
    "gpu_allowlist": "$GPU_ALLOWLIST",
    "model_id": "$MODEL_ID",
    "model_revision": "$MODEL_REVISION",
    "frozen_sglang": "$FROZEN_SGLANG",
    "frozen_sglang_sha": "$FROZEN_SGLANG_SHA",
    "server_port": "$SERVER_PORT",
    "server_bind": "$SERVER_BIND",
    "prompt_len": "$PROMPT_LEN",
    "batch_size": "$BATCH_SIZE",
    "new_tokens": "$NEW_TOKENS",
    "n_warmup": "$N_WARMUP",
    "n_timed": "$N_TIMED",
    "libcuda_preload": "$LIBCUDA_PRELOAD",
    "dry_run": bool($DRY_RUN),
}
json.dump(payload, sys.stdout, indent=2, sort_keys=True)
sys.stdout.write("\n")
PY

echo "gdn_runner: context blob written to $CONTEXT_BLOB"

# --- dry-run exit --------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
    echo "gdn_runner: --dry-run — exiting after context blob."
    exit 0
fi

# --- GDN preflight (blocking) --------------------------------------

if ! python3 "$HERE/gdn_preflight.py" --strict --json > "$RESULTS_DIR/preflight.json"; then
    echo "FATAL: gdn_preflight rejected the environment." >&2
    exit 74
fi

# --- foreign-PID guard (target GPU only, by UUID) ------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FATAL: nvidia-smi not available." >&2
    exit 74
fi

target_uuid="$(nvidia-smi --id="$GPU_ID" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)"
if [ -z "$target_uuid" ]; then
    echo "FATAL: could not read UUID for GPU $GPU_ID." >&2
    exit 74
fi

foreign_pids=""
while read -r line; do
    pid=$(echo "$line" | awk -F', ' '{print $1}')
    uuid=$(echo "$line" | awk -F', ' '{print $2}')
    if [ "$uuid" = "$target_uuid" ] && [ -e "/proc/$pid" ]; then
        foreign_pids="$foreign_pids $pid"
    fi
done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)

if [ -n "$foreign_pids" ]; then
    echo "FATAL: foreign PIDs on GPU $GPU_ID ($target_uuid):$foreign_pids" >&2
    exit 74
fi

# --- server startup -----------------------------------------------

RAW_DIR="$RESULTS_DIR/raw"
mkdir -p "$RAW_DIR"
SERVER_LOG="$RAW_DIR/server_${ARM}_p${PROMPT_LEN}_b${BATCH_SIZE}.log"

LAUNCHER_ARGS=(
    --instrumentation "$HERE/gdn_instrumentation.py"
    --frozen-sglang "$FROZEN_SGLANG"
    --frozen-sglang-sha "$FROZEN_SGLANG_SHA"
)

SGLANG_FORWARDED=(
    --model-path "$MODEL_ID"
    --revision "$MODEL_REVISION"
    --host "$SERVER_BIND"
    --port "$SERVER_PORT"
    --tp 1
    --mem-fraction-static 0.7
    "${ARM_FLAGS[@]}"
)

# Set up subprocess env; the sitecustomize bootstrap propagates the
# same environment into SGLang scheduler / worker spawn children.
export PYTHONPATH="$FROZEN_SGLANG/python:${PYTHONPATH:-}"
export PYTHONPATH="$DS_SCRIPTS/bootstrap:$PYTHONPATH"
export LD_PRELOAD="${LD_PRELOAD:-$LIBCUDA_PRELOAD}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export QWEN35_GDN_RUN_MODE="$ARM"

set -m
(
    setsid python3 "$DS_SCRIPTS/server_launcher.py" \
        "${LAUNCHER_ARGS[@]}" -- "${SGLANG_FORWARDED[@]}" \
        > "$SERVER_LOG" 2>&1
) &
SERVER_PGID=$!
set +m
echo "gdn_runner: server PGID $SERVER_PGID; log $SERVER_LOG"

# --- teardown trap (own PGID only) --------------------------------

trap 'echo "gdn_runner: teardown for PGID $SERVER_PGID"; kill -TERM -"$SERVER_PGID" 2>/dev/null || true; sleep 3; kill -KILL -"$SERVER_PGID" 2>/dev/null || true' EXIT

# --- readiness wait -----------------------------------------------

start_ts=$SECONDS
ready=0
while [ $((SECONDS - start_ts)) -lt "$LAUNCH_WAIT_SECONDS" ]; do
    if curl -sf "http://$SERVER_BIND:$SERVER_PORT/health" >/dev/null 2>&1; then
        ready=1
        break
    fi
    if ! kill -0 -"$SERVER_PGID" 2>/dev/null; then
        echo "FATAL: server PGID $SERVER_PGID died during startup." >&2
        exit 75
    fi
    sleep 2
done

if [ "$ready" -ne 1 ]; then
    echo "FATAL: server did not become ready within ${LAUNCH_WAIT_SECONDS}s." >&2
    exit 75
fi

echo "gdn_runner: server ready after $((SECONDS - start_ts))s"

# --- client dispatch ---------------------------------------------

CLIENT_LOG="$RAW_DIR/client_${ARM}_p${PROMPT_LEN}_b${BATCH_SIZE}.log"

python3 "$HERE/gdn_client.py" \
    --server "http://$SERVER_BIND:$SERVER_PORT" \
    --arm "$ARM" \
    --prompt-len "$PROMPT_LEN" \
    --batch-size "$BATCH_SIZE" \
    --new-tokens "$NEW_TOKENS" \
    --n-warmup "$N_WARMUP" \
    --n-timed "$N_TIMED" \
    --fixtures-dir "$FIXTURES_DIR" \
    --output "$RAW_DIR/records_${ARM}_p${PROMPT_LEN}_b${BATCH_SIZE}.jsonl" \
    2>&1 | tee "$CLIENT_LOG"

echo "gdn_runner: cell complete (arm=$ARM prompt=$PROMPT_LEN batch=$BATCH_SIZE)"
