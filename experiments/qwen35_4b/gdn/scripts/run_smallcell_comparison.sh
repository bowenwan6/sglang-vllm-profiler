#!/usr/bin/env bash
# Phase-6 smallest-cell A1/A2/A3 comparison + Nsight captures + R3
# Nsight-overhead disclosure (execution_plan.md §3).
#
# For the smallest cell (default p=128, b=1) — deterministic per the
# Phase-5 baseline — this driver runs:
#   1. A1 unprofiled + one Nsight capture
#   2. A2 unprofiled + one Nsight capture
#   3. A3 unprofiled + one Nsight capture
#   4. A0 Nsight capture (R3 overhead disclosure — compare against
#      Phase-5 A0 unprofiled for the same cell)
#
# Every arm gets a fresh cold server bring-up (runner default).
#
# Usage:
#   bash scripts/run_smallcell_comparison.sh --gpu-id 6 \
#       [--attempt-id gdn_smallcell_<ts>]
#       [--prompt-len 128] [--batch-size 1]
#       [--n-warmup 2] [--n-timed 8] [--new-tokens 128]

set -euo pipefail

GPU_ID=""
ATTEMPT_ID=""
FIXTURES_DIR=""
FROZEN_SGLANG=""
PROMPT_LEN=128
BATCH_SIZE=1
N_WARMUP=2
N_TIMED=8
NEW_TOKENS=128

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --attempt-id) ATTEMPT_ID="$2"; shift 2 ;;
        --fixtures-dir) FIXTURES_DIR="$2"; shift 2 ;;
        --frozen-sglang) FROZEN_SGLANG="$2"; shift 2 ;;
        --prompt-len) PROMPT_LEN="$2"; shift 2 ;;
        --batch-size) BATCH_SIZE="$2"; shift 2 ;;
        --n-warmup) N_WARMUP="$2"; shift 2 ;;
        --n-timed) N_TIMED="$2"; shift 2 ;;
        --new-tokens) NEW_TOKENS="$2"; shift 2 ;;
        *) echo "unknown: $1" >&2; exit 64 ;;
    esac
done

if [ -z "$GPU_ID" ]; then
    echo "FATAL: --gpu-id required" >&2; exit 64
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GDN_ROOT="$(cd "$HERE/.." && pwd)"

: "${FIXTURES_DIR:=$GDN_ROOT/fixtures}"
: "${FROZEN_SGLANG:=/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/sglang_checkout/sglang}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
: "${ATTEMPT_ID:=gdn_smallcell_gpu${GPU_ID}_${TS}}"
CELL_ROOT="$GDN_ROOT/results/$ATTEMPT_ID"

mkdir -p "$CELL_ROOT"
LOG="$CELL_ROOT/driver.log"
SUMMARY="$CELL_ROOT/smallcell_summary.txt"
echo "phase6: cell_root = $CELL_ROOT" | tee -a "$LOG"
: > "$SUMMARY"
echo "run,arm,mode,rc,server_ready_seconds,client_wallclock_seconds,client_exit_code,gpu_returned_clean" >> "$SUMMARY"

# --- helper: wait for GPU idle -----------------------------------

wait_gpu_idle() {
    local gpu_id="$1"
    local target_uuid deadline mem foreign
    target_uuid="$(nvidia-smi --id="$gpu_id" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)"
    deadline=$((SECONDS + 180))
    while [ $SECONDS -lt $deadline ]; do
        mem="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1)"
        foreign=""
        while read -r line; do
            pid=$(echo "$line" | awk -F', ' '{print $1}')
            uuid=$(echo "$line" | awk -F', ' '{print $2}')
            if [ "$uuid" = "$target_uuid" ] && [ -e "/proc/$pid" ]; then
                foreign="$foreign $pid"
            fi
        done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null)
        if [ -z "$foreign" ] && [ "${mem:-999}" -le 500 ]; then
            return 0
        fi
        sleep 5
    done
    echo "wait_gpu_idle: TIMEOUT (mem=$mem foreign=$foreign)" >&2
    return 1
}

# --- one run: unprofiled or profiled ------------------------------

run_one() {
    local arm="$1" mode="$2"  # mode = "unprof" | "nsys"
    local run_id="${arm}_${mode}"
    local results_dir="$CELL_ROOT/$run_id"
    mkdir -p "$results_dir"
    echo "" | tee -a "$LOG"
    echo "=== phase6 arm=$arm mode=$mode ===" | tee -a "$LOG"
    if ! wait_gpu_idle "$GPU_ID"; then
        echo "$run_id,$arm,$mode,GPU_NOT_IDLE,,,,, " >> "$SUMMARY"
        return
    fi
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits > "$results_dir/gpu_pre.txt"

    set +e
    if [ "$mode" = "unprof" ]; then
        bash "$HERE/gdn_runner.sh" \
            --gpu-id "$GPU_ID" \
            --arm "$arm" \
            --attempt-id "$run_id" \
            --results-dir "$results_dir" \
            --frozen-sglang "$FROZEN_SGLANG" \
            --fixtures-dir "$FIXTURES_DIR" \
            --prompt-len "$PROMPT_LEN" \
            --batch-size "$BATCH_SIZE" \
            --new-tokens "$NEW_TOKENS" \
            --n-warmup "$N_WARMUP" \
            --n-timed "$N_TIMED" \
            > "$results_dir/runner_stdout.log" 2> "$results_dir/runner_stderr.log"
    else
        bash "$HERE/nsys_capture.sh" -- \
            --gpu-id "$GPU_ID" \
            --arm "$arm" \
            --attempt-id "$run_id" \
            --results-dir "$results_dir" \
            --frozen-sglang "$FROZEN_SGLANG" \
            --fixtures-dir "$FIXTURES_DIR" \
            --prompt-len "$PROMPT_LEN" \
            --batch-size "$BATCH_SIZE" \
            --new-tokens "$NEW_TOKENS" \
            --n-warmup "$N_WARMUP" \
            --n-timed "$N_TIMED" \
            > "$results_dir/runner_stdout.log" 2> "$results_dir/runner_stderr.log"
    fi
    rc=$?
    set -e

    local metadata="$results_dir/metadata.json"
    local srs="" cws="" cec="" clean=""
    if [ -f "$metadata" ]; then
        srs=$(python3 -c "import json; print(json.load(open('$metadata')).get('server_ready_seconds',''))" 2>/dev/null)
        cws=$(python3 -c "import json; print(json.load(open('$metadata')).get('client_wallclock_seconds',''))" 2>/dev/null)
        cec=$(python3 -c "import json; print(json.load(open('$metadata')).get('client_exit_code',''))" 2>/dev/null)
        clean=$(python3 -c "import json; print(json.load(open('$metadata')).get('gpu_returned_clean',''))" 2>/dev/null)
    fi
    echo "$run_id,$arm,$mode,$rc,$srs,$cws,$cec,$clean" >> "$SUMMARY"
    echo "phase6: $run_id done rc=$rc" | tee -a "$LOG"
}

# --- 6 runs sequentially -----------------------------------------

for arm in A1 A2 A3; do
    run_one "$arm" "unprof"
    run_one "$arm" "nsys"
done
# R3 Nsight-overhead disclosure — A0 profiled with same nsys settings.
run_one "A0" "nsys"

echo "" | tee -a "$LOG"
echo "phase6: complete — summary at $SUMMARY" | tee -a "$LOG"
cat "$SUMMARY"
