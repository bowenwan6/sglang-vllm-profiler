#!/usr/bin/env bash
# Stage-1 T9 — first-token cross-arm comparison at (p=128, b=1).
#
# Runs A0, A1, A2, A3 with max_new_tokens=1 so autoregressive
# divergence cannot inflate any downstream logprob difference. Each
# record has one generated token plus its top-5 alternates (via T7).
# Downstream analysis compares first-token identity + top-1/top-2
# margins across arms to distinguish "razor-thin greedy boundary flip"
# from "real correctness defect".
#
# 4 cold-server bring-ups on the same GPU, rotated ports.
#
# Usage:
#   bash scripts/run_first_token.sh --gpu-id 6
#     [--attempt-id gdn_firsttoken_<ts>]
#     [--fixtures-dir <path>] [--frozen-sglang <path>]
#     [--prompt-len 128] [--batch-size 1]
#     [--n-warmup 2] [--n-timed 8]

set -euo pipefail

GPU_ID=""
ATTEMPT_ID=""
FIXTURES_DIR=""
FROZEN_SGLANG=""
PROMPT_LEN=128
BATCH_SIZE=1
N_WARMUP=2
N_TIMED=8

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
: "${ATTEMPT_ID:=gdn_firsttoken_gpu${GPU_ID}_${TS}}"
ROOT="$GDN_ROOT/results/$ATTEMPT_ID"

mkdir -p "$ROOT"
LOG="$ROOT/driver.log"
SUMMARY="$ROOT/firsttoken_summary.txt"
echo "firsttoken: root = $ROOT" | tee -a "$LOG"
: > "$SUMMARY"
echo "arm,rc,server_ready_seconds,client_wallclock_seconds,client_exit_code,gpu_returned_clean" >> "$SUMMARY"

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
    return 1
}

RUN_COUNTER=0

run_one() {
    local arm="$1"
    local results_dir="$ROOT/$arm"
    mkdir -p "$results_dir"
    local server_port=$((30300 + RUN_COUNTER))
    RUN_COUNTER=$((RUN_COUNTER + 1))

    echo "" | tee -a "$LOG"
    echo "=== firsttoken arm=$arm port=$server_port ===" | tee -a "$LOG"
    if ! wait_gpu_idle "$GPU_ID"; then
        echo "$arm,GPU_NOT_IDLE,,,, " >> "$SUMMARY"
        return
    fi
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits > "$results_dir/gpu_pre.txt"

    set +e
    bash "$HERE/gdn_runner.sh" \
        --gpu-id "$GPU_ID" \
        --arm "$arm" \
        --attempt-id "${arm}_firsttoken" \
        --results-dir "$results_dir" \
        --frozen-sglang "$FROZEN_SGLANG" \
        --fixtures-dir "$FIXTURES_DIR" \
        --server-port "$server_port" \
        --prompt-len "$PROMPT_LEN" \
        --batch-size "$BATCH_SIZE" \
        --new-tokens 1 \
        --n-warmup "$N_WARMUP" \
        --n-timed "$N_TIMED" \
        > "$results_dir/runner_stdout.log" 2> "$results_dir/runner_stderr.log"
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
    echo "$arm,$rc,$srs,$cws,$cec,$clean" >> "$SUMMARY"
    echo "firsttoken: $arm done rc=$rc srs=${srs}s" | tee -a "$LOG"
}

for arm in A0 A1 A2 A3; do
    run_one "$arm"
done

echo "" | tee -a "$LOG"
echo "firsttoken: complete — summary at $SUMMARY" | tee -a "$LOG"
cat "$SUMMARY"
