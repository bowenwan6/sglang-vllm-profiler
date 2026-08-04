#!/usr/bin/env bash
# Stage-2 T11 — reproducibility captures for A0 vs A1 at (p=128, b=1)
# to satisfy MIN_CAPTURES_FOR_REPRO=2 and firm up the H_A signal on
# steady-state (not capture-inflated) metrics.
#
# 4 sequential cold-server bring-ups on GPU 6, all under Nsight,
# rotated ports:
#   A0 rep1, A0 rep2, A1 rep1, A1 rep2.
#
# Each capture is post-processed by extract_nsys_metrics.py with
# --capture-cutoff-seconds set to the run's server_ready_seconds
# (read from metadata.json), producing 3 rows per capture:
# window=all|capture|steady_state, with kernels_per_request +
# graph_launches_per_request on the steady_state row.
#
# Usage:
#   bash scripts/run_stage2_reprocaptures.sh --gpu-id 6
#     [--attempt-id gdn_stage2_<ts>]

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
: "${ATTEMPT_ID:=gdn_stage2_gpu${GPU_ID}_${TS}}"
ROOT="$GDN_ROOT/results/$ATTEMPT_ID"

mkdir -p "$ROOT"
LOG="$ROOT/driver.log"
SUMMARY="$ROOT/stage2_summary.txt"
echo "stage2: root = $ROOT" | tee -a "$LOG"
: > "$SUMMARY"
echo "arm,rep,rc,server_ready_seconds,client_wallclock_seconds,client_exit_code,gpu_returned_clean,csv_path" >> "$SUMMARY"

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
    local arm="$1" rep="$2"
    local run_id="${arm}_${rep}"
    local results_dir="$ROOT/$run_id"
    mkdir -p "$results_dir"
    local server_port=$((30400 + RUN_COUNTER))
    RUN_COUNTER=$((RUN_COUNTER + 1))

    echo "" | tee -a "$LOG"
    echo "=== stage2 arm=$arm rep=$rep port=$server_port ===" | tee -a "$LOG"
    if ! wait_gpu_idle "$GPU_ID"; then
        echo "$arm,$rep,GPU_NOT_IDLE,,,,, " >> "$SUMMARY"
        return
    fi
    nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits > "$results_dir/gpu_pre.txt"

    set +e
    bash "$HERE/nsys_capture.sh" -- \
        --gpu-id "$GPU_ID" \
        --arm "$arm" \
        --attempt-id "$run_id" \
        --results-dir "$results_dir" \
        --frozen-sglang "$FROZEN_SGLANG" \
        --fixtures-dir "$FIXTURES_DIR" \
        --server-port "$server_port" \
        --prompt-len "$PROMPT_LEN" \
        --batch-size "$BATCH_SIZE" \
        --new-tokens "$NEW_TOKENS" \
        --n-warmup "$N_WARMUP" \
        --n-timed "$N_TIMED" \
        > "$results_dir/runner_stdout.log" 2> "$results_dir/runner_stderr.log"
    rc=$?
    set -e

    local metadata="$results_dir/metadata.json"
    local srs="" cws="" cec="" clean="" csv=""
    if [ -f "$metadata" ]; then
        srs=$(python3 -c "import json; print(json.load(open('$metadata')).get('server_ready_seconds',''))" 2>/dev/null)
        cws=$(python3 -c "import json; print(json.load(open('$metadata')).get('client_wallclock_seconds',''))" 2>/dev/null)
        cec=$(python3 -c "import json; print(json.load(open('$metadata')).get('client_exit_code',''))" 2>/dev/null)
        clean=$(python3 -c "import json; print(json.load(open('$metadata')).get('gpu_returned_clean',''))" 2>/dev/null)
    fi

    # Re-extract with the windowed cutoff (nsys_capture wrote an un-windowed
    # CSV; overwrite it with the windowed version).
    local nsys_rep="$results_dir/raw/${arm}_p${PROMPT_LEN}_b${BATCH_SIZE}.nsys-rep"
    local records="$results_dir/records_${arm}_p${PROMPT_LEN}_b${BATCH_SIZE}.jsonl"
    if [ ! -f "$records" ]; then
        records="$results_dir/raw/records_${arm}_p${PROMPT_LEN}_b${BATCH_SIZE}.jsonl"
    fi
    csv="$results_dir/nsys/${arm}_p${PROMPT_LEN}_b${BATCH_SIZE}.csv"
    if [ -f "$nsys_rep" ] && [ -n "$srs" ]; then
        # Cutoff = server_ready_seconds (from client-timestamp perspective,
        # the server accepted /health at that offset from run start).
        echo "stage2: re-extracting with --capture-cutoff-seconds $srs" | tee -a "$LOG"
        python3 "$HERE/extract_nsys_metrics.py" \
            --nsys-rep "$nsys_rep" --arm "$arm" \
            --prompt-len "$PROMPT_LEN" --batch "$BATCH_SIZE" \
            --records "$records" --output-csv "$csv" \
            --capture-cutoff-seconds "$srs" \
            --n-warmup "$N_WARMUP" --n-timed "$N_TIMED" \
            > "$results_dir/extract.log" 2>&1 || echo "stage2: WARN extract rc $?" | tee -a "$LOG"
    fi

    echo "$arm,$rep,$rc,$srs,$cws,$cec,$clean,$csv" >> "$SUMMARY"
    echo "stage2: $run_id done rc=$rc srs=${srs}s" | tee -a "$LOG"
}

for arm in A0 A1; do
    run_one "$arm" "rep1"
    run_one "$arm" "rep2"
done

echo "" | tee -a "$LOG"
echo "stage2: complete — summary at $SUMMARY" | tee -a "$LOG"
cat "$SUMMARY"
