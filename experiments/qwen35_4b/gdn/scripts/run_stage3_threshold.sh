#!/usr/bin/env bash
# Stage-3 T12 — threshold ladder for the <1024 alt-stream hypothesis.
#
# Per source_audit.md §3.3: GDN's _forward_input_proj forks in_proj_ba
# onto an alt CUDA stream when `seq_len < DUAL_STREAM_TOKEN_THRESHOLD`
# (= 1024), where seq_len is the padded BCG bucket size. The
# hypothesis is that this alt-stream branch is the mechanism behind
# Stage-2's steady-state +13.6% kernel inflation in A1 vs A0.
#
# Test: run A0 and A1 at 4 prompt-length targets that pad to buckets
# on both sides of the 1024 threshold:
#
#   L1 --prompt-len 128   → actual ~72-98,   bucket ≤112 (branch ON)
#   L2 --prompt-len 1024  → actual ~600-800, bucket ≤832 (branch ON)
#   L3 --prompt-len 2048  → actual ~1200,    bucket 1280 (branch OFF)
#   L4 --prompt-len 4096  → actual ~2400,    bucket 2560 (branch OFF)
#
# Each (arm, L) cell is captured twice with Nsight (MIN_CAPTURES_FOR_REPRO=2).
# Total: 2 arms × 4 sizes × 2 reps = 16 runs. Each run is a cold
# server bring-up on GPU 6 with a rotated port.
#
# Expected signature supporting the hypothesis:
#   * L1 and L2: A1 kernel-inflation delta vs A0 present and reproducible.
#   * L3 and L4: A1 kernel-inflation delta materially shrinks or
#     disappears (alt-stream branch disabled).
# If the delta persists at L3/L4, the hypothesis is revised or rejected.
#
# Usage:
#   bash scripts/run_stage3_threshold.sh --gpu-id 6
#     [--attempt-id gdn_stage3_<ts>]

set -euo pipefail

GPU_ID=""
ATTEMPT_ID=""
FIXTURES_DIR=""
FROZEN_SGLANG=""
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
: "${ATTEMPT_ID:=gdn_stage3_gpu${GPU_ID}_${TS}}"
ROOT="$GDN_ROOT/results/$ATTEMPT_ID"

mkdir -p "$ROOT"
LOG="$ROOT/driver.log"
SUMMARY="$ROOT/stage3_summary.txt"
echo "stage3: root = $ROOT" | tee -a "$LOG"
: > "$SUMMARY"
echo "arm,rep,prompt_len,rc,server_ready_seconds,client_wallclock_seconds,client_exit_code,gpu_returned_clean,csv_path" >> "$SUMMARY"

wait_gpu_idle() {
    local gpu_id="$1"
    local target_uuid deadline mem foreign
    target_uuid="$(nvidia-smi --id="$gpu_id" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)"
    deadline=$((SECONDS + 240))
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
    local arm="$1" prompt_len="$2" rep="$3"
    local run_id="${arm}_p${prompt_len}_${rep}"
    local results_dir="$ROOT/$run_id"
    mkdir -p "$results_dir"
    local server_port=$((30500 + RUN_COUNTER))
    RUN_COUNTER=$((RUN_COUNTER + 1))

    echo "" | tee -a "$LOG"
    echo "=== stage3 arm=$arm p=$prompt_len rep=$rep port=$server_port ===" | tee -a "$LOG"
    if ! wait_gpu_idle "$GPU_ID"; then
        echo "$arm,$rep,$prompt_len,GPU_NOT_IDLE,,,,, " >> "$SUMMARY"
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
        --prompt-len "$prompt_len" \
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

    # Re-extract with windowed split
    local nsys_rep="$results_dir/raw/${arm}_p${prompt_len}_b${BATCH_SIZE}.nsys-rep"
    local records="$results_dir/records_${arm}_p${prompt_len}_b${BATCH_SIZE}.jsonl"
    if [ ! -f "$records" ]; then
        records="$results_dir/raw/records_${arm}_p${prompt_len}_b${BATCH_SIZE}.jsonl"
    fi
    csv="$results_dir/nsys/${arm}_p${prompt_len}_b${BATCH_SIZE}.csv"
    if [ -f "$nsys_rep" ] && [ -n "$srs" ]; then
        python3 "$HERE/extract_nsys_metrics.py" \
            --nsys-rep "$nsys_rep" --arm "$arm" \
            --prompt-len "$prompt_len" --batch "$BATCH_SIZE" \
            --records "$records" --output-csv "$csv" \
            --capture-cutoff-seconds "$srs" \
            --n-warmup "$N_WARMUP" --n-timed "$N_TIMED" \
            > "$results_dir/extract.log" 2>&1 || echo "stage3: WARN extract rc $?" | tee -a "$LOG"
    fi

    echo "$arm,$rep,$prompt_len,$rc,$srs,$cws,$cec,$clean,$csv" >> "$SUMMARY"
    echo "stage3: $run_id done rc=$rc srs=${srs}s" | tee -a "$LOG"
}

# Ladder: 4 sizes × 2 arms × 2 reps = 16 runs.
for prompt_len in 128 1024 2048 4096; do
    for arm in A0 A1; do
        for rep in rep1 rep2; do
            run_one "$arm" "$prompt_len" "$rep"
        done
    done
done

echo "" | tee -a "$LOG"
echo "stage3: complete — summary at $SUMMARY" | tee -a "$LOG"
cat "$SUMMARY"
