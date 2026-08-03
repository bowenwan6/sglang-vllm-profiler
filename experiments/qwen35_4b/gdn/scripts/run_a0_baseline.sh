#!/usr/bin/env bash
# Phase-5 A0 baseline ladder driver (execution_plan.md §2).
#
# Runs A0 at 4 (prompt, batch) cells × 2 self-repeats = 8 sequential
# cells. Each cell is a cold server bring-up via gdn_runner.sh (which
# now writes gpu_post.txt + metadata.json per T1). Between cells,
# waits ~15 s for GPU memory to release and re-verifies target GPU
# idleness.
#
# All 8 cells share one $BASELINE_ROOT results dir; per-cell subdirs
# are keyed <cell_id>_rep<N>.
#
# Usage:
#   bash scripts/run_a0_baseline.sh --gpu-id 6
#     [--attempt-id gdn_a0_baseline_<ts>]
#     [--fixtures-dir experiments/qwen35_4b/gdn/fixtures]
#     [--frozen-sglang <path>]
#     [--n-warmup 2] [--n-timed 8] [--new-tokens 128]

set -euo pipefail

# --- arg parsing ------------------------------------------------

GPU_ID=""
ATTEMPT_ID=""
FIXTURES_DIR=""
FROZEN_SGLANG=""
N_WARMUP=2
N_TIMED=8
NEW_TOKENS=128

while [ $# -gt 0 ]; do
    case "$1" in
        --gpu-id) GPU_ID="$2"; shift 2 ;;
        --attempt-id) ATTEMPT_ID="$2"; shift 2 ;;
        --fixtures-dir) FIXTURES_DIR="$2"; shift 2 ;;
        --frozen-sglang) FROZEN_SGLANG="$2"; shift 2 ;;
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
REPO_ROOT="$(cd "$GDN_ROOT/../../.." && pwd)"

: "${FIXTURES_DIR:=$GDN_ROOT/fixtures}"
: "${FROZEN_SGLANG:=/tmp/claude-0/-data-sglang-vllm-profiler/1617f0f1-bb43-4914-afad-2284642acd9f/scratchpad/sglang_checkout/sglang}"

TS="$(date -u +%Y%m%dT%H%M%SZ)"
: "${ATTEMPT_ID:=gdn_a0_baseline_gpu${GPU_ID}_${TS}}"
BASELINE_ROOT="$GDN_ROOT/results/$ATTEMPT_ID"

mkdir -p "$BASELINE_ROOT"
LADDER_LOG="$BASELINE_ROOT/ladder.log"
LADDER_SUMMARY="$BASELINE_ROOT/ladder_summary.txt"
echo "phase5: baseline root = $BASELINE_ROOT" | tee -a "$LADDER_LOG"

# --- cells ------------------------------------------------------

CELLS=(
    "p128 b1  128 1"
    "p128 b4  128 4"
    "p512 b1  512 1"
    "p512 b4  512 4"
)
REPEATS=(rep1 rep2)

: > "$LADDER_SUMMARY"
echo "cell,rep,prompt_len,batch,rc,server_ready_seconds,client_wallclock_seconds,client_exit_code,gpu_returned_clean,records_path,metadata_path" >> "$LADDER_SUMMARY"

# --- helper: wait for target GPU to drop to ≤ 500 MiB ----------

wait_gpu_idle() {
    local gpu_id="$1"
    local target_uuid deadline mem foreign
    target_uuid="$(nvidia-smi --id="$gpu_id" --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -1)"
    deadline=$((SECONDS + 120))
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
    echo "wait_gpu_idle: TIMEOUT after 120s (mem=$mem foreign=$foreign)" >&2
    return 1
}

# --- run each cell × each repeat --------------------------------

for cell_spec in "${CELLS[@]}"; do
    read -r cell_id_p cell_id_b prompt_len batch <<< "$cell_spec"
    cell_id="${cell_id_p}_${cell_id_b}"
    for rep in "${REPEATS[@]}"; do
        run_id="${cell_id}_${rep}"
        results_dir="$BASELINE_ROOT/$run_id"
        mkdir -p "$results_dir"
        echo "" | tee -a "$LADDER_LOG"
        echo "=== phase5 cell=$cell_id rep=$rep prompt=$prompt_len batch=$batch ===" | tee -a "$LADDER_LOG"
        echo "phase5: waiting for GPU $GPU_ID to be idle..." | tee -a "$LADDER_LOG"
        if ! wait_gpu_idle "$GPU_ID"; then
            echo "$run_id,$rep,$prompt_len,$batch,GPU_NOT_IDLE,,,,,," >> "$LADDER_SUMMARY"
            continue
        fi
        # Capture pre-state.
        nvidia-smi --id="$GPU_ID" --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits \
            > "$results_dir/gpu_pre.txt"
        echo "phase5: launching cell $run_id" | tee -a "$LADDER_LOG"

        set +e
        bash "$HERE/gdn_runner.sh" \
            --gpu-id "$GPU_ID" \
            --arm A0 \
            --attempt-id "$run_id" \
            --results-dir "$results_dir" \
            --frozen-sglang "$FROZEN_SGLANG" \
            --fixtures-dir "$FIXTURES_DIR" \
            --prompt-len "$prompt_len" \
            --batch-size "$batch" \
            --new-tokens "$NEW_TOKENS" \
            --n-warmup "$N_WARMUP" \
            --n-timed "$N_TIMED" \
            > "$results_dir/runner_stdout.log" 2> "$results_dir/runner_stderr.log"
        rc=$?
        set -e

        # Extract per-cell fields from the runner's metadata.json.
        metadata="$results_dir/metadata.json"
        srs=""
        cws=""
        cec=""
        clean=""
        if [ -f "$metadata" ]; then
            srs=$(python3 -c "import json; print(json.load(open('$metadata')).get('server_ready_seconds',''))" 2>/dev/null)
            cws=$(python3 -c "import json; print(json.load(open('$metadata')).get('client_wallclock_seconds',''))" 2>/dev/null)
            cec=$(python3 -c "import json; print(json.load(open('$metadata')).get('client_exit_code',''))" 2>/dev/null)
            clean=$(python3 -c "import json; print(json.load(open('$metadata')).get('gpu_returned_clean',''))" 2>/dev/null)
        fi
        records="$results_dir/raw/records_A0_p${prompt_len}_b${batch}.jsonl"
        echo "$cell_id,$rep,$prompt_len,$batch,$rc,$srs,$cws,$cec,$clean,$records,$metadata" >> "$LADDER_SUMMARY"
        echo "phase5: cell $run_id done rc=$rc server_ready=${srs}s" | tee -a "$LADDER_LOG"
    done
done

echo "" | tee -a "$LADDER_LOG"
echo "phase5: ladder complete — summary at $LADDER_SUMMARY" | tee -a "$LADDER_LOG"
cat "$LADDER_SUMMARY"
