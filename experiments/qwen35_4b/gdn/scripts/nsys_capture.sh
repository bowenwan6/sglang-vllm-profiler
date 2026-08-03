#!/usr/bin/env bash
# Nsight Systems capture wrapper for one GDN cell.
#
# Wraps `gdn_runner.sh` under `nsys profile`, capturing CUDA + NVTX +
# OSRT + cuBLAS + cuDNN traces for the target (arm, prompt_len, batch)
# cell. Refuses to launch without --gpu-id from the {0..7} allowlist
# (Amendment 1).
#
# The raw .nsys-rep is written to <results-dir>/raw/ (gitignored). A
# post-run `nsys stats` extraction to CSV is left to
# scripts/extract_nsys_metrics.sh (a follow-up commit) — this wrapper
# focuses on the capture step so a failed post-process does not invalidate
# a good capture.
#
# The capture is only started AFTER server warmup so that graph capture
# and warmup GEMMs do not pollute the metrics. This is done by:
#   1. Launching the runner in --dry-run mode (validates env), then
#   2. Launching the real runner under `nsys start`/`stop` gated by an
#      NVTX range that the runner emits when the server is ready.
#
# For the initial baseline we take the simpler path: `nsys profile
# --sample=none --stats=false` wraps the whole runner invocation.
# Warmup requests are still discarded by gdn_client.py, so their trace
# volume is present in the .nsys-rep but ignorable by the extractor.

set -euo pipefail

# --- arg parsing (subset of gdn_runner + nsys knobs) ---------------

RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/gdn_runner.sh"
DRY_RUN=0
NSYS_BIN="${NSYS_BIN:-nsys}"
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx,osrt,cublas,cudnn}"
NSYS_SAMPLE="${NSYS_SAMPLE:-none}"
RUNNER_ARGS=()

show_usage() {
    cat >&2 <<'EOF'
usage: nsys_capture.sh -- <all-args-forwarded-to-gdn_runner.sh>
       nsys_capture.sh --dry-run -- <args-forwarded-to-gdn_runner.sh --dry-run>

Env knobs:
  NSYS_BIN     path to nsys binary (default: nsys on PATH)
  NSYS_TRACE   trace channels (default: cuda,nvtx,osrt,cublas,cudnn)
  NSYS_SAMPLE  --sample flag (default: none — profiling only)

The wrapper requires that gdn_runner.sh accepts every arg after `--`
unchanged. It resolves --gpu-id and --arm from those args to name the
output .nsys-rep file. It does NOT re-implement any of the runner's
preflight or GPU auth checks.
EOF
}

if [ $# -eq 0 ]; then
    show_usage
    exit 64
fi

# Split at `--`.
if [[ ! " $* " =~ " -- " ]]; then
    echo "FATAL: nsys_capture.sh requires '--' between wrapper flags and runner args" >&2
    show_usage
    exit 64
fi

# Parse pre-`--` wrapper flags.
while [ $# -gt 0 ]; do
    case "$1" in
        --dry-run) DRY_RUN=1; shift ;;
        --nsys-bin) NSYS_BIN="${2:-}"; shift 2 ;;
        --nsys-trace) NSYS_TRACE="${2:-}"; shift 2 ;;
        --nsys-sample) NSYS_SAMPLE="${2:-}"; shift 2 ;;
        --help|-h) show_usage; exit 0 ;;
        --) shift; RUNNER_ARGS=("$@"); break ;;
        *) echo "unknown wrapper argument: $1" >&2; show_usage; exit 64 ;;
    esac
done

# --- resolve cell identity from runner args -----------------------

gpu_id=""
arm=""
prompt_len=""
batch_size=""
results_dir=""
i=0
while [ $i -lt ${#RUNNER_ARGS[@]} ]; do
    case "${RUNNER_ARGS[$i]}" in
        --gpu-id) gpu_id="${RUNNER_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        --arm) arm="${RUNNER_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        --prompt-len) prompt_len="${RUNNER_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        --batch-size) batch_size="${RUNNER_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        --results-dir) results_dir="${RUNNER_ARGS[$((i + 1))]}"; i=$((i + 2)) ;;
        *) i=$((i + 1)) ;;
    esac
done

if [ -z "$gpu_id" ]; then
    echo "FATAL: --gpu-id not present in runner args" >&2
    exit 64
fi
case ",${GDN_GPU_ALLOWLIST:-0,1,2,3,4,5,6,7}," in
    *,"$gpu_id",*) : ;;
    *)
        echo "FATAL: --gpu-id=$gpu_id not in allowlist" >&2
        exit 64
        ;;
esac

if [ "$DRY_RUN" -eq 0 ] && { [ -z "$arm" ] || [ -z "$prompt_len" ] || [ -z "$batch_size" ] || [ -z "$results_dir" ]; }; then
    echo "FATAL: live run requires --arm, --prompt-len, --batch-size, --results-dir in runner args" >&2
    exit 64
fi

# --- output naming ------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
    NSYS_OUT="/tmp/gdn_nsys_dry_$(date -u +%Y%m%dT%H%M%SZ)_$$"
else
    mkdir -p "$results_dir/raw"
    NSYS_OUT="$results_dir/raw/${arm}_p${prompt_len}_b${batch_size}"
fi

# --- nsys command assembly ---------------------------------------

NSYS_CMD=(
    "$NSYS_BIN" profile
    --trace="$NSYS_TRACE"
    --sample="$NSYS_SAMPLE"
    --output="$NSYS_OUT"
    --force-overwrite=true
    --stats=false
    --duration=0
)

# --- exec --------------------------------------------------------

if [ "$DRY_RUN" -eq 1 ]; then
    echo "nsys_capture: dry-run — would exec:"
    printf '  %q' "${NSYS_CMD[@]}"; echo
    printf '  -- %q' "$RUNNER"; echo
    printf '  %q' "${RUNNER_ARGS[@]}"; echo
    echo "nsys_capture: output stem would be $NSYS_OUT"
    exit 0
fi

if ! command -v "$NSYS_BIN" >/dev/null 2>&1; then
    echo "FATAL: nsys binary not found: $NSYS_BIN" >&2
    exit 74
fi

"${NSYS_CMD[@]}" -- "$RUNNER" "${RUNNER_ARGS[@]}"
NSYS_EXIT=$?
echo "nsys_capture: nsys exit code $NSYS_EXIT"

# --- post-capture: run the extractor so gdn_verdict has CSV input --

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NSYS_REP="${NSYS_OUT}.nsys-rep"
CSV_DIR="$results_dir/nsys"
CSV_OUT="$CSV_DIR/${arm}_p${prompt_len}_b${batch_size}.csv"
RECORDS_JSONL="$results_dir/records_${arm}_p${prompt_len}_b${batch_size}.jsonl"

mkdir -p "$CSV_DIR"

if [ -f "$NSYS_REP" ]; then
    extractor_args=(
        --nsys-rep "$NSYS_REP"
        --arm "$arm"
        --prompt-len "$prompt_len"
        --batch "$batch_size"
        --output-csv "$CSV_OUT"
    )
    if [ -f "$RECORDS_JSONL" ]; then
        extractor_args+=( --records "$RECORDS_JSONL" )
    fi
    echo "nsys_capture: extracting metrics to $CSV_OUT"
    python3 "$HERE_DIR/extract_nsys_metrics.py" "${extractor_args[@]}" || \
        echo "nsys_capture: WARN extractor exit code $? (continuing)"
else
    echo "nsys_capture: WARN no .nsys-rep at $NSYS_REP; skipping extraction"
fi

exit "$NSYS_EXIT"
