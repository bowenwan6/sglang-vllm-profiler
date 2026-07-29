#!/usr/bin/env bash
# R5.C — correctness audit for clean (Y).
#
# Hypothesis: clean Y captures cuda graphs that read from
# input_deepstack_embeds at the address it had during the synthesis-
# at-warmup call. At inference, embed_mm_inputs allocates a fresh
# tensor. If the cuda-graph replay reads from the stale captured
# address, outputs are silently corrupted but still structurally
# valid 128-token responses.
#
# Test: run TWO short benches with the SAME --seed image+text input
# and greedy sampling, one with PCG-on-with-clean-Y, one with the
# default PCG-off (VLM auto-disables, prefill backend DISABLED ->
# eager language model forward). Compare generated_texts.
#
#   MATCH  -> clean Y is correctness-OK; observed ~104 ms TTFT is
#             the honest cost of PCG-on for this VLM workload
#   DIFFER -> clean Y has a real correctness bug; before any perf
#             discussion we need a static input_deepstack_embeds
#             buffer (or similar address-stability fix)
set -u

ROOT=/data/sglang-vllm-profiler/experiments/qwen3vl8b/v2/image_text_benchmarks/debug_pcg_capture_stream/root_cause
RESULTS="$ROOT/results/R5_clean_Y/R5C_correctness_audit"
RAW="$RESULTS/raw"
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
PORT=30003
FORK_PY=/data/sglang-fork/python

mkdir -p "$RAW"
[ -f "$RAW/.gitignore" ] || printf '*.log\n*.jsonl\n*.txt\n' > "$RAW/.gitignore"

# Fixed bench config across both runs.
N=2; SEED=42; INPUT_LEN=128; OUTPUT_LEN=64; CONCURRENCY=1

run_bench () {
  local LABEL="$1"
  local SERVER_EXTRA="$2"
  local SERVER_LOG="$RAW/${LABEL}_server.log"
  local BENCH_LOG="$RAW/${LABEL}_bench.log"
  local BENCH_OUT="$RAW/${LABEL}_bench.jsonl"

  rm -f "$SERVER_LOG" "$BENCH_LOG" "$BENCH_OUT"

  echo "=== ${LABEL}: launching server (${SERVER_EXTRA:-default flags}) ==="
  # `${SERVER_EXTRA}` may be empty; do not quote.
  python3 -m sglang.launch_server \
    --model-path "$SNAP" --dtype bfloat16 --port "$PORT" --tp 1 \
    --attention-backend flashinfer $SERVER_EXTRA \
    > "$SERVER_LOG" 2>&1 &
  local SRV_PID=$!

  local READY=0
  for i in $(seq 1 600); do
    if curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:$PORT/get_model_info 2>/dev/null | grep -q 200; then
      READY=1; echo "  ${LABEL}: ready after $((i*2)) s"; break
    fi
    if ! kill -0 $SRV_PID 2>/dev/null; then echo "  ${LABEL}: DIED"; break; fi
    sleep 2
  done
  if [ "$READY" -ne 1 ]; then
    echo "  ${LABEL}: server never came up; aborting this leg"
    pkill -9 -f "sglang.launch_server" 2>/dev/null; sleep 5
    return 1
  fi

  echo "  ${LABEL}: running bench (n=${N}, max_tokens=${OUTPUT_LEN}, seed=${SEED})"
  python3 -m sglang.benchmark.serving \
    --backend sglang-oai-chat --base-url http://127.0.0.1:$PORT --model "$SNAP" \
    --dataset-name image --image-count 1 --image-resolution 720p \
    --image-format png --image-content random \
    --random-input-len ${INPUT_LEN} --random-output-len ${OUTPUT_LEN} --random-range-ratio 1.0 \
    --max-concurrency ${CONCURRENCY} --num-prompts ${N} --warmup-requests 0 --seed ${SEED} \
    --extra-request-body '{"temperature": 0, "top_p": 1}' \
    --output-details --output-file "$BENCH_OUT" \
    > "$BENCH_LOG" 2>&1 || true
  echo "  ${LABEL}: bench done"

  pkill -TERM -P $SRV_PID 2>/dev/null
  kill -TERM $SRV_PID 2>/dev/null
  sleep 5
  pkill -9 -f "sglang.launch_server" 2>/dev/null
  sleep 5
  return 0
}

export CUDA_VISIBLE_DEVICES=0
export SGLANG_USE_CUDA_IPC_TRANSPORT=1
unset SGLANG_KERNEL_API_LOGLEVEL SGLANG_KERNEL_API_LOGDEST SGLANG_DEBUG_PCG_CALL_TRACE TORCH_LOGS
export PYTHONPATH="$FORK_PY${PYTHONPATH:+:$PYTHONPATH}"

# Leg 1: clean Y + PCG ON
run_bench "PCG_ON_cleanY" "--enforce-piecewise-cuda-graph"

# Leg 2: PCG OFF (VLM auto-disable -> Backend.DISABLED -> eager)
run_bench "PCG_OFF_eager"  ""

# Compare generated_texts
python3 - "$RAW/PCG_ON_cleanY_bench.jsonl" "$RAW/PCG_OFF_eager_bench.jsonl" "$RESULTS/audit_report.md" <<'PYEOF'
import json, sys, pathlib, difflib

on_path, off_path, report_path = sys.argv[1:4]
def load(p):
    line = pathlib.Path(p).read_text().strip().splitlines()
    if not line:
        return None, []
    d = json.loads(line[0])
    return d, d.get("generated_texts", [])

on_meta, on_texts   = load(on_path)
off_meta, off_texts = load(off_path)

lines = ["# R5.C correctness audit — clean Y vs PCG-OFF baseline", ""]
lines.append(f"- PCG ON (clean Y) bench jsonl: `{on_path}`")
lines.append(f"- PCG OFF (eager)   bench jsonl: `{off_path}`")
lines.append(f"- Each leg: {len(on_texts)} prompts, max_tokens=64, temperature=0, top_p=1, seed=42")
lines.append("")

if not on_texts or not off_texts:
    lines.append("**ABORT** — one or both legs produced no outputs (server may have failed to start).")
    pathlib.Path(report_path).write_text("\n".join(lines))
    print("AUDIT_INCOMPLETE")
    sys.exit(0)

if len(on_texts) != len(off_texts):
    lines.append(f"**ABORT** — leg sizes differ ({len(on_texts)} vs {len(off_texts)}).")
    pathlib.Path(report_path).write_text("\n".join(lines))
    print("AUDIT_INCOMPLETE")
    sys.exit(0)

all_match = True
for i, (a, b) in enumerate(zip(on_texts, off_texts)):
    if a != b:
        all_match = False
    lines.append(f"## Prompt {i}")
    lines.append("")
    lines.append("### PCG ON (clean Y) output")
    lines.append("```")
    lines.append(a)
    lines.append("```")
    lines.append("### PCG OFF (eager) output")
    lines.append("```")
    lines.append(b)
    lines.append("```")
    lines.append(f"**match: {a == b}**  (PCG ON length={len(a)}; PCG OFF length={len(b)})")
    if a != b:
        # First differing char index for quick orientation
        diff_idx = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        lines.append(f"first differing offset: {diff_idx}")
    lines.append("")

lines.insert(2, f"**VERDICT: {'OUTPUTS_MATCH (clean Y is correctness-OK)' if all_match else 'OUTPUTS_DIFFER (clean Y has silent corruption)'}**")
lines.insert(3, "")
pathlib.Path(report_path).write_text("\n".join(lines))
print("AUDIT_MATCH" if all_match else "AUDIT_DIFFER")
PYEOF

echo ""
echo "=== audit_report.md ==="
cat "$RESULTS/audit_report.md"
