#!/usr/bin/env bash
# prb_node.sh — a PR-B session on a RADIX node: environment, the unit tests in a real SGLang
# environment (D1), then prb_run.py for the phases given.
#
# Runs ON the node in tmux window "prb" of session "sgl-install". Copied to ~/sgl/prb/ with
# p1_bench.py, prb_run.py, prb_ladder.py, slo_client.py and stub_server.py; setup_node.sh is in
# ~/sgl/. Everything it writes goes to ~/sgl/logs/$RUN_NAME/, which the Mac pulls back.
#
#   RUN_NAME=p2 EXPECT_SHA=<head of exp/slo-node> BASE_SHA=<head of feat/bench-goodput> \
#   DEADLINE_EPOCH=<unix time> PHASES=ladder,calib,check,pilot,u3,u4,t1rep,t4rep bash ~/sgl/prb/prb_node.sh
set -uo pipefail

RUN_NAME="${RUN_NAME:?set RUN_NAME, for example p2}"
EXPECT_SHA="${EXPECT_SHA:?set EXPECT_SHA to the head of exp/slo-node}"
BASE_SHA="${BASE_SHA:?set BASE_SHA to the head of feat/bench-goodput}"
PHASES="${PHASES:?set PHASES}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"
SGL="$HOME/sgl"
HERE="$SGL/prb"
RUN="$SGL/logs/$RUN_NAME"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
SMALL_MODEL="${SMALL_MODEL:-Qwen/Qwen3-0.6B}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export SGLANG_REF="${SGLANG_REF:-origin/exp/slo-node}"
export FLASHINFER_VER="${FLASHINFER_VER:-0.7.0.post1}"
CONDA_ROOT="$HOME/miniforge3"
UNIT_FILES=(
  test/registered/unit/managers/test_scheduler_timeouts.py
  test/registered/unit/managers/test_io_struct.py
  test/registered/unit/entrypoints/openai/test_serving_completions.py
  test/registered/unit/entrypoints/openai/test_serving_chat.py
  test/registered/unit/bench/test_bench_serving_goodput.py
  test/registered/unit/bench/test_bench_serving_stream_error.py
)

mkdir -p "$RUN"
PROG="$RUN/progress.log"
SUMMARY="$RUN/step0_summary.txt"
: > "$SUMMARY"
T0=$(date +%s)
say()    { printf '[%s +%02dm] %s\n' "$(date +%H:%M:%S)" $(( ($(date +%s) - T0) / 60 )) "$*" | tee -a "$PROG"; }
record() { printf '%-44s %s\n' "$1" "$2" >> "$SUMMARY"; say "RESULT  $1 -> $2"; }
die()    { record "$1" "FAIL (fatal)"; say "STOPPED: $2"; touch "$RUN/.failed" "$RUN/.finished"; exit 1; }
activate() { set +u; source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sgl-profiler; set -u; }
setup()  { bash "$SGL/setup_node.sh" "$@" 2>&1 | grep --line-buffered -E '^=== |^FAILED|^present|^profiler @|^sglang   @|^torch |^sglang [0-9]|^nvcc|^wrote ' | sed -u 's/^/    /' | tee -a "$PROG"; return "${PIPESTATUS[0]}"; }

say "$RUN_NAME — per-request waiting timeout; expecting sglang $EXPECT_SHA, base $BASE_SHA; phases $PHASES"
say "node $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ---- environment -------------------------------------------------------------
say "===== environment + model download (window 'models') ====="
tmux new-window -d -t sgl-install: -n models \
  "MODELS='$MODEL $SMALL_MODEL' bash '$SGL/setup_node.sh' models && touch '$RUN/.model_done' || touch '$RUN/.model_failed'; exec bash" \
  2>/dev/null || say "note: could not open the 'models' window"
setup conda env repos sglang flashinfer verify manifest || die "setup" "setup_node.sh failed, see $SGL/logs/setup_*.log"
HEAD_SHA=$(git -C "$SGL/sglang" rev-parse HEAD)
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "sglang sha" "HEAD is $HEAD_SHA, expected $EXPECT_SHA"
if [ ! -d "$SGL/sglang-base" ]; then
  git -C "$SGL/sglang" worktree add -q --detach "$SGL/sglang-base" "$BASE_SHA" || die "base tree" "could not create the unpatched worktree"
fi
[ "$(git -C "$SGL/sglang-base" rev-parse HEAD)" = "$BASE_SHA" ] || die "base tree" "the unpatched worktree is not at $BASE_SHA"
record "environment" "ok (patched $(echo "$HEAD_SHA" | cut -c1-10), base $(echo "$BASE_SHA" | cut -c1-10))"
cp "$SGL/env_manifest.txt" "$RUN/" 2>/dev/null
activate
cd "$SGL/sglang"

# ---- D1: unit tests in a real environment, one file per process as in CI ------
say "===== D1: unit tests ====="
: > "$RUN/unit_tests.log"
D1_FAILED=0
for f in "${UNIT_FILES[@]}"; do
  python -m pytest -q -p no:cacheprovider "$f" > "$RUN/unit_one.log" 2>&1
  RC=$?
  { echo "##### $f (rc=$RC)"; cat "$RUN/unit_one.log"; } >> "$RUN/unit_tests.log"
  LAST=$(tail -1 "$RUN/unit_one.log" | tr -d '=')
  if [ $RC -eq 0 ]; then record "unit $(basename "$f")" "PASS ($LAST)"; else record "unit $(basename "$f")" "FAIL rc=$RC ($LAST)"; D1_FAILED=1; fi
done
[ $D1_FAILED -eq 0 ] && record "D1" "PASS" || record "D1" "FAIL (the session continues; see unit_tests.log)"

# The wider sweep runs beside the GPU work: every unit file of the two touched areas, then
# each failing file again on the unpatched tree, to separate what the patch broke from what
# was already broken.
cat > "$RUN/unit_sweep.sh" <<SWEEP
# No "set -u" here: conda's activation scripts read unset variables.
set -o pipefail
source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sgl-profiler
cd "$SGL/sglang"
# An ordinal that does not exist hides the GPU the benchmark server is using.
export CUDA_VISIBLE_DEVICES=99
: > "$RUN/unit_sweep.txt"
for f in \$(ls test/registered/unit/managers/test_*.py test/registered/unit/entrypoints/openai/test_*.py test/registered/unit/entrypoints/test_*.py 2>/dev/null); do
  if python -m pytest -q -p no:cacheprovider "\$f" > "$RUN/unit_sweep_one.log" 2>&1; then echo "PASS \$f" >> "$RUN/unit_sweep.txt"
  else
    echo "FAIL \$f :: \$(tail -1 "$RUN/unit_sweep_one.log")" >> "$RUN/unit_sweep.txt"
    { echo "##### patched: \$f"; tail -40 "$RUN/unit_sweep_one.log"; } >> "$RUN/unit_sweep_failures.log"
    if (cd "$SGL/sglang-base" && PYTHONPATH="$SGL/sglang-base/python" python -m pytest -q -p no:cacheprovider "\$f" > "$RUN/unit_sweep_one.log" 2>&1); then echo "     base: PASS  <-- new failure" >> "$RUN/unit_sweep.txt"
    else echo "     base: FAIL too :: \$(tail -1 "$RUN/unit_sweep_one.log")" >> "$RUN/unit_sweep.txt"; fi
  fi
done
echo "DONE \$(grep -c '^PASS' "$RUN/unit_sweep.txt") passed, \$(grep -c '^FAIL' "$RUN/unit_sweep.txt") failed, \$(grep -c 'new failure' "$RUN/unit_sweep.txt") new" >> "$RUN/unit_sweep.txt"
SWEEP
tmux new-window -d -t sgl-install: -n sweep "bash '$RUN/unit_sweep.sh'; exec bash" 2>/dev/null || say "note: could not open the 'sweep' window"

# ---- wait for the models -----------------------------------------------------
n=0
while [ ! -e "$RUN/.model_done" ]; do
  [ -e "$RUN/.model_failed" ] && die "model download" "the models did not download"
  [ $((n % 4)) -eq 0 ] && say "  waiting for the models … $(du -sh "$HF_HOME" 2>/dev/null | cut -f1) in \$HF_HOME"
  n=$((n + 1)); sleep 15
done
record "models" "ok ($(du -sh "$HF_HOME" | cut -f1))"

# ---- the phases --------------------------------------------------------------
say "===== phases: $PHASES (prb_run.py) ====="
cd "$HERE"
python "$HERE/prb_run.py" --out "$RUN" --model "$MODEL" --small-model "$SMALL_MODEL" \
  --deadline-epoch "$DEADLINE_EPOCH" --phases "$PHASES" --base-tree "$SGL/sglang-base" \
  ${C_CHAT:+--c-chat "$C_CHAT"} ${C_BATCH:+--c-batch "$C_BATCH"} ${HANGUP:+--hangup}
RC=$?
nvidia-smi > "$RUN/nvidia_smi_end.txt" 2>&1
say "prb_run.py exit $RC"
touch "$RUN/.finished"
say "ALL DONE — results in $RUN"
