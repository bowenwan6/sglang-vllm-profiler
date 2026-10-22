#!/usr/bin/env bash
# p1_node.sh — session P1 on a RADIX node: environment, step 0 (PR-A in a real SGLang
# environment), then tasks T1-T4 via p1_bench.py.
#
# Runs ON the node in tmux window "p1" of session "sgl-install". Copied to ~/sgl/p1/
# together with setup_node.sh (in ~/sgl/) and p1_bench.py. Everything it writes goes to
# ~/sgl/logs/p1/, which the Mac pulls back.
#
#   EXPECT_SHA=<fork branch head> DEADLINE_EPOCH=<unix time> bash ~/sgl/p1/p1_node.sh
set -uo pipefail

EXPECT_SHA="${EXPECT_SHA:?set EXPECT_SHA to the head of feat/bench-goodput}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"
SGL="$HOME/sgl"
HERE="$SGL/p1"
RUN="$SGL/logs/p1"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export SGLANG_REF="${SGLANG_REF:-origin/feat/bench-goodput}"
export FLASHINFER_VER="${FLASHINFER_VER:-0.7.0.post1}"
CONDA_ROOT="$HOME/miniforge3"
ENV_PREFIX="$CONDA_ROOT/envs/sgl-profiler"
PR_FILES=(
  python/sglang/benchmark/serving.py
  test/registered/unit/bench/test_bench_serving_goodput.py
  docs/docs/developer_guide/bench_serving.mdx
)

mkdir -p "$RUN"
PROG="$RUN/progress.log"
SUMMARY="$RUN/step0_summary.txt"
: > "$SUMMARY"
T0=$(date +%s)
say()    { printf '[%s +%02dm] %s\n' "$(date +%H:%M:%S)" $(( ($(date +%s) - T0) / 60 )) "$*" | tee -a "$PROG"; }
record() { printf '%-36s %s\n' "$1" "$2" >> "$SUMMARY"; say "RESULT  $1 -> $2"; }
die()    { record "$1" "FAIL (fatal)"; say "STOPPED: $2"; touch "$RUN/.failed" "$RUN/.finished"; exit 1; }
activate() { set +u; source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sgl-profiler; set -u; }
setup()  { bash "$SGL/setup_node.sh" "$@" 2>&1 | grep --line-buffered -E '^=== |^FAILED|^present|^profiler @|^sglang   @|^torch |^sglang [0-9]|^nvcc|^wrote ' | sed -u 's/^/    /' | tee -a "$PROG"; return "${PIPESTATUS[0]}"; }

say "P1 — goodput benchmark on real tasks; expecting sglang $EXPECT_SHA"
say "node $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ---- environment -------------------------------------------------------------
say "===== environment + model download (window 'models') ====="
tmux new-window -d -t sgl-install: -n models \
  "MODELS='$MODEL' bash '$SGL/setup_node.sh' models && touch '$RUN/.model_done' || touch '$RUN/.model_failed'; exec bash" \
  2>/dev/null || say "note: could not open the 'models' window"
setup conda env repos sglang flashinfer verify manifest || die "setup" "setup_node.sh failed, see $SGL/logs/setup_*.log"
HEAD_SHA=$(git -C "$SGL/sglang" rev-parse HEAD)
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "sglang sha" "HEAD is $HEAD_SHA, expected $EXPECT_SHA"
record "environment" "ok (sglang $(echo "$HEAD_SHA" | cut -c1-10))"
cp "$SGL/env_manifest.txt" "$RUN/" 2>/dev/null
activate
cd "$SGL/sglang"

# ---- step 0: PR-A in a real environment --------------------------------------
say "===== step 0: unit tests, lint, pre-commit ====="
python -m pytest -q -p no:cacheprovider \
  test/registered/unit/bench/test_bench_serving_goodput.py \
  test/registered/unit/bench/test_bench_serving_prompt_len.py > "$RUN/unit_tests.log" 2>&1
RC=$?
tail -3 "$RUN/unit_tests.log" | sed 's/^/    /' | tee -a "$PROG"
if [ $RC -eq 0 ]; then record "unit tests (2 bench files)" "PASS ($(tail -1 "$RUN/unit_tests.log" | tr -d '='))"; else die "unit tests" "the goodput unit file fails in the real environment"; fi

RUFF=("$ENV_PREFIX/bin/uv" tool run ruff@0.15.1)
"${RUFF[@]}" format --check --diff "${PR_FILES[@]:0:2}" > "$RUN/ruff_format.log" 2>&1 && record "ruff format" "PASS" || record "ruff format" "FAIL (see ruff_format.log)"
"${RUFF[@]}" check "${PR_FILES[@]:0:2}" > "$RUN/ruff_check.log" 2>&1 && record "ruff check" "PASS" || record "ruff check" "FAIL (see ruff_check.log)"
SKIP=no-commit-to-branch timeout 300 "$ENV_PREFIX/bin/uv" tool run pre-commit run --files "${PR_FILES[@]}" > "$RUN/pre_commit.log" 2>&1
RC=$?
tail -25 "$RUN/pre_commit.log" | sed 's/^/    /' >> "$PROG"
case $RC in 0) record "pre-commit (3 files)" "PASS";; 124) record "pre-commit (3 files)" "SKIPPED (timeout 300 s)";; *) record "pre-commit (3 files)" "FAIL rc=$RC (see pre_commit.log)";; esac

# ---- wait for the model ------------------------------------------------------
n=0
while [ ! -e "$RUN/.model_done" ]; do
  [ -e "$RUN/.model_failed" ] && die "model download" "$MODEL did not download"
  [ $((n % 4)) -eq 0 ] && say "  waiting for $MODEL … $(du -sh "$HF_HOME" 2>/dev/null | cut -f1) in \$HF_HOME"
  n=$((n + 1)); sleep 15
done
record "model" "ok ($(du -sh "$HF_HOME" | cut -f1))"

# ---- T1-T4 -------------------------------------------------------------------
say "===== tasks T1-T4 (p1_bench.py) ====="
python "$HERE/p1_bench.py" --out "$RUN" --model "$MODEL" --deadline-epoch "$DEADLINE_EPOCH" ${P1_ONLY:+--only "$P1_ONLY"} ${P1_C0:+--c0 "$P1_C0"}
RC=$?
nvidia-smi > "$RUN/nvidia_smi_end.txt" 2>&1
say "p1_bench.py exit $RC"
touch "$RUN/.finished"
say "ALL DONE — results in $RUN"
