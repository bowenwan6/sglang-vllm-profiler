#!/usr/bin/env bash
# pra_node.sh — verify the head of PR-A (pr/bench-goodput) on a RADIX node before it is opened.
#
# Runs ON the node in tmux window "pra" of session "sgl-install" (watch it with
# `tmux attach -t sgl-install`). Copied to ~/sgl/pra/ together with p1_bench.py,
# pra_verify.py, pra_mutate.py and the git bundle of the branch; setup_node.sh sits in
# ~/sgl/. Everything it writes goes to ~/sgl/logs/pra/, which the Mac pulls back.
#
#   EXPECT_SHA=<branch head> BASE_SHA=<upstream main it sits on> DEADLINE_EPOCH=<unix time> \
#     bash ~/sgl/pra/pra_node.sh
#
# Steps: environment, branch head from the bundle, unit files one per process the way CI
# runs them, mutation check, lint and pre-commit, the CLI help, then bench_serving against
# a live server (pra_verify.py).
set -uo pipefail

EXPECT_SHA="${EXPECT_SHA:?set EXPECT_SHA to the branch head}"
BASE_SHA="${BASE_SHA:?set BASE_SHA to the upstream main commit under the branch}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"
SGL="$HOME/sgl"
HERE="$SGL/pra"
RUN="$SGL/logs/pra"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
export HF_HOME="${HF_HOME:-$HOME/hf}"
CONDA_ROOT="$HOME/miniforge3"
ENV_PREFIX="$CONDA_ROOT/envs/sgl-profiler"
BRANCH=pr/bench-goodput
UNIT_TESTS=(
  test/registered/unit/bench/test_bench_serving_goodput.py
  test/registered/unit/bench/test_bench_serving_prompt_len.py
  test/registered/unit/bench/test_benchmark_endpoint.py
  test/registered/bench_fn/test_bench_serving_reasoning_stream.py
  test/registered/bench_fn/test_steady_state_benchmark.py
  test/registered/bench_fn/test_benchmark_datasets_api.py
)
PR_FILES=(
  python/sglang/benchmark/goodput.py
  python/sglang/benchmark/serving.py
  test/registered/unit/bench/test_bench_serving_goodput.py
  docs/docs/developer_guide/bench_serving.mdx
)
PY_FILES=("${PR_FILES[@]:0:3}")

mkdir -p "$RUN"
PROG="$RUN/progress.log"
SUMMARY="$RUN/summary.txt"
: > "$SUMMARY"
T0=$(date +%s)
STEP=0
TOTAL=8

elapsed() { printf '%dm%02ds' $(( ($(date +%s) - T0) / 60 )) $(( ($(date +%s) - T0) % 60 )); }
say()     { printf '[%s +%s] %s\n' "$(date +%H:%M:%S)" "$(elapsed)" "$*" | tee -a "$PROG"; }
step()    { STEP=$((STEP + 1)); printf '\n' | tee -a "$PROG"; say "===== STEP $STEP/$TOTAL: $* ====="; }
record()  { printf '%-44s %s\n' "$1" "$2" >> "$SUMMARY"; say "RESULT  $1 -> $2"; }
finish()  {
  printf '\n' | tee -a "$PROG"
  say "===== SUMMARY (total $(elapsed)) ====="
  tee -a "$PROG" < "$SUMMARY"
  if grep -q FAIL "$SUMMARY"; then say "OVERALL: FAIL"; else say "OVERALL: PASS"; fi
  touch "$RUN/.finished"
  say "ALL DONE — results in $RUN"
}
die()     { record "$1" "FAIL (fatal)"; say "STOPPED: $2"; finish; exit 1; }
activate() { set +u; source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sgl-profiler; set -u; }
setup()   { bash "$SGL/setup_node.sh" "$@" 2>&1 | grep --line-buffered -E '^=== |^FAILED|^present|^profiler @|^sglang   @|^torch |^sglang [0-9]|^nvcc|^wrote ' | sed -u 's/^/    /' | tee -a "$PROG"; return "${PIPESTATUS[0]}"; }

say "PR-A node check — expecting branch head $EXPECT_SHA on upstream main $BASE_SHA"
say "node $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------
step "environment (Miniforge, conda env, repos) + $MODEL download in window 'models'"
tmux new-window -d -t sgl-install: -n models \
  "MODELS='$MODEL' bash '$SGL/setup_node.sh' models && touch '$RUN/.model_done' || touch '$RUN/.model_failed'; exec bash" \
  2>/dev/null || say "note: could not open the 'models' window (not inside tmux session sgl-install?)"
setup conda env repos || die "setup conda/env/repos" "setup_node.sh failed, see $SGL/logs/setup_*.log"
record "setup conda/env/repos" "ok"

# ---------------------------------------------------------------------------
step "check out the branch head from the bundle and install sglang"
G=(git -C "$SGL/sglang")
"${G[@]}" bundle verify "$HERE/pra.bundle" > "$RUN/bundle_verify.log" 2>&1 || { cat "$RUN/bundle_verify.log"; die "bundle verify" "bundle prerequisites missing on the node"; }
"${G[@]}" fetch -q "$HERE/pra.bundle" "+refs/heads/$BRANCH:refs/heads/pr-a-head" 2>&1 | tee -a "$PROG"
GOT=$("${G[@]}" rev-parse pr-a-head)
[ "$GOT" = "$EXPECT_SHA" ] || die "branch head sha" "bundle has $GOT, expected $EXPECT_SHA"
[ "$("${G[@]}" rev-parse pr-a-head~1)" = "$BASE_SHA" ] || die "branch base sha" "the commit under the head is $("${G[@]}" rev-parse pr-a-head~1), expected $BASE_SHA"
FI_VER=$("${G[@]}" show pr-a-head:python/pyproject.toml | grep -oE 'flashinfer_python\[cu13\]==[^"]+' | sed 's/.*==//')
[ -n "$FI_VER" ] || die "flashinfer pin" "could not read the flashinfer pin from python/pyproject.toml"
say "flashinfer pin in the checkout: $FI_VER"
export FLASHINFER_VER="$FI_VER"
SGLANG_REF=pr-a-head setup repos sglang flashinfer verify manifest || die "sglang install" "setup_node.sh failed, see $SGL/logs/setup_*.log"
HEAD_SHA=$("${G[@]}" rev-parse HEAD)
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "checked-out sha" "HEAD is $HEAD_SHA, expected $EXPECT_SHA"
say "sglang HEAD        $("${G[@]}" log -1 --format='%h %an | %s')"
say "on upstream main   $("${G[@]}" log -1 --format='%h %cs %s' "$BASE_SHA" | cut -c1-110)"
say "node upstream/main $("${G[@]}" log -1 --format='%h %cs' upstream/main) ($("${G[@]}" rev-list --count "$BASE_SHA..upstream/main") commits newer than the base)"
say "PR delta: $("${G[@]}" diff --shortstat "$BASE_SHA" HEAD)"
record "branch head installed" "ok ($(echo "$HEAD_SHA" | cut -c1-10) on $(echo "$BASE_SHA" | cut -c1-10))"
cp "$SGL/env_manifest.txt" "$RUN/" 2>/dev/null
activate
cd "$SGL/sglang"
python -c "import sys, torch, transformers, flashinfer; print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| flashinfer', flashinfer.__version__)" 2>&1 | tail -1 | sed 's/^/    /' | tee -a "$PROG"

# ---------------------------------------------------------------------------
step "CPU unit files, one per process (${#UNIT_TESTS[@]} files, python3 <file> -f)"
failed=0
for f in "${UNIT_TESTS[@]}"; do
  log="$RUN/unit_$(basename "$f" .py).log"
  timeout 420 python3 "$f" -f > "$log" 2>&1; rc=$?
  ran=$(grep -E "^Ran [0-9]+ tests? in|[0-9]+ passed" "$log" | tail -1 | cut -c1-70)
  verdict=$(grep -E "^(OK|FAILED)" "$log" | tail -1)
  printf '    rc=%s | %s | %s | %s\n' "$rc" "${ran:-no count line}" "${verdict:--}" "$f" | tee -a "$PROG"
  [ $rc -eq 0 ] || failed=$((failed + 1))
done
if [ $failed -eq 0 ]; then record "unit files (ours + 5 neighbours)" "PASS (${#UNIT_TESTS[@]} of ${#UNIT_TESTS[@]} files)"; else record "unit files (ours + 5 neighbours)" "FAIL ($failed of ${#UNIT_TESTS[@]} files, see unit_*.log)"; fi
grep -n "register_.*_ci(\|CustomTestCase\|unittest.main" "${UNIT_TESTS[0]}" > "$RUN/unit_file_checklist.txt" 2>&1

# ---------------------------------------------------------------------------
step "mutation check: every listed regression must turn the unit file red"
python "$HERE/pra_mutate.py" "$SGL/sglang" > "$RUN/mutation.log" 2>&1; RC=$?
sed 's/^/    /' "$RUN/mutation.log" | tee -a "$PROG"
if [ $RC -eq 0 ]; then record "mutation check" "PASS ($(grep -c '^red' "$RUN/mutation.log") of $(grep -c ' | ' "$RUN/mutation.log") mutations caught)"; else record "mutation check" "FAIL (see mutation.log)"; fi
[ -z "$("${G[@]}" status --porcelain)" ] || die "clean tree after mutation" "the checkout is dirty: $("${G[@]}" status --porcelain | head -3)"

# ---------------------------------------------------------------------------
step "lint at the pinned versions and the pre-commit suite on the PR's ${#PR_FILES[@]} files"
UVT=("$ENV_PREFIX/bin/uv" tool run)
"${UVT[@]}" ruff@0.15.1 format --check --diff "${PY_FILES[@]}" > "$RUN/ruff_format.log" 2>&1 && record "ruff format --check" "PASS" || record "ruff format --check" "FAIL (see ruff_format.log)"
"${UVT[@]}" ruff@0.15.1 check --select=F401,F821,F823,UP037 "${PY_FILES[@]}" > "$RUN/ruff_check.log" 2>&1 && record "ruff check" "PASS" || record "ruff check" "FAIL (see ruff_check.log)"
"${UVT[@]}" isort@7.0.0 --check --diff "${PY_FILES[@]}" > "$RUN/isort.log" 2>&1 && record "isort --check" "PASS" || record "isort --check" "FAIL (see isort.log)"
"${UVT[@]}" codespell@2.4.1 --config .codespellrc "${PR_FILES[@]}" > "$RUN/codespell.log" 2>&1 && record "codespell" "PASS" || record "codespell" "FAIL (see codespell.log)"
# no-commit-to-branch only guards the name of the local branch; the node checkout is not the PR branch.
SKIP=no-commit-to-branch timeout 420 "${UVT[@]}" pre-commit run --files "${PR_FILES[@]}" > "$RUN/pre_commit.log" 2>&1; RC=$?
grep -E '\.\.\.+' "$RUN/pre_commit.log" | sed 's/^/    /' | tee -a "$PROG"
case $RC in
  0)   record "pre-commit (4 files)" "PASS ($(grep -c 'Passed$' "$RUN/pre_commit.log") hooks passed, $(grep -c 'Skipped$' "$RUN/pre_commit.log") skipped as not applicable)";;
  124) record "pre-commit (4 files)" "FAIL (timeout 420 s)";;
  *)   record "pre-commit (4 files)" "FAIL rc=$RC (see pre_commit.log)";;
esac
[ -z "$("${G[@]}" status --porcelain)" ] || { record "clean tree after pre-commit" "FAIL (a hook changed files)"; "${G[@]}" diff > "$RUN/pre_commit_changes.diff"; "${G[@]}" checkout -q -- .; }

# ---------------------------------------------------------------------------
step "the option in the CLI help (both entry points)"
python -m sglang.benchmark.serving --help > "$RUN/help_benchmark_serving.txt" 2>&1; RC1=$?
python -m sglang.bench_serving --help > "$RUN/help_bench_serving.txt" 2>&1; RC2=$?
grep -A7 -e '--goodput KEY:VALUE' "$RUN/help_benchmark_serving.txt" | sed 's/^/    /' | tee -a "$PROG"
if [ $RC1 -eq 0 ] && [ $RC2 -eq 0 ] && grep -q -e '--goodput KEY:VALUE' "$RUN/help_benchmark_serving.txt" && grep -q -e '--goodput KEY:VALUE' "$RUN/help_bench_serving.txt"; then record "--goodput in --help" "PASS"; else record "--goodput in --help" "FAIL (rc $RC1/$RC2)"; fi

# ---------------------------------------------------------------------------
step "wait for $MODEL"
n=0
while [ ! -e "$RUN/.model_done" ]; do
  [ -e "$RUN/.model_failed" ] && die "model download" "$MODEL did not download"
  [ $((n % 4)) -eq 0 ] && say "  waiting for the download … $(du -sh "$HF_HOME" 2>/dev/null | cut -f1) in \$HF_HOME"
  n=$((n + 1)); sleep 15
done
record "model" "ok ($(du -sh "$HF_HOME" | cut -f1))"

# ---------------------------------------------------------------------------
step "bench_serving against a live server (pra_verify.py)"
( cd "$HERE" && python "$HERE/pra_verify.py" --out "$RUN" --model "$MODEL" --deadline-epoch "$DEADLINE_EPOCH" ); RC=$?
if [ $RC -eq 0 ]; then
  record "live server: reported == recount" "PASS ($(python -c "import json; d=json.load(open('$RUN/verify_summary.json')); print(d['checks'], 'checks')") in $(grep -c . "$RUN/summary.jsonl") runs)"
else
  record "live server: reported == recount" "FAIL (see verify_summary.json)"
fi

# ---------------------------------------------------------------------------
nvidia-smi > "$RUN/nvidia_smi_end.txt" 2>&1
"${G[@]}" rev-parse HEAD > "$RUN/final_head_sha.txt"
"${G[@]}" status --porcelain > "$RUN/git_status_end.txt"
cp "$SGL"/logs/setup_*.log "$RUN/" 2>/dev/null
finish
