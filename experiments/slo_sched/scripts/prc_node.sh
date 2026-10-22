#!/usr/bin/env bash
# prc_node.sh — verify the head of PR-C (pr/goodput-metric) on a RADIX node before it is opened.
#
# Runs ON the node in tmux window "prc" of session "sgl-install" (watch it with
# `tmux attach -t sgl-install`). Copied to ~/sgl/prc/ together with p1_bench.py,
# prc_verify.py and prc_mutate.py; setup_node.sh sits in ~/sgl/. The branch comes from the
# fork (origin). Everything it writes goes to ~/sgl/logs/prc/, which the Mac pulls back.
#
#   EXPECT_SHA=<branch head> BASE_SHA=<upstream main it sits on> DEADLINE_EPOCH=<unix time> \
#     bash ~/sgl/prc/prc_node.sh
#
# Steps: environment, branch head from the fork, unit files one per process the way CI
# runs them (a failing file is repeated on the base commit), mutation check, lint and
# pre-commit, the CLI help, upstream's GPU metrics test with and without the wiring, then
# the counter against an exact recount on a live server (prc_verify.py).
set -uo pipefail

EXPECT_SHA="${EXPECT_SHA:?set EXPECT_SHA to the branch head}"
BASE_SHA="${BASE_SHA:?set BASE_SHA to the upstream main commit under the branch}"
DEADLINE_EPOCH="${DEADLINE_EPOCH:-0}"
SGL="$HOME/sgl"
HERE="$SGL/prc"
RUN="$SGL/logs/prc"
MODEL="${MODEL:-Qwen/Qwen3-8B}"
TEST_MODEL="Qwen/Qwen3-0.6B"
export HF_HOME="${HF_HOME:-$HOME/hf}"
CONDA_ROOT="$HOME/miniforge3"
ENV_PREFIX="$CONDA_ROOT/envs/sgl-profiler"
BRANCH=pr/goodput-metric
GPU_TEST=test/registered/observability/test_metrics.py
MANAGER=python/sglang/srt/managers/tokenizer_manager.py
WIRING='                goodput_slo=get_observability().goodput_slo,'

mkdir -p "$RUN"
PROG="$RUN/progress.log"
SUMMARY="$RUN/summary.txt"
: > "$SUMMARY"
T0=$(date +%s)
STEP=0
TOTAL=9

elapsed() { printf '%dm%02ds' $(( ($(date +%s) - T0) / 60 )) $(( ($(date +%s) - T0) % 60 )); }
say()     { printf '[%s +%s] %s\n' "$(date +%H:%M:%S)" "$(elapsed)" "$*" | tee -a "$PROG"; }
step()    { STEP=$((STEP + 1)); printf '\n' | tee -a "$PROG"; say "===== STEP $STEP/$TOTAL: $* ====="; }
record()  { printf '%-46s %s\n' "$1" "$2" >> "$SUMMARY"; say "RESULT  $1 -> $2"; }
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
wait_model() {  # $1 = marker stem
  local n=0
  while [ ! -e "$RUN/.$1_done" ]; do
    [ -e "$RUN/.$1_failed" ] && die "model download" "$1 did not download"
    [ $((n % 4)) -eq 0 ] && say "  waiting for the download … $(du -sh "$HF_HOME" 2>/dev/null | cut -f1) in \$HF_HOME"
    n=$((n + 1)); sleep 15
  done
}

say "PR-C node check — expecting branch head $EXPECT_SHA on upstream main $BASE_SHA"
say "node $(hostname), GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"

# ---------------------------------------------------------------------------
step "environment (Miniforge, conda env, repos) + model downloads in window 'models'"
tmux new-window -d -t sgl-install: -n models \
  "(MODELS='$TEST_MODEL' bash '$SGL/setup_node.sh' models && touch '$RUN/.small_done' || touch '$RUN/.small_failed'); (MODELS='$MODEL' bash '$SGL/setup_node.sh' models && touch '$RUN/.model_done' || touch '$RUN/.model_failed'); exec bash" \
  2>/dev/null || say "note: could not open the 'models' window (not inside tmux session sgl-install?)"
setup conda env repos || die "setup conda/env/repos" "setup_node.sh failed, see $SGL/logs/setup_*.log"
record "setup conda/env/repos" "ok"

# ---------------------------------------------------------------------------
step "check out the branch head from the fork and install sglang"
G=(git -C "$SGL/sglang")
"${G[@]}" fetch -q origin "+refs/heads/$BRANCH:refs/heads/pr-c-head" 2>&1 | tee -a "$PROG"
GOT=$("${G[@]}" rev-parse pr-c-head)
[ "$GOT" = "$EXPECT_SHA" ] || die "branch head sha" "the fork has $GOT, expected $EXPECT_SHA"
[ "$("${G[@]}" rev-parse pr-c-head~1)" = "$BASE_SHA" ] || die "branch base sha" "the commit under the head is $("${G[@]}" rev-parse pr-c-head~1), expected $BASE_SHA"
FI_VER=$("${G[@]}" show pr-c-head:python/pyproject.toml | grep -oE 'flashinfer_python\[cu13\]==[^"]+' | sed 's/.*==//')
[ -n "$FI_VER" ] || die "flashinfer pin" "could not read the flashinfer pin from python/pyproject.toml"
say "flashinfer pin in the checkout: $FI_VER"
export FLASHINFER_VER="$FI_VER"
SGLANG_REF=pr-c-head setup repos sglang flashinfer verify manifest || die "sglang install" "setup_node.sh failed, see $SGL/logs/setup_*.log"
HEAD_SHA=$("${G[@]}" rev-parse HEAD)
[ "$HEAD_SHA" = "$EXPECT_SHA" ] || die "checked-out sha" "HEAD is $HEAD_SHA, expected $EXPECT_SHA"
say "sglang HEAD        $("${G[@]}" log -1 --format='%h %an | %s')"
say "on upstream main   $("${G[@]}" log -1 --format='%h %cs %s' "$BASE_SHA" | cut -c1-110)"
say "PR delta: $("${G[@]}" diff --shortstat "$BASE_SHA" HEAD)"
record "branch head installed" "ok ($(echo "$HEAD_SHA" | cut -c1-10) on $(echo "$BASE_SHA" | cut -c1-10))"
cp "$SGL/env_manifest.txt" "$RUN/" 2>/dev/null
activate
cd "$SGL/sglang"
python -c "import sys, torch, transformers, flashinfer; print('python', sys.version.split()[0], '| torch', torch.__version__, '| transformers', transformers.__version__, '| flashinfer', flashinfer.__version__)" 2>&1 | tail -1 | sed 's/^/    /' | tee -a "$PROG"
mapfile -t PR_FILES < <("${G[@]}" diff --name-only "$BASE_SHA" HEAD)
mapfile -t PY_FILES < <(printf '%s\n' "${PR_FILES[@]}" | grep '\.py$')
mapfile -t UNIT_TESTS < <(ls test/registered/unit/observability/test_*.py test/registered/unit/server_args/test_*.py \
  test/registered/unit/managers/test_tokenizer_manager_rid_cleanup.py test/registered/unit/managers/test_multi_tokenizer_mixin.py \
  test/registered/unit/test_server_args_migration.py test/registered/unit/test_server_args_cli_metadata.py \
  test/registered/unit/test_runtime_context_config_bags.py test/registered/unit/test_global_config_read_ratchet.py \
  test/registered/unit/test_pre_publish_readers.py 2>/dev/null)

# ---------------------------------------------------------------------------
step "CPU unit files, one per process (${#UNIT_TESTS[@]} files, python3 <file> -f)"
failed=()
for f in "${UNIT_TESTS[@]}"; do
  log="$RUN/unit_$(basename "$f" .py).log"
  timeout 600 python3 "$f" -f > "$log" 2>&1; rc=$?
  ran=$(grep -E "^Ran [0-9]+ tests? in|[0-9]+ passed" "$log" | tail -1 | cut -c1-70)
  verdict=$(grep -E "^(OK|FAILED)" "$log" | tail -1)
  printf '    rc=%s | %s | %s | %s\n' "$rc" "${ran:-no count line}" "${verdict:--}" "$f" | tee -a "$PROG"
  [ $rc -eq 0 ] || failed+=("$f")
done
new_failures=0
if [ ${#failed[@]} -gt 0 ]; then
  say "  ${#failed[@]} file(s) failed; repeating them on the base commit $BASE_SHA"
  "${G[@]}" worktree add -q --detach "$SGL/base" "$BASE_SHA" 2>&1 | tee -a "$PROG"
  for f in "${failed[@]}"; do
    log="$RUN/base_$(basename "$f" .py).log"
    ( cd "$SGL/base" && PYTHONPATH="$SGL/base/python" timeout 600 python3 "$f" -f > "$log" 2>&1 ); rc=$?
    if [ $rc -eq 0 ]; then new_failures=$((new_failures + 1)); say "    NEW FAILURE (passes on the base): $f"; else say "    also fails on the base: $f"; fi
  done
  "${G[@]}" worktree remove --force "$SGL/base"
fi
if [ $new_failures -eq 0 ]; then
  record "unit files (changed + neighbours)" "PASS ($(( ${#UNIT_TESTS[@]} - ${#failed[@]} )) of ${#UNIT_TESTS[@]} files pass; ${#failed[@]} fail on the base commit too)"
else
  record "unit files (changed + neighbours)" "FAIL ($new_failures file(s) fail only with the PR, see unit_*.log)"
fi

# ---------------------------------------------------------------------------
step "mutation check: every listed regression must turn a unit file red"
python "$HERE/prc_mutate.py" "$SGL/sglang" > "$RUN/mutation.log" 2>&1; RC=$?
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
  0)   record "pre-commit (${#PR_FILES[@]} files)" "PASS ($(grep -c 'Passed$' "$RUN/pre_commit.log") hooks passed, $(grep -c 'Skipped$' "$RUN/pre_commit.log") skipped as not applicable)";;
  124) record "pre-commit (${#PR_FILES[@]} files)" "FAIL (timeout 420 s)";;
  *)   record "pre-commit (${#PR_FILES[@]} files)" "FAIL rc=$RC (see pre_commit.log)";;
esac
[ -z "$("${G[@]}" status --porcelain)" ] || { record "clean tree after pre-commit" "FAIL (a hook changed files)"; "${G[@]}" diff > "$RUN/pre_commit_changes.diff"; "${G[@]}" checkout -q -- .; }

# ---------------------------------------------------------------------------
step "the option in the CLI help"
python -m sglang.launch_server --help > "$RUN/help_launch_server.txt" 2>&1; RC=$?
grep -A4 -e '--goodput-slo' "$RUN/help_launch_server.txt" | sed 's/^/    /' | tee -a "$PROG"
if [ $RC -eq 0 ] && grep -q -e '--goodput-slo' "$RUN/help_launch_server.txt"; then record "--goodput-slo in --help" "PASS"; else record "--goodput-slo in --help" "FAIL (rc $RC)"; fi

# ---------------------------------------------------------------------------
step "upstream's GPU metrics test ($TEST_MODEL): green as committed, red without the wiring"
wait_model small
timeout 900 python3 "$GPU_TEST" TestEnableMetrics.test_metrics_1gpu -f > "$RUN/gpu_test.log" 2>&1; RC=$?
grep -E "^Ran |^OK|^FAILED" "$RUN/gpu_test.log" | sed 's/^/    /' | tee -a "$PROG"
grep "^sglang:num_good_requests_total" "$RUN/gpu_test.log" | head -3 | cut -c1-200 | sed 's/^/    /' | tee -a "$PROG"
if [ $RC -eq 0 ]; then record "GPU test test_metrics_1gpu" "PASS"; else record "GPU test test_metrics_1gpu" "FAIL rc=$RC (see gpu_test.log)"; fi
pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5
if [ "$(grep -cF "$WIRING" "$MANAGER")" = 1 ]; then
  grep -vF "$WIRING" "$MANAGER" > "$RUN/.manager_mutated" && cp "$RUN/.manager_mutated" "$MANAGER"
  timeout 900 python3 "$GPU_TEST" TestEnableMetrics.test_metrics_1gpu -f > "$RUN/gpu_test_unwired.log" 2>&1; RC=$?
  "${G[@]}" checkout -q -- "$MANAGER"; rm -f "$RUN/.manager_mutated"
  grep -E "^Ran |^OK|^FAILED|^KeyError" "$RUN/gpu_test_unwired.log" | tail -4 | cut -c1-160 | sed 's/^/    /' | tee -a "$PROG"
  if [ $RC -ne 0 ] && grep -q "KeyError: 'sglang:num_good_requests_total'" "$RUN/gpu_test_unwired.log"; then record "GPU test red without the wiring" "PASS"; else record "GPU test red without the wiring" "FAIL rc=$RC (see gpu_test_unwired.log)"; fi
else
  record "GPU test red without the wiring" "FAIL (wiring line not found once)"
fi
pkill -9 -f sglang.launch_server 2>/dev/null; sleep 5
[ -z "$("${G[@]}" status --porcelain)" ] || die "clean tree after the GPU test" "the checkout is dirty: $("${G[@]}" status --porcelain | head -3)"

# ---------------------------------------------------------------------------
step "wait for $MODEL"
wait_model model
record "models" "ok ($(du -sh "$HF_HOME" | cut -f1))"

# ---------------------------------------------------------------------------
step "the counter against an exact recount on a live server (prc_verify.py)"
( cd "$HERE" && python "$HERE/prc_verify.py" --out "$RUN" --model "$MODEL" --deadline-epoch "$DEADLINE_EPOCH" ); RC=$?
if [ $RC -eq 0 ]; then
  record "live server: counter == recount" "PASS ($(python -c "import json; d=json.load(open('$RUN/verify_summary.json')); print(d['checks'], 'checks')") in $(grep -c . "$RUN/summary.jsonl") phases)"
else
  record "live server: counter == recount" "FAIL (see verify_summary.json)"
fi

# ---------------------------------------------------------------------------
nvidia-smi > "$RUN/nvidia_smi_end.txt" 2>&1
"${G[@]}" rev-parse HEAD > "$RUN/final_head_sha.txt"
"${G[@]}" status --porcelain > "$RUN/git_status_end.txt"
cp "$SGL"/logs/setup_*.log "$RUN/" 2>/dev/null
finish
