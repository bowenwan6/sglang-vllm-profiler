#!/usr/bin/env bash
# pra_recheck.sh — move the node checkout to a later head of PR-A and repeat the CPU-side checks.
#
# For a head that differs from the one pra_node.sh verified only outside the benchmark
# package (a test or a comment): the script refuses to continue if any file under
# python/ or docs/ changed, so the live-server runs of pra_node.sh still describe it.
# Runs ON the node after pra_node.sh has finished and its server is down.
#
#   PREV_SHA=<head pra_node.sh verified> EXPECT_SHA=<new head> BUNDLE=~/sgl/pra/pra2.bundle \
#     bash ~/sgl/pra/pra_recheck.sh
set -uo pipefail

PREV_SHA="${PREV_SHA:?set PREV_SHA to the head pra_node.sh verified}"
EXPECT_SHA="${EXPECT_SHA:?set EXPECT_SHA to the new head}"
BUNDLE="${BUNDLE:?set BUNDLE to the git bundle that carries the new head}"
SGL="$HOME/sgl"
HERE="$SGL/pra"
RUN="$SGL/logs/pra/recheck"
CONDA_ROOT="$HOME/miniforge3"
ENV_PREFIX="$CONDA_ROOT/envs/sgl-profiler"
BRANCH=pr/bench-goodput
TEST=test/registered/unit/bench/test_bench_serving_goodput.py
PR_FILES=(
  python/sglang/benchmark/goodput.py
  python/sglang/benchmark/serving.py
  "$TEST"
  docs/docs/developer_guide/bench_serving.mdx
)

mkdir -p "$RUN"
SUMMARY="$RUN/summary.txt"
: > "$SUMMARY"
say()    { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$RUN/progress.log"; }
record() { printf '%-44s %s\n' "$1" "$2" >> "$SUMMARY"; say "RESULT  $1 -> $2"; }
finish() { if grep -q FAIL "$SUMMARY"; then say "OVERALL: FAIL"; else say "OVERALL: PASS"; fi; touch "$RUN/.finished"; }
die()    { record "$1" "FAIL (fatal)"; say "STOPPED: $2"; finish; exit 1; }

G=(git -C "$SGL/sglang")
pgrep -f sglang.launch_server > /dev/null && die "no server running" "a server is still running from this checkout"
[ "$("${G[@]}" rev-parse HEAD)" = "$PREV_SHA" ] || die "previous head" "the checkout is at $("${G[@]}" rev-parse HEAD), expected $PREV_SHA"
"${G[@]}" fetch -q "$BUNDLE" "+refs/heads/$BRANCH:refs/heads/pr-a-head-2" || die "bundle fetch" "could not fetch $BUNDLE"
[ "$("${G[@]}" rev-parse pr-a-head-2)" = "$EXPECT_SHA" ] || die "new head sha" "bundle has $("${G[@]}" rev-parse pr-a-head-2), expected $EXPECT_SHA"
[ "$("${G[@]}" rev-parse pr-a-head-2~1)" = "$("${G[@]}" rev-parse "$PREV_SHA~1")" ] || die "same base" "the two heads sit on different commits"
"${G[@]}" diff --stat "$PREV_SHA" pr-a-head-2 | tee -a "$RUN/progress.log"
CHANGED=$("${G[@]}" diff --name-only "$PREV_SHA" pr-a-head-2)
if echo "$CHANGED" | grep -q -E '^(python|docs)/'; then die "only tests changed" "files outside test/ changed: $CHANGED"; fi
record "delta to the verified head" "ok (only: $(echo $CHANGED))"
"${G[@]}" checkout -q -B main-upstream pr-a-head-2 || die "checkout" "could not check out the new head"
[ "$("${G[@]}" rev-parse HEAD)" = "$EXPECT_SHA" ] || die "checked-out sha" "HEAD is $("${G[@]}" rev-parse HEAD)"

set +u; source "$CONDA_ROOT/etc/profile.d/conda.sh"; conda activate sgl-profiler; set -u
cd "$SGL/sglang"

timeout 420 python3 "$TEST" -f > "$RUN/unit.log" 2>&1; RC=$?
if [ $RC -eq 0 ]; then record "unit file" "PASS ($(grep -E '^Ran [0-9]+ tests?' "$RUN/unit.log" | tail -1))"; else record "unit file" "FAIL rc=$RC"; fi

python "$HERE/pra_mutate.py" "$SGL/sglang" > "$RUN/mutation.log" 2>&1; RC=$?
tail -2 "$RUN/mutation.log" | sed 's/^/    /' | tee -a "$RUN/progress.log"
if [ $RC -eq 0 ]; then record "mutation check" "PASS ($(grep -c '^red' "$RUN/mutation.log") of $(grep -c ' | ' "$RUN/mutation.log") mutations caught)"; else record "mutation check" "FAIL (see mutation.log)"; fi

UVT=("$ENV_PREFIX/bin/uv" tool run)
SKIP=no-commit-to-branch timeout 420 "${UVT[@]}" pre-commit run --files "${PR_FILES[@]}" > "$RUN/pre_commit.log" 2>&1; RC=$?
if [ $RC -eq 0 ]; then record "pre-commit (4 files)" "PASS ($(grep -c 'Passed$' "$RUN/pre_commit.log") hooks passed)"; else record "pre-commit (4 files)" "FAIL rc=$RC (see pre_commit.log)"; fi

[ -z "$("${G[@]}" status --porcelain)" ] && record "clean tree" "ok" || record "clean tree" "FAIL ($("${G[@]}" status --porcelain | head -3))"
"${G[@]}" rev-parse HEAD > "$RUN/final_head_sha.txt"
finish
