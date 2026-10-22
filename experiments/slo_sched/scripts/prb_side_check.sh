#!/usr/bin/env bash
# Node side, CPU only: check a PR-ready commit in its own worktree while the GPU work runs.
# Unit files of the change, then the full pre-commit suite on the files the branch touches.
#
#   bash prb_side_check.sh <name> <sha> <first commit of the branch> <out dir> <unit file>...
set -o pipefail
NAME="$1"; SHA="$2"; FIRST="$3"; OUTDIR="$4"; shift 4
OUT="$OUTDIR/side_check_$NAME.txt"
TREE="$HOME/sgl/sglang-$NAME"
# No "set -u": conda's activation scripts read unset variables.
source "$HOME/miniforge3/etc/profile.d/conda.sh"; conda activate sgl-profiler
: > "$OUT"
cd "$HOME/sgl/sglang" || exit 1
git fetch -q origin 2>>"$OUT"
[ -d "$TREE" ] || git worktree add -q --detach "$TREE" "$SHA" 2>>"$OUT"
cd "$TREE" || { echo "no worktree" >> "$OUT"; exit 1; }
echo "worktree at $(git log -1 --format='%h %s'); base $(git log -1 --format='%h %cs %s' "$FIRST~1" | cut -c1-70)" >> "$OUT"
# An ordinal that does not exist hides the GPUs the benchmark servers are using.
export CUDA_VISIBLE_DEVICES=99 PYTHONPATH="$TREE/python"
python -c "import sglang; print('imports', sglang.__file__)" >> "$OUT" 2>&1
for f in "$@"; do
  if python -m pytest -q -p no:cacheprovider "$f" > "$OUTDIR/side_one_$NAME.log" 2>&1; then
    echo "PASS $f :: $(grep -E 'passed|failed' "$OUTDIR/side_one_$NAME.log" | tail -1)" >> "$OUT"
  else
    echo "FAIL $f :: $(grep -E 'passed|failed|rror' "$OUTDIR/side_one_$NAME.log" | tail -1)" >> "$OUT"
    tail -40 "$OUTDIR/side_one_$NAME.log" >> "$OUTDIR/side_failures_$NAME.log"
  fi
done
FILES=$(git diff --name-only "$FIRST~1" HEAD)
SKIP=no-commit-to-branch timeout 420 "$HOME/miniforge3/envs/sgl-profiler/bin/uv" tool run pre-commit run --files $FILES > "$OUTDIR/side_pre_commit_$NAME.log" 2>&1
echo "pre-commit on $(echo $FILES | wc -w) files: rc=$?" >> "$OUT"
grep -E "Failed" "$OUTDIR/side_pre_commit_$NAME.log" | sed 's/^/    /' >> "$OUT"
echo "    hooks passed: $(grep -c 'Passed' "$OUTDIR/side_pre_commit_$NAME.log"), failed: $(grep -c 'Failed' "$OUTDIR/side_pre_commit_$NAME.log")" >> "$OUT"
echo DONE >> "$OUT"
