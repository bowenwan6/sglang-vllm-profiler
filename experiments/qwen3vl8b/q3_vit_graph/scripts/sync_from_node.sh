#!/usr/bin/env bash
# Mac side. Pull Q3 results off the node every INTERVAL seconds and commit the tracked
# summaries, so an assignment expiry (home wiped) or a power cut costs at most one block.
#
#   NODE=bowenwan6@<ip> bash sync_from_node.sh            # loop every 120 s
#   NODE=bowenwan6@<ip> bash sync_from_node.sh 0          # one pass
#   Q3_SYNC_TRACES=0 ...                                  # skip the profiler traces (default: pull them, incrementally)
#
# What is pulled: results/ (cells/, *.json, *.md and raw/), the node's setup + server logs,
# env_manifest.txt and the profiler traces. What is committed: only the tracked part of
# results/ (cells/, *.json, *.md); results/raw/ and STATUS.json are git-ignored.
# The commit is path-scoped and pushed to exp/q3-vit-graph explicitly, whatever HEAD is;
# on any other branch the script only rsyncs.
set -euo pipefail
: "${NODE:?set NODE=user@ip  (radix machines mine shows the ssh target)}"
KEY="${RADIX_SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL="${1:-120}"
BRANCH=exp/q3-vit-graph
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXP_REL=experiments/qwen3vl8b/q3_vit_graph
DEST="$REPO/$EXP_REL/results"
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15"

pull() {  # $1 = remote path (relative to the node home), $2 = local dir, $3 = timeout
  mkdir -p "$2"
  rsync -az --timeout="${3:-60}" -e "$RSH" "$NODE:$1" "$2" 2>/dev/null \
    || echo "[$ts] rsync $1 failed (node unreachable or path absent)"
}

while true; do
  ts=$(date -u +%FT%TZ)
  pull "sgl/profiler/$EXP_REL/results/" "$DEST/" 120
  pull "sgl/logs/q3/" "$DEST/raw/logs/q3/" 60
  pull "sgl/logs/setup_*.log" "$DEST/raw/logs/" 30
  pull "sgl/env_manifest.txt" "$DEST/raw/" 30
  if [ "${Q3_SYNC_TRACES:-1}" = "1" ]; then
    pull "sgl/traces/q3/" "$DEST/raw/traces/" 600
  fi
  (
    cd "$REPO"
    br=$(git symbolic-ref --short -q HEAD || echo DETACHED)
    if [ "$br" != "$BRANCH" ]; then echo "[$ts] HEAD is $br, not $BRANCH: rsync only, no commit"; exit 0; fi
    git add -- "$EXP_REL/results" >/dev/null 2>&1 || true
    git reset -q -- "$EXP_REL/results/raw" "$EXP_REL/results/STATUS.json" 2>/dev/null || true
    if ! git diff --cached --quiet -- "$EXP_REL/results"; then
      if git commit -q -m "chore(q3): sync node results $ts" -- "$EXP_REL/results"; then
        if git push -q origin "HEAD:refs/heads/$BRANCH"; then echo "[$ts] committed + pushed"; else echo "[$ts] committed, push failed (retry next pass)"; fi
      else
        echo "[$ts] commit failed"
      fi
    else
      echo "[$ts] no new tracked results"
    fi
  )
  [ "$INTERVAL" = "0" ] && break
  sleep "$INTERVAL"
done
