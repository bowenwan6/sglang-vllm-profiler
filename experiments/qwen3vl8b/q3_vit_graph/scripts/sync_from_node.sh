#!/usr/bin/env bash
# Mac side. Pull Q3 results off the node every INTERVAL seconds and commit the
# summaries, so an assignment expiry (home wiped) or a power cut costs at most one cell.
#
#   NODE=bowenwan6@<ip> bash sync_from_node.sh            # loop every 120 s
#   NODE=bowenwan6@<ip> bash sync_from_node.sh 0          # one pass
#   Q3_SYNC_TRACES=1 ...                                  # also pull the raw profiler traces
#
# Only results/*.json|*.md are staged; results/raw/ (per-request jsonl, server logs,
# traces) is git-ignored and kept on the Mac as a backup.
set -euo pipefail
: "${NODE:?set NODE=user@ip  (radix machines mine shows the ssh target)}"
KEY="${RADIX_SSH_KEY:-$HOME/.ssh/id_ed25519}"
INTERVAL="${1:-120}"
REPO="$(cd "$(dirname "$0")/../../../.." && pwd)"
EXP_REL=experiments/qwen3vl8b/q3_vit_graph
SSH_OPTS=(-i "$KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15)
RSH="ssh ${SSH_OPTS[*]}"

while true; do
  ts=$(date -u +%FT%TZ)
  rsync -az --timeout=60 -e "$RSH" "$NODE:sgl/profiler/$EXP_REL/results/" "$REPO/$EXP_REL/results/" 2>/dev/null \
    || echo "[$ts] rsync results failed (node unreachable?)"
  rsync -az --timeout=60 -e "$RSH" "$NODE:sgl/logs/q3/" "$REPO/$EXP_REL/results/raw/logs/" 2>/dev/null || true
  if [ "${Q3_SYNC_TRACES:-0}" = "1" ]; then
    rsync -az --timeout=300 -e "$RSH" "$NODE:sgl/traces/q3/" "$REPO/$EXP_REL/results/raw/traces/" 2>/dev/null || true
  fi
  (
    cd "$REPO"
    git add "$EXP_REL/results" >/dev/null 2>&1 || true      # raw/ is ignored; summaries only
    if ! git diff --cached --quiet; then
      git commit -q -m "chore(q3): sync node results $ts" && git push -q origin HEAD \
        && echo "[$ts] committed + pushed" || echo "[$ts] commit ok, push failed (retry next pass)"
    else
      echo "[$ts] no new summaries"
    fi
  )
  [ "$INTERVAL" = "0" ] && break
  sleep "$INTERVAL"
done
