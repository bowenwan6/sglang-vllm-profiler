#!/usr/bin/env bash
# Mac side. Pull the PR-A node check's output into results/raw/pra/ (git-ignored).
#
#   NODE=<user>@<host> KNOWN_HOSTS=~/.radix/known_hosts.d/<ip> bash pra_sync.sh
#   ... WATCH_UNTIL=<unix time> bash pra_sync.sh    # repeat every 30 s until the node
#                                                   # script has finished, or that time
set -uo pipefail
: "${NODE:?set NODE=user@host}"
: "${KNOWN_HOSTS:?set KNOWN_HOSTS to the file holding the host keys of the node}"
KEY="${RADIX_SSH_KEY:-$HOME/.ssh/id_ed25519}"
PORT="${NODE_PORT:-22}"
WATCH_UNTIL="${WATCH_UNTIL:-0}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/results/raw/pra"
mkdir -p "$DEST"
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes -o ConnectTimeout=15 -p $PORT"

pull() { rsync -az --timeout=120 -e "$RSH" "${NODE}:sgl/logs/pra/" "$DEST/"; }

rounds=0
errors=0
while :; do
  pull 2> "$DEST/.sync_err" || errors=$((errors + 1))
  rounds=$((rounds + 1))
  [ -e "$DEST/.finished" ] && break
  [ "$(date +%s)" -ge "$WATCH_UNTIL" ] && break
  sleep 30
done
pull 2> "$DEST/.sync_err" || { errors=$((errors + 1)); cat "$DEST/.sync_err"; }
count=$(find "$DEST" -type f | wc -l)
echo "synced to $DEST: $((count)) files, $(du -sh "$DEST" | cut -f1) | rounds $rounds, failed pulls $errors | node script finished: $([ -e "$DEST/.finished" ] && echo yes || echo NO)"
