#!/usr/bin/env bash
# Mac side. Pull a PR-B session's output off the node into results/raw/<run>/ (git-ignored).
#
#   RUN_NAME=p2 NODE=<user>@<ip> KNOWN_HOSTS=~/.radix/known_hosts.d/<ip> bash prb_sync.sh
set -euo pipefail
: "${RUN_NAME:?set RUN_NAME, for example p2}"
: "${NODE:?set NODE=user@ip}"
: "${KNOWN_HOSTS:?set KNOWN_HOSTS to the host key file of the node}"
KEY="${RADIX_SSH_KEY:-$HOME/.ssh/id_ed25519}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/results/raw/$RUN_NAME"
mkdir -p "$DEST"
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes -o ConnectTimeout=15"
rsync -az --timeout=120 -e "$RSH" "$NODE:sgl/logs/$RUN_NAME/" "$DEST/"
rsync -az --timeout=60 -e "$RSH" "$NODE:sgl/logs/setup_*.log" "$DEST/" 2>/dev/null || true
count=$(find "$DEST" -type f | wc -l)
size=$(du -sh "$DEST" | cut -f1)
echo "synced to $DEST: $((count)) files, $size"
