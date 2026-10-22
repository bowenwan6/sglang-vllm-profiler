#!/usr/bin/env bash
# Mac side. Pull session P1's output off the node into results/raw/p1/ (git-ignored).
#
#   NODE=<user>@<ip> KNOWN_HOSTS=~/.radix/known_hosts.d/<ip> bash p1_sync.sh
set -euo pipefail
: "${NODE:?set NODE=user@ip}"
: "${KNOWN_HOSTS:?set KNOWN_HOSTS to the file holding the host keys of the node}"
KEY="${RADIX_SSH_KEY:-$HOME/.ssh/id_ed25519}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/results/raw/p1"
mkdir -p "$DEST"
RSH="ssh -i $KEY -o IdentitiesOnly=yes -o UserKnownHostsFile=$KNOWN_HOSTS -o StrictHostKeyChecking=yes -o ConnectTimeout=15"
rsync -az --timeout=120 -e "$RSH" "$NODE:sgl/logs/p1/" "$DEST/"
rsync -az --timeout=60 -e "$RSH" "$NODE:sgl/logs/setup_*.log" "$DEST/" 2>/dev/null || true
count=$(find "$DEST" -type f | wc -l)
size=$(du -sh "$DEST" | cut -f1)
echo "synced to $DEST: $((count)) files, $size"
