#!/usr/bin/env bash
# Idempotent installer for the VECTOR-INGEST cron job (§87).
# Embeds clinical records → ChromaDB twice daily (07:00 + 19:00).
#
#   scripts/install_vector_cron.sh            # install / refresh
#   scripts/install_vector_cron.sh --remove   # uninstall
#   scripts/install_vector_cron.sh --status   # show installed line
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
TAG="# VECTOR-INGEST (agenticfinder)"
LOG="$ROOT/jobs/logs/vector_ingest.log"
LINE="0 7,19 * * * cd $ROOT && $PY scripts/vector_ingest.py >> $LOG 2>&1 $TAG"

mkdir -p "$ROOT/jobs/logs"

case "${1:-install}" in
  --status) crontab -l 2>/dev/null | grep -F "$TAG" || echo "(not installed)";;
  --remove) crontab -l 2>/dev/null | grep -vF "$TAG" | crontab - || true; echo "Removed $TAG";;
  *) ( crontab -l 2>/dev/null | grep -vF "$TAG"; echo "$LINE" ) | crontab -
     echo "Installed VECTOR-INGEST cron (07:00 + 19:00 daily):"; echo "  $LINE"; echo "Log: $LOG";;
esac
