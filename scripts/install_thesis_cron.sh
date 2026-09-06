#!/usr/bin/env bash
# Idempotent installer for the thesis-asset refresh cron job.
# Re-collects thesis/ figures + tables twice daily (09:00 + 21:00) so the
# bundle stays current as new figures/results are generated.
#
# Usage:
#   scripts/install_thesis_cron.sh            # install / refresh
#   scripts/install_thesis_cron.sh --remove   # uninstall
#   scripts/install_thesis_cron.sh --status   # show installed lines
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
TAG="# THESIS-ASSET-REFRESH (agenticfinder)"
LOG="$ROOT/jobs/logs/thesis_cron.log"
CMD="cd $ROOT && $PY scripts/collect_thesis_assets.py >> $LOG 2>&1"
LINES="0 9,21 * * * $CMD $TAG"

mkdir -p "$ROOT/jobs/logs"

case "${1:-install}" in
  --status)
    crontab -l 2>/dev/null | grep -F "$TAG" || echo "(not installed)"
    ;;
  --remove)
    crontab -l 2>/dev/null | grep -vF "$TAG" | crontab - || true
    echo "Removed $TAG"
    ;;
  *)
    # Drop any prior copy of our tagged line, then add the fresh one.
    ( crontab -l 2>/dev/null | grep -vF "$TAG"; echo "$LINES" ) | crontab -
    echo "Installed thesis-asset refresh cron (09:00 + 21:00 daily):"
    echo "  $LINES"
    echo "Log: $LOG"
    ;;
esac
