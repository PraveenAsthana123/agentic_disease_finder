#!/usr/bin/env bash
# Idempotent installer for the VALIDATION-SUITE cron (weekly Sunday 06:00).
# Re-runs all benchmarks + refreshes VALIDATION_SUMMARY.md.
#   scripts/install_validation_cron.sh [--remove|--status]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
TAG="# VALIDATION-SUITE (agenticfinder)"
LOG="$ROOT/jobs/logs/validation_suite.log"
LINE="0 6 * * 0 cd $ROOT && $PY scripts/run_validation_suite.py >> $LOG 2>&1 $TAG"
mkdir -p "$ROOT/jobs/logs"
case "${1:-install}" in
  --status) crontab -l 2>/dev/null | grep -F "$TAG" || echo "(not installed)";;
  --remove) crontab -l 2>/dev/null | grep -vF "$TAG" | crontab - || true; echo "Removed $TAG";;
  *) ( crontab -l 2>/dev/null | grep -vF "$TAG"; echo "$LINE" ) | crontab -
     echo "Installed VALIDATION-SUITE cron (Sun 06:00):"; echo "  $LINE";;
esac
