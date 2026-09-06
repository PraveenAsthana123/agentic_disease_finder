#!/usr/bin/env bash
# Idempotent installer for the clinical-DB backup + audit cron job.
# Runs twice daily (08:00 + 20:00): backs up data/clinical.db and writes
# jobs/reports/clinical_audit_latest.{md,json}.
#
#   scripts/install_clinical_cron.sh            # install / refresh
#   scripts/install_clinical_cron.sh --remove   # uninstall
#   scripts/install_clinical_cron.sh --status   # show installed line
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$(command -v python3)"
TAG="# CLINICAL-DB-AUDIT (agenticfinder)"
LOG="$ROOT/jobs/logs/clinical_cron.log"
LINE="0 8,20 * * * cd $ROOT && $PY scripts/clinical_db_audit.py >> $LOG 2>&1 $TAG"

mkdir -p "$ROOT/jobs/logs"

case "${1:-install}" in
  --status) crontab -l 2>/dev/null | grep -F "$TAG" || echo "(not installed)";;
  --remove) crontab -l 2>/dev/null | grep -vF "$TAG" | crontab - || true; echo "Removed $TAG";;
  *) ( crontab -l 2>/dev/null | grep -vF "$TAG"; echo "$LINE" ) | crontab -
     echo "Installed clinical-DB audit cron (08:00 + 20:00 daily):"; echo "  $LINE"; echo "Log: $LOG";;
esac
