#!/bin/bash
# Install/remove the scheduled epilepsy training cron job.
# Runs the leakage-free training/eval daily at 02:30; result → jobs/reports/training_latest.json
# Usage: ./scripts/install_train_cron.sh [install|remove|status|run]
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TAG="# AGENTICFINDER-TRAIN"
# Canonical project venv (one-venv-per-project policy §61.11)
VENV_PY=/home/praveen/venv-ardupilot/bin/python3
LINE="30 2 * * * cd $ROOT && $VENV_PY scripts/scheduled_train.py >> jobs/logs/train.log 2>&1 $TAG"
mkdir -p "$ROOT/jobs/logs"
case "${1:-install}" in
  install)
    ( crontab -l 2>/dev/null | grep -v "$TAG"; echo "$LINE" ) | crontab -
    echo "installed: epilepsy training daily 02:30 → jobs/reports/training_latest.json"
    crontab -l | grep "$TAG" ;;
  remove)
    crontab -l 2>/dev/null | grep -v "$TAG" | crontab - || true
    echo "removed training cron" ;;
  status)
    crontab -l 2>/dev/null | grep "$TAG" && echo "(installed)" || echo "(not installed)" ;;
  run)
    cd "$ROOT" && "$VENV_PY" scripts/scheduled_train.py ;;
  *) echo "usage: $0 [install|remove|status|run]"; exit 1 ;;
esac
