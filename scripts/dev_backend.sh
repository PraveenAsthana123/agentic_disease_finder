#!/bin/bash
# Stable backend launcher (NO --reload).
#
# Why no --reload: this repo's data/ dir is ~247GB and the app writes to clinical.db +
# data/uploads/ on every upload. uvicorn --reload (WatchFiles) traverses the project to
# watch files; data/DB writes then trigger reloads that KILL in-flight upload requests
# ("upload failed"). So we run the plain, stable server. After adding/editing ENDPOINTS,
# re-run this script to pick them up (code is frozen at process start — §120).
#
# Usage:  ./scripts/dev_backend.sh        (foreground, Ctrl-C to stop)
#         ./scripts/dev_backend.sh bg     (background, logs to jobs/logs/backend.log)

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8010}"

OLD="$(fuser ${PORT}/tcp 2>/dev/null | tr -d ' ' || true)"
[ -n "$OLD" ] && { echo "stopping existing backend (pid $OLD)"; kill -9 $OLD 2>/dev/null || true; sleep 2; }

mkdir -p jobs/logs
if [ "$1" = "bg" ]; then
  nohup python3 api_backend.py >> jobs/logs/backend.log 2>&1 &
  disown
  echo "backend starting in background on :${PORT} (~30-40s for ML imports) — logs: jobs/logs/backend.log"
else
  echo "backend on :${PORT} (~30-40s for ML imports). Ctrl-C to stop."
  exec python3 api_backend.py
fi
