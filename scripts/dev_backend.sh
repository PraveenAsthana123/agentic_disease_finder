#!/bin/bash
# Dev backend with AUTO-RELOAD — eliminates the §120 stale-backend problem.
# With --reload, editing any .py file auto-restarts the server, so new endpoints
# appear in the UI immediately (no manual `kill + python3 api_backend.py`).
#
# Usage:  ./scripts/dev_backend.sh        (foreground, Ctrl-C to stop)
#         ./scripts/dev_backend.sh bg     (background, logs to jobs/logs/backend.log)
#
# Why: this session repeatedly hit "data gone / still see 4 / backend offline" because
# the prod-style launch (uvicorn.run(app) — no reload) froze code at process start, so
# every new route 404'd until a manual restart. --reload fixes that class of problem.

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PORT="${PORT:-8010}"

# free the port if something is already bound
OLD="$(fuser ${PORT}/tcp 2>/dev/null | tr -d ' ' || true)"
[ -n "$OLD" ] && { echo "stopping existing backend (pid $OLD)"; kill -9 $OLD 2>/dev/null || true; sleep 2; }

mkdir -p jobs/logs
CMD="python3 -m uvicorn api_backend:app --host 0.0.0.0 --port ${PORT} --reload --reload-include '*.py' --reload-include '*.json'"

if [ "$1" = "bg" ]; then
  nohup bash -c "$CMD" >> jobs/logs/backend.log 2>&1 &
  disown
  echo "dev backend (auto-reload) starting in background on :${PORT} — logs: jobs/logs/backend.log"
else
  echo "dev backend (auto-reload) on :${PORT} — edit .py/.json and it reloads automatically. Ctrl-C to stop."
  exec bash -c "$CMD"
fi
