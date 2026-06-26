#!/usr/bin/env bash
# Deterministic backend restart — kill ALL, wait port FREE, launch, wait HEALTH 200.
# Exit 0 only when serving. Prevents zombie pile-up + commit-before-verify (loop gates on this).
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 1
PORT=8010
# Serialize restarts: the auto_build_loop AND the watchdog can both call this. Without a
# lock, two restarts race and kill each other's booting process → extended downtime / flapping.
# flock makes a concurrent caller wait for the in-progress restart instead of stomping it.
mkdir -p jobs/logs
exec 9>jobs/logs/.restart.lock
if ! flock -w 150 9; then echo "restart already in progress (lock busy)"; exit 0; fi
# Got the lock — if a concurrent restart already brought the backend up, don't stomp it.
[ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/api/data-manager -m 5 2>/dev/null)" = "200" ] && {
  echo "backend already healthy (concurrent restart won) — skipping"; exit 0; }
# Canonical project venv (one-venv-per-project policy §61.11). Single source of truth.
VENV_PY=/home/praveen/venv-ardupilot/bin/python3
[ -x "$VENV_PY" ] || VENV_PY=python3   # fallback only if venv unreachable (§60)
# 1. kill every instance
pkill -9 -f "api_backend.py" 2>/dev/null
fuser -k ${PORT}/tcp 2>/dev/null
# 2. wait until port is actually free (max 20s)
for i in $(seq 1 20); do
  ss -ltn 2>/dev/null | grep -q ":${PORT} " || break
  sleep 1
done
# 3. launch detached under the canonical venv
setsid nohup "$VENV_PY" api_backend.py >> jobs/logs/backend.log 2>&1 < /dev/null &
disown 2>/dev/null
# 4. wait for health 200 (max ~120s; heavy ML imports)
for i in $(seq 1 90); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:${PORT}/api/data-manager -m 5 2>/dev/null)" = "200" ] && {
    bash scripts/track.sh "backend restarted + healthy" "ops"; echo "backend UP (~$((i*2))s, $(pgrep -cf 'python3 api_backend.py') proc)"; exit 0; }
  sleep 2
done
echo "backend FAILED to come up in 120s"; exit 1
