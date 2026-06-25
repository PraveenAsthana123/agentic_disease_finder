#!/usr/bin/env bash
# Robust backend launcher — fully detached, survives parent shell exit.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 1
fuser -k 8010/tcp 2>/dev/null
pkill -9 -f "api_backend.py" 2>/dev/null
sleep 2
setsid nohup python3 api_backend.py >> jobs/logs/backend.log 2>&1 < /dev/null &
disown
echo "launched backend pid $!"
