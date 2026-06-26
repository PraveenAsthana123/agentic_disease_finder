#!/usr/bin/env bash
# Server watchdog — checks backend health; auto-restarts if down; logs every up/down transition.
# "Server should not go down" → self-healing. Runs via cron (survives crash/reboot, editor-independent).
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
mkdir -p jobs/logs
STATE=jobs/logs/.server_state
NOW=$(TZ=America/Edmonton date '+%Y-%m-%d %H:%M:%S %Z')
prev=$(cat "$STATE" 2>/dev/null || echo "unknown")
code=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/api/data-manager -m 8 2>/dev/null)

if [ "$code" = "200" ]; then
  cur=up
  if [ "$prev" != "up" ]; then
    echo "{\"ts\":\"$NOW\",\"event\":\"RECOVERED\",\"prev\":\"$prev\"}" >> jobs/logs/uptime.jsonl
    bash scripts/track.sh "server RECOVERED (was $prev)" "watchdog"
    bash scripts/slack_notify.sh --level warn "server RECOVERED (was $prev)" >/dev/null 2>&1 &
  fi
else
  cur=down
  echo "{\"ts\":\"$NOW\",\"event\":\"DOWN\",\"http\":\"$code\"}" >> jobs/logs/uptime.jsonl
  bash scripts/track.sh "server DOWN (http=$code) — auto-restarting" "watchdog"
  bash scripts/slack_notify.sh --level error "server DOWN (http=$code) — auto-restarting" >/dev/null 2>&1 &
  bash scripts/restart_backend.sh >> jobs/logs/backend.log 2>&1
  code2=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/api/data-manager -m 8 2>/dev/null)
  [ "$code2" = "200" ] && { cur=up; echo "{\"ts\":\"$NOW\",\"event\":\"AUTO-RESTARTED\"}" >> jobs/logs/uptime.jsonl; bash scripts/track.sh "server auto-restarted OK" "watchdog"; }
fi
echo "$cur" > "$STATE"
echo "[$NOW] server=$cur (http=$code)"
