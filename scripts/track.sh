#!/usr/bin/env bash
# Timestamped event tracker — crash-recoverable dual-write (JSONL + Markdown), append-only.
# Usage: scripts/track.sh "event message" [level]
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
msg="${1:-}"; level="${2:-info}"
mkdir -p jobs/logs
utc=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
loc=$(TZ=America/Edmonton date +"%Y-%m-%d %H:%M:%S %Z")
host=$(hostname); usr=$(whoami)
# append-only: a crash mid-write loses at most this one line, never prior history
printf '{"ts_utc":"%s","ts_local":"%s","host":"%s","user":"%s","level":"%s","event":%s}\n' \
  "$utc" "$loc" "$host" "$usr" "$level" "$(printf '%s' "$msg" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" \
  >> jobs/logs/track.jsonl
printf '[%s] (%s@%s) %s — %s\n' "$loc" "$usr" "$host" "$level" "$msg" >> jobs/logs/track.md
echo "[$loc] $msg"
