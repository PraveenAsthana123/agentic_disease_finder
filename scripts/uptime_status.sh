#!/usr/bin/env bash
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
echo "════════ 🖥️ SERVER UPTIME — $(TZ=America/Edmonton date '+%Y-%m-%d %H:%M %Z') ════════"
now=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/api/data-manager -m 8 2>/dev/null)
echo "NOW: $([ "$now" = 200 ] && echo '✅ UP' || echo '❌ DOWN') (http=$now)"
echo "watchdog cron: $(crontab -l 2>/dev/null | grep -c '# WATCHDOG') · checks every 2 min, auto-restarts"
echo ""
echo "Recent up/down events (was it ever down?):"
tail -8 jobs/logs/uptime.jsonl 2>/dev/null | python3 -c "import sys,json;[print('  ',json.loads(l)['ts'],json.loads(l)['event']) for l in sys.stdin if l.strip()]" 2>/dev/null || echo "  (no transitions logged yet — server stable)"
