#!/usr/bin/env bash
# THE single status command — run this anytime to see EVERYTHING in one place.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  PROJECT STATUS — $(TZ=America/Edmonton date '+%Y-%m-%d %H:%M %Z')              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
bash scripts/uptime_status.sh 2>/dev/null | head -3
echo ""; bash scripts/health_check.sh 2>/dev/null | grep -E "BACKEND|API|DATABASE|FRONTEND|TOTAL"
echo ""; bash scripts/show_requests.sh 2>/dev/null | head -4
echo ""; echo "── PENDING ──"; python3 scripts/status_report.py >/dev/null 2>&1; grep -A0 "TOTAL PENDING" jobs/reports/auto_plan.md 2>/dev/null || echo "  (run show_pending.sh for full list)"
echo ""; echo "── ADVISOR (issues you may not know) ──"; python3 scripts/advisor_agent.py 2>/dev/null | grep -E "^\s+\[P" | head -6
echo ""; echo "── JOBS ──"; bash scripts/jobs_watch.sh 2>/dev/null | grep -cE "running" | sed 's/^/  jobs running: /'
echo ""
echo "✅ Safe to enter a NEW topic anytime — every input is logged (📥 Request Inbox), nothing is lost."
