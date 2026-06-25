#!/usr/bin/env bash
# Refresh STATUS.md then print the Pending section. Used by SessionStart hook + manually.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
python3 scripts/status_report.py >/dev/null 2>&1
echo "════════ ⏳ PENDING TASKS (auto) ════════"
sed -n '/## ⏳ Pending/,$p' STATUS.md 2>/dev/null | head -22
