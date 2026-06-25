#!/usr/bin/env bash
# View recent timestamped events (crash recovery: "what happened, when").
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
N="${1:-25}"
echo "════════ EVENT LOG (last $N · America/Edmonton) ════════"
tail -n "$N" jobs/logs/track.md 2>/dev/null || echo "(no events yet)"
