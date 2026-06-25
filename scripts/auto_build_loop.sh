#!/usr/bin/env bash
# TRUE unattended autonomous build — invokes Claude HEADLESS to build ONE pending item.
# Survives session close + system crash (cron-driven; cron auto-starts on reboot).
# Safety: kill-switch file · single-instance lock · safe_push (no force) · timestamped track.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 1

# kill-switch: only run if explicitly enabled (operator: `touch jobs/.autobuild_enabled` to arm, rm to stop)
[ -f jobs/.autobuild_enabled ] || { exit 0; }

# single instance (flock) — never two builds at once
exec 9>jobs/.autobuild.lock
flock -n 9 || { bash scripts/track.sh "auto-build skipped: another instance running" "autobuild"; exit 0; }

bash scripts/track.sh "auto-build START (headless claude)" "autobuild"

PROMPT='Autonomous build (§159), system unattended. Do ONE iteration only:
1. python3 scripts/next_pending.py — pick the TOP buildable pending item. SKIP: ictal/interictal retrain (too heavy), and anything needing operator credentials/decisions (Gmail/Slack/Drive/auth/EMR/FHIR).
2. Build it for real (backend endpoint + frontend panel + nav wiring), honest §57.7 (no stubs).
3. bash scripts/restart_backend.sh — restart+verify; ONLY proceed if it exits 0 (health 200).
4. Verify the new endpoint returns 200.
5. Mark the registry item built, refresh STATUS.md (python3 scripts/status_report.py).
6. Commit (§51 substrate, §54 NO Co-Authored-By trailer).
7. bash scripts/safe_push.sh (auto-push, fast-forward only).
8. bash scripts/track.sh "built+pushed: <item name>" "autobuild"
If you cannot complete + verify, do NOT commit; run scripts/track.sh with the failure reason and exit. NEVER force-push, NEVER first-publish, NEVER fabricate data, NEVER fake done.'

# headless, autonomous, scoped to this project; timeout guard 25 min
timeout 1500 /home/praveen/.local/bin/claude -p "$PROMPT" --permission-mode acceptEdits >> jobs/logs/autobuild.log 2>&1
rc=$?
bash scripts/track.sh "auto-build END (rc=$rc)" "autobuild"
