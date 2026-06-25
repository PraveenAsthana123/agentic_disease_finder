#!/usr/bin/env bash
# Per-job watch — see each cron job's schedule, last run, enabled status. Stop/start any job.
# Usage: jobs_watch.sh                → list all  |  jobs_watch.sh stop TAG  |  jobs_watch.sh start TAG
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
TAGS="AGENTICFINDER-TRAIN VECTOR-INGEST VIDEO-FRAMES GRAPH-DB VALIDATION CV-PIPELINE FAIRNESS DRIFT DATA-QUALITY STATUS-REPORT CONSISTENCY AUTO-PLAN AUTO-BUILD HEALTH WATCHDOG"
act="$1"; tag="$2"
if [ "$act" = stop ] && [ -n "$tag" ]; then
  crontab -l 2>/dev/null | sed "s|^\(.*# $tag\)$|#DISABLED \1|" | crontab -; echo "⏸️ stopped $tag"; exit 0; fi
if [ "$act" = start ] && [ -n "$tag" ]; then
  crontab -l 2>/dev/null | sed "s|^#DISABLED \(.*# $tag\)$|\1|" | crontab -; echo "▶️ started $tag"; exit 0; fi
echo "════════ ⏱️ JOB WATCH — $(TZ=America/Edmonton date '+%Y-%m-%d %H:%M %Z') ════════"
printf "%-22s %-13s %-8s %-22s\n" "JOB" "SCHEDULE" "STATE" "LAST RUN"
cron=$(crontab -l 2>/dev/null)
for t in $TAGS; do
  line=$(echo "$cron" | grep -E "# $t$")
  if echo "$line" | grep -q "^#DISABLED"; then state="⏸️ stopped"; else [ -n "$line" ] && state="▶️ running" || state="—absent"; fi
  sched=$(echo "$line" | sed 's/#DISABLED //' | awk '{print $1" "$2}')
  # last run = mtime of the job's report or last log mention
  rpt=$(python3 -c "import json;d=json.load(open('config/jobs.json'));print(next((j['report'] for j in d['jobs'] if j['cron_tag']=='$t'),''))" 2>/dev/null)
  last=$([ -n "$rpt" ] && [ -f "$rpt" ] && date -r "$rpt" '+%Y-%m-%d %H:%M' 2>/dev/null || echo "")
  printf "%-22s %-13s %-8s %-22s\n" "$t" "${sched:-?}" "$state" "${last:-never logged}"
done
