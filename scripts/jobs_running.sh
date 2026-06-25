#!/usr/bin/env bash
# Per-job STOPWATCH — which jobs are actually executing NOW + elapsed time (ps etime). Mandatory terminal view.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
echo "════════ ⏱️ JOB STOPWATCH — $(TZ=America/Edmonton date '+%Y-%m-%d %H:%M:%S %Z') ════════"
declare -A SC=( [AUTO-BUILD]=auto_build_loop.sh [WATCHDOG]=watchdog.sh [AUTO-PLAN]=auto_plan.py
  [DRIFT]=drift_job.py [FAIRNESS]=fairness_analysis.py [DATA-QUALITY]=data_quality_job.py
  [CV-PIPELINE]=cv_pipeline.py [VECTOR-INGEST]=vector_ingest.py [VIDEO-FRAMES]=video_to_frames.py
  [TRAIN]=scheduled_train.py [CONSISTENCY]=consistency_check.py [STATUS-REPORT]=status_report.py
  [HEALTH]=health_check.sh [GRAPH-DB]=build_graph.py [VALIDATION]=run_validation_suite.py )
SELF=$$
running=0
printf "%-16s %-8s %-12s %s\n" "JOB" "PID" "ELAPSED" "STATE"
for tag in AUTO-BUILD WATCHDOG AUTO-PLAN DRIFT FAIRNESS DATA-QUALITY CV-PIPELINE VECTOR-INGEST VIDEO-FRAMES TRAIN CONSISTENCY STATUS-REPORT HEALTH GRAPH-DB VALIDATION; do
  sc=${SC[$tag]}
  # match ONLY the real invocation: 'python3 .../scripts/X.py' or 'bash scripts/X.sh' — exclude this shell + viewers
  pid=$(ps -eo pid,args | grep -E "(python3|bash|claude).*scripts/$sc" | grep -v "grep" | grep -v "jobs_running\|status.sh\|grep -E" | awk '{print $1}' | grep -v "^$SELF$" | head -1)
  if [ -n "$pid" ]; then
    el=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
    printf "%-16s %-8s %-12s 🟢 running\n" "$tag" "$pid" "${el:-?}"
    running=$((running+1))
  fi
done
[ $running = 0 ] && echo "(no job executing this instant — all idle between cron ticks)"
echo "────────────────────────────────"
echo "RUNNING NOW: $running  ·  SCHEDULED: $(crontab -l 2>/dev/null | grep -cE '# [A-Z-]+$')  ·  (elapsed = stopwatch since start)"
