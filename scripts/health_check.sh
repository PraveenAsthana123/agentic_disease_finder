#!/usr/bin/env bash
# Terminal error + health summary: frontend, backend, API, database. Counts errors per surface.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
NOW=$(TZ=America/Edmonton date '+%Y-%m-%d %H:%M %Z')
echo "════════ 🩺 SYSTEM HEALTH / ERRORS — $NOW ════════"

# Backend
be=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/api/data-manager -m 8 2>/dev/null)
# only errors since the LAST backend start (current run = true live health)
since=$(grep -n "Started server process" jobs/logs/backend.log 2>/dev/null | tail -1 | cut -d: -f1); since=${since:-1}
be500=$(tail -n +$since jobs/logs/backend.log 2>/dev/null | grep -c "500 Internal Server Error")
betb=$(tail -n +$since jobs/logs/backend.log 2>/dev/null | grep -c "Traceback (most recent call last)")
echo "BACKEND : http=$be · 500s=$be500 · tracebacks=$betb $([ "$be" = 200 ]&&echo ✅||echo ❌)"

# API — probe key endpoints, count non-200
apifail=0; apitot=0
for e in data-manager requests automation-status clinical-trust drift eeg-viz/presets model-performance expert-roles seizure-timeline patient-compare?a=EPAT001\&b=EPAT002 fairness conversation ot ot/fall-risk psychologist psychologist/depression-anxiety coordinator coordinator/kpi data-manager/archival data-manager/terminology data-manager/standardization data-manager/dataset-version data-manager/label-validation data-manager/video-validation data-manager/mri-validation eeg-viz/recordings eeg-viz/traces eeg-viz/bad-channels eeg-viz/artifacts eeg-viz/ictal-interictal; do
  apitot=$((apitot+1))
  c=$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:8010/api/$e" -m 10 2>/dev/null)
  [ "$c" = 200 ] || { apifail=$((apifail+1)); echo "  ✗ /api/${e%%\?*} → $c"; }
done
echo "API     : $((apitot-apifail))/$apitot OK · errors=$apifail $([ $apifail = 0 ]&&echo ✅||echo ❌)"

# Database
dberr=$(python3 -c "import sqlite3;print(sqlite3.connect('data/clinical.db').execute('PRAGMA integrity_check').fetchone()[0])" 2>/dev/null)
echo "DATABASE: integrity=$dberr $([ "$dberr" = ok ]&&echo ✅||echo ❌)"

# Frontend (last build) + dev server
fe=$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3003/ -m 5 2>/dev/null)
echo "FRONTEND: dev-server :3003 http=$fe $([ "$fe" = 200 ]&&echo ✅||echo '⚠ (start: npm run dev)')"

TOTAL=$((apifail + be500 + betb)); [ "$dberr" = ok ] || TOTAL=$((TOTAL+1))
echo "────────────────────────────────"
echo "TOTAL ERRORS: $TOTAL $([ $TOTAL = 0 ]&&echo '✅ all healthy'||echo '❌ needs attention')"
# log to track + a json snapshot
mkdir -p jobs/reports
python3 -c "import json;json.dump({'now':'$NOW','backend_http':'$be','api_errors':$apifail,'api_total':$apitot,'backend_500s':$be500,'tracebacks':$betb,'db':'$dberr','frontend_http':'$fe','total_errors':$TOTAL},open('jobs/reports/health_latest.json','w'),indent=1)" 2>/dev/null
bash scripts/track.sh "healthcheck: $TOTAL errors (api_fail=$apifail be=$be db=$dberr)" "health" >/dev/null 2>&1
