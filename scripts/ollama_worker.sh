#!/usr/bin/env bash
# Independent OLLAMA worker lane (local/free/desktop, editor-independent via cron).
# SAFE scope: drafts a planned clinical-scale instrument config to a REVIEW QUEUE
# (jobs/ollama_drafts/) — NOT auto-merged. Clinical content requires human sign-off (§57.7).
# Separate lock from the claude builder → the two lanes never collide.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 1
[ -f jobs/.ollama_worker_enabled ] || exit 0
exec 8>jobs/.ollama_worker.lock
flock -n 8 || { bash scripts/track.sh "ollama-worker skipped: locked" "ollama"; exit 0; }

MODEL="qwen2.5-coder:14b"   # strong local code model for structured JSON
# pick next planned clinical scale not yet drafted
SCALE=$(python3 - <<'PY'
import json, glob, os
done = set(os.path.basename(f).replace('.json','') for f in glob.glob('jobs/ollama_drafts/*.json'))
for c in json.load(open('config/neuro_ai_ecosystem.json')).get('categories', []):
    for t in c.get('tools', []):
        if t['status']=='planned' and t['name'].replace('/','-').replace(' ','_') not in done:
            print(t['name']); raise SystemExit
PY
)
[ -z "$SCALE" ] && { bash scripts/track.sh "ollama-worker: no undrafted scales left" "ollama"; exit 0; }

bash scripts/track.sh "ollama-worker drafting: $SCALE (model $MODEL)" "ollama"
PROMPT="Output ONLY valid JSON (no prose) for a clinical assessment instrument named '$SCALE'. Schema: {\"instrument\":\"$SCALE\",\"items\":[{\"q\":\"question text\",\"max\":N}],\"scale\":\"sum|mean\",\"bands\":[{\"min\":0,\"max\":N,\"level\":\"normal|mild|moderate|severe\"}],\"source\":\"reference\",\"draft\":true,\"needs_clinical_review\":true}. Use the real published items/scoring for this instrument."
safe=$(echo "$SCALE" | tr '/ ' '--')
timeout 300 ollama run "$MODEL" "$PROMPT" 2>/dev/null > "jobs/ollama_drafts/${safe}.raw"
# validate JSON; keep only if parseable
python3 -c "import json,sys,re; t=open('jobs/ollama_drafts/${safe}.raw').read(); m=re.search(r'\{.*\}',t,re.S); d=json.loads(m.group()); d['_drafted_by']='$MODEL'; json.dump(d,open('jobs/ollama_drafts/${safe}.json','w'),indent=1)" 2>/dev/null \
  && bash scripts/track.sh "ollama-worker DRAFTED ${safe}.json (review queue, NOT merged)" "ollama" \
  || bash scripts/track.sh "ollama-worker FAILED to produce valid JSON for $SCALE" "ollama"
rm -f "jobs/ollama_drafts/${safe}.raw"
