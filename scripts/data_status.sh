#!/usr/bin/env bash
# Record-count evidence: data downloaded? jobs run? vector/graph DB updated? HOW MANY records?
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
echo "════════ 📦 DATA / RECORD COUNTS — $(TZ=America/Edmonton date '+%Y-%m-%d %H:%M %Z') ════════"
echo "── Raw data downloaded ──"
echo "  real EEG datasets : $(ls -d data/real_eeg/*/ 2>/dev/null | wc -l) folders"
echo "  EEG .edf files    : $(find data/real_eeg -name '*.edf' 2>/dev/null | wc -l)"
echo "  total real_eeg    : $(du -sh data/real_eeg 2>/dev/null | cut -f1)"
echo "── clinical.db records ──"
python3 - <<'PY'
import sqlite3
c=sqlite3.connect('data/clinical.db')
for t in ['patients','analyses','assessments','seizure_diary','clinical_decisions','operator_requests','conversation_log','medications','mri_findings']:
    try: print(f"  {t:20s} {c.execute(f'SELECT COUNT(*) FROM {t}').fetchone()[0]}")
    except: pass
PY
echo "── Vector DB (ChromaDB) ──"
python3 -c "import chromadb;print('  clinical collection:',chromadb.PersistentClient(path='data/vector_db').get_or_create_collection('clinical').count(),'embeddings')" 2>/dev/null || echo "  (chromadb unavailable)"
echo "── Graph DB ──"
[ -f jobs/reports/graph_latest.json ] && python3 -c "import json;d=json.load(open('jobs/reports/graph_latest.json'));print('  triples/nodes:',d.get('triples',d.get('nodes','?')),'· built',d.get('run_at_local','?')[:16])" 2>/dev/null || echo "  (no graph report yet)"
echo "── Last job runs (report mtime) ──"
for r in training_latest vector_latest graph_latest cv_pipeline_latest drift_latest fairness_latest data_quality_latest; do
  f="jobs/reports/$r.json"; [ -f "$f" ] && echo "  $(printf '%-22s' $r) $(date -r "$f" '+%Y-%m-%d %H:%M' 2>/dev/null)"
done
