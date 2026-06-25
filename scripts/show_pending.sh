#!/usr/bin/env bash
# §158: emit the FULL enumerated pending list (every item, numbered, counted). No truncation.
cd /media/praveen/Asthana4/rajveer/agenticfinder || exit 0
python3 scripts/status_report.py >/dev/null 2>&1
python3 - <<'PYEOF'
import json
def load(p):
    try: return json.load(open(p))
    except: return {}
n=0
def block(title, items):
    global n
    if not items: return
    print(f"\n## {title} ({len(items)})")
    for it in items:
        n+=1; print(f"  {n:3d}. {it}")
print("════════ ⏳ ALL PENDING TASKS (§158 full enumeration) ════════")
block("Clinical Data Manager", [f"{t['name']} [{t['status']}]" for t in load('config/data_manager.json').get('tasks',[]) if t['status']!='built'])
block("Expert Dashboards", [f"{t['name']} [{t['status']}] — {t.get('role','')}" for t in load('config/expert_dashboards.json').get('dashboards',[]) if t['status']!='built'])
er=[]
for r in load('config/expert_roles.json').get('roles',[]):
    for t in r['tasks']:
        if t['status']!='built': er.append(f"{r['role'].split('(')[0].strip()}: {t['name']} [{t['status']}]")
block("Expert Roles (8 roles)", er)
block("Neuro AI Ecosystem (scales + cognitive tests)", [t['name'] for c in load('config/neuro_ai_ecosystem.json').get('categories',[]) for t in c['tools'] if t['status']=='planned'])
block("EEG AI Stack libs", [t['name'] for L in load('config/eeg_ai_stack.json').get('layers',[]) for t in L['tools'] if t['status']=='cataloged'])
print(f"\n{'='*40}\nTOTAL PENDING: {n}")
print("🔒 BLOCKED (need operator): Gmail/Slack/Drive creds · multi-user auth · EMR/FHIR · device streaming")
print("🔴 GATED: push commits · ictal/interictal retrain (heavy)")
PYEOF
