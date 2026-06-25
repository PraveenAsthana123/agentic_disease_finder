#!/usr/bin/env python3
"""Writes jobs/reports/STATUS_NOW.md — a one-glance status refreshed every 5 min (STATUS-NOW cron).
Open it in the editor (auto-updates) or `tail -f`. Answers 'what's the status' without asking."""
import subprocess, sqlite3, json
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
def sh(c): return subprocess.run(c,shell=True,cwd=ROOT,capture_output=True,text=True,timeout=20).stdout.strip()
now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
be = sh("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8010/api/data-manager -m 8")
fe = sh("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3003/ -m 5")
try:
    c=sqlite3.connect(str(ROOT/"data/clinical.db"))
    unaddr=c.execute("SELECT COUNT(*) FROM operator_requests WHERE status IN ('open','pending')").fetchone()[0]
    total=c.execute("SELECT COUNT(*) FROM operator_requests").fetchone()[0]
except Exception: unaddr=total="?"
adv=json.loads((ROOT/"jobs/reports/advisor_latest.json").read_text()).get("issues",[]) if (ROOT/"jobs/reports/advisor_latest.json").exists() else []
p1=sum(1 for i in adv if i["severity"] in ("P0","P1"))
need = "🔴 INPUT NEEDED" if (p1>0 or (isinstance(unaddr,int) and unaddr>0)) else "🟢 all handled"
md=f"""# ⚡ STATUS NOW — {now}
- Backend: {'✅ UP' if be=='200' else '❌ DOWN ('+be+')'}  ·  Frontend: {'✅' if fe=='200' else '⚠ '+fe}
- Inputs: {total} total · {unaddr} unaddressed
- Advisor P0/P1 issues: {p1}
- **Do you need to input?** {need}
- Refreshed every 5 min (STATUS-NOW cron). Run anytime: bash scripts/status.sh
"""
(ROOT/"jobs/reports/STATUS_NOW.md").write_text(md)
print(md)
