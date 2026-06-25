#!/usr/bin/env python3
"""Advisor Agent — proactively scans for issues the operator may not be aware of, with guidance.
Files findings to advisor_issues table + jobs/reports/advisor_latest.{json,md}. Runs via cron.
Each finding: severity (P0/P1/P2/P3) · surface · issue · guidance. Honest §57.7."""
import json, subprocess, sqlite3
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent


def sh(c):
    return subprocess.run(c, shell=True, cwd=ROOT, capture_output=True, text=True, timeout=30).stdout.strip()


def jget(path):
    f = ROOT / "jobs" / "reports" / path
    try: return json.loads(f.read_text())
    except Exception: return {}


def scan():
    F = []
    def add(sev, surface, issue, guide):
        F.append({"severity": sev, "surface": surface, "issue": issue, "guidance": guide})

    # 1. model recall / clinical safety
    mp = jget("../reports/model_performance.json") or {}
    drift = jget("drift_latest.json")
    if drift.get("verdict", "").startswith("SEVERE"):
        add("P1", "model", "Drift monitor reports SEVERE drift (live features vs training distribution)",
            "Retrain on same-setup ictal/interictal data; until then trust confidence only with human sign-off.")
    # 2. model recall known low (from model bundle)
    try:
        import joblib
        b = joblib.load(ROOT / "models/epilepsy_model.joblib")
        m = b.get("metrics", {})
        if b.get("caveat"):
            add("P1", "model", f"Model has dataset-confound caveat (acc {m.get('subject_wise_cv_accuracy')}, control=motor-imagery)",
                "Confidence partly reflects dataset, not only epilepsy. Use ictal/interictal data for clinical claims.")
    except Exception:
        pass
    # 3. unaddressed operator requests
    try:
        import clinical_db as cdb
        req = cdb.list_requests()
        if req["open_count"] > 0:
            add("P2", "requests", f"{req['open_count']} operator input(s) unaddressed",
                "Review 📥 Request Inbox; mark addressed/not-implemented/rejected so nothing is lost.")
    except Exception:
        pass
    # 4. backend live errors (since restart)
    since = sh("grep -n 'Started server process' jobs/logs/backend.log | tail -1 | cut -d: -f1") or "1"
    e500 = sh(f"tail -n +{since} jobs/logs/backend.log 2>/dev/null | grep -c '500 Internal'") or "0"
    if int(e500 or 0) > 0:
        add("P1", "backend", f"{e500} HTTP-500 since last restart", "Check jobs/logs/backend.log; wrap NaN/serialization in _json_safe.")
    # 5. frontend dev server down
    fe = sh("curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:3003/ -m 5")
    if fe != "200":
        add("P2", "frontend", f"Dev server :3003 not reachable (http={fe})", "Run: cd frontend && npm run dev -- --port 3003 (or operator opens the UI).")
    # 6. data gaps
    dm = jget("data_quality_latest.json")
    cov = dm.get("modality_coverage_pct", {})
    if cov.get("MRI", 100) < 10:
        add("P3", "data", f"MRI coverage very low ({cov.get('MRI')}%)", "Expected — DICOM not ingested. Note as limitation, not a bug.")
    chb = len(sh("ls data/real_eeg/epilepsy_physionet/chb*/chb*-summary.txt 2>/dev/null").splitlines())
    if chb < 10:
        add("P2", "data", f"Only {chb} CHB-MIT subjects on disk", "Download more PhysioNet subjects (10-20) for stronger cross-patient claims.")
    # 7. security: no multi-user auth
    add("P2", "security", "No multi-user auth / RBAC (single-operator mode)",
        "Fine for research/dev; required before multi-clinician or PHI deployment (see §47.6).")
    # 8. unpushed commits
    ahead = sh("git rev-list --count origin/main..HEAD 2>/dev/null") or "0"
    if int(ahead or 0) > 0:
        add("P3", "git", f"{ahead} commit(s) unpushed", "safe_push.sh auto-pushes on shared repos; or push manually.")
    # 9. stale crons (not fired in 24h where expected)
    if not (ROOT / "jobs/logs/autobuild.log").exists():
        add("P3", "automation", "AUTO-BUILD has not logged a run yet", "Confirm jobs/.autobuild_enabled exists + cron */15 active.")

    order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}
    F.sort(key=lambda x: order.get(x["severity"], 9))
    return F


def main():
    F = scan()
    now = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    # persist to DB
    c = sqlite3.connect(str(ROOT / "data/clinical.db"))
    c.execute("""CREATE TABLE IF NOT EXISTS advisor_issues(id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity TEXT,surface TEXT,issue TEXT,guidance TEXT,status TEXT DEFAULT 'open',scanned_at TEXT)""")
    c.execute("DELETE FROM advisor_issues WHERE status='open'")  # refresh open set
    for f in F:
        c.execute("INSERT INTO advisor_issues(severity,surface,issue,guidance,scanned_at) VALUES(?,?,?,?,?)",
                  (f["severity"], f["surface"], f["issue"], f["guidance"], now))
    c.commit()
    rep = {"scanned_at": now, "count": len(F), "issues": F}
    (ROOT / "jobs/reports/advisor_latest.json").write_text(json.dumps(rep, indent=2))
    md = [f"# 🧭 Advisor — issues you may not be aware of ({now})\n", f"**{len(F)} findings**\n"]
    for f in F:
        md.append(f"- **[{f['severity']}] {f['surface']}** — {f['issue']}\n    ↳ {f['guidance']}")
    (ROOT / "jobs/reports/advisor_latest.md").write_text("\n".join(md) + "\n")
    print(f"🧭 ADVISOR — {len(F)} issues you may not be aware of ({now}):")
    for f in F:
        print(f"  [{f['severity']}] {f['surface']}: {f['issue']}")
        print(f"       ↳ {f['guidance']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
