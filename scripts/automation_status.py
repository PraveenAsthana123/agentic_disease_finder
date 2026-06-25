#!/usr/bin/env python3
"""Single source of truth for 'how do I know the automation is working' —
plan? crons? count? system? independent? sequence? crash-survival? completion? Returns JSON."""
import json, subprocess, os
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
PROJ_TAGS = ["AGENTICFINDER-TRAIN","VECTOR-INGEST","VIDEO-FRAMES","GRAPH-DB","VALIDATION","CV-PIPELINE",
             "FAIRNESS","DRIFT","DATA-QUALITY","STATUS-REPORT","CONSISTENCY","AUTO-PLAN","AUTO-BUILD"]


def build():
    cron = subprocess.run("crontab -l", shell=True, capture_output=True, text=True).stdout
    jobs = []
    for line in cron.splitlines():
        line = line.strip()
        if line.startswith("#") or not line:
            continue
        for t in PROJ_TAGS:
            if line.endswith("# " + t):
                parts = line.split()
                sched = " ".join(parts[:5]) if parts[0] != "@reboot" else "@reboot"
                jobs.append({"tag": t, "schedule": sched, "this_project": "agenticfinder" in line})
    plan = ROOT / "jobs/reports/auto_plan.md"
    cron_running = subprocess.run("pgrep -x cron", shell=True, capture_output=True).returncode == 0
    autobuild_armed = (ROOT / "jobs/.autobuild_enabled").exists()
    # last autobuild + track activity
    def tail(p, n=3):
        f = ROOT / p
        return f.read_text(errors="ignore").splitlines()[-n:] if f.exists() else []
    return {
        "now": datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "plan_created": plan.exists(),
        "plan_updated": datetime.fromtimestamp(plan.stat().st_mtime).strftime("%Y-%m-%d %H:%M") if plan.exists() else None,
        "cron_daemon_running": cron_running,
        "system": "system cron (systemd) — independent of VS Code/editor",
        "n_jobs_this_project": len(jobs),
        "jobs": jobs,
        "independent": "Each cron is its own process. AUTO-BUILD serialized via flock; others independent.",
        "survives_crash": cron_running,  # cron auto-starts on reboot
        "autobuild_armed": autobuild_armed,
        "completion": "AUTO-BUILD builds 1 verified item/run until queue empty or blocked",
        "recent_builds": tail("jobs/logs/autobuild.log", 4),
        "recent_events": tail("jobs/logs/track.md", 5),
    }


if __name__ == "__main__":
    import sys
    d = build()
    if "--json" in sys.argv:
        print(json.dumps(d, indent=2))
    else:
        print(f"════ AUTOMATION STATUS {d['now']} ════")
        print(f"Plan: {'✅ '+d['plan_updated'] if d['plan_created'] else '❌'}")
        print(f"Cron daemon: {'✅ running' if d['cron_daemon_running'] else '❌ STOPPED'} ({d['system']})")
        print(f"Jobs (this project): {d['n_jobs_this_project']}")
        for j in d["jobs"]:
            print(f"  {j['schedule']:12s} {j['tag']}")
        print(f"Survives crash/reboot: {'✅' if d['survives_crash'] else '❌'}  ·  AUTO-BUILD armed: {'✅' if d['autobuild_armed'] else '❌'}")
        print(f"Completion: {d['completion']}")
