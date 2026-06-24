#!/usr/bin/env python3
"""Scheduled training job — runs the leakage-free epilepsy accuracy evaluations and
writes a timestamped result so you can SEE training happened + what it scored.

Output: jobs/reports/training_latest.json (+ timestamped copy) and a transaction-log row.
Run by cron (see install_train_cron.sh) or manually:  python3 scripts/scheduled_train.py
"""
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "jobs" / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


def run(script: str) -> dict:
    """Run a training/accuracy script, capture its tail + exit code."""
    t0 = datetime.now(timezone.utc)
    try:
        p = subprocess.run([sys.executable, str(ROOT / "scripts" / script)],
                           capture_output=True, text=True, timeout=3600, cwd=str(ROOT))
        return {"script": script, "exit_code": p.returncode,
                "ok": p.returncode == 0,
                "tail": (p.stdout or p.stderr or "")[-1500:],
                "seconds": round((datetime.now(timezone.utc) - t0).total_seconds(), 1)}
    except subprocess.TimeoutExpired:
        return {"script": script, "exit_code": 124, "ok": False, "tail": "TIMEOUT (>1h)", "seconds": 3600}
    except Exception as exc:  # noqa: BLE001
        return {"script": script, "exit_code": 1, "ok": False, "tail": str(exc)[:500], "seconds": 0}


def main():
    now = datetime.now(timezone.utc).astimezone()
    # the two honest, leakage-free epilepsy evaluations
    results = [run("accuracy_patient_specific.py"), run("accuracy_all_options.py")]
    report = {
        "run_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "run_at_local": now.isoformat(timespec="seconds"),
        "dataset": "data/real_eeg/epilepsy_physionet (CHB-MIT)",
        "results": results,
        "summary": f"{sum(1 for r in results if r['ok'])}/{len(results)} training runs succeeded",
    }
    (REPORTS / "training_latest.json").write_text(json.dumps(report, indent=2))
    stamp = now.strftime("%Y%m%d_%H%M%S")
    (REPORTS / f"training_{stamp}.json").write_text(json.dumps(report, indent=2))

    # log to transaction history (best-effort)
    try:
        sys.path.insert(0, str(ROOT))
        import clinical_db as cdb
        cdb.log_transaction("_system", component="training", action="scheduled_train",
                            detail=f"epilepsy CHB-MIT · {report['summary']}")
    except Exception:  # noqa: BLE001
        pass

    print(f"[{report['run_at_local']}] {report['summary']} → {REPORTS / 'training_latest.json'}")
    for r in results:
        print(f"  {'✓' if r['ok'] else '✗'} {r['script']} ({r['seconds']}s)")
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
