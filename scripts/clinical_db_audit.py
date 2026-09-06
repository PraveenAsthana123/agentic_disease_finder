#!/usr/bin/env python3
"""Clinical DB backup + analysis-audit report (cron target).

- Backs up data/clinical.db via SQLite's online backup API (safe under WAL).
- Writes a Markdown + JSON audit: patient/analysis/survey counts, prediction
  distribution, average confidence, signal-quality breakdown.

Usage: python scripts/clinical_db_audit.py
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "clinical.db"
BACKUP_DIR = ROOT / "backups"
REPORT_DIR = ROOT / "jobs" / "reports"
KEEP_BACKUPS = 14  # retain ~2 weeks of twice-daily backups


def _now():
    return datetime.now(timezone.utc).astimezone()


def backup() -> Path | None:
    if not DB.exists():
        return None
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = _now().strftime("%Y%m%d_%H%M%S")
    dest = BACKUP_DIR / f"clinical_db_{ts}.db"
    src = sqlite3.connect(DB)
    dst = sqlite3.connect(dest)
    with dst:
        src.backup(dst)
    src.close(); dst.close()

    # Prune old backups.
    backups = sorted(BACKUP_DIR.glob("clinical_db_*.db"))
    for old in backups[:-KEEP_BACKUPS]:
        old.unlink(missing_ok=True)
    return dest


def audit() -> dict:
    if not DB.exists():
        return {"error": f"No database at {DB}"}
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row

    def scalar(q):
        return c.execute(q).fetchone()[0]

    patients = scalar("SELECT COUNT(*) FROM patients")
    analyses = scalar("SELECT COUNT(*) FROM analyses")
    surveys = scalar("SELECT COUNT(*) FROM surveys")
    avg_conf = c.execute("SELECT AVG(confidence) FROM analyses WHERE confidence IS NOT NULL").fetchone()[0]
    pred_dist = {r["predicted_label"]: r["n"] for r in c.execute(
        "SELECT predicted_label, COUNT(*) n FROM analyses GROUP BY predicted_label")}
    quality_dist = {r["signal_quality"]: r["n"] for r in c.execute(
        "SELECT signal_quality, COUNT(*) n FROM analyses GROUP BY signal_quality")}
    by_disease = {r["disease"]: r["n"] for r in c.execute(
        "SELECT disease, COUNT(*) n FROM analyses GROUP BY disease")}
    c.close()

    return {
        "generated_at": _now().isoformat(timespec="seconds"),
        "patients": patients,
        "analyses": analyses,
        "surveys": surveys,
        "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "prediction_distribution": pred_dist,
        "signal_quality_distribution": quality_dist,
        "analyses_by_disease": by_disease,
    }


def write_report(a: dict, backup_path: Path | None) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Clinical DB Audit",
        "",
        f"_Generated {a.get('generated_at')}_",
        f"_Backup: {backup_path.name if backup_path else '(none — no DB yet)'}_",
        "",
        f"- Patients: **{a.get('patients', 0)}**",
        f"- Analyses: **{a.get('analyses', 0)}**",
        f"- Surveys: **{a.get('surveys', 0)}**",
        f"- Avg confidence: **{a.get('avg_confidence')}**",
        "",
        "## Prediction distribution",
    ]
    lines += [f"- {k}: {v}" for k, v in (a.get("prediction_distribution") or {}).items()] or ["- (none)"]
    lines += ["", "## Signal quality"]
    lines += [f"- {k}: {v}" for k, v in (a.get("signal_quality_distribution") or {}).items()] or ["- (none)"]
    lines += ["", "## Analyses by disease"]
    lines += [f"- {k}: {v}" for k, v in (a.get("analyses_by_disease") or {}).items()] or ["- (none)"]
    lines += [""]

    (REPORT_DIR / "clinical_audit_latest.md").write_text("\n".join(lines), encoding="utf-8")
    (REPORT_DIR / "clinical_audit_latest.json").write_text(json.dumps(a, indent=2), encoding="utf-8")


def main() -> int:
    bk = backup()
    a = audit()
    write_report(a, bk)
    print(f"[clinical_db_audit] {a.get('generated_at')}")
    print(f"  backup: {bk}")
    print(f"  patients={a.get('patients')} analyses={a.get('analyses')} surveys={a.get('surveys')}")
    print(f"  report: {REPORT_DIR / 'clinical_audit_latest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
