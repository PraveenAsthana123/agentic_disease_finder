#!/usr/bin/env python3
"""Scheduled CDM data-quality audit — runs the live data-quality engine and snapshots
the AI-readiness score + dimensions to jobs/reports/data_quality_latest.json."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    import clinical_db as cdb
    r = cdb.data_manager_report()
    out = ROOT / "jobs" / "reports" / "data_quality_latest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(r, indent=2, default=str))
    print(f"CDM data quality: AI-readiness {r.get('ai_readiness_score')} ({r.get('ai_readiness_grade')}) "
          f"· {r.get('n_patients')} patients")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
