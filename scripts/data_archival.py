#!/usr/bin/env python3
"""Clinical Data Manager — Data Archival / Retention report.

Surfaces, per real table: row count, record date-range, age of the oldest
record, and archival candidates under a configurable retention policy.
100% real (reads live table timestamps) — no synthetic, no destructive action
(report only; archival execution is a separate, gated operation).

Retention policy follows the global data-retention standard (§7.4 / §41.2):
operational/audit logs are short-lived; clinical records are long-lived.
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "clinical.db")

# table -> (timestamp_column_or_None, retention_days_or_None, class)
# retention_days None = keep indefinitely (clinical record of value)
RETENTION = {
    "analyses": ("created_at", None, "clinical"),
    "assessments": ("created_at", None, "clinical"),
    "seizure_diary": ("created_at", None, "clinical"),
    "medications": ("created_at", None, "clinical"),
    "uploads": ("created_at", None, "clinical"),
    "expert_reviews": ("created_at", None, "clinical"),
    "team_messages": ("created_at", 90, "operational"),
    "transaction_log": (None, 30, "operational"),       # no ts column → age N/A
    "conversation_log": ("ts_utc", 90, "operational"),
    "operator_requests": (None, None, "operational"),   # no ts column
}


def _parse_ts(s):
    if not s:
        return None
    s = str(s).strip().replace(" ", "T")
    # normalize trailing Z / microseconds / offset for fromisoformat tolerance
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f"):
            try:
                return datetime.strptime(s[:26], fmt)
            except ValueError:
                continue
    return None


def _now():
    return datetime.now(timezone.utc)


def _age_days(ts):
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return (_now() - ts).days


def archival_report():
    """Per-table retention + archival-candidate report over real data."""
    c = sqlite3.connect(DB_PATH)
    tables = []
    total_candidates = 0
    for tbl, (tscol, retain_days, klass) in RETENTION.items():
        try:
            n = c.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        except sqlite3.OperationalError:
            continue
        row = {"table": tbl, "rows": n, "class": klass,
               "retention_days": retain_days, "timestamped": tscol is not None}
        candidates = 0
        oldest_age = None
        if tscol and n:
            tsvals = [_parse_ts(r[0]) for r in c.execute(f"SELECT {tscol} FROM {tbl}") if r[0]]
            ages = [_age_days(t) for t in tsvals if t is not None]
            if ages:
                oldest_age = max(ages)
                row["date_range_days"] = {"oldest": max(ages), "newest": min(ages)}
                if retain_days is not None:
                    candidates = sum(1 for a in ages if a > retain_days)
        row["oldest_age_days"] = oldest_age
        row["archival_candidates"] = candidates
        if not tscol and retain_days is not None:
            row["note"] = "no timestamp column — age-based archival N/A (needs created_at)"
        total_candidates += candidates
        tables.append(row)
    c.close()

    tables.sort(key=lambda r: r["archival_candidates"], reverse=True)
    return {
        "available": True,
        "generated_at": _now().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "policy": {"clinical": "retain indefinitely (record of value)",
                   "operational_logs": "transaction_log 30d · conversation/team 90d (§7.4/§41.2)"},
        "tables": tables,
        "summary": {
            "tables_tracked": len(tables),
            "total_rows": sum(t["rows"] for t in tables),
            "total_archival_candidates": total_candidates,
            "tables_without_timestamp": [t["table"] for t in tables if not t["timestamped"]],
        },
        "note": ("Report only — no rows are deleted/archived here. Archival execution is a "
                 "separate gated operation (§42). Clinical tables are retained indefinitely; "
                 "operational logs follow the retention policy."),
    }


if __name__ == "__main__":
    r = archival_report()
    print("Data Archival report:", r["summary"])
    for t in r["tables"][:6]:
        print(f"  {t['table']}: rows={t['rows']} class={t['class']} "
              f"oldest={t['oldest_age_days']}d candidates={t['archival_candidates']}")
