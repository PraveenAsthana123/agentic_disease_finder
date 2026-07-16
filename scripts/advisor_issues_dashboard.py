"""
Neuro AI Ecosystem — Advisor Issues Dashboard
==============================================
System health advisory analytics from advisor_issues table.

Severities: P0 (critical), P1 (high), P2 (medium), P3 (low)
Surfaces: model, backend, data, security, git
Statuses: open, resolved, wontfix

Real data: advisor_issues table in clinical.db, populated by scripts/advisor_agent.py.
"""

import sqlite3
from pathlib import Path
from collections import defaultdict

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview():
    """Advisor issues overview — totals, severity distribution, surface breakdown,
    status counts, open-issue rate, scan timeline."""
    conn = _conn()
    cur = conn.cursor()

    # Total issues
    cur.execute("SELECT COUNT(*) FROM advisor_issues")
    total_issues = cur.fetchone()[0]

    # Severity distribution
    cur.execute("""
        SELECT severity, COUNT(*) cnt
        FROM advisor_issues
        GROUP BY severity
        ORDER BY severity
    """)
    severity_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    # Surface distribution
    cur.execute("""
        SELECT surface, COUNT(*) cnt
        FROM advisor_issues
        GROUP BY surface
        ORDER BY cnt DESC
    """)
    surface_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    # Status distribution
    cur.execute("""
        SELECT status, COUNT(*) cnt
        FROM advisor_issues
        GROUP BY status
        ORDER BY cnt DESC
    """)
    status_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    # Open issue count and rate
    cur.execute("SELECT COUNT(*) FROM advisor_issues WHERE status = 'open'")
    open_count = cur.fetchone()[0]
    open_rate = round(open_count * 100.0 / total_issues, 1) if total_issues else 0

    # Critical/high (P0+P1) open count
    cur.execute("""
        SELECT COUNT(*) FROM advisor_issues
        WHERE status = 'open' AND severity IN ('P0', 'P1')
    """)
    critical_open = cur.fetchone()[0]

    # Most recent scan
    cur.execute("SELECT MAX(scanned_at) FROM advisor_issues")
    last_scan = cur.fetchone()[0]

    # Scan timeline (issues per scan date)
    cur.execute("""
        SELECT DATE(scanned_at) scan_date, COUNT(*) cnt
        FROM advisor_issues
        WHERE scanned_at IS NOT NULL
        GROUP BY DATE(scanned_at)
        ORDER BY scan_date
    """)
    scan_timeline = [{"date": r[0], "issues": r[1]} for r in cur.fetchall()]

    conn.close()
    return {
        "total_issues": total_issues,
        "open_count": open_count,
        "open_rate": open_rate,
        "critical_open": critical_open,
        "last_scan": last_scan,
        "severity_distribution": severity_dist,
        "surface_distribution": surface_dist,
        "status_distribution": status_dist,
        "scan_timeline": scan_timeline,
    }


def breakdown():
    """Advisor issues breakdown — full issue list with details, grouped by severity,
    open issues requiring attention, surface-severity cross-tabulation."""
    conn = _conn()
    cur = conn.cursor()

    # All issues ordered by severity then scanned_at
    cur.execute("""
        SELECT id, severity, surface, issue, guidance, status, scanned_at
        FROM advisor_issues
        ORDER BY
            CASE severity
                WHEN 'P0' THEN 0
                WHEN 'P1' THEN 1
                WHEN 'P2' THEN 2
                WHEN 'P3' THEN 3
                ELSE 4
            END,
            scanned_at DESC
    """)
    all_issues = _dict_rows(cur)

    # Open issues needing attention (P0/P1 first)
    open_issues = [i for i in all_issues if i.get("status") == "open"]

    # Surface x severity cross-tab
    cur.execute("""
        SELECT surface, severity, COUNT(*) cnt
        FROM advisor_issues
        GROUP BY surface, severity
        ORDER BY surface, severity
    """)
    cross_data = _dict_rows(cur)
    surface_severity = defaultdict(dict)
    for row in cross_data:
        surface_severity[row["surface"]][row["severity"]] = row["cnt"]
    surface_severity_list = [
        {"surface": s, **counts}
        for s, counts in sorted(surface_severity.items())
    ]

    # Per-surface summary
    cur.execute("""
        SELECT surface,
               COUNT(*) total,
               SUM(CASE WHEN status='open' THEN 1 ELSE 0 END) open_cnt
        FROM advisor_issues
        GROUP BY surface
        ORDER BY open_cnt DESC
    """)
    surface_summary = _dict_rows(cur)

    conn.close()
    return {
        "all_issues": all_issues,
        "open_issues": open_issues,
        "surface_severity": surface_severity_list,
        "surface_summary": surface_summary,
    }


def definitions():
    """Advisor issues definitions — severity tiers, surface categories,
    status meanings, advisor agent description, clinical glossary."""
    return {
        "severity_tiers": [
            {"tier": "P0", "label": "Critical", "description": "System-breaking or patient-safety issue requiring immediate action."},
            {"tier": "P1", "label": "High", "description": "Significant issue affecting accuracy, reliability, or compliance. Fix before clinical use."},
            {"tier": "P2", "label": "Medium", "description": "Moderate issue — data gaps, limited coverage. Address for production readiness."},
            {"tier": "P3", "label": "Low", "description": "Minor concern — informational, cosmetic, or expected limitation. Track but low urgency."},
        ],
        "surface_categories": [
            {"surface": "model", "description": "ML model accuracy, drift, dataset confounds, retraining needs."},
            {"surface": "backend", "description": "API errors, HTTP 500s, serialization issues, server health."},
            {"surface": "data", "description": "Data coverage gaps, missing subjects, ingestion limitations."},
            {"surface": "security", "description": "Authentication, authorization, RBAC, PHI protection."},
            {"surface": "git", "description": "Repository hygiene — unpushed commits, branch state."},
        ],
        "status_definitions": [
            {"status": "open", "description": "Issue identified and not yet resolved. Requires attention."},
            {"status": "resolved", "description": "Issue has been addressed and verified."},
            {"status": "wontfix", "description": "Issue acknowledged but will not be fixed (accepted limitation or out of scope)."},
        ],
        "advisor_agent": {
            "description": "The Advisor Agent (scripts/advisor_agent.py) performs automated system health scans, checking model drift, API errors, data coverage, security posture, and git state. Findings are written to the advisor_issues table with severity, surface, guidance, and scan timestamp.",
            "trigger": "Runs via cron or on-demand to maintain continuous awareness of system health.",
        },
        "glossary": [
            {"term": "Drift", "definition": "Statistical divergence between training data distribution and live/incoming data, indicating model may need retraining."},
            {"term": "Dataset Confound", "definition": "When model accuracy reflects dataset artifacts (e.g., different recording setups for classes) rather than genuine clinical signal."},
            {"term": "RBAC", "definition": "Role-Based Access Control — restricting system access based on user roles (neurologist, technician, admin)."},
            {"term": "PHI", "definition": "Protected Health Information — any individually identifiable health data subject to HIPAA/privacy regulations."},
            {"term": "HTTP 500", "definition": "Internal Server Error — indicates an unhandled exception in the backend API."},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston–MIT Scalp EEG Database — a benchmark epilepsy EEG dataset from PhysioNet."},
        ],
    }
