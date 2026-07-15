"""
Operator Requests Dashboard
Request lifecycle analytics: intake tracking, status distribution, category breakdown,
source analysis, resolution rates, implementation coverage, daily trends.
Real operator_requests table in clinical.db (310 rows).
"""

import sqlite3
import os
import json
from datetime import datetime, timedelta

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


def _rows(cur, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def overview():
    """Request KPIs, status/category/source distribution, resolution rate, daily volume."""
    con = _conn()
    cur = con.cursor()

    cur.execute("SELECT COUNT(*) FROM operator_requests")
    total = cur.fetchone()[0]

    status_counts = {}
    for row in _rows(cur, "SELECT status, COUNT(*) as cnt FROM operator_requests GROUP BY status"):
        status_counts[row["status"]] = row["cnt"]

    open_count = status_counts.get("open", 0)
    logged = status_counts.get("logged", 0)
    addressed = status_counts.get("addressed", 0)
    pending = status_counts.get("pending", 0)
    not_implemented = status_counts.get("not-implemented", 0)
    rejected = status_counts.get("rejected", 0)

    resolution_rate = round((addressed / total * 100), 1) if total else 0.0
    actionable = open_count + pending
    closed = addressed + not_implemented + rejected

    # Category distribution
    category_distribution = _rows(
        cur,
        "SELECT category, COUNT(*) as count FROM operator_requests GROUP BY category ORDER BY count DESC"
    )

    # Source distribution
    source_distribution = _rows(
        cur,
        "SELECT source, COUNT(*) as count FROM operator_requests GROUP BY source ORDER BY count DESC"
    )

    # Status distribution
    status_distribution = _rows(
        cur,
        "SELECT status, COUNT(*) as count FROM operator_requests GROUP BY status ORDER BY count DESC"
    )

    # Daily volume (last 21 days)
    cutoff = (datetime.utcnow() - timedelta(days=21)).strftime("%Y-%m-%d")
    daily_volume = _rows(
        cur,
        "SELECT DATE(ts_utc) as date, COUNT(*) as count "
        "FROM operator_requests WHERE DATE(ts_utc) >= ? "
        "GROUP BY DATE(ts_utc) ORDER BY date",
        (cutoff,),
    )

    # Implementation coverage: requests with impl_module or impl_api set
    cur.execute(
        "SELECT COUNT(*) FROM operator_requests "
        "WHERE impl_module IS NOT NULL AND impl_module != ''"
    )
    with_module = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM operator_requests "
        "WHERE impl_api IS NOT NULL AND impl_api != ''"
    )
    with_api = cur.fetchone()[0]
    cur.execute(
        "SELECT COUNT(*) FROM operator_requests "
        "WHERE tested IS NOT NULL AND tested != ''"
    )
    tested_count = cur.fetchone()[0]

    implementation_coverage = {
        "with_module": with_module,
        "with_api": with_api,
        "tested": tested_count,
        "module_pct": round(with_module / total * 100, 1) if total else 0,
        "api_pct": round(with_api / total * 100, 1) if total else 0,
        "tested_pct": round(tested_count / total * 100, 1) if total else 0,
    }

    # Category × status cross-tab
    cross_tab = _rows(
        cur,
        "SELECT category, status, COUNT(*) as count "
        "FROM operator_requests GROUP BY category, status ORDER BY category, count DESC"
    )

    con.close()
    return {
        "total_requests": total,
        "open": open_count,
        "logged": logged,
        "addressed": addressed,
        "pending": pending,
        "not_implemented": not_implemented,
        "rejected": rejected,
        "actionable": actionable,
        "closed": closed,
        "resolution_rate": resolution_rate,
        "category_distribution": category_distribution,
        "source_distribution": source_distribution,
        "status_distribution": status_distribution,
        "daily_volume": daily_volume,
        "implementation_coverage": implementation_coverage,
        "cross_tab": cross_tab,
    }


def breakdown():
    """Per-category detail, recent requests, implementation tracking, unaddressed list."""
    con = _conn()
    cur = con.cursor()

    # All requests
    all_rows = _rows(
        cur,
        "SELECT id, request_text, category, status, source, "
        "ts_utc, updated_at, impl_module, impl_tab, impl_api, impl_db, impl_ui, "
        "weblink, tested, notes "
        "FROM operator_requests ORDER BY id DESC",
    )

    # Per category
    per_category = {}
    for r in all_rows:
        cat = r["category"] or "uncategorized"
        per_category.setdefault(cat, []).append({
            "id": r["id"],
            "request_text": (r["request_text"] or "")[:120],
            "status": r["status"],
            "source": r["source"],
            "ts_utc": r["ts_utc"],
        })

    # Recent 30
    recent_requests = [{
        "id": r["id"],
        "request_text": (r["request_text"] or "")[:120],
        "category": r["category"],
        "status": r["status"],
        "source": r["source"],
        "ts_utc": r["ts_utc"],
    } for r in all_rows[:30]]

    # Unaddressed: open or pending
    unaddressed = [{
        "id": r["id"],
        "request_text": (r["request_text"] or "")[:120],
        "category": r["category"],
        "source": r["source"],
        "ts_utc": r["ts_utc"],
    } for r in all_rows if r["status"] in ("open", "pending")]

    # Implemented (have impl_module)
    implemented = [{
        "id": r["id"],
        "request_text": (r["request_text"] or "")[:80],
        "impl_module": r["impl_module"],
        "impl_tab": r["impl_tab"],
        "impl_api": r["impl_api"],
        "tested": r["tested"],
    } for r in all_rows if r.get("impl_module")]

    # Source × status cross-tab
    source_status = _rows(
        cur,
        "SELECT source, status, COUNT(*) as count "
        "FROM operator_requests GROUP BY source, status ORDER BY source, count DESC"
    )

    con.close()
    return {
        "per_category": per_category,
        "recent_requests": recent_requests,
        "unaddressed": unaddressed,
        "unaddressed_count": len(unaddressed),
        "implemented": implemented,
        "implemented_count": len(implemented),
        "source_status": source_status,
    }


def definitions():
    """Request statuses, categories, sources, implementation fields, glossary."""
    return {
        "request_statuses": {
            "open": "New request, not yet triaged or assigned — needs review",
            "logged": "Request recorded from transcript/history — acknowledged but not actioned",
            "addressed": "Request completed and verified — feature built or answer provided",
            "pending": "Request triaged, awaiting operator decision or external dependency",
            "not-implemented": "Request reviewed but intentionally not built (out of scope or deferred)",
            "rejected": "Request declined — invalid, duplicate, or not feasible",
        },
        "categories": {
            "general": "Ad-hoc requests from chat interaction — feature asks, questions, fixes",
            "session": "Requests generated during a working session — multi-step builds",
            "history": "Historical requests imported from conversation transcripts",
            "meta": "Requests about the system itself — tooling, process, configuration",
        },
        "sources": {
            "chat": "Submitted via live chat interaction with the operator",
            "transcript": "Extracted from conversation transcript review",
        },
        "implementation_fields": {
            "impl_module": "Backend module or script that implements the request",
            "impl_tab": "Frontend tab/panel where the feature is accessible",
            "impl_api": "API endpoint(s) that serve the feature",
            "impl_db": "Database table(s) or data source backing the feature",
            "impl_ui": "UI section or navigation group where the feature appears",
            "weblink": "URL to access the feature in the running application",
            "tested": "Verification status — whether endpoint returns 200 and UI renders",
        },
        "glossary": {
            "operator request": "A feature ask, question, or task submitted by the system operator",
            "triage": "The process of reviewing and categorizing incoming requests",
            "resolution": "Completing a request by building the feature or providing the answer",
            "resolution rate": "Percentage of total requests that have been addressed (completed)",
            "actionable": "Requests that are open or pending — need work or a decision",
            "backlog": "Accumulated open requests not yet triaged or started",
            "implementation coverage": "Percentage of requests with backend module, API, or test verification",
            "SLA": "Service Level Agreement — target time from request to resolution",
            "transcript extraction": "Process of mining past conversations for implicit feature requests",
            "stale request": "An open request older than 7 days with no updates",
            "request velocity": "Rate of new requests per day — measures operator engagement",
            "burndown": "Trend of remaining open requests over time — shows resolution momentum",
        },
    }


if __name__ == "__main__":
    print(json.dumps({"overview": overview(), "breakdown": breakdown(), "definitions": definitions()}, indent=2, default=str))
