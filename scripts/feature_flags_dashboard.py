"""Feature Flags Management Dashboard — flag lifecycle analytics from clinical.db.

Tracks feature flag status (enabled/disabled), rollout percentages, category
distribution, owner workload, and flag age/staleness analytics.

Sources:
- feature_flags table (flag_id, name, description, enabled, rollout_percentage,
  created_at, updated_at, owner, category)
"""

import sqlite3
import os
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        return cur.fetchone()[0]
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _days_since(date_str):
    """Return days between date_str and now."""
    if not date_str:
        return None
    try:
        dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return (datetime.now() - dt.replace(tzinfo=None)).days
    except Exception:
        return None


def overview():
    """Feature flags overview — total flags, enabled/disabled counts, category
    distribution, rollout tiers, owner workload, staleness analysis."""
    conn = _conn()
    cur = conn.cursor()

    total = _safe(cur, "SELECT COUNT(*) FROM feature_flags")
    enabled = _safe(cur, "SELECT COUNT(*) FROM feature_flags WHERE enabled = 1")
    disabled = total - enabled

    avg_rollout = _safe(cur, "SELECT AVG(rollout_percentage) FROM feature_flags WHERE enabled = 1", default=0)
    full_rollout = _safe(cur, "SELECT COUNT(*) FROM feature_flags WHERE enabled = 1 AND rollout_percentage = 100")

    # Category distribution
    cat_rows = _safe_rows(cur, """
        SELECT category, COUNT(*) as cnt,
               SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) as enabled_cnt
        FROM feature_flags
        GROUP BY category
        ORDER BY cnt DESC
    """)
    category_distribution = [
        {"category": r[0], "total": r[1], "enabled": r[2], "disabled": r[1] - r[2]}
        for r in cat_rows
    ]

    # Rollout tiers
    tier_rows = _safe_rows(cur, """
        SELECT
            CASE
                WHEN rollout_percentage = 0 THEN 'Off (0%)'
                WHEN rollout_percentage <= 25 THEN 'Canary (1-25%)'
                WHEN rollout_percentage <= 50 THEN 'Beta (26-50%)'
                WHEN rollout_percentage <= 75 THEN 'Staged (51-75%)'
                WHEN rollout_percentage < 100 THEN 'Wide (76-99%)'
                ELSE 'Full (100%)'
            END as tier,
            COUNT(*) as cnt
        FROM feature_flags
        GROUP BY tier
        ORDER BY MIN(rollout_percentage)
    """)
    rollout_tiers = [{"tier": r[0], "count": r[1]} for r in tier_rows]

    # Owner workload
    owner_rows = _safe_rows(cur, """
        SELECT owner, COUNT(*) as cnt,
               SUM(CASE WHEN enabled = 1 THEN 1 ELSE 0 END) as active
        FROM feature_flags
        GROUP BY owner
        ORDER BY cnt DESC
    """)
    owner_workload = [
        {"owner": r[0], "total_flags": r[1], "active": r[2]}
        for r in owner_rows
    ]

    # Staleness analysis — flags not updated in 90+ days
    all_flags = _safe_rows(cur, "SELECT flag_id, name, updated_at, enabled FROM feature_flags")
    stale_count = 0
    for f in all_flags:
        days = _days_since(f[2])
        if days is not None and days > 90:
            stale_count += 1

    conn.close()
    return {
        "available": True,
        "kpis": {
            "total_flags": total,
            "enabled": enabled,
            "disabled": disabled,
            "full_rollout": full_rollout,
            "avg_rollout_pct": round(avg_rollout, 1),
            "stale_flags": stale_count,
            "owners": len(owner_workload),
            "categories": len(category_distribution),
        },
        "category_distribution": category_distribution,
        "rollout_tiers": rollout_tiers,
        "owner_workload": owner_workload,
    }


def breakdown():
    """Per-flag detail, recently updated flags, disabled flags that may need
    cleanup, rollout progression candidates."""
    conn = _conn()
    cur = conn.cursor()

    # All flags with computed fields
    rows = _safe_rows(cur, """
        SELECT flag_id, name, description, enabled, rollout_percentage,
               created_at, updated_at, owner, category
        FROM feature_flags
        ORDER BY updated_at DESC
    """)
    all_flags = []
    for r in rows:
        age_days = _days_since(r[5])
        stale_days = _days_since(r[6])
        all_flags.append({
            "flag_id": r[0],
            "name": r[1],
            "description": r[2],
            "enabled": bool(r[3]),
            "rollout_percentage": r[4],
            "created_at": r[5],
            "updated_at": r[6],
            "owner": r[7],
            "category": r[8],
            "age_days": age_days,
            "days_since_update": stale_days,
        })

    # Recently updated (last 30 days)
    recent = [f for f in all_flags if f["days_since_update"] is not None and f["days_since_update"] <= 30]

    # Stale flags (not updated in 90+ days)
    stale = [f for f in all_flags if f["days_since_update"] is not None and f["days_since_update"] > 90]

    # Disabled flags — cleanup candidates
    disabled = [f for f in all_flags if not f["enabled"]]

    # Rollout candidates — enabled but not at 100%
    rollout_candidates = [
        f for f in all_flags
        if f["enabled"] and f["rollout_percentage"] < 100
    ]

    # Per-category breakdown
    cat_flags = {}
    for f in all_flags:
        cat = f["category"]
        if cat not in cat_flags:
            cat_flags[cat] = []
        cat_flags[cat].append({
            "flag_id": f["flag_id"],
            "name": f["name"],
            "enabled": f["enabled"],
            "rollout_percentage": f["rollout_percentage"],
            "owner": f["owner"],
        })

    conn.close()
    return {
        "available": True,
        "all_flags": all_flags,
        "recently_updated": recent,
        "stale_flags": stale,
        "disabled_flags": disabled,
        "rollout_candidates": rollout_candidates,
        "by_category": cat_flags,
    }


def definitions():
    """Feature flag glossary — statuses, rollout tiers, staleness thresholds,
    category descriptions, best practices."""
    return {
        "available": True,
        "statuses": {
            "enabled": "Flag is active and controls feature visibility based on rollout percentage",
            "disabled": "Flag is off — feature is hidden for all users",
        },
        "rollout_tiers": {
            "Off (0%)": "Feature completely disabled or enabled but not yet rolled out",
            "Canary (1-25%)": "Small percentage of users for initial validation",
            "Beta (26-50%)": "Moderate rollout for broader testing",
            "Staged (51-75%)": "Majority rollout with holdback for comparison",
            "Wide (76-99%)": "Near-complete rollout, final validation phase",
            "Full (100%)": "Fully rolled out to all users",
        },
        "staleness_thresholds": {
            "fresh": "Updated within last 30 days",
            "aging": "31-90 days since last update",
            "stale": "90+ days since last update — review for cleanup or promotion",
        },
        "categories": {
            "AI Model": "Machine learning model features (training, inference, XAI)",
            "Data Pipeline": "Data processing and ETL pipeline features",
            "Integration": "Third-party service integrations (Slack, SIEM, etc.)",
            "Reporting": "Clinical and operational reporting features",
            "Security": "Authentication, authorization, compliance features",
            "UI": "User interface and experience features",
        },
        "best_practices": [
            "Remove flags that have been at 100% rollout for 30+ days",
            "Review disabled flags quarterly for cleanup",
            "Each flag should have a single owner responsible for lifecycle",
            "Document rollout criteria and rollback plan per flag",
            "Use canary rollout (10-25%) before broader deployment",
        ],
    }
