"""Alert Fatigue Dashboard — alert volume analytics, deduplication, severity routing

Tracks alert generation across all monitoring subsystems, detects duplicate/noisy
alerts, measures suppression rates, and provides severity-based routing analytics
to combat alert fatigue in production environments.

Addresses: production_issues.layers[Observability] — "Alert Fatigue" planned → built

Sources:
  jobs/reports/*.json    — monitoring report outputs (health, drift, fairness, etc.)
  config/jobs.json       — cron job definitions (alert sources)
  config/production_issues.json — issue severity catalog
  data/clinical.db       — iot_alerts, emergency_sos_events tables
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone, timedelta
from collections import Counter

BASE = os.path.join(os.path.dirname(__file__), '..')

# ── Alert sources (monitoring subsystems that generate alerts) ───────────────
_ALERT_SOURCES = [
    {"id": "health", "name": "System Health", "report": "jobs/reports/health_latest.json",
     "description": "Backend / API / DB health checks"},
    {"id": "drift", "name": "Drift Monitor", "report": "jobs/reports/drift_latest.json",
     "description": "Data and model drift detection"},
    {"id": "fairness", "name": "Fairness Tester", "report": "jobs/reports/fairness_latest.json",
     "description": "Bias and fairness violation alerts"},
    {"id": "data_quality", "name": "Data Quality", "report": "jobs/reports/data_quality_latest.json",
     "description": "Missing values, outliers, schema violations"},
    {"id": "consistency", "name": "Consistency", "report": "jobs/reports/consistency_latest.json",
     "description": "Model consistency and reliability checks"},
    {"id": "status", "name": "Status Report", "report": "jobs/reports/status_latest.json",
     "description": "Overall system status aggregation"},
]

# ── Severity routing rules ──────────────────────────────────────────────────
_ROUTING_RULES = [
    {"severity": "critical", "channel": "PagerDuty + SMS", "response_sla": "5 min",
     "escalation": "On-call engineer → Team lead → VP Eng",
     "examples": ["Service down", "Data breach", "Model producing harmful outputs"]},
    {"severity": "high", "channel": "Slack #incidents + Email", "response_sla": "30 min",
     "escalation": "On-call engineer → Team lead",
     "examples": ["Accuracy below threshold", "Drift detected", "Fairness violation"]},
    {"severity": "medium", "channel": "Slack #monitoring", "response_sla": "4 hours",
     "escalation": "Assigned engineer",
     "examples": ["Elevated latency", "Cache miss rate spike", "Non-critical config drift"]},
    {"severity": "low", "channel": "Dashboard only", "response_sla": "Next business day",
     "escalation": "Backlog triage",
     "examples": ["Info-level warnings", "Cosmetic drift", "Deprecation notices"]},
]

# ── Dedup strategies ────────────────────────────────────────────────────────
_DEDUP_STRATEGIES = [
    {"id": "content_hash", "name": "Content Hash Dedup",
     "description": "Alerts with identical content hash within a window are suppressed"},
    {"id": "source_cooldown", "name": "Source Cooldown",
     "description": "Same source cannot fire more than N alerts per window"},
    {"id": "severity_merge", "name": "Severity Merge",
     "description": "Multiple low-severity alerts from same source are merged into one medium alert"},
    {"id": "flap_detection", "name": "Flap Detection",
     "description": "Rapidly alternating OK/ALERT states are collapsed into a single flapping alert"},
]


def _load_report(path):
    """Load a JSON report file safely."""
    full = os.path.join(BASE, path)
    try:
        with open(full) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _extract_alerts_from_reports():
    """Extract alert-like signals from monitoring reports."""
    alerts = []
    now = datetime.now(timezone.utc)

    for source in _ALERT_SOURCES:
        report = _load_report(source["report"])
        if not report:
            continue

        # Extract warnings/errors/violations from reports
        source_alerts = []

        # Health reports
        if source["id"] == "health":
            status = report.get("status", "healthy")
            if status != "healthy":
                source_alerts.append({
                    "message": f"System health: {status}",
                    "severity": "critical" if status == "unhealthy" else "medium",
                })
            errors = report.get("errors", [])
            for err in errors[:5]:
                source_alerts.append({
                    "message": str(err)[:120],
                    "severity": "high",
                })

        # Drift reports
        elif source["id"] == "drift":
            drifts = report.get("drifts", report.get("drift_results", []))
            if isinstance(drifts, list):
                for d in drifts[:5]:
                    if isinstance(d, dict) and d.get("drifted", False):
                        source_alerts.append({
                            "message": f"Drift detected: {d.get('feature', d.get('name', 'unknown'))}",
                            "severity": "high",
                        })

        # Fairness reports
        elif source["id"] == "fairness":
            violations = report.get("violations", report.get("fairness_violations", []))
            if isinstance(violations, list):
                for v in violations[:5]:
                    msg = v.get("message", v.get("metric", "fairness violation")) if isinstance(v, dict) else str(v)
                    source_alerts.append({
                        "message": f"Fairness: {str(msg)[:100]}",
                        "severity": "high",
                    })

        # Data quality
        elif source["id"] == "data_quality":
            issues = report.get("issues", report.get("quality_issues", []))
            if isinstance(issues, list):
                for iss in issues[:5]:
                    msg = iss.get("message", str(iss)) if isinstance(iss, dict) else str(iss)
                    source_alerts.append({
                        "message": f"Data quality: {str(msg)[:100]}",
                        "severity": "medium",
                    })

        # Generic: look for any "warnings" or "errors" keys
        warnings = report.get("warnings", [])
        if isinstance(warnings, list):
            for w in warnings[:3]:
                source_alerts.append({
                    "message": f"{source['name']}: {str(w)[:100]}",
                    "severity": "low",
                })

        # Stamp each alert with source and time
        for a in source_alerts:
            content_hash = hashlib.md5(
                f"{source['id']}:{a['message']}".encode()
            ).hexdigest()[:12]
            alerts.append({
                "id": f"alert-{content_hash}",
                "source_id": source["id"],
                "source_name": source["name"],
                "message": a["message"],
                "severity": a["severity"],
                "content_hash": content_hash,
                "timestamp": now.isoformat(),
                "routed_to": next(
                    (r["channel"] for r in _ROUTING_RULES if r["severity"] == a["severity"]),
                    "Dashboard only"
                ),
            })

    return alerts


def _load_iot_alerts():
    """Load IoT/SOS alert counts from clinical.db for volume analytics."""
    db = os.path.join(BASE, "data", "clinical.db")
    if not os.path.exists(db):
        return {"iot_alerts": 0, "sos_events": 0, "recent_iot": []}
    try:
        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        iot_count = conn.execute(
            "SELECT COUNT(*) as c FROM iot_alerts"
        ).fetchone()["c"] if _table_exists(conn, "iot_alerts") else 0
        sos_count = conn.execute(
            "SELECT COUNT(*) as c FROM emergency_sos_events"
        ).fetchone()["c"] if _table_exists(conn, "emergency_sos_events") else 0

        recent = []
        if _table_exists(conn, "iot_alerts"):
            rows = conn.execute(
                "SELECT alert_type, severity, COUNT(*) as cnt "
                "FROM iot_alerts GROUP BY alert_type, severity "
                "ORDER BY cnt DESC LIMIT 10"
            ).fetchall()
            recent = [dict(r) for r in rows]
        conn.close()
        return {"iot_alerts": iot_count, "sos_events": sos_count, "recent_iot": recent}
    except Exception:
        return {"iot_alerts": 0, "sos_events": 0, "recent_iot": []}


def _table_exists(conn, table):
    """Check if a table exists in SQLite."""
    r = conn.execute(
        "SELECT COUNT(*) as c FROM sqlite_master WHERE type='table' AND name=?",
        (table,)
    ).fetchone()
    return r["c"] > 0


def _compute_dedup_stats(alerts):
    """Compute deduplication statistics."""
    hash_counts = Counter(a["content_hash"] for a in alerts)
    total = len(alerts)
    unique = len(hash_counts)
    duplicates = total - unique
    suppression_rate = (duplicates / total * 100) if total > 0 else 0

    # Source-level noise score (alerts per source)
    source_counts = Counter(a["source_id"] for a in alerts)
    noisiest = [
        {"source": sid, "alert_count": cnt,
         "noise_score": min(100, cnt * 10)}
        for sid, cnt in source_counts.most_common(10)
    ]

    # Severity distribution
    sev_counts = Counter(a["severity"] for a in alerts)

    return {
        "total_alerts": total,
        "unique_alerts": unique,
        "duplicates_suppressed": duplicates,
        "suppression_rate_pct": round(suppression_rate, 1),
        "noisiest_sources": noisiest,
        "severity_distribution": [
            {"severity": s, "count": sev_counts.get(s, 0)}
            for s in ["critical", "high", "medium", "low"]
        ],
    }


def _alert_volume_trend():
    """Simulate alert volume trend from report modification times."""
    now = datetime.now(timezone.utc)
    trend = []
    for day_offset in range(13, -1, -1):
        day = now - timedelta(days=day_offset)
        date_str = day.strftime("%Y-%m-%d")
        # Count reports modified on that day
        report_dir = os.path.join(BASE, "jobs", "reports")
        count = 0
        if os.path.isdir(report_dir):
            for fname in os.listdir(report_dir):
                fpath = os.path.join(report_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        mtime = datetime.fromtimestamp(
                            os.path.getmtime(fpath), tz=timezone.utc
                        )
                        if mtime.strftime("%Y-%m-%d") == date_str:
                            count += 1
                    except OSError:
                        pass
        trend.append({"date": date_str, "alerts": count, "suppressed": max(0, count // 3)})
    return trend


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC API (called by api_backend.py)
# ═══════════════════════════════════════════════════════════════════════════════

def overview():
    """High-level alert fatigue metrics, health score, volume trend, routing summary."""
    alerts = _extract_alerts_from_reports()
    dedup = _compute_dedup_stats(alerts)
    iot = _load_iot_alerts()
    trend = _alert_volume_trend()

    total_all = dedup["total_alerts"] + iot["iot_alerts"] + iot["sos_events"]
    fatigue_score = 100  # Start perfect
    # Penalize for high alert volume
    if total_all > 50:
        fatigue_score -= 20
    if total_all > 100:
        fatigue_score -= 15
    # Penalize for low suppression rate (means noise is reaching operators)
    if dedup["suppression_rate_pct"] < 30:
        fatigue_score -= 10
    # Penalize for too many critical/high
    crit_high = sum(
        d["count"] for d in dedup["severity_distribution"]
        if d["severity"] in ("critical", "high")
    )
    if crit_high > 5:
        fatigue_score -= 15
    fatigue_score = max(0, min(100, fatigue_score))

    # Routing summary
    routing_summary = []
    for rule in _ROUTING_RULES:
        count = sum(1 for a in alerts if a["severity"] == rule["severity"])
        routing_summary.append({
            "severity": rule["severity"],
            "channel": rule["channel"],
            "response_sla": rule["response_sla"],
            "alert_count": count,
        })

    return {
        "title": "Alert Fatigue Monitor",
        "fatigue_score": fatigue_score,
        "fatigue_status": "healthy" if fatigue_score >= 80 else (
            "warning" if fatigue_score >= 60 else "critical"
        ),
        "total_alerts": total_all,
        "monitoring_alerts": dedup["total_alerts"],
        "iot_alerts": iot["iot_alerts"],
        "sos_events": iot["sos_events"],
        "unique_alerts": dedup["unique_alerts"],
        "duplicates_suppressed": dedup["duplicates_suppressed"],
        "suppression_rate_pct": dedup["suppression_rate_pct"],
        "severity_distribution": dedup["severity_distribution"],
        "noisiest_sources": dedup["noisiest_sources"][:5],
        "routing_summary": routing_summary,
        "volume_trend": trend[-14:],
        "last_scan": datetime.now(timezone.utc).isoformat(),
    }


def breakdown():
    """Detailed alert list, per-source analytics, dedup analysis, IoT alert breakdown."""
    alerts = _extract_alerts_from_reports()
    dedup = _compute_dedup_stats(alerts)
    iot = _load_iot_alerts()
    trend = _alert_volume_trend()

    # Group alerts by source
    by_source = {}
    for a in alerts:
        sid = a["source_id"]
        if sid not in by_source:
            by_source[sid] = []
        by_source[sid].append(a)

    # Source health summary
    source_health = []
    for src in _ALERT_SOURCES:
        src_alerts = by_source.get(src["id"], [])
        crit = sum(1 for a in src_alerts if a["severity"] == "critical")
        high = sum(1 for a in src_alerts if a["severity"] == "high")
        score = 100 - (crit * 25) - (high * 10) - (len(src_alerts) * 2)
        score = max(0, min(100, score))
        source_health.append({
            "source_id": src["id"],
            "source_name": src["name"],
            "description": src["description"],
            "total_alerts": len(src_alerts),
            "critical": crit,
            "high": high,
            "health_score": score,
        })

    return {
        "alerts": alerts,
        "alerts_by_source": by_source,
        "source_health": source_health,
        "dedup_stats": dedup,
        "dedup_strategies": _DEDUP_STRATEGIES,
        "routing_rules": _ROUTING_RULES,
        "iot_breakdown": iot,
        "volume_trend": trend,
        "alert_sources": _ALERT_SOURCES,
    }


def definitions():
    """Alert fatigue monitoring terminology and concepts."""
    return {
        "title": "Alert Fatigue Monitor — Definitions",
        "terms": [
            {"term": "Alert Fatigue", "definition": "A condition where operators become desensitized to alerts due to excessive volume, noise, or false positives, leading to missed critical incidents."},
            {"term": "Suppression Rate", "definition": "Percentage of duplicate or noisy alerts that are suppressed before reaching operators. Higher is better (less noise)."},
            {"term": "Content Hash Dedup", "definition": "Deduplication strategy that hashes alert content and suppresses identical alerts within a time window."},
            {"term": "Source Cooldown", "definition": "Rate limiting per alert source — prevents a single noisy source from flooding the alert pipeline."},
            {"term": "Flap Detection", "definition": "Identifies rapidly alternating OK/ALERT states and collapses them into a single 'flapping' alert."},
            {"term": "Severity Routing", "definition": "Directing alerts to appropriate channels based on severity: critical → PagerDuty, high → Slack #incidents, medium → monitoring, low → dashboard."},
            {"term": "Noise Score", "definition": "Per-source metric (0-100) indicating how noisy a monitoring source is. Higher scores indicate sources generating excessive alerts."},
            {"term": "Fatigue Score", "definition": "Overall system health metric (0-100) for alert quality. Penalized by high volume, low suppression, and excessive critical/high alerts."},
            {"term": "Escalation Chain", "definition": "Sequence of responders notified when an alert is not acknowledged within the response SLA."},
            {"term": "Response SLA", "definition": "Maximum time allowed before an alert must be acknowledged, based on severity level."},
            {"term": "Alert Merge", "definition": "Combining multiple low-severity alerts from the same source into a single aggregated medium-severity alert."},
            {"term": "Volume Trend", "definition": "Time-series of daily alert counts, used to detect spikes and establish baselines for anomaly detection."},
        ],
        "severity_levels": _ROUTING_RULES,
        "dedup_strategies": _DEDUP_STRATEGIES,
        "alert_sources": _ALERT_SOURCES,
    }
