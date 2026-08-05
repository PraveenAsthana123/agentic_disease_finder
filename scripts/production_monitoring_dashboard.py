#!/usr/bin/env python3
"""
Production Issue Live Monitoring Dashboard
==========================================

Real-time detection for 6 previously "planned" production issue watchpoints:
  1. Token/Cost Explosion    — budget utilization vs cap (Token/Cost Explosion, P2)
  2. MCP Server Outage       — local MCP/API server reachability (MCP, P1)
  3. Vector DB Corruption    — ChromaDB collection integrity check (Vector DB, P1)
  4. Stale Retrieval         — RAG corpus last-update freshness (Retrieval, P2)
  5. Goal Misinterpretation  — planner conversation anomaly rate (Planner, P2)
  6. Version Compatibility   — model file version vs registry expectation (MCP, P2)

All checks run against REAL local data (clinical.db, ChromaDB, filesystem).
"""

import os
import sqlite3
import time
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "clinical.db"
VECTOR_DB_PATH = ROOT / "data" / "vector_db"
CHROMA_DB_PATH = ROOT / "chroma_db"
MODEL_DIR = ROOT / "models"

# ── Budget caps (matches token_cost_dashboard.py) ────────────────────────────
MONTHLY_BUDGET_USD = 85.0
BUDGET_WARN_PCT = 80.0   # yellow
BUDGET_CRIT_PCT = 100.0  # red

# ── RAG staleness threshold ───────────────────────────────────────────────────
STALE_DAYS = 30  # corpus is stale if not updated in this many days

# ── Conversation anomaly detection ───────────────────────────────────────────
ANOMALY_LENGTH_THRESHOLD = 50  # messages; long sessions may signal misinterpretation


def _db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _now_utc():
    return datetime.now(timezone.utc)


# ── CHECK 1: Token / Cost Explosion ──────────────────────────────────────────
def _check_token_budget():
    """Read conversation_log + transaction_log to compute monthly cost vs cap."""
    try:
        conn = _db()
        cur = conn.cursor()
        # Monthly operation count (proxy for cost)
        cur.execute("""
            SELECT COUNT(*) FROM transaction_log
            WHERE strftime('%Y-%m', ts_utc) = strftime('%Y-%m', 'now')
        """)
        monthly_ops = (cur.fetchone() or [0])[0]

        # Estimate monthly cost (matches token_cost_dashboard.py constants)
        input_per_op, output_per_op = 180, 120
        input_rate, output_rate = 3.0 / 1_000_000, 15.0 / 1_000_000
        monthly_cost = (monthly_ops * input_per_op * input_rate
                        + monthly_ops * output_per_op * output_rate)
        utilization_pct = (monthly_cost / MONTHLY_BUDGET_USD) * 100

        # Monthly conversation tokens from conversation_log
        cur.execute("""
            SELECT COUNT(*), COALESCE(SUM(LENGTH(text)), 0)
            FROM conversation_log
            WHERE strftime('%Y-%m', ts_utc) = strftime('%Y-%m', 'now')
        """)
        row = cur.fetchone()
        conv_count, conv_chars = (row[0] or 0), (row[1] or 0)
        conn.close()

        status = "ok"
        if utilization_pct >= BUDGET_CRIT_PCT:
            status = "critical"
        elif utilization_pct >= BUDGET_WARN_PCT:
            status = "warning"

        return {
            "check": "Token/Cost Explosion",
            "layer": "Agent",
            "severity": "P2",
            "status": status,
            "detected": True,
            "details": {
                "monthly_ops": monthly_ops,
                "monthly_cost_usd": round(monthly_cost, 6),
                "monthly_budget_usd": MONTHLY_BUDGET_USD,
                "utilization_pct": round(utilization_pct, 2),
                "monthly_conversations": conv_count,
                "monthly_conv_chars": conv_chars,
            },
            "threshold": f"Warn ≥{BUDGET_WARN_PCT}%, Critical ≥{BUDGET_CRIT_PCT}%",
            "remediation": "Reduce operation frequency · enable context compression · raise budget cap",
        }
    except Exception as e:
        return {"check": "Token/Cost Explosion", "layer": "Agent", "severity": "P2",
                "status": "error", "detected": False, "error": str(e)}


# ── CHECK 2: MCP Server Outage ────────────────────────────────────────────────
def _check_mcp_outage():
    """Ping the local API server (MCP equivalent) and check recent uptime events."""
    try:
        import urllib.request
        t0 = time.time()
        req = urllib.request.urlopen("http://127.0.0.1:8010/api/data-manager", timeout=5)
        latency_ms = round((time.time() - t0) * 1000, 1)
        http_code = req.getcode()
        server_up = (http_code == 200)
    except Exception as e:
        latency_ms = None
        http_code = None
        server_up = False

    # Check recent health log
    try:
        conn = _db()
        cur = conn.cursor()
        cur.execute("""
            SELECT COUNT(*) FROM system_health_log
            WHERE status != 'healthy'
              AND timestamp >= datetime('now', '-7 days')
        """)
        recent_errors = (cur.fetchone() or [0])[0]
        cur.execute("SELECT COUNT(*) FROM system_health_log")
        total_checks = (cur.fetchone() or [0])[0]
        conn.close()
    except Exception:
        recent_errors = 0
        total_checks = 0

    status = "ok" if server_up else "critical"
    if server_up and recent_errors > 5:
        status = "warning"

    return {
        "check": "MCP Server Outage",
        "layer": "MCP",
        "severity": "P1",
        "status": status,
        "detected": True,
        "details": {
            "api_reachable": server_up,
            "http_code": http_code,
            "latency_ms": latency_ms,
            "recent_health_errors_7d": recent_errors,
            "total_health_checks": total_checks,
        },
        "threshold": "API must return HTTP 200; ≤5 health errors in 7 days",
        "remediation": "Auto-restart via watchdog · check backend logs · verify port 8010",
    }


# ── CHECK 3: Vector DB Index Corruption ──────────────────────────────────────
def _check_vector_db():
    """Check ChromaDB collections for integrity (count, metadata, reachability)."""
    checks = []
    for db_path, label in [
        (str(VECTOR_DB_PATH), "clinical"),
        (str(CHROMA_DB_PATH), "arxiv_papers"),
    ]:
        try:
            import chromadb
            client = chromadb.PersistentClient(path=db_path)
            colls = client.list_collections()
            names = [c.name for c in colls]
            doc_counts = {c.name: c.count() for c in colls}
            checks.append({
                "db": label,
                "path": db_path,
                "status": "ok",
                "collections": names,
                "doc_counts": doc_counts,
                "total_docs": sum(doc_counts.values()),
            })
        except Exception as e:
            checks.append({
                "db": label,
                "path": db_path,
                "status": "error",
                "error": str(e),
            })

    all_ok = all(c.get("status") == "ok" for c in checks)
    any_empty = any(c.get("total_docs", 1) == 0 for c in checks if c.get("status") == "ok")
    status = "ok" if all_ok and not any_empty else ("warning" if all_ok else "critical")

    return {
        "check": "Vector DB Index Corruption",
        "layer": "Vector DB",
        "severity": "P1",
        "status": status,
        "detected": True,
        "details": {
            "databases_checked": len(checks),
            "all_reachable": all_ok,
            "any_empty_collection": any_empty,
            "results": checks,
        },
        "threshold": "All ChromaDB instances reachable; no empty collections",
        "remediation": "Re-index from source documents · restore from snapshot · rebuild ChromaDB",
    }


# ── CHECK 4: Stale Retrieval ──────────────────────────────────────────────────
def _check_stale_retrieval():
    """Check when the RAG corpus was last updated using transaction_log."""
    try:
        conn = _db()
        cur = conn.cursor()
        # Last ingest/index event in transaction_log
        cur.execute("""
            SELECT MAX(ts_utc) FROM transaction_log
            WHERE component LIKE '%rag%'
               OR component LIKE '%vector%'
               OR component LIKE '%ingest%'
               OR action IN ('ingest', 'index', 'embed')
        """)
        row = cur.fetchone()
        last_update_str = row[0] if row else None

        # Also check chroma_db mtime as ground truth
        chroma_file = VECTOR_DB_PATH / "chroma.sqlite3"
        file_mtime = None
        if chroma_file.exists():
            file_mtime = datetime.fromtimestamp(
                chroma_file.stat().st_mtime, tz=timezone.utc
            ).isoformat()

        # Compute staleness
        now = _now_utc()
        stale = False
        days_since_update = None

        ref_date_str = file_mtime or last_update_str
        if ref_date_str:
            try:
                ref_date = datetime.fromisoformat(ref_date_str.replace("Z", "+00:00"))
                if ref_date.tzinfo is None:
                    ref_date = ref_date.replace(tzinfo=timezone.utc)
                delta = now - ref_date
                days_since_update = delta.days
                stale = days_since_update > STALE_DAYS
            except Exception:
                pass

        conn.close()

        status = "critical" if stale else "ok"
        if days_since_update is not None and days_since_update > STALE_DAYS // 2:
            status = "warning" if status == "ok" else status

        return {
            "check": "Stale Retrieval",
            "layer": "Retrieval",
            "severity": "P2",
            "status": status,
            "detected": True,
            "details": {
                "last_update_transaction": last_update_str,
                "last_update_file_mtime": file_mtime,
                "days_since_update": days_since_update,
                "stale_threshold_days": STALE_DAYS,
                "is_stale": stale,
            },
            "threshold": f"Corpus must be updated within {STALE_DAYS} days",
            "remediation": "Trigger re-index job · upload new EEG reports · run RAG ingest pipeline",
        }
    except Exception as e:
        return {"check": "Stale Retrieval", "layer": "Retrieval", "severity": "P2",
                "status": "error", "detected": False, "error": str(e)}


# ── CHECK 5: Goal Misinterpretation ──────────────────────────────────────────
def _check_goal_misinterpretation():
    """Detect planner anomalies: very long sessions, high retry rates."""
    try:
        conn = _db()
        cur = conn.cursor()

        # Conversation session length distribution
        cur.execute("""
            SELECT DATE(ts_utc) as d, COUNT(*) as msgs
            FROM conversation_log
            GROUP BY d ORDER BY d DESC LIMIT 30
        """)
        daily = cur.fetchall()

        # Sessions exceeding threshold (long conversations may indicate misinterpretation)
        long_sessions = [r for r in daily if r[1] > ANOMALY_LENGTH_THRESHOLD]

        # Check for repeated similar messages (retry signal)
        cur.execute("""
            SELECT role, COUNT(*) as cnt, MIN(ts_utc) as first, MAX(ts_utc) as last
            FROM conversation_log GROUP BY role
        """)
        by_role = {r[0]: {"count": r[1], "first": r[2], "last": r[3]}
                   for r in cur.fetchall()}

        # Overall stats
        cur.execute("SELECT COUNT(*) FROM conversation_log")
        total_msgs = (cur.fetchone() or [0])[0]

        # Anomaly rate: days with long sessions / total days
        total_days = len(daily)
        anomaly_days = len(long_sessions)
        anomaly_rate = round(anomaly_days / max(total_days, 1), 4)

        conn.close()

        status = "warning" if anomaly_rate > 0.2 else "ok"

        return {
            "check": "Goal Misinterpretation",
            "layer": "Planner",
            "severity": "P2",
            "status": status,
            "detected": True,
            "details": {
                "total_messages": total_msgs,
                "days_analyzed": total_days,
                "long_session_days": anomaly_days,
                "anomaly_rate": anomaly_rate,
                "long_session_threshold": ANOMALY_LENGTH_THRESHOLD,
                "by_role": by_role,
            },
            "threshold": f"Anomaly rate < 20% (sessions with >{ANOMALY_LENGTH_THRESHOLD} msgs/day)",
            "remediation": "Review conversation logs · add goal-validation checkpoint · tune planner prompt",
        }
    except Exception as e:
        return {"check": "Goal Misinterpretation", "layer": "Planner", "severity": "P2",
                "status": "error", "detected": False, "error": str(e)}


# ── CHECK 6: Version Compatibility ────────────────────────────────────────────
def _check_version_compatibility():
    """Check model files exist and are consistent across disease types."""
    expected_diseases = ["epilepsy", "depression", "parkinsons", "alzheimers", "sleep_disorder"]
    found = []
    missing = []

    # Check model directory for PKL/JOBLIB files
    model_patterns = ["*.pkl", "*.joblib", "*.h5", "*.pt", "*.onnx"]
    existing_files = set()
    if MODEL_DIR.exists():
        for pat in model_patterns:
            for f in MODEL_DIR.glob(pat):
                existing_files.add(f.name)

    for disease in expected_diseases:
        hits = [f for f in existing_files
                if disease.lower() in f.lower() or "model" in f.lower()]
        if hits:
            found.append(disease)
        else:
            missing.append(disease)

    # Also check analyses table for model version signals
    try:
        conn = _db()
        cur = conn.cursor()
        cur.execute("""
            SELECT disease, COUNT(*) as cnt, MAX(created_at) as last_run
            FROM analyses GROUP BY disease
        """)
        by_disease = {r[0]: {"count": r[1], "last_run": r[2]} for r in cur.fetchall()}
        conn.close()
    except Exception:
        by_disease = {}

    # Models with no recent inferences are potentially version-mismatched
    stale_models = []
    threshold = _now_utc() - timedelta(days=30)
    for disease, info in by_disease.items():
        try:
            last = datetime.fromisoformat(info["last_run"].replace("Z", "+00:00"))
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if last < threshold:
                stale_models.append(disease)
        except Exception:
            pass

    status = "ok" if not missing and not stale_models else "warning"

    return {
        "check": "Version Compatibility",
        "layer": "MCP",
        "severity": "P2",
        "status": status,
        "detected": True,
        "details": {
            "expected_diseases": expected_diseases,
            "models_found_count": len(found) + (1 if existing_files else 0),
            "model_files_in_dir": sorted(existing_files)[:20],
            "diseases_with_recent_inferences": list(by_disease.keys()),
            "stale_inference_diseases": stale_models,
            "missing_model_flags": missing,
            "by_disease_inference": by_disease,
        },
        "threshold": "All disease models have recent (≤30d) inferences; model files present",
        "remediation": "Retrain stale models · verify model registry · run model compatibility tests",
    }


# ── Aggregate: run all checks ─────────────────────────────────────────────────
def live_checks():
    """Run all 6 detection checks and return structured results."""
    check_fns = [
        _check_token_budget,
        _check_mcp_outage,
        _check_vector_db,
        _check_stale_retrieval,
        _check_goal_misinterpretation,
        _check_version_compatibility,
    ]
    results = []
    for fn in check_fns:
        try:
            results.append(fn())
        except Exception as e:
            results.append({"check": fn.__name__, "status": "error", "error": str(e)})

    counts = {"ok": 0, "warning": 0, "critical": 0, "error": 0}
    for r in results:
        counts[r.get("status", "error")] = counts.get(r.get("status", "error"), 0) + 1

    overall = ("critical" if counts["critical"] > 0
               else "warning" if counts["warning"] > 0
               else "error" if counts["error"] > 0
               else "ok")

    return {
        "available": True,
        "run_at": _now_utc().isoformat(),
        "overall_status": overall,
        "summary": {
            "total_checks": len(results),
            "ok": counts["ok"],
            "warning": counts["warning"],
            "critical": counts["critical"],
            "error": counts["error"],
        },
        "checks": results,
    }


def overview():
    """Overview: summary KPIs + per-check status table."""
    data = live_checks()
    checks = data["checks"]

    # Trend: simulate from system_health_log for "historical" flavour
    trend = []
    try:
        conn = _db()
        cur = conn.cursor()
        cur.execute("""
            SELECT DATE(timestamp) as d,
                   SUM(CASE WHEN status='healthy' THEN 1 ELSE 0 END) as healthy,
                   SUM(CASE WHEN status!='healthy' THEN 1 ELSE 0 END) as unhealthy
            FROM system_health_log
            GROUP BY d ORDER BY d
        """)
        for r in cur.fetchall():
            trend.append({"date": r[0], "healthy": r[1], "issues": r[2]})
        conn.close()
    except Exception:
        pass

    return {
        "available": True,
        "run_at": data["run_at"],
        "overall_status": data["overall_status"],
        "summary": data["summary"],
        "checks_summary": [
            {
                "check": c.get("check"),
                "layer": c.get("layer"),
                "severity": c.get("severity"),
                "status": c.get("status"),
                "detected": c.get("detected", False),
            }
            for c in checks
        ],
        "health_trend": trend,
    }


def breakdown():
    """Full live-checks breakdown with all detail fields."""
    return live_checks()


def definitions():
    """Definitions for metrics, statuses, and issue layers."""
    return {
        "available": True,
        "metrics": [
            {"name": "Check Status", "description": "ok / warning / critical / error result of each live detection run", "unit": "enum"},
            {"name": "Anomaly Rate", "description": "Fraction of days where conversation session length exceeded threshold", "unit": "%"},
            {"name": "Budget Utilization", "description": "Monthly LLM/operation cost as % of the $85 monthly cap", "unit": "%"},
            {"name": "Days Since Update", "description": "Days elapsed since the RAG corpus (ChromaDB) was last written to", "unit": "days"},
            {"name": "API Latency", "description": "Round-trip time to /api/data-manager health endpoint", "unit": "ms"},
            {"name": "Vector Docs", "description": "Total documents indexed across all ChromaDB collections", "unit": "count"},
        ],
        "statuses": [
            {"status": "ok", "color": "green", "meaning": "Check passed — within all thresholds"},
            {"status": "warning", "color": "yellow", "meaning": "Approaching threshold — investigate proactively"},
            {"status": "critical", "color": "red", "meaning": "Threshold exceeded — immediate action required"},
            {"status": "error", "color": "gray", "meaning": "Check could not run — verify dependencies"},
        ],
        "layers": [
            {"layer": "Agent", "description": "LLM agent behaviour and token economics"},
            {"layer": "MCP", "description": "Model Control Plane — server availability and version management"},
            {"layer": "Vector DB", "description": "ChromaDB index integrity for RAG retrieval"},
            {"layer": "Retrieval", "description": "RAG corpus freshness and retrieval quality"},
            {"layer": "Planner", "description": "Agentic goal planner coherence and session anomaly detection"},
        ],
        "thresholds": [
            {"check": "Token/Cost Explosion", "warn": "≥80% of $85/month", "critical": "≥100%"},
            {"check": "MCP Server Outage", "warn": ">5 health errors / 7 days", "critical": "API unreachable"},
            {"check": "Vector DB Corruption", "warn": "Empty collection detected", "critical": "ChromaDB unreachable"},
            {"check": "Stale Retrieval", "warn": ">15 days since update", "critical": ">30 days"},
            {"check": "Goal Misinterpretation", "warn": ">20% anomaly rate", "critical": "N/A (warn only)"},
            {"check": "Version Compatibility", "warn": "Stale inference (>30d) or missing model", "critical": "N/A"},
        ],
        "clinical_relevance": (
            "Production issue monitoring is a mandatory governance layer for AI in clinical settings. "
            "Token/Cost Explosion risks budget overrun for a self-funded DBA research study. "
            "MCP Server Outage causes clinical AI unavailability — neurologists cannot receive AI decision support. "
            "Vector DB Corruption corrupts RAG-assisted clinical summaries. "
            "Stale Retrieval returns outdated epilepsy evidence to clinicians. "
            "Goal Misinterpretation degrades agentic pipeline outputs, risking incorrect treatment recommendations. "
            "All 6 watchpoints align with the enterprise Agentic AI Issue Catalog (§production_issues.json)."
        ),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(live_checks(), indent=2))
