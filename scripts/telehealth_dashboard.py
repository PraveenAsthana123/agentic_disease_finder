#!/usr/bin/env python3
"""
Telehealth Sessions Dashboard
==============================

Analyses REAL telehealth session data from clinical.db:

  1. **telehealth_sessions** — session_date, session_type, provider_name,
     duration_minutes, connection_quality, patient_satisfaction,
     technical_issues, platform, notes

The dashboard visualises telehealth utilisation and quality:
  session volume → provider workload → platform reliability → patient satisfaction

Functions:
  overview()     — KPIs, session-type distribution, platform usage, satisfaction trends
  breakdown()    — per-provider stats, per-patient session history, quality metrics
  definitions()  — glossary, session types, quality thresholds, clinical references
"""

import json
import math
import os
import sqlite3
from collections import Counter, defaultdict
from typing import Any, Dict, List

_BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
_DB_PATH = os.path.join(_BASE_DIR, "data", "clinical.db")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _conn():
    c = sqlite3.connect(_DB_PATH)
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    if not os.path.exists(_DB_PATH):
        return []
    conn = _conn()
    try:
        return [dict(r) for r in conn.execute(query, params).fetchall()]
    except Exception:
        return []
    finally:
        conn.close()


def _avg(vals):
    return round(sum(vals) / len(vals), 2) if vals else 0.0


def _std(vals):
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return round(math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1)), 2)


def _pct(num, denom):
    return round(num / denom, 4) if denom else 0.0


def _safe_float(val, default=0.0):
    """Convert value to float, returning default for None/NaN/Inf."""
    if val is None:
        return default
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# overview — fleet-level KPIs + distributions
# ---------------------------------------------------------------------------

def overview() -> Dict[str, Any]:
    sessions = _rows("SELECT * FROM telehealth_sessions ORDER BY session_date DESC")
    if not sessions:
        return {"kpis": {}, "session_type_distribution": [], "platform_distribution": [],
                "quality_distribution": [], "satisfaction_histogram": [],
                "monthly_trend": [], "technical_issue_rate": 0}

    total = len(sessions)
    patients = len(set(s["patient_id"] for s in sessions))
    providers = len(set(s["provider_name"] for s in sessions))
    durations = [_safe_float(s["duration_minutes"]) for s in sessions]
    satisfactions = [_safe_float(s["patient_satisfaction"]) for s in sessions if s["patient_satisfaction"] is not None]
    tech_issues = sum(1 for s in sessions if _safe_float(s["technical_issues"]) > 0)

    # Session type distribution
    type_counts = Counter(s["session_type"] for s in sessions)
    session_type_dist = [{"name": t, "count": c, "pct": _pct(c, total)}
                         for t, c in type_counts.most_common()]

    # Platform distribution
    plat_counts = Counter(s["platform"] for s in sessions)
    platform_dist = [{"name": p, "count": c, "pct": _pct(c, total)}
                     for p, c in plat_counts.most_common()]

    # Connection quality distribution
    qual_counts = Counter(s["connection_quality"] for s in sessions)
    quality_order = ["excellent", "good", "fair", "poor"]
    quality_dist = []
    for q in quality_order:
        if q in qual_counts:
            quality_dist.append({"name": q, "count": qual_counts[q], "pct": _pct(qual_counts[q], total)})

    # Satisfaction histogram (1-5)
    sat_counts = Counter(int(_safe_float(s["patient_satisfaction"])) for s in sessions if s["patient_satisfaction"] is not None)
    satisfaction_hist = [{"rating": r, "count": sat_counts.get(r, 0)} for r in range(1, 6)]

    # Monthly trend
    monthly = defaultdict(lambda: {"count": 0, "durations": [], "satisfactions": [], "issues": 0})
    for s in sessions:
        month = s["session_date"][:7] if s["session_date"] else "unknown"
        monthly[month]["count"] += 1
        monthly[month]["durations"].append(_safe_float(s["duration_minutes"]))
        if s["patient_satisfaction"] is not None:
            monthly[month]["satisfactions"].append(_safe_float(s["patient_satisfaction"]))
        if _safe_float(s["technical_issues"]) > 0:
            monthly[month]["issues"] += 1

    monthly_trend = []
    for m in sorted(monthly.keys()):
        d = monthly[m]
        monthly_trend.append({
            "month": m,
            "sessions": d["count"],
            "avg_duration": _avg(d["durations"]),
            "avg_satisfaction": _avg(d["satisfactions"]),
            "issue_count": d["issues"],
        })

    kpis = {
        "total_sessions": total,
        "unique_patients": patients,
        "unique_providers": providers,
        "avg_duration_min": _avg(durations),
        "avg_satisfaction": _avg(satisfactions),
        "technical_issue_rate": _pct(tech_issues, total),
        "sessions_with_issues": tech_issues,
    }

    return {
        "kpis": kpis,
        "session_type_distribution": session_type_dist,
        "platform_distribution": platform_dist,
        "quality_distribution": quality_dist,
        "satisfaction_histogram": satisfaction_hist,
        "monthly_trend": monthly_trend,
    }


# ---------------------------------------------------------------------------
# breakdown — per-provider, per-patient, per-platform detail
# ---------------------------------------------------------------------------

def breakdown() -> Dict[str, Any]:
    sessions = _rows("SELECT * FROM telehealth_sessions ORDER BY session_date DESC")
    if not sessions:
        return {"provider_stats": [], "patient_sessions": [], "platform_quality": [],
                "recent_sessions": []}

    # Per-provider stats
    prov = defaultdict(lambda: {"sessions": 0, "durations": [], "satisfactions": [], "issues": 0, "patients": set()})
    for s in sessions:
        p = s["provider_name"] or "Unknown"
        prov[p]["sessions"] += 1
        prov[p]["durations"].append(_safe_float(s["duration_minutes"]))
        if s["patient_satisfaction"] is not None:
            prov[p]["satisfactions"].append(_safe_float(s["patient_satisfaction"]))
        if _safe_float(s["technical_issues"]) > 0:
            prov[p]["issues"] += 1
        prov[p]["patients"].add(s["patient_id"])

    provider_stats = []
    for name in sorted(prov.keys()):
        d = prov[name]
        provider_stats.append({
            "provider": name,
            "sessions": d["sessions"],
            "patients": len(d["patients"]),
            "avg_duration": _avg(d["durations"]),
            "avg_satisfaction": _avg(d["satisfactions"]),
            "issue_rate": _pct(d["issues"], d["sessions"]),
        })
    provider_stats.sort(key=lambda x: x["sessions"], reverse=True)

    # Per-patient session summary
    pat = defaultdict(lambda: {"sessions": 0, "types": [], "satisfactions": [], "last_date": ""})
    for s in sessions:
        pid = s["patient_id"]
        pat[pid]["sessions"] += 1
        pat[pid]["types"].append(s["session_type"])
        if s["patient_satisfaction"] is not None:
            pat[pid]["satisfactions"].append(_safe_float(s["patient_satisfaction"]))
        if not pat[pid]["last_date"] or s["session_date"] > pat[pid]["last_date"]:
            pat[pid]["last_date"] = s["session_date"]

    patient_sessions = []
    for pid in sorted(pat.keys()):
        d = pat[pid]
        type_counts = Counter(d["types"])
        patient_sessions.append({
            "patient_id": pid,
            "total_sessions": d["sessions"],
            "avg_satisfaction": _avg(d["satisfactions"]),
            "last_session": d["last_date"],
            "primary_type": type_counts.most_common(1)[0][0] if type_counts else "N/A",
        })
    patient_sessions.sort(key=lambda x: x["total_sessions"], reverse=True)

    # Platform quality comparison
    plat = defaultdict(lambda: {"sessions": 0, "qualities": [], "satisfactions": [], "issues": 0, "durations": []})
    for s in sessions:
        p = s["platform"] or "Unknown"
        plat[p]["sessions"] += 1
        plat[p]["qualities"].append(s["connection_quality"] or "unknown")
        if s["patient_satisfaction"] is not None:
            plat[p]["satisfactions"].append(_safe_float(s["patient_satisfaction"]))
        if _safe_float(s["technical_issues"]) > 0:
            plat[p]["issues"] += 1
        plat[p]["durations"].append(_safe_float(s["duration_minutes"]))

    quality_score_map = {"excellent": 4, "good": 3, "fair": 2, "poor": 1}
    platform_quality = []
    for name in sorted(plat.keys()):
        d = plat[name]
        qscores = [quality_score_map.get(q, 0) for q in d["qualities"]]
        platform_quality.append({
            "platform": name,
            "sessions": d["sessions"],
            "avg_quality_score": _avg(qscores),
            "avg_satisfaction": _avg(d["satisfactions"]),
            "avg_duration": _avg(d["durations"]),
            "issue_rate": _pct(d["issues"], d["sessions"]),
        })
    platform_quality.sort(key=lambda x: x["sessions"], reverse=True)

    # Recent sessions table (top 20)
    recent = []
    for s in sessions[:20]:
        recent.append({
            "patient_id": s["patient_id"],
            "date": s["session_date"],
            "type": s["session_type"],
            "provider": s["provider_name"],
            "duration_min": s["duration_minutes"],
            "quality": s["connection_quality"],
            "satisfaction": s["patient_satisfaction"],
            "issues": s["technical_issues"],
            "platform": s["platform"],
            "notes": s["notes"],
        })

    return {
        "provider_stats": provider_stats,
        "patient_sessions": patient_sessions,
        "platform_quality": platform_quality,
        "recent_sessions": recent,
    }


# ---------------------------------------------------------------------------
# definitions — glossary and reference
# ---------------------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    return {
        "session_types": {
            "video-visit": "Live video consultation between provider and patient",
            "phone-consult": "Voice-only telephone consultation",
            "async-message": "Asynchronous messaging (secure text/email exchange)",
            "remote-monitoring-review": "Provider reviews remotely-collected patient data (EEG, wearable, diary)",
        },
        "connection_quality": {
            "excellent": "No interruptions, HD video/audio, latency < 100ms",
            "good": "Minor buffering, acceptable quality, latency 100-300ms",
            "fair": "Intermittent issues, reduced quality, latency 300-500ms",
            "poor": "Frequent drops, pixelated/choppy, latency > 500ms",
        },
        "satisfaction_scale": {
            "1": "Very dissatisfied",
            "2": "Dissatisfied",
            "3": "Neutral",
            "4": "Satisfied",
            "5": "Very satisfied",
        },
        "kpi_definitions": [
            {"kpi": "Total Sessions", "desc": "Count of all telehealth encounters in the system"},
            {"kpi": "Unique Patients", "desc": "Distinct patient IDs with at least one telehealth session"},
            {"kpi": "Avg Duration", "desc": "Mean session length in minutes across all encounters"},
            {"kpi": "Avg Satisfaction", "desc": "Mean patient-reported satisfaction (1-5 Likert scale)"},
            {"kpi": "Technical Issue Rate", "desc": "Fraction of sessions reporting at least one technical problem"},
        ],
        "clinical_references": [
            "AAN Practice Advisory: Telemedicine (2020)",
            "AES Guidelines: Telehealth for Epilepsy Management",
            "CMS Telehealth Policy: Eligible Services & Billing",
            "HIPAA Telehealth Compliance: Platform Security Requirements",
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN ===")
    pprint.pprint(breakdown())
    print("\n=== DEFINITIONS ===")
    pprint.pprint(definitions())
