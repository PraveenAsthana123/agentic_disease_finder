"""
Guided Assessment Flow Dashboard
=================================
Real data from clinical.db guided_assessment_sessions table — session
tracking, completion analytics, instrument usage, channel breakdown,
and per-patient guided assessment history.

This backs the Conversational AI guided-assessment feature: item-by-item
clinical assessments (PHQ-9, GAD-7, MoCA, etc.) delivered via chat or voice.

Data Sources:
  - guided_assessment_sessions  (45 rows, 30 patients)
  - patients                    — demographics
  - assessments                 — cross-reference with standard assessments

Author: Research Team
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict, Counter

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        return {}


def _fmt(val, decimals=1):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


# ── Overview ────────────────────────────────────────────────────────

def overview():
    """KPIs, status distribution, instrument usage, channel breakdown,
    completion trend, duration histogram."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = [dict(r) for r in cur.execute(
        "SELECT * FROM guided_assessment_sessions ORDER BY started_at"
    ).fetchall()]

    patients = {r["patient_id"] for r in rows}

    # Status counts
    status_counts = Counter(r["status"] for r in rows)
    completed = [r for r in rows if r["status"] == "completed"]
    in_progress = [r for r in rows if r["status"] == "in_progress"]
    abandoned = [r for r in rows if r["status"] == "abandoned"]

    # Completion rate
    total = len(rows)
    completion_rate = round(len(completed) / total * 100, 1) if total else 0

    # Avg duration for completed
    durations = [r["duration_seconds"] for r in completed if r["duration_seconds"]]
    avg_duration = round(sum(durations) / len(durations)) if durations else 0

    # Instrument distribution
    inst_counts = Counter(r["instrument"] for r in rows)
    instrument_dist = [{"name": k, "value": v} for k, v in
                       sorted(inst_counts.items(), key=lambda x: -x[1])]

    # Channel distribution
    chan_counts = Counter(r["channel"] for r in rows)
    channel_dist = [{"name": k, "value": v} for k, v in
                    sorted(chan_counts.items(), key=lambda x: -x[1])]

    # Status distribution for pie
    status_dist = [{"name": k, "value": v} for k, v in
                   sorted(status_counts.items(), key=lambda x: -x[1])]

    # Daily completion trend
    daily = defaultdict(lambda: {"completed": 0, "started": 0, "abandoned": 0})
    for r in rows:
        day = r["started_at"][:10] if r["started_at"] else "unknown"
        daily[day]["started"] += 1
        if r["status"] == "completed":
            daily[day]["completed"] += 1
        elif r["status"] == "abandoned":
            daily[day]["abandoned"] += 1
    daily_trend = [{"date": d, **v} for d, v in sorted(daily.items())]

    # Duration histogram (for completed sessions)
    dur_buckets = [
        ("< 2 min", 0, 120),
        ("2-5 min", 120, 300),
        ("5-8 min", 300, 480),
        ("8+ min", 480, 99999),
    ]
    dur_hist = []
    for label, lo, hi in dur_buckets:
        count = sum(1 for d in durations if lo <= d < hi)
        dur_hist.append({"label": label, "count": count})

    # Score distribution for completed
    scores = []
    for r in completed:
        if r["score"] is not None and r["max_score"]:
            pct = round(r["score"] / r["max_score"] * 100, 1)
            scores.append(pct)
    score_buckets = [
        ("0-25%", 0, 25), ("25-50%", 25, 50),
        ("50-75%", 50, 75), ("75-100%", 75, 101),
    ]
    score_dist = []
    for label, lo, hi in score_buckets:
        count = sum(1 for s in scores if lo <= s < hi)
        score_dist.append({"label": label, "count": count})

    kpis = [
        {"label": "Total Sessions", "value": str(total)},
        {"label": "Completed", "value": str(len(completed)), "color": "#10b981"},
        {"label": "In Progress", "value": str(len(in_progress)), "color": "#f59e0b"},
        {"label": "Abandoned", "value": str(len(abandoned)), "color": "#ef4444"},
        {"label": "Completion Rate", "value": f"{completion_rate}%", "color": "#3b82f6"},
        {"label": "Patients", "value": str(len(patients))},
        {"label": "Instruments", "value": str(len(inst_counts))},
        {"label": "Avg Duration", "value": f"{avg_duration // 60}m {avg_duration % 60}s",
         "color": "#8b5cf6"},
    ]

    conn.close()
    return {
        "available": True,
        "kpis": kpis,
        "status_distribution": status_dist,
        "instrument_distribution": instrument_dist,
        "channel_distribution": channel_dist,
        "daily_trend": daily_trend,
        "duration_histogram": dur_hist,
        "score_distribution": score_dist,
    }


# ── Breakdown ───────────────────────────────────────────────────────

def breakdown():
    """Per-session detail, per-instrument stats, per-patient history,
    active sessions, recently completed."""
    conn = _conn()
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    rows = [dict(r) for r in cur.execute(
        "SELECT g.*, p.name as patient_name, p.age, p.gender "
        "FROM guided_assessment_sessions g "
        "LEFT JOIN patients p ON g.patient_id = p.patient_id "
        "ORDER BY g.started_at DESC"
    ).fetchall()]

    # Session log (most recent 50)
    session_log = []
    for r in rows[:50]:
        progress = round(r["current_item"] / r["total_items"] * 100) if r["total_items"] else 0
        session_log.append({
            "session_id": r["session_id"],
            "patient_id": r["patient_id"],
            "patient_name": r.get("patient_name", ""),
            "instrument": r["instrument"],
            "status": r["status"],
            "progress": f"{progress}%",
            "current_item": r["current_item"],
            "total_items": r["total_items"],
            "score": _fmt(r["score"]) if r["score"] is not None else "-",
            "max_score": _fmt(r["max_score"]) if r["max_score"] is not None else "-",
            "interpretation": r["interpretation"] or "-",
            "channel": r["channel"],
            "started_at": r["started_at"],
            "completed_at": r["completed_at"] or "-",
            "duration": f"{r['duration_seconds'] // 60}m {r['duration_seconds'] % 60}s"
                        if r["duration_seconds"] else "-",
        })

    # Per-instrument stats
    inst_stats = defaultdict(lambda: {
        "total": 0, "completed": 0, "abandoned": 0, "in_progress": 0,
        "durations": [], "scores": [],
    })
    for r in rows:
        s = inst_stats[r["instrument"]]
        s["total"] += 1
        s[r["status"]] += 1
        if r["duration_seconds"] and r["status"] == "completed":
            s["durations"].append(r["duration_seconds"])
        if r["score"] is not None:
            s["scores"].append(r["score"])

    instrument_summary = []
    for name, s in sorted(inst_stats.items()):
        avg_dur = round(sum(s["durations"]) / len(s["durations"])) if s["durations"] else 0
        avg_score = round(sum(s["scores"]) / len(s["scores"]), 1) if s["scores"] else None
        comp_rate = round(s["completed"] / s["total"] * 100, 1) if s["total"] else 0
        instrument_summary.append({
            "instrument": name,
            "total": s["total"],
            "completed": s["completed"],
            "in_progress": s["in_progress"],
            "abandoned": s["abandoned"],
            "completion_rate": f"{comp_rate}%",
            "avg_duration": f"{avg_dur // 60}m {avg_dur % 60}s",
            "avg_score": _fmt(avg_score) if avg_score is not None else "-",
        })

    # Per-patient summary
    patient_stats = defaultdict(lambda: {"sessions": 0, "completed": 0, "instruments": set()})
    for r in rows:
        ps = patient_stats[r["patient_id"]]
        ps["sessions"] += 1
        ps["name"] = r.get("patient_name", r["patient_id"])
        if r["status"] == "completed":
            ps["completed"] += 1
        ps["instruments"].add(r["instrument"])

    patient_summary = []
    for pid, ps in sorted(patient_stats.items()):
        patient_summary.append({
            "patient_id": pid,
            "patient_name": ps["name"],
            "total_sessions": ps["sessions"],
            "completed": ps["completed"],
            "instruments_used": len(ps["instruments"]),
            "instruments": ", ".join(sorted(ps["instruments"])),
        })

    # Active sessions (in_progress)
    active = [s for s in session_log if s["status"] == "in_progress"]

    conn.close()
    return {
        "session_log": session_log,
        "instrument_summary": instrument_summary,
        "patient_summary": patient_summary,
        "active_sessions": active,
    }


# ── Definitions ─────────────────────────────────────────────────────

def definitions():
    """Metric definitions, instrument descriptions, guided flow concepts."""
    return {
        "concepts": [
            {"name": "Guided Assessment Flow",
             "description": "Item-by-item clinical assessment delivered conversationally via chat (LLM) or voice (STT). Each question is presented one at a time, with the system recording and scoring responses in real time."},
            {"name": "Session",
             "description": "A single guided assessment interaction — one patient completing one instrument through the conversational or voice channel."},
            {"name": "Completion Rate",
             "description": "Percentage of started sessions that reach full completion (all items answered and scored)."},
            {"name": "Abandonment",
             "description": "A session where the patient stopped responding before completing all items. May indicate assessment fatigue, technical issues, or clinical distress."},
            {"name": "Channel",
             "description": "Delivery method — 'conversational_ai' (chat-based, LLM-guided) or 'voice_ai' (spoken responses via STT)."},
            {"name": "Duration",
             "description": "Wall-clock time from session start to completion or abandonment."},
        ],
        "instruments": [
            {"name": "PHQ-9", "items": 9, "max_score": 27,
             "description": "Patient Health Questionnaire — depression screening (9 items, 0-3 per item)."},
            {"name": "GAD-7", "items": 7, "max_score": 21,
             "description": "Generalized Anxiety Disorder scale — anxiety screening (7 items, 0-3 per item)."},
            {"name": "MoCA", "items": 12, "max_score": 30,
             "description": "Montreal Cognitive Assessment — cognitive screening (12 sections, 30 points total)."},
            {"name": "NDDI-E", "items": 6, "max_score": 24,
             "description": "Neurological Disorders Depression Inventory for Epilepsy (6 items, 1-4 per item)."},
            {"name": "MMSE", "items": 11, "max_score": 30,
             "description": "Mini-Mental State Examination — cognitive screening (11 sections, 30 points total)."},
            {"name": "Epworth", "items": 8, "max_score": 24,
             "description": "Epworth Sleepiness Scale — daytime sleepiness (8 items, 0-3 per item)."},
            {"name": "Barthel", "items": 10, "max_score": 100,
             "description": "Barthel Index — activities of daily living (10 items, 0/5/10/15 per item)."},
        ],
        "quality_metrics": [
            {"name": "Session Completion Rate",
             "description": "Fraction of sessions completed vs started — target > 80%."},
            {"name": "Average Duration",
             "description": "Mean time to complete a guided session — shorter is better for patient experience, but too short may indicate skipping."},
            {"name": "Abandonment Rate",
             "description": "Fraction of sessions abandoned — high rates may indicate UX or fatigue issues."},
            {"name": "Channel Utilization",
             "description": "Distribution of sessions across conversational AI vs voice AI channels."},
        ],
        "compliance": [
            {"name": "Standardized Administration",
             "description": "Each item presented exactly as published in the instrument manual — no paraphrasing or reordering."},
            {"name": "Response Validation",
             "description": "Patient responses validated against allowed options before recording (e.g., 0-3 for PHQ-9)."},
            {"name": "Audit Trail",
             "description": "Every session recorded with timestamps, patient ID, answers, and channel for regulatory traceability."},
        ],
        "remediation": [
            {"name": "High Abandonment",
             "description": "If abandonment rate exceeds 15%, review session UX — consider shorter instruments, progress indicators, or break points."},
            {"name": "Score Alerts",
             "description": "Scores exceeding clinical thresholds (e.g., PHQ-9 >= 15) trigger clinical review alerts."},
            {"name": "Stuck Sessions",
             "description": "In-progress sessions older than 24 hours should be flagged for follow-up or closure."},
        ],
    }


if __name__ == "__main__":
    import json as _json
    print("=== Overview ===")
    print(_json.dumps(overview(), indent=2)[:2000])
    print("\n=== Breakdown (keys) ===")
    bd = breakdown()
    for k, v in bd.items():
        print(f"  {k}: {len(v) if isinstance(v, list) else type(v).__name__}")
    print("\n=== Definitions (keys) ===")
    df = definitions()
    for k, v in df.items():
        print(f"  {k}: {len(v)}")
