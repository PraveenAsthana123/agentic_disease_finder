"""Seizure Burden & Trigger Dashboard — real clinical.db data.

Sources:
  seizure_diary        — 25 patient-reported seizure events
  seizure_trigger_logs — 203 daily trigger-monitoring entries (53 seizure days / 150 non-seizure days)

Exports: overview(), breakdown(), definitions()
"""

import sqlite3
from collections import defaultdict
from pathlib import Path
import math

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    return sqlite3.connect(str(DB))


# ── helpers ──────────────────────────────────────────────────────────────────

def _safe(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _round(v, n=1):
    try:
        return round(float(v), n)
    except Exception:
        return None


# ── overview ─────────────────────────────────────────────────────────────────

def overview():
    con = _conn()
    cur = con.cursor()

    # ── KPIs from seizure_diary ──────────────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM seizure_diary")
    total_diary = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM seizure_diary WHERE er_visit='Yes'")
    er_visits = cur.fetchone()[0]

    cur.execute("SELECT COUNT(*) FROM seizure_diary WHERE severity='Severe'")
    severe_events = cur.fetchone()[0]

    cur.execute(
        "SELECT COUNT(*) FROM seizure_diary "
        "WHERE injury IS NOT NULL AND injury != '' AND injury != 'No' AND injury != 'None'"
    )
    injury_events = cur.fetchone()[0]

    cur.execute("SELECT AVG(duration_sec), MAX(duration_sec) FROM seizure_diary")
    dur_row = cur.fetchone()
    avg_dur = round(dur_row[0]) if dur_row[0] else 0
    max_dur = dur_row[1] or 0

    cur.execute(
        "SELECT COUNT(*) FROM seizure_diary "
        "WHERE rescue_med IS NOT NULL AND rescue_med != '' AND rescue_med != 'None'"
    )
    rescue_med_used = cur.fetchone()[0]

    # ── Seizure rate from trigger logs ───────────────────────────────────────
    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs WHERE seizure_occurred=1")
    sz_days = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM seizure_trigger_logs")
    total_log_days = cur.fetchone()[0]
    seizure_rate_pct = _round(sz_days / total_log_days * 100, 1) if total_log_days else 0

    # ── Severity distribution ─────────────────────────────────────────────────
    cur.execute("SELECT severity, COUNT(*) FROM seizure_diary GROUP BY severity")
    severity_distribution = {r[0] or "Unknown": r[1] for r in cur.fetchall()}

    # ── Diary trigger distribution ────────────────────────────────────────────
    cur.execute(
        "SELECT trigger, COUNT(*) FROM seizure_diary "
        "WHERE trigger IS NOT NULL AND trigger != '' "
        "GROUP BY trigger ORDER BY 2 DESC"
    )
    diary_trigger_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # ── Log trigger distribution ──────────────────────────────────────────────
    cur.execute(
        "SELECT primary_trigger, COUNT(*) FROM seizure_trigger_logs "
        "WHERE primary_trigger IS NOT NULL AND primary_trigger != '' "
        "GROUP BY primary_trigger ORDER BY 2 DESC"
    )
    log_trigger_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # ── Physiological comparison: seizure vs non-seizure days ────────────────
    def _phys(sz_flag):
        cur.execute(
            "SELECT COUNT(*), AVG(sleep_hours), AVG(stress_level), "
            "AVG(fatigue_level), AVG(missed_doses), AVG(caffeine_mg) "
            "FROM seizure_trigger_logs WHERE seizure_occurred=?",
            (sz_flag,),
        )
        r = cur.fetchone()
        return {
            "n": r[0],
            "avg_sleep_hours": _round(r[1]),
            "avg_stress_level": _round(r[2]),
            "avg_fatigue_level": _round(r[3]),
            "avg_missed_doses": _round(r[4], 2),
            "avg_caffeine_mg": _round(r[5]),
        }

    physiological_comparison = {
        "seizure_days": _phys(1),
        "non_seizure_days": _phys(0),
    }

    # ── Patient summary from seizure_diary ────────────────────────────────────
    cur.execute(
        """SELECT patient_id,
                  COUNT(*) as total_events,
                  SUM(CASE WHEN severity='Severe' THEN 1 ELSE 0 END),
                  SUM(CASE WHEN severity='Mild'   THEN 1 ELSE 0 END),
                  SUM(CASE WHEN er_visit='Yes'    THEN 1 ELSE 0 END),
                  AVG(duration_sec),
                  MAX(duration_sec),
                  GROUP_CONCAT(event_date ORDER BY event_date)
           FROM seizure_diary
           GROUP BY patient_id
           ORDER BY total_events DESC, patient_id""",
    )
    patient_summary = []
    for r in cur.fetchall():
        dates = [d for d in (r[7] or "").split(",") if d]
        patient_summary.append(
            {
                "patient_id": r[0],
                "total_events": r[1],
                "severe_count": r[2] or 0,
                "mild_count": r[3] or 0,
                "er_visits": r[4] or 0,
                "avg_duration_sec": round(r[5]) if r[5] else 0,
                "max_duration_sec": r[6] or 0,
                "dates": sorted(set(dates)),
            }
        )

    con.close()
    return {
        "kpis": {
            "total_diary_events": total_diary,
            "er_visits": er_visits,
            "severe_events": severe_events,
            "injury_events": injury_events,
            "avg_duration_sec": avg_dur,
            "max_duration_sec": max_dur,
            "seizure_rate_in_logs_pct": seizure_rate_pct,
            "rescue_med_used": rescue_med_used,
            "total_log_days": total_log_days,
            "seizure_days_in_logs": sz_days,
        },
        "severity_distribution": severity_distribution,
        "diary_trigger_distribution": diary_trigger_distribution,
        "log_trigger_distribution": log_trigger_distribution,
        "physiological_comparison": physiological_comparison,
        "patient_summary": patient_summary,
    }


# ── breakdown ─────────────────────────────────────────────────────────────────

def breakdown():
    con = _conn()
    cur = con.cursor()

    # ── Duration histogram from trigger_logs ─────────────────────────────────
    cur.execute(
        "SELECT seizure_duration_sec FROM seizure_trigger_logs "
        "WHERE seizure_occurred=1 AND seizure_duration_sec IS NOT NULL"
    )
    durations = [r[0] for r in cur.fetchall()]

    buckets = [
        ("< 30s", 0, 30),
        ("30–60s", 30, 60),
        ("60–120s", 60, 120),
        ("120–300s", 120, 300),
        ("> 300s", 300, 99999),
    ]
    duration_histogram = []
    for label, lo, hi in buckets:
        count = sum(1 for d in durations if lo <= d < hi)
        duration_histogram.append({"label": label, "count": count})

    # ── Sleep distribution ────────────────────────────────────────────────────
    def _sleep_dist(sz_flag):
        cur.execute(
            "SELECT ROUND(sleep_hours) as hr, COUNT(*) "
            "FROM seizure_trigger_logs WHERE seizure_occurred=? "
            "GROUP BY hr ORDER BY hr",
            (sz_flag,),
        )
        return {str(int(r[0])) + "h": r[1] for r in cur.fetchall() if r[0] is not None}

    sleep_distribution = {
        "seizure": _sleep_dist(1),
        "non_seizure": _sleep_dist(0),
    }

    # ── Per-patient detail cards from seizure_diary ───────────────────────────
    cur.execute(
        """SELECT patient_id, event_date, event_time, duration_sec, severity,
                  aura, awareness, motor_signs, injury, recovery_min,
                  er_visit, trigger, location, rescue_med
           FROM seizure_diary
           ORDER BY patient_id, event_date""",
    )
    rows = cur.fetchall()

    grouped = defaultdict(list)
    for r in rows:
        grouped[r[0]].append(
            {
                "date": r[1],
                "time": r[2] or "—",
                "duration_sec": r[3],
                "severity": r[4] or "Unknown",
                "aura": r[5] or "—",
                "awareness": r[6] or "—",
                "motor_signs": r[7] or "—",
                "injury": r[8] or "None",
                "recovery_min": r[9],
                "er_visit": r[10] or "No",
                "trigger": r[11] or "Unknown",
                "location": r[12] or "—",
                "rescue_med": r[13] or "—",
            }
        )

    patient_cards = [
        {
            "patient_id": pid,
            "event_count": len(events),
            "avg_duration_sec": round(
                sum(e["duration_sec"] or 0 for e in events) / len(events)
            ),
            "events": events,
        }
        for pid, events in sorted(
            grouped.items(), key=lambda x: -len(x[1])
        )
    ]

    # ── Stress / fatigue comparison ───────────────────────────────────────────
    cur.execute(
        "SELECT stress_level, COUNT(*) FROM seizure_trigger_logs "
        "WHERE seizure_occurred=1 AND stress_level IS NOT NULL "
        "GROUP BY stress_level ORDER BY stress_level"
    )
    stress_sz = {r[0]: r[1] for r in cur.fetchall()}

    cur.execute(
        "SELECT stress_level, COUNT(*) FROM seizure_trigger_logs "
        "WHERE seizure_occurred=0 AND stress_level IS NOT NULL "
        "GROUP BY stress_level ORDER BY stress_level"
    )
    stress_nsz = {r[0]: r[1] for r in cur.fetchall()}

    # ── Exercise & alcohol on seizure vs non-seizure days ────────────────────
    cur.execute(
        "SELECT AVG(exercise_minutes), AVG(alcohol_units), AVG(screen_time_hours) "
        "FROM seizure_trigger_logs WHERE seizure_occurred=1"
    )
    r = cur.fetchone()
    lifestyle_sz = {
        "avg_exercise_min": _round(r[0]),
        "avg_alcohol_units": _round(r[1], 2),
        "avg_screen_time_hrs": _round(r[2]),
    }
    cur.execute(
        "SELECT AVG(exercise_minutes), AVG(alcohol_units), AVG(screen_time_hours) "
        "FROM seizure_trigger_logs WHERE seizure_occurred=0"
    )
    r = cur.fetchone()
    lifestyle_nsz = {
        "avg_exercise_min": _round(r[0]),
        "avg_alcohol_units": _round(r[1], 2),
        "avg_screen_time_hrs": _round(r[2]),
    }

    con.close()
    return {
        "seizure_duration_histogram": duration_histogram,
        "sleep_distribution": sleep_distribution,
        "stress_distribution": {"seizure": stress_sz, "non_seizure": stress_nsz},
        "lifestyle_comparison": {
            "seizure_days": lifestyle_sz,
            "non_seizure_days": lifestyle_nsz,
        },
        "patient_cards": patient_cards,
    }


# ── definitions ───────────────────────────────────────────────────────────────

def definitions():
    return {
        "title": "Seizure Burden & Trigger Dashboard — Definitions",
        "description": (
            "Quantifies the clinical impact of epilepsy by measuring seizure frequency, "
            "duration, severity, and modifiable environmental/physiological triggers. "
            "Supports patient-centred AED optimisation and lifestyle counselling."
        ),
        "clinical_context": (
            "Seizure burden is a composite measure used to guide treatment escalation, "
            "surgical referral, and quality-of-life tracking (Fisher et al. 2017; ILAE). "
            "Trigger analysis draws on the biopsychosocial model of seizure precipitation "
            "(Frucht et al. 2000; Haut et al. 2007)."
        ),
        "kpi_definitions": [
            {"name": "Total Events", "description": "Count of self-reported seizure events in the seizure diary."},
            {"name": "ER Visits", "description": "Events requiring emergency department attendance — marker of seizure severity and healthcare utilisation."},
            {"name": "Severe Events", "description": "Seizures classified as Severe by the reporting clinician or patient (e.g., prolonged, convulsive, or resulting in injury)."},
            {"name": "Injury Events", "description": "Seizures associated with physical injury (falls, lacerations, burns) recorded in the diary."},
            {"name": "Avg Duration (s)", "description": "Mean seizure duration in seconds across all diary entries; >5 min meets the ILAE operational definition of status epilepticus."},
            {"name": "Max Duration (s)", "description": "Longest single seizure in the diary. Duration >5 min (300 s) is flagged as a red-alert threshold."},
            {"name": "Seizure Rate (logs)", "description": "Percentage of 203 daily trigger-log entries on which a seizure occurred (seizure_trigger_logs.seizure_occurred=1)."},
            {"name": "Rescue Meds Used", "description": "Diary events where a rescue medication (benzodiazepine, levetiracetam IM) was administered."},
        ],
        "seizure_metrics": [
            {"name": "Seizure Diary", "description": "Structured patient/caregiver log capturing each event's date, time, duration, severity, aura, awareness, motor signs, injury, recovery, ER visit, trigger, and location."},
            {"name": "Trigger Log", "description": "Daily diary tracking physiological and behavioural risk factors: sleep, stress, fatigue, mood, exercise, caffeine, alcohol, screen time, medication adherence, and primary trigger."},
            {"name": "Severity — Mild", "description": "Brief, self-limited seizure without injury or post-ictal deficit. Recovery < 30 min."},
            {"name": "Severity — Moderate", "description": "Seizure with post-ictal confusion or minor injury; no ER visit required."},
            {"name": "Severity — Severe", "description": "Prolonged seizure (>5 min), cluster, status epilepticus, significant injury, or requiring ER/ICU attendance."},
            {"name": "Physiological Profile", "description": "Comparison of sleep hours, stress, fatigue, missed doses, and caffeine between seizure days and non-seizure days — identifies modifiable precipitants."},
        ],
        "trigger_factors": [
            {"name": "Sleep Deprivation", "description": "The most consistently identified trigger across all epilepsy syndromes; even 1 hr reduction increases cortical excitability (Rajna & Veres 1993)."},
            {"name": "Missed Medication", "description": "Non-adherence is the leading avoidable cause of breakthrough seizures in controlled epilepsy (Faught et al. 2008)."},
            {"name": "Stress", "description": "Psychological stress activates the HPA axis and raises cortisol, altering seizure thresholds (Frucht et al. 2000)."},
            {"name": "Photosensitivity", "description": "Flickering light at 15–25 Hz triggers photo-paroxysmal responses in ~3% of epilepsy patients (Kasteleijn-Nolst Trenité 2012)."},
            {"name": "Fatigue", "description": "Physical or cognitive fatigue shares neuroexcitatory mechanisms with sleep deprivation; often co-occurs."},
            {"name": "Alcohol", "description": "Both intoxication (excitatory) and withdrawal (rebound excitability) precipitate seizures."},
            {"name": "Hormonal Changes", "description": "Catamenial epilepsy: seizure clusters around menstruation due to progesterone withdrawal (Herzog et al. 2004)."},
            {"name": "Illness / Fever", "description": "Systemic illness lowers seizure threshold through cytokine-mediated neuroinflammation and electrolyte shifts."},
            {"name": "Dehydration", "description": "Reduces seizure threshold by altering ion concentrations and cortical perfusion."},
        ],
        "severity_levels": [
            {"level": "Mild", "description": "Brief, no injury, <30 min recovery; no emergency intervention."},
            {"level": "Moderate", "description": "Post-ictal confusion, minor injury; managed at home or outpatient."},
            {"level": "Severe", "description": "Prolonged (>5 min), convulsive, significant injury, or ER visit."},
        ],
        "data_sources": [
            "seizure_diary — 25 events from patient-reported diary (clinical.db)",
            "seizure_trigger_logs — 203 daily trigger-monitoring entries (clinical.db)",
            "References: Fisher et al. 2017 (ILAE seizure burden); Frucht et al. 2000; Haut et al. 2007; Faught et al. 2008; Herzog et al. 2004; Rajna & Veres 1993; ILAE 2014",
        ],
    }
