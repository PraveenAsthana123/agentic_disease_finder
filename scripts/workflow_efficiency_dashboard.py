"""Clinical Workflow Efficiency Dashboard — script module.
Real data: patient_appointments (191 rows, 8 types, 6 providers, 4 locations),
appointments (120 rows, 5 departments), analyses (133 rows), hospitalization (115 rows).
KPIs: completion rate, no-show rate, avg duration, provider utilisation.
"""
import sqlite3
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


# ── helpers ────────────────────────────────────────────────────────────────────

def _r(row):
    """Row → plain dict."""
    return dict(row) if row else {}


def _pct(n, d):
    return round(n / d * 100, 1) if d else 0


# ── overview ──────────────────────────────────────────────────────────────────

def overview():
    with _conn() as db:
        cur = db.cursor()

        # Totals from patient_appointments
        cur.execute("SELECT COUNT(*) FROM patient_appointments")
        total = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM patient_appointments WHERE status = 'completed'")
        completed = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM patient_appointments WHERE status = 'no-show'")
        no_shows = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM patient_appointments WHERE status = 'cancelled'")
        cancelled = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM patient_appointments WHERE status = 'rescheduled'")
        rescheduled = cur.fetchone()[0]

        cur.execute("SELECT COUNT(*) FROM patient_appointments WHERE status = 'scheduled'")
        upcoming = cur.fetchone()[0]

        cur.execute("SELECT AVG(duration_minutes) FROM patient_appointments WHERE status = 'completed'")
        avg_dur = round(cur.fetchone()[0] or 0, 1)

        # Provider count
        cur.execute("SELECT COUNT(DISTINCT provider_name) FROM patient_appointments")
        provider_count = cur.fetchone()[0]

        # Distinct patients served
        cur.execute("SELECT COUNT(DISTINCT patient_id) FROM patient_appointments")
        patients_served = cur.fetchone()[0]

        # Appointment type distribution
        cur.execute("""
            SELECT appointment_type,
                   COUNT(*) as total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as comp,
                   AVG(duration_minutes) as avg_dur
            FROM patient_appointments
            GROUP BY appointment_type
            ORDER BY total DESC
        """)
        type_rows = [dict(r) for r in cur.fetchall()]

        # Location utilisation
        cur.execute("""
            SELECT location,
                   COUNT(*) as total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as comp
            FROM patient_appointments
            GROUP BY location
            ORDER BY total DESC
        """)
        loc_rows = [dict(r) for r in cur.fetchall()]

        # Provider workload
        cur.execute("""
            SELECT provider_name,
                   COUNT(*) as total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as comp,
                   SUM(CASE WHEN status='no-show' THEN 1 ELSE 0 END) as no_show
            FROM patient_appointments
            GROUP BY provider_name
            ORDER BY total DESC
        """)
        provider_rows = [dict(r) for r in cur.fetchall()]

        # EEG reads from analyses
        cur.execute("SELECT COUNT(*) FROM analyses")
        eeg_reads = cur.fetchone()[0]

        cur.execute("SELECT AVG(confidence) FROM analyses")
        avg_conf = round((cur.fetchone()[0] or 0) * 100, 1)

        # Hospitalization throughput (fields_json-backed)
        import json as _json
        cur.execute("SELECT fields_json FROM hospitalization")
        hosp_rows_raw = cur.fetchall()
        hosp_total = len(hosp_rows_raw)
        los_vals = []
        for hr in hosp_rows_raw:
            try:
                fj = _json.loads(hr[0] or "{}")
                if fj.get("length_of_stay_days") is not None:
                    los_vals.append(float(fj["length_of_stay_days"]))
            except Exception:
                pass
        avg_los = round(sum(los_vals) / len(los_vals), 1) if los_vals else 0

        return {
            "kpis": {
                "total_appointments": total,
                "completed": completed,
                "completion_rate_pct": _pct(completed, total),
                "no_show_rate_pct": _pct(no_shows, total),
                "cancelled_pct": _pct(cancelled, total),
                "rescheduled_pct": _pct(rescheduled, total),
                "upcoming_scheduled": upcoming,
                "avg_visit_duration_min": avg_dur,
                "providers_active": provider_count,
                "patients_served": patients_served,
                "eeg_reads_total": eeg_reads,
                "avg_eeg_confidence_pct": avg_conf,
                "hospital_admissions": hosp_total,
                "avg_los_days": avg_los,
            },
            "appointment_types": [
                {
                    "type": r["appointment_type"],
                    "total": r["total"],
                    "completion_rate_pct": _pct(r["comp"], r["total"]),
                    "avg_duration_min": round(r["avg_dur"] or 0, 1),
                }
                for r in type_rows
            ],
            "location_utilisation": [
                {
                    "location": r["location"],
                    "total": r["total"],
                    "completed": r["comp"],
                    "completion_rate_pct": _pct(r["comp"], r["total"]),
                }
                for r in loc_rows
            ],
            "provider_workload": [
                {
                    "provider": r["provider_name"],
                    "total_appointments": r["total"],
                    "completed": r["comp"],
                    "no_shows": r["no_show"],
                    "completion_rate_pct": _pct(r["comp"], r["total"]),
                    "no_show_rate_pct": _pct(r["no_show"], r["total"]),
                }
                for r in provider_rows
            ],
            "source": "patient_appointments (191 rows), analyses (133 rows), hospitalization (115 rows)",
        }


# ── breakdown ─────────────────────────────────────────────────────────────────

def breakdown():
    with _conn() as db:
        cur = db.cursor()

        # Per-type status cross-tab
        cur.execute("""
            SELECT appointment_type, status, COUNT(*) as cnt
            FROM patient_appointments
            GROUP BY appointment_type, status
            ORDER BY appointment_type, cnt DESC
        """)
        status_matrix = {}
        for r in cur.fetchall():
            t, s, c = r["appointment_type"], r["status"], r["cnt"]
            if t not in status_matrix:
                status_matrix[t] = {}
            status_matrix[t][s] = c

        # Dept breakdown from appointments table
        cur.execute("""
            SELECT department,
                   COUNT(*) as total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as comp,
                   AVG(duration_min) as avg_dur
            FROM appointments
            GROUP BY department
            ORDER BY total DESC
        """)
        dept_rows = [dict(r) for r in cur.fetchall()]

        # Monthly trend (patient_appointments — by appointment_date month)
        cur.execute("""
            SELECT substr(appointment_date, 1, 7) as month,
                   COUNT(*) as total,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as comp
            FROM patient_appointments
            WHERE appointment_date IS NOT NULL
            GROUP BY month
            ORDER BY month
        """)
        monthly = [dict(r) for r in cur.fetchall()]

        # Per-patient summary (top 15 by visit count)
        cur.execute("""
            SELECT patient_id,
                   COUNT(*) as visits,
                   SUM(CASE WHEN status='completed' THEN 1 ELSE 0 END) as comp,
                   SUM(CASE WHEN status='no-show' THEN 1 ELSE 0 END) as no_show,
                   AVG(duration_minutes) as avg_dur
            FROM patient_appointments
            GROUP BY patient_id
            ORDER BY visits DESC
            LIMIT 15
        """)
        patient_rows = [dict(r) for r in cur.fetchall()]

        # EEG reads breakdown by disease / signal quality
        cur.execute("""
            SELECT disease, signal_quality, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM analyses
            GROUP BY disease, signal_quality
            ORDER BY cnt DESC
        """)
        eeg_rows = [
            {**dict(r), "avg_conf_pct": round((r["avg_conf"] or 0) * 100, 1)}
            for r in cur.fetchall()
        ]

        # Hospitalization — ward / disposition breakdown (from fields_json)
        import json as _json2
        cur.execute("SELECT fields_json FROM hospitalization")
        _hosp_raw = cur.fetchall()
        ward_map = {}
        for _hr in _hosp_raw:
            try:
                _fj = _json2.loads(_hr[0] or "{}")
                ward = _fj.get("ward", "Unknown")
                disp = _fj.get("discharge_disposition", "Unknown")
                los = float(_fj.get("length_of_stay_days", 0) or 0)
                key = (ward, disp)
                if key not in ward_map:
                    ward_map[key] = {"cnt": 0, "los_sum": 0}
                ward_map[key]["cnt"] += 1
                ward_map[key]["los_sum"] += los
            except Exception:
                pass
        hosp_rows = [
            {
                "ward": k[0],
                "discharge_disposition": k[1],
                "cnt": v["cnt"],
                "avg_los": round(v["los_sum"] / v["cnt"], 1) if v["cnt"] else 0,
            }
            for k, v in sorted(ward_map.items(), key=lambda x: -x[1]["cnt"])
        ][:20]

        return {
            "status_matrix": status_matrix,
            "departments": [
                {
                    "department": r["department"],
                    "total": r["total"],
                    "completed": r["comp"],
                    "completion_rate_pct": _pct(r["comp"], r["total"]),
                    "avg_duration_min": round(r["avg_dur"] or 0, 1),
                }
                for r in dept_rows
            ],
            "monthly_trend": monthly,
            "per_patient": [
                {
                    "patient_id": r["patient_id"],
                    "total_visits": r["visits"],
                    "completed": r["comp"],
                    "no_shows": r["no_show"],
                    "completion_rate_pct": _pct(r["comp"], r["visits"]),
                    "avg_duration_min": round(r["avg_dur"] or 0, 1),
                }
                for r in patient_rows
            ],
            "eeg_reads": eeg_rows,
            "hospitalization_breakdown": hosp_rows,
        }


# ── definitions ──────────────────────────────────────────────────────────────

def definitions():
    return {
        "dashboard": "Clinical Workflow Efficiency Dashboard",
        "purpose": (
            "Operational performance analytics for the epilepsy centre — appointment throughput, "
            "provider workload, location utilisation, no-show tracking, EEG read pipeline, "
            "and inpatient admission efficiency."
        ),
        "data_sources": [
            {
                "table": "patient_appointments",
                "rows": 191,
                "description": (
                    "Per-patient appointment records: type (8 categories), provider (6), "
                    "location (4), status (completed/scheduled/no-show/cancelled/rescheduled), "
                    "duration in minutes."
                ),
            },
            {
                "table": "appointments",
                "rows": 120,
                "description": (
                    "Department-level appointment records: 5 departments (Neurology, EEG Lab, "
                    "Neuropsychology, Psychiatry, Surgical Recovery), booked/completed timestamps."
                ),
            },
            {
                "table": "analyses",
                "rows": 133,
                "description": (
                    "EEG AI read results: disease prediction, confidence score, signal quality. "
                    "Used to calculate EEG pipeline throughput and average confidence."
                ),
            },
            {
                "table": "hospitalization",
                "rows": 115,
                "description": (
                    "Inpatient admissions: ward, LOS, discharge disposition, admission cost. "
                    "Used for inpatient workflow efficiency metrics."
                ),
            },
        ],
        "metrics": [
            {
                "metric": "Completion Rate (%)",
                "formula": "completed / total_appointments × 100",
                "benchmark": ">85% per NAEC epilepsy centre standards",
            },
            {
                "metric": "No-Show Rate (%)",
                "formula": "no_shows / total_appointments × 100",
                "benchmark": "<10% target; >15% triggers intervention",
            },
            {
                "metric": "Average Visit Duration (min)",
                "formula": "Mean duration of completed appointments by type",
                "benchmark": "15–60 min depending on appointment type",
            },
            {
                "metric": "Provider Utilisation",
                "formula": "Completed visits per provider / total scheduled",
                "benchmark": "70–80% target utilisation",
            },
            {
                "metric": "EEG AI Confidence (%)",
                "formula": "Mean model confidence across all reads × 100",
                "benchmark": ">75% acceptable; >85% high-confidence",
            },
            {
                "metric": "Average Length of Stay (days)",
                "formula": "Mean length_of_stay_days across all admissions",
                "benchmark": "EMU average 4–7 days; ICU varies by severity",
            },
        ],
        "appointment_types": [
            {"type": "EEG Review", "description": "Post-recording EEG read with neurologist"},
            {"type": "Neurology Follow-Up", "description": "Routine neurology clinic visit"},
            {"type": "Medication Review", "description": "AED therapy adjustment or TDM review"},
            {"type": "Epilepsy Surgery Consult", "description": "Pre/post-surgical neurosurgeon meeting"},
            {"type": "Neuropsychology", "description": "Cognitive battery or counselling session"},
            {"type": "Diet Therapy Review", "description": "Ketogenic or modified Atkins review"},
            {"type": "VNS Check", "description": "Vagus Nerve Stimulator device interrogation"},
            {"type": "Telehealth Follow-Up", "description": "Remote video consultation"},
        ],
        "status_definitions": {
            "completed": "Appointment attended and service delivered",
            "scheduled": "Future appointment confirmed",
            "no-show": "Patient did not attend without prior notice",
            "cancelled": "Appointment cancelled by patient or provider",
            "rescheduled": "Moved to a different date/time",
        },
        "locations": {
            "Epilepsy Center Main": "On-site main clinic and EMU floor",
            "Outpatient Clinic B": "Secondary outpatient neurology space",
            "Telehealth": "Synchronous video consultation platform",
            "Home Video": "Patient-recorded home EEG / diary video submission",
        },
        "abbreviations": {
            "AED": "Anti-Epileptic Drug",
            "EMU": "Epilepsy Monitoring Unit",
            "EEG": "Electroencephalogram",
            "TDM": "Therapeutic Drug Monitoring",
            "VNS": "Vagus Nerve Stimulator",
            "LOS": "Length of Stay",
            "NAEC": "National Association of Epilepsy Centers",
        },
        "references": [
            "NAEC Epilepsy Center Accreditation Standards 2023",
            "ILAE Quality of Care Standards for Epilepsy Centers (2021)",
            "Engel J. et al. Report of the ILAE Classification Core Group (2017)",
        ],
    }
