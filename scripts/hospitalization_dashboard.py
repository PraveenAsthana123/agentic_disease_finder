"""
Neuro AI Ecosystem — Hospitalization Dashboard
===============================================
Hospitalization analytics from the hospitalization table in clinical.db.

Admission types: emergency, planned, observation, transfer
Wards: Epilepsy Monitoring Unit, Neurology Ward, ICU, Emergency, Surgical Recovery
Dispositions: home, rehabilitation, transferred, ama, null (still admitted)

Real data: hospitalization table in clinical.db (115 rows).
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
    """Hospitalization overview — KPIs, type/reason/ward/disposition distributions,
    monthly timeline, insurance breakdown."""
    conn = _conn()
    cur = conn.cursor()

    # Total admissions
    cur.execute("SELECT COUNT(*) FROM hospitalization")
    total_admissions = cur.fetchone()[0]

    # Currently admitted (discharge_date IS NULL)
    cur.execute("""
        SELECT COUNT(*) FROM hospitalization
        WHERE json_extract(fields_json, '$.discharge_date') IS NULL
    """)
    currently_admitted = cur.fetchone()[0]

    # Average length of stay (for discharged patients)
    cur.execute("""
        SELECT AVG(CAST(json_extract(fields_json, '$.length_of_stay_days') AS REAL))
        FROM hospitalization
        WHERE json_extract(fields_json, '$.discharge_date') IS NOT NULL
    """)
    avg_los_raw = cur.fetchone()[0]
    avg_length_of_stay_days = round(avg_los_raw, 1) if avg_los_raw is not None else 0.0

    # Readmission rate (readmission_within_30d = true / total with known discharge)
    cur.execute("""
        SELECT COUNT(*) FROM hospitalization
        WHERE json_extract(fields_json, '$.discharge_date') IS NOT NULL
    """)
    discharged_total = cur.fetchone()[0]

    cur.execute("""
        SELECT COUNT(*) FROM hospitalization
        WHERE json_extract(fields_json, '$.readmission_within_30d') = 1
    """)
    readmit_count = cur.fetchone()[0]
    readmission_rate_pct = round(readmit_count * 100.0 / discharged_total, 1) if discharged_total else 0.0

    # Seizure-free at discharge rate
    cur.execute("""
        SELECT COUNT(*) FROM hospitalization
        WHERE json_extract(fields_json, '$.seizure_free_at_discharge') = 1
          AND json_extract(fields_json, '$.discharge_date') IS NOT NULL
    """)
    seizure_free_count = cur.fetchone()[0]
    seizure_free_discharge_rate_pct = round(seizure_free_count * 100.0 / discharged_total, 1) if discharged_total else 0.0

    # Average cost per admission
    cur.execute("""
        SELECT AVG(CAST(json_extract(fields_json, '$.total_cost_usd') AS REAL))
        FROM hospitalization
        WHERE json_extract(fields_json, '$.total_cost_usd') IS NOT NULL
    """)
    avg_cost_raw = cur.fetchone()[0]
    avg_cost_per_admission = round(avg_cost_raw, 2) if avg_cost_raw is not None else 0.0

    # Admission type distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.admission_type') admission_type, COUNT(*) cnt
        FROM hospitalization
        GROUP BY admission_type
        ORDER BY cnt DESC
    """)
    admission_type_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Admission reason distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.admission_reason') admission_reason, COUNT(*) cnt
        FROM hospitalization
        GROUP BY admission_reason
        ORDER BY cnt DESC
    """)
    admission_reason_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Ward distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.ward') ward, COUNT(*) cnt
        FROM hospitalization
        GROUP BY ward
        ORDER BY cnt DESC
    """)
    ward_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Discharge disposition distribution
    cur.execute("""
        SELECT COALESCE(json_extract(fields_json, '$.discharge_disposition'), 'currently_admitted') disposition,
               COUNT(*) cnt
        FROM hospitalization
        GROUP BY disposition
        ORDER BY cnt DESC
    """)
    disposition_distribution = {r[0]: r[1] for r in cur.fetchall()}

    # Insurance type distribution
    cur.execute("""
        SELECT json_extract(fields_json, '$.insurance_type') insurance_type, COUNT(*) cnt
        FROM hospitalization
        GROUP BY insurance_type
        ORDER BY cnt DESC
    """)
    insurance_distribution = {r[0]: r[1] for r in cur.fetchall() if r[0] is not None}

    # Monthly timeline — last 12 months (by admission_date)
    cur.execute("""
        SELECT DISTINCT substr(json_extract(fields_json, '$.admission_date'), 1, 7) month
        FROM hospitalization
        WHERE json_extract(fields_json, '$.admission_date') IS NOT NULL
        ORDER BY month DESC
        LIMIT 12
    """)
    month_rows = cur.fetchall()
    months = sorted([r[0] for r in month_rows])

    monthly_timeline = []
    for m in months:
        cur.execute("""
            SELECT COUNT(*) FROM hospitalization
            WHERE substr(json_extract(fields_json, '$.admission_date'), 1, 7) = ?
        """, (m,))
        adm_count = cur.fetchone()[0]

        cur.execute("""
            SELECT COUNT(*) FROM hospitalization
            WHERE substr(json_extract(fields_json, '$.discharge_date'), 1, 7) = ?
        """, (m,))
        dis_count = cur.fetchone()[0]

        cur.execute("""
            SELECT AVG(CAST(json_extract(fields_json, '$.length_of_stay_days') AS REAL))
            FROM hospitalization
            WHERE substr(json_extract(fields_json, '$.admission_date'), 1, 7) = ?
              AND json_extract(fields_json, '$.discharge_date') IS NOT NULL
        """, (m,))
        los_raw = cur.fetchone()[0]
        avg_los = round(los_raw, 1) if los_raw is not None else 0.0

        monthly_timeline.append({
            "month": m,
            "admissions": adm_count,
            "discharges": dis_count,
            "avg_los": avg_los,
        })

    conn.close()
    return {
        "total_admissions": total_admissions,
        "currently_admitted": currently_admitted,
        "avg_length_of_stay_days": avg_length_of_stay_days,
        "readmission_rate_pct": readmission_rate_pct,
        "seizure_free_discharge_rate_pct": seizure_free_discharge_rate_pct,
        "avg_cost_per_admission": avg_cost_per_admission,
        "admission_type_distribution": admission_type_distribution,
        "admission_reason_distribution": admission_reason_distribution,
        "ward_distribution": ward_distribution,
        "disposition_distribution": disposition_distribution,
        "monthly_timeline": monthly_timeline,
        "insurance_distribution": insurance_distribution,
    }


def breakdown():
    """Hospitalization breakdown — per-patient summary, currently admitted list,
    recent discharges, physician stats, complication summary."""
    conn = _conn()
    cur = conn.cursor()

    # -- Per-patient summary ---------------------------------------------------
    cur.execute("""
        SELECT
            patient_id,
            COUNT(*) total_admissions,
            SUM(CAST(json_extract(fields_json, '$.length_of_stay_days') AS REAL)) total_days,
            AVG(CAST(json_extract(fields_json, '$.length_of_stay_days') AS REAL)) avg_los,
            SUM(CASE WHEN json_extract(fields_json, '$.readmission_within_30d') = 1 THEN 1 ELSE 0 END) readmissions,
            SUM(CASE WHEN json_extract(fields_json, '$.seizure_free_at_discharge') = 1 THEN 1 ELSE 0 END) seizure_free_count,
            SUM(CAST(json_extract(fields_json, '$.total_cost_usd') AS REAL)) total_cost
        FROM hospitalization
        GROUP BY patient_id
        ORDER BY total_admissions DESC
    """)
    per_patient = []
    for r in cur.fetchall():
        total_adm = r[1]
        sf_count = r[5]
        discharged = cur.execute("""
            SELECT COUNT(*) FROM hospitalization
            WHERE patient_id = ?
              AND json_extract(fields_json, '$.discharge_date') IS NOT NULL
        """, (r[0],)).fetchone()[0]
        seizure_free_rate = round(sf_count * 100.0 / discharged, 1) if discharged else 0.0
        per_patient.append({
            "patient_id": r[0],
            "total_admissions": total_adm,
            "total_days": int(r[2]) if r[2] is not None else 0,
            "avg_los": round(r[3], 1) if r[3] is not None else 0.0,
            "readmissions": r[4],
            "seizure_free_rate": seizure_free_rate,
            "total_cost": round(r[6], 2) if r[6] is not None else 0.0,
        })

    # -- Currently admitted (discharge_date IS NULL) ---------------------------
    cur.execute("""
        SELECT
            patient_id,
            json_extract(fields_json, '$.admission_date') admission_date,
            json_extract(fields_json, '$.ward') ward,
            json_extract(fields_json, '$.admission_reason') admission_reason,
            json_extract(fields_json, '$.attending_physician') attending_physician,
            CAST(
                ROUND((julianday('now') - julianday(json_extract(fields_json, '$.admission_date'))))
            AS INTEGER) days_so_far
        FROM hospitalization
        WHERE json_extract(fields_json, '$.discharge_date') IS NULL
        ORDER BY admission_date ASC
    """)
    currently_admitted = []
    for r in cur.fetchall():
        currently_admitted.append({
            "patient_id": r[0],
            "admission_date": r[1],
            "ward": r[2],
            "admission_reason": r[3],
            "attending_physician": r[4],
            "days_so_far": r[5] if r[5] is not None else 0,
        })

    # -- Recent discharges (last 20) -------------------------------------------
    cur.execute("""
        SELECT
            patient_id,
            json_extract(fields_json, '$.admission_date') admission_date,
            json_extract(fields_json, '$.discharge_date') discharge_date,
            json_extract(fields_json, '$.ward') ward,
            json_extract(fields_json, '$.length_of_stay_days') los_days,
            json_extract(fields_json, '$.discharge_disposition') disposition,
            json_extract(fields_json, '$.seizure_free_at_discharge') seizure_free
        FROM hospitalization
        WHERE json_extract(fields_json, '$.discharge_date') IS NOT NULL
        ORDER BY json_extract(fields_json, '$.discharge_date') DESC
        LIMIT 20
    """)
    recent_discharges = []
    for r in cur.fetchall():
        recent_discharges.append({
            "patient_id": r[0],
            "admission_date": r[1],
            "discharge_date": r[2],
            "ward": r[3],
            "los_days": r[4],
            "disposition": r[5],
            "seizure_free": bool(r[6]) if r[6] is not None else None,
        })

    # -- Physician stats -------------------------------------------------------
    cur.execute("""
        SELECT
            json_extract(fields_json, '$.attending_physician') physician,
            COUNT(*) total_patients,
            AVG(CAST(json_extract(fields_json, '$.length_of_stay_days') AS REAL)) avg_los,
            SUM(CASE WHEN json_extract(fields_json, '$.seizure_free_at_discharge') = 1 THEN 1 ELSE 0 END) sf_count,
            SUM(CASE WHEN json_extract(fields_json, '$.discharge_date') IS NOT NULL THEN 1 ELSE 0 END) discharged
        FROM hospitalization
        WHERE json_extract(fields_json, '$.attending_physician') IS NOT NULL
        GROUP BY physician
        ORDER BY total_patients DESC
    """)
    physician_stats = []
    for r in cur.fetchall():
        discharged = r[4]
        sf_rate = round(r[3] * 100.0 / discharged, 1) if discharged else 0.0
        physician_stats.append({
            "physician": r[0],
            "total_patients": r[1],
            "avg_los": round(r[2], 1) if r[2] is not None else 0.0,
            "seizure_free_rate": sf_rate,
        })

    # -- Complication summary --------------------------------------------------
    cur.execute("""
        SELECT
            COALESCE(json_extract(fields_json, '$.complications'), 'none') complication,
            COUNT(*) cnt
        FROM hospitalization
        GROUP BY complication
        ORDER BY cnt DESC
    """)
    complication_summary = [{"complication": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()
    return {
        "per_patient": per_patient,
        "currently_admitted": currently_admitted,
        "recent_discharges": recent_discharges,
        "physician_stats": physician_stats,
        "complication_summary": complication_summary,
    }


def definitions():
    """Hospitalization definitions — admission types, reasons, wards, dispositions,
    and clinical glossary."""
    return {
        "admission_types": {
            "emergency": "Unplanned admission through the emergency department due to acute seizure activity, status epilepticus, or sudden neurological deterioration.",
            "planned": "Scheduled admission for a pre-defined clinical purpose such as pre-surgical evaluation, video-EEG monitoring, or elective medication adjustment.",
            "transfer": "Admission originating from another hospital or facility, typically for specialist epilepsy care not available at the referring centre.",
            "observation": "Short-stay admission (usually < 24 hours) to monitor a patient's condition without committing to full inpatient status.",
        },
        "admission_reasons": {
            "status_epilepticus": "Medical emergency involving prolonged seizure activity (> 5 minutes) or recurrent seizures without recovery of consciousness between episodes.",
            "seizure_cluster": "A series of multiple seizures occurring within a short period (typically 24 hours) with incomplete recovery between events.",
            "medication_adjustment": "Admission to safely initiate, titrate, or switch anti-epileptic drugs under monitored conditions.",
            "pre_surgical_evaluation": "Comprehensive inpatient workup — including video-EEG, MRI, neuropsychology — to determine surgical candidacy for drug-resistant epilepsy.",
            "post_surgical_monitoring": "Observation period following epilepsy surgery to manage recovery, wound care, and early post-operative seizure activity.",
            "breakthrough_seizure": "Seizure occurrence in a patient who was previously well-controlled, suggesting medication failure, non-adherence, or disease progression.",
            "aed_toxicity": "Admission to manage adverse drug reactions or toxic serum levels of anti-epileptic medications.",
            "first_seizure": "Evaluation following a first unprovoked seizure to establish diagnosis, identify aetiology, and initiate treatment if indicated.",
            "video_eeg_monitoring": "Prolonged inpatient recording combining video surveillance with continuous EEG to characterise seizure type, localise onset zone, or rule out PNES.",
        },
        "wards": {
            "Epilepsy Monitoring Unit": "Dedicated inpatient unit equipped with continuous video-EEG monitoring, staffed by epilepsy-specialist nurses and neurophysiology technicians.",
            "Neurology Ward": "General neurology inpatient ward providing care for patients requiring neurological assessment, medication management, or observation.",
            "ICU": "Intensive Care Unit — for critically ill patients with refractory status epilepticus, post-operative complications, or multi-organ involvement.",
            "Emergency": "Emergency department — initial stabilisation and assessment of acute seizure events before admission or discharge.",
            "Surgical Recovery": "Post-operative recovery ward for patients following epilepsy surgery, resective procedures, or neuromodulation device implantation.",
        },
        "disposition_types": {
            "home": "Patient discharged directly to their home environment, typically with updated medication plan and outpatient follow-up arranged.",
            "rehabilitation": "Transfer to a rehabilitation facility for functional recovery, including physiotherapy, occupational therapy, or neuropsychological rehabilitation.",
            "transferred": "Transfer to another inpatient facility for ongoing specialist care, higher acuity management, or geographic proximity to family.",
            "ama": "Discharge Against Medical Advice — patient left the hospital before the clinical team considered it safe to do so.",
            "currently_admitted": "Patient remains hospitalised with no recorded discharge date.",
        },
        "glossary": [
            {"term": "LOS", "definition": "Length of Stay — total number of inpatient days from admission date to discharge date."},
            {"term": "AED", "definition": "Anti-Epileptic Drug — pharmacological agent used to prevent or reduce seizure frequency and severity."},
            {"term": "Status Epilepticus", "definition": "A prolonged seizure or series of seizures lasting > 5 minutes without recovery of consciousness, constituting a neurological emergency."},
            {"term": "EMU", "definition": "Epilepsy Monitoring Unit — a specialist inpatient ward providing continuous video-EEG monitoring for seizure characterisation and surgical evaluation."},
            {"term": "VEE", "definition": "Video-EEG — simultaneous recording of clinical behaviour on video with electrophysiological data from scalp or intracranial electrodes."},
            {"term": "Readmission", "definition": "An unplanned return to hospital within 30 days of a previous discharge, used as a quality indicator for care transitions."},
            {"term": "AMA", "definition": "Against Medical Advice — a discharge category where the patient leaves the hospital without clinician approval."},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizure — paroxysmal events resembling seizures but without epileptiform EEG activity, associated with psychological factors."},
            {"term": "Seizure-Free at Discharge", "definition": "Clinical status indicating no seizure activity was recorded during the inpatient stay or within the 24 hours prior to discharge."},
            {"term": "Breakthrough Seizure", "definition": "A seizure occurring in a previously well-controlled patient, indicating potential medication failure or non-adherence."},
            {"term": "Pre-Surgical Evaluation", "definition": "A structured inpatient workup combining video-EEG, neuroimaging, and neuropsychological testing to determine epilepsy surgery candidacy."},
            {"term": "SUDEP", "definition": "Sudden Unexpected Death in Epilepsy — unexpected death in a person with epilepsy where no other cause is found post-mortem."},
            {"term": "Drug-Resistant Epilepsy", "definition": "Failure to achieve sustained seizure freedom following adequate trials of two or more appropriately chosen AEDs."},
        ],
    }


if __name__ == "__main__":
    import json

    print("=== OVERVIEW ===")
    ov = overview()
    print(json.dumps(ov, indent=2))

    print("\n=== BREAKDOWN ===")
    bk = breakdown()
    # Truncate per_patient for readability
    bk_display = dict(bk)
    bk_display["per_patient"] = bk["per_patient"][:5]
    print(json.dumps(bk_display, indent=2))

    print("\n=== DEFINITIONS (keys only) ===")
    df = definitions()
    print(list(df.keys()))
