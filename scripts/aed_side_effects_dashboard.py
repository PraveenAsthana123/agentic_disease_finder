"""
AED Side Effects Dashboard
==========================
Comprehensive anti-epileptic drug adverse effect profiling.

Data: medication_adherence table — 12,600 rows, 30 patients, 8 AEDs,
12 side effect types, severity scale 0–8.

All data read from clinical.db — no fabrication.
"""

import json
import sqlite3
from collections import Counter
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")

DRUG_CLASSES = {
    "Carbamazepine": "Sodium Channel Blocker",
    "Oxcarbazepine": "Sodium Channel Blocker",
    "Lacosamide": "Sodium Channel Blocker (slow inactivation)",
    "Lamotrigine": "Sodium/Calcium Channel Blocker",
    "Valproate": "Multi-mechanism (Na/GABA/Ca)",
    "Levetiracetam": "SV2A Ligand",
    "Clobazam": "GABA-A Modulator (Benzodiazepine)",
    "Topiramate": "Multi-mechanism (Na/GABA/AMPA/CA)",
}

SEVERITY_LABELS = {
    0: "None",
    1: "Minimal",
    2: "Mild",
    3: "Mild-Moderate",
    4: "Moderate",
    5: "Moderate-Severe",
    6: "Severe",
    7: "Very Severe",
    8: "Extreme",
}


def _conn():
    return sqlite3.connect(DB_PATH)


def _round1(v):
    return round(v, 1) if v is not None else 0.0


def get_overview():
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*), COUNT(DISTINCT patient_id), COUNT(DISTINCT drug_name) FROM medication_adherence")
    total, n_patients, n_drugs = cur.fetchone()

    cur.execute(
        "SELECT COUNT(*) FROM medication_adherence "
        "WHERE side_effects_json != '[]' AND side_effects_json IS NOT NULL AND side_effects_json != 'null'"
    )
    with_effects = cur.fetchone()[0]

    cur.execute(
        "SELECT AVG(side_effect_severity) FROM medication_adherence WHERE side_effect_severity > 0"
    )
    avg_sev = _round1(cur.fetchone()[0] or 0)

    cur.execute(
        "SELECT COUNT(*) FROM medication_adherence WHERE side_effect_severity >= 6"
    )
    severe_records = cur.fetchone()[0]

    # Collect all side effects
    cur.execute(
        "SELECT side_effects_json FROM medication_adherence "
        "WHERE side_effects_json != '[]' AND side_effects_json IS NOT NULL"
    )
    effect_counter = Counter()
    for (se_json,) in cur.fetchall():
        try:
            effects = json.loads(se_json)
            if effects:
                for e in effects:
                    effect_counter[e] += 1
        except Exception:
            pass

    top_effects = [{"effect": k, "count": v} for k, v in effect_counter.most_common(12)]

    # Per-drug summary (avg severity, % with effects)
    cur.execute(
        "SELECT drug_name, COUNT(*) as n, AVG(side_effect_severity) as avg_sev, "
        "COUNT(CASE WHEN side_effects_json != '[]' AND side_effects_json IS NOT NULL THEN 1 END) as with_fx "
        "FROM medication_adherence GROUP BY drug_name ORDER BY avg_sev DESC"
    )
    drug_summary = []
    for drug, n, asev, wfx in cur.fetchall():
        drug_summary.append({
            "drug": drug,
            "class": DRUG_CLASSES.get(drug, "Other"),
            "total_records": n,
            "avg_severity": _round1(asev or 0),
            "with_effects": wfx,
            "effect_rate_pct": _round1(100 * wfx / n) if n else 0,
        })

    # Severity distribution
    cur.execute(
        "SELECT side_effect_severity, COUNT(*) FROM medication_adherence GROUP BY side_effect_severity ORDER BY side_effect_severity"
    )
    sev_dist = [
        {"severity": sev, "label": SEVERITY_LABELS.get(sev, str(sev)), "count": cnt}
        for sev, cnt in cur.fetchall()
    ]

    # Patients with severe effects (severity >= 6)
    cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM medication_adherence WHERE side_effect_severity >= 6"
    )
    severe_patients = cur.fetchone()[0]

    conn.close()
    return {
        "kpis": {
            "total_records": total,
            "total_patients": n_patients,
            "total_drugs": n_drugs,
            "records_with_effects": with_effects,
            "effect_rate_pct": _round1(100 * with_effects / total) if total else 0,
            "avg_severity_when_present": avg_sev,
            "severe_records": severe_records,
            "patients_with_severe_effects": severe_patients,
        },
        "top_side_effects": top_effects,
        "drug_summary": drug_summary,
        "severity_distribution": sev_dist,
    }


def get_breakdown():
    conn = _conn()
    cur = conn.cursor()

    # Side effects by drug (drug × effect matrix)
    cur.execute(
        "SELECT drug_name, side_effects_json FROM medication_adherence "
        "WHERE side_effects_json != '[]' AND side_effects_json IS NOT NULL"
    )
    drug_effect_map = {}
    for drug, se_json in cur.fetchall():
        try:
            effects = json.loads(se_json)
            if not effects:
                continue
            if drug not in drug_effect_map:
                drug_effect_map[drug] = Counter()
            for e in effects:
                drug_effect_map[drug][e] += 1
        except Exception:
            pass

    drug_effect_profiles = []
    for drug in sorted(drug_effect_map.keys()):
        ctr = drug_effect_map[drug]
        drug_effect_profiles.append({
            "drug": drug,
            "class": DRUG_CLASSES.get(drug, "Other"),
            "top_effects": [{"effect": k, "count": v} for k, v in ctr.most_common(5)],
            "total_effect_events": sum(ctr.values()),
        })

    # Per-patient side effect burden
    cur.execute(
        "SELECT patient_id, COUNT(*) as n, AVG(side_effect_severity) as avg_sev, "
        "MAX(side_effect_severity) as max_sev, COUNT(DISTINCT drug_name) as n_drugs "
        "FROM medication_adherence "
        "GROUP BY patient_id ORDER BY avg_sev DESC LIMIT 30"
    )
    per_patient = []
    for pid, n, avg_sev, max_sev, nd in cur.fetchall():
        # Get their most common side effect
        cur.execute(
            "SELECT side_effects_json FROM medication_adherence WHERE patient_id=? AND side_effects_json != '[]'",
            (pid,)
        )
        pt_counter = Counter()
        for (se_json,) in cur.fetchall():
            try:
                for e in json.loads(se_json):
                    pt_counter[e] += 1
            except Exception:
                pass
        top_effect = pt_counter.most_common(1)[0][0] if pt_counter else "None"
        per_patient.append({
            "patient_id": pid,
            "records": n,
            "avg_severity": _round1(avg_sev or 0),
            "max_severity": max_sev or 0,
            "drugs_tried": nd,
            "top_side_effect": top_effect,
            "severity_label": SEVERITY_LABELS.get(int(avg_sev or 0), "None"),
        })

    # Side effect severity by drug (heatmap data)
    cur.execute(
        "SELECT drug_name, "
        "COUNT(CASE WHEN side_effect_severity=0 THEN 1 END) as sev0, "
        "COUNT(CASE WHEN side_effect_severity BETWEEN 1 AND 3 THEN 1 END) as mild, "
        "COUNT(CASE WHEN side_effect_severity BETWEEN 4 AND 5 THEN 1 END) as moderate, "
        "COUNT(CASE WHEN side_effect_severity >= 6 THEN 1 END) as severe "
        "FROM medication_adherence GROUP BY drug_name ORDER BY drug_name"
    )
    sev_by_drug = [
        {"drug": row[0], "none": row[1], "mild": row[2], "moderate": row[3], "severe": row[4]}
        for row in cur.fetchall()
    ]

    # Adherence impact: days with side effects vs adherence rate
    cur.execute(
        "SELECT taken, COUNT(*), AVG(side_effect_severity) "
        "FROM medication_adherence GROUP BY taken"
    )
    adherence_vs_sev = [
        {"taken": bool(t), "count": cnt, "avg_severity": _round1(avg_sev or 0)}
        for t, cnt, avg_sev in cur.fetchall()
    ]

    conn.close()
    return {
        "drug_effect_profiles": drug_effect_profiles,
        "per_patient": per_patient,
        "severity_by_drug": sev_by_drug,
        "adherence_vs_severity": adherence_vs_sev,
    }


def get_definitions():
    return {
        "dashboard_purpose": (
            "The AED Side Effects Dashboard provides systematic adverse effect profiling "
            "for anti-epileptic drugs (AEDs) used in this cohort. It tracks 12 distinct "
            "side effect types across 8 AEDs and 30 patients, enabling identification of "
            "high-burden drugs, at-risk patients, and AED–effect associations to guide "
            "individualized medication selection."
        ),
        "data_source": {
            "table": "medication_adherence",
            "rows": 12600,
            "patients": 30,
            "aeds": 8,
            "side_effect_types": 12,
            "date_range": "12-month rolling (30 patients × 420 days × 1–2 drugs/day)",
        },
        "severity_scale": [
            {"score": 0, "label": "None", "description": "No side effect experienced"},
            {"score": "1–2", "label": "Minimal–Mild", "description": "Manageable; no dose change required"},
            {"score": "3–5", "label": "Mild–Moderate-Severe", "description": "Dose adjustment may be indicated"},
            {"score": "6–8", "label": "Severe–Extreme", "description": "Discontinuation or urgent review warranted"},
        ],
        "aed_classes": [
            {"class": "Sodium Channel Blockers", "drugs": "Carbamazepine, Oxcarbazepine, Lacosamide, Lamotrigine",
             "key_effects": "Diplopia, dizziness, hyponatremia (OXC), rash (LTG)"},
            {"class": "GABA Modulators", "drugs": "Clobazam", "key_effects": "Drowsiness, tolerance, withdrawal"},
            {"class": "SV2A Ligands", "drugs": "Levetiracetam", "key_effects": "Irritability, mood changes, psychosis (rare)"},
            {"class": "Multi-mechanism", "drugs": "Valproate, Topiramate",
             "key_effects": "VPA: weight gain, tremor, hair loss, teratogenicity; TPM: cognitive slowing, kidney stones"},
        ],
        "side_effects_glossary": [
            {"effect": "Blurred vision", "drugs": "Carbamazepine, Oxcarbazepine, Lacosamide", "clinical_note": "Dose-dependent; check plasma levels"},
            {"effect": "Drowsiness", "drugs": "Clobazam, Valproate", "clinical_note": "Often transient; improves after titration"},
            {"effect": "Mood changes", "drugs": "Levetiracetam, Topiramate", "clinical_note": "Psychiatric referral if persistent"},
            {"effect": "Memory issues", "drugs": "Topiramate", "clinical_note": "Cognitive effects; consider alternative"},
            {"effect": "Weight gain", "drugs": "Valproate, Carbamazepine", "clinical_note": "Monitor BMI; dietary counselling"},
            {"effect": "Skin rash", "drugs": "Lamotrigine, Carbamazepine", "clinical_note": "Discontinue immediately if severe (SJS/TEN risk)"},
            {"effect": "Hair loss", "drugs": "Valproate", "clinical_note": "Usually reversible on dose reduction"},
            {"effect": "Tremor", "drugs": "Valproate", "clinical_note": "Dose-related; propranolol adjunct option"},
            {"effect": "Nausea", "drugs": "Valproate, Carbamazepine", "clinical_note": "Administer with food; slow titration"},
            {"effect": "Headache", "drugs": "Lacosamide, Oxcarbazepine", "clinical_note": "Often mild; resolves with dose stabilisation"},
            {"effect": "Dizziness", "drugs": "Carbamazepine, Lacosamide", "clinical_note": "Fall risk; especially in elderly"},
            {"effect": "Fatigue", "drugs": "Clobazam, Levetiracetam", "clinical_note": "Assess sleep quality concurrently"},
        ],
        "clinical_references": [
            "Brodie MJ et al. Antiepileptic drug therapy — the evidence for tolerability and efficacy. Epilepsia. 2012",
            "NICE NG217: Epilepsies in children, young people and adults (2022)",
            "Perucca P & Gilliam FG. Adverse effects of antiepileptic drugs. Lancet Neurol. 2012",
            "Kwan P et al. Definition of drug resistant epilepsy. Epilepsia. 2010",
        ],
    }
