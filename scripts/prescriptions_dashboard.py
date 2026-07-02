"""Prescriptions Dashboard — prescribing analytics from clinical.db.

Tracks medication prescriptions, drug utilization, polypharmacy,
dose distributions, per-patient drug lists, and AED combinations.

Sources:
- medications table (9 records: drug_name, dose_mg, frequency, aed)
- patients table (40 rows for coverage calculations)
"""

import sqlite3
import os
import json
from collections import Counter, defaultdict

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql, params=(), default=0):
    try:
        cur.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row else default
    except Exception:
        return default


def _safe_rows(cur, sql, params=()):
    try:
        cur.execute(sql, params)
        return cur.fetchall()
    except Exception:
        return []


def _parse_fields(fields_json_str):
    """Parse fields_json column into a dict."""
    try:
        return json.loads(fields_json_str) if fields_json_str else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def prescriptions_overview():
    """KPI summary: total prescriptions, unique drugs, coverage, polypharmacy, distributions."""
    conn = _conn()
    cur = conn.cursor()

    # Fetch all medication rows
    rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM medications ORDER BY created_at")
    total_prescriptions = len(rows)

    # Parse all fields
    parsed = []
    for row in rows:
        fields = _parse_fields(row[2])
        parsed.append({
            "id": row[0],
            "patient_id": row[1],
            "drug_name": fields.get("drug_name", "Unknown"),
            "dose_mg": fields.get("dose_mg"),
            "frequency": fields.get("frequency", "Unknown"),
            "aed": fields.get("aed", []),
            "created_at": row[3]
        })

    # Unique drugs and patients
    all_drugs = [p["drug_name"] for p in parsed]
    unique_drugs = len(set(all_drugs))
    med_patients = set(p["patient_id"] for p in parsed)
    unique_patients = len(med_patients)

    total_patients = _safe(cur, "SELECT COUNT(*) FROM patients")
    prescription_coverage = round(unique_patients / total_patients * 100, 1) if total_patients > 0 else 0

    # Avg drugs per patient
    patient_drug_map = defaultdict(set)
    for p in parsed:
        patient_drug_map[p["patient_id"]].add(p["drug_name"])
    avg_drugs_per_patient = round(sum(len(v) for v in patient_drug_map.values()) / len(patient_drug_map), 1) if patient_drug_map else 0

    # Most common drug
    drug_counts = Counter(all_drugs)
    most_common_drug = drug_counts.most_common(1)[0][0] if drug_counts else "N/A"

    # Most common frequency
    freq_list = [p["frequency"] for p in parsed if p["frequency"] and p["frequency"] != "Unknown"]
    freq_counts = Counter(freq_list)
    most_common_frequency = freq_counts.most_common(1)[0][0] if freq_counts else "N/A"

    # Drug distribution
    drug_distribution = []
    for drug, count in drug_counts.most_common():
        drug_distribution.append({
            "drug_name": drug,
            "count": count,
            "percentage": round(count / total_prescriptions * 100, 1) if total_prescriptions > 0 else 0
        })

    # Patient drug counts
    patient_drug_counts = [
        {"patient_id": pid, "num_drugs": len(drugs)}
        for pid, drugs in sorted(patient_drug_map.items(), key=lambda x: -len(x[1]))
    ]

    # Frequency distribution
    all_freqs = [p["frequency"] for p in parsed]
    freq_counter = Counter(all_freqs)
    frequency_distribution = [
        {"frequency": f, "count": c}
        for f, c in freq_counter.most_common()
    ]

    # Dose distribution (buckets)
    dose_buckets = {"0-100": 0, "101-200": 0, "201-500": 0, "500+": 0}
    for p in parsed:
        dose = p["dose_mg"]
        if dose is not None:
            if dose <= 100:
                dose_buckets["0-100"] += 1
            elif dose <= 200:
                dose_buckets["101-200"] += 1
            elif dose <= 500:
                dose_buckets["201-500"] += 1
            else:
                dose_buckets["500+"] += 1
    dose_distribution = [
        {"dose_range": k, "count": v}
        for k, v in dose_buckets.items()
    ]

    # Prescriptions by date
    date_counter = Counter()
    for p in parsed:
        if p["created_at"]:
            date_str = p["created_at"][:10]  # YYYY-MM-DD
            date_counter[date_str] += 1
    prescriptions_by_date = [
        {"date": d, "count": c}
        for d, c in sorted(date_counter.items())
    ]

    # Polypharmacy patients (>= 2 distinct drugs)
    polypharmacy_patients = sum(1 for drugs in patient_drug_map.values() if len(drugs) >= 2)

    conn.close()
    return {
        "kpis": {
            "total_prescriptions": total_prescriptions,
            "unique_drugs": unique_drugs,
            "unique_patients": unique_patients,
            "total_patients": total_patients,
            "prescription_coverage_pct": prescription_coverage,
            "avg_drugs_per_patient": avg_drugs_per_patient,
            "most_common_drug": most_common_drug,
            "most_common_frequency": most_common_frequency,
            "polypharmacy_patients": polypharmacy_patients
        },
        "drug_distribution": drug_distribution,
        "patient_drug_counts": patient_drug_counts,
        "frequency_distribution": frequency_distribution,
        "dose_distribution": dose_distribution,
        "prescriptions_by_date": prescriptions_by_date
    }


def prescriptions_breakdown():
    """Per-patient and per-drug detailed breakdowns."""
    conn = _conn()
    cur = conn.cursor()

    rows = _safe_rows(cur, "SELECT id, patient_id, fields_json, created_at FROM medications ORDER BY created_at")

    parsed = []
    for row in rows:
        fields = _parse_fields(row[2])
        parsed.append({
            "id": row[0],
            "patient_id": row[1],
            "drug_name": fields.get("drug_name", "Unknown"),
            "dose_mg": fields.get("dose_mg"),
            "frequency": fields.get("frequency", "Unknown"),
            "aed": fields.get("aed", []),
            "created_at": row[3]
        })

    # Per-patient breakdown
    patient_map = defaultdict(list)
    for p in parsed:
        patient_map[p["patient_id"]].append(p)

    per_patient = []
    for pid, meds in sorted(patient_map.items()):
        drugs = [
            {
                "drug_name": m["drug_name"],
                "dose_mg": m["dose_mg"],
                "frequency": m["frequency"],
                "created_at": m["created_at"]
            }
            for m in meds
        ]
        unique_drug_names = set(m["drug_name"] for m in meds)
        per_patient.append({
            "patient_id": pid,
            "drugs": drugs,
            "total_drugs": len(unique_drug_names),
            "has_polypharmacy": len(unique_drug_names) >= 2
        })

    # Per-drug breakdown
    drug_map = defaultdict(list)
    for p in parsed:
        drug_map[p["drug_name"]].append(p)

    per_drug = []
    for drug, meds in sorted(drug_map.items()):
        doses = [m["dose_mg"] for m in meds if m["dose_mg"] is not None]
        patients = sorted(set(m["patient_id"] for m in meds))
        per_drug.append({
            "drug_name": drug,
            "patients": patients,
            "total_prescribed": len(meds),
            "avg_dose_mg": round(sum(doses) / len(doses), 1) if doses else None,
            "dose_range": {
                "min": min(doses) if doses else None,
                "max": max(doses) if doses else None
            }
        })

    # AED combinations: patients who have aed lists with multiple entries
    aed_combinations = []
    for pid, meds in patient_map.items():
        all_aeds = set()
        for m in meds:
            if m["aed"]:
                for a in m["aed"]:
                    all_aeds.add(a)
        if len(all_aeds) >= 2:
            aed_combinations.append({
                "patient_id": pid,
                "aeds": sorted(all_aeds),
                "count": len(all_aeds)
            })

    # Recent prescriptions (last 10 by created_at desc)
    recent_sorted = sorted(parsed, key=lambda x: x["created_at"] or "", reverse=True)[:10]
    recent_prescriptions = [
        {
            "patient_id": r["patient_id"],
            "drug_name": r["drug_name"],
            "dose_mg": r["dose_mg"],
            "frequency": r["frequency"],
            "created_at": r["created_at"]
        }
        for r in recent_sorted
    ]

    conn.close()
    return {
        "per_patient": per_patient,
        "per_drug": per_drug,
        "aed_combinations": aed_combinations,
        "recent_prescriptions": recent_prescriptions
    }


def prescriptions_definitions():
    """Clinical definitions for the Prescriptions dashboard."""
    return {
        "sections": [
            {
                "title": "Prescription Concepts",
                "items": [
                    {"term": "AED (Anti-Epileptic Drug)", "definition": "Medications used to control seizures in epilepsy. Common AEDs include Levetiracetam, Lamotrigine, Valproate, Carbamazepine, and Topiramate."},
                    {"term": "Polypharmacy", "definition": "The concurrent use of two or more medications for the same patient. In epilepsy, polypharmacy often means multiple AEDs when monotherapy fails to achieve seizure freedom."},
                    {"term": "Monotherapy", "definition": "Treatment with a single AED. ILAE guidelines recommend monotherapy as first-line treatment, escalating to polytherapy only when necessary."},
                    {"term": "Dose Titration", "definition": "The process of gradually adjusting medication dosage to find the optimal therapeutic level while minimizing side effects."}
                ]
            },
            {
                "title": "Quality Metrics",
                "items": [
                    {"term": "Prescription Coverage", "definition": "Percentage of registered patients who have at least one active prescription. Low coverage may indicate documentation gaps or untreated patients."},
                    {"term": "Drug Utilization Rate", "definition": "How frequently each drug is prescribed across the patient population. Helps identify prescribing patterns and formulary alignment."},
                    {"term": "Polypharmacy Rate", "definition": "Percentage of patients on two or more distinct drugs. High rates may warrant clinical review for drug interactions and side effects."},
                    {"term": "Dose Distribution", "definition": "Distribution of prescribed doses across standard ranges. Outliers may indicate off-label use or dosing errors."}
                ]
            },
            {
                "title": "Clinical Relevance",
                "items": [
                    {"term": "ILAE AED Guidelines", "definition": "International League Against Epilepsy treatment guidelines recommend evidence-based AED selection based on seizure type, tolerability, and patient factors."},
                    {"term": "FDA Drug Labeling", "definition": "FDA-approved indications, dosing ranges, and contraindications must be considered when prescribing. Off-label use should be documented."},
                    {"term": "IEC 62304", "definition": "Medical device software lifecycle standard. Prescribing analytics that inform clinical decisions must meet software validation requirements."},
                    {"term": "Drug Interaction Monitoring", "definition": "AED combinations (e.g., Valproate + Lamotrigine) require monitoring due to pharmacokinetic interactions affecting drug levels and toxicity risk."}
                ]
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {"term": "Low Prescription Coverage", "definition": "Review patients without prescriptions to determine if they need medication initiation, are managed elsewhere, or have documentation gaps."},
                    {"term": "High Polypharmacy Rate", "definition": "Conduct medication reconciliation reviews. Evaluate whether patients on 3+ AEDs could be simplified without increasing seizure risk."},
                    {"term": "Dose Outliers", "definition": "Flag prescriptions outside standard dose ranges for pharmacist review. Ensure dose adjustments are documented with clinical rationale."},
                    {"term": "Missing Frequency Data", "definition": "Ensure all prescriptions include dosing frequency. Incomplete prescriptions risk medication errors and non-adherence."}
                ]
            }
        ]
    }


if __name__ == '__main__':
    import json as _json
    print("=== OVERVIEW ===")
    print(_json.dumps(prescriptions_overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(_json.dumps(prescriptions_breakdown(), indent=2, default=str))
