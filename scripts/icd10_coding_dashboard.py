"""ICD-10 Coding Records Dashboard — diagnostic coding analytics from clinical.db.

Tracks ICD-10 coding for epilepsy/neuro encounters, including AI-assisted
auto-coding, clinician confirmation workflow, rejection analysis, and coding
accuracy metrics.

Sources:
- icd10_coding_records table (id, patient_id, encounter_date, primary_code,
  primary_desc, secondary_codes, status, confidence, coder, rejection_reason, notes)
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


# ──────────────────────────────────────────────────────────────
#  /api/icd10-coding/overview
# ──────────────────────────────────────────────────────────────

def overview():
    """High-level ICD-10 coding KPIs, distributions, coder workload, timeline."""
    conn = _conn()
    cur = conn.cursor()

    # KPIs
    cur.execute("SELECT COUNT(*) FROM icd10_coding_records")
    total_records = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT patient_id) FROM icd10_coding_records")
    total_patients = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(DISTINCT coder) FROM icd10_coding_records")
    total_coders = cur.fetchone()[0] or 0

    cur.execute("SELECT COUNT(*) FROM icd10_coding_records WHERE status = 'auto_coded'")
    auto_coded_count = cur.fetchone()[0] or 0
    auto_coded_pct = round(auto_coded_count / total_records * 100, 1) if total_records else 0.0

    cur.execute("SELECT COUNT(*) FROM icd10_coding_records WHERE status = 'confirmed'")
    confirmed_count = cur.fetchone()[0] or 0
    confirmed_pct = round(confirmed_count / total_records * 100, 1) if total_records else 0.0

    cur.execute("SELECT COUNT(*) FROM icd10_coding_records WHERE status = 'rejected'")
    rejected_count = cur.fetchone()[0] or 0

    cur.execute("SELECT ROUND(AVG(confidence), 3) FROM icd10_coding_records WHERE confidence IS NOT NULL")
    avg_confidence = cur.fetchone()[0] or 0.0

    # Status distribution (for pie chart)
    cur.execute("SELECT status, COUNT(*) FROM icd10_coding_records GROUP BY status")
    status_distribution = dict(cur.fetchall())

    # Top ICD-10 codes (for bar chart)
    cur.execute("""
        SELECT primary_code, primary_desc, COUNT(*) AS cnt
        FROM icd10_coding_records
        GROUP BY primary_code
        ORDER BY cnt DESC
        LIMIT 15
    """)
    top_codes = _dict_rows(cur)

    # Coder workload (for bar chart)
    cur.execute("""
        SELECT coder, COUNT(*) AS total,
               SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
               SUM(CASE WHEN status = 'pending_review' THEN 1 ELSE 0 END) AS pending,
               SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected
        FROM icd10_coding_records
        GROUP BY coder ORDER BY total DESC
    """)
    coder_workload = _dict_rows(cur)

    # Monthly volume timeline
    cur.execute("""
        SELECT strftime('%Y-%m', encounter_date) AS month, COUNT(*) AS records,
               SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
               SUM(CASE WHEN status = 'auto_coded' THEN 1 ELSE 0 END) AS auto_coded
        FROM icd10_coding_records
        WHERE encounter_date IS NOT NULL
        GROUP BY month ORDER BY month
    """)
    monthly_timeline = _dict_rows(cur)

    conn.close()
    return {
        "total_records": total_records,
        "total_patients": total_patients,
        "total_coders": total_coders,
        "auto_coded_pct": auto_coded_pct,
        "confirmed_pct": confirmed_pct,
        "rejected_count": rejected_count,
        "avg_confidence": avg_confidence,
        "status_distribution": status_distribution,
        "top_codes": top_codes,
        "coder_workload": coder_workload,
        "monthly_timeline": monthly_timeline,
    }


# ──────────────────────────────────────────────────────────────
#  /api/icd10-coding/breakdown
# ──────────────────────────────────────────────────────────────

def breakdown():
    """Detailed ICD-10 coding breakdown — rejections, per-coder stats, low confidence, records."""
    conn = _conn()
    cur = conn.cursor()

    # Rejection reason distribution
    cur.execute("""
        SELECT rejection_reason, COUNT(*) FROM icd10_coding_records
        WHERE status = 'rejected' AND rejection_reason IS NOT NULL
        GROUP BY rejection_reason ORDER BY COUNT(*) DESC
    """)
    rejection_distribution = dict(cur.fetchall())

    # Per-coder summary (records, confirmed rate, avg confidence)
    cur.execute("""
        SELECT coder,
               COUNT(*) AS total,
               SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
               SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected,
               SUM(CASE WHEN status = 'pending_review' THEN 1 ELSE 0 END) AS pending,
               SUM(CASE WHEN status = 'auto_coded' THEN 1 ELSE 0 END) AS auto_coded,
               ROUND(AVG(confidence), 3) AS avg_confidence
        FROM icd10_coding_records
        GROUP BY coder ORDER BY total DESC
    """)
    per_coder_summary = _dict_rows(cur)
    for row in per_coder_summary:
        reviewed = row["confirmed"] + row["rejected"]
        row["confirmed_rate_pct"] = round(row["confirmed"] / reviewed * 100, 1) if reviewed else 0.0

    # Low confidence records (confidence < 0.6)
    cur.execute("""
        SELECT patient_id, encounter_date, primary_code, primary_desc,
               status, confidence, coder, rejection_reason
        FROM icd10_coding_records
        WHERE confidence < 0.6
        ORDER BY confidence ASC
        LIMIT 30
    """)
    low_confidence_records = _dict_rows(cur)

    # Recent records (last 30)
    cur.execute("""
        SELECT patient_id, encounter_date, primary_code, primary_desc,
               secondary_codes, status, confidence, coder, rejection_reason, notes
        FROM icd10_coding_records
        ORDER BY encounter_date DESC
        LIMIT 30
    """)
    recent_records = _dict_rows(cur)

    # Status-by-coder cross-tab
    cur.execute("""
        SELECT coder, status, COUNT(*) AS cnt
        FROM icd10_coding_records
        GROUP BY coder, status
    """)
    status_by_coder = defaultdict(dict)
    for row in cur.fetchall():
        status_by_coder[row[0]][row[1]] = row[2]
    status_by_coder = dict(status_by_coder)

    # Code category analysis (G40=epilepsy, G43=migraine, R56=convulsions, F06=organic, G93=other cerebral)
    cur.execute("""
        SELECT
            CASE
                WHEN primary_code LIKE 'G40%' THEN 'G40 - Epilepsy'
                WHEN primary_code LIKE 'G41%' THEN 'G41 - Status Epilepticus'
                WHEN primary_code LIKE 'G43%' THEN 'G43 - Migraine'
                WHEN primary_code LIKE 'R56%' THEN 'R56 - Convulsions'
                WHEN primary_code LIKE 'F06%' THEN 'F06 - Organic Mental'
                WHEN primary_code LIKE 'F44%' THEN 'F44 - Conversion/PNES'
                WHEN primary_code LIKE 'G93%' THEN 'G93 - Other Cerebral'
                WHEN primary_code LIKE 'R51%' THEN 'R51 - Headache'
                ELSE 'Other'
            END AS category,
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'confirmed' THEN 1 ELSE 0 END) AS confirmed,
            SUM(CASE WHEN status = 'rejected' THEN 1 ELSE 0 END) AS rejected,
            ROUND(AVG(confidence), 3) AS avg_confidence
        FROM icd10_coding_records
        GROUP BY category
        ORDER BY total DESC
    """)
    code_category_analysis = _dict_rows(cur)
    for row in code_category_analysis:
        reviewed = row["confirmed"] + row["rejected"]
        row["accuracy_pct"] = round(row["confirmed"] / reviewed * 100, 1) if reviewed else 0.0

    conn.close()
    return {
        "rejection_distribution": rejection_distribution,
        "per_coder_summary": per_coder_summary,
        "low_confidence_records": low_confidence_records,
        "recent_records": recent_records,
        "status_by_coder": status_by_coder,
        "code_category_analysis": code_category_analysis,
    }


# ──────────────────────────────────────────────────────────────
#  /api/icd10-coding/definitions
# ──────────────────────────────────────────────────────────────

def definitions():
    """Definitions and glossary for ICD-10 coding domain."""
    return {
        "code_categories": [
            {"category": "G40.x", "name": "Epilepsy & Recurrent Seizures", "description": "Focal, generalised, and unspecified epilepsy subtypes with intractability and status epilepticus modifiers"},
            {"category": "G41.x", "name": "Status Epilepticus", "description": "Grand mal, petit mal, complex partial, and other forms of prolonged seizure activity"},
            {"category": "G43.x", "name": "Migraine", "description": "Included when EEG monitoring is indicated for differential diagnosis with epilepsy"},
            {"category": "R56.x", "name": "Convulsions", "description": "Febrile convulsions (simple and complex), post-traumatic seizures, unspecified convulsions"},
            {"category": "F06.x", "name": "Organic Mental Disorders", "description": "Mental disorders due to brain damage and dysfunction -- hallucinosis, catatonia, delusional"},
            {"category": "F44.5", "name": "Conversion / PNES", "description": "Psychogenic nonepileptic seizures diagnosed via video-EEG monitoring"},
            {"category": "G93.x", "name": "Other Cerebral Disorders", "description": "Anoxic brain damage, encephalopathy, and other structural/functional brain conditions"},
            {"category": "R51", "name": "Headache", "description": "Primary presenting symptom requiring neurological workup and differential diagnosis"},
        ],
        "status_descriptions": [
            {"status": "auto_coded", "description": "AI auto-coder analysed clinical documentation and suggested an ICD-10 code. Not yet reviewed by a clinician."},
            {"status": "confirmed", "description": "Clinician or certified coder reviewed and confirmed the auto-coded assignment as accurate. Finalised for billing."},
            {"status": "pending_review", "description": "AI auto-coder generated a suggestion, but it awaits clinician review before finalisation."},
            {"status": "rejected", "description": "Clinician rejected the auto-coded suggestion as inaccurate. A rejection reason is recorded and code is replaced manually."},
        ],
        "rejection_reasons": [
            {"reason": "incorrect_specificity", "description": "Wrong level of ICD-10 code detail (e.g. G40.3 instead of G40.309)"},
            {"reason": "wrong_laterality", "description": "Incorrect laterality modifier for the diagnosed condition"},
            {"reason": "missing_clinical_evidence", "description": "Insufficient clinical documentation to support the assigned code"},
            {"reason": "code_superseded", "description": "The assigned code has been replaced by a more current ICD-10 code"},
            {"reason": "duplicate_entry", "description": "Same code already assigned for this encounter"},
            {"reason": "non_neuro_condition", "description": "The assigned code does not fall within the neurology scope"},
        ],
        "confidence_thresholds": [
            {"range": ">= 0.80", "label": "High Confidence", "description": "Strong textual evidence supports the code. Auto-coding is reliable.", "color": "#22c55e"},
            {"range": "0.50 - 0.79", "label": "Medium Confidence", "description": "Moderate evidence. Recommended for clinician review before finalisation.", "color": "#f59e0b"},
            {"range": "< 0.50", "label": "Low Confidence", "description": "Weak evidence. Requires mandatory clinician review. Likely to be rejected.", "color": "#ef4444"},
        ],
        "clinical_notes": [
            "All auto-coded records with confidence < 0.70 are automatically routed to pending_review",
            "Coding accuracy target: >= 92% agreement rate for production deployment",
            "Rejection rates above 15% trigger AI model retraining pipeline",
            "Accuracy is stratified by ICD-10 category to identify model weaknesses",
            "Weekly accuracy audits with monthly trend reporting are mandatory",
        ],
        "glossary": [
            {"term": "ICD-10-CM", "definition": "International Classification of Diseases, Tenth Revision, Clinical Modification -- standard diagnostic coding system"},
            {"term": "Auto-Coding", "definition": "AI-assisted assignment of ICD-10 codes from unstructured clinical documentation (notes, EEG, imaging)"},
            {"term": "Coding Confidence", "definition": "Score 0.0-1.0 representing the AI model's certainty in its suggested ICD-10 code"},
            {"term": "Specificity Level", "definition": "Granularity of the ICD-10 code. More digits = more clinical detail (e.g. G40.309 vs G40.3)"},
            {"term": "Primary Code", "definition": "Principal diagnosis driving the encounter, used for billing and clinical records"},
            {"term": "Secondary Code", "definition": "Comorbid conditions or complications documented during the same encounter"},
            {"term": "Rejection Reason", "definition": "Structured reason recorded when a clinician rejects an auto-coded assignment"},
            {"term": "Coding Accuracy Rate", "definition": "Percentage of AI-suggested codes confirmed by clinician review: confirmed / (confirmed + rejected)"},
            {"term": "Encounter", "definition": "A single patient-provider interaction generating clinical documentation requiring ICD-10 coding"},
            {"term": "PNES", "definition": "Psychogenic Non-Epileptic Seizures -- events resembling seizures without epileptic EEG correlate"},
        ],
    }


if __name__ == "__main__":
    import json
    print("=== OVERVIEW ===")
    print(json.dumps(overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(breakdown(), indent=2, default=str))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(definitions(), indent=2, default=str))
