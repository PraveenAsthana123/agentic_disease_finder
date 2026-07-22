"""
Neuro AI Ecosystem — ICD-10 Coding Dashboard
=============================================
ICD-10 diagnostic coding analytics from icd10_coding_records table.

Statuses: confirmed, auto_coded, pending_review, rejected
Coders: AI-AutoCoder v2.1, human coders (Dr. Patel, Dr. Sharma, Dr. Gupta,
        Coder Analyst Singh, Coder Analyst Reddy)
Tracks: primary/secondary codes, confidence, rejection reasons

Real data: icd10_coding_records (85 rows, 41 patients) in clinical.db.
"""

import sqlite3
from pathlib import Path

DB_PATH = str(Path(__file__).parent.parent / "data" / "clinical.db")


def _conn():
    return sqlite3.connect(DB_PATH)


def _dict_rows(cursor):
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, r)) for r in cursor.fetchall()]


def overview(patient_id=None):
    conn = _conn()
    cur = conn.cursor()
    where = ""
    params = []
    if patient_id:
        where = "WHERE patient_id = ?"
        params = [patient_id]

    cur.execute(f"SELECT COUNT(*) FROM icd10_coding_records {where}", params)
    total_records = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(DISTINCT patient_id) FROM icd10_coding_records {where}", params)
    total_patients = cur.fetchone()[0]

    cur.execute(f"SELECT COUNT(DISTINCT primary_code) FROM icd10_coding_records {where}", params)
    unique_codes = cur.fetchone()[0]

    cur.execute(f"SELECT status, COUNT(*) cnt FROM icd10_coding_records {where} GROUP BY status ORDER BY cnt DESC", params)
    status_dist = [{"name": r[0], "value": r[1]} for r in cur.fetchall()]

    cur.execute(f"SELECT primary_code, primary_desc, COUNT(*) cnt FROM icd10_coding_records {where} GROUP BY primary_code, primary_desc ORDER BY cnt DESC LIMIT 15", params)
    top_codes = [{"code": r[0], "description": r[1], "count": r[2]} for r in cur.fetchall()]

    cur.execute(f"SELECT ROUND(AVG(confidence), 3), ROUND(MIN(confidence), 3), ROUND(MAX(confidence), 3), COUNT(CASE WHEN confidence >= 0.9 THEN 1 END), COUNT(CASE WHEN confidence >= 0.7 AND confidence < 0.9 THEN 1 END), COUNT(CASE WHEN confidence < 0.7 THEN 1 END) FROM icd10_coding_records {where}", params)
    cr = cur.fetchone()
    confidence_stats = {"avg": cr[0], "min": cr[1], "max": cr[2], "high_confidence": cr[3], "medium_confidence": cr[4], "low_confidence": cr[5]}
    confidence_tiers = [{"name": "High (>=0.9)", "value": cr[3]}, {"name": "Medium (0.7-0.9)", "value": cr[4]}, {"name": "Low (<0.7)", "value": cr[5]}]

    cur.execute(f"SELECT coder, COUNT(*) cnt, ROUND(AVG(confidence), 3) avg_conf FROM icd10_coding_records {where} GROUP BY coder ORDER BY cnt DESC", params)
    coder_breakdown = [{"coder": r[0], "count": r[1], "avg_confidence": r[2]} for r in cur.fetchall()]

    cur.execute(f"SELECT CASE WHEN coder LIKE 'AI-%%' THEN 'AI' ELSE 'Human' END as coder_type, COUNT(*) cnt, ROUND(AVG(confidence), 3) avg_conf, COUNT(CASE WHEN status = 'confirmed' THEN 1 END) confirmed, COUNT(CASE WHEN status = 'rejected' THEN 1 END) rejected FROM icd10_coding_records {where} GROUP BY coder_type", params)
    ai_vs_human = [{"type": r[0], "count": r[1], "avg_confidence": r[2], "confirmed": r[3], "rejected": r[4]} for r in cur.fetchall()]

    cur.execute(f"SELECT COUNT(CASE WHEN secondary_codes IS NOT NULL AND secondary_codes != '' THEN 1 END), COUNT(*) FROM icd10_coding_records {where}", params)
    sr = cur.fetchone()
    secondary_coverage = {"with_secondary": sr[0], "total": sr[1], "pct": round(sr[0] * 100.0 / sr[1], 1) if sr[1] else 0}

    cur.execute(f"SELECT substr(encounter_date, 1, 7) mo, COUNT(*) cnt FROM icd10_coding_records {where} GROUP BY mo ORDER BY mo", params)
    monthly_trend = [{"month": r[0], "count": r[1]} for r in cur.fetchall()]

    rej_where = "WHERE rejection_reason IS NOT NULL AND rejection_reason != ''"
    if patient_id:
        rej_where += " AND patient_id = ?"
    cur.execute(f"SELECT rejection_reason, COUNT(*) cnt FROM icd10_coding_records {rej_where} GROUP BY rejection_reason ORDER BY cnt DESC", params)
    rejection_reasons = [{"reason": r[0], "count": r[1]} for r in cur.fetchall()]

    conn.close()
    confirmed_count = next((s["value"] for s in status_dist if s["name"] == "confirmed"), 0)
    return {
        "kpis": {"total_records": total_records, "total_patients": total_patients, "unique_codes": unique_codes, "confirmation_rate": round(confirmed_count * 100.0 / total_records, 1) if total_records else 0, "avg_confidence": confidence_stats["avg"], "secondary_coverage_pct": secondary_coverage["pct"]},
        "status_distribution": status_dist,
        "top_codes": top_codes,
        "confidence_stats": confidence_stats,
        "confidence_tiers": confidence_tiers,
        "coder_breakdown": coder_breakdown,
        "ai_vs_human": ai_vs_human,
        "secondary_coverage": secondary_coverage,
        "monthly_trend": monthly_trend,
        "rejection_reasons": rejection_reasons,
    }


def breakdown(patient_id=None):
    conn = _conn()
    cur = conn.cursor()
    where = ""
    params = []
    if patient_id:
        where = "WHERE patient_id = ?"
        params = [patient_id]

    cur.execute(f"SELECT patient_id, COUNT(*) total_codes, COUNT(DISTINCT primary_code) unique_codes, ROUND(AVG(confidence), 3) avg_conf, COUNT(CASE WHEN status = 'confirmed' THEN 1 END) confirmed, COUNT(CASE WHEN status = 'rejected' THEN 1 END) rejected, MAX(encounter_date) latest_encounter FROM icd10_coding_records {where} GROUP BY patient_id ORDER BY total_codes DESC", params)
    patient_summary = _dict_rows(cur)

    cur.execute(f"SELECT id, patient_id, encounter_date, primary_code, primary_desc, status, confidence, coder, rejection_reason FROM icd10_coding_records {where} ORDER BY encounter_date DESC LIMIT 20", params)
    recent_activity = _dict_rows(cur)

    cur.execute(f"SELECT substr(primary_code, 1, 3) chapter, COUNT(*) cnt FROM icd10_coding_records {where} GROUP BY chapter ORDER BY cnt DESC", params)
    chapter_dist = [{"chapter": r[0], "count": r[1]} for r in cur.fetchall()]

    lc_where = "WHERE confidence < 0.7"
    if patient_id:
        lc_where += " AND patient_id = ?"
    cur.execute(f"SELECT id, patient_id, encounter_date, primary_code, primary_desc, confidence, coder, status FROM icd10_coding_records {lc_where} ORDER BY confidence ASC", params)
    low_confidence = _dict_rows(cur)

    pr_where = "WHERE status = 'pending_review'"
    if patient_id:
        pr_where += " AND patient_id = ?"
    cur.execute(f"SELECT id, patient_id, encounter_date, primary_code, primary_desc, confidence, coder FROM icd10_coding_records {pr_where} ORDER BY encounter_date DESC", params)
    pending_review = _dict_rows(cur)

    cur.execute(f"SELECT coder, COUNT(*) total, COUNT(CASE WHEN status = 'confirmed' THEN 1 END) confirmed, COUNT(CASE WHEN status = 'rejected' THEN 1 END) rejected, COUNT(CASE WHEN status = 'pending_review' THEN 1 END) pending, ROUND(AVG(confidence), 3) avg_conf, ROUND(MIN(confidence), 3) min_conf FROM icd10_coding_records {where} GROUP BY coder ORDER BY total DESC", params)
    coder_performance = _dict_rows(cur)

    conn.close()
    return {"patient_summary": patient_summary, "recent_activity": recent_activity, "chapter_distribution": chapter_dist, "low_confidence_records": low_confidence, "pending_review_queue": pending_review, "coder_performance": coder_performance}


def definitions():
    return {
        "statuses": {"confirmed": "Code verified and accepted by clinical coder or supervising physician", "auto_coded": "Automatically assigned by AI-AutoCoder; awaiting human validation", "pending_review": "Flagged for manual review due to low confidence or edge case", "rejected": "Code rejected after review"},
        "confidence_thresholds": {"high": {"range": ">=0.9", "action": "Auto-accept eligible"}, "medium": {"range": "0.7-0.9", "action": "Requires human review"}, "low": {"range": "<0.7", "action": "Priority manual review"}},
        "coding_workflow": ["1. Encounter documentation ingested", "2. AI-AutoCoder assigns primary ICD-10 code with confidence score", "3. Secondary codes extracted from comorbidities", "4. High-confidence codes auto-confirmed", "5. Low/medium confidence routed to human coder queue", "6. Human coder confirms, corrects, or rejects", "7. Confirmed codes pushed to billing/claims"],
        "icd10_chapters": {"G40": "Epilepsy and recurrent seizures", "G43": "Migraine", "G93": "Other disorders of brain", "R56": "Convulsions, not elsewhere classified", "F06": "Other mental disorders due to known physiological condition"},
        "metrics": {"confirmation_rate": "Percentage of records with status confirmed", "avg_confidence": "Mean confidence score across all coding records", "secondary_coverage": "Percentage of records with secondary ICD-10 codes assigned", "rejection_rate": "Percentage of records rejected after review"},
    }
