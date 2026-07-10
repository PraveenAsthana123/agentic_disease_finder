"""RAG Report Generation Dashboard — EEG AI RAG Pipeline Step 20

Generates structured clinical reports that unify:
- Prediction results (seizure classification, risk scores)
- Biomarkers (EEG features, spectral power, asymmetry)
- XAI explanations (SHAP feature importance, confidence)
- Retrieved evidence (RAG knowledge-base citations, literature)
- Clinical metadata (demographics, medications, seizure history)

All data from clinical.db — no fabrication, no stubs.

Sources:
  analyses          — EEG analysis results + predictions
  assessments       — clinical scale scores (PHQ-9, MoCA, etc.)
  patients          — demographics
  patient_demographics — extended demographics
  seizure_diary     — seizure event log
  medications       — current prescriptions
  clinical_decisions — AI decision audit trail
  explainability_gt — XAI ground-truth explanations
  conversation_log  — RAG conversation/retrieval history
  mri_findings      — imaging biomarkers
  pro_outcomes      — patient-reported outcomes
"""

import sqlite3
import json
import os
import hashlib
from datetime import datetime, timezone
from collections import Counter, defaultdict

BASE = os.path.join(os.path.dirname(__file__), '..')
DB = os.path.join(BASE, 'data', 'clinical.db')


def _conn():
    return sqlite3.connect(DB)


def _safe_query(cur, sql):
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _safe_json(raw):
    if not raw:
        return {}
    try:
        return json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        return {}


def _det_hash(key):
    """Deterministic hash from a string key for reproducible seeding."""
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16)


# ── Report assembly ──────────────────────────────────────────────

def _patient_prediction_section(cur, patient_id):
    """Pull latest prediction/classification for a patient from analyses."""
    rows = _safe_query(cur,
        f"SELECT * FROM analyses WHERE patient_id='{patient_id}' ORDER BY analysis_date DESC")
    if not rows:
        return {"available": False, "note": "No EEG analyses on file"}

    latest = rows[0]
    classification = latest.get('classification', latest.get('result', 'N/A'))
    confidence = latest.get('confidence')
    risk_score = latest.get('risk_score')
    method = latest.get('method', latest.get('analysis_type', 'EEG'))

    return {
        "available": True,
        "total_analyses": len(rows),
        "latest_classification": classification,
        "confidence": confidence,
        "risk_score": risk_score,
        "method": method,
        "analysis_date": latest.get('analysis_date'),
        "prior_analyses": [
            {"date": r.get('analysis_date'), "result": r.get('classification', r.get('result'))}
            for r in rows[1:5]
        ]
    }


def _patient_biomarker_section(cur, patient_id):
    """Assemble biomarker panel: EEG features + MRI + labs."""
    # EEG-derived biomarkers from analyses
    analyses = _safe_query(cur,
        f"SELECT * FROM analyses WHERE patient_id='{patient_id}' ORDER BY analysis_date DESC LIMIT 1")
    eeg_biomarkers = {}
    if analyses:
        a = analyses[0]
        for key in ('dominant_frequency', 'theta_power', 'alpha_power', 'beta_power',
                     'delta_power', 'gamma_power', 'spectral_entropy', 'asymmetry_index',
                     'spike_rate', 'slowing_index'):
            val = a.get(key)
            if val is not None:
                eeg_biomarkers[key] = val

    # MRI biomarkers
    mri = _safe_query(cur,
        f"SELECT * FROM mri_findings WHERE patient_id='{patient_id}' ORDER BY scan_date DESC LIMIT 1")
    mri_biomarkers = {}
    if mri:
        m = mri[0]
        for key in ('finding_type', 'location', 'laterality', 'clinical_significance'):
            val = m.get(key)
            if val is not None:
                mri_biomarkers[key] = val

    # Assessment-derived biomarkers (cognitive, mood)
    assessments = _safe_query(cur,
        f"SELECT instrument, score, max_score FROM assessments WHERE patient_id='{patient_id}' ORDER BY assessment_date DESC")
    cognitive_markers = {}
    for a in assessments:
        inst = a.get('instrument', '')
        if inst and inst not in cognitive_markers:
            score = a.get('score')
            max_s = a.get('max_score')
            cognitive_markers[inst] = {
                "score": score,
                "max_score": max_s,
                "pct": round(score / max_s * 100, 1) if score and max_s else None
            }

    return {
        "eeg": eeg_biomarkers if eeg_biomarkers else {"note": "No spectral features extracted"},
        "mri": mri_biomarkers if mri_biomarkers else {"note": "No MRI on file"},
        "cognitive_mood": cognitive_markers if cognitive_markers else {"note": "No assessments on file"},
        "biomarker_count": len(eeg_biomarkers) + len(mri_biomarkers) + len(cognitive_markers)
    }


def _patient_xai_section(cur, patient_id):
    """XAI explanations: SHAP importance, explainability ground-truth."""
    xai_rows = _safe_query(cur,
        f"SELECT * FROM explainability_gt WHERE patient_id='{patient_id}'")

    # Clinical decisions with AI reasoning
    decisions = _safe_query(cur,
        f"SELECT * FROM clinical_decisions WHERE patient_id='{patient_id}' ORDER BY created_at DESC")

    feature_importance = []
    explanation_text = []

    for row in xai_rows:
        feat = row.get('feature_name', row.get('feature'))
        imp = row.get('importance', row.get('shap_value'))
        if feat:
            feature_importance.append({"feature": feat, "importance": imp})

    for d in decisions[:3]:
        reason = d.get('ai_recommendation') or d.get('ai_reason') or d.get('reasoning')
        if reason:
            explanation_text.append({
                "date": d.get('created_at'),
                "recommendation": reason,
                "agreement": d.get('neurologist_agreement', 'unknown')
            })

    return {
        "feature_importance": sorted(feature_importance,
            key=lambda x: abs(x.get('importance', 0) or 0), reverse=True)[:10],
        "ai_explanations": explanation_text,
        "xai_available": bool(feature_importance or explanation_text),
        "total_xai_records": len(xai_rows)
    }


def _patient_evidence_section(cur, patient_id):
    """Retrieved evidence from RAG conversations + knowledge base."""
    convos = _safe_query(cur,
        f"SELECT * FROM conversation_log WHERE patient_id='{patient_id}' ORDER BY timestamp DESC")

    evidence_items = []
    query_topics = []

    for c in convos:
        role = c.get('role', '')
        content = c.get('content', c.get('message', ''))
        if not content:
            continue

        if role in ('assistant', 'system', 'ai'):
            # AI-generated responses are evidence
            evidence_items.append({
                "timestamp": c.get('timestamp'),
                "type": "rag_response",
                "content_preview": str(content)[:200],
                "session": c.get('session_id')
            })
        elif role in ('user', 'clinician'):
            query_topics.append(str(content)[:100])

    return {
        "total_conversations": len(convos),
        "evidence_retrieved": len(evidence_items),
        "recent_evidence": evidence_items[:5],
        "query_topics": query_topics[:10],
        "rag_coverage": "full" if len(evidence_items) >= 3 else "partial" if evidence_items else "none"
    }


def _patient_metadata_section(cur, patient_id):
    """Clinical metadata: demographics, medications, seizure history."""
    # Demographics
    patients = _safe_query(cur, f"SELECT * FROM patients WHERE patient_id='{patient_id}'")
    demographics = {}
    if patients:
        p = patients[0]
        for key in ('age', 'gender', 'epilepsy_type', 'onset_age', 'diagnosis',
                     'diagnosis_date', 'ethnicity'):
            val = p.get(key)
            if val is not None:
                demographics[key] = val

    # Medications
    meds = _safe_query(cur,
        f"SELECT drug_name, dose, frequency, start_date, status FROM medications WHERE patient_id='{patient_id}'")

    # Seizure history
    seizures = _safe_query(cur,
        f"SELECT * FROM seizure_diary WHERE patient_id='{patient_id}' ORDER BY event_date DESC")
    seizure_summary = {
        "total_events": len(seizures),
        "recent_3": [
            {"date": s.get('event_date'), "type": s.get('seizure_type'),
             "duration_sec": s.get('duration_seconds'), "severity": s.get('severity')}
            for s in seizures[:3]
        ]
    }

    return {
        "demographics": demographics,
        "medications": meds,
        "seizure_history": seizure_summary,
        "metadata_completeness": sum([
            bool(demographics), bool(meds), bool(seizures)
        ])
    }


def _generate_single_report(cur, patient_id):
    """Assemble a full RAG report for one patient."""
    prediction = _patient_prediction_section(cur, patient_id)
    biomarkers = _patient_biomarker_section(cur, patient_id)
    xai = _patient_xai_section(cur, patient_id)
    evidence = _patient_evidence_section(cur, patient_id)
    metadata = _patient_metadata_section(cur, patient_id)

    # Report quality score
    sections_present = sum([
        prediction.get('available', False),
        biomarkers.get('biomarker_count', 0) > 0,
        xai.get('xai_available', False),
        evidence.get('rag_coverage') != 'none',
        metadata.get('metadata_completeness', 0) >= 2
    ])

    return {
        "patient_id": patient_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_quality": f"{sections_present}/5 sections populated",
        "sections": {
            "prediction": prediction,
            "biomarkers": biomarkers,
            "xai_explanations": xai,
            "retrieved_evidence": evidence,
            "clinical_metadata": metadata
        },
        "summary": {
            "classification": prediction.get('latest_classification', 'N/A'),
            "confidence": prediction.get('confidence'),
            "biomarker_count": biomarkers.get('biomarker_count', 0),
            "xai_features": len(xai.get('feature_importance', [])),
            "evidence_items": evidence.get('evidence_retrieved', 0),
            "seizure_count": metadata.get('seizure_history', {}).get('total_events', 0)
        }
    }


# ── Public API ────────────────────────────────────────────────────

def overview():
    """Aggregate RAG report generation metrics across all patients."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    total_patients = _safe_count(cur, "SELECT COUNT(*) FROM patients")
    total_analyses = _safe_count(cur, "SELECT COUNT(*) FROM analyses")
    total_assessments = _safe_count(cur, "SELECT COUNT(*) FROM assessments")
    total_xai = _safe_count(cur, "SELECT COUNT(*) FROM explainability_gt")
    total_convos = _safe_count(cur, "SELECT COUNT(*) FROM conversation_log")
    total_meds = _safe_count(cur, "SELECT COUNT(*) FROM medications")
    total_seizures = _safe_count(cur, "SELECT COUNT(*) FROM seizure_diary")
    total_mri = _safe_count(cur, "SELECT COUNT(*) FROM mri_findings")
    total_decisions = _safe_count(cur, "SELECT COUNT(*) FROM clinical_decisions")

    # Patients with at least one analysis (reportable)
    reportable = _safe_count(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM analyses")

    # Patients with XAI data
    xai_covered = _safe_count(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM explainability_gt")

    # Patients with RAG conversations
    rag_covered = _safe_count(cur,
        "SELECT COUNT(DISTINCT patient_id) FROM conversation_log")

    # Generate sample reports for quality assessment
    rows = _safe_query(cur, "SELECT DISTINCT patient_id FROM patients ORDER BY patient_id LIMIT 10")
    patient_ids = [r['patient_id'] for r in rows if r.get('patient_id')]

    quality_scores = []
    for pid in patient_ids[:10]:
        report = _generate_single_report(cur, pid)
        q = report.get('report_quality', '0/5')
        try:
            score = int(q.split('/')[0])
        except (ValueError, IndexError):
            score = 0
        quality_scores.append(score)

    avg_quality = round(sum(quality_scores) / len(quality_scores), 1) if quality_scores else 0

    conn.close()

    return {
        "title": "RAG Report Generation — Pipeline Step 20",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_sources": {
            "patients": total_patients,
            "analyses": total_analyses,
            "assessments": total_assessments,
            "xai_records": total_xai,
            "conversations": total_convos,
            "medications": total_meds,
            "seizure_events": total_seizures,
            "mri_scans": total_mri,
            "clinical_decisions": total_decisions
        },
        "report_coverage": {
            "total_patients": total_patients,
            "reportable_with_analysis": reportable,
            "xai_covered": xai_covered,
            "rag_covered": rag_covered,
            "coverage_pct": round(reportable / total_patients * 100, 1) if total_patients else 0
        },
        "report_quality": {
            "sample_size": len(quality_scores),
            "avg_sections_populated": avg_quality,
            "max_possible": 5,
            "quality_pct": round(avg_quality / 5 * 100, 1)
        },
        "report_sections": [
            "Prediction (classification + confidence + risk)",
            "Biomarkers (EEG spectral + MRI + cognitive scales)",
            "XAI Explanations (SHAP features + AI reasoning)",
            "Retrieved Evidence (RAG conversations + citations)",
            "Clinical Metadata (demographics + meds + seizure history)"
        ]
    }


def breakdown(patient_id=None):
    """Per-patient RAG report generation. If patient_id given, single report;
    otherwise all patients with summary."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    if patient_id:
        report = _generate_single_report(cur, int(patient_id))
        conn.close()
        return report

    # All patients — summary table
    rows = _safe_query(cur, "SELECT patient_id FROM patients ORDER BY patient_id")
    reports = []
    for r in rows:
        pid = r['patient_id']
        report = _generate_single_report(cur, pid)
        reports.append({
            "patient_id": pid,
            "report_quality": report['report_quality'],
            "classification": report['summary']['classification'],
            "confidence": report['summary']['confidence'],
            "biomarker_count": report['summary']['biomarker_count'],
            "xai_features": report['summary']['xai_features'],
            "evidence_items": report['summary']['evidence_items'],
            "seizure_count": report['summary']['seizure_count']
        })

    conn.close()

    return {
        "title": "RAG Report Breakdown — All Patients",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_patients": len(reports),
        "reports": reports
    }


def definitions():
    """Clinical definitions and methodology for RAG report generation."""
    return {
        "title": "RAG Report Generation — Definitions & Methodology",
        "pipeline_step": 20,
        "purpose": "Generate structured clinical reports combining prediction, biomarkers, XAI, retrieved evidence, and clinical metadata via RAG pipeline",
        "report_sections": {
            "prediction": {
                "description": "Latest EEG classification/prediction with confidence score and risk level",
                "sources": ["analyses table"],
                "includes": ["classification", "confidence", "risk_score", "method", "prior_analyses"]
            },
            "biomarkers": {
                "description": "Multi-modal biomarker panel: EEG spectral features, MRI findings, cognitive/mood scales",
                "sources": ["analyses", "mri_findings", "assessments"],
                "eeg_features": ["dominant_frequency", "theta/alpha/beta/delta/gamma power",
                                  "spectral_entropy", "asymmetry_index", "spike_rate", "slowing_index"],
                "imaging": ["finding_type", "location", "laterality", "clinical_significance"],
                "cognitive": ["MoCA", "PHQ-9", "GAD-7", "NDDI-E", "QOLIE-31"]
            },
            "xai_explanations": {
                "description": "Explainability layer: SHAP feature importance + AI decision reasoning",
                "sources": ["explainability_gt", "clinical_decisions"],
                "methods": ["SHAP values", "feature importance ranking", "AI recommendation text",
                            "neurologist agreement/override audit"]
            },
            "retrieved_evidence": {
                "description": "RAG-retrieved evidence: knowledge-base citations, conversation history, query topics",
                "sources": ["conversation_log"],
                "coverage_levels": {
                    "full": "≥3 evidence items retrieved",
                    "partial": "1-2 evidence items",
                    "none": "No RAG conversations on file"
                }
            },
            "clinical_metadata": {
                "description": "Patient context: demographics, medications, seizure diary",
                "sources": ["patients", "medications", "seizure_diary"],
                "completeness": "Scored 0-3 (demographics + meds + seizures present)"
            }
        },
        "quality_scoring": {
            "method": "Count of 5 sections with meaningful data",
            "max_score": "5/5",
            "criteria": [
                "Prediction: at least one analysis result",
                "Biomarkers: at least one biomarker value",
                "XAI: feature importance or AI explanation present",
                "Evidence: at least one RAG conversation",
                "Metadata: ≥2 of demographics/meds/seizures present"
            ]
        },
        "references": [
            "EEG AI RAG Pipeline — config/eeg_ai_rag_pipeline.json step 20",
            "Agentic RAG Dashboard — scripts/agentic_rag_dashboard.py",
            "Patient Report Dashboard — scripts/patient_report_dashboard.py",
            "XAI Dashboard — scripts/xai_dashboard.py"
        ]
    }
