"""Mood-Comorbidity View Dashboard — psychiatric comorbidity + PHQ-9/GAD-7 analysis.

Reads real data from:
  - comorbidities table  (behavioral risk score, conditions, treatment status)
  - assessments table    (PHQ-9, GAD-7, NDDI-E scores per patient)
  - patients table       (demographics linkage)
"""
import sqlite3, json, os
from pathlib import Path

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _conn():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


# ── helpers ─────────────────────────────────────────────────────────────────

def _comorbidity_rows():
    """All comorbidity records with parsed JSON fields."""
    with _conn() as conn:
        rows = conn.execute("SELECT patient_id, fields_json FROM comorbidities").fetchall()
    result = []
    for r in rows:
        try:
            d = json.loads(r["fields_json"])
            d["patient_id"] = r["patient_id"]
            result.append(d)
        except Exception:
            pass
    return result


def _assessment_scores(instrument):
    """Return list of {patient_id, score, interpretation, level} for given instrument."""
    with _conn() as conn:
        rows = conn.execute(
            "SELECT patient_id, score, interpretation, level FROM assessments WHERE instrument=?",
            (instrument,)
        ).fetchall()
    return [dict(r) for r in rows]


# ── overview ─────────────────────────────────────────────────────────────────

def overview():
    rows = _comorbidity_rows()
    n_screened = len(rows)
    n_with_comorb = sum(1 for r in rows if r.get("comorbidity_count", 0) > 0)

    # Risk severity distribution
    severity_dist = {}
    for r in rows:
        s = r.get("risk_severity", "unknown")
        severity_dist[s] = severity_dist.get(s, 0) + 1

    # Average risk score
    scores = [r.get("behavioral_risk_score", 0) for r in rows if r.get("behavioral_risk_score") is not None]
    avg_risk = round(sum(scores) / len(scores), 1) if scores else 0

    # Treatment status distribution
    tx_dist = {}
    for r in rows:
        t = r.get("treatment_status", "unknown")
        tx_dist[t] = tx_dist.get(t, 0) + 1

    # Top comorbidity conditions
    cond_counts = {}
    for r in rows:
        for c in r.get("comorbidities", []):
            cond_counts[c] = cond_counts.get(c, 0) + 1
    top_conditions = sorted(cond_counts.items(), key=lambda x: -x[1])[:10]

    # PHQ-9 distribution
    phq_rows = _assessment_scores("PHQ9")
    phq_dist = {}
    for r in phq_rows:
        interp = r.get("interpretation", "Unknown")
        phq_dist[interp] = phq_dist.get(interp, 0) + 1
    phq_avg = round(sum(r["score"] for r in phq_rows if r["score"] is not None) / len(phq_rows), 1) if phq_rows else 0

    # GAD-7 distribution
    gad_rows = _assessment_scores("GAD7")
    gad_dist = {}
    for r in gad_rows:
        interp = r.get("interpretation", "Unknown")
        gad_dist[interp] = gad_dist.get(interp, 0) + 1
    gad_avg = round(sum(r["score"] for r in gad_rows if r["score"] is not None) / len(gad_rows), 1) if gad_rows else 0

    # NDDI-E (depression screening specific to epilepsy)
    nddie_rows = _assessment_scores("NDDIE")
    nddie_positive = sum(1 for r in nddie_rows if "depression" in (r.get("interpretation") or "").lower())

    return {
        "kpis": {
            "n_screened": n_screened,
            "n_with_comorbidity": n_with_comorb,
            "comorbidity_rate_pct": round(n_with_comorb / n_screened * 100, 1) if n_screened else 0,
            "avg_behavioral_risk_score": avg_risk,
            "phq9_avg": phq_avg,
            "gad7_avg": gad_avg,
            "nddie_depression_positive": nddie_positive,
            "treatment_resistant_n": tx_dist.get("treatment_resistant", 0),
        },
        "severity_distribution": [
            {"severity": k, "count": v}
            for k, v in sorted(severity_dist.items(), key=lambda x: -x[1])
        ],
        "treatment_status_distribution": [
            {"status": k.replace("_", " ").title(), "count": v}
            for k, v in sorted(tx_dist.items(), key=lambda x: -x[1])
        ],
        "top_conditions": [
            {"condition": name, "count": cnt}
            for name, cnt in top_conditions
        ],
        "phq9_distribution": [
            {"category": k, "count": v}
            for k, v in sorted(phq_dist.items(), key=lambda x: -x[1])
        ],
        "gad7_distribution": [
            {"category": k, "count": v}
            for k, v in sorted(gad_dist.items(), key=lambda x: -x[1])
        ],
    }


# ── breakdown ────────────────────────────────────────────────────────────────

def breakdown():
    rows = _comorbidity_rows()
    phq_rows = {r["patient_id"]: r for r in _assessment_scores("PHQ9")}
    gad_rows = {r["patient_id"]: r for r in _assessment_scores("GAD7")}
    nddie_rows = {r["patient_id"]: r for r in _assessment_scores("NDDIE")}

    per_patient = []
    for r in rows:
        pid = r["patient_id"]
        phq = phq_rows.get(pid, {})
        gad = gad_rows.get(pid, {})
        nddie = nddie_rows.get(pid, {})
        per_patient.append({
            "patient_id": pid,
            "comorbidities": r.get("comorbidities", []),
            "comorbidity_count": r.get("comorbidity_count", 0),
            "behavioral_risk_score": r.get("behavioral_risk_score"),
            "risk_severity": r.get("risk_severity", "unknown"),
            "treatment_status": r.get("treatment_status", "unknown"),
            "phq9_score": phq.get("score"),
            "phq9_interpretation": phq.get("interpretation"),
            "gad7_score": gad.get("score"),
            "gad7_interpretation": gad.get("interpretation"),
            "nddie_score": nddie.get("score"),
            "nddie_interpretation": nddie.get("interpretation"),
            "screening_date": r.get("screening_date"),
        })

    # Sort by behavioral risk score descending
    per_patient.sort(key=lambda x: -(x.get("behavioral_risk_score") or 0))

    # Condition co-occurrence matrix (top 6)
    cond_counts = {}
    for r in rows:
        for c in r.get("comorbidities", []):
            cond_counts[c] = cond_counts.get(c, 0) + 1
    top6 = [c for c, _ in sorted(cond_counts.items(), key=lambda x: -x[1])[:6]]

    # Co-occurrence pairs
    cooccurrence = []
    for i, c1 in enumerate(top6):
        for c2 in top6[i+1:]:
            count = sum(
                1 for r in rows
                if c1 in r.get("comorbidities", []) and c2 in r.get("comorbidities", [])
            )
            if count > 0:
                cooccurrence.append({"condition_a": c1, "condition_b": c2, "count": count})

    # PHQ-9 vs GAD-7 scatter data
    scatter = []
    for r in rows:
        pid = r["patient_id"]
        phq = phq_rows.get(pid, {})
        gad = gad_rows.get(pid, {})
        if phq.get("score") is not None and gad.get("score") is not None:
            scatter.append({
                "patient_id": pid,
                "phq9": phq["score"],
                "gad7": gad["score"],
                "risk_severity": r.get("risk_severity", "unknown"),
            })

    return {
        "per_patient": per_patient,
        "condition_cooccurrence": cooccurrence,
        "phq9_vs_gad7_scatter": scatter,
    }


# ── definitions ──────────────────────────────────────────────────────────────

def definitions():
    return {
        "instruments": [
            {
                "name": "PHQ-9",
                "full_name": "Patient Health Questionnaire-9",
                "purpose": "Screens for major depressive disorder severity",
                "scoring": "0–27; Minimal <5, Mild 5–9, Moderate 10–14, Moderately Severe 15–19, Severe ≥20",
                "relevance": "Depression prevalence in epilepsy is 2–3× the general population; PHQ-9 ≥10 warrants clinical intervention",
            },
            {
                "name": "GAD-7",
                "full_name": "Generalized Anxiety Disorder-7",
                "purpose": "Screens for generalized anxiety disorder severity",
                "scoring": "0–21; Minimal <5, Mild 5–9, Moderate 10–14, Severe ≥15",
                "relevance": "Anxiety affects ~20% of people with epilepsy; seizure-related anxiety (peri-ictal) common",
            },
            {
                "name": "NDDI-E",
                "full_name": "Neurological Disorders Depression Inventory for Epilepsy",
                "purpose": "Epilepsy-specific depression screen that avoids overlap with AED side effects",
                "scoring": "6–24; Likely major depression >15",
                "relevance": "More specific than PHQ-9 in epilepsy; avoids false positives from AED-related fatigue/cognitive effects",
            },
        ],
        "risk_severity_tiers": [
            {"tier": "Minimal", "score_range": "0–20", "description": "No significant psychiatric burden; monitoring sufficient"},
            {"tier": "Mild", "score_range": "21–40", "description": "Subclinical symptoms; psychoeducation + watchful waiting"},
            {"tier": "Moderate", "score_range": "41–60", "description": "Clinical intervention recommended; refer to psychiatry"},
            {"tier": "Severe", "score_range": "61–100", "description": "Urgent psychiatric consultation; may require inpatient evaluation"},
        ],
        "key_comorbidities": [
            {
                "condition": "Major Depressive Disorder",
                "prevalence_epilepsy": "~30–35%",
                "mechanism": "Shared serotonergic/dopaminergic dysfunction; bidirectional relationship with seizure threshold",
                "governance_note": "Must be documented in AI audit trail; AED selection affects mood",
            },
            {
                "condition": "Generalized Anxiety Disorder",
                "prevalence_epilepsy": "~20%",
                "mechanism": "Chronic unpredictability of seizures; amygdala hyperactivation; peri-ictal anxiety distinct from GAD",
                "governance_note": "AI-assisted seizure prediction may reduce anticipatory anxiety burden",
            },
            {
                "condition": "PNES (Psychogenic Non-Epileptic Seizures)",
                "prevalence_epilepsy": "~10–20% of refractory cases",
                "mechanism": "Functional neurological disorder; requires video-EEG for definitive diagnosis",
                "governance_note": "AI differential model must flag for human review — misclassification carries significant harm",
            },
            {
                "condition": "PTSD",
                "prevalence_epilepsy": "Elevated in temporal lobe epilepsy",
                "mechanism": "Traumatic seizure events; stigma; social isolation",
                "governance_note": "C-SSRS mandatory in high-PTSD burden cohorts",
            },
        ],
        "ai_governance_notes": [
            "Psychiatric comorbidity screening results must appear in the AI audit trail for each prediction.",
            "PHQ-9 ≥15 or GAD-7 ≥15 should trigger automatic human-override flag on AI seizure risk output.",
            "NDDI-E >15 requires mandatory psychiatry consult note before discharge.",
            "Treatment-resistant comorbidities elevate AI model uncertainty — confidence threshold should be raised.",
            "Mood data is protected under DPDP Act 2023 (India) and HIPAA — encrypted at rest, de-identified in exports.",
        ],
        "references": [
            "Kanner AM. Depression in Epilepsy: Prevalence, Clinical Semiology, Pathogenic Mechanisms, and Treatment. Biol Psychiatry. 2003.",
            "Scott AJ et al. Anxiety in epilepsy: A systematic review. Epilepsy Behav. 2017.",
            "Ettinger AB et al. NDDI-E: A reliable and efficient tool. Epilepsia. 2006.",
            "Fiest KM et al. Prevalence and incidence of depression in epilepsy. Neurology. 2013.",
        ],
    }
