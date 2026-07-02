"""AI Risk Management Dashboard — risk identification, assessment, mitigation
from real clinical.db data.

Sources:
- assessments (risk levels: normal/mild/moderate/severe/critical, alert flags)
- seizure_diary (seizure frequency per patient — elevated risk indicator)
- medications (multi-medication regimen complexity)
- clinical_decisions (artifact_risk, ai_confidence)
- model_governance (governance records)
- transaction_log (risk-related events: risk/alert/escalat/block)
- expert_reviews (disagree-with-AI as risk indicator)
- hitl_reviews (mitigation actions)
"""

import sqlite3
import os
import json
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
CONFIG = os.path.join(os.path.dirname(__file__), '..', 'config')

LEVEL_MAP = {'normal': 0, 'none': 0, 'mild': 1, 'moderate': 2, 'severe': 3,
             'critical': 4, 'high': 3}


def _conn():
    return sqlite3.connect(DB)


def _safe_count(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchone()[0]
    except Exception:
        return 0


def _safe_query(cur, sql):
    try:
        cur.execute(sql)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]
    except Exception:
        return []


def risk_overview():
    """Aggregate risk posture from real clinical.db data — KPIs, severity
    distribution, risk categories, trend, recent risk events."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── Assessments: risk levels ──
    assessments = _safe_query(cur,
        "SELECT patient_id, level, alert, created_at FROM assessments")
    level_counts = {}
    for a in assessments:
        lvl = (a.get('level') or 'unknown').lower()
        level_counts[lvl] = level_counts.get(lvl, 0) + 1

    severe_critical = sum(1 for a in assessments
                          if (a.get('level') or '').lower()
                          in ('severe', 'critical', 'high'))
    alert_count = sum(1 for a in assessments
                      if a.get('alert') and str(a['alert']).strip() != '')

    # Patients with severe/critical assessments or alerts
    high_risk_patients = set()
    for a in assessments:
        lvl = (a.get('level') or '').lower()
        if lvl in ('severe', 'critical', 'high'):
            high_risk_patients.add(a['patient_id'])
        if a.get('alert') and str(a['alert']).strip() != '':
            high_risk_patients.add(a['patient_id'])

    # ── Seizure diary: high-frequency patients ──
    seizure_freq = _safe_query(cur,
        "SELECT patient_id, count(*) as cnt FROM seizure_diary GROUP BY patient_id")
    for s in seizure_freq:
        if s['cnt'] >= 3:
            high_risk_patients.add(s['patient_id'])

    # ── Medications: multi-medication complexity ──
    med_per_patient = _safe_query(cur,
        "SELECT patient_id, count(*) as cnt FROM medications GROUP BY patient_id")
    multi_med_patients = sum(1 for m in med_per_patient if m['cnt'] >= 3)

    # ── Clinical decisions: artifact_risk, ai_confidence ──
    decisions = _safe_query(cur, "SELECT * FROM clinical_decisions")
    low_confidence_count = sum(1 for d in decisions
                               if d.get('ai_confidence') is not None
                               and d['ai_confidence'] < 0.5)
    artifact_risks = {}
    for d in decisions:
        ar = (d.get('artifact_risk') or 'Unknown').strip()
        artifact_risks[ar] = artifact_risks.get(ar, 0) + 1

    # ── Model governance ──
    gov_records = _safe_count(cur, "SELECT count(*) FROM model_governance")

    # ── Transaction log: risk-related events ──
    risk_events = _safe_query(cur,
        "SELECT * FROM transaction_log "
        "WHERE action LIKE '%risk%' OR action LIKE '%alert%' "
        "OR action LIKE '%escalat%' OR action LIKE '%block%' "
        "ORDER BY ts_utc DESC")
    risk_events_count = len(risk_events)

    # ── Expert reviews: disagree = risk indicator ──
    expert_disagree = _safe_count(cur,
        "SELECT count(*) FROM expert_reviews WHERE agree_with_ai = 'disagree'")
    total_expert = _safe_count(cur, "SELECT count(*) FROM expert_reviews")

    # ── Mitigation counts ──
    total_hitl = _safe_count(cur, "SELECT count(*) FROM hitl_reviews")

    # ── KPIs ──
    total_risks_identified = severe_critical + alert_count + expert_disagree
    patients_at_risk = len(high_risk_patients)
    mitigations = total_hitl + total_expert
    risk_mitigation_rate = round(
        mitigations / max(total_risks_identified, 1) * 100, 1)
    if risk_mitigation_rate > 100:
        risk_mitigation_rate = 100.0
    open_risks = max(total_risks_identified - mitigations, 0)

    # Average severity
    severity_values = []
    for a in assessments:
        lvl = (a.get('level') or '').lower()
        if lvl in LEVEL_MAP:
            severity_values.append(LEVEL_MAP[lvl])
    avg_severity = round(
        sum(severity_values) / max(len(severity_values), 1), 2)

    kpis = {
        "total_risks_identified": total_risks_identified,
        "patients_at_risk": patients_at_risk,
        "risk_mitigation_rate": risk_mitigation_rate,
        "avg_assessment_severity": avg_severity,
        "open_risks": open_risks,
        "risk_events_30d": risk_events_count,
        "medication_risk_flags": multi_med_patients,
        "ai_confidence_risk": low_confidence_count,
    }

    # ── Risk by category ──
    clinical_risks = severe_critical + alert_count
    ai_model_risks = low_confidence_count + expert_disagree
    data_quality_risks = sum(1 for ar in artifact_risks
                             if ar.lower() not in ('none', 'unknown', ''))
    operational_risks = risk_events_count
    compliance_risks = max(0, gov_records)

    risk_by_category = [
        {"name": "Clinical", "value": clinical_risks},
        {"name": "AI Model", "value": ai_model_risks},
        {"name": "Data Quality", "value": data_quality_risks},
        {"name": "Operational", "value": operational_risks},
        {"name": "Compliance", "value": compliance_risks},
    ]

    # ── Severity distribution ──
    sev_order = ['normal', 'none', 'mild', 'moderate', 'severe', 'critical']
    severity_distribution = []
    for lvl in sev_order:
        if lvl in level_counts:
            severity_distribution.append(
                {"name": lvl.capitalize(), "count": level_counts[lvl]})
    for lvl, cnt in sorted(level_counts.items()):
        if lvl not in sev_order:
            severity_distribution.append(
                {"name": lvl.capitalize(), "count": cnt})

    # ── Risk trend (grouped by month from assessments created_at) ──
    monthly = {}
    for a in assessments:
        dt = a.get('created_at', '')
        if dt:
            month_key = dt[:7]  # YYYY-MM
            lvl = (a.get('level') or '').lower()
            has_alert = a.get('alert') and str(a['alert']).strip() != ''
            if lvl in ('severe', 'critical', 'high') or has_alert:
                monthly[month_key] = monthly.get(month_key, 0) + 1
    risk_trend = [{"date": k, "risks": v}
                  for k, v in sorted(monthly.items())]

    # ── Recent risk events (last 20) ──
    recent_risk_events = []
    for e in risk_events[:20]:
        recent_risk_events.append({
            "id": e.get("id"),
            "patient_id": e.get("patient_id"),
            "component": e.get("component"),
            "action": e.get("action"),
            "actor": e.get("actor"),
            "detail": e.get("detail"),
            "timestamp": e.get("ts_utc"),
        })

    conn.close()

    return {
        "available": True,
        "kpis": kpis,
        "risk_by_category": risk_by_category,
        "severity_distribution": severity_distribution,
        "risk_trend": risk_trend,
        "recent_risk_events": recent_risk_events,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def risk_breakdown():
    """Detailed per-patient risk profiles, risk register, 5x5 risk matrix,
    mitigation log from hitl_reviews + expert_reviews."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()

    # ── Per-patient risk profiles ──
    patients_assessments = _safe_query(cur,
        "SELECT patient_id, level, alert FROM assessments")
    seizure_counts = {}
    for row in _safe_query(cur,
            "SELECT patient_id, count(*) as cnt "
            "FROM seizure_diary GROUP BY patient_id"):
        seizure_counts[row['patient_id']] = row['cnt']

    patient_data = {}
    for a in patients_assessments:
        pid = a['patient_id']
        if pid not in patient_data:
            patient_data[pid] = {"levels": [], "alerts": []}
        lvl = (a.get('level') or '').lower()
        patient_data[pid]["levels"].append(lvl)
        if a.get('alert') and str(a['alert']).strip() != '':
            patient_data[pid]["alerts"].append(a['alert'])

    per_patient_risks = []
    for pid, data in patient_data.items():
        score = 0
        risk_factors = []
        # Assessment severity contribution (up to 60)
        for lvl in data["levels"]:
            score += LEVEL_MAP.get(lvl, 0) * 5
        if score > 60:
            score = 60
        # Alert contribution (up to 20)
        alert_bonus = min(len(data["alerts"]) * 10, 20)
        score += alert_bonus
        if data["alerts"]:
            risk_factors.append(f"{len(data['alerts'])} clinical alerts")
        # Seizure contribution (up to 20)
        sz = seizure_counts.get(pid, 0)
        if sz > 0:
            score += min(sz * 5, 20)
            risk_factors.append(f"{sz} seizure events")
        # Severity factors
        sev_count = sum(1 for l in data["levels"]
                        if l in ('severe', 'critical', 'high'))
        if sev_count:
            risk_factors.append(
                f"{sev_count} severe/critical assessments")
        mod_count = sum(1 for l in data["levels"] if l == 'moderate')
        if mod_count:
            risk_factors.append(f"{mod_count} moderate assessments")

        score = min(score, 100)
        mitigation = "reviewed" if score < 50 else "needs_review"

        per_patient_risks.append({
            "patient_id": pid,
            "risk_score": score,
            "risk_factors": risk_factors,
            "mitigation_status": mitigation,
            "assessment_count": len(data["levels"]),
            "alert_count": len(data["alerts"]),
            "seizure_count": sz,
        })
    per_patient_risks.sort(key=lambda x: x['risk_score'], reverse=True)

    # ── Risk register ──
    risk_register = []
    rid = 1

    # Severe/critical assessments
    severe_rows = _safe_query(cur,
        "SELECT id, patient_id, instrument, level, alert, interpretation, "
        "created_at FROM assessments "
        "WHERE level IN ('severe','critical','high') "
        "ORDER BY created_at DESC")
    for a in severe_rows:
        lvl = (a.get('level') or '').lower()
        risk_register.append({
            "id": rid,
            "category": "Clinical",
            "description": (f"{a.get('instrument', 'Assessment')}: "
                            f"{a.get('interpretation', lvl)} ({lvl})"),
            "severity": lvl.capitalize(),
            "likelihood": "High" if lvl == 'critical' else "Medium",
            "impact": "Critical" if lvl == 'critical' else "High",
            "mitigation": a.get('alert') or "Standard monitoring",
            "status": "Open",
            "patient_id": a.get('patient_id'),
        })
        rid += 1

    # Alert-based risks
    alert_rows = _safe_query(cur,
        "SELECT id, patient_id, instrument, level, alert, created_at "
        "FROM assessments "
        "WHERE alert IS NOT NULL AND alert != '' "
        "ORDER BY created_at DESC")
    for a in alert_rows:
        risk_register.append({
            "id": rid,
            "category": "Clinical",
            "description": f"Alert: {a.get('alert', 'Unknown alert')}",
            "severity": "High",
            "likelihood": "Medium",
            "impact": "High",
            "mitigation": "Clinical review required",
            "status": "Open",
            "patient_id": a.get('patient_id'),
        })
        rid += 1

    # Expert disagreements
    disagrees = _safe_query(cur,
        "SELECT id, patient_id, role, expert, finding, note, created_at "
        "FROM expert_reviews WHERE agree_with_ai = 'disagree'")
    for d in disagrees:
        risk_register.append({
            "id": rid,
            "category": "AI Model",
            "description": (f"Expert ({d.get('role')}) disagrees with AI: "
                            f"{d.get('finding', 'N/A')}"),
            "severity": "Medium",
            "likelihood": "Medium",
            "impact": "Medium",
            "mitigation": d.get('note') or "Expert review override",
            "status": "Mitigated",
            "patient_id": d.get('patient_id'),
        })
        rid += 1

    # Low-confidence AI decisions
    low_conf = _safe_query(cur,
        "SELECT id, patient_id, ai_prediction, ai_confidence, artifact_risk "
        "FROM clinical_decisions WHERE ai_confidence < 0.5")
    for lc in low_conf:
        risk_register.append({
            "id": rid,
            "category": "AI Model",
            "description": (f"Low confidence ({lc.get('ai_confidence')}) "
                            f"prediction: {lc.get('ai_prediction', 'N/A')}"),
            "severity": "Medium",
            "likelihood": "High",
            "impact": "Medium",
            "mitigation": "Mandatory human review",
            "status": "Open",
            "patient_id": lc.get('patient_id'),
        })
        rid += 1

    # Blocked operations from transaction_log
    blocked = _safe_query(cur,
        "SELECT id, patient_id, component, detail, ts_utc "
        "FROM transaction_log WHERE action = 'blocked'")
    for b in blocked:
        risk_register.append({
            "id": rid,
            "category": "Operational",
            "description": (f"Blocked action in {b.get('component', 'system')}: "
                            f"{b.get('detail', 'guardrail triggered')}"),
            "severity": "Medium",
            "likelihood": "Low",
            "impact": "Medium",
            "mitigation": "Guardrail active — blocked and logged",
            "status": "Mitigated",
            "patient_id": b.get('patient_id'),
        })
        rid += 1

    # ── Risk matrix (5x5: likelihood x impact) ──
    likelihood_levels = ["Very Low", "Low", "Medium", "High", "Very High"]
    impact_levels = ["Negligible", "Low", "Medium", "High", "Critical"]
    risk_matrix = []
    for li, lname in enumerate(likelihood_levels):
        for ii, iname in enumerate(impact_levels):
            count = sum(1 for r in risk_register
                        if r['likelihood'] == lname and r['impact'] == iname)
            risk_matrix.append({
                "likelihood": lname,
                "impact": iname,
                "likelihood_idx": li,
                "impact_idx": ii,
                "count": count,
                "severity_score": (li + 1) * (ii + 1),
            })

    # ── Mitigation log from hitl_reviews + expert_reviews ──
    mitigation_log = []

    hitl_reviews = _safe_query(cur,
        "SELECT id, patient_id, fields_json, created_at "
        "FROM hitl_reviews ORDER BY created_at DESC")
    for h in hitl_reviews:
        fj = {}
        try:
            fj = json.loads(h.get('fields_json', '{}'))
        except Exception:
            pass
        mitigation_log.append({
            "id": h.get('id'),
            "source": "HITL Review",
            "patient_id": h.get('patient_id'),
            "action": fj.get('decision', 'review'),
            "detail": (fj.get('human_decision')
                       or fj.get('reason_code', '')),
            "ai_prediction": fj.get('ai_prediction', ''),
            "date": h.get('created_at'),
        })

    expert_reviews = _safe_query(cur,
        "SELECT id, patient_id, role, expert, finding, agree_with_ai, "
        "note, created_at FROM expert_reviews ORDER BY created_at DESC")
    for e in expert_reviews:
        mitigation_log.append({
            "id": e.get('id'),
            "source": "Expert Review",
            "patient_id": e.get('patient_id'),
            "action": (f"{e.get('agree_with_ai', 'review')} "
                       f"({e.get('role', '')})"),
            "detail": e.get('finding', ''),
            "ai_prediction": '',
            "date": e.get('created_at'),
        })

    mitigation_log.sort(key=lambda x: x.get('date') or '', reverse=True)

    conn.close()

    return {
        "available": True,
        "per_patient_risks": per_patient_risks,
        "risk_register": risk_register,
        "risk_matrix": risk_matrix,
        "mitigation_log": mitigation_log,
    }


def risk_definitions():
    """AI Risk Management concepts, metrics, clinical relevance, remediation."""
    return {
        "concepts": [
            {"term": "Clinical Risk",
             "definition": "Risk arising from patient assessment severity "
                           "levels (severe/critical), seizure frequency, and "
                           "clinical alerts requiring immediate attention."},
            {"term": "AI Model Risk",
             "definition": "Risk from low-confidence AI predictions, expert "
                           "disagreements with AI outputs, and artifact "
                           "contamination in EEG data."},
            {"term": "Data Quality Risk",
             "definition": "Risk from EEG artifacts, missing data, signal "
                           "quality issues that may compromise AI model "
                           "accuracy."},
            {"term": "Operational Risk",
             "definition": "Risk from system events including blocked "
                           "actions, escalations, and process failures "
                           "logged in the transaction pipeline."},
            {"term": "Compliance Risk",
             "definition": "Risk from gaps in model governance, audit "
                           "trails, and regulatory documentation "
                           "requirements."},
            {"term": "Risk Score",
             "definition": "Composite 0-100 score per patient combining "
                           "assessment severity, seizure frequency, alert "
                           "flags, and AI confidence levels."},
            {"term": "Risk Matrix",
             "definition": "5x5 grid mapping likelihood (Very Low to Very "
                           "High) against impact (Negligible to Critical) "
                           "for systematic risk prioritization."},
            {"term": "Risk Register",
             "definition": "Comprehensive catalog of all identified risks "
                           "with category, severity, likelihood, impact, "
                           "mitigation strategy, and current status."},
        ],
        "metrics": [
            {"metric": "Total Risks Identified",
             "description": "Count of severe/critical assessments + clinical "
                            "alerts + expert disagreements.",
             "formula": "severe_count + critical_count + alert_count + "
                        "expert_disagree_count"},
            {"metric": "Patients at Risk",
             "description": "Distinct patients with any severe/critical "
                            "assessment, clinical alert, or high seizure "
                            "frequency.",
             "formula": "DISTINCT(patient_id) WHERE level IN "
                        "(severe, critical) OR alert != '' OR "
                        "seizure_count >= 3"},
            {"metric": "Risk Mitigation Rate",
             "description": "Percentage of identified risks addressed "
                            "through HITL or expert review.",
             "formula": "(hitl_reviews + expert_reviews) / "
                        "total_risks * 100"},
            {"metric": "Average Severity",
             "description": "Numeric mean of assessment severity "
                            "(normal=0, mild=1, moderate=2, severe=3, "
                            "critical=4).",
             "formula": "AVG(LEVEL_MAP[level])"},
            {"metric": "AI Confidence Risk",
             "description": "Count of AI predictions with confidence below "
                            "0.5, indicating unreliable outputs.",
             "formula": "COUNT(*) WHERE ai_confidence < 0.5"},
            {"metric": "Medication Risk Flags",
             "description": "Patients on 3+ concurrent medications "
                            "indicating polypharmacy complexity risk.",
             "formula": "COUNT(DISTINCT patient_id) WHERE "
                        "medication_count >= 3"},
        ],
        "clinical_relevance": [
            {"standard": "EU AI Act (Article 9)",
             "requirement": "High-risk AI systems shall have a risk "
                            "management system established, implemented, "
                            "documented, and maintained. Risk identification, "
                            "estimation, evaluation, and mitigation must be "
                            "continuous."},
            {"standard": "ISO 14971:2019",
             "requirement": "Application of risk management to medical "
                            "devices — requires risk analysis, risk "
                            "evaluation, risk control, and monitoring of "
                            "residual risk throughout product lifecycle."},
            {"standard": "IEC 62304",
             "requirement": "Medical device software lifecycle processes — "
                            "risk management activities must be integrated "
                            "into software development and maintenance."},
            {"standard": "FDA AI/ML SaMD",
             "requirement": "Good Machine Learning Practice requires "
                            "continuous monitoring of AI model performance, "
                            "risk categorization, and predetermined change "
                            "control protocols."},
            {"standard": "NIST AI RMF 1.0",
             "requirement": "AI Risk Management Framework — MAP, MEASURE, "
                            "MANAGE, GOVERN functions for trustworthy AI "
                            "with emphasis on risk identification and "
                            "mitigation."},
            {"standard": "ILAE Guidelines",
             "requirement": "International League Against Epilepsy "
                            "classification standards require validated "
                            "clinical assessments with documented risk "
                            "escalation pathways."},
        ],
        "remediation": [
            {"risk_type": "High Clinical Severity",
             "strategy": "Implement automated escalation for severe/critical "
                         "assessments. Ensure mandatory specialist review "
                         "within 24 hours. Deploy safety intervention "
                         "protocols for immediate-risk alerts."},
            {"risk_type": "Low AI Confidence",
             "strategy": "Flag predictions below 0.5 confidence for "
                         "mandatory human review. Retrain model on edge "
                         "cases. Add uncertainty quantification to output."},
            {"risk_type": "Expert Disagreement",
             "strategy": "Trigger multi-expert panel review. Document "
                         "disagreement rationale. Feed corrections back "
                         "into model training pipeline."},
            {"risk_type": "Data Quality Issues",
             "strategy": "Implement real-time artifact detection. Reject or "
                         "flag contaminated EEG segments. Add data quality "
                         "gates before AI inference."},
            {"risk_type": "Medication Polypharmacy",
             "strategy": "Flag patients on 3+ AEDs for pharmacology review. "
                         "Cross-reference drug interactions. Monitor for "
                         "treatment-emergent adverse effects."},
            {"risk_type": "Operational Failures",
             "strategy": "Implement circuit breakers for blocked actions. "
                         "Automated incident reporting. Regular operational "
                         "risk reviews with documented remediation plans."},
        ],
    }


if __name__ == "__main__":
    print("=== OVERVIEW ===")
    print(json.dumps(risk_overview(), indent=2, default=str))
    print("\n=== BREAKDOWN ===")
    print(json.dumps(risk_breakdown(), indent=2, default=str))
    print("\n=== DEFINITIONS ===")
    print(json.dumps(risk_definitions(), indent=2, default=str))
