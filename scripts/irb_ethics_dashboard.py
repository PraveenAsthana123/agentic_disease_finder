"""
IRB / Ethics Officer Dashboard
==============================
Real analytics from clinical.db for Institutional Review Board oversight:
  - Protocol compliance tracking (workflow step completion)
  - Informed consent coverage (assessment-based proxy)
  - Risk-benefit analysis (AI decisions vs human overrides)
  - AI oversight summary (expert agreement rates)
  - Audit trail completeness (transaction log coverage)
  - Vulnerable population flagging (age-based IRB scrutiny)
  - Per-patient ethics profiles with risk classification
"""
import json
import sqlite3
from pathlib import Path
from collections import defaultdict
from datetime import datetime

DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _connect():
    if not DB.exists():
        return None
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    return conn


# ════════════════════════════════════════════════════════════════════════
# 1. overview()
# ════════════════════════════════════════════════════════════════════════

def overview():
    """KPI cards + chart data for the IRB / Ethics Officer dashboard."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found"}

    cur = conn.cursor()

    # ── Basic counts ────────────────────────────────────────────────────
    total_patients = cur.execute("SELECT COUNT(*) FROM patients").fetchone()[0]
    total_actions = cur.execute("SELECT COUNT(*) FROM transaction_log").fetchone()[0]
    expert_reviews_count = cur.execute("SELECT COUNT(*) FROM expert_reviews").fetchone()[0]
    ai_decisions_count = cur.execute("SELECT COUNT(*) FROM clinical_decisions").fetchone()[0]
    hitl_reviews_count = cur.execute("SELECT COUNT(*) FROM hitl_reviews").fetchone()[0]
    assessments_count = cur.execute("SELECT COUNT(*) FROM assessments").fetchone()[0]

    # ── Informed consent coverage (patients with at least one assessment) ─
    patients_with_assessments = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM assessments"
    ).fetchone()[0]
    consent_coverage_pct = round(
        (patients_with_assessments / max(total_patients, 1)) * 100, 1
    )

    # ── Audit trail completeness (patients with transaction_log entries) ─
    patients_with_txn = cur.execute(
        "SELECT COUNT(DISTINCT patient_id) FROM transaction_log WHERE patient_id IS NOT NULL"
    ).fetchone()[0]
    audit_completeness_pct = round(
        (patients_with_txn / max(total_patients, 1)) * 100, 1
    )

    # ── KPI cards ───────────────────────────────────────────────────────
    kpis = {
        "total_patients_enrolled": total_patients,
        "total_data_actions": total_actions,
        "expert_reviews_conducted": expert_reviews_count,
        "ai_decisions_made": ai_decisions_count,
        "human_overrides": hitl_reviews_count,
        "assessments_administered": assessments_count,
        "informed_consent_coverage_pct": consent_coverage_pct,
        "audit_trail_completeness_pct": audit_completeness_pct,
    }

    # ── Protocol compliance ─────────────────────────────────────────────
    # Check how many patients completed each key workflow step:
    # upload -> assessment -> expert_review -> clinical_decision
    all_patient_ids = [r[0] for r in cur.execute("SELECT patient_id FROM patients").fetchall()]

    patients_with_upload = set(
        r[0] for r in cur.execute("SELECT DISTINCT patient_id FROM uploads").fetchall()
    )
    patients_with_assessment = set(
        r[0] for r in cur.execute("SELECT DISTINCT patient_id FROM assessments").fetchall()
    )
    patients_with_expert_review = set(
        r[0] for r in cur.execute("SELECT DISTINCT patient_id FROM expert_reviews").fetchall()
    )
    patients_with_clinical_decision = set(
        r[0] for r in cur.execute("SELECT DISTINCT patient_id FROM clinical_decisions").fetchall()
    )

    protocol_compliance = {
        "workflow_steps": [
            {"step": "Upload (EEG/data)",            "patients_completed": len(patients_with_upload),            "total": total_patients, "pct": round(len(patients_with_upload) / max(total_patients, 1) * 100, 1)},
            {"step": "Assessment administered",      "patients_completed": len(patients_with_assessment),        "total": total_patients, "pct": round(len(patients_with_assessment) / max(total_patients, 1) * 100, 1)},
            {"step": "Expert review conducted",      "patients_completed": len(patients_with_expert_review),     "total": total_patients, "pct": round(len(patients_with_expert_review) / max(total_patients, 1) * 100, 1)},
            {"step": "Clinical decision recorded",   "patients_completed": len(patients_with_clinical_decision), "total": total_patients, "pct": round(len(patients_with_clinical_decision) / max(total_patients, 1) * 100, 1)},
        ],
        "fully_compliant_patients": len(
            patients_with_upload & patients_with_assessment & patients_with_expert_review & patients_with_clinical_decision
        ),
    }

    # ── Consent tracking (per-patient) ──────────────────────────────────
    consent_tracking = []
    for pid in all_patient_ids:
        has = pid in patients_with_assessment
        consent_tracking.append({"patient_id": pid, "has_consent": has})

    # ── Risk-benefit summary ────────────────────────────────────────────
    cd_rows = cur.execute(
        "SELECT ai_prediction, ai_confidence, neurologist_agreement, final_decision FROM clinical_decisions"
    ).fetchall()

    accepted = 0
    overridden = 0
    confidences = []
    for row in cd_rows:
        conf = row["ai_confidence"]
        if conf is not None:
            confidences.append(conf)
        agreement = (row["neurologist_agreement"] or "").lower()
        if agreement in ("yes", "agree", "true", "1"):
            accepted += 1
        else:
            overridden += 1

    # Include HITL overrides
    hitl_total = hitl_reviews_count

    conf_buckets = {"high_gte_0.8": 0, "medium_0.5_0.8": 0, "low_lt_0.5": 0}
    for c in confidences:
        if c >= 0.8:
            conf_buckets["high_gte_0.8"] += 1
        elif c >= 0.5:
            conf_buckets["medium_0.5_0.8"] += 1
        else:
            conf_buckets["low_lt_0.5"] += 1

    risk_benefit_summary = {
        "ai_decisions_total": ai_decisions_count,
        "ai_accepted": accepted,
        "ai_overridden": overridden,
        "hitl_overrides": hitl_total,
        "confidence_distribution": [
            {"name": "High (>=0.8)", "value": conf_buckets["high_gte_0.8"]},
            {"name": "Medium (0.5-0.8)", "value": conf_buckets["medium_0.5_0.8"]},
            {"name": "Low (<0.5)", "value": conf_buckets["low_lt_0.5"]},
        ],
    }

    # ── AI oversight summary ────────────────────────────────────────────
    er_rows = cur.execute(
        "SELECT agree_with_ai, role, expert FROM expert_reviews"
    ).fetchall()
    agree_count = 0
    disagree_count = 0
    override_reasons = []
    for row in er_rows:
        agreement = (row["agree_with_ai"] or "").lower()
        if agreement in ("yes", "agree", "true", "1"):
            agree_count += 1
        else:
            disagree_count += 1
            if row["expert"]:
                override_reasons.append({
                    "expert": row["expert"],
                    "role": row["role"],
                })

    agreement_rate = round(
        agree_count / max(agree_count + disagree_count, 1) * 100, 1
    )

    ai_oversight_summary = {
        "expert_reviews_total": expert_reviews_count,
        "expert_agrees_with_ai": agree_count,
        "expert_disagrees_with_ai": disagree_count,
        "agreement_rate_pct": agreement_rate,
        "override_instances": override_reasons,
    }

    # ── Data action timeline (last 30 days) ─────────────────────────────
    # Group transaction_log by date
    txn_dates = cur.execute(
        "SELECT DATE(ts_utc) as d, COUNT(*) as cnt FROM transaction_log "
        "WHERE ts_utc IS NOT NULL GROUP BY DATE(ts_utc) ORDER BY d"
    ).fetchall()
    data_action_timeline = [
        {"date": row["d"], "count": row["cnt"]} for row in txn_dates
    ]

    conn.close()

    return {
        "kpis": kpis,
        "protocol_compliance": protocol_compliance,
        "consent_tracking": consent_tracking,
        "risk_benefit_summary": risk_benefit_summary,
        "ai_oversight_summary": ai_oversight_summary,
        "data_action_timeline": data_action_timeline,
    }


# ════════════════════════════════════════════════════════════════════════
# 2. breakdown()
# ════════════════════════════════════════════════════════════════════════

def breakdown():
    """Per-patient ethics profiles and audit breakdowns."""
    conn = _connect()
    if not conn:
        return {"error": "clinical.db not found"}

    cur = conn.cursor()

    # ── Load all patients ───────────────────────────────────────────────
    patients = [dict(r) for r in cur.execute(
        "SELECT patient_id, name, age, gender, disease FROM patients"
    ).fetchall()]

    # ── Pre-load per-patient data ───────────────────────────────────────
    # Assessments
    assess_rows = cur.execute(
        "SELECT patient_id, instrument FROM assessments"
    ).fetchall()
    assess_by_patient = defaultdict(list)
    for r in assess_rows:
        assess_by_patient[r["patient_id"]].append(r["instrument"])

    # Transaction log counts
    txn_counts = dict(cur.execute(
        "SELECT patient_id, COUNT(*) FROM transaction_log "
        "WHERE patient_id IS NOT NULL GROUP BY patient_id"
    ).fetchall())

    # Uploads counts
    upload_counts = dict(cur.execute(
        "SELECT patient_id, COUNT(*) FROM uploads GROUP BY patient_id"
    ).fetchall())

    # Expert reviews
    er_rows = cur.execute(
        "SELECT patient_id, role, expert, finding, agree_with_ai, note, created_at "
        "FROM expert_reviews"
    ).fetchall()
    er_by_patient = defaultdict(list)
    for r in er_rows:
        er_by_patient[r["patient_id"]].append({
            "role": r["role"],
            "expert": r["expert"],
            "finding": r["finding"],
            "agree_with_ai": r["agree_with_ai"],
            "note": r["note"],
            "created_at": r["created_at"],
        })

    # Clinical decisions
    cd_rows = cur.execute(
        "SELECT patient_id, ai_prediction, ai_confidence, neurologist_agreement, "
        "final_decision, reviewer, note, created_at FROM clinical_decisions"
    ).fetchall()
    cd_by_patient = defaultdict(list)
    for r in cd_rows:
        cd_by_patient[r["patient_id"]].append({
            "ai_prediction": r["ai_prediction"],
            "ai_confidence": r["ai_confidence"],
            "neurologist_agreement": r["neurologist_agreement"],
            "final_decision": r["final_decision"],
            "reviewer": r["reviewer"],
            "note": r["note"],
            "created_at": r["created_at"],
        })

    # HITL reviews
    hitl_rows = cur.execute(
        "SELECT patient_id, fields_json, created_at FROM hitl_reviews"
    ).fetchall()
    hitl_by_patient = defaultdict(list)
    for r in hitl_rows:
        fields = {}
        try:
            fields = json.loads(r["fields_json"]) if r["fields_json"] else {}
        except (json.JSONDecodeError, TypeError):
            pass
        hitl_by_patient[r["patient_id"]].append({
            "fields": fields,
            "created_at": r["created_at"],
        })

    # ── Per-patient ethics profile ──────────────────────────────────────
    per_patient_ethics_profile = []
    for pat in patients:
        pid = pat["patient_id"]
        assessments_list = assess_by_patient.get(pid, [])
        has_consent = len(assessments_list) > 0
        n_data_actions = txn_counts.get(pid, 0)
        n_uploads = upload_counts.get(pid, 0)
        expert_reviews = er_by_patient.get(pid, [])
        ai_decisions = cd_by_patient.get(pid, [])
        hitl_reviews_list = hitl_by_patient.get(pid, [])

        # Risk level determination
        risk_level = "low"
        risk_reasons = []

        # Check AI confidence
        for dec in ai_decisions:
            conf = dec.get("ai_confidence")
            if conf is not None and conf < 0.7:
                risk_level = "medium"
                risk_reasons.append(f"Low AI confidence ({conf})")

        # Check expert disagreement
        for rev in expert_reviews:
            agreement = (rev.get("agree_with_ai") or "").lower()
            if agreement not in ("yes", "agree", "true", "1"):
                if risk_level == "low":
                    risk_level = "medium"
                risk_reasons.append("Expert disagreed with AI")

        # Check HITL overrides
        if len(hitl_reviews_list) > 0:
            risk_level = "high"
            risk_reasons.append(f"Human override applied ({len(hitl_reviews_list)} review(s))")

        # Check low confidence + override combination
        for dec in ai_decisions:
            conf = dec.get("ai_confidence")
            agreement = (dec.get("neurologist_agreement") or "").lower()
            if conf is not None and conf < 0.7 and agreement not in ("yes", "agree", "true", "1"):
                risk_level = "high"
                risk_reasons.append("Low confidence AND neurologist disagreement")

        per_patient_ethics_profile.append({
            "patient_id": pid,
            "name": pat["name"],
            "age": pat["age"],
            "gender": pat["gender"],
            "disease": pat["disease"],
            "has_consent": has_consent,
            "n_assessments": len(assessments_list),
            "assessment_types": list(set(assessments_list)),
            "n_data_actions": n_data_actions,
            "n_uploads": n_uploads,
            "expert_reviews": expert_reviews,
            "ai_decisions": ai_decisions,
            "hitl_reviews": hitl_reviews_list,
            "risk_level": risk_level,
            "risk_reasons": risk_reasons,
        })

    # ── Component audit ─────────────────────────────────────────────────
    comp_rows = cur.execute(
        "SELECT component, action, COUNT(*) as cnt "
        "FROM transaction_log WHERE component IS NOT NULL "
        "GROUP BY component, action ORDER BY cnt DESC"
    ).fetchall()

    component_audit = defaultdict(list)
    for r in comp_rows:
        component_audit[r["component"]].append({
            "action": r["action"],
            "count": r["cnt"],
        })
    component_audit = dict(component_audit)

    # ── Actor audit ─────────────────────────────────────────────────────
    actor_rows = cur.execute(
        "SELECT actor, COUNT(*) as cnt "
        "FROM transaction_log WHERE actor IS NOT NULL "
        "GROUP BY actor ORDER BY cnt DESC"
    ).fetchall()
    actor_audit = [{"actor": r["actor"], "action_count": r["cnt"]} for r in actor_rows]

    # ── Data access log (last 50 entries) ───────────────────────────────
    recent_rows = cur.execute(
        "SELECT id, patient_id, component, action, actor, ref_id, detail, ts_utc, ts_local "
        "FROM transaction_log ORDER BY id DESC LIMIT 50"
    ).fetchall()
    data_access_log = [
        {
            "id": r["id"],
            "patient_id": r["patient_id"],
            "component": r["component"],
            "action": r["action"],
            "actor": r["actor"],
            "ref_id": r["ref_id"],
            "detail": r["detail"],
            "ts_utc": r["ts_utc"],
            "ts_local": r["ts_local"],
        }
        for r in recent_rows
    ]

    # ── Vulnerable population flags ─────────────────────────────────────
    vulnerable_population_flags = []
    for pat in patients:
        age = pat.get("age")
        if age is not None and (age < 18 or age > 65):
            flag_type = "minor" if age < 18 else "elderly"
            vulnerable_population_flags.append({
                "patient_id": pat["patient_id"],
                "name": pat["name"],
                "age": age,
                "flag": flag_type,
                "irb_note": (
                    "Minor (< 18) — requires parental/guardian consent, assent documentation, "
                    "and additional IRB protections per 45 CFR 46 Subpart D."
                    if flag_type == "minor" else
                    "Elderly (> 65) — assess decisional capacity, consider LAR consent, "
                    "monitor for cognitive decline affecting voluntary participation."
                ),
            })

    conn.close()

    return {
        "per_patient_ethics_profile": per_patient_ethics_profile,
        "component_audit": component_audit,
        "actor_audit": actor_audit,
        "data_access_log": data_access_log,
        "vulnerable_population_flags": vulnerable_population_flags,
    }


# ════════════════════════════════════════════════════════════════════════
# 3. definitions()
# ════════════════════════════════════════════════════════════════════════

def definitions():
    """IRB/ethics concepts, quality metrics, compliance refs, remediation."""
    return {
        "terms": [
            {
                "term": "IRB (Institutional Review Board)",
                "definition": (
                    "An independent committee established to review and approve research "
                    "involving human subjects. The IRB ensures that research protocols protect "
                    "participants' rights, safety, and welfare, and comply with federal regulations "
                    "(45 CFR 46). In AI-assisted clinical research, the IRB also evaluates "
                    "algorithmic decision-making impact on patient care."
                ),
            },
            {
                "term": "Informed Consent",
                "definition": (
                    "The process by which a research participant voluntarily agrees to take part "
                    "in a study after being informed of its purpose, procedures, risks, benefits, "
                    "and alternatives. For AI-driven EEG analysis, informed consent must disclose "
                    "that algorithmic predictions will be used alongside clinical judgment."
                ),
            },
            {
                "term": "Risk-Benefit Analysis",
                "definition": (
                    "A systematic evaluation of the potential risks (physical, psychological, social, "
                    "economic) against anticipated benefits of research participation. In this system, "
                    "risk-benefit is assessed by comparing AI prediction confidence, expert agreement "
                    "rates, and human override frequency to determine if AI-assisted diagnosis "
                    "provides net benefit to patients."
                ),
            },
            {
                "term": "Human-in-the-Loop (HITL) Review",
                "definition": (
                    "A safeguard mechanism where a qualified human expert reviews, validates, or "
                    "overrides AI-generated predictions before they affect clinical decisions. "
                    "HITL reviews are a core IRB requirement for AI-assisted clinical research "
                    "to prevent algorithmic harm."
                ),
            },
            {
                "term": "Data Audit Trail",
                "definition": (
                    "A chronological record of all data access, modifications, and clinical actions "
                    "performed within the system. The transaction log provides a complete audit trail "
                    "for IRB compliance, enabling reconstruction of who accessed what data, when, "
                    "and what actions were taken."
                ),
            },
            {
                "term": "Vulnerable Populations",
                "definition": (
                    "Groups requiring additional IRB protections due to diminished autonomy or "
                    "increased susceptibility to coercion. In epilepsy research, this includes "
                    "minors (< 18, 45 CFR 46 Subpart D), elderly patients (> 65, cognitive "
                    "decline risk), pregnant women, and prisoners. Additional consent procedures "
                    "and monitoring are required."
                ),
            },
            {
                "term": "Protocol Compliance",
                "definition": (
                    "Adherence to the approved research protocol, including required workflow steps "
                    "(data upload, assessment, expert review, clinical decision). Protocol deviations "
                    "must be reported to the IRB. This dashboard tracks completion rates for each "
                    "mandated step across all enrolled patients."
                ),
            },
            {
                "term": "AI Decision Oversight",
                "definition": (
                    "The systematic monitoring and evaluation of AI-generated clinical predictions "
                    "to ensure safety, accuracy, and fairness. Includes tracking prediction confidence "
                    "levels, expert agreement rates, override frequency, and identifying patterns "
                    "of algorithmic bias or error that may warrant protocol modification."
                ),
            },
        ],
        "quality_metrics": [
            {
                "metric": "Consent Rate",
                "description": (
                    "Percentage of enrolled patients with documented informed consent "
                    "(proxied by assessment participation). Target: 100%."
                ),
                "target": "100%",
            },
            {
                "metric": "Audit Completeness",
                "description": (
                    "Percentage of enrolled patients with at least one transaction log entry, "
                    "confirming their data interactions are recorded. Target: 100%."
                ),
                "target": "100%",
            },
            {
                "metric": "Override Rate",
                "description": (
                    "Percentage of AI clinical decisions that were overridden by human reviewers. "
                    "A high override rate may indicate algorithmic issues requiring protocol amendment."
                ),
                "target": "<15%",
            },
            {
                "metric": "Expert Agreement Rate",
                "description": (
                    "Percentage of expert reviews that agree with the AI prediction. "
                    "Low agreement rates trigger IRB review of the AI model's clinical validity."
                ),
                "target": ">85%",
            },
        ],
        "compliance_references": [
            {
                "standard": "45 CFR 46 (Common Rule)",
                "description": (
                    "U.S. federal regulations for the protection of human subjects in research. "
                    "Establishes IRB requirements, informed consent standards, and additional "
                    "protections for vulnerable populations (Subparts B, C, D)."
                ),
            },
            {
                "standard": "Declaration of Helsinki",
                "description": (
                    "World Medical Association ethical principles for medical research involving "
                    "human subjects. Requires that research benefit outweigh risks, informed "
                    "consent be obtained, and an independent ethics committee review all protocols."
                ),
            },
            {
                "standard": "ICH-GCP E6(R2)",
                "description": (
                    "International Council for Harmonisation guideline on Good Clinical Practice. "
                    "Provides a unified standard for designing, conducting, recording, and reporting "
                    "clinical trials. Requires audit trails, data integrity, and participant safety monitoring."
                ),
            },
            {
                "standard": "HIPAA",
                "description": (
                    "Health Insurance Portability and Accountability Act. Governs the privacy and "
                    "security of protected health information (PHI) in research settings. Requires "
                    "de-identification or authorization for use of patient data in AI model training."
                ),
            },
        ],
        "remediation_strategies": [
            {
                "strategy": "Consent Gap Remediation",
                "description": (
                    "Identify patients without documented consent engagement. Implement re-consent "
                    "protocol with study coordinator follow-up within 7 days. For vulnerable "
                    "populations, engage legally authorized representatives."
                ),
            },
            {
                "strategy": "Audit Trail Gap Closure",
                "description": (
                    "Patients without transaction log entries may indicate data access outside "
                    "the monitored system. Investigate data handling pathways, enforce all access "
                    "through the audited platform, and implement automated logging alerts."
                ),
            },
            {
                "strategy": "AI Override Protocol Review",
                "description": (
                    "When the override rate exceeds 15%, convene an IRB subcommittee to review "
                    "the AI model's clinical validity. Evaluate retraining needs, update risk "
                    "classification thresholds, and document corrective actions."
                ),
            },
            {
                "strategy": "Vulnerable Population Safeguards",
                "description": (
                    "For minors: obtain parental consent + child assent, assign pediatric advocate. "
                    "For elderly: assess decisional capacity (MoCA/MMSE), consider LAR consent, "
                    "implement additional monitoring for cognitive decline during study participation."
                ),
            },
        ],
    }
