"""Release Management Dashboard — real dataset/model version, approval, and
change-log analytics from clinical.db + on-disk model artefacts.

Sources:
- transaction_log (562)  — build/submit/create/sign-off/scheduled_train actions
- analyses (21)          — model inference runs per disease
- expert_reviews (3)     — clinician sign-offs on AI predictions
- hitl_reviews (2)       — human-in-the-loop override decisions
- clinical_decisions (1) — final clinical sign-off
- uploads (21)           — dataset ingestion records
- models/*.joblib        — on-disk model artefacts (size, last-modified)
"""

import glob
import os
import sqlite3
from collections import Counter
from datetime import datetime, timezone

DB = os.path.join(os.path.dirname(__file__), '..', 'data', 'clinical.db')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')


def _conn():
    return sqlite3.connect(DB)


def _safe(cur, sql):
    try:
        cur.execute(sql)
        r = cur.fetchone()
        return r[0] if r else 0
    except Exception:
        return 0


def _safe_rows(cur, sql):
    try:
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return []


# ── Model artefact inventory ────────────────────────────────────

def _model_inventory():
    """Scan models/*.joblib for on-disk model artefacts."""
    artefacts = []
    for path in sorted(glob.glob(os.path.join(MODELS_DIR, '*.joblib'))):
        name = os.path.basename(path)
        stat = os.stat(path)
        disease = name.replace('_model.joblib', '')
        artefacts.append({
            "file": name,
            "disease": disease,
            "size_bytes": stat.st_size,
            "size_kb": round(stat.st_size / 1024, 1),
            "last_modified": datetime.fromtimestamp(
                stat.st_mtime, tz=timezone.utc
            ).isoformat(),
        })
    return artefacts


# ── Overview ─────────────────────────────────────────────────────

def release_overview():
    """Release KPIs: model count, dataset records, approval rate,
    training runs, sign-offs, daily change trend."""
    if not os.path.exists(DB):
        return {"available": False, "note": "clinical.db not found"}

    conn = _conn()
    cur = conn.cursor()
    now = datetime.now(timezone.utc)
    result = {"available": True, "generated_at": now.isoformat()}

    # ── Model artefacts ──
    models = _model_inventory()
    total_models = len(models)
    total_model_size_kb = sum(m["size_kb"] for m in models)

    # ── Dataset stats ──
    total_uploads = _safe(cur, "SELECT count(*) FROM uploads")
    distinct_patients = _safe(
        cur, "SELECT count(DISTINCT patient_id) FROM uploads"
    )
    distinct_diseases_uploaded = _safe(
        cur, "SELECT count(DISTINCT disease) FROM uploads"
    )

    # ── Analysis (inference) runs ──
    total_analyses = _safe(cur, "SELECT count(*) FROM analyses")
    avg_confidence = _safe(
        cur, "SELECT round(avg(confidence), 3) FROM analyses"
    )

    # ── Approval / sign-off stats ──
    expert_reviews = _safe(cur, "SELECT count(*) FROM expert_reviews")
    hitl_reviews = _safe(cur, "SELECT count(*) FROM hitl_reviews")
    clinical_decisions = _safe(cur, "SELECT count(*) FROM clinical_decisions")
    total_approvals = expert_reviews + hitl_reviews + clinical_decisions

    # Agreement rate
    agree_count = _safe(
        cur,
        "SELECT count(*) FROM expert_reviews WHERE agree_with_ai='agree'"
    )
    agreement_rate = round(agree_count / expert_reviews, 3) if expert_reviews else 0

    # Override rate from hitl
    override_count = len([
        r for r in _safe_rows(
            cur, "SELECT fields_json FROM hitl_reviews"
        )
        if 'override' in (r[0] or '')
    ])
    override_rate = round(
        override_count / hitl_reviews, 3
    ) if hitl_reviews else 0

    # ── Training runs ──
    training_runs = _safe(
        cur,
        "SELECT count(*) FROM transaction_log "
        "WHERE component='training' AND action='scheduled_train'"
    )

    # ── Sign-off actions ──
    signoff_count = _safe(
        cur,
        "SELECT count(*) FROM transaction_log WHERE action='sign-off'"
    )

    # ── Change log actions (build/create/update/submit) ──
    change_actions = _safe_rows(
        cur,
        "SELECT action, count(*) FROM transaction_log "
        "WHERE action IN ('build','create','update','submit','scheduled_train','sign-off') "
        "GROUP BY action ORDER BY count(*) DESC"
    )
    change_action_dist = [
        {"action": r[0], "count": r[1]} for r in change_actions
    ]
    total_changes = sum(r[1] for r in change_actions)

    # ── Daily change trend ──
    daily_rows = _safe_rows(
        cur,
        "SELECT substr(ts_utc, 1, 10) AS day, count(*) "
        "FROM transaction_log "
        "WHERE action IN ('build','create','update','submit','scheduled_train','sign-off') "
        "GROUP BY day ORDER BY day"
    )
    daily_trend = [{"date": r[0], "changes": r[1]} for r in daily_rows]

    # ── Disease-level analysis coverage ──
    disease_analysis = _safe_rows(
        cur,
        "SELECT disease, count(*), round(avg(confidence),3) "
        "FROM analyses GROUP BY disease ORDER BY count(*) DESC"
    )
    disease_coverage = [
        {"disease": r[0], "analyses": r[1], "avg_confidence": r[2]}
        for r in disease_analysis
    ]

    # ── Actor activity on release-related actions ──
    actor_rows = _safe_rows(
        cur,
        "SELECT actor, count(*) FROM transaction_log "
        "WHERE action IN ('build','create','update','submit','scheduled_train','sign-off') "
        "GROUP BY actor ORDER BY count(*) DESC"
    )
    actor_distribution = [
        {"actor": r[0], "changes": r[1]} for r in actor_rows
    ]

    result["summary"] = {
        "total_models": total_models,
        "total_model_size_kb": total_model_size_kb,
        "total_uploads": total_uploads,
        "distinct_patients": distinct_patients,
        "distinct_diseases_uploaded": distinct_diseases_uploaded,
        "total_analyses": total_analyses,
        "avg_confidence": avg_confidence,
        "total_approvals": total_approvals,
        "expert_reviews": expert_reviews,
        "hitl_reviews": hitl_reviews,
        "clinical_decisions": clinical_decisions,
        "agreement_rate": agreement_rate,
        "override_rate": override_rate,
        "training_runs": training_runs,
        "signoff_count": signoff_count,
        "total_changes": total_changes,
    }
    result["model_inventory"] = models
    result["disease_coverage"] = disease_coverage
    result["change_action_distribution"] = change_action_dist
    result["daily_trend"] = daily_trend
    result["actor_distribution"] = actor_distribution

    conn.close()
    return result


# ── Breakdown ────────────────────────────────────────────────────

def release_breakdown():
    """Detailed release breakdown: per-patient upload history, expert review
    log, HITL override log, training schedule, component change log."""
    if not os.path.exists(DB):
        return {"available": False}

    conn = _conn()
    cur = conn.cursor()
    result = {"available": True}

    # ── Per-patient upload inventory ──
    upload_rows = _safe_rows(
        cur,
        "SELECT patient_id, file_name, disease, department, created_at "
        "FROM uploads ORDER BY created_at DESC"
    )
    upload_log = [
        {
            "patient_id": r[0], "file": r[1], "disease": r[2],
            "department": r[3], "uploaded_at": r[4]
        }
        for r in upload_rows
    ]

    # ── Expert review log ──
    review_rows = _safe_rows(
        cur,
        "SELECT patient_id, role, expert, finding, agree_with_ai, note, "
        "created_at FROM expert_reviews ORDER BY created_at DESC"
    )
    expert_log = [
        {
            "patient_id": r[0], "role": r[1], "expert": r[2],
            "finding": r[3], "agreement": r[4], "note": r[5],
            "reviewed_at": r[6]
        }
        for r in review_rows
    ]

    # ── HITL override log ──
    hitl_rows = _safe_rows(
        cur,
        "SELECT patient_id, fields_json, created_at "
        "FROM hitl_reviews ORDER BY created_at DESC"
    )
    import json as _json
    hitl_log = []
    for r in hitl_rows:
        fields = {}
        try:
            fields = _json.loads(r[1]) if r[1] else {}
        except Exception:
            pass
        hitl_log.append({
            "patient_id": r[0],
            "ai_prediction": fields.get("ai_prediction", ""),
            "decision": fields.get("decision", ""),
            "human_decision": fields.get("human_decision", ""),
            "reason_code": fields.get("reason_code", ""),
            "reviewed_at": r[2]
        })

    # ── Clinical decision log ──
    decision_rows = _safe_rows(
        cur,
        "SELECT patient_id, ai_prediction, ai_confidence, "
        "neurologist_agreement, final_decision, reviewer, note, created_at "
        "FROM clinical_decisions ORDER BY created_at DESC"
    )
    decision_log = [
        {
            "patient_id": r[0], "ai_prediction": r[1],
            "ai_confidence": r[2], "neurologist_agreement": r[3],
            "final_decision": r[4], "reviewer": r[5],
            "note": r[6], "decided_at": r[7]
        }
        for r in decision_rows
    ]

    # ── Training schedule log ──
    train_rows = _safe_rows(
        cur,
        "SELECT patient_id, actor, detail, ts_utc FROM transaction_log "
        "WHERE component='training' AND action='scheduled_train' "
        "ORDER BY ts_utc DESC"
    )
    training_log = [
        {
            "patient_id": r[0], "actor": r[1],
            "detail": r[2], "timestamp": r[3]
        }
        for r in train_rows
    ]

    # ── Component change log (build/create/update/submit/sign-off) ──
    change_rows = _safe_rows(
        cur,
        "SELECT component, action, actor, patient_id, detail, ts_utc "
        "FROM transaction_log "
        "WHERE action IN ('build','create','update','submit','scheduled_train','sign-off') "
        "ORDER BY ts_utc DESC LIMIT 50"
    )
    change_log = [
        {
            "component": r[0], "action": r[1], "actor": r[2],
            "patient_id": r[3], "detail": r[4], "timestamp": r[5]
        }
        for r in change_rows
    ]

    # ── Component-level change counts ──
    comp_change_rows = _safe_rows(
        cur,
        "SELECT component, count(*) FROM transaction_log "
        "WHERE action IN ('build','create','update','submit','scheduled_train','sign-off') "
        "GROUP BY component ORDER BY count(*) DESC"
    )
    component_changes = [
        {"component": r[0], "changes": r[1]} for r in comp_change_rows
    ]

    # ── Hourly change pattern ──
    hourly_rows = _safe_rows(
        cur,
        "SELECT CAST(substr(ts_utc, 12, 2) AS INTEGER) AS hr, count(*) "
        "FROM transaction_log "
        "WHERE action IN ('build','create','update','submit','scheduled_train','sign-off') "
        "GROUP BY hr ORDER BY hr"
    )
    hourly_pattern = [{"hour": r[0], "changes": r[1]} for r in hourly_rows]

    result["upload_log"] = upload_log
    result["expert_log"] = expert_log
    result["hitl_log"] = hitl_log
    result["decision_log"] = decision_log
    result["training_log"] = training_log
    result["change_log"] = change_log
    result["component_changes"] = component_changes
    result["hourly_pattern"] = hourly_pattern

    conn.close()
    return result


# ── Definitions ──────────────────────────────────────────────────

def release_definitions():
    """Metric definitions for the Release Management dashboard."""
    return {
        "metrics": [
            {
                "metric": "Total Models",
                "definition": "Number of .joblib model artefacts on disk (models/ directory)."
            },
            {
                "metric": "Model Size (KB)",
                "definition": "Combined size of all model artefact files in kilobytes."
            },
            {
                "metric": "Total Uploads",
                "definition": "Count of EEG file uploads ingested into the platform (uploads table)."
            },
            {
                "metric": "Distinct Patients",
                "definition": "Number of unique patient IDs with at least one uploaded EEG file."
            },
            {
                "metric": "Total Analyses",
                "definition": "Number of model inference runs recorded in the analyses table."
            },
            {
                "metric": "Avg Confidence",
                "definition": "Mean prediction confidence across all inference runs."
            },
            {
                "metric": "Expert Reviews",
                "definition": "Number of clinician expert reviews on AI predictions (expert_reviews table)."
            },
            {
                "metric": "Agreement Rate",
                "definition": "Fraction of expert reviews where the clinician agreed with the AI prediction."
            },
            {
                "metric": "HITL Reviews",
                "definition": "Human-in-the-loop override reviews (hitl_reviews table)."
            },
            {
                "metric": "Override Rate",
                "definition": "Fraction of HITL reviews that resulted in an override of the AI prediction."
            },
            {
                "metric": "Clinical Decisions",
                "definition": "Final clinical sign-off decisions recorded (clinical_decisions table)."
            },
            {
                "metric": "Training Runs",
                "definition": "Scheduled model training events from transaction_log (action=scheduled_train)."
            },
            {
                "metric": "Sign-offs",
                "definition": "Formal sign-off actions recorded in the transaction log (action=sign-off)."
            },
            {
                "metric": "Total Changes",
                "definition": "Sum of build, create, update, submit, scheduled_train, and sign-off actions in the transaction log."
            },
            {
                "metric": "Daily Trend",
                "definition": "Number of release-relevant change actions per calendar day."
            },
            {
                "metric": "Component Changes",
                "definition": "Per-component count of release-relevant actions in the transaction log."
            },
        ]
    }
