"""
AI / ML Advisor Dashboard — EEG Epilepsy Platform
Surfaces model performance benchmarks, validation study status, clinical decision
AI vs clinician agreement, multimodal fusion stats, and responsible-AI notes.

Reads from clinical.db: analyses, clinical_decisions, model_comparison,
validation_studies, multimodal_fusion, hitl_reviews, finops_costs.
"""

import sqlite3, json, os, statistics
from datetime import datetime

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clinical.db")


def _conn():
    return sqlite3.connect(DB)


# ─── helpers ──────────────────────────────────────────────────────────────────

def _safe_mean(lst):
    lst = [x for x in lst if x is not None]
    return round(statistics.mean(lst), 4) if lst else None


def _pct(n, d):
    return round(n / d * 100, 1) if d else 0


# ─── overview ─────────────────────────────────────────────────────────────────

def overview():
    conn = _conn()
    c = conn.cursor()

    # ── analyses / predictions ──
    total_analyses = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    diseases = c.execute(
        "SELECT disease, COUNT(*) as n FROM analyses GROUP BY disease ORDER BY n DESC"
    ).fetchall()
    avg_confidence = c.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0]

    # signal quality distribution
    sq_rows = c.execute(
        "SELECT signal_quality, COUNT(*) FROM analyses GROUP BY signal_quality"
    ).fetchall()

    # ── clinical decisions (AI vs clinician) ──
    cd_total = c.execute("SELECT COUNT(*) FROM clinical_decisions").fetchone()[0]
    agreed = c.execute(
        "SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement='Full'"
    ).fetchone()[0]
    partial = c.execute(
        "SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement='Partial'"
    ).fetchone()[0]
    override = c.execute(
        "SELECT COUNT(*) FROM clinical_decisions WHERE neurologist_agreement='Disagreed'"
    ).fetchone()[0]

    # ── model comparison ──
    model_count = c.execute("SELECT COUNT(*) FROM model_comparison").fetchone()[0]
    best_auc = c.execute("SELECT model_name, MAX(auc_roc) FROM model_comparison WHERE auc_roc IS NOT NULL").fetchone()
    avg_auc = c.execute("SELECT AVG(auc_roc) FROM model_comparison WHERE auc_roc IS NOT NULL").fetchone()[0]
    tasks = c.execute(
        "SELECT task, COUNT(*) FROM model_comparison GROUP BY task ORDER BY COUNT(*) DESC"
    ).fetchall()

    # ── validation studies ──
    val_total = c.execute("SELECT COUNT(*) FROM validation_studies").fetchone()[0]
    val_passed = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status LIKE '%Passed%' OR status='Completed'"
    ).fetchone()[0]
    val_failed = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status LIKE '%Failed%'"
    ).fetchone()[0]
    val_ongoing = c.execute(
        "SELECT COUNT(*) FROM validation_studies WHERE status LIKE '%In Progress%' OR status='Active'"
    ).fetchone()[0]

    # ── multimodal fusion ──
    mf_total = c.execute("SELECT COUNT(*) FROM multimodal_fusion").fetchone()[0]
    mf_patients = c.execute("SELECT COUNT(DISTINCT patient_id) FROM multimodal_fusion").fetchone()[0]
    mf_avg_conf = c.execute(
        "SELECT AVG(confidence) FROM multimodal_fusion WHERE confidence IS NOT NULL"
    ).fetchone()[0]
    mf_avg_modalities = c.execute(
        "SELECT AVG(modalities_count) FROM multimodal_fusion WHERE modalities_count IS NOT NULL"
    ).fetchone()[0]

    # ── hitl reviews ──
    hitl_total = c.execute("SELECT COUNT(*) FROM hitl_reviews").fetchone()[0]

    conn.close()
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "kpis": {
            "total_ai_analyses": total_analyses,
            "avg_ai_confidence": round(avg_confidence, 3) if avg_confidence else None,
            "clinical_decisions_reviewed": cd_total,
            "neurologist_agreement_rate_pct": _pct(agreed, cd_total),
            "override_rate_pct": _pct(override, cd_total),
            "partial_agreement_rate_pct": _pct(partial, cd_total),
            "model_experiments": model_count,
            "best_model_auc": round(best_auc[1], 3) if best_auc and best_auc[1] else None,
            "best_model_name": best_auc[0] if best_auc else None,
            "avg_model_auc": round(avg_auc, 3) if avg_auc else None,
            "validation_studies": val_total,
            "validation_passed": val_passed,
            "validation_failed": val_failed,
            "validation_ongoing": val_ongoing,
            "multimodal_fusion_sessions": mf_total,
            "multimodal_patients": mf_patients,
            "avg_fusion_confidence": round(mf_avg_conf, 3) if mf_avg_conf else None,
            "avg_modalities_fused": round(mf_avg_modalities, 1) if mf_avg_modalities else None,
            "hitl_reviews": hitl_total,
        },
        "disease_distribution": [
            {"disease": d, "count": n} for d, n in diseases
        ],
        "signal_quality_distribution": [
            {"quality": q, "count": n} for q, n in sq_rows
        ],
        "task_distribution": [
            {"task": t, "count": n} for t, n in tasks
        ],
        "alert": (
            "⚠ Override rate {:.1f}% exceeds 20% vigilance threshold — "
            "human sign-off mandatory per ISO 14971 / EU MDR Art.83".format(_pct(override, cd_total))
        ) if cd_total and _pct(override, cd_total) > 20 else None,
    }


# ─── breakdown ────────────────────────────────────────────────────────────────

def breakdown():
    conn = _conn()
    c = conn.cursor()

    # ── model comparison table ──
    models = c.execute("""
        SELECT model_name, model_type, task, accuracy, precision_score, recall,
               f1_score, auc_roc, training_time_sec, inference_time_ms, status, version
        FROM model_comparison
        ORDER BY auc_roc DESC NULLS LAST
        LIMIT 30
    """).fetchall()
    model_cols = ["model_name", "model_type", "task", "accuracy", "precision",
                  "recall", "f1", "auc_roc", "train_sec", "infer_ms", "status", "version"]

    # ── performance by model type ──
    by_type = c.execute("""
        SELECT model_type,
               COUNT(*) as n,
               ROUND(AVG(auc_roc),3) as avg_auc,
               ROUND(MAX(auc_roc),3) as best_auc,
               ROUND(AVG(f1_score),3) as avg_f1,
               ROUND(AVG(inference_time_ms),1) as avg_infer_ms
        FROM model_comparison
        WHERE auc_roc IS NOT NULL
        GROUP BY model_type
        ORDER BY avg_auc DESC
    """).fetchall()

    # ── performance by task ──
    by_task = c.execute("""
        SELECT task,
               COUNT(*) as n,
               ROUND(AVG(auc_roc),3) as avg_auc,
               ROUND(MAX(auc_roc),3) as best_auc
        FROM model_comparison
        WHERE auc_roc IS NOT NULL
        GROUP BY task
        ORDER BY avg_auc DESC
    """).fetchall()

    # ── validation studies ──
    val_rows = c.execute("""
        SELECT study_id, study_type, title, status, sample_size,
               sensitivity, specificity, auc_roc,
               principal_investigator, site, start_date, end_date, findings
        FROM validation_studies
        ORDER BY created_at DESC
        LIMIT 20
    """).fetchall()
    val_cols = ["study_id", "study_type", "title", "status", "sample_size",
                "sensitivity", "specificity", "auc_roc", "pi", "site",
                "start_date", "end_date", "findings"]

    # ── multimodal fusion breakdown ──
    fusion_by_method = c.execute("""
        SELECT fusion_method,
               COUNT(*) as n,
               ROUND(AVG(confidence),3) as avg_conf,
               ROUND(AVG(concordance_score),3) as avg_concordance,
               ROUND(AVG(modalities_count),1) as avg_modalities,
               ROUND(AVG(processing_time_sec),1) as avg_sec
        FROM multimodal_fusion
        GROUP BY fusion_method
        ORDER BY avg_conf DESC
    """).fetchall()

    fusion_by_subtype = c.execute("""
        SELECT predicted_subtype, COUNT(*) as n,
               ROUND(AVG(confidence),3) as avg_conf
        FROM multimodal_fusion
        GROUP BY predicted_subtype
        ORDER BY n DESC
    """).fetchall()

    # ── AI decision confidence buckets ──
    conf_buckets = []
    for lo, hi, label in [(0, 0.5, "<0.5"), (0.5, 0.6, "0.5-0.6"),
                          (0.6, 0.7, "0.6-0.7"), (0.7, 0.8, "0.7-0.8"),
                          (0.8, 0.9, "0.8-0.9"), (0.9, 1.01, "≥0.9")]:
        n = c.execute(
            "SELECT COUNT(*) FROM clinical_decisions WHERE ai_confidence >= ? AND ai_confidence < ?",
            (lo, hi)
        ).fetchone()[0]
        conf_buckets.append({"range": label, "count": n})

    # ── agreement by prediction class ──
    by_class = c.execute("""
        SELECT ai_prediction,
               COUNT(*) as total,
               SUM(CASE WHEN neurologist_agreement='Full' THEN 1 ELSE 0 END) as agreed,
               SUM(CASE WHEN neurologist_agreement='Disagreed' THEN 1 ELSE 0 END) as overridden,
               ROUND(AVG(ai_confidence),3) as avg_conf
        FROM clinical_decisions
        GROUP BY ai_prediction
        ORDER BY total DESC
    """).fetchall()

    conn.close()
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "model_leaderboard": [
            dict(zip(model_cols, row)) for row in models
        ],
        "performance_by_type": [
            {"model_type": r[0], "n": r[1], "avg_auc": r[2],
             "best_auc": r[3], "avg_f1": r[4], "avg_infer_ms": r[5]}
            for r in by_type
        ],
        "performance_by_task": [
            {"task": r[0], "n": r[1], "avg_auc": r[2], "best_auc": r[3]}
            for r in by_task
        ],
        "validation_studies": [
            dict(zip(val_cols, row)) for row in val_rows
        ],
        "fusion_by_method": [
            {"method": r[0], "n": r[1], "avg_conf": r[2],
             "avg_concordance": r[3], "avg_modalities": r[4], "avg_sec": r[5]}
            for r in fusion_by_method
        ],
        "fusion_by_subtype": [
            {"subtype": r[0], "n": r[1], "avg_conf": r[2]}
            for r in fusion_by_subtype
        ],
        "confidence_distribution": conf_buckets,
        "agreement_by_class": [
            {"prediction": r[0], "total": r[1], "agreed": r[2],
             "overridden": r[3], "avg_conf": r[4]}
            for r in by_class
        ],
    }


# ─── definitions ──────────────────────────────────────────────────────────────

def definitions():
    return {
        "generated_at": datetime.now().isoformat(timespec="minutes"),
        "role": {
            "title": "AI / ML Advisor",
            "scope": (
                "The AI/ML Advisor monitors model lifecycle: training, benchmarking, "
                "validation, clinical deployment, drift, and decommission. Advises on "
                "architecture selection, feature engineering, bias mitigation, and "
                "regulatory alignment (EU MDR, FDA SaMD, IEC 62304)."
            ),
            "responsibilities": [
                "Model selection and hyperparameter optimisation",
                "Train/test split strategy (GroupKFold, patient-level hold-out)",
                "Calibration and confidence thresholds",
                "Concept drift monitoring and retrain triggers",
                "Explainability (SHAP, LIME, GradCAM) for clinical sign-off",
                "Responsible AI: fairness, bias audits, class-imbalance remediation",
                "Regulatory evidence packages (IEC 62304 / ISO 13485 / FDA 510(k))",
            ],
        },
        "concepts": [
            {
                "term": "AUC-ROC",
                "definition": (
                    "Area Under the Receiver Operating Characteristic Curve. "
                    "Measures discrimination ability (1.0 = perfect, 0.5 = random). "
                    "Preferred over accuracy for class-imbalanced EEG data."
                ),
            },
            {
                "term": "GroupKFold",
                "definition": (
                    "Cross-validation strategy where all recordings from one patient "
                    "stay in the same fold. Prevents data leakage and gives realistic "
                    "cross-patient generalisation estimates."
                ),
            },
            {
                "term": "Confidence Calibration",
                "definition": (
                    "A model is well-calibrated when predicted probabilities match "
                    "observed event frequencies. Use Platt scaling or temperature "
                    "scaling post-training. Critical for clinical threshold setting."
                ),
            },
            {
                "term": "Concept Drift",
                "definition": (
                    "Statistical shift between training and deployment data distributions. "
                    "Detected by monitoring feature means/covariance or prediction "
                    "confidence over time. Triggers retraining when PSI > 0.2."
                ),
            },
            {
                "term": "HITL (Human-in-the-Loop)",
                "definition": (
                    "Mechanism requiring clinician review before AI predictions are "
                    "acted upon. Mandated when AI confidence < threshold or for "
                    "high-stakes decisions. Logged for audit trail."
                ),
            },
            {
                "term": "Override Rate",
                "definition": (
                    "Fraction of AI predictions overridden by clinicians. A rate "
                    ">20% triggers post-market surveillance vigilance review "
                    "(ISO 14971 / EU MDR Art.83 / FDA 21 CFR 803)."
                ),
            },
            {
                "term": "Multimodal Fusion",
                "definition": (
                    "Integration of EEG, video, MRI, clinical notes, and medication "
                    "data into a single prediction. Fusion methods: early (feature "
                    "concatenation), late (decision combination), attention (learned "
                    "modality weighting). Improves subtype discrimination."
                ),
            },
            {
                "term": "Validation Study",
                "definition": (
                    "Prospective or retrospective clinical study confirming AI "
                    "performance on an independent dataset. Required for SaMD "
                    "regulatory submissions (sensitivity ≥ 80%, specificity ≥ 80%, "
                    "AUC ≥ 0.85 for epilepsy classification)."
                ),
            },
            {
                "term": "SaMD (Software as a Medical Device)",
                "definition": (
                    "Software intended for medical purposes running on general-purpose "
                    "hardware. Regulated by: EU MDR (Class IIa/IIb), FDA (510(k)/De Novo), "
                    "IEC 62304 (software lifecycle), ISO 13485 (QMS), ISO 14971 (risk)."
                ),
            },
            {
                "term": "Inference Latency",
                "definition": (
                    "Wall-clock time from EEG feature submission to AI prediction "
                    "return. Target < 500 ms for real-time clinical workflow "
                    "integration. Benchmarked per model type on CPU (no GPU assumed)."
                ),
            },
        ],
        "regulatory_context": [
            {"standard": "EU MDR 2017/745 Art.83", "note": "Post-market surveillance — mandatory when override rate > 20%"},
            {"standard": "FDA 21 CFR Part 803", "note": "Medical device adverse event reporting"},
            {"standard": "IEC 62304:2006+AMD1:2015", "note": "Medical device software lifecycle"},
            {"standard": "ISO 14971:2019", "note": "Risk management for medical devices"},
            {"standard": "ISO 13485:2016", "note": "Quality management system for medical devices"},
            {"standard": "IMDRF SaMD N41", "note": "Principles for software as a medical device"},
            {"standard": "ICH E6(R2) GCP", "note": "Good Clinical Practice — validation study design"},
        ],
        "references": [
            "Obermeyer Z & Emanuel EJ. Predicting the Future — Big Data, ML, and Clinical Medicine. NEJM 2016.",
            "Rajpurkar P et al. AI in Health and Medicine. Nature Medicine 2022.",
            "Esteva A et al. Deep Learning in Medical Imaging. Nature 2021.",
            "FDA AI/ML Action Plan for SaMD, January 2021.",
            "EU MDR 2017/745 — Annex XIV Clinical Evaluation + Art.83 PMS.",
        ],
        "ai_governance_notes": [
            "Class imbalance: use SMOTE / class-weighted loss; epilepsy data is inherently imbalanced.",
            "Fairness audit: check performance across age groups, sex, ethnicity — GroupKFold by demographic subgroup.",
            "Explainability: SHAP channel attribution required for clinical sign-off (HITL).",
            "Leakage prevention: patient-level split only; no shared recordings across train/test.",
            "Model registry: all experiments logged in model_comparison table with hyperparams + metrics.",
        ],
        "performance_thresholds": [
            {"metric": "Sensitivity", "target": "≥ 80%", "rationale": "Minimise missed seizures (clinical safety)"},
            {"metric": "Specificity", "target": "≥ 80%", "rationale": "Minimise false alarms (clinical burden)"},
            {"metric": "AUC-ROC", "target": "≥ 0.85", "rationale": "FDA SaMD / CE Mark validation floor"},
            {"metric": "Override Rate", "target": "< 20%", "rationale": "EU MDR Art.83 vigilance trigger"},
            {"metric": "Inference Latency", "target": "< 500 ms", "rationale": "Real-time clinical workflow"},
            {"metric": "HITL Coverage", "target": "100% high-risk", "rationale": "ICH GCP / ISO 14971 risk class"},
        ],
    }
