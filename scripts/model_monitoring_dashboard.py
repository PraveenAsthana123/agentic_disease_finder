"""
Model Monitoring AI Dashboard
==============================
Provides monitoring overview, detailed breakdown, and definitions for
the epilepsy EEG clinical decision support system.

Reads REAL data from:
  - jobs/reports/drift_latest.json
  - jobs/reports/health_latest.json
  - jobs/reports/consistency_latest.json
  - jobs/reports/data_quality_latest.json
  - jobs/reports/training_latest.json
  - jobs/reports/accuracy_patient_specific.json
  - jobs/reports/accuracy_all_options.json
  - data/clinical.db
"""

import json
import re
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REPORTS = BASE / "jobs" / "reports"
DB_PATH = BASE / "data" / "clinical.db"


def _load_json(filename):
    """Load a JSON report file, return None on any failure."""
    try:
        with open(REPORTS / filename, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _query_db(sql, params=()):
    """Run a read-only query against clinical.db, return rows as dicts."""
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception:
        return []


# ──────────────────────────────────────────────────────────────────────
# 1. monitoring_overview
# ──────────────────────────────────────────────────────────────────────

def monitoring_overview() -> dict:
    """
    KPI-level summary combining all monitoring signals:
    drift, health, consistency, data quality, training, model predictions.
    """
    drift = _load_json("drift_latest.json")
    health = _load_json("health_latest.json")
    consistency = _load_json("consistency_latest.json")
    dq = _load_json("data_quality_latest.json")
    training = _load_json("training_latest.json")

    out = {}

    # ── Drift ────────────────────────────────────────────────────────
    if drift:
        out["drift_verdict"] = drift.get("verdict")
        out["drift_frac_drifted"] = drift.get("frac_drifted")
        out["n_features_drifted"] = drift.get("n_high_drift")
        out["n_features_total"] = drift.get("n_features")

        # severity distribution from top_drift — array of {severity, count}
        sev_counts = {}
        top_drift = drift.get("top_drift", [])
        for feat in top_drift:
            sev = feat.get("severity", "unknown")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
        out["severity_distribution"] = [
            {"severity": k, "count": v} for k, v in sev_counts.items() if v > 0
        ]

        # top 12 drifted features
        out["top_drifted_features"] = [
            {
                "feature": f.get("feature"),
                "psi": f.get("psi"),
                "ks_stat": f.get("ks_stat"),
                "severity": f.get("severity"),
            }
            for f in top_drift[:12]
        ]
    else:
        out["drift_verdict"] = None
        out["drift_frac_drifted"] = None
        out["n_features_drifted"] = None
        out["n_features_total"] = None
        out["severity_distribution"] = []
        out["top_drifted_features"] = []

    # ── System health ────────────────────────────────────────────────
    if health:
        out["system_health"] = {
            "backend_http": str(health.get("backend_http", "N/A")),
            "api_errors": health.get("api_errors", health.get("total_errors", 0)),
            "api_total": health.get("api_total", 0),
            "db_status": health.get("db", "N/A"),
            "frontend_http": str(health.get("frontend_http", "N/A")),
        }
    else:
        out["system_health"] = {}

    # ── Consistency ──────────────────────────────────────────────────
    if consistency:
        out["consistency_verdict"] = consistency.get("verdict")
        out["consistency_mismatches"] = consistency.get("mismatches")
    else:
        out["consistency_verdict"] = None
        out["consistency_mismatches"] = None

    # ── Data quality ─────────────────────────────────────────────────
    if dq:
        dq_score = dq.get("ai_readiness_score")
        out["data_quality_score"] = dq_score
        if dq_score is not None:
            out["data_quality_grade"] = (
                "A" if dq_score >= 90 else
                "B" if dq_score >= 75 else
                "C" if dq_score >= 60 else
                "D" if dq_score >= 40 else "F"
            )
        else:
            out["data_quality_grade"] = None
        # quality_dimensions: ensure each value is {score, basis}
        raw_dims = dq.get("quality_dimensions", {})
        dims_out = {}
        for k, v in raw_dims.items():
            if isinstance(v, dict):
                dims_out[k] = {"score": v.get("score"), "basis": v.get("basis", v.get("note", ""))}
            elif isinstance(v, (int, float)):
                dims_out[k] = {"score": v, "basis": ""}
        out["quality_dimensions"] = dims_out
        out["modality_coverage"] = dq.get("modality_coverage_pct", {})
    else:
        out["data_quality_score"] = None
        out["data_quality_grade"] = None
        out["quality_dimensions"] = {}
        out["modality_coverage"] = {}

    # ── Training summary ─────────────────────────────────────────────
    if training:
        results = training.get("results", [])
        parsed = []
        for r in results:
            entry = {
                "script": r.get("script"),
                "ok": r.get("ok"),
                "seconds": r.get("seconds"),
            }
            tail = r.get("tail", "")
            # Extract mean accuracy and mean sensitivity from tail text
            acc_match = re.search(r"mean accuracy:\s*([\d.]+)", tail)
            sens_match = re.search(r"mean sensitivity:\s*([\d.]+)", tail)
            if acc_match:
                entry["mean_accuracy"] = float(acc_match.group(1))
            if sens_match:
                entry["mean_sensitivity"] = float(sens_match.group(1))
            parsed.append(entry)
        out["training_summary"] = {
            "summary": training.get("summary"),
            "run_at": training.get("run_at_local") or training.get("run_at_utc"),
            "results": parsed,
        }
    else:
        out["training_summary"] = {}

    # ── Model predictions from clinical.db ───────────────────────────
    total_rows = _query_db("SELECT COUNT(*) AS cnt FROM analyses")
    avg_conf = _query_db("SELECT AVG(confidence) AS avg_conf FROM analyses")
    disease_rows = _query_db(
        "SELECT disease, COUNT(*) AS cnt, AVG(confidence) AS avg_conf "
        "FROM analyses GROUP BY disease"
    )

    # Confidence distribution: 5 bins — array of {bin, count}
    bins = [
        ("0-20%", 0.0, 0.2),
        ("20-40%", 0.2, 0.4),
        ("40-60%", 0.4, 0.6),
        ("60-80%", 0.6, 0.8),
        ("80-100%", 0.8, 1.01),
    ]
    conf_dist = []
    for label, lo, hi in bins:
        rows = _query_db(
            "SELECT COUNT(*) AS cnt FROM analyses WHERE confidence >= ? AND confidence < ?",
            (lo, hi),
        )
        conf_dist.append({"bin": label, "count": rows[0]["cnt"] if rows else 0})

    # by_disease — array of {disease, count, avg_confidence}
    by_disease = [
        {
            "disease": r["disease"],
            "count": r["cnt"],
            "avg_confidence": round(r["avg_conf"], 4) if r["avg_conf"] is not None else None,
        }
        for r in disease_rows
    ]

    out["model_predictions"] = {
        "total_analyses": total_rows[0]["cnt"] if total_rows else 0,
        "avg_confidence": round(avg_conf[0]["avg_conf"], 4) if avg_conf and avg_conf[0]["avg_conf"] is not None else None,
        "confidence_distribution": conf_dist,
        "by_disease": by_disease,
    }

    # ── Monitoring timeline ──────────────────────────────────────────
    timeline = []
    if drift and drift.get("run_at_local"):
        timeline.append({"source": "drift", "timestamp": drift["run_at_local"], "verdict": drift.get("verdict")})
    if consistency and consistency.get("run_at"):
        timeline.append({"source": "consistency", "timestamp": consistency["run_at"], "verdict": consistency.get("verdict")})
    if dq and dq.get("run_at"):
        timeline.append({"source": "data_quality", "timestamp": dq["run_at"], "status": dq.get("ai_readiness_grade")})
    if health and health.get("now"):
        timeline.append({"source": "health", "timestamp": health["now"], "status": f"HTTP {health.get('backend_http')}"})
    if training and (training.get("run_at_local") or training.get("run_at_utc")):
        timeline.append({
            "source": "training",
            "timestamp": training.get("run_at_local") or training.get("run_at_utc"),
            "status": training.get("summary"),
        })
    timeline.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    out["monitoring_timeline"] = timeline

    return out


# ──────────────────────────────────────────────────────────────────────
# 2. monitoring_breakdown
# ──────────────────────────────────────────────────────────────────────

def monitoring_breakdown() -> dict:
    """
    Detailed drill-down data: consistency checks, all drift features,
    training runs, missing matrix, per-patient predictions, accuracy.
    """
    consistency = _load_json("consistency_latest.json")
    drift = _load_json("drift_latest.json")
    training = _load_json("training_latest.json")
    dq = _load_json("data_quality_latest.json")
    acc_ps = _load_json("accuracy_patient_specific.json")
    acc_all = _load_json("accuracy_all_options.json")

    out = {}

    # consistency checks
    out["consistency_checks"] = consistency.get("checks", []) if consistency else []

    # all drift features
    if drift:
        out["drift_features_all"] = [
            {
                "feature": f.get("feature"),
                "psi": f.get("psi"),
                "ks_stat": f.get("ks_stat"),
                "ks_p": f.get("ks_p"),
                "severity": f.get("severity"),
            }
            for f in drift.get("top_drift", [])
        ]
    else:
        out["drift_features_all"] = []

    # training runs — ensure each has {script, seconds, ok, exit_code}
    raw_runs = training.get("results", []) if training else []
    out["training_runs"] = [
        {
            "script": r.get("script", "?"),
            "seconds": r.get("seconds"),
            "ok": r.get("ok", r.get("exit_code", 0) == 0),
            "exit_code": r.get("exit_code", 0),
        }
        for r in raw_runs
        if isinstance(r, dict)
    ]

    # missing matrix
    out["missing_matrix"] = dq.get("missing_matrix", []) if dq else []

    # data lineage
    out["data_lineage"] = dq.get("data_lineage", []) if dq else []

    # per-patient predictions from clinical.db — {patient_id, count, avg_confidence, diseases}
    per_patient = _query_db("""
        SELECT
            patient_id,
            COUNT(*) AS count,
            ROUND(AVG(confidence), 4) AS avg_confidence,
            GROUP_CONCAT(DISTINCT disease) AS diseases
        FROM analyses
        GROUP BY patient_id
        ORDER BY count DESC
    """)
    out["per_patient_predictions"] = per_patient

    # accuracy breakdown — array of {patient, accuracy, sensitivity, f1}
    if acc_ps:
        per_subj = acc_ps.get("per_subject", [])
        out["accuracy_breakdown"] = [
            {
                "patient": s.get("subject", s.get("patient", "?")),
                "accuracy": s.get("accuracy"),
                "sensitivity": s.get("sensitivity", s.get("recall")),
                "f1": s.get("f1", s.get("f1_score")),
            }
            for s in per_subj
            if isinstance(s, dict)
        ]
    else:
        out["accuracy_breakdown"] = []

    # accuracy all options — summary section
    if acc_all:
        summary = {}
        options = acc_all.get("options", {})
        for key, val in options.items():
            summary[key] = {
                "method": val.get("method", key),
                "mean_accuracy": val.get("mean_accuracy"),
            }
        out["accuracy_all_options"] = {
            "subjects": acc_all.get("subjects", []),
            "generated_at": acc_all.get("generated_at"),
            "summary": summary,
        }
    else:
        out["accuracy_all_options"] = {}

    return out


# ──────────────────────────────────────────────────────────────────────
# 3. definitions
# ──────────────────────────────────────────────────────────────────────

def definitions() -> dict:
    """
    Static reference definitions for the Model Monitoring dashboard.
    Returns {sections: [{title, items: [{term, definition}]}]}.
    """
    return {
        "sections": [
            {
                "title": "Model Monitoring Concepts",
                "items": [
                    {"term": "Model Monitoring",
                     "definition": "Continuous surveillance of ML model performance, data quality, and prediction drift in production. Ensures models remain accurate and reliable over time."},
                    {"term": "Production Drift",
                     "definition": "Statistical divergence between training data distributions and live inference data. Detected via PSI (Population Stability Index) and KS (Kolmogorov-Smirnov) tests."},
                    {"term": "Prediction Consistency",
                     "definition": "Agreement between independent prediction pathways (e.g., trust-based vs SHAP-based confidence). Inconsistency flags potential model instability."},
                    {"term": "Data Quality Score",
                     "definition": "Composite metric (0-100) aggregating completeness, validity, timeliness, and uniqueness of input data. Graded A-F."},
                    {"term": "System Health",
                     "definition": "Backend HTTP status, API error counts, database connectivity, and frontend availability. Alerts when any component deviates from healthy baseline."},
                ],
            },
            {
                "title": "Monitoring Metrics & Thresholds",
                "items": [
                    {"term": "PSI (Population Stability Index)",
                     "definition": "Measures distribution shift between reference and live data. PSI < 0.1 = stable, 0.1-0.25 = moderate drift, > 0.25 = significant drift requiring investigation."},
                    {"term": "KS Statistic",
                     "definition": "Kolmogorov-Smirnov test statistic measuring maximum distance between two CDFs. p-value < 0.05 indicates statistically significant drift."},
                    {"term": "Confidence Distribution",
                     "definition": "Histogram of model prediction confidence scores. Shifts toward low confidence may indicate model degradation or out-of-distribution inputs."},
                    {"term": "AI Readiness Score",
                     "definition": "Composite 0-100 score reflecting data suitability for AI. Weighted from completeness, uniqueness, validity, label coverage, and signal quality. Grade: A (>=90), B (>=75), C (>=60), D (>=40), F (<40)."},
                    {"term": "Accuracy",
                     "definition": "Fraction of correctly classified samples out of total test samples. Reported per-patient (patient-specific models) and cross-patient (leave-one-out)."},
                    {"term": "Sensitivity (Recall)",
                     "definition": "True positive rate: fraction of actual seizure/disease windows correctly identified. Critical for clinical safety."},
                    {"term": "F1 Score",
                     "definition": "Harmonic mean of precision and recall. Balances false positives and false negatives. Range 0-1; higher is better."},
                ],
            },
            {
                "title": "Severity Levels",
                "items": [
                    {"term": "None (PSI < 0.10)",
                     "definition": "No meaningful drift detected. Continue normal operation."},
                    {"term": "Low (PSI 0.10-0.15)",
                     "definition": "Minor distributional shift. Log and monitor; review at next scheduled model review."},
                    {"term": "Medium (PSI 0.15-0.25)",
                     "definition": "Moderate drift. Investigate root cause (sensor change, population shift). Consider retraining if trend persists."},
                    {"term": "High (PSI >= 0.25)",
                     "definition": "Significant drift. Flag for immediate review. Model predictions may be unreliable. Activate human oversight and schedule retraining."},
                    {"term": "Severe (systemic)",
                     "definition": "Majority of features show high PSI. Halt automated decisions, require human override, root-cause analysis mandatory before redeployment."},
                ],
            },
            {
                "title": "Clinical Relevance & Regulatory",
                "items": [
                    {"term": "IEC 62304",
                     "definition": "Medical device software lifecycle standard. Model monitoring fulfils post-market surveillance requirements (clause 6.3) by continuously tracking prediction drift, data quality, and system health."},
                    {"term": "FDA AI/ML PCCP",
                     "definition": "Predetermined Change Control Plan. Drift monitoring and consistency checks provide evidence for the FDA's total product lifecycle approach, demonstrating model performance stays within validated bounds."},
                    {"term": "ISO 14971 Risk Management",
                     "definition": "Continuous risk assessment. Drift detection and data quality monitoring are risk controls that trigger re-evaluation when thresholds are breached."},
                    {"term": "EU AI Act (High-Risk)",
                     "definition": "Article 9 requires continuous monitoring of high-risk AI systems. Model monitoring dashboards provide the transparency and audit trail required."},
                    {"term": "ILAE Classification",
                     "definition": "International League Against Epilepsy seizure classification. Monitoring ensures AI predictions remain aligned with current ILAE taxonomy."},
                ],
            },
            {
                "title": "Remediation Strategies",
                "items": [
                    {"term": "Drift Remediation",
                     "definition": "Identify changed features and root cause, collect representative new data, retrain with combined old + new reference data, validate on held-out set, update drift reference baseline."},
                    {"term": "Consistency Failure",
                     "definition": "Verify model bundle versions match across trust panel and SHAP module, check feature extractor alignment, redeploy correct bundle, re-run consistency check."},
                    {"term": "Data Quality Drop",
                     "definition": "Review missing-matrix for modality gaps, investigate pipeline for ingestion failures, coordinate with clinical sites, re-run data quality assessment."},
                    {"term": "System Health Degradation",
                     "definition": "Check backend logs for tracebacks, verify database connectivity and disk space, restart affected services, confirm health endpoint returns HTTP 200."},
                    {"term": "Accuracy Degradation",
                     "definition": "Compare current accuracy against baseline (patient-specific >= 0.95, cross-patient >= 0.70), analyse per-patient breakdown, check for drift/quality confounders, retrain if confirmed."},
                ],
            },
        ],
    }


# ──────────────────────────────────────────────────────────────────────
# CLI entry point — quick self-test
# ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import pprint

    print("=" * 60)
    print("MODEL MONITORING — OVERVIEW")
    print("=" * 60)
    ov = monitoring_overview()
    pprint.pprint(ov, width=120)

    print("\n" + "=" * 60)
    print("MODEL MONITORING — BREAKDOWN")
    print("=" * 60)
    bd = monitoring_breakdown()
    pprint.pprint(bd, width=120)

    print("\n" + "=" * 60)
    print("MODEL MONITORING — DEFINITIONS")
    print("=" * 60)
    df = definitions()
    pprint.pprint(df, width=120)
