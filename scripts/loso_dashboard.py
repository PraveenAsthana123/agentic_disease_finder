"""
LOSO (Leave-One-Subject-Out) Cross-Validation Dashboard
CHB-MIT per-subject LOSO results — the patient-independent, viva-defensible metric.
24 subjects, real epilepsy seizure detection performance.
Also queries model_comparison table for in-sample benchmark context.
"""

import sqlite3
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(BASE, "data", "clinical.db")

# Real CHB-MIT LOSO per-subject results
# Source: Leave-one-subject-out cross-validation on CHB-MIT Scalp EEG dataset
# Mean: Sensitivity 35.1% · Specificity 96.9% · AUC 0.846
CHB_MIT_LOSO = [
    {"subject": "chb01", "seizure_epochs": 6,   "sensitivity": 0.167, "specificity": 1.000, "auc": 0.951},
    {"subject": "chb02", "seizure_epochs": 11,  "sensitivity": 0.727, "specificity": 1.000, "auc": 1.000},
    {"subject": "chb03", "seizure_epochs": 56,  "sensitivity": 0.429, "specificity": 0.998, "auc": 0.981},
    {"subject": "chb04", "seizure_epochs": 52,  "sensitivity": 0.077, "specificity": 0.995, "auc": 0.937},
    {"subject": "chb05", "seizure_epochs": 75,  "sensitivity": 0.813, "specificity": 0.958, "auc": 0.959},
    {"subject": "chb06", "seizure_epochs": 29,  "sensitivity": 0.448, "specificity": 0.569, "auc": 0.620},
    {"subject": "chb07", "seizure_epochs": 42,  "sensitivity": 0.762, "specificity": 0.985, "auc": 0.982},
    {"subject": "chb08", "seizure_epochs": 120, "sensitivity": 0.175, "specificity": 1.000, "auc": 0.841},
    {"subject": "chb09", "seizure_epochs": 39,  "sensitivity": 0.897, "specificity": 0.987, "auc": 0.968},
    {"subject": "chb10", "seizure_epochs": 61,  "sensitivity": 0.623, "specificity": 0.982, "auc": 0.957},
    {"subject": "chb11", "seizure_epochs": 104, "sensitivity": 0.490, "specificity": 0.987, "auc": 0.949},
    {"subject": "chb12", "seizure_epochs": 224, "sensitivity": 0.076, "specificity": 0.943, "auc": 0.650},
    {"subject": "chb13", "seizure_epochs": 79,  "sensitivity": 0.089, "specificity": 0.880, "auc": 0.533},
    {"subject": "chb14", "seizure_epochs": 30,  "sensitivity": 0.000, "specificity": 1.000, "auc": 0.625},
    {"subject": "chb15", "seizure_epochs": 270, "sensitivity": 0.000, "specificity": 1.000, "auc": 0.489},
    {"subject": "chb16", "seizure_epochs": 21,  "sensitivity": 0.048, "specificity": 0.976, "auc": 0.721},
    {"subject": "chb17", "seizure_epochs": 39,  "sensitivity": 0.051, "specificity": 0.997, "auc": 0.891},
    {"subject": "chb18", "seizure_epochs": 47,  "sensitivity": 0.255, "specificity": 1.000, "auc": 0.881},
    {"subject": "chb19", "seizure_epochs": 34,  "sensitivity": 0.618, "specificity": 1.000, "auc": 0.951},
    {"subject": "chb20", "seizure_epochs": 45,  "sensitivity": 0.022, "specificity": 1.000, "auc": 0.687},
    {"subject": "chb21", "seizure_epochs": 28,  "sensitivity": 0.179, "specificity": 1.000, "auc": 0.968},
    {"subject": "chb22", "seizure_epochs": 29,  "sensitivity": 0.621, "specificity": 0.996, "auc": 0.985},
    {"subject": "chb23", "seizure_epochs": 60,  "sensitivity": 0.233, "specificity": 1.000, "auc": 0.934},
    {"subject": "chb24", "seizure_epochs": 79,  "sensitivity": 0.620, "specificity": 0.998, "auc": 0.854},
]


def _tier(sens):
    if sens >= 0.6:
        return "high"
    elif sens >= 0.2:
        return "moderate"
    elif sens > 0.0:
        return "low"
    else:
        return "zero"


def overview():
    """KPIs, sensitivity distribution tiers, subject count, dataset summary, in-sample vs LOSO gap."""
    subjects = len(CHB_MIT_LOSO)
    total_seizure_epochs = sum(r["seizure_epochs"] for r in CHB_MIT_LOSO)
    mean_sens = round(sum(r["sensitivity"] for r in CHB_MIT_LOSO) / subjects, 4)
    mean_spec = round(sum(r["specificity"] for r in CHB_MIT_LOSO) / subjects, 4)
    mean_auc  = round(sum(r["auc"]         for r in CHB_MIT_LOSO) / subjects, 4)

    high    = [r for r in CHB_MIT_LOSO if r["sensitivity"] >= 0.6]
    moderate = [r for r in CHB_MIT_LOSO if 0.2 <= r["sensitivity"] < 0.6]
    low     = [r for r in CHB_MIT_LOSO if 0.0 < r["sensitivity"] < 0.2]
    zero    = [r for r in CHB_MIT_LOSO if r["sensitivity"] == 0.0]

    # In-sample benchmark from model_comparison table (seizure_detection on chb_mit)
    in_sample_accuracy = None
    in_sample_auc      = None
    in_sample_model    = None
    try:
        con = sqlite3.connect(DB)
        cur = con.cursor()
        cur.execute("""
            SELECT model_name, accuracy, auc_roc
            FROM model_comparison
            WHERE task='seizure_detection' AND dataset='chb_mit' AND status='completed'
            ORDER BY auc_roc DESC LIMIT 1
        """)
        row = cur.fetchone()
        if row:
            in_sample_model    = row[0]
            in_sample_accuracy = round(row[1], 4)
            in_sample_auc      = round(row[2], 4)
        con.close()
    except Exception:
        pass

    loso_gap = None
    if in_sample_auc is not None:
        loso_gap = round(in_sample_auc - mean_auc, 4)

    return {
        "kpis": {
            "subjects":             subjects,
            "total_seizure_epochs": total_seizure_epochs,
            "mean_sensitivity":     mean_sens,
            "mean_specificity":     mean_spec,
            "mean_auc":             mean_auc,
            "high_sens_subjects":   len(high),
            "zero_sens_subjects":   len(zero),
        },
        "sensitivity_tiers": {
            "high":     {"label": "High (≥60%)",    "count": len(high),    "subjects": [r["subject"] for r in high]},
            "moderate": {"label": "Moderate (20-60%)", "count": len(moderate), "subjects": [r["subject"] for r in moderate]},
            "low":      {"label": "Low (1-20%)",    "count": len(low),     "subjects": [r["subject"] for r in low]},
            "zero":     {"label": "Zero (0%)",      "count": len(zero),    "subjects": [r["subject"] for r in zero]},
        },
        "in_sample_benchmark": {
            "model":    in_sample_model,
            "accuracy": in_sample_accuracy,
            "auc":      in_sample_auc,
        },
        "loso_vs_insample_gap": loso_gap,
        "dataset": {
            "name":          "CHB-MIT Scalp EEG Database",
            "source":        "PhysioNet",
            "subjects":      subjects,
            "sampling_rate": "256 Hz",
            "channels":      23,
            "cv_method":     "Leave-One-Subject-Out (LOSO)",
            "window_size":   "1-second epochs",
            "note":          "Patient-independent generalization metric. Wide per-subject spread is the key finding: seizure morphology is highly patient-specific.",
        },
    }


def breakdown():
    """Full per-subject LOSO table with tier annotations, sorted by sensitivity desc."""
    # Annotate each subject
    annotated = []
    for r in CHB_MIT_LOSO:
        annotated.append({
            **r,
            "tier":              _tier(r["sensitivity"]),
            "sensitivity_pct":   round(r["sensitivity"] * 100, 1),
            "specificity_pct":   round(r["specificity"] * 100, 1),
            "auc_pct":           round(r["auc"] * 100, 1),
        })

    sorted_by_sens = sorted(annotated, key=lambda x: x["sensitivity"], reverse=True)

    # In-sample model comparison (top 6 CHB-MIT results)
    in_sample_rows = []
    try:
        con = sqlite3.connect(DB)
        cur = con.cursor()
        cur.execute("""
            SELECT model_name, model_type, accuracy, f1_score, auc_roc, training_time_sec
            FROM model_comparison
            WHERE task='seizure_detection' AND dataset='chb_mit' AND status='completed'
            ORDER BY auc_roc DESC LIMIT 6
        """)
        cols = ["model_name", "model_type", "accuracy", "f1_score", "auc_roc", "training_time_sec"]
        in_sample_rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        con.close()
    except Exception:
        pass

    # AUC distribution buckets for chart
    auc_dist = {">=0.95": 0, "0.85-0.95": 0, "0.70-0.85": 0, "<0.70": 0}
    for r in CHB_MIT_LOSO:
        a = r["auc"]
        if a >= 0.95:
            auc_dist[">=0.95"] += 1
        elif a >= 0.85:
            auc_dist["0.85-0.95"] += 1
        elif a >= 0.70:
            auc_dist["0.70-0.85"] += 1
        else:
            auc_dist["<0.70"] += 1

    return {
        "subjects":          sorted_by_sens,
        "total_subjects":    len(CHB_MIT_LOSO),
        "auc_distribution":  [{"bucket": k, "count": v} for k, v in auc_dist.items()],
        "in_sample_models":  in_sample_rows,
    }


def definitions():
    """LOSO methodology, metric definitions, CHB-MIT dataset, and generalization glossary."""
    return {
        "cv_method": {
            "loso": "Leave-One-Subject-Out — train on all subjects except one, test on the held-out subject. Repeats for all 24 subjects.",
            "patient_specific": "Train and test on the same subject (high accuracy, no generalization claim).",
            "k_fold": "Random k-fold cross-validation — folds may share subject data (overestimates generalization).",
        },
        "metrics": {
            "sensitivity": "True Positive Rate — fraction of actual seizure epochs correctly detected. Critical for patient safety; missing a seizure is dangerous.",
            "specificity": "True Negative Rate — fraction of non-seizure epochs correctly rejected. High specificity ≈ low false-alarm rate.",
            "auc_roc":     "Area Under the ROC Curve — overall discriminative ability across all thresholds (1.0 = perfect, 0.5 = random).",
        },
        "dataset": {
            "name":        "CHB-MIT Scalp EEG Database",
            "source":      "PhysioNet (physionet.org/content/chbmit)",
            "subjects":    24,
            "recordings":  "23-channel scalp EEG, 256 Hz, continuous long-term monitoring",
            "seizures":    "Total 198 seizures across all subjects",
            "note":        "Gold-standard public epilepsy EEG dataset for benchmarking seizure detection algorithms.",
        },
        "sensitivity_tiers": {
            "high":     "Sensitivity ≥ 60% — reliable detection, clinically useful for that subject",
            "moderate": "Sensitivity 20–60% — partial detection, misses majority of seizures",
            "low":      "Sensitivity 1–20% — nearly misses all seizures, high false-negative rate",
            "zero":     "Sensitivity 0% — model trained on other subjects cannot detect seizures in this subject",
        },
        "key_finding": (
            "The wide per-subject spread (chb09 89.7% → chb14/15 0% sensitivity) is the primary research finding: "
            "seizure morphology is highly patient-specific. A universal cross-patient detector has fundamentally "
            "different accuracy than a patient-specific model. Mean LOSO AUC 0.846 vs in-sample AUC 0.97+ illustrates "
            "the generalization gap that motivates personalized epilepsy AI."
        ),
        "glossary": {
            "LOSO":             "Leave-One-Subject-Out cross-validation",
            "epoch":            "Fixed-length EEG segment (here: 1 second at 256 Hz = 256 samples)",
            "generalization":   "Model's ability to perform on subjects not seen during training",
            "patient-specific": "Model trained and evaluated on the same patient — not a generalization claim",
            "seizure epoch":    "1-second window containing ictal (seizure) EEG activity",
        },
    }
