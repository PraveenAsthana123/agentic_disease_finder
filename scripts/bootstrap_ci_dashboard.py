"""Bootstrap CI Baselines Dashboard — reads jobs/reports/bootstrap_ci_baselines.json.
Three entry points: overview(), breakdown(), definitions()."""
from pathlib import Path
import json

_ROOT = Path(__file__).resolve().parent.parent
_REPORT = _ROOT / "jobs" / "reports" / "bootstrap_ci_baselines.json"


def _load():
    return json.loads(_REPORT.read_text())


def overview():
    d = _load()
    ps_acc  = d["patient_specific_accuracy_ci"]
    ps_sens = d["patient_specific_sensitivity_ci"]
    cp_acc  = d["cross_patient_rf_accuracy_ci"]
    return {
        "generated_at": d["generated_at"],
        "method": d["method"],
        "kpis": {
            "patient_specific_accuracy_mean":  round(ps_acc["mean"] * 100, 2),
            "patient_specific_accuracy_ci95":  f"{round(ps_acc['ci95_low']*100,1)}–{round(ps_acc['ci95_high']*100,1)}%",
            "patient_specific_sensitivity_mean": round(ps_sens["mean"] * 100, 2),
            "patient_specific_sensitivity_ci95": f"{round(ps_sens['ci95_low']*100,1)}–{round(ps_sens['ci95_high']*100,1)}%",
            "cross_patient_accuracy_mean": round(cp_acc["mean"] * 100, 2),
            "cross_patient_accuracy_ci95": f"{round(cp_acc['ci95_low']*100,1)}–{round(cp_acc['ci95_high']*100,1)}%",
            "n_subjects": ps_acc["n_subjects"],
            "n_bootstrap_iters": ps_acc["n_boot"],
        },
        "confidence_intervals": [
            {
                "metric": "Patient-specific Accuracy",
                "setting": "patient-specific",
                "mean_pct": round(ps_acc["mean"] * 100, 2),
                "ci95_low_pct": round(ps_acc["ci95_low"] * 100, 2),
                "ci95_high_pct": round(ps_acc["ci95_high"] * 100, 2),
                "n_subjects": ps_acc["n_subjects"],
                "n_boot": ps_acc["n_boot"],
            },
            {
                "metric": "Patient-specific Sensitivity",
                "setting": "patient-specific",
                "mean_pct": round(ps_sens["mean"] * 100, 2),
                "ci95_low_pct": round(ps_sens["ci95_low"] * 100, 2),
                "ci95_high_pct": round(ps_sens["ci95_high"] * 100, 2),
                "n_subjects": ps_sens["n_subjects"],
                "n_boot": ps_sens["n_boot"],
            },
            {
                "metric": "Cross-patient Accuracy (RF LOSO)",
                "setting": "cross-patient",
                "mean_pct": round(cp_acc["mean"] * 100, 2),
                "ci95_low_pct": round(cp_acc["ci95_low"] * 100, 2),
                "ci95_high_pct": round(cp_acc["ci95_high"] * 100, 2),
                "n_subjects": cp_acc["n_subjects"],
                "n_boot": cp_acc["n_boot"],
            },
        ],
        "interpretation": (
            "Patient-specific performance is high and narrow-CI (clinically robust). "
            "Cross-patient CI is wide (4 subjects) — honest generalization bound. "
            "Full CHB-MIT (24 subjects) would tighten the cross-patient CI."
        ),
    }


def breakdown():
    d = _load()
    methods = d["baseline_comparison"]["methods"]
    note    = d["baseline_comparison"]["note"]
    # Enrich with our CI data where applicable
    ps_acc  = d["patient_specific_accuracy_ci"]
    cp_acc  = d["cross_patient_rf_accuracy_ci"]
    rows = []
    for m in methods:
        rows.append({
            "method": m["method"],
            "setting": m["setting"],
            "reported": m["reported"],
            "source": m["source"],
            "is_ours": "Ours" in m["method"],
            "ci95": (
                f"{round(ps_acc['ci95_low']*100,1)}–{round(ps_acc['ci95_high']*100,1)}%"
                if "patient-specific" in m["setting"] and "Ours" in m["method"]
                else (
                    f"{round(cp_acc['ci95_low']*100,1)}–{round(cp_acc['ci95_high']*100,1)}%"
                    if "cross-patient" in m["setting"] and "Ours" in m["method"]
                    else "reported only"
                )
            ),
        })
    return {
        "generated_at": d["generated_at"],
        "note": note,
        "methods": rows,
        "summary": {
            "total_methods": len(rows),
            "ours_patient_specific_rank": "competitive (≥ Shoeb 2010 SVM)",
            "ours_cross_patient_rank": "within literature range (0.65–0.85)",
            "honest_caveat": "4 subjects only; do not over-interpret wide cross-patient CI",
        },
    }


def definitions():
    return {
        "terms": [
            {
                "term": "Bootstrap CI",
                "definition": "Confidence interval computed by resampling subjects (not windows) 2 000 times, then taking the 2.5th and 97.5th percentiles of the resampled metric distribution.",
                "why_subjects": "EEG windows from the same subject are correlated. Resampling windows would underestimate variance. Subject-level bootstrap is the statistically correct approach.",
            },
            {
                "term": "Patient-specific setting",
                "definition": "Model trained on earlier epochs of the SAME patient and tested on later epochs (temporal split). High accuracy expected; not generalizable to new patients.",
            },
            {
                "term": "Cross-patient / LOSO setting",
                "definition": "Leave-One-Subject-Out: model trained on N-1 subjects, tested on the held-out subject. Honest generalization metric. Wide CI with 4 subjects; narrows with more data.",
            },
            {
                "term": "95% CI width interpretation",
                "definition": "Narrow CI (e.g. 97.3–98.7%) → reliable estimate. Wide CI (e.g. 40–93%) → high variance; more subjects needed to draw strong conclusions.",
            },
            {
                "term": "Shoeb (2010) benchmark",
                "definition": "Patient-specific SVM on CHB-MIT achieving ~0.96 sensitivity. Published in MIT PhD thesis; widely cited as a baseline.",
            },
            {
                "term": "Truong et al. (2018)",
                "definition": "CNN-based patient-specific approach reporting ~0.97 AUC on CHB-MIT. Published in Neural Networks journal.",
            },
        ],
        "methodology_note": (
            "All our metrics use subject-level bootstrap (not window-level) to avoid "
            "inflated confidence. This is a conservative, defensible approach for a DBA thesis."
        ),
    }
