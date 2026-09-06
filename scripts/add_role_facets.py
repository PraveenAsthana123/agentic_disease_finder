#!/usr/bin/env python3
"""Add `tools` + `ai_solutions` facets to each consultant role (idempotent)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = ROOT / "config" / "consultant_matrix.json"

TOOLS = {
    "neurologist": ["EEG viewer", "ILAE classifier", "ICD-10 coder", "PACS / MRI viewer", "EMR"],
    "neurophysiologist": ["EEG montage tool", "Spike/HFO detector", "Spectral analyzer", "ICA artifact tool"],
    "eeg_technologist": ["EEG amplifier", "Impedance meter", "EDF/BDF exporter", "Montage editor"],
    "psychiatrist": ["PHQ-9", "GAD-7", "Sleep scales", "Medication database", "EMR notes"],
    "psychologist": ["MoCA", "MMSE", "Neuropsych battery", "Norm-reference tables"],
    "occupational_therapist": ["ADL scales", "QOLIE-31", "Functional assessment kit"],
    "biostatistician": ["R", "Python (statsmodels)", "Power-analysis tools", "GroupKFold/CV"],
    "ai_advisor": ["SHAP", "scikit-learn", "MLflow", "Drift monitor", "Surrogate-tree"],
    "governance_advisor": ["OPA / policy engine", "Audit log", "DLP / PII scanner", "Fairness toolkit (AIF360/Fairlearn)"],
    "methodology_advisor": ["Reference manager", "Plagiarism check", "Stats reproducibility tools"],
}
AI_SOLUTIONS = {
    "neurologist": [
        {"challenge": "Inter-rater variability", "ai": "AI seizure-classification suggestion + SHAP → reduces disagreement"},
        {"challenge": "Long-term EEG review time", "ai": "AI pre-screens & flags IED segments → focused review"},
        {"challenge": "Subtle focal abnormalities missed", "ai": "Sensitivity-tuned detector surfaces low-amplitude spikes"},
    ],
    "neurophysiologist": [
        {"challenge": "EEG interpretation subjectivity", "ai": "Quantitative biomarkers (band power, entropy) standardize reads"},
        {"challenge": "Artifacts confound signal", "ai": "ICA + ML artifact rejection cleans before analysis"},
        {"challenge": "Montage/site variability", "ai": "Auto-harmonization to 10-20 reference"},
    ],
    "eeg_technologist": [
        {"challenge": "Poor signal quality", "ai": "Real-time AI signal-quality scoring + alerts during recording"},
        {"challenge": "Artifact identification", "ai": "Auto artifact tagging (eye-blink/muscle/ECG)"},
        {"challenge": "Inconsistent metadata", "ai": "Auto metadata capture from device + EDF header"},
    ],
    "psychiatrist": [
        {"challenge": "Depression/anxiety overlap with seizures", "ai": "NLP extracts mood/anxiety from notes → confounder flag"},
        {"challenge": "Repetitive PHQ/GAD scoring", "ai": "Auto PHQ-9/GAD-7 scoring + summary"},
        {"challenge": "Medication mood effects", "ai": "Med-effect detector correlates AED with mood change"},
    ],
    "psychologist": [
        {"challenge": "Manual cognitive scoring", "ai": "Auto MoCA/MMSE scoring + domain breakdown"},
        {"challenge": "Low score attribution", "ai": "EEG-cognition correlation isolates epilepsy effect"},
        {"challenge": "Long reports", "ai": "LLM summarizes neuropsych reports"},
    ],
    "occupational_therapist": [
        {"challenge": "Functional impact under-documented", "ai": "Auto ADL scoring from structured intake"},
        {"challenge": "Self-report bias", "ai": "Cross-check ADL vs outcome data"},
    ],
    "biostatistician": [
        {"challenge": "Small sample / imbalance", "ai": "Auto subject-wise CV + balanced metrics (F1/AUC)"},
        {"challenge": "Leakage risk", "ai": "Automated leakage/data-split audit"},
        {"challenge": "Manual stats reporting", "ai": "Auto power analysis + significance reporting"},
    ],
    "ai_advisor": [
        {"challenge": "Model learns artifacts", "ai": "SHAP/Grad-CAM verifies attention on brain regions not artifacts"},
        {"challenge": "Overfitting / black-box", "ai": "Surrogate tree + fidelity score for interpretability"},
        {"challenge": "Biomarker instability", "ai": "Drift detection + retraining triggers"},
    ],
    "governance_advisor": [
        {"challenge": "De-identification / PII", "ai": "Auto PII detection + redaction (50-pattern scanner)"},
        {"challenge": "Bias / fairness", "ai": "Fairness gates (disparate impact, equal opportunity)"},
        {"challenge": "Audit completeness", "ai": "Auto decision-audit rows (UTC+local timestamp)"},
    ],
    "methodology_advisor": [
        {"challenge": "Scope creep", "ai": "AI checks claims-to-evidence coverage"},
        {"challenge": "Weak methodology framing", "ai": "Methodology-gap detector vs publication standards"},
    ],
}


def main():
    d = json.loads(F.read_text())
    for c in d["consultants"]:
        cid = c["id"]
        c["tools"] = TOOLS.get(cid, [])
        c["ai_solutions"] = AI_SOLUTIONS.get(cid, [])
    F.write_text(json.dumps(d, indent=2), encoding="utf-8")
    print(f"Added tools + ai_solutions to {len(d['consultants'])} roles -> {F}")


if __name__ == "__main__":
    main()
