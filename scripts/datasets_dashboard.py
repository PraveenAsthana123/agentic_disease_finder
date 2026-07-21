"""Datasets Dashboard — 6-disease EEG dataset registry visualization
from config/datasets.json.
6 diseases, 9 datasets, 739 total subjects, all REAL_DATA."""

import json
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "datasets.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: disease count, dataset count, total subjects,
    accuracy stats, format distribution, subject/accuracy charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "datasets.json missing"}

    diseases = cfg.get("diseases", {})
    total_diseases = len(diseases)

    # Collect all datasets and per-disease summaries
    all_datasets = []
    disease_list = []
    accuracies = []
    subject_distribution = []
    accuracy_distribution = []
    format_counter = Counter()

    for dname, dinfo in diseases.items():
        ds_list = dinfo.get("datasets", [])
        formats = list(set(d.get("format", "unknown") for d in ds_list))
        disease_list.append({
            "name": dname,
            "status": dinfo.get("status", ""),
            "total_subjects": dinfo.get("total_subjects", 0),
            "accuracy": dinfo.get("accuracy", 0),
            "dataset_count": len(ds_list),
            "formats": formats,
        })
        subject_distribution.append({
            "name": dname,
            "value": dinfo.get("total_subjects", 0),
        })
        accuracy_distribution.append({
            "name": dname,
            "value": dinfo.get("accuracy", 0),
        })
        acc = dinfo.get("accuracy")
        if acc is not None:
            accuracies.append(acc)
        for ds in ds_list:
            fmt = ds.get("format", "unknown")
            format_counter[fmt] += 1
            all_datasets.append(ds)

    total_datasets = len(all_datasets)
    total_subjects = sum(d.get("total_subjects", 0) for d in diseases.values())
    all_real = all(d.get("status") == "REAL_DATA" for d in diseases.values())
    unique_formats = sorted(set(format_counter.keys()))
    avg_accuracy = round(sum(accuracies) / len(accuracies), 2) if accuracies else None

    format_distribution = [
        {"name": fmt, "value": cnt}
        for fmt, cnt in sorted(format_counter.items(), key=lambda x: -x[1])
    ]

    return {
        "available": True,
        "title": "Datasets Registry",
        "version": cfg.get("version", ""),
        "project": cfg.get("project", ""),
        "kpis": {
            "total_diseases": total_diseases,
            "total_datasets": total_datasets,
            "total_subjects": total_subjects,
            "all_real_data": all_real,
            "formats": unique_formats,
            "format_count": len(unique_formats),
            "avg_accuracy": avg_accuracy,
        },
        "diseases": disease_list,
        "subject_distribution": subject_distribution,
        "accuracy_distribution": accuracy_distribution,
        "format_distribution": format_distribution,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-disease detail with all datasets expanded, plus flat
    all-datasets list with disease field."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "datasets.json missing"}

    diseases_raw = cfg.get("diseases", {})
    diseases_out = []
    all_datasets = []

    for dname, dinfo in diseases_raw.items():
        ds_list = dinfo.get("datasets", [])
        expanded = []
        for ds in ds_list:
            row = {
                "name": ds.get("name", ""),
                "subjects": ds.get("subjects", 0),
                "channels": ds.get("channels"),
                "sampling_rate": ds.get("sampling_rate"),
                "format": ds.get("format", ""),
                "source": ds.get("source", ""),
                "is_downloaded": ds.get("is_downloaded", False),
            }
            expanded.append(row)
            all_datasets.append({**row, "disease": dname})

        diseases_out.append({
            "name": dname,
            "status": dinfo.get("status", ""),
            "total_subjects": dinfo.get("total_subjects", 0),
            "accuracy": dinfo.get("accuracy", 0),
            "datasets": expanded,
        })

    return {
        "available": True,
        "diseases": diseases_out,
        "all_datasets": all_datasets,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "REAL_DATA", "description": "Disease uses verified, downloaded real-world EEG datasets — no synthetic or placeholder data"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — recording of electrical activity of the brain via scalp electrodes"},
            {"term": "EDF", "definition": "European Data Format — standard file format for multi-channel biomedical signals including EEG"},
            {"term": "CSV", "definition": "Comma-Separated Values — tabular data format used for pre-processed EEG feature matrices"},
            {"term": "MAT", "definition": "MATLAB data format — binary format used by some EEG acquisition and analysis tools"},
            {"term": "CHB-MIT", "definition": "CHB-MIT Scalp EEG Database — benchmark epilepsy dataset from Children's Hospital Boston / MIT"},
            {"term": "PhysioNet", "definition": "Open-access repository of physiological signal databases maintained by MIT Laboratory for Computational Physiology"},
            {"term": "Sampling Rate", "definition": "Number of data points recorded per second (Hz) — determines temporal resolution of EEG signals"},
            {"term": "Channels", "definition": "Number of electrode positions used for EEG recording — higher count provides better spatial resolution"},
            {"term": "MHRC", "definition": "Mental Health Research Center — Russian dataset of schizophrenia EEG recordings"},
            {"term": "ASZED", "definition": "African Schizophrenia EEG Dataset — multi-site schizophrenia EEG dataset hosted on Zenodo"},
            {"term": "SAM-40", "definition": "Stress Analysis using Multimodal signals — 40-subject cognitive stress dataset with EEG and physiological data"},
            {"term": "SMOTE", "definition": "Synthetic Minority Oversampling Technique — class-balancing method for imbalanced EEG datasets"},
        ],
        "clinical_notes": [
            "All 6 diseases use verified real-world EEG datasets — no synthetic data in any classification pipeline.",
            "Total subject pool across all diseases is 739, spanning datasets from 5+ countries and institutions.",
            "Sampling rates range from 128 Hz to 500 Hz; channel counts from 14 to 64 — all clinically standard configurations.",
            "Dataset formats include EDF (clinical standard), CSV (pre-processed features), and MAT (MATLAB) — covering raw and processed stages.",
        ],
        "references": [
            {"ref": "PhysioNet", "detail": "Goldberger et al. (2000) — open-access physiological signal repository (CHB-MIT, EEGMAT, EEG-MMIDB)"},
            {"ref": "Kaggle", "detail": "Kaggle open datasets platform — source for MHRC schizophrenia EEG dataset"},
            {"ref": "Zenodo", "detail": "CERN open-access research data repository — source for ASZED schizophrenia EEG dataset"},
            {"ref": "RepOD", "detail": "Repository for Open Data (Poland) — source for RepOD schizophrenia EEG dataset"},
        ],
    }
