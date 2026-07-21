"""Datasets Config Dashboard — 6-disease, 10-dataset real EEG registry
from config/datasets_config.yaml.
6 diseases (schizophrenia/autism/parkinson/stress/epilepsy/depression),
10 datasets, 730 real subjects, accuracy per disease, paths/formats/sources."""

import yaml
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "datasets_config.yaml"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return yaml.safe_load(f)


_DISEASES = ["schizophrenia", "autism", "parkinson", "stress", "epilepsy", "depression"]


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: disease count, dataset count, total subjects, accuracy, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "datasets_config.yaml missing"}

    diseases = []
    total_subjects = 0
    total_datasets = 0
    format_counts = {}
    source_counts = {}

    for disease_key in _DISEASES:
        d = cfg.get(disease_key)
        if not d:
            continue
        subj = d.get("total_subjects", 0)
        acc = d.get("accuracy", "N/A")
        ds_list = d.get("datasets", [])
        n_ds = len(ds_list)
        total_subjects += subj
        total_datasets += n_ds
        diseases.append({
            "disease": disease_key.title(),
            "status": d.get("status", ""),
            "subjects": subj,
            "accuracy": acc,
            "datasets": n_ds,
        })
        for ds in ds_list:
            fmt = ds.get("format", "Unknown")
            for f in fmt.split("/"):
                f = f.strip()
                format_counts[f] = format_counts.get(f, 0) + 1
            src = ds.get("source", "Unknown")
            short_src = src.split(" - ")[0].strip() if " - " in src else src
            source_counts[short_src] = source_counts.get(short_src, 0) + 1

    summary = cfg.get("summary", {})

    # Charts
    subjects_by_disease = [
        {"name": d["disease"], "value": d["subjects"]} for d in diseases
    ]
    datasets_by_disease = [
        {"name": d["disease"], "value": d["datasets"]} for d in diseases
    ]
    format_dist = [
        {"name": k, "value": v} for k, v in sorted(format_counts.items(), key=lambda x: -x[1])
    ]
    source_dist = [
        {"name": k, "value": v} for k, v in sorted(source_counts.items(), key=lambda x: -x[1])
    ]

    return {
        "available": True,
        "title": "Datasets Config (Real EEG Registry)",
        "note": f"{len(diseases)} diseases, {total_datasets} datasets, {total_subjects} real subjects",
        "updated_at": cfg.get("version", "1.0"),
        "kpis": {
            "total_diseases": len(diseases),
            "total_datasets": total_datasets,
            "total_subjects": total_subjects,
            "all_real_data": all(d["status"] == "REAL_DATA" for d in diseases),
            "project": cfg.get("project", "AgenticFinder"),
        },
        "charts": {
            "subjects_by_disease": subjects_by_disease,
            "datasets_by_disease": datasets_by_disease,
            "format_distribution": format_dist,
            "source_distribution": source_dist,
        },
        "summary_table": diseases,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-disease dataset details: paths, subjects, channels, formats, sources."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "datasets_config.yaml missing"}

    disease_cards = []
    for disease_key in _DISEASES:
        d = cfg.get(disease_key)
        if not d:
            continue
        ds_list = d.get("datasets", [])
        datasets = []
        for ds in ds_list:
            datasets.append({
                "name": ds.get("name", ""),
                "path": ds.get("path", ""),
                "file": ds.get("file", ""),
                "subjects": ds.get("subjects", 0),
                "channels": ds.get("channels", "N/A"),
                "sampling_rate": ds.get("sampling_rate", "N/A"),
                "format": ds.get("format", ""),
                "source": ds.get("source", ""),
                "download_url": ds.get("download_url", ""),
            })
        disease_cards.append({
            "disease": disease_key.title(),
            "disease_key": disease_key,
            "status": d.get("status", ""),
            "total_subjects": d.get("total_subjects", 0),
            "accuracy": d.get("accuracy", "N/A"),
            "datasets": datasets,
        })

    base_paths = cfg.get("base_paths", {})

    return {
        "available": True,
        "diseases": disease_cards,
        "base_paths": [{"label": k, "path": v} for k, v in base_paths.items()],
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Glossary, clinical notes, references for datasets config."""
    return {
        "available": True,
        "status_legend": [
            {"status": "REAL_DATA", "meaning": "Real patient EEG recordings used for training/validation"},
            {"status": "SYNTHETIC", "meaning": "Computationally generated data based on real study patterns"},
            {"status": "DOWNLOADING", "meaning": "Dataset download in progress"},
        ],
        "glossary": [
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston / MIT scalp EEG database — 24 pediatric epilepsy patients"},
            {"term": "MHRC", "definition": "Mental Health Research Center Russia — 84-subject schizophrenia EEG dataset"},
            {"term": "RepOD", "definition": "Repository of Open Data (Poland) — 28-subject schizophrenia EEG"},
            {"term": "ASZED", "definition": "African Schizophrenia EEG Dataset — 153 subjects from Zenodo"},
            {"term": "SAM40", "definition": "SAM-40 Cognitive Stress Dataset — 40 subjects, 14-channel EEG"},
            {"term": "EEGMAT", "definition": "EEG Mental Arithmetic Tasks — PhysioNet, 25 subjects, 19 channels"},
            {"term": "EDF", "definition": "European Data Format — standard EEG file format with header metadata"},
            {"term": "CSV", "definition": "Comma-Separated Values — tabular data format for preprocessed features"},
            {"term": "MAT", "definition": "MATLAB binary file format for signal data"},
            {"term": "EEG-MMIDB", "definition": "EEG Motor Movement/Imagery Database — PhysioNet, 85 subjects, 64 channels"},
            {"term": "PhysioNet", "definition": "Open repository of physiological signal databases (physionet.org)"},
            {"term": "Sampling Rate", "definition": "Number of data points recorded per second (Hz) — determines frequency resolution"},
        ],
        "clinical_notes": [
            "All 6 diseases use real patient EEG data for model training and validation.",
            "Schizophrenia has the largest subject pool (265 across 3 datasets) for robust cross-validation.",
            "Channel counts vary from 5 (Emotiv) to 64 (MMIDB) — models must handle heterogeneous montages.",
            "Sampling rates range from 128 Hz to 500 Hz — resampling to a common rate is applied during preprocessing.",
        ],
        "references": [
            {"label": "datasets_config.yaml", "note": "Source config for this dashboard"},
            {"label": "PhysioNet", "note": "physionet.org — open access physiological signal data repository"},
            {"label": "Kaggle", "note": "kaggle.com — community datasets for schizophrenia EEG"},
            {"label": "Zenodo", "note": "zenodo.org — open-access research data repository (ASZED dataset)"},
        ],
    }
