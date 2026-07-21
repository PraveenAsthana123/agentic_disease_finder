"""Data Configuration Dashboard — 7-disease EEG data configuration visualization
from config/data_config.json.
7 diseases, 70 total datasets, 47 features, 3 validation datasets."""

import json
from pathlib import Path
from collections import Counter

_CFG = Path(__file__).resolve().parent.parent / "config" / "data_config.json"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return json.load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: diseases, datasets, features, validation, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "data_config.json missing"}

    diseases = cfg.get("diseases", [])
    total_diseases = cfg.get("total_diseases", len(diseases))
    total_datasets = cfg.get("total_datasets", 0)
    features_cfg = cfg.get("features", {})
    total_features = features_cfg.get("total_count", 0)
    categories = features_cfg.get("categories", {})
    feature_categories = len(categories)
    validation = cfg.get("validation_datasets", {})
    validation_count = len(validation)
    total_validation_size_mb = sum(v.get("size_mb", 0) for v in validation.values())
    compatibility = cfg.get("compatibility", {})
    platforms = compatibility.get("platforms", [])

    # Collect all primary datasets across diseases
    all_primary = []
    datasets_per_disease = []
    subjects_per_disease = []
    diseases_summary = []
    format_counter = Counter()
    license_counter = Counter()
    auto_count = 0
    manual_count = 0
    source_set = set()

    for d in diseases:
        pds = d.get("primary_datasets", [])
        ds_count = d.get("datasets_count", len(pds))
        total_subj = sum(pd.get("subjects", 0) for pd in pds)

        datasets_per_disease.append({"name": d.get("display_name", d.get("name", "")), "value": ds_count})
        subjects_per_disease.append({"name": d.get("display_name", d.get("name", "")), "value": total_subj})
        diseases_summary.append({
            "name": d.get("name", ""),
            "display_name": d.get("display_name", ""),
            "description": d.get("description", ""),
            "datasets_count": ds_count,
            "primary_count": len(pds),
            "total_subjects": total_subj,
        })

        for pd in pds:
            all_primary.append(pd)
            fmt = pd.get("format")
            if fmt:
                format_counter[fmt] += 1
            lic = pd.get("license")
            if lic:
                license_counter[lic] += 1
            if pd.get("auto_download"):
                auto_count += 1
            else:
                manual_count += 1
            src = pd.get("source")
            if src:
                source_set.add(src)

    format_distribution = [
        {"name": fmt, "value": cnt}
        for fmt, cnt in sorted(format_counter.items(), key=lambda x: -x[1])
    ]
    license_distribution = [
        {"name": lic, "value": cnt}
        for lic, cnt in sorted(license_counter.items(), key=lambda x: -x[1])
    ]
    auto_download_distribution = [
        {"name": "Auto", "value": auto_count},
        {"name": "Manual", "value": manual_count},
    ]

    return {
        "available": True,
        "title": "Data Configuration",
        "version": cfg.get("version", ""),
        "project": cfg.get("project", ""),
        "kpis": {
            "total_diseases": total_diseases,
            "total_datasets": total_datasets,
            "total_primary_datasets": len(all_primary),
            "total_features": total_features,
            "feature_categories": feature_categories,
            "validation_datasets": validation_count,
            "total_validation_size_mb": round(total_validation_size_mb, 1),
            "platforms_supported": len(platforms),
            "download_sources": len(source_set),
        },
        "datasets_per_disease": datasets_per_disease,
        "subjects_per_disease": subjects_per_disease,
        "format_distribution": format_distribution,
        "license_distribution": license_distribution,
        "auto_download_distribution": auto_download_distribution,
        "diseases_summary": diseases_summary,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-disease datasets, features breakdown, validation, download URLs."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "data_config.json missing"}

    # Diseases with primary datasets
    diseases_raw = cfg.get("diseases", [])
    diseases_out = []
    for d in diseases_raw:
        pds = d.get("primary_datasets", [])
        expanded = []
        for pd in pds:
            expanded.append({
                "id": pd.get("id", ""),
                "name": pd.get("name", ""),
                "source": pd.get("source", ""),
                "subjects": pd.get("subjects", 0),
                "channels": pd.get("channels"),
                "sampling_rate": pd.get("sampling_rate"),
                "format": pd.get("format", ""),
                "license": pd.get("license", ""),
                "auto_download": pd.get("auto_download", False),
                "url": pd.get("url", ""),
            })
        diseases_out.append({
            "name": d.get("name", ""),
            "display_name": d.get("display_name", ""),
            "description": d.get("description", ""),
            "datasets_count": d.get("datasets_count", len(pds)),
            "primary_datasets": expanded,
        })

    # Validation datasets
    val_raw = cfg.get("validation_datasets", {})
    val_out = []
    for vid, vinfo in val_raw.items():
        files = vinfo.get("files", [])
        val_out.append({
            "id": vid,
            "path": vinfo.get("path", ""),
            "files": files,
            "file_count": len(files),
            "size_mb": vinfo.get("size_mb", 0),
            "status": vinfo.get("status", "pending"),
        })

    # Features
    features_cfg = cfg.get("features", {})
    categories_raw = features_cfg.get("categories", {})
    categories_out = {}
    for cat_name, cat_val in categories_raw.items():
        if isinstance(cat_val, list):
            categories_out[cat_name] = cat_val
        elif isinstance(cat_val, dict):
            # Spectral: flatten bands into features list
            feats = list(cat_val.get("features", []))
            bands = cat_val.get("bands", {})
            for band_name in bands:
                feats.append(band_name)
            categories_out[cat_name] = feats
            # Also include bands detail
            categories_out[cat_name + "_bands"] = {
                k: v for k, v in bands.items()
            }

    features_out = {
        "total_count": features_cfg.get("total_count", 0),
        "categories": categories_out,
    }

    # Download URLs
    download_raw = cfg.get("download_urls", {})
    download_out = {}
    for group, urls in download_raw.items():
        download_out[group] = [
            {"id": uid, "url": uurl} for uid, uurl in urls.items()
        ]

    return {
        "available": True,
        "diseases": diseases_out,
        "validation_datasets": val_out,
        "features": features_out,
        "download_urls": download_out,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Status legend, glossary, clinical notes, references."""
    return {
        "available": True,
        "status_legend": [
            {"status": "downloaded", "color": "#22c55e", "label": "Downloaded"},
            {"status": "pending", "color": "#f97316", "label": "Pending"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — recording of electrical brain activity via scalp electrodes"},
            {"term": "EDF", "definition": "European Data Format — standard file format for multi-channel biomedical signals"},
            {"term": "BIDS", "definition": "Brain Imaging Data Structure — standardized format for organizing neuroimaging datasets"},
            {"term": "PhysioNet", "definition": "Open-access repository of physiological signal databases maintained by MIT"},
            {"term": "OpenNeuro", "definition": "Free and open platform for sharing neuroimaging data in BIDS format"},
            {"term": "Sampling Rate", "definition": "Number of data points recorded per second (Hz) — determines temporal resolution"},
            {"term": "Channels", "definition": "Number of electrode positions used for EEG recording — determines spatial resolution"},
            {"term": "DUA", "definition": "Data Use Agreement — legal contract required before accessing restricted datasets"},
            {"term": "ODC-BY", "definition": "Open Data Commons Attribution License — requires attribution for data reuse"},
            {"term": "CC0", "definition": "Creative Commons Zero — public domain dedication, no restrictions on data use"},
            {"term": "Auto Download", "definition": "Dataset can be programmatically downloaded without manual steps or DUA signing"},
            {"term": "Validation Dataset", "definition": "Pre-downloaded reference dataset used to verify pipeline correctness and accuracy"},
        ],
        "clinical_notes": [
            "All 7 diseases have curated primary datasets from peer-reviewed sources — no synthetic data in validation pipelines.",
            "Datasets requiring DUA (Data Use Agreement) are marked as manual download to ensure compliance.",
            "Validation datasets are pre-downloaded and version-locked to ensure reproducible benchmarking.",
            "Feature extraction covers 47 features across 4 categories: statistical, spectral, temporal, and nonlinear.",
        ],
        "references": [
            "PhysioNet — Goldberger et al. (2000), open-access physiological signal repository (CHB-MIT, EEGMMIDB, Sleep-EDF)",
            "OpenNeuro — Free platform for sharing BIDS-formatted neuroimaging data (ds003490, ds004504, ds004186)",
            "BIDS Standard — Brain Imaging Data Structure specification for organizing and describing neuroimaging datasets",
            "MNE-Python — Open-source Python library for EEG/MEG analysis, preprocessing, and feature extraction",
        ],
    }
