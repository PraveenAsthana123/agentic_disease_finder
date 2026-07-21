"""Data Sources Catalog Dashboard — EEG dataset sources, features, splits
from config/data_sources.yaml.
7 internal datasets, 3 external validation, 10 public EEG datasets,
feature extraction parameters, data split strategy."""

import yaml
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "data_sources.yaml"


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return yaml.safe_load(f)


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: dataset counts, sample totals, feature counts, charts."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "data_sources.yaml missing"}

    internal = cfg.get("internal_datasets", {})
    external = cfg.get("external_validation", {})
    public = cfg.get("public_datasets", {})
    feat = cfg.get("feature_extraction", {})
    split = cfg.get("data_split", {})

    n_internal = len(internal)
    n_external = len(external)
    n_public = len(public)
    total_datasets = n_internal + n_external + n_public

    # Total internal samples (augmented)
    internal_samples = sum(
        d.get("samples", {}).get("augmented", d.get("samples", {}).get("original", 0))
        for d in internal.values()
    )
    external_samples = sum(d.get("samples", 0) for d in external.values())
    public_samples = sum(d.get("samples", 0) for d in public.values())

    # Feature counts
    time_features = len(feat.get("time_domain_features", []))
    freq_features = len(feat.get("frequency_domain_features", []))
    nonlinear_features = len(feat.get("nonlinear_features", []))
    total_features = time_features + freq_features + nonlinear_features
    n_bands = len(feat.get("frequency_bands", {}))

    # Charts: samples per internal disease
    samples_by_disease = [
        {
            "name": d.get("name", k).replace(" EEG Dataset", ""),
            "original": d.get("samples", {}).get("original", 0),
            "augmented": d.get("samples", {}).get("augmented", 0),
        }
        for k, d in internal.items()
    ]

    # Dataset category pie
    category_dist = [
        {"name": "Internal", "value": n_internal},
        {"name": "External Validation", "value": n_external},
        {"name": "Public", "value": n_public},
    ]

    # Public datasets by disease
    disease_counts = {}
    for d in public.values():
        dis = d.get("disease", "unknown")
        disease_counts[dis] = disease_counts.get(dis, 0) + 1
    public_by_disease = [{"name": k.title(), "value": v} for k, v in disease_counts.items()]

    # Feature type distribution
    feature_dist = [
        {"name": "Time-domain", "value": time_features},
        {"name": "Frequency-domain", "value": freq_features},
        {"name": "Nonlinear", "value": nonlinear_features},
    ]

    # Public dataset sample sizes (bar chart)
    public_sample_bars = [
        {"name": d.get("name", k), "value": d.get("samples", 0)}
        for k, d in public.items()
    ]

    return {
        "available": True,
        "title": "Data Sources Catalog",
        "note": f"{total_datasets} datasets across internal, external validation, and public sources",
        "updated_at": "2026-01-26",
        "kpis": {
            "total_datasets": total_datasets,
            "internal": n_internal,
            "external": n_external,
            "public": n_public,
            "internal_samples": internal_samples,
            "external_samples": external_samples,
            "public_samples": public_samples,
            "total_features": total_features,
            "frequency_bands": n_bands,
            "sampling_rate": feat.get("sampling_rate", "N/A"),
        },
        "charts": {
            "category_distribution": category_dist,
            "samples_by_disease": samples_by_disease,
            "public_by_disease": public_by_disease,
            "feature_distribution": feature_dist,
            "public_sample_sizes": public_sample_bars,
        },
        "summary_table": [
            {
                "name": d.get("name", k),
                "category": "Internal",
                "disease": k.title(),
                "samples_original": d.get("samples", {}).get("original", 0),
                "samples_augmented": d.get("samples", {}).get("augmented", 0),
                "features": d.get("features", ""),
                "source": d.get("source", ""),
            }
            for k, d in internal.items()
        ] + [
            {
                "name": d.get("name", k),
                "category": "External",
                "disease": k.replace("_external", "").replace("_ucsd", "").replace("_bonn", "").title(),
                "samples_original": d.get("samples", 0),
                "samples_augmented": "-",
                "features": "-",
                "source": d.get("url", ""),
            }
            for k, d in external.items()
        ],
        "split": {
            "train_method": split.get("training_validation", {}).get("method", ""),
            "train_ratio": split.get("training_validation", {}).get("train_ratio", ""),
            "val_ratio": split.get("training_validation", {}).get("validation_ratio", ""),
            "external_method": split.get("external_validation", {}).get("method", ""),
            "external_ratio": split.get("external_validation", {}).get("ratio", ""),
            "total_samples_per_disease": split.get("total_samples_per_disease", ""),
            "total_diseases": split.get("total_diseases", ""),
            "total_samples": split.get("total_samples", ""),
        },
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Per-category dataset details: internal, external, public, features."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "data_sources.yaml missing"}

    internal = cfg.get("internal_datasets", {})
    external = cfg.get("external_validation", {})
    public = cfg.get("public_datasets", {})
    feat = cfg.get("feature_extraction", {})

    internal_cards = [
        {
            "id": k,
            "name": d.get("name", k),
            "path": d.get("path", ""),
            "original_file": d.get("original_file", ""),
            "augmented_file": d.get("augmented_file", ""),
            "samples_original": d.get("samples", {}).get("original", 0),
            "samples_augmented": d.get("samples", {}).get("augmented", 0),
            "features": d.get("features", 0),
            "classes": d.get("classes", {}),
            "source": d.get("source", ""),
            "preprocessing": d.get("preprocessing", ""),
        }
        for k, d in internal.items()
    ]

    external_cards = [
        {
            "id": k,
            "name": d.get("name", k),
            "path": d.get("path", ""),
            "samples": d.get("samples", 0),
            "url": d.get("url", ""),
            "citation": d.get("citation", ""),
            "status": d.get("status", ""),
        }
        for k, d in external.items()
    ]

    public_cards = [
        {
            "id": k,
            "name": d.get("name", k),
            "url": d.get("url", ""),
            "samples": d.get("samples", 0),
            "disease": d.get("disease", ""),
            "format": d.get("format", ""),
            "access": d.get("access", ""),
            "kaggle": d.get("kaggle", ""),
            "physionet_id": d.get("physionet_id", ""),
            "openneuro_id": d.get("openneuro_id", ""),
            "downloaded": d.get("downloaded", False),
            "local_path": d.get("local_path", ""),
        }
        for k, d in public.items()
    ]

    bands = feat.get("frequency_bands", {})
    feature_params = {
        "sampling_rate": feat.get("sampling_rate", ""),
        "window_size": feat.get("window_size", ""),
        "overlap": feat.get("overlap", ""),
        "bands": [{"name": k, "range_hz": f"{v[0]}–{v[1]}"} for k, v in bands.items()],
        "time_domain": feat.get("time_domain_features", []),
        "frequency_domain": feat.get("frequency_domain_features", []),
        "nonlinear": feat.get("nonlinear_features", []),
    }

    return {
        "available": True,
        "internal": internal_cards,
        "external": external_cards,
        "public": public_cards,
        "feature_params": feature_params,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Glossary, clinical notes, references for data sources."""
    return {
        "available": True,
        "status_legend": [
            {"status": "built", "meaning": "Dataset loaded and used in training/validation"},
            {"status": "simulated", "meaning": "Synthetic dataset based on real study patterns"},
            {"status": "planned", "meaning": "Identified for future integration"},
        ],
        "glossary": [
            {"term": "EEG", "definition": "Electroencephalography — recording of brain electrical activity via scalp electrodes"},
            {"term": "Augmentation", "definition": "Data augmentation via noise injection to increase training set size"},
            {"term": "StandardScaler", "definition": "Z-score normalization (mean=0, std=1) per feature"},
            {"term": "Cross-validation", "definition": "K-fold splitting to estimate model generalization without holdout bias"},
            {"term": "EDF", "definition": "European Data Format — standard file format for EEG recordings"},
            {"term": "BIDS", "definition": "Brain Imaging Data Structure — standardized neuroimaging data organization"},
            {"term": "PhysioNet", "definition": "Open repository of physiological signal databases"},
            {"term": "OpenNeuro", "definition": "Free platform for sharing neuroimaging data in BIDS format"},
            {"term": "Band Power", "definition": "Signal power within a specific frequency band (delta/theta/alpha/beta/gamma)"},
            {"term": "Hjorth Parameters", "definition": "Activity, mobility, complexity — time-domain EEG descriptors"},
            {"term": "Spectral Entropy", "definition": "Shannon entropy of the power spectral density — measures signal complexity"},
            {"term": "DFA", "definition": "Detrended Fluctuation Analysis — measures long-range temporal correlations"},
        ],
        "clinical_notes": [
            "Internal datasets are synthetic, modeled on real study patterns for development; clinical deployment requires real patient data.",
            "External validation uses holdout datasets to detect overfitting before clinical use.",
            "Public datasets vary in quality, labeling, and recording conditions — preprocessing must account for heterogeneity.",
            "Feature extraction uses 256 Hz sampling rate with 50% overlapping windows, consistent across all diseases.",
        ],
        "references": [
            {"label": "data_sources.yaml", "note": "Source config for this dashboard"},
            {"label": "Bonn University Epilepsy Dataset", "note": "Andrzejak RG et al. (2001) — gold-standard epilepsy EEG benchmark"},
            {"label": "CHB-MIT", "note": "PhysioNet scalp EEG database — 23 pediatric epilepsy patients"},
            {"label": "TUH EEG Corpus", "note": "Temple University Hospital — largest public clinical EEG corpus (25k+ recordings)"},
        ],
    }
