"""Data Inventory Dashboard — per-disease EEG data inventory tracking
from config/data_inventory.yaml.
7 diseases, 3 additional datasets, synthetic + real EEG + external validation,
download status, disk usage summary."""

import yaml
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "data_inventory.yaml"

DISEASES = ["epilepsy", "parkinson", "alzheimer", "schizophrenia", "depression", "autism", "stress"]
ADDITIONAL = ["motor_imagery", "siena_epilepsy", "sleep"]


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return yaml.safe_load(f)


def _parse_records(val):
    """Parse record strings like '200 (augmented from 50)' or plain ints."""
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        import re
        m = re.match(r"(\d[\d,]*)", val.replace(",", ""))
        return int(m.group(1)) if m else 0
    return 0


def _parse_size(val):
    """Return size string as-is for display."""
    if val is None:
        return "0 B"
    return str(val)


def _status_key(s):
    """Normalize status string to a category key."""
    if not s:
        return "unknown"
    s = s.upper()
    if "COMPLETE" in s or "DOWNLOADED" in s or "SYMLINKED" in s:
        return "downloaded"
    if "PARTIAL" in s:
        return "partial"
    if "DOWNLOADING" in s:
        return "downloading"
    if "FAIL" in s or "MISSING" in s:
        return "failed"
    if "SIMULATED" in s:
        return "simulated"
    if "CACHE" in s or "INDEX" in s:
        return "cache_only"
    return "other"


# ── overview ────────────────────────────────────────────────────────────
def overview():
    """Summary KPIs: disease count, file counts, sizes, download status."""
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "data_inventory.yaml missing"}

    summary = cfg.get("summary", {})
    download_status = cfg.get("download_status", {})
    disk = cfg.get("disk_usage", {})

    # Per-disease stats
    disease_rows = []
    total_synthetic_files = 0
    total_real_files = 0
    total_ext_files = 0
    status_dist = {}

    for d in DISEASES:
        info = cfg.get(d, {})
        syn = info.get("synthetic", {})
        real = info.get("real_eeg", {})
        ext = info.get("external_validation", {})
        eeg_ds = info.get("eeg_datasets", {})

        syn_files = syn.get("files", 0) if isinstance(syn.get("files"), int) else 0
        real_files = real.get("files", 0) if isinstance(real.get("files"), int) else 0
        ext_files = ext.get("files", 0) if isinstance(ext.get("files"), int) else 0
        eeg_files = eeg_ds.get("files", 0) if isinstance(eeg_ds.get("files"), int) else 0

        total_synthetic_files += syn_files
        total_real_files += real_files + eeg_files
        total_ext_files += ext_files

        real_status = _status_key(real.get("status", ""))
        status_dist[real_status] = status_dist.get(real_status, 0) + 1

        syn_records = _parse_records(syn.get("records", 0))

        disease_rows.append({
            "disease": d.replace("_", " ").title(),
            "synthetic_files": syn_files,
            "synthetic_records": syn_records,
            "synthetic_size": _parse_size(syn.get("size")),
            "synthetic_status": syn.get("status", "N/A"),
            "real_files": real_files + eeg_files,
            "real_size": _parse_size(real.get("size")),
            "real_source": real.get("source", "N/A"),
            "real_subjects": real.get("subjects", 0) if isinstance(real.get("subjects"), int) else 0,
            "real_status": real.get("status", "N/A"),
            "real_format": real.get("format", "N/A"),
            "ext_files": ext_files,
            "ext_status": ext.get("status", "N/A"),
        })

    # Additional datasets
    additional_rows = []
    for a in ADDITIONAL:
        info = cfg.get(a, {})
        real = info.get("real_eeg", {})
        real_files = real.get("files", 0) if isinstance(real.get("files"), int) else 0
        total_real_files += real_files
        rs = _status_key(real.get("status", ""))
        status_dist[rs] = status_dist.get(rs, 0) + 1
        additional_rows.append({
            "name": a.replace("_", " ").title(),
            "files": real_files,
            "size": _parse_size(real.get("size")),
            "source": real.get("source", "N/A"),
            "subjects": real.get("subjects", 0) if isinstance(real.get("subjects"), int) else 0,
            "format": real.get("format", "N/A"),
            "status": real.get("status", "N/A"),
        })

    # Download status chart
    dl_fully = len(download_status.get("fully_downloaded", []))
    dl_downloading = len(download_status.get("downloading", []))
    dl_failed = len(download_status.get("failed", []))
    dl_not = len(download_status.get("not_downloaded", []))
    dl_reg = len(download_status.get("registration_required", []))

    download_chart = [
        {"name": "Downloaded", "value": dl_fully},
        {"name": "Downloading", "value": dl_downloading},
        {"name": "Failed", "value": dl_failed},
        {"name": "Not Downloaded", "value": dl_not},
        {"name": "Registration Required", "value": dl_reg},
    ]
    download_chart = [x for x in download_chart if x["value"] > 0]

    # Status distribution for pie
    status_chart = [{"name": k.replace("_", " ").title(), "value": v} for k, v in status_dist.items() if v > 0]

    # Real EEG subjects per disease bar chart
    subjects_chart = [
        {"name": r["disease"], "subjects": r["real_subjects"]}
        for r in disease_rows if r["real_subjects"] > 0
    ]

    # Synthetic records bar chart
    synthetic_chart = [
        {"name": r["disease"], "records": r["synthetic_records"]}
        for r in disease_rows
    ]

    return {
        "available": True,
        "title": "Data Inventory & Tracking",
        "note": "Per-disease EEG data inventory — synthetic, real EEG, external validation, download status",
        "updated_at": summary.get("last_updated", "N/A"),
        "kpis": {
            "total_diseases": summary.get("total_diseases", len(DISEASES)),
            "additional_datasets": summary.get("additional_datasets", len(ADDITIONAL)),
            "total_synthetic_records": summary.get("total_synthetic_records", "1,400"),
            "total_real_eeg_files": summary.get("total_real_eeg_files", f"{total_real_files}+"),
            "total_real_eeg_size": summary.get("total_real_eeg_size", "N/A"),
            "total_disk": disk.get("total", "N/A"),
            "downloaded": dl_fully,
            "failed": dl_failed,
        },
        "charts": {
            "download_status": download_chart,
            "real_eeg_status": status_chart,
            "subjects_per_disease": subjects_chart,
            "synthetic_records": synthetic_chart,
        },
        "disease_summary": disease_rows,
        "additional_datasets": additional_rows,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    """Detailed per-disease breakdown: paths, sources, formats, licenses."""
    cfg = _load()
    if not cfg:
        return {"available": False}

    diseases = {}
    for d in DISEASES:
        info = cfg.get(d, {})
        diseases[d] = {
            "synthetic": info.get("synthetic", {}),
            "real_eeg": info.get("real_eeg", {}),
            "external_validation": info.get("external_validation", {}),
            "eeg_datasets": info.get("eeg_datasets", {}),
        }
        # Also include special keys like ppmi, adni
        for k, v in info.items():
            if k not in ("synthetic", "real_eeg", "external_validation", "eeg_datasets") and isinstance(v, dict):
                diseases[d][k] = v

    additional = {}
    for a in ADDITIONAL:
        additional[a] = cfg.get(a, {})

    general = cfg.get("general", {})
    download_status = cfg.get("download_status", {})
    disk_usage = cfg.get("disk_usage", {})

    return {
        "available": True,
        "diseases": diseases,
        "additional": additional,
        "general": general,
        "download_status": download_status,
        "disk_usage": disk_usage,
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    """Glossary, clinical notes, references for data inventory."""
    return {
        "available": True,
        "status_legend": [
            {"status": "COMPLETE", "meaning": "All synthetic data generated and validated"},
            {"status": "DOWNLOADED", "meaning": "Real EEG dataset fully downloaded and accessible"},
            {"status": "PARTIAL", "meaning": "Subset of dataset downloaded (e.g., limited subjects)"},
            {"status": "DOWNLOADING", "meaning": "Download in progress"},
            {"status": "DOWNLOAD FAILED", "meaning": "Download attempted but failed (timeout/404)"},
            {"status": "SIMULATED", "meaning": "External validation data is synthetic/simulated"},
            {"status": "MISSING", "meaning": "Dataset not yet created or downloaded"},
            {"status": "SYMLINKED", "meaning": "Data files symlinked from another location"},
            {"status": "CACHE ONLY", "meaning": "Requires registration/DUA; only cache placeholder exists"},
            {"status": "INDEX ONLY", "meaning": "Index/manifest exists but raw data not stored"},
        ],
        "glossary": [
            {"term": "EDF", "definition": "European Data Format — standard for EEG/PSG time-series files"},
            {"term": "BIDS", "definition": "Brain Imaging Data Structure — standardized neuroimaging directory layout"},
            {"term": "PhysioNet", "definition": "MIT repository of physiological signal databases (open access)"},
            {"term": "OpenNeuro", "definition": "Free platform for sharing neuroimaging datasets in BIDS format"},
            {"term": "DUA", "definition": "Data Use Agreement — legal contract required for restricted datasets"},
            {"term": "Augmented", "definition": "Synthetic samples generated via SMOTE/noise injection to increase training size"},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston EEG dataset for seizure detection (23 subjects)"},
            {"term": "PPMI", "definition": "Parkinson's Progression Markers Initiative — longitudinal clinical study"},
            {"term": "ADNI", "definition": "Alzheimer's Disease Neuroimaging Initiative — multi-site biomarker study"},
            {"term": "Figshare", "definition": "Open-access repository for research datasets and supplementary materials"},
            {"term": "CC0 / CC-BY", "definition": "Creative Commons licenses — CC0 is public domain, CC-BY requires attribution"},
            {"term": "ODC-BY", "definition": "Open Data Commons Attribution License — open data with attribution requirement"},
        ],
        "clinical_notes": [
            "Real EEG datasets range from 10 MB to 7.6 GB per disease; total storage ~10.9 GB",
            "Synthetic data augments small real datasets from 50-100 original to 200 samples each",
            "External validation datasets are simulated for diseases where no public benchmark exists",
            "Registration-gated datasets (PPMI, ADNI, DEAP) require institutional DUA approval",
        ],
        "references": [
            {"name": "data_inventory.yaml", "path": "config/data_inventory.yaml", "role": "Source config for this dashboard"},
            {"name": "PhysioNet", "note": "physionet.org — CHB-MIT, EEGMAT, Sleep-EDF, Siena datasets"},
            {"name": "OpenNeuro", "note": "openneuro.org — Parkinson ds002778, Alzheimer ds004504, Autism ds004141"},
            {"name": "Figshare", "note": "figshare.com — Mumtaz MDD EEG dataset (article 4244171)"},
        ],
    }
