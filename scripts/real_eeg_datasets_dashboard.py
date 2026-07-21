"""Real EEG Datasets Dashboard — per-disease real EEG dataset registry
from config/real_eeg_datasets.yaml.
7 primary diseases + 6 additional datasets, subjects/files/sizes/formats/sources/licenses/status."""

import yaml
from pathlib import Path

_CFG = Path(__file__).resolve().parent.parent / "config" / "real_eeg_datasets.yaml"

PRIMARY = ["epilepsy", "parkinson", "alzheimer", "autism", "depression", "schizophrenia", "stress"]
ADDITIONAL = ["depression_figshare", "schizophrenia_msu", "uci_eye_state",
              "motor_imagery", "siena_epilepsy", "sleep_edf"]


def _load():
    if not _CFG.exists():
        return None
    with open(_CFG) as f:
        return yaml.safe_load(f)


def _parse_int(val):
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        import re
        m = re.match(r"(\d[\d,]*)", val.replace(",", ""))
        return int(m.group(1)) if m else 0
    return 0


def _status_cat(s):
    if not s:
        return "unknown"
    s = str(s).lower()
    if s in ("downloaded", "complete"):
        return "downloaded"
    if s == "partial":
        return "partial"
    if s == "downloading":
        return "downloading"
    if s == "symlinked":
        return "symlinked"
    if "fail" in s:
        return "failed"
    return "other"


# ── overview ────────────────────────────────────────────────────────────
def overview():
    cfg = _load()
    if not cfg:
        return {"available": False, "note": "real_eeg_datasets.yaml missing"}

    primary_data = cfg.get("real_eeg_datasets", {})
    additional_data = cfg.get("additional", {})
    summary = cfg.get("summary", {})

    # Primary disease rows
    primary_rows = []
    total_subjects = 0
    total_files = 0
    status_dist = {}

    for d in PRIMARY:
        info = primary_data.get(d, {})
        subj = _parse_int(info.get("subjects", 0))
        files = _parse_int(info.get("files", 0))
        total_subjects += subj
        total_files += files
        st = _status_cat(info.get("status", ""))
        status_dist[st] = status_dist.get(st, 0) + 1
        primary_rows.append({
            "disease": info.get("name", d.replace("_", " ").title()),
            "disease_key": d,
            "subjects": subj,
            "files": files,
            "size": str(info.get("size", "N/A")),
            "format": info.get("format", "N/A"),
            "source": info.get("source", "N/A"),
            "license": info.get("license", "N/A"),
            "status": info.get("status", "N/A"),
        })

    # Additional rows
    additional_rows = []
    for a in ADDITIONAL:
        info = additional_data.get(a, {})
        if not info:
            continue
        subj = _parse_int(info.get("subjects", 0))
        files = _parse_int(info.get("files", 0))
        records = _parse_int(info.get("records", 0))
        total_subjects += subj
        total_files += files
        st = _status_cat(info.get("status", ""))
        status_dist[st] = status_dist.get(st, 0) + 1
        additional_rows.append({
            "name": info.get("name", a.replace("_", " ").title()),
            "key": a,
            "disease": info.get("disease", "N/A"),
            "subjects": subj,
            "files": files,
            "records": records,
            "size": str(info.get("size", "N/A")),
            "format": info.get("format", "N/A"),
            "source": info.get("source", "N/A"),
            "license": info.get("license", "N/A"),
            "status": info.get("status", "N/A"),
            "notes": info.get("notes", ""),
        })

    # Charts
    status_chart = [{"name": k.replace("_", " ").title(), "value": v}
                    for k, v in status_dist.items() if v > 0]

    subjects_chart = [{"name": r["disease_key"].title(), "subjects": r["subjects"]}
                      for r in primary_rows if r["subjects"] > 0]

    files_chart = [{"name": r["disease_key"].title(), "files": r["files"]}
                   for r in primary_rows if r["files"] > 0]

    # Format distribution
    format_dist = {}
    for r in primary_rows + additional_rows:
        fmt = r.get("format", "N/A")
        format_dist[fmt] = format_dist.get(fmt, 0) + 1
    format_chart = [{"name": k, "value": v} for k, v in format_dist.items()]

    # Source distribution
    source_dist = {}
    for r in primary_rows + additional_rows:
        src = r.get("source", "N/A")
        source_dist[src] = source_dist.get(src, 0) + 1
    source_chart = [{"name": k, "value": v} for k, v in source_dist.items()]

    return {
        "available": True,
        "title": "Real EEG Datasets Registry",
        "note": "Per-disease real EEG dataset inventory — subjects, files, sizes, formats, sources, licenses",
        "updated_at": summary.get("last_updated", "N/A"),
        "kpis": {
            "primary_diseases": len(PRIMARY),
            "additional_datasets": len(additional_rows),
            "total_datasets": len(PRIMARY) + len(additional_rows),
            "total_subjects": total_subjects,
            "total_files": summary.get("total_files", total_files),
            "total_size": summary.get("total_size", "~30 GB"),
            "downloaded": status_dist.get("downloaded", 0),
            "partial": status_dist.get("partial", 0),
            "downloading": status_dist.get("downloading", 0),
        },
        "charts": {
            "status_distribution": status_chart,
            "subjects_per_disease": subjects_chart,
            "files_per_disease": files_chart,
            "format_distribution": format_chart,
            "source_distribution": source_chart,
        },
        "primary_datasets": primary_rows,
        "additional_datasets": additional_rows,
    }


# ── breakdown ───────────────────────────────────────────────────────────
def breakdown():
    cfg = _load()
    if not cfg:
        return {"available": False}

    primary_data = cfg.get("real_eeg_datasets", {})
    additional_data = cfg.get("additional", {})

    primary = {}
    for d in PRIMARY:
        info = primary_data.get(d, {})
        primary[d] = {
            "name": info.get("name", d.replace("_", " ").title()),
            "disease": info.get("disease", d),
            "path": info.get("path", "N/A"),
            "format": info.get("format", "N/A"),
            "subjects": info.get("subjects", 0),
            "files": info.get("files", 0),
            "size": str(info.get("size", "N/A")),
            "source": info.get("source", "N/A"),
            "url": info.get("url", ""),
            "license": info.get("license", "N/A"),
            "status": info.get("status", "N/A"),
            "notes": info.get("notes", ""),
            "symlink_target": info.get("symlink_target", ""),
        }

    additional = {}
    for a in ADDITIONAL:
        info = additional_data.get(a, {})
        if not info:
            continue
        additional[a] = {
            "name": info.get("name", a.replace("_", " ").title()),
            "disease": info.get("disease", "N/A"),
            "path": info.get("path", "N/A"),
            "format": info.get("format", "N/A"),
            "subjects": info.get("subjects", 0),
            "files": info.get("files", 0),
            "records": info.get("records", 0),
            "size": str(info.get("size", "N/A")),
            "source": info.get("source", "N/A"),
            "url": info.get("url", ""),
            "license": info.get("license", "N/A"),
            "status": info.get("status", "N/A"),
            "notes": info.get("notes", ""),
        }

    return {
        "available": True,
        "primary": primary,
        "additional": additional,
        "summary": cfg.get("summary", {}),
    }


# ── definitions ─────────────────────────────────────────────────────────
def definitions():
    return {
        "available": True,
        "status_legend": [
            {"status": "downloaded", "meaning": "Full dataset downloaded and accessible locally"},
            {"status": "partial", "meaning": "Only a subset of subjects downloaded"},
            {"status": "downloading", "meaning": "Download in progress"},
            {"status": "symlinked", "meaning": "Data files symlinked from another disk/location"},
            {"status": "failed", "meaning": "Download attempted but failed"},
        ],
        "glossary": [
            {"term": "EDF", "definition": "European Data Format — standard for EEG/PSG time-series files"},
            {"term": "BDF", "definition": "BioSemi Data Format — 24-bit variant of EDF for high-resolution EEG"},
            {"term": "BIDS", "definition": "Brain Imaging Data Structure — standardized neuroimaging directory layout"},
            {"term": "SET/FDT", "definition": "EEGLAB file format — .set header + .fdt float data"},
            {"term": "ARFF", "definition": "Attribute-Relation File Format — Weka machine learning format"},
            {"term": "PhysioNet", "definition": "MIT repository of physiological signal databases (open access)"},
            {"term": "OpenNeuro", "definition": "Free platform for sharing neuroimaging datasets in BIDS format"},
            {"term": "Figshare", "definition": "Open-access repository for research datasets"},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston EEG dataset for seizure detection"},
            {"term": "EEGMAT", "definition": "EEG during Mental Arithmetic Tasks — stress dataset from PhysioNet"},
            {"term": "CC0", "definition": "Creative Commons Zero — public domain dedication, no restrictions"},
            {"term": "CC-BY", "definition": "Creative Commons Attribution — requires attribution"},
            {"term": "ODC-BY", "definition": "Open Data Commons Attribution License — open data with attribution"},
        ],
        "clinical_notes": [
            "Primary datasets cover 7 neurological/psychiatric conditions with real EEG recordings",
            "Total storage across all datasets is approximately 30 GB",
            "CHB-MIT (epilepsy) is partial — 4 of 23 subjects; full dataset available on PhysioNet",
            "Stress dataset (EEGMAT) is symlinked from an external drive to save disk space",
        ],
        "references": [
            {"name": "real_eeg_datasets.yaml", "path": "config/real_eeg_datasets.yaml",
             "role": "Source config for this dashboard"},
            {"name": "PhysioNet", "note": "physionet.org — CHB-MIT, EEGMAT, Sleep-EDF, Siena, Motor Imagery"},
            {"name": "OpenNeuro", "note": "openneuro.org — Parkinson ds002778, Alzheimer ds004504, Autism ds004141, Depression ds003478, Schizophrenia ds004215"},
            {"name": "Figshare", "note": "figshare.com — Mumtaz MDD EEG dataset (article 4244171)"},
        ],
    }
