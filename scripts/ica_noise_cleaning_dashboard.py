"""ICA Noise Cleaning Dashboard — Independent Component Analysis artifact removal analytics.

Reads REAL ICA noise cleaning results from jobs/reports/ica_noise_cleaning.json (generated
by the MNE ICA pipeline on CHB-MIT EEG data) and also queries clinical.db analyses table
for per-patient EEG analysis context.

Pipeline stages:
  1. Load raw EEG (EDF/CSV)
  2. Band-pass filter (1-45 Hz) + notch (60 Hz)
  3. ICA decomposition (FastICA, n_components from data rank)
  4. Artifact component detection (kurtosis + variance thresholds)
  5. Reconstruct cleaned signal (remove artifact components)

Reference:
  Delorme A, Makeig S. EEGLAB: an open source toolbox for analysis of
  single-trial EEG dynamics including independent component analysis.
  J Neurosci Methods. 2004;134(1):9-21.

Author: Research Team
"""
import json
import sqlite3
from pathlib import Path
from collections import Counter, defaultdict

REPORT = Path(__file__).resolve().parent.parent / "jobs" / "reports" / "ica_noise_cleaning.json"
DB = Path(__file__).resolve().parent.parent / "data" / "clinical.db"


def _load_report():
    if REPORT.exists():
        with open(REPORT) as f:
            return json.load(f)
    return None


def _conn():
    c = sqlite3.connect(str(DB))
    c.row_factory = sqlite3.Row
    return c


def _rows(query, params=()):
    with _conn() as c:
        return [dict(r) for r in c.execute(query, params).fetchall()]


def overview():
    """Summary KPIs + per-file ICA results + subject-level aggregates."""
    report = _load_report()
    if not report:
        return {"total_files": 0, "message": "No ICA noise cleaning data yet. Run the ICA pipeline first."}

    per_file = report.get("per_file", [])
    n_files = len(per_file)

    if n_files == 0:
        return {"total_files": 0, "message": "ICA report has no per-file results."}

    # KPIs
    total_components_removed = sum(f["artifact_components_removed"] for f in per_file)
    total_components = sum(f["n_components"] for f in per_file)
    avg_variance_removed = report.get("mean_variance_removed_pct", 0)
    avg_components_per_file = round(total_components_removed / n_files, 1)
    removal_rate = round(total_components_removed / total_components * 100, 1) if total_components else 0

    # Per-subject aggregate
    subject_data = defaultdict(list)
    for f in per_file:
        subject_data[f["subject"]].append(f)

    subject_summary = []
    for subj, files in sorted(subject_data.items()):
        avg_var = round(sum(f["variance_removed_pct"] for f in files) / len(files), 1)
        total_art = sum(f["artifact_components_removed"] for f in files)
        subject_summary.append({
            "subject": subj,
            "n_files": len(files),
            "total_artifact_components": total_art,
            "avg_variance_removed_pct": avg_var,
            "n_channels": files[0]["n_channels"],
            "quality": "good" if avg_var < 40 else ("moderate" if avg_var < 60 else "heavy_artifact"),
        })

    # Quality distribution
    quality_dist = Counter(s["quality"] for s in subject_summary)

    # Variance distribution buckets for chart
    var_buckets = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
    for f in per_file:
        v = f["variance_removed_pct"]
        if v < 20:
            var_buckets["0-20%"] += 1
        elif v < 40:
            var_buckets["20-40%"] += 1
        elif v < 60:
            var_buckets["40-60%"] += 1
        elif v < 80:
            var_buckets["60-80%"] += 1
        else:
            var_buckets["80-100%"] += 1

    variance_distribution = [{"range": k, "count": v} for k, v in var_buckets.items()]

    # Get analyses count from DB for context
    analyses = _rows("SELECT COUNT(*) as cnt FROM analyses")
    db_analyses_count = analyses[0]["cnt"] if analyses else 0

    return {
        "generated_at": report.get("generated_at", ""),
        "method": report.get("method", ""),
        "note": report.get("note", ""),
        "total_files_processed": n_files,
        "total_subjects": len(subject_data),
        "total_components_fitted": total_components,
        "total_artifact_components_removed": total_components_removed,
        "avg_artifact_components_per_file": avg_components_per_file,
        "artifact_removal_rate_pct": removal_rate,
        "mean_variance_removed_pct": avg_variance_removed,
        "quality_distribution": dict(quality_dist),
        "variance_distribution": variance_distribution,
        "subject_summary": subject_summary,
        "per_file": per_file,
        "db_analyses_count": db_analyses_count,
    }


def breakdown():
    """Per-file detail, component analysis, channel coverage."""
    report = _load_report()
    if not report:
        return {"message": "No ICA data"}

    per_file = report.get("per_file", [])

    # Per-file detail with derived metrics
    file_details = []
    for f in per_file:
        retained = f["n_components"] - f["artifact_components_removed"]
        retention_pct = round(retained / f["n_components"] * 100, 1) if f["n_components"] else 0
        signal_retained_pct = round(100 - f["variance_removed_pct"], 1)
        file_details.append({
            "file": f["file"],
            "subject": f["subject"],
            "n_channels": f["n_channels"],
            "n_components": f["n_components"],
            "artifact_components": f["artifact_components_removed"],
            "retained_components": retained,
            "retention_pct": retention_pct,
            "variance_removed_pct": f["variance_removed_pct"],
            "signal_retained_pct": signal_retained_pct,
        })

    # Component analysis summary
    all_components = [f["n_components"] for f in per_file]
    all_artifact = [f["artifact_components_removed"] for f in per_file]
    component_stats = {
        "min_components": min(all_components) if all_components else 0,
        "max_components": max(all_components) if all_components else 0,
        "avg_components": round(sum(all_components) / len(all_components), 1) if all_components else 0,
        "min_artifacts": min(all_artifact) if all_artifact else 0,
        "max_artifacts": max(all_artifact) if all_artifact else 0,
        "avg_artifacts": round(sum(all_artifact) / len(all_artifact), 1) if all_artifact else 0,
    }

    # Channel coverage: which channel counts appear
    channel_counts = Counter(f["n_channels"] for f in per_file)
    channel_coverage = [{"n_channels": k, "file_count": v} for k, v in sorted(channel_counts.items())]

    # Pipeline stages with status
    stages = [
        {"step": 1, "name": "Load Raw EEG", "description": "Read EDF/CSV files, extract channel data and sampling rate", "status": "complete"},
        {"step": 2, "name": "Band-pass Filter", "description": "1-45 Hz band-pass to remove DC offset and high-frequency noise", "status": "complete"},
        {"step": 3, "name": "Notch Filter", "description": "60 Hz notch to remove power line interference", "status": "complete"},
        {"step": 4, "name": "ICA Decomposition", "description": "FastICA decomposition into independent components", "status": "complete"},
        {"step": 5, "name": "Artifact Detection", "description": "Kurtosis + variance thresholding to identify artifact components (eye blinks, muscle, line noise)", "status": "complete"},
        {"step": 6, "name": "Signal Reconstruction", "description": "Remove artifact components and reconstruct cleaned signal", "status": "complete"},
    ]

    return {
        "file_details": file_details,
        "component_stats": component_stats,
        "channel_coverage": channel_coverage,
        "pipeline_stages": stages,
    }


def definitions():
    """Metric definitions for the ICA Noise Cleaning dashboard."""
    return {
        "title": "ICA Noise Cleaning Pipeline — Metric Definitions",
        "definitions": [
            {"term": "ICA", "definition": "Independent Component Analysis — a blind source separation technique that decomposes multi-channel EEG into statistically independent components. Each component represents a distinct source (neural, ocular, muscular, etc.)."},
            {"term": "FastICA", "definition": "A fast fixed-point algorithm for ICA. Uses negentropy as the contrast function. Converges faster than gradient-based ICA for EEG data (Hyvärinen & Oja, 2000)."},
            {"term": "Artifact Component", "definition": "An ICA component identified as non-neural (eye blink, muscle, line noise, channel noise). Detected via high kurtosis (>5) and/or high variance relative to median."},
            {"term": "Variance Removed %", "definition": "Percentage of total signal variance accounted for by removed artifact components. Lower is better (less data lost). Typical range: 15-50%. Above 60% suggests heavy contamination or aggressive removal."},
            {"term": "Signal Retained %", "definition": "100% minus variance removed. Represents the proportion of original signal preserved after cleaning. Higher is better."},
            {"term": "Components Fitted", "definition": "Number of ICA components estimated. Set by data rank (usually n_channels - 1 for average-referenced data, or lower if rank-deficient)."},
            {"term": "Retention Rate", "definition": "Percentage of ICA components classified as neural (kept). Components removed / total components."},
            {"term": "Band-pass Filter", "definition": "1-45 Hz. Removes DC drift (<1 Hz) and high-frequency EMG/noise (>45 Hz) before ICA, improving decomposition quality."},
            {"term": "Notch Filter", "definition": "60 Hz (North American power line). Removes electrical interference that ICA may fail to separate cleanly."},
            {"term": "Kurtosis", "definition": "Fourth statistical moment measuring 'tailedness.' Artifact components (eye blinks) have high kurtosis (>5) due to sparse, large-amplitude events."},
            {"term": "Quality Rating", "definition": "Good: <40% variance removed (clean data). Moderate: 40-60% (some artifact). Heavy Artifact: >60% (heavily contaminated, review recommended)."},
            {"term": "CHB-MIT", "definition": "Children's Hospital Boston EEG dataset (Shoeb & Guttag, 2010). 23-channel scalp EEG from 23 pediatric epilepsy patients. Used here for ICA validation."},
        ],
    }
