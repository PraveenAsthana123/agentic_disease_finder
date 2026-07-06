"""
EEG Noise Cleaning / ICA Artifact Removal Dashboard — reads real ICA
noise-cleaning results from jobs/reports/ica_noise_cleaning.json.

Provides per-file and per-subject artifact removal statistics,
variance distribution, and methodology definitions.
"""

import json, pathlib
from collections import defaultdict

BASE = pathlib.Path(__file__).resolve().parent.parent
ICA_REPORT = BASE / "jobs" / "reports" / "ica_noise_cleaning.json"


def _load_data():
    if not ICA_REPORT.exists():
        return {}
    try:
        return json.loads(ICA_REPORT.read_text())
    except Exception:
        return {}


def _bucket_label(pct):
    if pct < 20:
        return "0-20%"
    elif pct < 40:
        return "20-40%"
    elif pct < 60:
        return "40-60%"
    elif pct < 80:
        return "60-80%"
    else:
        return "80-100%"


# ── overview ─────────────────────────────────────────────────────
def overview():
    data = _load_data()
    if not data:
        return {
            "available": False,
            "kpis": [],
            "per_subject_summary": [],
            "variance_distribution": [],
            "timeline": [],
            "generated_at": "",
            "method": "",
            "note": "",
        }

    per_file = data.get("per_file", [])
    method = data.get("method", "")
    note = data.get("note", "")
    generated_at = data.get("generated_at", "")
    mean_var = data.get("mean_variance_removed_pct", 0.0)

    total_files = len(per_file)
    total_artifacts = sum(f.get("artifact_components_removed", 0) for f in per_file)

    kpis = [
        {"label": "Total Files Processed", "value": total_files, "unit": "files"},
        {"label": "Mean Variance Removed", "value": mean_var, "unit": "%"},
        {"label": "Total Artifact Components Removed", "value": total_artifacts, "unit": "components"},
        {"label": "Method", "value": method, "unit": ""},
    ]

    # per-subject summary
    subj = defaultdict(lambda: {"files": 0, "var_sum": 0.0, "artifacts": 0})
    for f in per_file:
        s = f.get("subject", "unknown")
        subj[s]["files"] += 1
        subj[s]["var_sum"] += f.get("variance_removed_pct", 0.0)
        subj[s]["artifacts"] += f.get("artifact_components_removed", 0)
    per_subject_summary = sorted(
        [
            {
                "subject": s,
                "files_count": v["files"],
                "avg_variance_removed": round(v["var_sum"] / v["files"], 2) if v["files"] else 0.0,
                "total_artifacts_removed": v["artifacts"],
            }
            for s, v in subj.items()
        ],
        key=lambda x: x["subject"],
    )

    # variance distribution buckets
    buckets = {"0-20%": 0, "20-40%": 0, "40-60%": 0, "60-80%": 0, "80-100%": 0}
    for f in per_file:
        buckets[_bucket_label(f.get("variance_removed_pct", 0.0))] += 1
    variance_distribution = [{"range": k, "count": v} for k, v in buckets.items()]

    return {
        "available": True,
        "kpis": kpis,
        "per_subject_summary": per_subject_summary,
        "variance_distribution": variance_distribution,
        "timeline": per_file,
        "generated_at": generated_at,
        "method": method,
        "note": note,
    }


# ── breakdown ────────────────────────────────────────────────────
def breakdown():
    data = _load_data()
    if not data:
        return {
            "available": False,
            "per_file_details": [],
            "channel_stats": {"avg_channels": 0, "min_channels": 0, "max_channels": 0},
            "component_stats": {"avg_components": 0, "avg_artifacts": 0, "artifact_ratio_pct": 0.0},
            "subject_comparison": [],
            "quality_tiers": [],
        }

    per_file = data.get("per_file", [])
    n = len(per_file) or 1

    # channel stats
    channels = [f.get("n_channels", 0) for f in per_file]
    channel_stats = {
        "avg_channels": round(sum(channels) / n, 1),
        "min_channels": min(channels) if channels else 0,
        "max_channels": max(channels) if channels else 0,
    }

    # component stats
    components = [f.get("n_components", 0) for f in per_file]
    artifacts = [f.get("artifact_components_removed", 0) for f in per_file]
    avg_comp = sum(components) / n
    avg_art = sum(artifacts) / n
    component_stats = {
        "avg_components": round(avg_comp, 1),
        "avg_artifacts": round(avg_art, 1),
        "artifact_ratio_pct": round(avg_art / avg_comp * 100, 2) if avg_comp else 0.0,
    }

    # subject comparison (aggregated)
    subj = defaultdict(lambda: {"var_sum": 0.0, "art_sum": 0, "count": 0})
    for f in per_file:
        s = f.get("subject", "unknown")
        subj[s]["var_sum"] += f.get("variance_removed_pct", 0.0)
        subj[s]["art_sum"] += f.get("artifact_components_removed", 0)
        subj[s]["count"] += 1
    subject_comparison = sorted(
        [
            {
                "subject": s,
                "variance_removed_pct": round(v["var_sum"] / v["count"], 2) if v["count"] else 0.0,
                "artifact_components": v["art_sum"],
                "n_files": v["count"],
            }
            for s, v in subj.items()
        ],
        key=lambda x: x["subject"],
    )

    # quality tiers
    tier_defs = [
        ("Minimal", "0-20%", 0, 20),
        ("Moderate", "20-40%", 20, 40),
        ("Significant", "40-60%", 40, 60),
        ("Heavy", "60-80%", 60, 80),
        ("Extreme", "80-100%", 80, 100),
    ]
    tier_counts = {t[0]: 0 for t in tier_defs}
    for f in per_file:
        v = f.get("variance_removed_pct", 0.0)
        for name, _, lo, hi in tier_defs:
            if lo <= v < hi or (hi == 100 and v == 100):
                tier_counts[name] += 1
                break
    quality_tiers = [
        {"tier": name, "count": tier_counts[name], "description": f"{rng} variance removed"}
        for name, rng, _, _ in tier_defs
    ]

    return {
        "available": True,
        "per_file_details": per_file,
        "channel_stats": channel_stats,
        "component_stats": component_stats,
        "subject_comparison": subject_comparison,
        "quality_tiers": quality_tiers,
    }


# ── definitions ──────────────────────────────────────────────────
def definitions():
    return {
        "available": True,
        "metrics": [
            {"name": "ICA (Independent Component Analysis)", "description": "A blind source separation technique that decomposes multi-channel EEG into statistically independent components, allowing identification and removal of artifact sources (eye blinks, muscle activity, line noise) without discarding entire channels.", "unit": ""},
            {"name": "Variance Removed (%)", "description": "Percentage of total signal variance attributed to the rejected artifact components. Higher values indicate more aggressive cleaning; typical clinical range is 20-50%.", "unit": "%"},
            {"name": "Artifact Components Removed", "description": "Number of ICA components classified as artifacts and excluded from the reconstructed signal. Components are flagged by kurtosis and variance thresholds.", "unit": "components"},
            {"name": "Number of Components", "description": "Total ICA components extracted per recording. Set to min(n_channels - 1, 15) to balance decomposition quality and computation.", "unit": "components"},
            {"name": "Number of Channels", "description": "EEG electrode count in the recording. CHB-MIT uses a standard 23-channel bipolar montage.", "unit": "channels"},
            {"name": "Band-Pass Filter (1-45 Hz)", "description": "Pre-ICA filtering that removes DC drift (<1 Hz) and high-frequency EMG/electronic noise (>45 Hz), retaining clinically relevant EEG bands (delta through gamma).", "unit": "Hz"},
            {"name": "Notch Filter (60 Hz)", "description": "Removes power-line interference at 60 Hz (North American mains frequency). Applied before ICA to prevent line noise from dominating a component.", "unit": "Hz"},
            {"name": "Kurtosis Threshold", "description": "Components with excess kurtosis above the threshold are flagged as artifacts. High-kurtosis components typically capture eye blinks and sharp transients.", "unit": ""},
            {"name": "Quality Tier", "description": "Classification of cleaning intensity: Minimal (0-20%), Moderate (20-40%), Significant (40-60%), Heavy (60-80%), Extreme (80-100%) based on variance removed percentage.", "unit": ""},
        ],
        "methodology": (
            "Pipeline: (1) Load raw EDF via MNE-Python. "
            "(2) Apply 1-45 Hz band-pass FIR filter. "
            "(3) Apply 60 Hz notch filter. "
            "(4) Fit ICA (FastICA, max 15 components). "
            "(5) Auto-detect artifact components via kurtosis (>2.0) and variance thresholds. "
            "(6) Remove flagged components and reconstruct cleaned EEG. "
            "Note: No EOG/EMG reference channels available in CHB-MIT, so detection is fully automated. "
            "Clinical deployment would add manual component review and EOG correlation."
        ),
        "quality_notes": [
            "Auto-detection without EOG reference may miss subtle ocular artifacts or misclassify cerebral components as noise.",
            "Variance removed >60% warrants manual review — aggressive removal risks discarding epileptiform activity.",
            "CHB-MIT recordings use bipolar montage; results may not generalize to average-reference or Laplacian montages.",
            "ICA assumes stationary mixing — long recordings with posture/impedance changes may violate this assumption.",
            "Clinical use requires board-certified EEG technologist review of excluded components before downstream analysis.",
        ],
    }


if __name__ == "__main__":
    import pprint
    print("=== OVERVIEW ===")
    pprint.pprint(overview())
    print("\n=== BREAKDOWN ===")
    pprint.pprint(breakdown())
    print("\n=== DEFINITIONS ===")
    pprint.pprint(definitions())
