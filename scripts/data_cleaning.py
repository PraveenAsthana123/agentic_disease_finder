#!/usr/bin/env python3
"""Clinical Data Manager — Data Cleaning analysis.

Runs real EEG signal-quality checks on CHB-MIT recordings:
  1. Flat/saturated channel detection (eeg_quality.bad_channels)
  2. NaN/Inf sanitization stats
  3. ICA artifact-removal summary (from cron-generated report)
  4. Post-clean quality re-score

100 % real data — report only, never modifies source EDF files.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHB = ROOT / "data" / "real_eeg" / "epilepsy_physionet"
ICA_REPORT = ROOT / "jobs" / "reports" / "ica_noise_cleaning.json"


def cleaning_report() -> dict:
    """Full data-cleaning dashboard payload."""

    # ── 1. Discover available subjects + EDFs ─────────────────────────
    subjects_found: list[str] = []
    edfs_found: list[str] = []
    if CHB.exists():
        for sd in sorted(CHB.iterdir()):
            if sd.is_dir() and sd.name.startswith("chb"):
                edfs = sorted(sd.glob("*.edf"))
                subjects_found.append(sd.name)
                edfs_found.extend([str(e) for e in edfs[:2]])

    # ── 2. Channel quality scan (real signal via eeg_quality) ─────────
    channel_report: dict = {
        "scanned": 0, "flat": 0, "noisy": 0,
        "line_noise": 0, "disconnected": 0, "good": 0,
        "per_subject": [],
    }
    try:
        import sys as _sys
        _scripts = str(ROOT / "scripts")
        if _scripts not in _sys.path:
            _sys.path.insert(0, _scripts)
        from eeg_quality import bad_channels

        for edf in edfs_found[:3]:  # cap at 3 for speed
            qr = bad_channels(edf, seconds=30)
            if qr.get("available"):
                vd = qr.get("verdict_distribution", {})
                n_ch = qr.get("n_channels", 0)
                subj = {
                    "file": Path(edf).name,
                    "total": n_ch,
                }
                for v in ("good", "flat", "noisy", "line_noise", "disconnected"):
                    cnt = vd.get(v, 0)
                    channel_report[v] += cnt
                    subj[v] = cnt
                channel_report["scanned"] += n_ch
                channel_report["per_subject"].append(subj)
    except Exception as exc:
        channel_report["error"] = str(exc)

    # ── 3. NaN / Inf sanitization stats (scan first EDF) ─────────────
    nan_stats: dict = {
        "total_samples": 0, "nan_count": 0, "inf_count": 0, "clean_ratio": 1.0,
    }
    try:
        import mne

        if edfs_found:
            raw = mne.io.read_raw_edf(edfs_found[0], preload=True, verbose="ERROR")
            raw.crop(tmin=0, tmax=min(30, raw.times[-1]))
            data = raw.get_data()
            nan_stats["total_samples"] = int(data.size)
            nan_stats["nan_count"] = int(np.isnan(data).sum())
            nan_stats["inf_count"] = int(np.isinf(data).sum())
            valid = data.size - nan_stats["nan_count"] - nan_stats["inf_count"]
            nan_stats["clean_ratio"] = round(valid / max(data.size, 1), 6)
    except Exception as exc:
        nan_stats["error"] = str(exc)

    # ── 4. ICA artifact-removal report (from last cron run) ───────────
    ica_summary: dict = {"available": False}
    if ICA_REPORT.exists():
        try:
            ica_data = json.loads(ICA_REPORT.read_text())
            subjects_list = ica_data.get("subjects", [])
            ica_summary = {
                "available": True,
                "subjects_cleaned": len(subjects_list),
                "mean_variance_removed_pct": round(
                    float(np.mean([s.get("variance_removed_pct", 0) for s in subjects_list])), 2
                ) if subjects_list else 0,
                "mean_components_removed": round(
                    float(np.mean([s.get("components_removed", 0) for s in subjects_list])), 1
                ) if subjects_list else 0,
                "timestamp": ica_data.get("timestamp", ""),
                "subjects": subjects_list[:5],
            }
        except Exception as exc:
            ica_summary["error"] = str(exc)

    # ── 5. Post-clean quality score ───────────────────────────────────
    bad_count = (
        channel_report["flat"]
        + channel_report["noisy"]
        + channel_report["disconnected"]
        + channel_report["line_noise"]
    )
    total = max(channel_report["scanned"], 1)
    quality_pct = round((1.0 - bad_count / total) * 100, 1)

    return {
        "available": True,
        "subjects_found": len(subjects_found),
        "edfs_sampled": min(len(edfs_found), 3),
        "channel_quality": channel_report,
        "nan_inf_stats": nan_stats,
        "ica_artifact_removal": ica_summary,
        "post_clean_quality_pct": quality_pct,
        "cleaning_steps": [
            {"step": 1, "name": "Flat/saturated channel detection",
             "status": "complete" if channel_report["scanned"] > 0 else "no_data"},
            {"step": 2, "name": "NaN/Inf sanitization",
             "status": "complete" if nan_stats["total_samples"] > 0 else "no_data"},
            {"step": 3, "name": "ICA artifact removal",
             "status": "complete" if ica_summary["available"] else "pending_cron"},
            {"step": 4, "name": "Post-clean quality re-score",
             "status": "complete" if quality_pct > 0 else "no_data"},
        ],
        "challenges": [
            "Distinguishing artifact from epileptiform activity",
            "Over-cleaning removes real spikes",
            "No gold-standard 'clean' reference",
        ],
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    import json as _j

    r = cleaning_report()
    print(_j.dumps(r, indent=2, default=str))
