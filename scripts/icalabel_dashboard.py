#!/usr/bin/env python3
"""ICLabel ICA Component Classification Dashboard.

Uses mne-icalabel to automatically classify ICA components from real
CHB-MIT EEG recordings into 7 categories:
  brain, eye, heart, muscle, line_noise, channel_noise, other

100% real data — reads EDF files, runs ICA, applies ICLabel neural-net
classifier, returns per-component class probabilities.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CHB = ROOT / "data" / "real_eeg" / "epilepsy_physionet"

# ICLabel class names (fixed order from mne-icalabel)
IC_CLASSES = ["brain", "muscle", "eye", "heart", "line_noise", "channel_noise", "other"]


def _find_edfs(max_subjects: int = 3, max_per_subject: int = 1) -> list[Path]:
    """Find real EDF files from CHB-MIT dataset."""
    edfs: list[Path] = []
    if not CHB.exists():
        return edfs
    for sd in sorted(CHB.iterdir()):
        if sd.is_dir() and sd.name.startswith("chb"):
            for edf in sorted(sd.glob("*.edf"))[:max_per_subject]:
                edfs.append(edf)
            if len(edfs) >= max_subjects:
                break
    return edfs


def _bipolar_to_unipolar(raw) -> "mne.io.Raw":
    """Convert CHB-MIT bipolar channels (FP1-F7, F7-T7, ...) into unipolar
    channels named by the first electrode of each pair, then set a standard
    10-20 montage.  ICLabel needs electrode positions — bipolar names have none."""
    import mne

    montage = mne.channels.make_standard_montage("standard_1020")
    montage_lower = {n.lower(): n for n in montage.ch_names}

    rename = {}
    drop = []
    seen = set()

    for ch in raw.ch_names:
        # CHB-MIT pattern: "FP1-F7", "F7-T7", "FZ-CZ", "T8-P8-0" (dup suffix)
        parts = ch.upper().replace(" ", "").split("-")
        first = parts[0] if parts else ch

        # Map common CHB-MIT names to standard 10-20
        electrode_map = {
            "FP1": "Fp1", "FP2": "Fp2", "F7": "F7", "F8": "F8",
            "F3": "F3", "F4": "F4", "FZ": "Fz", "CZ": "Cz", "PZ": "Pz",
            "C3": "C3", "C4": "C4", "T7": "T7", "T8": "T8",
            "P3": "P3", "P4": "P4", "P7": "P7", "P8": "P8",
            "O1": "O1", "O2": "O2", "FT9": "FT9", "FT10": "FT10",
        }
        std_name = electrode_map.get(first)
        if std_name and std_name.lower() in montage_lower and std_name not in seen:
            rename[ch] = std_name
            seen.add(std_name)
        else:
            drop.append(ch)

    if len(rename) < 5:
        return None  # not enough channels to map

    if drop:
        raw.drop_channels(drop)
    raw.rename_channels(rename)
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage(montage, on_missing="ignore")
    return raw


def _run_icalabel_on_edf(edf_path: Path, duration_sec: float = 60.0) -> dict:
    """Run ICA + ICLabel on one EDF file, return per-component classification."""
    import mne
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    mne.set_log_level("ERROR")

    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)

    # Crop to keep analysis fast
    if raw.times[-1] > duration_sec:
        raw.crop(tmax=duration_sec)

    # Convert bipolar CHB-MIT channels to unipolar + montage
    raw = _bipolar_to_unipolar(raw)
    if raw is None:
        return {"available": False, "error": "Could not map channels to standard 10-20 montage"}

    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude="bads")
    if len(eeg_picks) < 4:
        return {"available": False, "error": "Too few EEG channels for ICA"}

    raw.pick(eeg_picks)

    # Bandpass filter (required for ICA stability)
    raw.filter(1.0, 40.0, verbose=False)

    # Run ICA
    n_components = min(15, len(raw.ch_names) - 1)
    if n_components < 3:
        return {"available": False, "error": "Too few channels for meaningful ICA"}

    ica = mne.preprocessing.ICA(
        n_components=n_components,
        method="fastica",
        random_state=42,
        max_iter=500,
    )
    ica.fit(raw, verbose=False)

    # Apply ICLabel
    from mne_icalabel import label_components

    label_result = label_components(raw, ica, method="iclabel")

    # mne-icalabel returns:
    #   labels: list of string labels (e.g. "eye blink", "brain", "muscle artifact")
    #   y_pred_proba: 1D array of max-class confidence per component
    pred_labels = label_result.get("labels", [])
    pred_proba = label_result.get("y_pred_proba", None)

    # Map ICLabel string labels to our canonical 7-class names
    LABEL_MAP = {
        "brain": "brain",
        "muscle artifact": "muscle",
        "eye blink": "eye",
        "heart beat": "heart",
        "line noise": "line_noise",
        "channel noise": "channel_noise",
        "other": "other",
    }

    components = []
    class_counts = {c: 0 for c in IC_CLASSES}

    for i, lbl in enumerate(pred_labels):
        lbl_lower = str(lbl).lower().strip()
        canonical = LABEL_MAP.get(lbl_lower, "other")
        confidence = float(pred_proba[i]) if pred_proba is not None and i < len(pred_proba) else 1.0
        class_counts[canonical] = class_counts.get(canonical, 0) + 1

        components.append({
            "index": i,
            "label": canonical,
            "original_label": str(lbl),
            "confidence": round(confidence, 4),
        })

    brain_count = class_counts.get("brain", 0)
    artifact_count = sum(v for k, v in class_counts.items() if k != "brain")
    total = len(components)

    return {
        "available": True,
        "file": edf_path.name,
        "subject": edf_path.parent.name,
        "n_channels": len(raw.ch_names),
        "n_components": total,
        "duration_sec": round(float(raw.times[-1]), 1),
        "sfreq": raw.info["sfreq"],
        "brain_components": brain_count,
        "artifact_components": artifact_count,
        "brain_ratio": round(brain_count / max(total, 1), 4),
        "class_distribution": class_counts,
        "components": components,
    }


def icalabel_report() -> dict:
    """Full ICLabel dashboard payload — runs on real CHB-MIT data."""
    t0 = time.time()
    edfs = _find_edfs(max_subjects=3, max_per_subject=1)

    if not edfs:
        return {
            "available": False,
            "error": "No CHB-MIT EDF files found at data/real_eeg/epilepsy_physionet/",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }

    file_results = []
    aggregate_classes = {c: 0 for c in IC_CLASSES}
    total_components = 0
    total_brain = 0
    errors = []

    for edf in edfs:
        try:
            result = _run_icalabel_on_edf(edf)
            if result.get("available"):
                file_results.append(result)
                for cls, cnt in result.get("class_distribution", {}).items():
                    aggregate_classes[cls] = aggregate_classes.get(cls, 0) + cnt
                total_components += result.get("n_components", 0)
                total_brain += result.get("brain_components", 0)
            else:
                errors.append({"file": edf.name, "error": result.get("error", "unknown")})
        except Exception as exc:
            errors.append({"file": edf.name, "error": f"{type(exc).__name__}: {exc}"})

    elapsed = round(time.time() - t0, 2)

    return {
        "available": len(file_results) > 0,
        "tool": "mne-icalabel",
        "method": "ICLabel neural-net classifier",
        "classes": IC_CLASSES,
        "files_analyzed": len(file_results),
        "files_errored": len(errors),
        "total_components": total_components,
        "total_brain": total_brain,
        "total_artifact": total_components - total_brain,
        "brain_ratio": round(total_brain / max(total_components, 1), 4),
        "aggregate_class_distribution": aggregate_classes,
        "per_file": file_results,
        "errors": errors,
        "elapsed_sec": elapsed,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


if __name__ == "__main__":
    report = icalabel_report()
    print(json.dumps(report, indent=2, default=str))
