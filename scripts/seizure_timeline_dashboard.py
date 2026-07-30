"""
Seizure Timeline Dashboard — parse CHB-MIT PhysioNet summary files,
extract peri-onset EEG from .edf via MNE, run threshold-based spike
detection, and return a structured seizure catalog for the frontend.

Data source: data/real_eeg/epilepsy_physionet/chbNN/
Each subject folder contains chbNN-summary.txt (seizure annotations)
and chbNN_XX.edf files (raw EEG recordings, 256 Hz, 10-20 bipolar).
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Root of the real EEG data relative to the project
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "real_eeg" / "epilepsy_physionet"

# Number of seconds before/after seizure onset to extract
_PERI_ONSET_BEFORE = 5.0
_PERI_ONSET_AFTER = 5.0
# Downsample peri-onset waveform to roughly this many points
_DOWNSAMPLE_POINTS = 100
# Top N channels (by spike count) to include in the peri-onset payload
_TOP_CHANNELS = 4
# Spike detection: threshold = N * median absolute deviation
_MAD_MULTIPLIER = 3.0


# ---------------------------------------------------------------------------
# Summary file parser
# ---------------------------------------------------------------------------

def _parse_summary(summary_path: Path) -> list[dict[str, Any]]:
    """Parse a chbNN-summary.txt and return a list of seizure records.

    Each record:
        {
            "file": "chb01_03.edf",
            "file_start_time": "13:43:04",
            "seizure_index": 1,        # 1-based within the file
            "start_sec": 2996,
            "end_sec": 3036,
        }

    Handles both single-seizure format (``Seizure Start Time:``) and
    multi-seizure format (``Seizure 1 Start Time:``).
    """
    text = summary_path.read_text(errors="replace")
    seizures: list[dict[str, Any]] = []

    # Split into per-file blocks.  Each block starts with "File Name:".
    blocks = re.split(r"(?=^File Name:)", text, flags=re.MULTILINE)

    for block in blocks:
        fname_m = re.search(r"File Name:\s*(\S+)", block)
        if not fname_m:
            continue
        fname = fname_m.group(1).strip()

        start_time_m = re.search(r"File Start Time:\s*(\S+)", block)
        file_start_time = start_time_m.group(1).strip() if start_time_m else "00:00:00"

        n_seizures_m = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        if not n_seizures_m or int(n_seizures_m.group(1)) == 0:
            continue

        n_seizures = int(n_seizures_m.group(1))

        if n_seizures == 1:
            # Single seizure: "Seizure Start Time: 2996 seconds"
            ss = re.search(r"Seizure Start Time:\s*(\d+)", block)
            se = re.search(r"Seizure End Time:\s*(\d+)", block)
            if ss and se:
                seizures.append({
                    "file": fname,
                    "file_start_time": file_start_time,
                    "seizure_index": 1,
                    "start_sec": int(ss.group(1)),
                    "end_sec": int(se.group(1)),
                })
        else:
            # Multi-seizure: "Seizure 1 Start Time: 1679 seconds"
            for idx in range(1, n_seizures + 1):
                ss = re.search(
                    rf"Seizure\s+{idx}\s+Start Time:\s*(\d+)", block
                )
                se = re.search(
                    rf"Seizure\s+{idx}\s+End Time:\s*(\d+)", block
                )
                if ss and se:
                    seizures.append({
                        "file": fname,
                        "file_start_time": file_start_time,
                        "seizure_index": idx,
                        "start_sec": int(ss.group(1)),
                        "end_sec": int(se.group(1)),
                    })

    return seizures


def _clock_time(file_start: str, offset_sec: int) -> str:
    """Compute clock time = file_start + offset_sec, returned as HH:MM:SS."""
    try:
        base = datetime.strptime(file_start, "%H:%M:%S")
        result = base + timedelta(seconds=offset_sec)
        return result.strftime("%H:%M:%S")
    except Exception:
        return "??:??:??"


# ---------------------------------------------------------------------------
# EEG analysis helpers
# ---------------------------------------------------------------------------

def _analyse_seizure(
    edf_path: Path,
    start_sec: int,
    end_sec: int,
) -> dict[str, Any] | None:
    """Load a peri-onset EEG segment, run spike detection, and return stats.

    Returns None if the file cannot be read or the seizure window is
    outside the recording.
    """
    try:
        import mne  # noqa: F811
    except ImportError:
        return None

    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception as exc:
        logger.warning("Cannot read %s: %s", edf_path, exc)
        return None

    sfreq = raw.info["sfreq"]
    max_time = raw.n_times / sfreq

    # Peri-onset window
    t_start = max(0.0, start_sec - _PERI_ONSET_BEFORE)
    t_end = min(max_time, start_sec + _PERI_ONSET_AFTER)
    if t_start >= t_end:
        return None

    try:
        raw.load_data()
        raw_crop = raw.copy().crop(tmin=t_start, tmax=t_end)
        data = raw_crop.get_data()  # (n_channels, n_times)
        ch_names = raw_crop.ch_names
    except Exception as exc:
        logger.warning("Crop/load failed for %s: %s", edf_path, exc)
        return None

    # Convert to microvolts (EDF stores volts by default in MNE)
    data_uv = data * 1e6

    # --- Spike detection per channel (MAD threshold) ---
    spike_counts: dict[str, int] = {}
    for i, ch in enumerate(ch_names):
        signal = data_uv[i]
        median_val = np.median(np.abs(signal))
        mad = np.median(np.abs(signal - np.median(signal)))
        if mad == 0:
            mad = 1e-6
        threshold = _MAD_MULTIPLIER * mad
        spikes = int(np.sum(np.abs(signal - np.median(signal)) > threshold))
        spike_counts[ch] = spikes

    total_spikes = sum(spike_counts.values())
    channels_with_spikes = [
        ch for ch, cnt in spike_counts.items() if cnt > 0
    ]

    # Peak amplitude & channel
    abs_max_per_ch = np.max(np.abs(data_uv), axis=1)
    peak_idx = int(np.argmax(abs_max_per_ch))
    peak_amp = float(round(abs_max_per_ch[peak_idx], 2))
    peak_channel = ch_names[peak_idx]

    # --- Top channels by spike count for peri-onset waveform ---
    sorted_chs = sorted(spike_counts.items(), key=lambda x: x[1], reverse=True)
    top_chs = [ch for ch, _ in sorted_chs[:_TOP_CHANNELS]]

    # Downsample waveform
    n_samples = data_uv.shape[1]
    step = max(1, n_samples // _DOWNSAMPLE_POINTS)
    indices = list(range(0, n_samples, step))

    times_raw = raw_crop.times  # relative to crop start
    times_relative = times_raw - _PERI_ONSET_BEFORE  # relative to onset
    times_ds = [float(round(times_relative[i], 4)) for i in indices]

    channels_data: dict[str, list[float]] = {}
    for ch in top_chs:
        ch_idx = ch_names.index(ch)
        channels_data[ch] = [
            float(round(data_uv[ch_idx, i], 2)) for i in indices
        ]

    return {
        "spike_count": total_spikes,
        "peak_amplitude_uv": peak_amp,
        "peak_channel": peak_channel,
        "channels_with_spikes": channels_with_spikes,
        "peri_onset_eeg": {
            "times": times_ds,
            "channels": channels_data,
        },
    }


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_seizure_timeline_report(*, include_eeg: bool = False) -> dict[str, Any]:
    """Build the full seizure timeline report across all CHB-MIT subjects.

    Scans ``data/real_eeg/epilepsy_physionet/chbNN/`` for summary files,
    parses seizure annotations, and returns a structured dict suitable
    for the Seizure Timeline Dashboard frontend.

    Args:
        include_eeg: If True, also load .edf files via MNE for peri-onset
            EEG extraction and spike detection (slow — loads GBs of data).
            Default False for fast endpoint response.

    Returns:
        dict with keys: available, total_seizures, subjects, timeline,
        per_subject_summary.  If data is unavailable, returns
        {"available": False, "error": "reason"}.
    """
    has_mne = False
    if include_eeg:
        try:
            import mne  # noqa: F811 — verify import
            has_mne = True
        except ImportError:
            pass

    if not _DATA_ROOT.is_dir():
        return {
            "available": False,
            "error": f"Data directory not found: {_DATA_ROOT}",
        }

    # Discover subjects
    subject_dirs = sorted([
        d for d in _DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("chb")
    ])

    if not subject_dirs:
        return {"available": False, "error": "No subject directories found"}

    timeline: list[dict[str, Any]] = []
    subjects_seen: list[str] = []

    for subj_dir in subject_dirs:
        subj = subj_dir.name  # e.g. "chb01"
        summary_file = subj_dir / f"{subj}-summary.txt"
        if not summary_file.is_file():
            logger.info("No summary for %s, skipping", subj)
            continue

        seizure_records = _parse_summary(summary_file)
        if not seizure_records:
            continue

        subjects_seen.append(subj)

        for rec in seizure_records:
            edf_path = subj_dir / rec["file"]
            if not edf_path.is_file():
                logger.info("EDF missing: %s, skipping seizure", edf_path)
                continue

            duration = rec["end_sec"] - rec["start_sec"]
            onset_clock = _clock_time(rec["file_start_time"], rec["start_sec"])

            entry: dict[str, Any] = {
                "subject": subj,
                "file": rec["file"],
                "seizure_index": rec["seizure_index"],
                "start_sec": rec["start_sec"],
                "end_sec": rec["end_sec"],
                "duration_sec": duration,
                "file_start_time": rec["file_start_time"],
                "onset_clock": onset_clock,
                # Defaults if EEG analysis fails
                "spike_count": 0,
                "peak_amplitude_uv": 0.0,
                "peak_channel": "",
                "channels_with_spikes": [],
                "peri_onset_eeg": {"times": [], "channels": {}},
            }

            if include_eeg and has_mne:
                analysis = _analyse_seizure(edf_path, rec["start_sec"], rec["end_sec"])
                if analysis:
                    entry.update(analysis)

            timeline.append(entry)

    # --- Per-subject summary ---
    per_subject: dict[str, dict[str, Any]] = {}
    for ev in timeline:
        subj = ev["subject"]
        if subj not in per_subject:
            per_subject[subj] = {
                "subject": subj,
                "total_seizures": 0,
                "total_duration_sec": 0,
                "total_spikes": 0,
                "files_with_seizures": [],
            }
        ps = per_subject[subj]
        ps["total_seizures"] += 1
        ps["total_duration_sec"] += ev["duration_sec"]
        ps["total_spikes"] += ev["spike_count"]
        if ev["file"] not in ps["files_with_seizures"]:
            ps["files_with_seizures"].append(ev["file"])

    per_subject_summary = []
    for ps in per_subject.values():
        n = ps["total_seizures"]
        ps["mean_duration_sec"] = round(ps["total_duration_sec"] / n, 1) if n else 0.0
        per_subject_summary.append(ps)

    return {
        "available": True,
        "total_seizures": len(timeline),
        "subjects": subjects_seen,
        "timeline": timeline,
        "per_subject_summary": per_subject_summary,
    }


if __name__ == "__main__":
    import json

    report = generate_seizure_timeline_report()
    print(json.dumps(report, indent=2, default=str))
