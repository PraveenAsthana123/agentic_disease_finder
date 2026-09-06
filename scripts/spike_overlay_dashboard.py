"""
Spike / Sharp-Wave Overlay Analysis — detect individual spike and sharp-wave
events with exact timestamps and morphological features from CHB-MIT
PhysioNet EEG data.  Designed for overlaying markers on EEG waveform traces.

Data source: data/real_eeg/epilepsy_physionet/chbNN/
Each subject folder contains chbNN-summary.txt (seizure annotations)
and chbNN_XX.edf files (raw EEG recordings, 256 Hz, 10-20 bipolar).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_ROOT = _PROJECT_ROOT / "data" / "real_eeg" / "epilepsy_physionet"

# ---------------------------------------------------------------------------
# Detection parameters
# ---------------------------------------------------------------------------
_MAD_SPIKE_MULTIPLIER = 3.0
_MAD_SHARP_MULTIPLIER = 2.5

_SPIKE_MIN_MS = 20.0
_SPIKE_MAX_MS = 70.0
_SHARP_MIN_MS = 70.0
_SHARP_MAX_MS = 200.0

# Bandpass filter edges (Hz)
_BANDPASS_LOW = 1.0
_BANDPASS_HIGH = 70.0

# Limits
_MAX_SEIZURE_FILES_PER_SUBJECT = 2
_MAX_EVENTS_TOTAL = 500
_MAX_WAVEFORM_SAMPLES = 10  # 5 spikes + 5 sharp waves
_WAVEFORM_HALF_WINDOW_MS = 100.0  # ±100 ms around event center
_WAVEFORM_POINTS = 50


# ---------------------------------------------------------------------------
# Reuse summary parser from the seizure timeline dashboard
# ---------------------------------------------------------------------------

def _import_parse_summary():
    """Import _parse_summary from seizure_timeline_dashboard."""
    try:
        from scripts.seizure_timeline_dashboard import _parse_summary
        return _parse_summary
    except ImportError:
        pass
    # Fallback: direct import when running as a script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "seizure_timeline_dashboard",
        Path(__file__).resolve().parent / "seizure_timeline_dashboard.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._parse_summary


def _clock_time(file_start: str, offset_sec: float) -> str:
    """Compute clock time = file_start + offset_sec, returned as HH:MM:SS."""
    from datetime import datetime, timedelta
    try:
        base = datetime.strptime(file_start, "%H:%M:%S")
        result = base + timedelta(seconds=int(offset_sec))
        return result.strftime("%H:%M:%S")
    except Exception:
        return "??:??:??"


# ---------------------------------------------------------------------------
# Core spike / sharp-wave detector
# ---------------------------------------------------------------------------

def _detect_events_in_signal(
    signal_uv: np.ndarray,
    sfreq: float,
) -> list[dict[str, Any]]:
    """Detect spike and sharp-wave events in a single-channel signal.

    Uses zero-crossing interval analysis on the bandpass-filtered signal.
    Returns a list of event dicts with keys: sample_idx, type, amplitude_uv,
    duration_ms.
    """
    # Compute MAD threshold
    median_val = np.median(signal_uv)
    mad = np.median(np.abs(signal_uv - median_val))
    if mad < 1e-6:
        return []

    spike_thresh = _MAD_SPIKE_MULTIPLIER * mad
    sharp_thresh = _MAD_SHARP_MULTIPLIER * mad

    # Find zero crossings (relative to median)
    centered = signal_uv - median_val
    sign_changes = np.where(np.diff(np.sign(centered)))[0]

    if len(sign_changes) < 2:
        return []

    events: list[dict[str, Any]] = []

    for i in range(len(sign_changes) - 1):
        start_idx = sign_changes[i]
        end_idx = sign_changes[i + 1]
        segment = centered[start_idx:end_idx + 1]

        if len(segment) < 2:
            continue

        duration_samples = end_idx - start_idx
        duration_ms = (duration_samples / sfreq) * 1000.0

        peak_val = np.max(np.abs(segment))
        peak_offset = np.argmax(np.abs(segment))
        peak_sample = start_idx + peak_offset

        # Classify by duration and amplitude
        if _SPIKE_MIN_MS <= duration_ms <= _SPIKE_MAX_MS and peak_val > spike_thresh:
            events.append({
                "sample_idx": int(peak_sample),
                "type": "spike",
                "amplitude_uv": float(round(peak_val, 2)),
                "duration_ms": float(round(duration_ms, 2)),
            })
        elif _SHARP_MIN_MS <= duration_ms <= _SHARP_MAX_MS and peak_val > sharp_thresh:
            events.append({
                "sample_idx": int(peak_sample),
                "type": "sharp_wave",
                "amplitude_uv": float(round(peak_val, 2)),
                "duration_ms": float(round(duration_ms, 2)),
            })

    return events


_PERI_SEIZURE_SEC = 30.0  # analyse ±30 s around each seizure onset


def _analyse_file(
    edf_path: Path,
    subject: str,
    file_start_time: str,
    seizure_windows: list[tuple[int, int]],
    mne,
) -> list[dict[str, Any]]:
    """Analyse peri-seizure windows in an EDF file for spike/sharp-wave events.

    Only crops and filters ±30 s around each seizure onset to keep
    processing fast (full-file filtering is too slow for API use).

    Returns a list of event dicts ready for the report.
    """
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception as exc:
        logger.warning("Cannot read %s: %s", edf_path, exc)
        return []

    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    max_time = raw.n_times / sfreq

    all_events: list[dict[str, Any]] = []

    for start_sec, end_sec in seizure_windows:
        t_start = max(0.0, start_sec - _PERI_SEIZURE_SEC)
        t_end = min(max_time, end_sec + _PERI_SEIZURE_SEC)
        if t_start >= t_end:
            continue

        try:
            raw.load_data()
            crop = raw.copy().crop(tmin=t_start, tmax=t_end)
            crop.filter(
                l_freq=_BANDPASS_LOW,
                h_freq=_BANDPASS_HIGH,
                verbose=False,
                n_jobs=1,
            )
            data = crop.get_data() * 1e6  # to microvolts
        except Exception as exc:
            logger.warning("Crop/filter failed for %s [%.0f-%.0f]: %s",
                           edf_path, t_start, t_end, exc)
            continue

        for ch_idx, ch_name in enumerate(ch_names):
            signal = data[ch_idx]
            raw_events = _detect_events_in_signal(signal, sfreq)

            for ev in raw_events:
                # Map sample index back to file time
                time_sec = t_start + (ev["sample_idx"] / sfreq)
                file_sample = int(time_sec * sfreq)
                all_events.append({
                    "subject": subject,
                    "file": edf_path.name,
                    "type": ev["type"],
                    "time_sec": float(round(time_sec, 4)),
                    "channel": ch_name,
                    "amplitude_uv": ev["amplitude_uv"],
                    "duration_ms": ev["duration_ms"],
                    "onset_clock": _clock_time(file_start_time, time_sec),
                    "_sample_idx": file_sample,
                    "_ch_idx": ch_idx,
                })

    return all_events


def _extract_waveform(
    edf_path: Path,
    ch_idx: int,
    sample_idx: int,
    sfreq: float,
    mne,
) -> dict[str, Any] | None:
    """Extract a short waveform snippet around an event for visualization."""
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception:
        return None

    half_samples = int((_WAVEFORM_HALF_WINDOW_MS / 1000.0) * sfreq)
    start = max(0, sample_idx - half_samples)
    end = min(raw.n_times, sample_idx + half_samples)

    if end - start < 4:
        return None

    try:
        raw.load_data()
        data = raw.get_data(picks=[ch_idx], start=start, stop=end) * 1e6
    except Exception:
        return None

    signal = data[0]
    n = len(signal)

    # Downsample to ~_WAVEFORM_POINTS
    step = max(1, n // _WAVEFORM_POINTS)
    indices = list(range(0, n, step))[:_WAVEFORM_POINTS]

    center_ms = (sample_idx - start) / sfreq * 1000.0
    times = [float(round((idx / sfreq * 1000.0) - center_ms, 3)) for idx in indices]
    values = [float(round(signal[idx], 2)) for idx in indices]
    peak_uv = float(round(np.max(np.abs(signal)), 2))

    return {
        "times": times,
        "values": values,
        "peak_uv": peak_uv,
    }


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def generate_spike_overlay_report() -> dict[str, Any]:
    """Build the spike/sharp-wave overlay report across CHB-MIT subjects.

    Scans seizure-containing EDF files, detects individual spike and
    sharp-wave events with morphological features, and returns a
    structured dict suitable for waveform overlay visualization.

    Returns:
        dict with keys: available, total_spikes, total_sharp_waves,
        subjects_analyzed, analysis_params, events, per_channel_summary,
        per_subject_summary, waveform_samples.
        If MNE or data is unavailable, returns
        {"available": False, "error": "reason"}.
    """
    try:
        import mne
    except ImportError:
        return {"available": False, "error": "MNE library not installed"}

    if not _DATA_ROOT.is_dir():
        return {
            "available": False,
            "error": f"Data directory not found: {_DATA_ROOT}",
        }

    try:
        _parse_summary = _import_parse_summary()
    except Exception as exc:
        return {
            "available": False,
            "error": f"Cannot import summary parser: {exc}",
        }

    # Discover subjects
    subject_dirs = sorted([
        d for d in _DATA_ROOT.iterdir()
        if d.is_dir() and d.name.startswith("chb")
    ])

    if not subject_dirs:
        return {"available": False, "error": "No subject directories found"}

    all_events: list[dict[str, Any]] = []
    subjects_analyzed: list[str] = []
    # Track which EDF files we loaded (path -> file_start_time) for waveform extraction
    file_start_times: dict[str, str] = {}

    for subj_dir in subject_dirs:
        subj = subj_dir.name
        summary_file = subj_dir / f"{subj}-summary.txt"
        if not summary_file.is_file():
            continue

        seizure_records = _parse_summary(summary_file)
        if not seizure_records:
            continue

        # Group seizure windows by file, limit to first N unique files
        file_records: dict[str, list[dict[str, Any]]] = {}
        for rec in seizure_records:
            fname = rec["file"]
            if fname not in file_records:
                if len(file_records) >= _MAX_SEIZURE_FILES_PER_SUBJECT:
                    break
                file_records[fname] = []
            file_records[fname].append(rec)

        subject_had_events = False
        for fname, recs in file_records.items():
            edf_path = subj_dir / fname
            if not edf_path.is_file():
                logger.info("EDF missing: %s, skipping", edf_path)
                continue

            file_start_times[str(edf_path)] = recs[0]["file_start_time"]
            windows = [(r["start_sec"], r["end_sec"]) for r in recs]
            events = _analyse_file(
                edf_path, subj, recs[0]["file_start_time"], windows, mne,
            )
            if events:
                subject_had_events = True
                all_events.extend(events)

        if subject_had_events:
            subjects_analyzed.append(subj)

    if not all_events:
        return {
            "available": True,
            "total_spikes": 0,
            "total_sharp_waves": 0,
            "subjects_analyzed": 0,
            "analysis_params": {
                "mad_threshold": _MAD_SPIKE_MULTIPLIER,
                "min_spike_duration_ms": _SPIKE_MIN_MS,
                "max_spike_duration_ms": _SPIKE_MAX_MS,
                "min_sharp_wave_duration_ms": _SHARP_MIN_MS,
                "max_sharp_wave_duration_ms": _SHARP_MAX_MS,
            },
            "events": [],
            "per_channel_summary": [],
            "per_subject_summary": [],
            "waveform_samples": [],
        }

    # Sort by amplitude descending and limit to top N events
    all_events.sort(key=lambda e: e["amplitude_uv"], reverse=True)
    truncated_events = all_events[:_MAX_EVENTS_TOTAL]

    # --- Waveform samples: top 5 spikes + top 5 sharp waves ---
    spikes_sorted = [e for e in all_events if e["type"] == "spike"]
    sharps_sorted = [e for e in all_events if e["type"] == "sharp_wave"]

    waveform_candidates = spikes_sorted[:5] + sharps_sorted[:5]
    waveform_samples: list[dict[str, Any]] = []

    for ev in waveform_candidates:
        subj_dir = _DATA_ROOT / ev["subject"]
        edf_path = subj_dir / ev["file"]
        if not edf_path.is_file():
            continue

        try:
            raw_tmp = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
            sfreq = raw_tmp.info["sfreq"]
            del raw_tmp
        except Exception:
            continue

        wf = _extract_waveform(
            edf_path, ev["_ch_idx"], ev["_sample_idx"], sfreq, mne,
        )
        if wf is not None:
            waveform_samples.append({
                "type": ev["type"],
                "subject": ev["subject"],
                "channel": ev["channel"],
                "times": wf["times"],
                "values": wf["values"],
                "peak_uv": wf["peak_uv"],
            })

        if len(waveform_samples) >= _MAX_WAVEFORM_SAMPLES:
            break

    # --- Per-channel summary ---
    channel_agg: dict[str, dict[str, Any]] = {}
    for ev in all_events:
        ch = ev["channel"]
        if ch not in channel_agg:
            channel_agg[ch] = {
                "channel": ch,
                "spikes": 0,
                "sharp_waves": 0,
                "_amplitudes": [],
            }
        if ev["type"] == "spike":
            channel_agg[ch]["spikes"] += 1
        else:
            channel_agg[ch]["sharp_waves"] += 1
        channel_agg[ch]["_amplitudes"].append(ev["amplitude_uv"])

    per_channel_summary = []
    for info in channel_agg.values():
        amps = info.pop("_amplitudes")
        info["mean_amplitude_uv"] = float(round(np.mean(amps), 2)) if amps else 0.0
        per_channel_summary.append(info)
    per_channel_summary.sort(
        key=lambda x: x["spikes"] + x["sharp_waves"], reverse=True,
    )

    # --- Per-subject summary ---
    subject_agg: dict[str, dict[str, Any]] = {}
    for ev in all_events:
        subj = ev["subject"]
        if subj not in subject_agg:
            subject_agg[subj] = {
                "subject": subj,
                "spikes": 0,
                "sharp_waves": 0,
                "_files": set(),
            }
        if ev["type"] == "spike":
            subject_agg[subj]["spikes"] += 1
        else:
            subject_agg[subj]["sharp_waves"] += 1
        subject_agg[subj]["_files"].add(ev["file"])

    per_subject_summary = []
    for info in subject_agg.values():
        files = info.pop("_files")
        info["files_analyzed"] = len(files)
        per_subject_summary.append(info)
    per_subject_summary.sort(key=lambda x: x["subject"])

    # --- Clean internal fields from events ---
    clean_events = []
    for ev in truncated_events:
        clean = {k: v for k, v in ev.items() if not k.startswith("_")}
        clean_events.append(clean)

    total_spikes = sum(1 for e in all_events if e["type"] == "spike")
    total_sharps = sum(1 for e in all_events if e["type"] == "sharp_wave")

    return {
        "available": True,
        "total_spikes": total_spikes,
        "total_sharp_waves": total_sharps,
        "subjects_analyzed": len(subjects_analyzed),
        "analysis_params": {
            "mad_threshold": _MAD_SPIKE_MULTIPLIER,
            "min_spike_duration_ms": _SPIKE_MIN_MS,
            "max_spike_duration_ms": _SPIKE_MAX_MS,
            "min_sharp_wave_duration_ms": _SHARP_MIN_MS,
            "max_sharp_wave_duration_ms": _SHARP_MAX_MS,
        },
        "events": clean_events,
        "per_channel_summary": per_channel_summary,
        "per_subject_summary": per_subject_summary,
        "waveform_samples": waveform_samples,
    }


if __name__ == "__main__":
    import json

    report = generate_spike_overlay_report()
    print(json.dumps(report, indent=2, default=str))
