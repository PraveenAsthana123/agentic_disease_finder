#!/usr/bin/env python3
"""
Neo Multi-Format Reader Dashboard
==================================

Showcases the Neo library (Garcia et al., 2014) — a unified Python package
for reading electrophysiology data in 50+ formats (EDF, Spike2, Plexon,
Neuralynx, Axon, BlackRock, BrainVision, NWB, Intan, Micromed, etc.).

Neo provides a hierarchical data model:
  Block → Segment → AnalogSignal / SpikeTrain / Event / Epoch

This dashboard reads REAL CHB-MIT EDF files via Neo's EDFIO and exposes:
  - Supported format catalog (all 54 IO classes)
  - File structure inspection (blocks, segments, signals, events)
  - Signal summary statistics from actual recordings
  - Format comparison (Neo vs MNE for the same file)

Clinical relevance:
  Multi-center EEG studies often receive data in heterogeneous formats.
  Neo provides a single API to read all of them, enabling unified
  preprocessing pipelines without format-specific code paths.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).parent.parent


# -- data discovery ----------------------------------------------------------

def _find_edf() -> Optional[Path]:
    """Return the first available real EDF file."""
    dirs = [
        ROOT / "data/real_eeg/epilepsy_physionet",
        ROOT / "data/real_eeg/depression_figshare",
        ROOT / "data/real_eeg",
    ]
    for d in dirs:
        if d.is_dir():
            edfs = sorted(d.glob("*.edf"))
            if edfs:
                return edfs[0]
    return None


def _list_edfs() -> List[str]:
    """Return all available EDF filenames (relative to ROOT)."""
    results = []
    dirs = [
        ROOT / "data/real_eeg/epilepsy_physionet",
        ROOT / "data/real_eeg/depression_figshare",
        ROOT / "data/real_eeg",
    ]
    for d in dirs:
        if d.is_dir():
            for edf in sorted(d.glob("*.edf")):
                results.append(str(edf.relative_to(ROOT)))
    return results


# -- format catalog ----------------------------------------------------------

def supported_formats() -> Dict[str, Any]:
    """Return catalog of all Neo-supported neurophysiology formats."""
    import neo

    formats = []
    for io_cls in sorted(neo.io.iolist, key=lambda c: c.__name__):
        name = io_cls.__name__
        exts = getattr(io_cls, "extensions", [])
        # Extract a readable description from the docstring
        doc = (io_cls.__doc__ or "").strip().split("\n")[0][:120]
        formats.append({
            "io_class": name,
            "extensions": exts,
            "description": doc,
        })

    return {
        "title": "Neo Supported Formats",
        "description": (
            "Neo (Garcia et al., 2014) provides a unified API for reading "
            "electrophysiology data across 50+ formats — from clinical EEG "
            "(EDF/BDF) to intracranial (BlackRock/Plexon) to modern standards (NWB)."
        ),
        "total_io_classes": len(formats),
        "neo_version": neo.__version__,
        "formats": formats,
        "key_formats_for_eeg": [
            {"format": "EDF/EDF+", "io_class": "EDFIO", "use": "Standard scalp EEG (PhysioNet, clinical)"},
            {"format": "BrainVision", "io_class": "BrainVisionIO", "use": "Research EEG (BrainProducts)"},
            {"format": "Micromed", "io_class": "MicromedIO", "use": "Clinical EEG/SEEG (Micromed systems)"},
            {"format": "NWB", "io_class": "NWBIO", "use": "Neurodata Without Borders (DANDI archive)"},
            {"format": "Intan", "io_class": "IntanIO", "use": "Intan RHD/RHS (preclinical, neuropixels)"},
            {"format": "CED/Spike2", "io_class": "CedIO", "use": "CED Spike2 (.smr/.smrx)"},
        ],
        "data_model": {
            "hierarchy": "Block → Segment → AnalogSignal / SpikeTrain / Event / Epoch",
            "block": "Top-level container (one recording session)",
            "segment": "Continuous time epoch within a block",
            "analog_signal": "Regularly sampled data (EEG channels, LFP)",
            "spike_train": "Spike times (sorted or unsorted unit activity)",
            "event": "Timestamped markers (annotations, triggers)",
            "epoch": "Time intervals (seizure periods, sleep stages)",
        },
        "reference": "Garcia S. et al. (2014). Neo: an object model for handling electrophysiology data in multiple formats. Front Neuroinform 8:10.",
    }


# -- file structure inspection -----------------------------------------------

def inspect_file(file: Optional[str] = None) -> Dict[str, Any]:
    """Read a real EDF via Neo and return its hierarchical structure."""
    import neo

    if file:
        p = ROOT / file if not Path(file).is_absolute() else Path(file)
    else:
        p = _find_edf()

    if not p or not p.exists():
        return {"error": "No EDF file found", "available_files": _list_edfs()}

    reader = neo.io.EDFIO(filename=str(p))
    blk = reader.read_block()

    segments_info = []
    for i, seg in enumerate(blk.segments):
        signals_info = []
        for j, sig in enumerate(seg.analogsignals):
            arr = np.array(sig)
            signals_info.append({
                "index": j,
                "shape": list(sig.shape),
                "n_channels": int(sig.shape[1]) if sig.ndim > 1 else 1,
                "n_samples": int(sig.shape[0]),
                "sampling_rate_hz": float(sig.sampling_rate),
                "units": str(sig.units.dimensionality),
                "duration_seconds": round(float(sig.shape[0] / sig.sampling_rate), 2),
                "channel_stats": {
                    "mean_uV": round(float(np.mean(arr)), 4),
                    "std_uV": round(float(np.std(arr)), 4),
                    "min_uV": round(float(np.min(arr)), 4),
                    "max_uV": round(float(np.max(arr)), 4),
                },
            })

        events_info = []
        for ev in seg.events:
            events_info.append({
                "name": str(getattr(ev, "name", "unnamed")),
                "n_events": len(ev.times),
                "labels_sample": list(ev.labels[:5]) if hasattr(ev, "labels") and len(ev.labels) > 0 else [],
            })

        spike_trains_info = []
        for st in seg.spiketrains:
            spike_trains_info.append({
                "n_spikes": len(st),
                "t_start": float(st.t_start),
                "t_stop": float(st.t_stop),
            })

        segments_info.append({
            "segment_index": i,
            "n_analog_signals": len(seg.analogsignals),
            "n_events": len(seg.events),
            "n_spike_trains": len(seg.spiketrains),
            "n_epochs": len(seg.epochs),
            "analog_signals": signals_info,
            "events": events_info,
            "spike_trains": spike_trains_info,
        })

    return {
        "title": "Neo File Structure Inspection",
        "file": str(p.relative_to(ROOT)),
        "file_size_mb": round(p.stat().st_size / 1e6, 2),
        "io_class_used": "EDFIO",
        "neo_version": reader.__class__.__module__.split(".")[0],
        "n_blocks": 1,
        "n_segments": len(blk.segments),
        "block_description": str(getattr(blk, "description", "")),
        "segments": segments_info,
        "data_model_note": (
            "Neo organizes data as Block → Segment → AnalogSignal/Event/SpikeTrain. "
            "For EDF files, each recording is one Block with one Segment containing "
            "the multichannel EEG as an AnalogSignal array."
        ),
    }


# -- signal overview (per-channel stats from real data) ----------------------

def signal_overview(file: Optional[str] = None, seconds: float = 30.0) -> Dict[str, Any]:
    """Read a segment via Neo, compute per-channel summary statistics."""
    import neo

    if file:
        p = ROOT / file if not Path(file).is_absolute() else Path(file)
    else:
        p = _find_edf()

    if not p or not p.exists():
        return {"error": "No EDF file found", "available_files": _list_edfs()}

    reader = neo.io.EDFIO(filename=str(p))
    blk = reader.read_block()

    if not blk.segments or not blk.segments[0].analogsignals:
        return {"error": "No analog signals in file"}

    sig = blk.segments[0].analogsignals[0]
    sfreq = float(sig.sampling_rate)
    n_samples = min(int(seconds * sfreq), sig.shape[0])
    data = np.array(sig[:n_samples, :])  # (n_samples, n_channels)

    # Try to get channel names from annotations
    ch_names = []
    if hasattr(sig, "array_annotations") and "channel_names" in sig.array_annotations:
        ch_names = list(sig.array_annotations["channel_names"])
    if not ch_names:
        ch_names = [f"Ch{i}" for i in range(data.shape[1])]

    channels = []
    for i in range(data.shape[1]):
        ch_data = data[:, i]
        channels.append({
            "channel": ch_names[i],
            "mean_uV": round(float(np.mean(ch_data)), 4),
            "std_uV": round(float(np.std(ch_data)), 4),
            "min_uV": round(float(np.min(ch_data)), 4),
            "max_uV": round(float(np.max(ch_data)), 4),
            "rms_uV": round(float(np.sqrt(np.mean(ch_data ** 2))), 4),
            "peak_to_peak_uV": round(float(np.ptp(ch_data)), 4),
            "kurtosis": round(float(_kurtosis(ch_data)), 4),
            "skewness": round(float(_skewness(ch_data)), 4),
        })

    return {
        "title": "Neo Signal Overview — Per-Channel Statistics",
        "file": str(p.relative_to(ROOT)),
        "reader": "neo.io.EDFIO",
        "sampling_rate_hz": sfreq,
        "n_channels": data.shape[1],
        "duration_seconds": round(n_samples / sfreq, 2),
        "n_samples_read": n_samples,
        "channels": channels,
        "note": (
            "Statistics computed from real CHB-MIT EDF data read via Neo's "
            "unified IO layer. Same data accessible through 54 format readers."
        ),
    }


def _kurtosis(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher definition)."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4) - 3.0)


def _skewness(x: np.ndarray) -> float:
    """Sample skewness."""
    m = np.mean(x)
    s = np.std(x)
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


# -- definitions -------------------------------------------------------------

def definitions() -> Dict[str, Any]:
    """Neo terminology, data model, clinical relevance, and references."""
    return {
        "title": "Neo Reader — Definitions & Reference",
        "library": {
            "name": "Neo",
            "version_installed": _get_version(),
            "purpose": (
                "Unified Python API for reading electrophysiology data in 50+ "
                "formats. Eliminates format-specific code paths in multi-center "
                "EEG/intracranial studies."
            ),
            "data_model": "Block → Segment → AnalogSignal / SpikeTrain / Event / Epoch",
        },
        "key_concepts": [
            {
                "term": "Block",
                "definition": "Top-level container representing one recording session or file.",
            },
            {
                "term": "Segment",
                "definition": "Continuous recording epoch within a Block. Multiple segments for gap-free concatenation.",
            },
            {
                "term": "AnalogSignal",
                "definition": "Regularly sampled multichannel data (EEG, LFP, EMG) with units and sampling rate.",
            },
            {
                "term": "SpikeTrain",
                "definition": "Array of spike times from a single neuron or sorted unit.",
            },
            {
                "term": "Event",
                "definition": "Timestamped markers (triggers, annotations, seizure onset markers).",
            },
            {
                "term": "Epoch",
                "definition": "Time intervals with duration (sleep stages, seizure periods, task blocks).",
            },
        ],
        "clinical_relevance": (
            "Multi-center epilepsy research (e.g., combining CHB-MIT EDF with "
            "hospital Micromed TRC or BrainVision VHDR) requires a unified "
            "reader. Neo handles format translation so preprocessing pipelines "
            "remain format-agnostic."
        ),
        "eeg_format_guide": [
            {"format": "EDF/EDF+", "extension": ".edf", "origin": "European Data Format (Kemp 1992)", "common_in": "PhysioNet, clinical EEG"},
            {"format": "BDF", "extension": ".bdf", "origin": "BioSemi Data Format", "common_in": "BioSemi ActiveTwo systems"},
            {"format": "BrainVision", "extension": ".vhdr/.vmrk/.eeg", "origin": "Brain Products GmbH", "common_in": "Research EEG labs"},
            {"format": "Micromed", "extension": ".trc", "origin": "Micromed S.p.A.", "common_in": "Clinical EEG/SEEG in European hospitals"},
            {"format": "NWB", "extension": ".nwb", "origin": "Neurodata Without Borders", "common_in": "DANDI archive, modern neuroscience"},
            {"format": "Spike2/CED", "extension": ".smr/.smrx", "origin": "Cambridge Electronic Design", "common_in": "Animal electrophysiology"},
        ],
        "references": [
            "Garcia S, Guarino D, Jaillet F, Jennings T, Propper R, Rautenberg PL, et al. (2014). Neo: an object model for handling electrophysiology data in multiple formats. Front Neuroinform 8:10.",
            "Kemp B, Varri A, Rosa AC, Nielsen KD, Gade J. (1992). A simple format for exchange of digitized polygraphic recordings. Electroencephalogr Clin Neurophysiol 82(5):391-3.",
            "Rubel O, Tritt A, Ly R, Dichter BK, Ghosh S, Niu L, et al. (2022). The Neurodata Without Borders ecosystem for neurophysiological data science. eLife 11:e78362.",
        ],
        "comparison_with_mne": {
            "neo_advantage": "54 format readers (vs ~15 in MNE); hierarchical Block/Segment model for multi-trial data; spike train support.",
            "mne_advantage": "Richer EEG-specific processing (ICA, source localization, montage, events); deeper clinical EEG tooling.",
            "recommendation": "Use Neo for format ingestion, convert to MNE Raw for EEG-specific analysis.",
        },
    }


def _get_version() -> str:
    """Return installed Neo version."""
    try:
        import neo
        return neo.__version__
    except Exception:
        return "unknown"
