#!/usr/bin/env python3
"""
Synchrosqueezing Transform Dashboard — ssqueezepy (Muradeli, 2020)
===================================================================

Computes real synchrosqueezing continuous wavelet transforms (SSQ-CWT)
from EEG recordings on disk using the ssqueezepy library.  Every value
comes from actual EDF data — nothing fabricated.

The synchrosqueezing transform sharpens the time-frequency representation
produced by a CWT by reassigning energy to the instantaneous-frequency
ridges, yielding a high-resolution spectrogram suitable for non-stationary
signals like EEG.

Functions provided:
  - synchrosqueezing_overview   — per-channel peak freq, energy, band breakdown
  - synchrosqueezing_spectrum   — marginal frequency spectrum per channel
  - synchrosqueezing_definitions — method definitions and clinical relevance

Clinical relevance:
  SSQ-CWT resolves transient oscillatory bursts (epileptiform spikes,
  spindles, K-complexes) that standard FFT or short-time Fourier
  transform smear across frequency bins.  It is used in seizure onset
  detection, sleep-stage scoring, and event-related synchronization /
  desynchronization analysis.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).parent.parent

# ── EEG band definitions (Hz) ─────────────────────────────────────
_BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


# ── data discovery ──────────────────────────────────────────────────

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


def _load_raw(file: Optional[str] = None, seconds: float = 10.0):
    """Load raw EDF via MNE, return (data_array, sfreq, ch_names)."""
    import mne
    mne.set_log_level("ERROR")

    if file:
        p = ROOT / file if not Path(file).is_absolute() else Path(file)
    else:
        p = _find_edf()
    if not p or not p.exists():
        return None, None, None

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose=False)
    raw.filter(0.5, 45.0, verbose=False)
    sfreq = raw.info["sfreq"]
    n_samples = min(int(seconds * sfreq), raw.n_times)
    data = raw.get_data(start=0, stop=n_samples)  # (n_ch, n_times)
    ch_names = raw.ch_names
    return data, sfreq, ch_names


# ── helpers ─────────────────────────────────────────────────────────

def _classify_band(freq_hz: float) -> str:
    """Classify a frequency into a clinical EEG band name."""
    for band_name, (lo, hi) in _BANDS.items():
        if lo <= freq_hz < hi:
            return band_name
    if freq_hz >= 45.0:
        return "gamma"
    return "sub-delta"


def _band_energy(freqs: np.ndarray, power_spectrum: np.ndarray) -> Dict[str, float]:
    """Compute total energy within each clinical EEG band."""
    result = {}
    for band_name, (lo, hi) in _BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if mask.any():
            result[band_name] = round(float(np.sum(power_spectrum[mask])), 6)
        else:
            result[band_name] = 0.0
    return result


# ── synchrosqueezing computation ───────────────────────────────────

def _compute_ssq_channel(signal: np.ndarray, sfreq: float):
    """
    Compute SSQ-CWT for a single channel.

    Returns (Twx, ssq_freqs) where Twx is the synchrosqueezed matrix
    and ssq_freqs are the corresponding frequency values.
    """
    from ssqueezepy import ssq_cwt

    # ssq_cwt returns: Tx, Wx, ssq_freqs, scales, ...
    Twx, Wx, ssq_freqs, scales, *_ = ssq_cwt(
        signal, wavelet='morlet', fs=sfreq
    )
    return Twx, ssq_freqs


# ── public API ──────────────────────────────────────────────────────

def synchrosqueezing_overview(
    file: Optional[str] = None, seconds: float = 10.0
) -> Dict[str, Any]:
    """
    Full synchrosqueezing dashboard: per-channel peak frequency, energy,
    bandwidth, dominant band, and aggregate frequency-band breakdown.
    """
    try:
        from ssqueezepy import ssq_cwt  # noqa: F401
    except ImportError:
        return {"available": False, "error": "ssqueezepy not installed"}

    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {
            "available": False,
            "note": "No EDF files found. Place .edf files under data/real_eeg/",
        }

    n_ch = min(data.shape[0], 10)  # cap at 10 channels for speed
    per_channel: List[Dict[str, Any]] = []
    all_band_energy: Dict[str, float] = {b: 0.0 for b in _BANDS}

    for i in range(n_ch):
        sig = data[i]

        # Downsample if very long (keep ≤4096 points for speed)
        if len(sig) > 4096:
            step = len(sig) // 4096
            sig = sig[::step]
            effective_sfreq = sfreq / step
        else:
            effective_sfreq = sfreq

        try:
            Twx, ssq_freqs = _compute_ssq_channel(sig, effective_sfreq)
        except Exception:
            continue

        # Energy = |Twx|^2
        energy_matrix = np.abs(Twx) ** 2

        # Marginal frequency spectrum: sum energy across time
        freq_power = np.sum(energy_matrix, axis=1)  # shape (n_freqs,)

        # Filter to valid positive frequencies ≤ Nyquist
        valid = (ssq_freqs > 0) & (ssq_freqs <= effective_sfreq / 2)
        if not valid.any():
            continue

        freqs_valid = ssq_freqs[valid]
        power_valid = freq_power[valid]

        # Peak frequency
        peak_idx = int(np.argmax(power_valid))
        peak_freq = float(freqs_valid[peak_idx])

        # Mean energy (use log10 scale for readability since raw SSQ energy is very small)
        mean_energy_raw = float(np.mean(power_valid))
        mean_energy = float(np.log10(mean_energy_raw + 1e-30))

        # Bandwidth: frequency spread (std of power-weighted frequencies)
        total_power = np.sum(power_valid)
        if total_power > 0:
            weighted_mean = np.sum(freqs_valid * power_valid) / total_power
            weighted_var = np.sum(
                power_valid * (freqs_valid - weighted_mean) ** 2
            ) / total_power
            bandwidth = float(np.sqrt(weighted_var))
        else:
            bandwidth = 0.0

        # Band energy breakdown for this channel
        ch_band = _band_energy(freqs_valid, power_valid)
        for b in _BANDS:
            all_band_energy[b] += ch_band[b]

        per_channel.append({
            "channel": ch_names[i],
            "peak_frequency_hz": round(peak_freq, 3),
            "mean_energy_log10": round(mean_energy, 2),
            "bandwidth_hz": round(bandwidth, 3),
            "dominant_band": _classify_band(peak_freq),
            "time_frequency_matrix_shape": list(Twx.shape),
        })

    if not per_channel:
        return {
            "available": False,
            "note": "SSQ-CWT computation failed for all channels.",
        }

    # Summary across channels
    peak_freqs = [ch["peak_frequency_hz"] for ch in per_channel]
    mean_energies = [ch["mean_energy_log10"] for ch in per_channel]

    summary = {
        "peak_frequency_hz": {
            "mean": round(float(np.mean(peak_freqs)), 3),
            "std": round(float(np.std(peak_freqs)), 3),
            "min": round(float(np.min(peak_freqs)), 3),
            "max": round(float(np.max(peak_freqs)), 3),
        },
        "mean_energy_log10": {
            "mean": round(float(np.mean(mean_energies)), 2),
            "std": round(float(np.std(mean_energies)), 2),
            "min": round(float(np.min(mean_energies)), 2),
            "max": round(float(np.max(mean_energies)), 2),
        },
    }

    # Normalize band energy to proportions
    total_band = sum(all_band_energy.values())
    if total_band > 0:
        frequency_bands = {
            b: {
                "energy": round(v, 6),
                "proportion": round(v / total_band, 4),
                "range_hz": list(_BANDS[b]),
            }
            for b, v in all_band_energy.items()
        }
    else:
        frequency_bands = {
            b: {"energy": 0.0, "proportion": 0.0, "range_hz": list(_BANDS[b])}
            for b in _BANDS
        }

    edf_name = (
        str(Path(file).name) if file else str(_find_edf().name)
    )

    return {
        "available": True,
        "file": edf_name,
        "sfreq": sfreq,
        "n_channels": len(per_channel),
        "seconds_analyzed": seconds,
        "transform_type": "cwt",
        "per_channel": per_channel,
        "summary": summary,
        "frequency_bands": frequency_bands,
        "note": f"ssqueezepy synchrosqueezing CWT on real EEG — {edf_name}",
    }


def synchrosqueezing_spectrum(
    file: Optional[str] = None, seconds: float = 10.0
) -> Dict[str, Any]:
    """
    Marginal frequency spectrum from the SSQ transform for each channel.
    Limited to 5 channels max to keep response size manageable.
    """
    try:
        from ssqueezepy import ssq_cwt  # noqa: F401
    except ImportError:
        return {"available": False, "error": "ssqueezepy not installed"}

    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {
            "available": False,
            "note": "No EDF files found. Place .edf files under data/real_eeg/",
        }

    n_ch = min(data.shape[0], 5)  # limit to 5 channels
    per_channel: List[Dict[str, Any]] = []

    for i in range(n_ch):
        sig = data[i]

        # Downsample if very long
        if len(sig) > 4096:
            step = len(sig) // 4096
            sig = sig[::step]
            effective_sfreq = sfreq / step
        else:
            effective_sfreq = sfreq

        try:
            Twx, ssq_freqs = _compute_ssq_channel(sig, effective_sfreq)
        except Exception:
            continue

        energy_matrix = np.abs(Twx) ** 2
        freq_power = np.sum(energy_matrix, axis=1)

        # Filter to valid positive frequencies ≤ Nyquist
        valid = (ssq_freqs > 0) & (ssq_freqs <= effective_sfreq / 2)
        if not valid.any():
            continue

        freqs_out = ssq_freqs[valid]
        power_out = freq_power[valid]

        # Subsample the spectrum to keep response size small (max 200 points)
        if len(freqs_out) > 200:
            indices = np.linspace(0, len(freqs_out) - 1, 200, dtype=int)
            freqs_out = freqs_out[indices]
            power_out = power_out[indices]

        per_channel.append({
            "channel": ch_names[i],
            "frequencies": [round(float(f), 3) for f in freqs_out],
            "power": [round(float(p), 6) for p in power_out],
        })

    if not per_channel:
        return {
            "available": False,
            "note": "SSQ spectrum computation failed for all channels.",
        }

    return {
        "available": True,
        "per_channel": per_channel,
        "note": "Marginal frequency spectrum from ssqueezepy SSQ-CWT.",
    }


def synchrosqueezing_definitions() -> Dict[str, Any]:
    """
    Definitions of synchrosqueezing, CWT, time-frequency analysis,
    and their clinical relevance to EEG.
    """
    return {
        "synchrosqueezing_transform": {
            "name": "Synchrosqueezing Transform (SST)",
            "description": (
                "A time-frequency reassignment method that sharpens a wavelet "
                "scalogram by squeezing energy towards instantaneous-frequency "
                "ridges. Unlike standard CWT, SST produces a concentrated "
                "representation where each oscillatory component appears as a "
                "thin curve rather than a broad smear, enabling reliable "
                "extraction of individual modes from multi-component signals."
            ),
            "key_property": (
                "Invertible — the original signal can be reconstructed from "
                "the synchrosqueezed representation, unlike many other "
                "reassignment methods."
            ),
            "reference": "Daubechies, Lu & Wu (2011). Applied & Computational Harmonic Analysis.",
        },
        "continuous_wavelet_transform": {
            "name": "Continuous Wavelet Transform (CWT)",
            "description": (
                "Decomposes a signal into time-frequency space by convolving "
                "with scaled and translated copies of a mother wavelet "
                "(typically Morlet for EEG). Provides multi-resolution analysis: "
                "good time resolution at high frequencies, good frequency "
                "resolution at low frequencies."
            ),
            "advantage_over_stft": (
                "Unlike the fixed-window STFT, CWT adapts its analysis window "
                "to each frequency scale, better capturing transient EEG events "
                "like epileptiform spikes (10–50 ms duration)."
            ),
        },
        "time_frequency_analysis": {
            "name": "Time-Frequency Analysis",
            "description": (
                "A family of methods that represent how the spectral content "
                "of a signal changes over time. Essential for non-stationary "
                "signals like EEG where brain rhythms start, stop, and shift "
                "frequency on sub-second timescales."
            ),
            "methods_hierarchy": [
                "STFT (Short-Time Fourier Transform) — simplest, fixed resolution",
                "CWT (Continuous Wavelet Transform) — multi-resolution",
                "SST (Synchrosqueezing Transform) — sharpened CWT",
                "Empirical Mode Decomposition (EMD) — data-driven, no basis",
            ],
        },
        "clinical_relevance": {
            "seizure_detection": (
                "SSQ-CWT resolves the onset of ictal rhythmic activity with "
                "higher temporal precision than FFT-based methods. Seizure "
                "epochs show concentrated energy in the delta-theta band "
                "(2–8 Hz) that SSQ separates from concurrent alpha/beta "
                "background."
            ),
            "sleep_staging": (
                "Sleep spindles (12–16 Hz, 0.5–1.5 s) and K-complexes are "
                "well-localized by SSQ. The reassigned spectrogram enables "
                "automatic spindle detection with fewer false positives than "
                "band-pass filtering."
            ),
            "event_related_oscillations": (
                "Motor-imagery BCI and cognitive neuroscience paradigms use "
                "SSQ to measure event-related synchronization (ERS) and "
                "desynchronization (ERD) in the mu (8–12 Hz) and beta "
                "(18–25 Hz) bands with single-trial precision."
            ),
            "artifact_identification": (
                "Muscle artifacts (>30 Hz, broadband) and eye blinks "
                "(0.5–3 Hz, frontal channels) appear as distinct time-frequency "
                "blobs in the synchrosqueezed representation, aiding automated "
                "artifact rejection pipelines."
            ),
        },
        "library": {
            "name": "ssqueezepy",
            "version": "0.6.6",
            "author": "John Muradeli",
            "url": "https://github.com/OverLordGoldDragon/ssqueezepy",
            "license": "MIT",
            "reference": "Muradeli (2020). ssqueezepy: Synchrosqueezing, wavelet transforms, and time-frequency analysis in Python.",
        },
    }


if __name__ == "__main__":
    print("=== Synchrosqueezing Overview ===")
    r = synchrosqueezing_overview(seconds=10)
    # Print overview without per_channel for readability
    print(json.dumps(
        {k: v for k, v in r.items() if k != "per_channel"},
        indent=2,
    ))
    if r.get("per_channel"):
        print(f"\nChannels computed: {len(r['per_channel'])}")
        print("First channel detail:")
        print(json.dumps(r["per_channel"][0], indent=2))

    print("\n=== Synchrosqueezing Spectrum (first channel) ===")
    s = synchrosqueezing_spectrum(seconds=10)
    if s.get("per_channel"):
        ch0 = s["per_channel"][0]
        print(f"Channel: {ch0['channel']}, "
              f"freq points: {len(ch0['frequencies'])}, "
              f"power points: {len(ch0['power'])}")
    else:
        print(json.dumps(s, indent=2))
