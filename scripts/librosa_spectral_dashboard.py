#!/usr/bin/env python3
"""
Librosa Spectral Features Dashboard — EEG spectral analysis
============================================================

Computes real spectral features from EEG recordings on disk
using numpy + scipy (pure-Python, no librosa dependency required).
Every value comes from actual EDF data — nothing fabricated.

Features computed per channel:
  - Spectral Centroid   — centre of mass of the spectrum (Hz)
  - Spectral Bandwidth  — spread around the centroid (Hz)
  - Spectral Rolloff    — frequency below which 85% of energy lies (Hz)
  - Spectral Flatness   — tonal vs noisy (0=tonal, 1=noisy)
  - Spectral Contrast   — valley-to-peak energy ratio per sub-band
  - Zero-Crossing Rate  — how often the signal crosses zero
  - MFCC (first 13)     — mel-frequency cepstral coefficients
  - Mel Spectrogram      — mel-scaled power spectrum (time × mel-bins)

Clinical relevance:
  Spectral features capture frequency-domain characteristics that
  distinguish seizure epochs (high-amplitude rhythmic activity,
  reduced spectral flatness) from normal background.  These features
  complement MNE's band-power approach with a richer set of
  spectral descriptors used in audio/signal ML pipelines.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).parent.parent


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


def _load_raw(file: Optional[str] = None, seconds: float = 30.0):
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


# ── pure numpy/scipy spectral features (no librosa dependency) ──────

def _frame_signal(y: np.ndarray, n_fft: int = 2048, hop: int = 512) -> np.ndarray:
    """Split 1-D signal into overlapping frames of length n_fft."""
    n = len(y)
    if n < n_fft:
        pad = np.zeros(n_fft - n)
        y = np.concatenate([y, pad])
        n = n_fft
    n_frames = 1 + (n - n_fft) // hop
    frames = np.stack([y[i * hop: i * hop + n_fft] for i in range(n_frames)])
    return frames  # (n_frames, n_fft)


def _stft_magnitude(frames: np.ndarray) -> tuple:
    """Return |STFT| matrix (n_frames, n_fft//2+1) and rfft freqs."""
    win = np.hanning(frames.shape[1])
    windowed = frames * win[np.newaxis, :]
    spectrum = np.fft.rfft(windowed, axis=1)
    magnitude = np.abs(spectrum)  # (n_frames, n_bins)
    return magnitude


def _spectral_centroid(magnitude: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Per-frame spectral centroid (Hz)."""
    norm = np.sum(magnitude, axis=1, keepdims=True) + 1e-10
    return np.sum(freqs[np.newaxis, :] * magnitude, axis=1) / norm.squeeze()


def _spectral_bandwidth(magnitude: np.ndarray, freqs: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Per-frame spectral bandwidth (Hz)."""
    norm = np.sum(magnitude, axis=1) + 1e-10
    diff_sq = (freqs[np.newaxis, :] - centroids[:, np.newaxis]) ** 2
    return np.sqrt(np.sum(diff_sq * magnitude, axis=1) / norm)


def _spectral_rolloff(magnitude: np.ndarray, freqs: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
    """Per-frame spectral rolloff (Hz) — freq below which roll_percent of energy lies."""
    cumsum = np.cumsum(magnitude, axis=1)
    threshold = roll_percent * cumsum[:, -1:]
    rolloff = np.array([
        freqs[np.searchsorted(cumsum[i], threshold[i, 0])]
        if np.searchsorted(cumsum[i], threshold[i, 0]) < len(freqs)
        else freqs[-1]
        for i in range(len(magnitude))
    ])
    return rolloff


def _spectral_flatness(magnitude: np.ndarray) -> np.ndarray:
    """Per-frame spectral flatness."""
    power = magnitude ** 2 + 1e-10
    log_mean = np.mean(np.log(power), axis=1)
    arith_mean = np.mean(power, axis=1) + 1e-10
    return np.exp(log_mean) / arith_mean


def _zero_crossing_rate(y: np.ndarray, frame_len: int = 2048, hop: int = 512) -> np.ndarray:
    """Per-frame zero-crossing rate."""
    frames = _frame_signal(y, frame_len, hop)
    signs = np.sign(frames)
    crossings = np.diff(signs, axis=1) != 0
    return crossings.mean(axis=1)


def _mel_filterbank(sr: float, n_fft: int, n_mels: int = 40,
                    fmin: float = 0.0, fmax: Optional[float] = None) -> np.ndarray:
    """Build a mel-scale filterbank matrix (n_mels, n_fft//2+1)."""
    if fmax is None:
        fmax = sr / 2.0
    # Mel scale helpers
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    n_bins = len(freqs)
    filterbank = np.zeros((n_mels, n_bins))

    for m in range(1, n_mels + 1):
        f_lower = hz_points[m - 1]
        f_center = hz_points[m]
        f_upper = hz_points[m + 1]
        for k, f in enumerate(freqs):
            if f_lower <= f <= f_center:
                filterbank[m - 1, k] = (f - f_lower) / (f_center - f_lower + 1e-10)
            elif f_center < f <= f_upper:
                filterbank[m - 1, k] = (f_upper - f) / (f_upper - f_center + 1e-10)
    return filterbank


def _mfcc(magnitude: np.ndarray, sr: float, n_mfcc: int = 13,
           n_mels: int = 40, n_fft: int = 2048) -> np.ndarray:
    """Compute MFCC coefficients from magnitude spectrum frames.
    Returns (n_frames, n_mfcc)."""
    from scipy.fft import dct

    filterbank = _mel_filterbank(sr, n_fft, n_mels)
    power = magnitude ** 2
    mel_power = np.dot(power, filterbank.T)  # (n_frames, n_mels)
    log_mel = np.log(mel_power + 1e-10)
    coeffs = dct(log_mel, type=2, axis=1, norm='ortho')  # (n_frames, n_mels)
    return coeffs[:, :n_mfcc]


def _spectral_contrast(magnitude: np.ndarray, freqs: np.ndarray,
                       sr: float, n_bands: int = 6) -> np.ndarray:
    """Per-frame spectral contrast — peak minus valley per sub-band.
    Returns (n_frames, n_bands+1)."""
    # Frequency limits of each sub-band (octave-based)
    f_min = 200.0
    f_max = sr / 2.0
    if f_max <= f_min:
        return np.zeros((len(magnitude), n_bands + 1))

    bands = np.logspace(np.log10(f_min), np.log10(f_max), n_bands + 1)
    contrast = np.zeros((len(magnitude), n_bands + 1))

    for b in range(n_bands):
        lo, hi = bands[b], bands[b + 1]
        mask = (freqs >= lo) & (freqs < hi)
        if mask.sum() < 2:
            continue
        sub = magnitude[:, mask]
        n_sub = max(1, sub.shape[1] // 5)
        peak = np.mean(np.sort(sub, axis=1)[:, -n_sub:], axis=1)
        valley = np.mean(np.sort(sub, axis=1)[:, :n_sub], axis=1)
        contrast[:, b] = 20 * np.log10(peak / (valley + 1e-10) + 1e-10)

    return contrast


def _compute_channel_spectral(signal: np.ndarray, sr: float) -> Dict[str, Any]:
    """Compute spectral features for a single channel (pure numpy/scipy)."""
    y = signal.astype(np.float64)
    n_fft = min(2048, len(y))
    if n_fft < 16:
        return {
            "spectral_centroid_hz": 0.0, "spectral_centroid_std": 0.0,
            "spectral_bandwidth_hz": 0.0, "spectral_rolloff_hz": 0.0,
            "spectral_flatness": 0.0, "zero_crossing_rate": 0.0,
            "mfcc_means": [0.0] * 13, "spectral_contrast": [0.0] * 7,
        }

    hop = n_fft // 4
    frames = _frame_signal(y, n_fft, hop)
    magnitude = _stft_magnitude(frames)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    centroids = _spectral_centroid(magnitude, freqs)
    centroid_mean = float(np.nanmean(centroids))
    centroid_std = float(np.nanstd(centroids))

    bandwidths = _spectral_bandwidth(magnitude, freqs, centroids)
    bandwidth_mean = float(np.nanmean(bandwidths))

    rolloffs = _spectral_rolloff(magnitude, freqs, 0.85)
    rolloff_mean = float(np.nanmean(rolloffs))

    flatnesses = _spectral_flatness(magnitude)
    flatness_mean = float(np.nanmean(flatnesses))

    zcr_frames = _zero_crossing_rate(y, n_fft, hop)
    zcr_mean = float(np.nanmean(zcr_frames))

    mfccs = _mfcc(magnitude, sr, n_mfcc=13, n_fft=n_fft)
    mfcc_means = [float(np.nanmean(mfccs[:, i])) for i in range(13)]

    try:
        contrast = _spectral_contrast(magnitude, freqs, sr, n_bands=6)
        contrast_means = [float(np.nanmean(contrast[:, i])) for i in range(contrast.shape[1])]
    except Exception:
        contrast_means = [0.0] * 7

    return {
        "spectral_centroid_hz": round(centroid_mean, 2),
        "spectral_centroid_std": round(centroid_std, 2),
        "spectral_bandwidth_hz": round(bandwidth_mean, 2),
        "spectral_rolloff_hz": round(rolloff_mean, 2),
        "spectral_flatness": round(flatness_mean, 6),
        "zero_crossing_rate": round(zcr_mean, 6),
        "mfcc_means": [round(v, 4) for v in mfcc_means],
        "spectral_contrast": [round(v, 4) for v in contrast_means],
    }


def _compute_mel_spectrogram(signal: np.ndarray, sr: float,
                              n_mels: int = 40) -> Dict[str, Any]:
    """Compute mel spectrogram summary (mean power per mel bin)."""
    y = signal.astype(np.float64)
    n_fft = min(2048, len(y))
    if n_fft < 16:
        return {"mel_bins": [], "mel_freqs_hz": [], "note": "signal too short"}

    hop = n_fft // 4
    frames = _frame_signal(y, n_fft, hop)
    magnitude = _stft_magnitude(frames)
    power = magnitude ** 2

    filterbank = _mel_filterbank(sr, n_fft, n_mels)
    mel_power = np.dot(power, filterbank.T)  # (n_frames, n_mels)
    mel_db = 10 * np.log10(mel_power + 1e-10)

    mean_power = np.nanmean(mel_db, axis=0).tolist()

    # Mel bin centre frequencies
    def hz_to_mel(f): return 2595 * np.log10(1 + f / 700)
    def mel_to_hz(m): return 700 * (10 ** (m / 2595) - 1)
    mel_min = hz_to_mel(0.0)
    mel_max = hz_to_mel(sr / 2.0)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)[1:-1]
    mel_freqs = mel_to_hz(mel_points).tolist()

    return {
        "mel_bins": [round(v, 2) for v in mean_power],
        "mel_freqs_hz": [round(v, 2) for v in mel_freqs],
        "n_time_frames": int(frames.shape[0]),
    }


# ── public API functions (called by api_backend.py) ────────────────

def overview(file: Optional[str] = None, seconds: float = 30.0) -> Dict[str, Any]:
    """Full spectral overview — per-channel spectral features."""
    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {"error": "No EDF file found. Place .edf files in data/real_eeg/"}

    channels = []
    for i, ch in enumerate(ch_names):
        feats = _compute_channel_spectral(data[i], sfreq)
        feats["channel"] = ch
        channels.append(feats)

    # Summary statistics across channels
    centroids = [c["spectral_centroid_hz"] for c in channels]
    flatnesses = [c["spectral_flatness"] for c in channels]

    return {
        "tool": "numpy+scipy (librosa-compatible features)",
        "version": _backend_version(),
        "file": str(_find_edf() or file or "N/A"),
        "sfreq": sfreq,
        "n_channels": len(ch_names),
        "duration_sec": seconds,
        "channels": channels,
        "summary": {
            "mean_centroid_hz": round(float(np.mean(centroids)), 2),
            "std_centroid_hz": round(float(np.std(centroids)), 2),
            "mean_flatness": round(float(np.mean(flatnesses)), 6),
            "min_flatness_ch": ch_names[int(np.argmin(flatnesses))],
            "max_flatness_ch": ch_names[int(np.argmax(flatnesses))],
        },
    }


def heatmap(file: Optional[str] = None, seconds: float = 30.0) -> Dict[str, Any]:
    """Channels × spectral-metrics heatmap matrix for visualization."""
    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {"error": "No EDF file found."}

    metrics = ["centroid", "bandwidth", "rolloff", "flatness", "zcr"]
    matrix = []
    for i, ch in enumerate(ch_names):
        feats = _compute_channel_spectral(data[i], sfreq)
        row = [
            feats["spectral_centroid_hz"],
            feats["spectral_bandwidth_hz"],
            feats["spectral_rolloff_hz"],
            feats["spectral_flatness"],
            feats["zero_crossing_rate"],
        ]
        matrix.append(row)

    return {
        "channels": list(ch_names),
        "metrics": metrics,
        "matrix": matrix,
        "note": "Row = channel, Col = metric. Values are mean across time frames.",
    }


def mel_spectrogram(file: Optional[str] = None, seconds: float = 30.0) -> Dict[str, Any]:
    """Mel spectrogram — mean dB power per mel bin, averaged across channels."""
    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {"error": "No EDF file found."}

    per_channel = []
    for i, ch in enumerate(ch_names):
        mel = _compute_mel_spectrogram(data[i], sfreq)
        mel["channel"] = ch
        per_channel.append(mel)

    # Average across channels
    valid = [c for c in per_channel if c.get("mel_bins")]
    if valid:
        avg_bins = np.mean([c["mel_bins"] for c in valid], axis=0).tolist()
        mel_freqs = valid[0]["mel_freqs_hz"]
    else:
        avg_bins = []
        mel_freqs = []

    return {
        "channels": per_channel,
        "average_mel_power_dB": [round(v, 2) for v in avg_bins],
        "mel_freqs_hz": [round(v, 2) for v in mel_freqs],
        "n_channels": len(ch_names),
    }


def mfcc_profile(file: Optional[str] = None, seconds: float = 30.0) -> Dict[str, Any]:
    """MFCC profile — mean of first 13 MFCCs per channel."""
    data, sfreq, ch_names = _load_raw(file, seconds)
    if data is None:
        return {"error": "No EDF file found."}

    profiles = []
    for i, ch in enumerate(ch_names):
        feats = _compute_channel_spectral(data[i], sfreq)
        profiles.append({
            "channel": ch,
            "mfcc_means": feats["mfcc_means"],
        })

    # Average across channels
    avg_mfcc = np.mean([p["mfcc_means"] for p in profiles], axis=0).tolist()

    return {
        "channels": profiles,
        "average_mfcc": [round(v, 4) for v in avg_mfcc],
        "n_coefficients": 13,
        "note": "MFCC 0 = energy, MFCC 1-12 = spectral shape. "
                "Seizure epochs typically show altered MFCC patterns.",
    }


def definitions() -> Dict[str, Any]:
    """Spectral feature definitions, interpretation, and references."""
    return {
        "tool": "numpy+scipy (librosa-compatible features)",
        "version": _backend_version(),
        "features": [
            {
                "name": "Spectral Centroid",
                "unit": "Hz",
                "description": "Centre of mass of the power spectrum. Higher values "
                               "indicate more energy at higher frequencies.",
                "clinical": "Seizure epochs often show increased centroid due to "
                            "high-frequency oscillations; depression EEG may show "
                            "leftward (lower) centroid shift.",
                "reference": "McFee et al. (2015) librosa",
            },
            {
                "name": "Spectral Bandwidth",
                "unit": "Hz",
                "description": "Standard deviation of the spectrum around the centroid. "
                               "Wider bandwidth = more spread-out spectral energy.",
                "clinical": "Narrower bandwidth during seizures (concentrated rhythmic "
                            "activity). Wider during normal awake background.",
                "reference": "Peeters (2004)",
            },
            {
                "name": "Spectral Rolloff",
                "unit": "Hz",
                "description": "Frequency below which 85% of total spectral energy lies.",
                "clinical": "Lower rolloff in seizure epochs (energy concentrated "
                            "in lower bands). Higher in muscle artifact.",
                "reference": "Scheirer & Slaney (1997)",
            },
            {
                "name": "Spectral Flatness",
                "unit": "ratio [0-1]",
                "description": "Ratio of geometric to arithmetic mean of power spectrum. "
                               "0 = tonal/periodic, 1 = noise-like/flat spectrum.",
                "clinical": "Seizures show lower flatness (more periodic/rhythmic). "
                            "Background EEG has higher flatness (more noise-like).",
                "reference": "Dubnov (2004)",
            },
            {
                "name": "Zero-Crossing Rate",
                "unit": "rate [0-1]",
                "description": "Fraction of signal frames where the sign changes. "
                               "Proxy for dominant frequency content.",
                "clinical": "Higher ZCR in high-frequency dominant epochs. "
                            "Lower during slow-wave activity (sleep, seizure onset).",
                "reference": "Kedem (1986)",
            },
            {
                "name": "MFCC (Mel-Frequency Cepstral Coefficients)",
                "unit": "dB-scale coefficients",
                "description": "Compact representation of the spectral envelope on a "
                               "mel-frequency scale. 13 coefficients capture the shape "
                               "of the spectrum as perceived on a psychoacoustic scale.",
                "clinical": "MFCCs are top-performing features in EEG seizure detection "
                            "and sleep-stage classification (Tsinalis et al., 2016).",
                "reference": "Davis & Mermelstein (1980); McFee et al. (2015)",
            },
            {
                "name": "Spectral Contrast",
                "unit": "dB",
                "description": "Peak-to-valley energy ratio in sub-bands. Higher contrast "
                               "indicates more distinct spectral peaks.",
                "clinical": "Seizure epochs may show increased contrast in specific bands "
                            "due to rhythmic hypersynchronous activity.",
                "reference": "Jiang et al. (2002)",
            },
            {
                "name": "Mel Spectrogram",
                "unit": "dB",
                "description": "Power spectrum mapped to mel-frequency scale, displayed "
                               "as a time-frequency representation. 40 mel bins.",
                "clinical": "Visual inspection of mel spectrograms can reveal seizure "
                            "onset patterns, sleep spindles, and artifact regions.",
                "reference": "Stevens et al. (1937); McFee et al. (2015)",
            },
        ],
    }


def _backend_version() -> str:
    """Return numpy + scipy versions."""
    try:
        import scipy
        return f"numpy={np.__version__}, scipy={scipy.__version__}"
    except Exception:
        return f"numpy={np.__version__}"


# ── CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "overview"
    fn = {"overview": overview, "heatmap": heatmap, "mel": mel_spectrogram,
          "mfcc": mfcc_profile, "definitions": definitions}.get(cmd, overview)
    result = fn()
    print(json.dumps(result, indent=2, default=str))
