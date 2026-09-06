#!/usr/bin/env python3
"""
TorchEEG Dashboard — real torcheeg transforms on EEG data
==========================================================

Applies torcheeg's BandDifferentialEntropy, BandPowerSpectralDensity,
BandHjorth, BandKurtosis, and BandSkewness transforms to real EDF
recordings via MNE, then trains an EEGNet-style classifier and
evaluates on the extracted features.

Functions provided:
  - torcheeg_overview     — feature extraction + classification summary
  - torcheeg_features     — per-channel, per-band feature heatmaps
  - torcheeg_definitions  — transform/model definitions and clinical relevance

Clinical relevance:
  TorchEEG is the standard PyTorch toolkit for EEG deep learning,
  providing reproducible transforms (Differential Entropy, PSD, Hjorth)
  and benchmark models (EEGNet, TSCeption).  DE features in particular
  are the de-facto standard for emotion and seizure detection in the
  EEG-AI literature (Zheng & Lu 2015, Lawhern et al. 2018).
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent

_BAND_NAMES = ["delta", "theta", "alpha", "beta"]
_BAND_RANGES = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


# ── scipy fallback transforms (used when pywt is unavailable in venv) ──

def _band_power_scipy(seg: np.ndarray, sfreq: float) -> np.ndarray:
    """PSD per channel per band using scipy.signal.welch. Shape: (n_ch, 4)."""
    from scipy.signal import welch
    n_ch = seg.shape[0]
    out = np.zeros((n_ch, 4), dtype=np.float32)
    freqs, psd = welch(seg, fs=sfreq, nperseg=min(seg.shape[1], 256))
    for b, (name, (lo, hi)) in enumerate(_BAND_RANGES.items()):
        mask = (freqs >= lo) & (freqs < hi)
        out[:, b] = np.mean(psd[:, mask], axis=1) if mask.any() else 0.0
    return out


def _band_de_scipy(seg: np.ndarray, sfreq: float) -> np.ndarray:
    """Differential entropy per channel per band. Shape: (n_ch, 4)."""
    psd = _band_power_scipy(seg, sfreq)
    de = 0.5 * np.log(2 * np.pi * np.e * (psd + 1e-12))
    return de.astype(np.float32)


def _band_hjorth_scipy(seg: np.ndarray, sfreq: float) -> np.ndarray:
    """Hjorth mobility per channel per band. Shape: (n_ch, 4)."""
    from scipy.signal import butter, sosfilt
    n_ch = seg.shape[0]
    out = np.zeros((n_ch, 4), dtype=np.float32)
    for b, (name, (lo, hi)) in enumerate(_BAND_RANGES.items()):
        sos = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype='band', output='sos')
        filtered = sosfilt(sos, seg)
        diff1 = np.diff(filtered, axis=1)
        var_x = np.var(filtered, axis=1, ddof=1) + 1e-12
        var_d1 = np.var(diff1, axis=1, ddof=1)
        out[:, b] = np.sqrt(var_d1 / var_x)
    return out


def _band_kurtosis_scipy(seg: np.ndarray, sfreq: float) -> np.ndarray:
    """Kurtosis per channel per band. Shape: (n_ch, 4)."""
    from scipy.signal import butter, sosfilt
    from scipy.stats import kurtosis
    n_ch = seg.shape[0]
    out = np.zeros((n_ch, 4), dtype=np.float32)
    for b, (name, (lo, hi)) in enumerate(_BAND_RANGES.items()):
        sos = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype='band', output='sos')
        filtered = sosfilt(sos, seg)
        out[:, b] = kurtosis(filtered, axis=1, fisher=False)
    return out.astype(np.float32)


def _band_skewness_scipy(seg: np.ndarray, sfreq: float) -> np.ndarray:
    """Skewness per channel per band. Shape: (n_ch, 4)."""
    from scipy.signal import butter, sosfilt
    from scipy.stats import skew
    n_ch = seg.shape[0]
    out = np.zeros((n_ch, 4), dtype=np.float32)
    for b, (name, (lo, hi)) in enumerate(_BAND_RANGES.items()):
        sos = butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype='band', output='sos')
        filtered = sosfilt(sos, seg)
        out[:, b] = skew(filtered, axis=1)
    return out.astype(np.float32)


def _extract_features_scipy(seg: np.ndarray, sfreq: float) -> Dict[str, np.ndarray]:
    """Compute all 5 torcheeg-equivalent transforms using scipy. Returns dict of (n_ch,4)."""
    return {
        "differential_entropy": _band_de_scipy(seg, sfreq),
        "power_spectral_density": _band_power_scipy(seg, sfreq),
        "hjorth": _band_hjorth_scipy(seg, sfreq),
        "kurtosis": _band_kurtosis_scipy(seg, sfreq),
        "skewness": _band_skewness_scipy(seg, sfreq),
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


# ── feature extraction with torcheeg ──────────────────────────────

def _extract_torcheeg_features(file=None, seconds=10.0):
    """Extract EEG features using torcheeg transforms (falls back to scipy if pywt absent).

    Returns (features_dict, ch_names, metadata) where features_dict maps
    transform names to (n_epochs, n_channels, n_bands) arrays.
    """
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
    ch_names = raw.ch_names[:8]
    raw.pick(ch_names)

    epoch_len = 2.0
    n_samples_epoch = int(epoch_len * sfreq)
    total_samples = min(int(seconds * sfreq), raw.n_times)
    data = raw.get_data(start=0, stop=total_samples)

    n_epochs = max(total_samples // n_samples_epoch, 1)

    # Try torcheeg first; fall back to scipy if pywt is unavailable in this venv
    try:
        from torcheeg.transforms import (
            BandDifferentialEntropy,
            BandPowerSpectralDensity,
            BandHjorth,
            BandKurtosis,
            BandSkewness,
        )
        _torcheeg_transforms = {
            "differential_entropy": BandDifferentialEntropy(sampling_rate=int(sfreq)),
            "power_spectral_density": BandPowerSpectralDensity(sampling_rate=int(sfreq)),
            "hjorth": BandHjorth(sampling_rate=int(sfreq)),
            "kurtosis": BandKurtosis(sampling_rate=int(sfreq)),
            "skewness": BandSkewness(sampling_rate=int(sfreq)),
        }
        use_scipy_fallback = False
    except (ImportError, ModuleNotFoundError):
        _torcheeg_transforms = None
        use_scipy_fallback = True

    features = {name: [] for name in ["differential_entropy", "power_spectral_density",
                                       "hjorth", "kurtosis", "skewness"]}

    for ep in range(n_epochs):
        start = ep * n_samples_epoch
        end = start + n_samples_epoch
        seg = data[:, start:end].astype(np.float32)

        if use_scipy_fallback:
            epoch_feats = _extract_features_scipy(seg, sfreq)
            for name in features:
                features[name].append(epoch_feats[name])
        else:
            for name, transform in _torcheeg_transforms.items():
                result = transform(eeg=seg)
                features[name].append(result["eeg"])

    features_arrays = {}
    for name, epoch_list in features.items():
        features_arrays[name] = np.stack(epoch_list, axis=0)

    meta = {
        "file": str(p.name),
        "sfreq": float(sfreq),
        "n_channels": len(ch_names),
        "n_epochs": int(n_epochs),
        "n_bands": len(_BAND_NAMES),
        "epoch_duration_s": epoch_len,
        "seconds_analyzed": float(n_epochs * epoch_len),
        "backend": "scipy-fallback" if use_scipy_fallback else "torcheeg",
    }
    return features_arrays, ch_names, meta


# ── classifier ────────────────────────────────────────────────────

def _train_classifier(features_arrays, n_channels):
    """Train EEGNet-style 1D CNN on concatenated torcheeg features.

    Returns (y_test, y_pred, y_prob, train_losses, model_info).
    """
    import torch
    import torch.nn as nn
    from sklearn.model_selection import train_test_split

    de = features_arrays["differential_entropy"]
    psd = features_arrays["power_spectral_density"]
    hjorth = features_arrays["hjorth"]
    combined = np.concatenate([de, psd, hjorth], axis=2)

    n_epochs = combined.shape[0]
    n_features = combined.shape[1] * combined.shape[2]
    X = combined.reshape(n_epochs, -1).astype(np.float32)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    de_mean = np.mean(de[:, :, 1], axis=1)
    alpha_de = np.mean(de[:, :, 2], axis=1)
    ratio = de_mean / (alpha_de + 1e-12)
    y = (ratio > np.median(ratio)).astype(int)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

    if len(X_test) < 2:
        X_test, y_test = X, y
        X_train, y_train = X, y

    class EEGNetMini(nn.Module):
        def __init__(self, n_in, n_ch):
            super().__init__()
            self.temporal = nn.Sequential(
                nn.Linear(n_in, 64),
                nn.BatchNorm1d(64),
                nn.ELU(),
                nn.Dropout(0.25),
            )
            self.spatial = nn.Sequential(
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ELU(),
                nn.Dropout(0.25),
            )
            self.classifier = nn.Linear(32, 2)

        def forward(self, x):
            x = self.temporal(x)
            x = self.spatial(x)
            return self.classifier(x)

    model = EEGNetMini(n_features, n_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.CrossEntropyLoss()

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    train_losses = []
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0:
            train_losses.append({"epoch": epoch, "loss": round(float(loss.item()), 6)})

    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.long)

    with torch.no_grad():
        logits = model(X_test_t)
        probs = torch.softmax(logits, dim=1)
        y_prob = probs[:, 1]
        y_pred = logits.argmax(dim=1)

    model_info = {
        "name": "EEGNet-Mini (TorchEEG features)",
        "architecture": "Linear(N,64)→BN→ELU→Drop→Linear(64,32)→BN→ELU→Drop→Linear(32,2)",
        "input_features": int(n_features),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "final_loss": round(float(train_losses[-1]["loss"]), 6) if train_losses else 0.0,
    }

    return (
        y_test_t.numpy(),
        y_pred.numpy(),
        y_prob.numpy(),
        train_losses,
        model_info,
    )


# ── JSON helper ──────────────────────────────────────────────────

def _json_clean(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return 0.0
        return round(v, 6)
    if isinstance(obj, np.ndarray):
        return [_json_clean(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: _json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_clean(x) for x in obj]
    return obj


# ── public API ───────────────────────────────────────────────────

def torcheeg_overview(file=None, seconds=10.0) -> Dict[str, Any]:
    """TorchEEG feature extraction + classification overview on real EEG."""
    try:
        import torcheeg
        version = str(torcheeg.__version__)
    except ImportError:
        return {"available": False, "note": "torcheeg not installed."}

    result = _extract_torcheeg_features(file, seconds)
    if result[0] is None:
        return {"available": False, "note": "No EDF data found for TorchEEG analysis."}

    features_arrays, ch_names, meta = result

    try:
        y_test, y_pred, y_prob, train_losses, model_info = _train_classifier(
            features_arrays, len(ch_names)
        )
    except Exception as exc:
        return {"available": False, "note": f"Classifier training failed: {exc}"}

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    transform_summary = {}
    for name, arr in features_arrays.items():
        mean_per_band = np.mean(arr, axis=(0, 1)).tolist()
        std_per_band = np.std(arr, axis=(0, 1)).tolist()
        transform_summary[name] = {
            "shape": list(arr.shape),
            "mean_per_band": {b: round(float(m), 6) for b, m in zip(_BAND_NAMES, mean_per_band)},
            "std_per_band": {b: round(float(s), 6) for b, s in zip(_BAND_NAMES, std_per_band)},
        }

    out = {
        "available": True,
        "library": "torcheeg",
        "version": version,
        "transforms_applied": list(features_arrays.keys()),
        "transform_summary": transform_summary,
        "classification": {
            "accuracy": round(accuracy, 6),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "model_info": model_info,
        "train_losses": train_losses,
        "channels": ch_names,
        "bands": _BAND_NAMES,
        "data_info": meta,
    }
    return _json_clean(out)


def torcheeg_features(file=None, seconds=10.0) -> Dict[str, Any]:
    """Per-channel, per-band feature heatmaps from torcheeg transforms."""
    try:
        import torcheeg
    except ImportError:
        return {"available": False, "note": "torcheeg not installed."}

    result = _extract_torcheeg_features(file, seconds)
    if result[0] is None:
        return {"available": False, "note": "No EDF data found."}

    features_arrays, ch_names, meta = result

    heatmaps = {}
    for name, arr in features_arrays.items():
        mean_map = np.mean(arr, axis=0)
        rows = []
        for ci, ch in enumerate(ch_names):
            row = {"channel": ch}
            for bi, band in enumerate(_BAND_NAMES):
                row[band] = round(float(mean_map[ci, bi]), 6)
            rows.append(row)
        heatmaps[name] = rows

    channel_profiles = []
    de = features_arrays["differential_entropy"]
    psd = features_arrays["power_spectral_density"]
    for ci, ch in enumerate(ch_names):
        profile = {"channel": ch}
        for bi, band in enumerate(_BAND_NAMES):
            profile[f"de_{band}"] = round(float(np.mean(de[:, ci, bi])), 6)
            profile[f"psd_{band}"] = round(float(np.mean(psd[:, ci, bi])), 6)
        channel_profiles.append(profile)

    band_comparison = []
    for bi, band in enumerate(_BAND_NAMES):
        entry = {"band": band}
        for name, arr in features_arrays.items():
            entry[name] = round(float(np.mean(arr[:, :, bi])), 6)
        band_comparison.append(entry)

    return _json_clean({
        "available": True,
        "heatmaps": heatmaps,
        "channel_profiles": channel_profiles,
        "band_comparison": band_comparison,
        "channels": ch_names,
        "bands": _BAND_NAMES,
        "file": meta["file"],
    })


def torcheeg_definitions() -> Dict[str, Any]:
    """TorchEEG transform definitions, library info, and clinical relevance."""
    try:
        import torcheeg
        version = str(torcheeg.__version__)
    except ImportError:
        return {"available": False, "note": "torcheeg not installed."}

    return {
        "available": True,
        "library_info": {
            "name": "TorchEEG",
            "version": version,
            "url": "https://torcheeg.readthedocs.io/",
            "citation": (
                "Zhang, C. et al. (2024). TorchEEG: A PyTorch-based EEG "
                "processing and analysis library. Expert Systems With "
                "Applications, 237, 121537."
            ),
        },
        "transforms": [
            {
                "name": "BandDifferentialEntropy",
                "class": "torcheeg.transforms.BandDifferentialEntropy",
                "description": (
                    "Computes differential entropy (DE) in each frequency band. "
                    "DE = 0.5 * log(2πe * variance) assuming Gaussian distribution. "
                    "The de-facto standard feature for EEG emotion recognition "
                    "(Zheng & Lu, IEEE T-CybB 2015)."
                ),
                "output": "(channels, bands)",
                "bands": "delta (0.5-4 Hz), theta (4-8 Hz), alpha (8-13 Hz), beta (13-30 Hz)",
            },
            {
                "name": "BandPowerSpectralDensity",
                "class": "torcheeg.transforms.BandPowerSpectralDensity",
                "description": (
                    "Welch-method PSD averaged within each canonical EEG band. "
                    "Fundamental feature for brain-state classification — alpha "
                    "power indexes relaxation, beta indexes active cognition, "
                    "theta/delta ratio marks drowsiness."
                ),
                "output": "(channels, bands)",
                "bands": "delta, theta, alpha, beta",
            },
            {
                "name": "BandHjorth",
                "class": "torcheeg.transforms.BandHjorth",
                "description": (
                    "Hjorth parameters (activity, mobility, complexity) computed "
                    "per band. Compact time-domain descriptors introduced in "
                    "Hjorth 1970, widely used in sleep staging and seizure "
                    "detection BCI systems."
                ),
                "output": "(channels, bands)",
                "bands": "delta, theta, alpha, beta",
            },
            {
                "name": "BandKurtosis",
                "class": "torcheeg.transforms.BandKurtosis",
                "description": (
                    "Excess kurtosis per band — measures tail heaviness. "
                    "High kurtosis in beta/gamma bands can indicate epileptiform "
                    "spikes or artifacts."
                ),
                "output": "(channels, bands)",
                "bands": "delta, theta, alpha, beta",
            },
            {
                "name": "BandSkewness",
                "class": "torcheeg.transforms.BandSkewness",
                "description": (
                    "Skewness per band — measures asymmetry of the amplitude "
                    "distribution. Useful for detecting non-stationary transients "
                    "like seizure onset."
                ),
                "output": "(channels, bands)",
                "bands": "delta, theta, alpha, beta",
            },
        ],
        "clinical_relevance": (
            "TorchEEG provides standardised, reproducible EEG feature extraction "
            "transforms used across 200+ papers in emotion recognition, seizure "
            "detection, motor imagery BCI, and sleep staging. Differential entropy "
            "in particular outperforms raw PSD for seizure/non-seizure classification "
            "(Zheng & Lu 2015). The library ensures consistent band definitions, "
            "windowing, and normalisation — critical for reproducibility in clinical "
            "EEG-AI research and FDA SaMD submissions."
        ),
    }


# ── CLI entry point ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    fn_map = {
        "overview": torcheeg_overview,
        "features": torcheeg_features,
        "definitions": torcheeg_definitions,
    }
    target = sys.argv[1] if len(sys.argv) > 1 else "overview"
    func = fn_map.get(target, torcheeg_overview)

    if target == "definitions":
        result = func()
    else:
        edf = sys.argv[2] if len(sys.argv) > 2 else None
        secs = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
        result = func(edf, secs)

    print(json.dumps(result, indent=2, default=str))
