"""EEG analysis pipeline — parse → features → model → analysis → report.

Single source of truth for "what happens when a user uploads EEG data".
Reuses the canonical 47-feature extractor (matches how the per-disease
models in models/<disease>_model.joblib were trained) so predictions are
valid, not fabricated.

Pipeline stages:
    1. parse_eeg(path)        EDF / NPZ / CSV  -> (data[ch, samples], sfreq, ch_names)
    2. analyze_signal(...)    duration, band powers, signal quality
    3. extract_features(...)  47-feature vector (first channel, 10s, normalized)
    4. classify(...)          load trained model bundle -> label + confidence
    5. build_report(...)      assemble a JSON report (persisted by db layer)
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).parent
MODELS_DIR = ROOT / "models"

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
}

# Canonical 47-feature order — must match scripts/create_real_samples.py.
FEATURE_NAMES = [
    "mean", "std", "var", "min", "max", "median", "ptp", "skewness", "kurtosis",
    "q25", "q75", "rms", "mav", "line_length", "zero_crossings",
    "delta_power", "theta_power", "alpha_power", "beta_power", "gamma_power",
    "total_power", "dominant_freq", "spectral_entropy", "psd_std", "psd_mean",
    "psd_median", "psd_q10", "psd_q90", "peak_ratio", "spectral_flatness",
    "spectral_centroid", "spectral_bandwidth", "spectral_rolloff",
    "mean_abs_diff", "std_diff", "max_diff", "hjorth_mobility", "hjorth_complexity",
    "autocorr", "slope_changes", "trend", "crest_factor",
    "approx_entropy", "sample_entropy", "hurst_exponent", "dfa_alpha", "lz_complexity",
]


def now_stamp() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# 1. PARSE
# ---------------------------------------------------------------------------
def parse_eeg(path: str | Path):
    """Return (data[n_channels, n_samples], sfreq, ch_names). Raises ValueError."""
    p = Path(path)
    ext = p.suffix.lower()

    if ext in (".edf", ".bdf"):
        import mne  # local import keeps API import light
        raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR") if ext == ".edf" \
            else mne.io.read_raw_bdf(str(p), preload=True, verbose="ERROR")
        return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)

    if ext in (".fif", ".fiff"):  # MNE-native — gold standard for processed EEG
        import mne
        raw = mne.io.read_raw_fif(str(p), preload=True, verbose="ERROR")
        return raw.get_data(), float(raw.info["sfreq"]), list(raw.ch_names)

    if ext == ".mat":  # MATLAB — common in EEG research datasets
        from scipy.io import loadmat
        m = loadmat(str(p))
        # find the largest 2-D numeric array (the signal matrix)
        cand = [(k, v) for k, v in m.items() if not k.startswith("__")
                and hasattr(v, "ndim") and getattr(v, "ndim", 0) == 2 and np.issubdtype(v.dtype, np.number)]
        if not cand:
            raise ValueError("No 2-D numeric signal array found in .mat")
        key, arr = max(cand, key=lambda kv: kv[1].size)
        arr = np.asarray(arr, dtype=float)
        if arr.shape[0] > arr.shape[1]:  # ensure channels x samples (channels < samples)
            arr = arr.T
        sf = 256.0
        for sk in ("fs", "sfreq", "Fs", "srate", "sampling_rate"):
            if sk in m:
                try:
                    sf = float(np.asarray(m[sk]).flatten()[0]); break
                except Exception:
                    pass
        return arr, sf, [f"ch{i}" for i in range(arr.shape[0])]

    if ext == ".npz":
        d = np.load(p)
        if "X" in d:  # feature matrix sample, not raw signal
            return None, None, None  # signaled to caller as feature-mode
        # else assume an array of raw signal
        key = list(d.keys())[0]
        arr = d[key]
        arr = arr if arr.ndim == 2 else arr.reshape(1, -1)
        return arr, 256.0, [f"ch{i}" for i in range(arr.shape[0])]

    if ext in (".csv", ".tsv", ".txt"):
        import pandas as pd
        sep = "\t" if ext == ".tsv" else ","
        df = pd.read_csv(p, sep=sep)
        num = df.select_dtypes("number")
        arr = num.to_numpy().T  # rows=samples -> channels x samples
        return arr, 256.0, list(num.columns)

    raise ValueError(f"Unsupported EEG format: {ext}")


# ---------------------------------------------------------------------------
# 2. SIGNAL ANALYSIS
# ---------------------------------------------------------------------------
def analyze_signal(data: np.ndarray, sfreq: float) -> dict:
    n_channels, n_samples = data.shape
    duration_s = round(n_samples / sfreq, 2)

    # Average band power across channels (relative).
    band_power = {b: 0.0 for b in BANDS}
    per_channel = []
    for ch in range(n_channels):
        x = data[ch]
        freqs, psd = sp_signal.welch(x, fs=sfreq, nperseg=min(256, len(x)))
        total = float(np.sum(psd)) + 1e-10
        for b, (lo, hi) in BANDS.items():
            idx = np.logical_and(freqs >= lo, freqs < hi)
            band_power[b] += float(np.sum(psd[idx])) / total
        per_channel.append({
            "channel": ch,
            "mean": round(float(np.mean(x)), 4),
            "std": round(float(np.std(x)), 4),
            "min": round(float(np.min(x)), 4),
            "max": round(float(np.max(x)), 4),
        })
    band_power = {b: round(v / n_channels, 4) for b, v in band_power.items()}

    # Simple signal-quality heuristic: flat or extreme channels -> lower score.
    flat = sum(1 for ch in range(n_channels) if np.std(data[ch]) < 1e-6)
    quality = "Good" if flat == 0 else ("Fair" if flat < n_channels * 0.2 else "Poor")

    return {
        "n_channels": int(n_channels),
        "n_samples": int(n_samples),
        "sampling_rate": float(sfreq),
        "duration_seconds": duration_s,
        "band_power_relative": band_power,
        "flat_channels": int(flat),
        "signal_quality": quality,
        "per_channel": per_channel[:16],  # cap payload
    }


# ---------------------------------------------------------------------------
# 3. FEATURES (canonical 47 — first channel, 10s segment, normalized)
# ---------------------------------------------------------------------------
def extract_features(data: np.ndarray, sfreq: float) -> np.ndarray:
    seg_len = int(min(data.shape[1], sfreq * 10))
    eeg = data[0, :seg_len].astype(float)
    eeg = (eeg - np.mean(eeg)) / (np.std(eeg) + 1e-10)
    f: dict[str, float] = {}

    f["mean"], f["std"], f["var"] = np.mean(eeg), np.std(eeg), np.var(eeg)
    f["min"], f["max"], f["median"] = np.min(eeg), np.max(eeg), np.median(eeg)
    f["ptp"], f["skewness"], f["kurtosis"] = np.ptp(eeg), skew(eeg), kurtosis(eeg)
    f["q25"], f["q75"] = np.percentile(eeg, 25), np.percentile(eeg, 75)
    f["rms"] = np.sqrt(np.mean(eeg ** 2))
    f["mav"] = np.mean(np.abs(eeg))
    f["line_length"] = np.sum(np.abs(np.diff(eeg)))
    f["zero_crossings"] = np.sum(np.diff(np.sign(eeg)) != 0)

    freqs, psd = sp_signal.welch(eeg, fs=sfreq, nperseg=min(256, len(eeg)))
    for b, (lo, hi) in BANDS.items():
        idx = np.logical_and(freqs >= lo, freqs < hi)
        f[f"{b}_power"] = float(np.sum(psd[idx])) if np.any(idx) else 0.0
    f["total_power"] = float(np.sum(psd))
    f["dominant_freq"] = float(freqs[np.argmax(psd)]) if len(psd) else 0.0
    psd_norm = psd / (np.sum(psd) + 1e-10)
    f["spectral_entropy"] = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-10)))
    f["psd_std"], f["psd_mean"], f["psd_median"] = np.std(psd), np.mean(psd), np.median(psd)
    f["psd_q10"], f["psd_q90"] = np.percentile(psd, 10), np.percentile(psd, 90)
    f["peak_ratio"] = np.max(psd) / (np.mean(psd) + 1e-10)
    f["spectral_flatness"] = np.exp(np.mean(np.log(psd + 1e-10))) / (np.mean(psd) + 1e-10)
    f["spectral_centroid"] = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
    f["spectral_bandwidth"] = np.sqrt(np.sum(((freqs - f["spectral_centroid"]) ** 2) * psd) / (np.sum(psd) + 1e-10))
    f["spectral_rolloff"] = float(freqs[np.searchsorted(np.cumsum(psd), 0.85 * np.sum(psd))]) if len(freqs) else 0.0

    diff = np.diff(eeg)
    f["mean_abs_diff"], f["std_diff"], f["max_diff"] = np.mean(np.abs(diff)), np.std(diff), np.max(np.abs(diff))
    var_s, var_d, var_d2 = np.var(eeg), np.var(diff), np.var(np.diff(diff))
    f["hjorth_mobility"] = np.sqrt(var_d / (var_s + 1e-10))
    f["hjorth_complexity"] = np.sqrt(var_d2 / (var_d + 1e-10)) / (f["hjorth_mobility"] + 1e-10)
    ac = np.correlate(eeg, eeg, mode="full")[len(eeg) - 1:]
    f["autocorr"] = ac[1] / (ac[0] + 1e-10) if len(ac) > 1 else 0.0
    f["slope_changes"] = np.sum(np.diff(np.sign(diff)) != 0)
    f["trend"] = np.polyfit(np.arange(len(eeg)), eeg, 1)[0] if len(eeg) > 1 else 0.0
    f["crest_factor"] = np.max(np.abs(eeg)) / (f["rms"] + 1e-10)

    f["approx_entropy"] = _approx_entropy(eeg)
    f["sample_entropy"] = _sample_entropy(eeg)
    f["hurst_exponent"] = _hurst(eeg)
    f["dfa_alpha"] = _dfa(eeg)
    f["lz_complexity"] = _lz(eeg)

    return np.array([float(f[name]) for name in FEATURE_NAMES], dtype=float)


def _approx_entropy(s, m=2, r=0.2):
    N = len(s); r *= np.std(s)
    def phi(m):
        x = np.array([s[i:i + m] for i in range(N - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1) / (N - m + 1)
        return np.sum(np.log(C + 1e-10)) / (N - m + 1)
    try:
        return float(phi(m) - phi(m + 1))
    except Exception:
        return 0.5


def _sample_entropy(s, m=2, r=0.2):
    N = len(s); r *= np.std(s)
    def cnt(m):
        x = np.array([s[i:i + m] for i in range(N - m)])
        d = np.max(np.abs(x[:, None] - x[None, :]), axis=2)
        return np.sum(d <= r) - (N - m)
    try:
        A, B = cnt(m + 1), cnt(m)
        return float(-np.log(A / (B + 1e-10) + 1e-10))
    except Exception:
        return 0.5


def _hurst(s):
    N = len(s)
    if N < 20:
        return 0.5
    try:
        max_k = min(N // 4, 100)
        ns = np.unique(np.logspace(1, np.log10(max_k), 10).astype(int))
        rs = []
        for n in ns:
            seg = N // n
            vals = []
            for i in range(seg):
                x = s[i * n:(i + 1) * n]
                c = np.cumsum(x - np.mean(x))
                R = np.max(c) - np.min(c); S = np.std(x)
                if S > 0:
                    vals.append(R / S)
            if vals:
                rs.append(np.mean(vals))
        if len(rs) > 2:
            return float(np.polyfit(np.log(ns[:len(rs)]), np.log(np.array(rs) + 1e-10), 1)[0])
    except Exception:
        pass
    return 0.5


def _dfa(s, scale_min=4):
    N = len(s)
    try:
        y = np.cumsum(s - np.mean(s))
        scales = np.unique(np.logspace(np.log10(scale_min), np.log10(N // 4), 10).astype(int))
        flucts = []
        for sc in scales:
            seg = N // sc
            if seg < 1:
                continue
            rms = []
            for i in range(seg):
                x = y[i * sc:(i + 1) * sc]
                t = np.arange(len(x))
                fit = np.polyval(np.polyfit(t, x, 1), t)
                rms.append(np.sqrt(np.mean((x - fit) ** 2)))
            if rms:
                flucts.append(np.mean(rms))
        if len(flucts) > 2:
            return float(np.polyfit(np.log(scales[:len(flucts)]), np.log(np.array(flucts) + 1e-10), 1)[0])
    except Exception:
        pass
    return 0.5


def _lz(s, threshold=None):
    try:
        b = (s > (threshold if threshold is not None else np.median(s))).astype(int)
        seq = "".join(map(str, b))
        i, c, ln = 0, 1, 1
        k, k_max = 1, 1
        n = len(seq)
        while True:
            if seq[i + k - 1] == seq[ln + k - 1]:
                k += 1
                if ln + k > n:
                    c += 1
                    break
            else:
                if k > k_max:
                    k_max = k
                i += 1
                if i == ln:
                    c += 1
                    ln += k_max
                    if ln + 1 > n:
                        break
                    i, k, k_max = 0, 1, 1
                else:
                    k = 1
        return float(c / (n / np.log2(n)) if n > 1 else 0.0)
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# 4. CLASSIFY (real trained model bundle)
# ---------------------------------------------------------------------------
def classify(features: np.ndarray, disease: str) -> dict:
    import joblib
    bundle_path = MODELS_DIR / f"{disease.lower()}_model.joblib"
    if not bundle_path.exists():
        return {"available": False, "reason": f"No trained model for '{disease}'"}

    bundle = joblib.load(bundle_path)
    model = bundle["model"]
    class_names = bundle.get("class_names", ["Control", disease.title()])
    n_expected = bundle.get("n_features", getattr(model, "n_features_in_", len(features)))

    if len(features) != n_expected:
        return {"available": False, "reason": f"Feature mismatch: got {len(features)}, model expects {n_expected}"}

    X = features.reshape(1, -1)
    # CRITICAL: the model was trained on scaler.fit_transform → selector.fit_transform
    # (see train_aggressive_final.py). Inference MUST apply the same transforms or
    # confidence is wrong (raw-input predictions are inflated). Apply if present.
    if bundle.get("scaler") is not None:
        X = bundle["scaler"].transform(X)
    if bundle.get("selector") is not None:
        X = bundle["selector"].transform(X)
    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist() if hasattr(model, "predict_proba") else None
    confidence = round(float(max(proba)), 4) if proba else None

    return {
        "available": True,
        "predicted_label": class_names[pred] if pred < len(class_names) else str(pred),
        "predicted_index": pred,
        "confidence": confidence,
        "class_probabilities": {class_names[i]: round(p, 4) for i, p in enumerate(proba)} if proba else None,
        "preprocessing_applied": [k for k in ("scaler", "selector") if bundle.get(k) is not None],
        "model_metrics": bundle.get("metrics", {}),
        "model_trained": bundle.get("training_date", "unknown"),
        "note": "Demonstrator model trained on reference feature samples. Validate before clinical use (see leakage caveat).",
    }


# ---------------------------------------------------------------------------
# 5. FULL PIPELINE
# ---------------------------------------------------------------------------
def run_pipeline(path: str | Path, disease: str, patient_id: Optional[str] = None) -> dict:
    """End-to-end: parse → analyze → features → classify → report dict."""
    data, sfreq, ch_names = parse_eeg(path)
    if data is None:
        return {"status": "error", "message": "NPZ is a feature matrix, not raw signal. Upload EDF/CSV raw EEG."}

    analysis = analyze_signal(data, sfreq)
    feats = extract_features(data, sfreq)
    prediction = classify(feats, disease)

    return {
        "status": "success",
        "generated_at": now_stamp(),
        "patient_id": patient_id,
        "disease": disease.lower(),
        "file": Path(path).name,
        "channels": ch_names[:32],
        "analysis": analysis,
        "features": {name: round(float(v), 5) for name, v in zip(FEATURE_NAMES, feats)},
        "prediction": prediction,
    }
