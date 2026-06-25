#!/usr/bin/env python3
"""Bad Channel Dashboard — per-channel EEG signal-quality QC from real EDF.

Classifies each channel good / flat / disconnected / noisy / line-noise using
amplitude (µV std + peak-to-peak), flatline ratio, and 50/60 Hz line-noise
relative power, computed via MNE + SciPy over a real recording window.

100% real (reads EDF signal) — report only. Runs under the canonical venv
(mne/scipy). Mirrors the Raw EEG Viewer source.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Thresholds (µV) — screening-grade, documented (not clinician-calibrated)
FLAT_STD_UV = 1.0        # std below this ⇒ flat / disconnected
NOISY_STD_UV = 500.0     # std above this ⇒ noisy / artifact-laden
LINE_REL_HI = 0.30       # >30% relative power at 50/60 Hz ⇒ line interference


def bad_channels(edf_path: str, seconds: float = 30.0) -> dict:
    import mne
    import numpy as np
    from scipy import signal as sps

    p = Path(edf_path)
    if not p.is_absolute():
        p = ROOT / edf_path
    if not p.exists():
        return {"available": False, "error": f"EDF not found: {edf_path}"}

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    raw.crop(tmin=0, tmax=min(seconds, raw.times[-1]))
    data = raw.get_data()  # volts
    ch_names = raw.ch_names
    uv = data * 1e6

    freqs, psd = sps.welch(data, fs=sf, nperseg=int(min(sf * 2, data.shape[1])))
    total_p = psd[:, freqs <= 70].sum(axis=1) + 1e-20
    # line-noise band: whichever of 50/60 has more energy in this recording
    def band_rel(lo, hi):
        m = (freqs >= lo) & (freqs < hi)
        return psd[:, m].sum(axis=1) / total_p
    line_rel = np.maximum(band_rel(48, 52), band_rel(58, 62))

    channels, verdict_count = [], {}
    for i, name in enumerate(ch_names):
        std_uv = float(uv[i].std())
        p2p_uv = float(uv[i].max() - uv[i].min())
        # flatline ratio: fraction of consecutive near-equal samples
        diff = np.abs(np.diff(uv[i]))
        flat_ratio = float((diff < 0.1).mean())
        lr = float(line_rel[i])
        if std_uv < FLAT_STD_UV or flat_ratio > 0.5:
            verdict = "disconnected" if std_uv < 0.2 else "flat"
        elif std_uv > NOISY_STD_UV:
            verdict = "noisy"
        elif lr > LINE_REL_HI:
            verdict = "line-noise"
        else:
            verdict = "good"
        verdict_count[verdict] = verdict_count.get(verdict, 0) + 1
        channels.append({"channel": name, "std_uv": round(std_uv, 1), "p2p_uv": round(p2p_uv, 1),
                         "flat_ratio": round(flat_ratio, 3), "line_noise_rel": round(lr, 3),
                         "verdict": verdict})
    bad = [c for c in channels if c["verdict"] != "good"]
    channels.sort(key=lambda c: (c["verdict"] == "good", c["channel"]))
    return {
        "available": True, "file": p.name, "sfreq": sf, "n_channels": len(ch_names),
        "seconds_analyzed": round(min(seconds, raw.times[-1]), 1),
        "verdict_distribution": verdict_count,
        "n_bad": len(bad), "bad_channels": bad,
        "channels": channels,
        "quality": "PASS" if not bad else "REVIEW",
        "thresholds": {"flat_std_uv": FLAT_STD_UV, "noisy_std_uv": NOISY_STD_UV,
                       "line_noise_rel": LINE_REL_HI},
        "note": ("Screening-grade channel QC (amplitude + flatline + line-noise). "
                 "Real EDF via MNE/SciPy. Not a substitute for clinician channel review."),
    }


if __name__ == "__main__":
    import glob
    f = sorted(glob.glob("data/real_eeg/epilepsy_physionet/chb*.edf"))[0]
    r = bad_channels(f)
    print("Bad channel QC:", r["verdict_distribution"], "| quality:", r["quality"])
    for c in r["bad_channels"][:6]:
        print(f"  {c['channel']}: {c['verdict']} (std={c['std_uv']}µV line={c['line_noise_rel']})")


def artifact_review(edf_path: str, seconds: float = 60.0, window_s: float = 2.0) -> dict:
    """Window-based artifact detection from real EDF — eye-blink (frontal transients),
    muscle/EMG (high-freq power), line-noise (50/60 Hz), and movement (broadband
    high amplitude). Returns per-window flags + clean-data percentage."""
    import mne
    import numpy as np
    from scipy import signal as sps

    p = Path(edf_path)
    if not p.is_absolute():
        p = ROOT / edf_path
    if not p.exists():
        return {"available": False, "error": f"EDF not found: {edf_path}"}

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    raw.crop(tmin=0, tmax=min(seconds, raw.times[-1]))
    uv = raw.get_data() * 1e6
    ch = raw.ch_names
    frontal = [i for i, c in enumerate(ch) if "FP" in c.upper() or "FT" in c.upper()]
    win = int(window_s * sf)
    n_win = max(1, uv.shape[1] // win)

    types = {"eye_blink": 0, "muscle": 0, "line_noise": 0, "movement": 0}
    windows = []
    for w in range(n_win):
        seg = uv[:, w * win:(w + 1) * win]
        if seg.shape[1] < win // 2:
            break
        flags = []
        # eye-blink: large low-freq deflection in frontal channels
        if frontal:
            fseg = seg[frontal]
            if float(np.abs(fseg).max()) > 150 and float(fseg.std()) > 40:
                flags.append("eye_blink")
        # muscle: high-freq (>30Hz) power ratio elevated
        f, ps = sps.welch(seg, fs=sf, nperseg=int(min(sf, seg.shape[1])))
        hf = ps[:, (f >= 30) & (f < 70)].sum(axis=1)
        lf = ps[:, (f >= 1) & (f < 30)].sum(axis=1) + 1e-20
        if float((hf / lf).max()) > 1.0:
            flags.append("muscle")
        # line noise: 50/60 Hz dominant
        ln = np.maximum(ps[:, (f >= 48) & (f < 52)].sum(axis=1),
                        ps[:, (f >= 58) & (f < 62)].sum(axis=1))
        tot = ps[:, f <= 70].sum(axis=1) + 1e-20
        if float((ln / tot).max()) > 0.30:
            flags.append("line_noise")
        # movement: broadband high amplitude across many channels
        if float(np.median(np.abs(seg).max(axis=1))) > 300:
            flags.append("movement")
        for fl in flags:
            types[fl] += 1
        windows.append({"window": w, "start_s": round(w * window_s, 1),
                        "artifacts": flags, "clean": not flags})

    clean = sum(1 for w in windows if w["clean"])
    return {
        "available": True, "file": p.name, "sfreq": sf, "n_channels": len(ch),
        "window_s": window_s, "n_windows": len(windows),
        "clean_windows": clean,
        "clean_pct": round(100 * clean / len(windows), 1) if windows else 0.0,
        "artifact_type_counts": types,
        "windows": windows[:60],
        "frontal_channels": [ch[i] for i in frontal],
        "quality": "PASS" if windows and clean / len(windows) >= 0.8 else "REVIEW",
        "note": ("Screening-grade artifact detection (eye-blink/muscle/line-noise/movement) "
                 "over real EDF windows. Not ICA-based clinician artifact rejection."),
    }
