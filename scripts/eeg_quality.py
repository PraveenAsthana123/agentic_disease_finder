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
