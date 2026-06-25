#!/usr/bin/env python3
"""Montage Comparison Dashboard — referential vs re-referenced EEG from real EDF.

Shows how the SAME recording looks under different montages: original
(linked-ear referential), Common Average Reference (CAR), and a bipolar
longitudinal derivation. Contrasts per-band relative power + signal amplitude
so the clinician sees what each montage emphasises.

100% real (monopolar EDF via MNE) — runs under the canonical venv.
"""
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def _bandpower(data, sf):
    import numpy as np
    from scipy import signal as sps
    f, ps = sps.welch(data, fs=sf, nperseg=int(min(sf * 2, data.shape[1])))
    tot = ps[:, f <= 45].sum() + 1e-20
    return {b: round(float(ps[:, (f >= lo) & (f < hi)].sum() / tot), 4) for b, (lo, hi) in BANDS.items()}


def _default_monopolar():
    g = sorted(glob.glob(str(ROOT / "data" / "real_eeg" / "depression_figshare" / "*.edf")))
    return str(Path(g[0]).relative_to(ROOT)) if g else None


def compare(file: str = None) -> dict:
    import mne
    import numpy as np

    file = file or _default_monopolar()
    if not file:
        return {"available": False, "error": "no monopolar EDF for montage comparison"}
    p = ROOT / file if not Path(file).is_absolute() else Path(file)
    if not p.exists():
        return {"available": False, "error": f"EDF not found: {file}"}

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    raw.crop(tmin=0, tmax=min(30, raw.times[-1]))
    montages = {}

    # 1. Original referential (as recorded)
    d0 = raw.get_data()
    montages["referential_original"] = {
        "description": "As recorded (linked-ear / common reference)",
        "n_channels": len(raw.ch_names),
        "mean_amplitude_uv": round(float(np.abs(d0).mean() * 1e6), 2),
        "band_power": _bandpower(d0, sf),
    }
    # 2. Common Average Reference
    car = raw.copy().set_eeg_reference("average", verbose="ERROR")
    dc = car.get_data()
    montages["common_average"] = {
        "description": "Common Average Reference (CAR) — re-referenced to mean of all channels",
        "n_channels": len(car.ch_names),
        "mean_amplitude_uv": round(float(np.abs(dc).mean() * 1e6), 2),
        "band_power": _bandpower(dc, sf),
    }
    # 3. Bipolar longitudinal (sequential channel differences)
    chs = raw.ch_names
    bip = d0[:-1] - d0[1:]
    montages["bipolar_longitudinal"] = {
        "description": "Bipolar longitudinal — sequential channel differences (e.g. Fp1-F3)",
        "n_channels": bip.shape[0],
        "mean_amplitude_uv": round(float(np.abs(bip).mean() * 1e6), 2),
        "band_power": _bandpower(bip, sf),
        "example_derivations": [f"{re.sub(r'^EEG ','',chs[i])} − {re.sub(r'^EEG ','',chs[i+1])}"
                                for i in range(min(4, len(chs) - 1))],
    }

    # contrast: how much does the montage change relative band power?
    ref_bp = montages["referential_original"]["band_power"]
    deltas = {m: {b: round(montages[m]["band_power"][b] - ref_bp[b], 4) for b in BANDS}
              for m in ("common_average", "bipolar_longitudinal")}
    return {
        "available": True, "file": p.name, "sfreq": sf, "seconds": round(min(30, raw.times[-1]), 1),
        "montages": montages,
        "band_power_delta_vs_referential": deltas,
        "note": ("Same real recording under 3 montages. CAR removes common-mode signal; "
                 "bipolar emphasises local gradients (better for focal/lateralised activity). "
                 "Illustrative — montage choice is clinician-driven."),
        "source": "Real monopolar EDF via MNE-Python.",
    }


if __name__ == "__main__":
    r = compare()
    if r["available"]:
        print("Montage comparison:", r["file"])
        for m, v in r["montages"].items():
            print(f"  {m}: {v['n_channels']}ch amp={v['mean_amplitude_uv']}µV bp={v['band_power']}")
    else:
        print(r)
