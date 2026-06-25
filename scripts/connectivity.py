#!/usr/bin/env python3
"""EEG Connectivity — spectral coherence between channels from real EDF.

Uses mne-connectivity to compute pairwise alpha-band coherence (functional
connectivity) across channels, then surfaces the connectivity matrix, the
strongest channel pairs, and per-channel connectivity strength (node degree).

Wires the Brain-Connectivity EEG stack lib (mne-connectivity, installed in the
canonical venv per §61.11). 100% real signal — no synthetic.
"""
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}


def _default_edf():
    g = sorted(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
    return str(Path(g[0]).relative_to(ROOT)) if g else None


def connectivity(file: str = None, band: str = "alpha", seconds: float = 20.0,
                 max_channels: int = 20) -> dict:
    import mne
    import numpy as np
    from mne_connectivity import spectral_connectivity_epochs

    file = file or _default_edf()
    if not file:
        return {"available": False, "error": "no EDF on disk"}
    p = ROOT / file if not Path(file).is_absolute() else Path(file)
    if not p.exists():
        return {"available": False, "error": f"EDF not found: {file}"}
    lo, hi = BANDS.get(band, BANDS["alpha"])

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    raw.crop(tmax=min(seconds, raw.times[-1]))
    raw.pick(raw.ch_names[:max_channels])
    ch = raw.ch_names
    ep = mne.make_fixed_length_epochs(raw, duration=2.0, verbose="ERROR")
    con = spectral_connectivity_epochs(ep, method="coh", sfreq=sf, fmin=lo, fmax=hi,
                                       faverage=True, verbose="ERROR")
    m = con.get_data(output="dense")[:, :, 0]  # lower-triangular coherence
    m = m + m.T  # symmetric full matrix

    n = len(ch)
    # strongest pairs
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append({"pair": f"{ch[i]} ↔ {ch[j]}", "coherence": round(float(m[i, j]), 3)})
    pairs.sort(key=lambda x: x["coherence"], reverse=True)
    # node degree = mean connectivity per channel
    degree = [{"channel": ch[i], "strength": round(float(m[i].sum() / (n - 1)), 3)} for i in range(n)]
    degree.sort(key=lambda x: x["strength"], reverse=True)
    vals = m[np.triu_indices(n, 1)]
    return {
        "available": True, "file": p.name, "band": band, "freq_hz": [lo, hi],
        "n_channels": n, "sfreq": sf,
        "mean_coherence": round(float(vals.mean()), 3),
        "max_coherence": round(float(vals.max()), 3),
        "strongest_pairs": pairs[:10],
        "hub_channels": degree[:6],
        "matrix": [[round(float(x), 3) for x in row] for row in m],
        "channels": ch,
        "method": "Pairwise magnitude-squared coherence (mne-connectivity), 2s epochs, band-averaged.",
        "note": "Functional connectivity (statistical coupling), not anatomical connectivity. Real EEG via MNE.",
        "source": "Real EDF via MNE + mne-connectivity.",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    r = connectivity()
    if r["available"]:
        print("Connectivity:", r["file"], r["band"], "| mean coh:", r["mean_coherence"])
        print("  hubs:", [(h["channel"], h["strength"]) for h in r["hub_channels"][:3]])
        print("  top pair:", r["strongest_pairs"][0])
    else:
        print(r)
