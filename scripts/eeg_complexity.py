#!/usr/bin/env python3
"""EEG Complexity / Entropy features — AntroPy + Nolds on real EDF.

Per-channel nonlinear & entropy measures used in seizure / depression EEG
research: spectral entropy, permutation entropy, sample entropy (AntroPy),
and detrended fluctuation analysis / Hurst exponent (Nolds). Computed over a
real recording window via MNE.

Wires EEG AI stack libs AntroPy + Nolds (installed in the canonical venv per
§61.11). 100% real signal — no synthetic.
"""
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _default_edf():
    g = sorted(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
    return str(Path(g[0]).relative_to(ROOT)) if g else None


def complexity(file: str = None, seconds: float = 10.0, max_channels: int = 12) -> dict:
    import antropy as ant
    import mne
    import nolds
    import numpy as np

    file = file or _default_edf()
    if not file:
        return {"available": False, "error": "no EDF on disk"}
    p = ROOT / file if not Path(file).is_absolute() else Path(file)
    if not p.exists():
        return {"available": False, "error": f"EDF not found: {file}"}

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    raw.crop(tmin=0, tmax=min(seconds, raw.times[-1]))
    data = raw.get_data()[:max_channels]
    ch = raw.ch_names[:max_channels]

    rows = []
    for i, name in enumerate(ch):
        sig = data[i].astype(float)
        # decimate for the O(n^2) measures (sample entropy / DFA) to stay tractable
        dec = sig[:: max(1, len(sig) // 2000)]
        try:
            row = {
                "channel": name,
                "spectral_entropy": round(float(ant.spectral_entropy(sig, sf, method="welch", normalize=True)), 4),
                "permutation_entropy": round(float(ant.perm_entropy(sig, normalize=True)), 4),
                "sample_entropy": round(float(ant.sample_entropy(dec)), 4),
                "dfa": round(float(nolds.dfa(dec)), 4),
                "hurst": round(float(nolds.hurst_rs(dec)), 4),
            }
        except Exception as e:  # noqa: BLE001 — report-only; never crash the feature pass
            row = {"channel": name, "error": f"{type(e).__name__}: {str(e)[:50]}"}
        rows.append(row)

    ok = [r for r in rows if "spectral_entropy" in r]
    def avg(k):
        vals = [r[k] for r in ok if isinstance(r.get(k), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None
    return {
        "available": True, "file": p.name, "sfreq": sf, "seconds": round(min(seconds, raw.times[-1]), 1),
        "n_channels": len(ch), "per_channel": rows,
        "summary": {"mean_spectral_entropy": avg("spectral_entropy"),
                    "mean_permutation_entropy": avg("permutation_entropy"),
                    "mean_sample_entropy": avg("sample_entropy"),
                    "mean_dfa": avg("dfa"), "mean_hurst": avg("hurst")},
        "libraries": {"antropy": "spectral/permutation/sample entropy", "nolds": "DFA + Hurst (R/S)"},
        "note": ("Nonlinear EEG complexity (entropy + fractal). Lower entropy / altered DFA "
                 "associate with ictal/abnormal states. Research-grade features, not a diagnosis."),
        "source": "Real EDF via MNE + AntroPy + Nolds.",
    }


if __name__ == "__main__":
    r = complexity()
    if r["available"]:
        print("EEG complexity:", r["file"], "| summary:", r["summary"])
        for row in r["per_channel"][:3]:
            print("  ", row)
    else:
        print(r)
