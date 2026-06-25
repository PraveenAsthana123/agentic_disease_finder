#!/usr/bin/env python3
"""TSFEL time-series features — automated EEG feature extraction from real EDF.

Uses TSFEL to extract statistical + temporal domain features per channel from a
real recording window — the kind of broad feature bank used as ML inputs for
seizure/state classification. Returns per-channel feature counts + a sample of
named features + cross-channel summary stats.

Wires the TSFEL EEG-stack lib (installed in the canonical venv per §61.11).
100% real signal — no synthetic.
"""
import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _default_edf():
    g = sorted(glob.glob(str(ROOT / "data" / "real_eeg" / "epilepsy_physionet" / "chb*.edf")))
    return str(Path(g[0]).relative_to(ROOT)) if g else None


def extract(file: str = None, seconds: float = 10.0, max_channels: int = 8) -> dict:
    import mne
    import numpy as np
    import tsfel

    file = file or _default_edf()
    if not file:
        return {"available": False, "error": "no EDF on disk"}
    p = ROOT / file if not Path(file).is_absolute() else Path(file)
    if not p.exists():
        return {"available": False, "error": f"EDF not found: {file}"}

    raw = mne.io.read_raw_edf(str(p), preload=True, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    raw.crop(tmax=min(seconds, raw.times[-1]))
    data = raw.get_data()[:max_channels]
    ch = raw.ch_names[:max_channels]

    # statistical + temporal domains (fast, no overlap with the dedicated spectral views)
    cfg = {**tsfel.get_features_by_domain("statistical"), **tsfel.get_features_by_domain("temporal")}
    per_channel, feat_names = [], None
    agg = {}
    for i, name in enumerate(ch):
        try:
            feat = tsfel.time_series_features_extractor(cfg, data[i], fs=sf, verbose=0)
            row = {c.split("_", 1)[-1]: round(float(v), 4) for c, v in zip(feat.columns, feat.values[0])}
            if feat_names is None:
                feat_names = list(row.keys())
            for k, v in row.items():
                if np.isfinite(v):
                    agg.setdefault(k, []).append(v)
            per_channel.append({"channel": name, "n_features": len(row),
                                "sample": dict(list(row.items())[:6])})
        except Exception as e:  # noqa: BLE001 — report-only
            per_channel.append({"channel": name, "error": f"{type(e).__name__}: {str(e)[:50]}"})

    summary = {k: round(float(sum(v) / len(v)), 4) for k, v in list(agg.items())[:12] if v}
    return {
        "available": True, "file": p.name, "sfreq": sf,
        "n_channels": len(ch),
        "n_features_per_channel": (len(feat_names) if feat_names else 0),
        "feature_domains": ["statistical", "temporal"],
        "feature_names": feat_names[:30] if feat_names else [],
        "per_channel": per_channel,
        "cross_channel_mean": summary,
        "note": "Automated TSFEL feature bank (statistical+temporal) per channel — ML-ready inputs.",
        "source": "Real EDF via MNE + TSFEL.",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    r = extract()
    if r["available"]:
        print("TSFEL:", r["file"], "|", r["n_features_per_channel"], "features ×", r["n_channels"], "channels")
        print("  sample features:", r["feature_names"][:5])
    else:
        print(r)
