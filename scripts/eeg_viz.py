#!/usr/bin/env python3
"""Real EEG visualizations (P0 clinical visuals) from raw EDF via MNE/SciPy.
PSD curve + band power (interactive data) + spectrogram + scalp topomap (PNG).
Honest: topomap only when channels are monopolar 10-20; bipolar montages (CHB-MIT)
get per-channel band power instead, with a note. No synthetic data."""
from __future__ import annotations

import base64
import io
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _clean_name(ch: str) -> str:
    """Map 'EEG Fp1-LE' / 'Fp1-Ref' → 'Fp1' for montage; bipolar 'FP1-F7' → '' (no single pos)."""
    n = re.sub(r"^EEG\s+", "", ch, flags=re.I).strip()
    n = re.sub(r"-(LE|RE|REF|A1|A2|M1|M2|AVG)$", "", n, flags=re.I)
    # bipolar pair like FP1-F7 → not a single electrode
    if re.match(r"^[A-Za-z]+\d*-[A-Za-z]+\d*$", n):
        return ""
    return n


def render(edf_path: str, seconds: float = 10.0) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
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
    data = raw.get_data()  # (n_ch, n_samples)
    ch_names = raw.ch_names
    n_ch = len(ch_names)

    # ── PSD via Welch (mean across channels) ──
    freqs, psd = sps.welch(data, fs=sf, nperseg=int(min(sf * 2, data.shape[1])))
    psd_mean = psd.mean(axis=0)
    mask = freqs <= 45
    psd_curve = [{"freq": round(float(f), 2), "power_db": round(float(10 * np.log10(v + 1e-12)), 2)}
                 for f, v in zip(freqs[mask], psd_mean[mask])]

    # ── Band power (relative, mean across channels) ──
    total = float(psd_mean[mask].sum()) + 1e-12
    band_power = []
    for b, (lo, hi) in BANDS.items():
        bidx = (freqs >= lo) & (freqs < hi)
        band_power.append({"band": b, "rel_power": round(float(psd_mean[bidx].sum()) / total, 4)})

    # ── Spectrogram on highest-variance channel ──
    var_ch = int(np.argmax(data.var(axis=1)))
    f_s, t_s, Sxx = sps.spectrogram(data[var_ch], fs=sf, nperseg=int(min(sf, data.shape[1] // 4 or 1)))
    fmask = f_s <= 45
    fig1, ax1 = plt.subplots(figsize=(6, 2.6))
    ax1.pcolormesh(t_s, f_s[fmask], 10 * np.log10(Sxx[fmask] + 1e-12), shading="gouraud", cmap="viridis")
    ax1.set_ylabel("Hz"); ax1.set_xlabel("Time (s)")
    ax1.set_title(f"Spectrogram — {ch_names[var_ch]} (highest-variance channel)", fontsize=9)
    spectrogram_png = _png_b64(fig1)

    # ── Topomap (only if monopolar 10-20) ──
    cleaned = [_clean_name(c) for c in ch_names]
    montage = mne.channels.make_standard_montage("standard_1020")
    valid = {c.lower(): i for i, c in enumerate(montage.ch_names)}
    mappable = [(i, cleaned[i]) for i in range(n_ch) if cleaned[i] and cleaned[i].lower() in valid]
    topomap_png, topomap_note = None, None
    if len(mappable) >= 8:
        try:
            idxs = [i for i, _ in mappable]
            sub = raw.copy().pick([ch_names[i] for i in idxs])
            sub.rename_channels({ch_names[i]: cleaned[i] for i in idxs})
            sub.set_montage(montage, on_missing="ignore", verbose="ERROR")
            # alpha-band relative power per channel
            fa, pa = sps.welch(sub.get_data(), fs=sf, nperseg=int(min(sf * 2, sub.get_data().shape[1])))
            am = (fa >= 8) & (fa < 13)
            tot = pa[:, fa <= 45].sum(axis=1) + 1e-12
            alpha_rel = pa[:, am].sum(axis=1) / tot
            fig2, ax2 = plt.subplots(figsize=(3.4, 3.4))
            mne.viz.plot_topomap(alpha_rel, sub.info, axes=ax2, show=False, cmap="RdBu_r", contours=4)
            ax2.set_title(f"Alpha relative power ({len(idxs)} ch)", fontsize=9)
            topomap_png = _png_b64(fig2)
            topomap_note = f"Scalp topomap of alpha-band relative power across {len(idxs)} monopolar electrodes."
        except Exception as e:  # noqa: BLE001
            topomap_note = f"Topomap render failed: {str(e)[:80]}"
    else:
        topomap_note = (f"Bipolar/non-standard montage ({n_ch} channels like '{ch_names[0]}') — "
                        "scalp topomap requires monopolar 10-20 referencing. Per-channel band power shown instead.")

    # per-channel band power (always available — works for bipolar)
    fch, pch = sps.welch(data, fs=sf, nperseg=int(min(sf * 2, data.shape[1])))
    am = (fch >= 8) & (fch < 13)
    tot = pch[:, fch <= 45].sum(axis=1) + 1e-12
    per_channel_alpha = [{"channel": ch_names[i], "alpha_rel": round(float(pch[i, am].sum() / tot[i]), 4)}
                         for i in range(n_ch)]

    # ── Lateralization: L/R hemisphere band-power asymmetry (10-20: odd=Left, even=Right) ──
    import re as _re
    def _hemi(nm):
        m = _re.search(r"(\d+)", _clean_name(nm) or nm)
        if not m:
            return None
        return "L" if int(m.group(1)) % 2 == 1 else "R"
    lat = None
    try:
        f2, p2 = sps.welch(data, fs=sf, nperseg=int(min(sf * 2, data.shape[1])))
        bands_idx = {b: (f2 >= lo) & (f2 < hi) for b, (lo, hi) in BANDS.items()}
        L = [i for i in range(n_ch) if _hemi(ch_names[i]) == "L"]
        R = [i for i in range(n_ch) if _hemi(ch_names[i]) == "R"]
        if L and R:
            rows = []
            for b, idx in bands_idx.items():
                lp = float(p2[L][:, idx].sum()); rp = float(p2[R][:, idx].sum())
                ai = round((lp - rp) / (lp + rp + 1e-12), 4)  # +ve = left-dominant
                rows.append({"band": b, "left": round(lp, 4), "right": round(rp, 4), "asymmetry_index": ai,
                             "lateralization": "Left" if ai > 0.1 else "Right" if ai < -0.1 else "Symmetric"})
            overall = round(sum(r["asymmetry_index"] for r in rows) / len(rows), 4)
            lat = {"available": True, "n_left": len(L), "n_right": len(R), "by_band": rows,
                   "overall_index": overall,
                   "focus": "Left-hemisphere" if overall > 0.1 else "Right-hemisphere" if overall < -0.1 else "Symmetric",
                   "basis": "(L-R)/(L+R) band power; odd electrodes=Left, even=Right; +ve=left-dominant",
                   "note": "Screening asymmetry only — not seizure localization (needs ictal recording + clinician)."}
        else:
            lat = {"available": False, "note": "Bipolar/non-standard montage — hemisphere split needs monopolar 10-20."}
    except Exception as _e:
        lat = {"available": False, "note": f"lateralization failed: {str(_e)[:60]}"}

    return {
        "available": True, "file": p.name, "sfreq": sf, "n_channels": n_ch,
        "lateralization": lat,
        "seconds_analyzed": round(min(seconds, raw.times[-1]), 1),
        "channels": ch_names,
        "psd_curve": psd_curve,
        "band_power": band_power,
        "spectrogram_png": spectrogram_png,
        "topomap_png": topomap_png, "topomap_note": topomap_note,
        "per_channel_alpha": per_channel_alpha,
        "source": "Real EDF via MNE-Python + SciPy (no synthetic data).",
    }


def list_presets() -> dict:
    """Real EDF presets the viz can render (epilepsy bipolar + monopolar for topomap)."""
    presets = []
    chb = sorted((ROOT / "data/real_eeg/epilepsy_physionet").glob("chb*.edf"))
    if chb:
        presets.append({"key": "epilepsy_chb", "label": "Epilepsy (CHB-MIT, bipolar)",
                        "file": str(chb[0].relative_to(ROOT)), "topomap": False})
    mono = sorted((ROOT / "data/real_eeg/depression_figshare").glob("*.edf"))
    if mono:
        presets.append({"key": "monopolar", "label": "Monopolar 10-20 (topomap-capable)",
                        "file": str(mono[0].relative_to(ROOT)), "topomap": True})
    return {"presets": presets, "default": presets[0]["file"] if presets else None}


if __name__ == "__main__":
    import json
    pr = list_presets()
    print("presets:", json.dumps(pr, indent=1))
    if pr["default"]:
        r = render(pr["default"], seconds=10)
        print({k: (v[:40] + "..." if isinstance(v, str) and len(v) > 40 else v)
               for k, v in r.items() if k not in ("psd_curve", "channels", "per_channel_alpha")})
