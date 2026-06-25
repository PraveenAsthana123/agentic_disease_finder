#!/usr/bin/env python3
"""Ictal vs Interictal Dashboard — real seizure spectral analysis from CHB-MIT.

Parses the CHB-MIT summary.txt annotations (per-file seizure start/end times),
then for an annotated file extracts the ICTAL window (during seizure) and an
INTERICTAL window (seizure-free), and contrasts band power — the classic ictal
delta-dominance / alpha-suppression signature.

100% real (annotated EDF via MNE) — runs under the canonical venv.
"""
import glob
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30), "gamma": (30, 45)}


def parse_seizure_annotations():
    """Parse all chbXX-summary.txt → files with seizures + ictal time windows."""
    files = []
    for summ in sorted(ROOT.glob("data/**/chb*-summary.txt")):
        text = summ.read_text(errors="ignore")
        # split into per-file blocks
        blocks = re.split(r"File Name:", text)
        for b in blocks[1:]:
            name = b.strip().split()[0] if b.strip() else None
            if not name:
                continue
            nm = re.search(r"Number of Seizures in File:\s*(\d+)", b)
            n_sz = int(nm.group(1)) if nm else 0
            if n_sz <= 0:
                continue
            starts = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds", b)]
            ends = [int(x) for x in re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds", b)]
            edf = glob.glob(str(ROOT / "data" / "**" / name), recursive=True)
            files.append({"file": name, "n_seizures": n_sz,
                          "seizures": [{"start_s": s, "end_s": e} for s, e in zip(starts, ends)],
                          "edf_on_disk": bool(edf), "edf_path": (str(Path(edf[0]).relative_to(ROOT)) if edf else None)})
    return files


def _band_power(raw_seg, sf):
    import numpy as np
    from scipy import signal as sps
    d = raw_seg.get_data()
    f, ps = sps.welch(d, fs=sf, nperseg=int(min(sf * 2, d.shape[1])))
    tot = ps[:, f <= 45].sum() + 1e-20
    return {b: round(float(ps[:, (f >= lo) & (f < hi)].sum() / tot), 4) for b, (lo, hi) in BANDS.items()}


def ictal_interictal(file: str = None) -> dict:
    """Contrast ictal vs interictal band power for an annotated seizure file."""
    import mne

    ann = parse_seizure_annotations()
    ann = [a for a in ann if a["edf_on_disk"] and a["seizures"]]
    if not ann:
        return {"available": False, "error": "no annotated seizure EDF on disk"}
    rec = next((a for a in ann if file and (a["file"] == file or a["edf_path"] == file)), ann[0])

    p = ROOT / rec["edf_path"]
    raw = mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR")
    sf = float(raw.info["sfreq"])
    dur = float(raw.times[-1])
    sz = rec["seizures"][0]
    t0, t1 = sz["start_s"], min(sz["end_s"], dur)
    # interictal: a seizure-free window well before the first seizure (or after if none before)
    inter0 = 0.0 if t0 > 120 else min(t1 + 60, dur - 40)
    inter1 = min(inter0 + (t1 - t0), dur)

    ictal = raw.copy().crop(tmin=t0, tmax=t1); ictal.load_data(verbose="ERROR")
    inter = raw.copy().crop(tmin=inter0, tmax=inter1); inter.load_data(verbose="ERROR")
    bp_ictal = _band_power(ictal, sf)
    bp_inter = _band_power(inter, sf)
    delta_shift = round(bp_ictal["delta"] - bp_inter["delta"], 4)
    alpha_shift = round(bp_ictal["alpha"] - bp_inter["alpha"], 4)

    return {
        "available": True, "file": rec["file"], "sfreq": sf, "recording_s": round(dur, 1),
        "seizure_window": {"start_s": t0, "end_s": t1, "duration_s": round(t1 - t0, 1)},
        "interictal_window": {"start_s": round(inter0, 1), "end_s": round(inter1, 1)},
        "band_power": {"ictal": bp_ictal, "interictal": bp_inter},
        "ictal_signature": {
            "delta_shift": delta_shift, "alpha_shift": alpha_shift,
            "verdict": ("ictal slow-wave dominance (delta↑, alpha↓) — consistent with seizure"
                        if delta_shift > 0.05 and alpha_shift < 0 else
                        "atypical contrast — review (delta/alpha shift not in expected ictal direction)"),
        },
        "annotated_files": len(ann),
        "available_files": [{"file": a["file"], "n_seizures": a["n_seizures"]} for a in ann[:20]],
        "source": "CHB-MIT summary.txt seizure annotations + real EDF via MNE/SciPy.",
        "note": "Ictal vs interictal band-power contrast from annotated seizures. Screening illustration, not auto-detection.",
    }


if __name__ == "__main__":
    r = ictal_interictal()
    if r["available"]:
        print("Ictal/Interictal:", r["file"], r["seizure_window"])
        print("  ictal:", r["band_power"]["ictal"])
        print("  interictal:", r["band_power"]["interictal"])
        print("  signature:", r["ictal_signature"]["verdict"])
    else:
        print(r)
