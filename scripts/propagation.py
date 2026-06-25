#!/usr/bin/env python3
"""Seizure Propagation Map — per-channel onset timing during the ictal window.

For an annotated CHB-MIT seizure, band-pass filters each channel, finds when its
envelope first crosses a baseline-relative threshold, and orders channels by
onset time — the propagation sequence (which region leads, how it spreads).

100% real (annotated EDF via MNE/SciPy) — runs under the canonical venv.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def propagation(file: str = None, pre_s: float = 6.0) -> dict:
    import mne
    import numpy as np
    from scipy.signal import butter, filtfilt
    import scripts.ictal_analysis as ia
    import scripts.localization as loc

    ann = [a for a in ia.parse_seizure_annotations() if a["edf_on_disk"] and a["seizures"]]
    if not ann:
        return {"available": False, "error": "no annotated seizure EDF on disk"}
    rec = next((a for a in ann if file and (a["file"] == file or a["edf_path"] == file)), ann[0])
    p = ROOT / rec["edf_path"]
    raw = mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR")
    sf = float(raw.info["sfreq"]); dur = float(raw.times[-1]); ch = raw.ch_names
    sz = rec["seizures"][0]
    t0, t1 = sz["start_s"], min(sz["end_s"], dur)
    nyq = sf / 2.0
    b, a = butter(4, [1.0 / nyq, min(30.0, nyq - 1) / nyq], btype="band")

    # CLEAN interictal baseline (far from the seizure) for a stable per-channel threshold —
    # avoids the pre-ictal-contamination bug where the in-segment baseline fires early.
    bt0 = 60.0 if t0 > 160 else min(t1 + 60, dur - 40)
    bseg = raw.copy().crop(tmin=bt0, tmax=min(bt0 + 40, dur)); bseg.load_data(verbose="ERROR")
    bd = bseg.get_data()
    base_env = np.array([np.median(np.abs(filtfilt(b, a, bd[i]))) + 1e-12 for i in range(len(ch))])

    # detect onset WITHIN the ictal window (sliding 0.5s windows); onset = first window
    # whose envelope exceeds 5× the interictal baseline.
    iseg = raw.copy().crop(tmin=t0, tmax=t1); iseg.load_data(verbose="ERROR")
    d = iseg.get_data()
    step = max(1, int(0.5 * sf))
    onsets = []
    for i in range(len(ch)):
        env = np.abs(filtfilt(b, a, d[i]))
        thr = 5 * base_env[i]
        onset_rel = None
        for w in range(0, len(env) - step, step):
            if env[w:w + step].mean() > thr:
                onset_rel = w / sf
                break
        onsets.append({"channel": ch[i], "onset_s": (round(float(onset_rel), 2) if onset_rel is not None else None),
                       "region": loc._region(ch[i]), "hemisphere": loc._hemi(ch[i])})
    activated = [o for o in onsets if o["onset_s"] is not None]
    activated.sort(key=lambda o: o["onset_s"])
    leaders = activated[:5]
    spread_span = (round(activated[-1]["onset_s"] - activated[0]["onset_s"], 2)
                   if len(activated) >= 2 else None)
    return {
        "available": True, "file": rec["file"], "sfreq": sf,
        "seizure_window": {"start_s": t0, "end_s": t1},
        "onset_leaders": leaders,
        "propagation_order": activated,
        "n_activated": len(activated), "n_silent": len(onsets) - len(activated),
        "spread_span_s": spread_span,
        "lead_region": (leaders[0]["region"] if leaders else None),
        "lead_hemisphere": (leaders[0]["hemisphere"] if leaders else None),
        "method": "Band-pass 1-30Hz envelope; onset = first 5×baseline crossing; ordered by time.",
        "note": ("Scalp-EEG propagation timing — screening illustration of spread, NOT "
                 "intracranial onset-zone mapping for surgical planning."),
        "source": "CHB-MIT annotated EDF via MNE/SciPy.",
    }


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    r = propagation()
    if r["available"]:
        print("Propagation:", r["file"], "| lead:", r["lead_hemisphere"], r["lead_region"],
              "| spread span:", r["spread_span_s"], "s")
        for o in r["onset_leaders"]:
            print(f"  {o['channel']}: t{o['onset_s']:+.1f}s [{o['hemisphere']} {o['region']}]")
    else:
        print(r)
