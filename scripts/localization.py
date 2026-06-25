#!/usr/bin/env python3
"""Localization Dashboard — seizure-focus localization from annotated CHB-MIT EDF.

Using the seizure annotation windows (from ictal_analysis), ranks channels by
their ICTAL/INTERICTAL broadband power ratio — channels with the largest power
increase localize the seizure focus. Maps electrode names to region + hemisphere
(10-20 / CHB-MIT bipolar) to summarise the focus.

100% real (annotated EDF via MNE) — runs under the canonical venv.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# region heuristics from 10-20 electrode labels
REGION = [("F", "frontal"), ("FP", "frontal"), ("FT", "fronto-temporal"), ("T", "temporal"),
          ("C", "central"), ("P", "parietal"), ("O", "occipital")]


def _hemi(ch):
    nums = re.findall(r"\d+", ch)
    if not nums:
        return "midline"
    odd = any(int(n) % 2 == 1 for n in nums)
    even = any(int(n) % 2 == 0 for n in nums)
    if odd and not even:
        return "left"
    if even and not odd:
        return "right"
    return "bilateral"


def _region(ch):
    up = ch.upper()
    for pre, reg in sorted(REGION, key=lambda x: -len(x[0])):
        if up.startswith(pre) or f"-{pre}" in up or f" {pre}" in up:
            return reg
    return "unspecified"


def localize(file: str = None) -> dict:
    import mne
    import numpy as np
    from scipy import signal as sps
    import scripts.ictal_analysis as ia

    ann = [a for a in ia.parse_seizure_annotations() if a["edf_on_disk"] and a["seizures"]]
    if not ann:
        return {"available": False, "error": "no annotated seizure EDF on disk"}
    rec = next((a for a in ann if file and (a["file"] == file or a["edf_path"] == file)), ann[0])
    p = ROOT / rec["edf_path"]
    raw = mne.io.read_raw_edf(str(p), preload=False, verbose="ERROR")
    sf = float(raw.info["sfreq"]); dur = float(raw.times[-1]); ch = raw.ch_names
    sz = rec["seizures"][0]
    t0, t1 = sz["start_s"], min(sz["end_s"], dur)
    i0 = 0.0 if t0 > 120 else min(t1 + 60, dur - 40)
    i1 = min(i0 + (t1 - t0), dur)

    def power(a, b):
        r = raw.copy().crop(tmin=a, tmax=b); r.load_data(verbose="ERROR")
        f, ps = sps.welch(r.get_data(), fs=sf, nperseg=int(min(sf * 2, r.get_data().shape[1])))
        return ps[:, (f >= 1) & (f < 30)].sum(axis=1)

    ictal_p, inter_p = power(t0, t1), power(i0, i1)
    ratio = ictal_p / (inter_p + 1e-20)
    order = np.argsort(ratio)[::-1]
    ranked = [{"channel": ch[i], "ictal_increase_x": round(float(ratio[i]), 1),
               "region": _region(ch[i]), "hemisphere": _hemi(ch[i])} for i in order]
    top = ranked[:6]
    # focus summary = dominant region + hemisphere among top channels
    from collections import Counter
    reg = Counter(c["region"] for c in top).most_common(1)[0][0]
    hem = Counter(c["hemisphere"] for c in top if c["hemisphere"] in ("left", "right")).most_common(1)
    hemisphere = hem[0][0] if hem else "bilateral/midline"
    return {
        "available": True, "file": rec["file"], "sfreq": sf,
        "seizure_window": {"start_s": t0, "end_s": t1},
        "top_focus_channels": top,
        "localized_focus": {"region": reg, "hemisphere": hemisphere,
                            "summary": f"{hemisphere} {reg}",
                            "peak_increase_x": top[0]["ictal_increase_x"] if top else None},
        "all_channels_ranked": ranked,
        "method": "Per-channel ictal/interictal broadband (1-30Hz) power ratio; top channels = focus.",
        "note": ("Screening localization from scalp EEG power increase — NOT intracranial "
                 "seizure-onset-zone localization. Requires clinician + ictal recording for surgical use."),
        "source": "CHB-MIT annotated EDF via MNE/SciPy.",
    }


if __name__ == "__main__":
    r = localize()
    if r["available"]:
        print("Localization:", r["file"], "→", r["localized_focus"]["summary"],
              f"(peak {r['localized_focus']['peak_increase_x']}x)")
        for c in r["top_focus_channels"]:
            print(f"  {c['channel']}: {c['ictal_increase_x']}x [{c['hemisphere']} {c['region']}]")
    else:
        print(r)
